"""Check the sampler's log p / log q against an independent teacher-forced pass.

Draws a few completions with the dpp plugin (log p and log q of each sampled token from
the logits that drew it), then rescores prompt + completion (end-of-string included) with
full-vocabulary prompt logprobs, where log q = log p / T' - logsumexp(log p / T').
Usage: .venv/bin/python tests/check_logq.py [T'] [model]
"""
import math
import os
import sys

os.environ.setdefault("DPP_LOGQ", "1")
os.environ.setdefault("VLLM_USE_V2_MODEL_RUNNER", "0")

from transformers import AutoTokenizer, GenerationConfig  # noqa: E402
from vllm import LLM, SamplingParams  # noqa: E402
from vllm.inputs import TokensPrompt  # noqa: E402

Tp = float(sys.argv[1]) if len(sys.argv) > 1 else 1.5
MODEL = sys.argv[2] if len(sys.argv) > 2 else "Qwen/Qwen3-8B-AWQ"
PROMPT = "Name a fruit and say one thing about it."

tok = AutoTokenizer.from_pretrained(MODEL)
eos = GenerationConfig.from_pretrained(MODEL).eos_token_id
eos = set(eos if isinstance(eos, list) else [eos])
text = tok.apply_chat_template([{"role": "user", "content": PROMPT}], tokenize=False,
                               add_generation_prompt=True, enable_thinking=False)
pids = tok(text, add_special_tokens=False).input_ids

llm = LLM(MODEL, gpu_memory_utilization=float(os.environ.get("UTIL", 0.34)), max_model_len=1024,
          max_num_seqs=32, logprobs_mode="raw_logprobs", generation_config="vllm", max_logprobs=-1,
          enable_prefix_caching=False)
sp = SamplingParams(n=4, temperature=Tp, top_p=1.0, top_k=0, min_p=0.0, max_tokens=16, logprobs=1,
                    seed=1, stop_token_ids=sorted(eos), detokenize=False)
out = llm.generate(TokensPrompt(prompt_token_ids=pids), sp)[0]

worst = 0.0
for c in out.outputs:
    toks = list(c.token_ids)
    lp = lq = 0.0
    for t, d in zip(toks, c.logprobs):
        assert len(d) == 2, f"expected the sampled token plus the log q entry, got {d}"
        lp += d[t].logprob
        lq += next(v.logprob for k, v in d.items() if k != t)
    tf = llm.generate(TokensPrompt(prompt_token_ids=pids + toks),
                      SamplingParams(max_tokens=1, prompt_logprobs=-1, detokenize=False))[0]
    tlp = tlq = 0.0
    for j, t in enumerate(toks):
        d = tf.prompt_logprobs[len(pids) + j]
        vals = [v.logprob / Tp for v in d.values()]
        m = max(vals)
        lse = m + math.log(sum(math.exp(v - m) for v in vals))
        tlp += d[t].logprob
        tlq += d[t].logprob / Tp - lse
    diff = max(abs(lp - tlp), abs(lq - tlq))
    worst = max(worst, diff)
    print(f"{tok.decode([t for t in toks if t not in eos])!r:50.50} ended={c.finish_reason == 'stop' and toks[-1] in eos} "
          f"n={len(toks)} logp {lp:9.4f} vs {tlp:9.4f}  logq {lq:9.4f} vs {tlq:9.4f}  diff {diff:.2e}")
print(f"T'={Tp}: max |sampler - teacher forcing| = {worst:.3e} nats per string")
