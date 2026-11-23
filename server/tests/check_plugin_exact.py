"""Exact check of dpp_vllm_plugin against vLLM's own processed logprobs.

Runs the same seeded request twice in fresh processes:
  plugin  DPP_LOGQ=1, logprobs_mode=raw_logprobs: log p in column 0, log q in column 1
  vllm    DPP_LOGQ=0, logprobs_mode=processed_logprobs: vLLM's log softmax(z / T')
Sampling does not depend on the logprobs mode, so both draw the same tokens, and the
plugin's log q must equal vLLM's per token up to float32 rounding.
Usage: .venv/bin/python tests/check_plugin_exact.py [T'] [model]
"""
import json
import os
import subprocess
import sys

Tp = sys.argv[1] if len(sys.argv) > 1 else "1.5"
MODEL = sys.argv[2] if len(sys.argv) > 2 else "Qwen/Qwen3-8B-AWQ"

CHILD = r"""
import json, os, sys
from transformers import AutoTokenizer, GenerationConfig
from vllm import LLM, SamplingParams
from vllm.inputs import TokensPrompt
Tp, MODEL, mode, dest = float(sys.argv[1]), sys.argv[2], sys.argv[3], sys.argv[4]
tok = AutoTokenizer.from_pretrained(MODEL)
try:
    eos = GenerationConfig.from_pretrained(MODEL).eos_token_id
except Exception:
    eos = None
eos = tok.eos_token_id if eos is None else eos
eos = eos if isinstance(eos, list) else [eos]
# Chat-template kwargs: only Qwen3 takes enable_thinking (as in the server's MODELS table).
kw = {"enable_thinking": False} if "Qwen3" in MODEL else {}
text = tok.apply_chat_template([{"role": "user", "content": "Give one creative use for an old bicycle, in one sentence."}],
                               tokenize=False, add_generation_prompt=True, **kw)
pids = tok(text, add_special_tokens=False).input_ids
llm = LLM(MODEL, gpu_memory_utilization=float(os.environ.get("UTIL", "0.34")), max_model_len=1024, max_num_seqs=32, logprobs_mode=mode,
          generation_config="vllm", enable_prefix_caching=False)
sp = SamplingParams(n=8, temperature=Tp, top_p=1.0, top_k=0, min_p=0.0, max_tokens=48, logprobs=1, seed=7,
                    stop_token_ids=sorted(eos), detokenize=False)
out = llm.generate(TokensPrompt(prompt_token_ids=pids), sp, use_tqdm=False)[0]
res = []
for c in out.outputs:
    toks = list(c.token_ids)
    res.append({"toks": toks, "lp": [d[t].logprob for t, d in zip(toks, c.logprobs)],
                "other": [[v.logprob for k, v in d.items() if k != t] for t, d in zip(toks, c.logprobs)]})
json.dump(res, open(dest, "w"))
"""


def run(logq, mode):
    env = dict(os.environ, DPP_LOGQ=logq, VLLM_USE_V2_MODEL_RUNNER="0", HF_HUB_OFFLINE="1")
    dest = f"/tmp/dpp_check_{mode}_{os.getpid()}.json"
    logf = os.path.join(os.path.dirname(__file__), "..", "logs", f"check_child_{mode}_logq{logq}.log")
    with open(logf, "w") as lf:
        p = subprocess.run([sys.executable, "-c", CHILD, Tp, MODEL, mode, dest], env=env, stdout=lf, stderr=lf, text=True)
    if p.returncode or not os.path.exists(dest):
        sys.exit(open(logf).read()[-3000:])
    for ln in open(logf):
        if "dpp_vllm_plugin check" in ln:
            last = ln.strip()
    if "last" in locals():
        print("  same-pass check:", last.split("check: ", 1)[1])
    with open(dest) as f:
        res = json.load(f)
    os.remove(dest)
    return res


a = run("1", "raw_logprobs")
b = run("0", "processed_logprobs")
c = run("0", "raw_logprobs")
noise = max(abs(u - v) for x, y in zip(a, c) for u, v in zip(x["lp"], y["lp"]))
print(f"run-to-run: max |log p (plugin run) - log p (plain vLLM run)| = {noise:.2e} per token")
worst_tok, worst_str, n = 0.0, 0.0, 0
for x, y in zip(a, b):
    assert x["toks"] == y["toks"], "the two runs drew different tokens"
    lq_plugin = [o[0] for o in x["other"]]
    lq_vllm = y["lp"]
    d = [abs(u - v) for u, v in zip(lq_plugin, lq_vllm)]
    worst_tok = max(worst_tok, max(d))
    worst_str = max(worst_str, abs(sum(lq_plugin) - sum(lq_vllm)))
    n += len(d)
print(f"T'={Tp}: {len(a)} strings, {n} tokens, same tokens in both runs")
print(f"max |log q plugin - log q vLLM|: {worst_tok:.2e} per token, {worst_str:.2e} per string")
