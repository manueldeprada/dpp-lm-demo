"""vLLM plugin: log q next to log p for every sampled token, from the same logits.

The PoolDPP demo needs, for each sampled token y_t with logits z,
    log p(y_t | prefix) = log softmax(z)[y_t]        (the model at T = 1)
    log q(y_t | prefix) = log softmax(z / T')[y_t]   (the proposal that drew y_t)
vLLM returns one of the two, depending on `logprobs_mode`. With DPP_LOGQ=1 in the
environment and logprobs_mode="raw_logprobs", requests made with logprobs=1 get
    column 0: the sampled token and its raw log p (unchanged vLLM behavior),
    column 1: log q of the same token, under the token id SENTINEL
              (SENTINEL + 1 when the sampled token is SENTINEL itself),
in place of the top-1 alternative. A client reads log p as the entry keyed by the
sampled token and log q as the other entry.

Both values come from one log-sum-exp each over the logits the sampler sees (before
temperature), so they agree with a teacher-forced forward pass up to float32 rounding.
Only the V1 model runner's `vllm.v1.sample.sampler.Sampler` is patched: run with
VLLM_USE_V2_MODEL_RUNNER=0. Without DPP_LOGQ=1 the plugin does nothing.

DPP_LOGQ_CHECK=1 also computes log softmax(z / T') at the sampled tokens directly from
a copy of the same logits and logs the largest difference from the formula (slow; for tests).
"""
import logging
import os
import sys

SENTINEL = 0
CHUNK = 64  # rows per log-sum-exp, to bound the extra float32 memory
CHECK = os.environ.get("DPP_LOGQ_CHECK") == "1"
_STATS = {"q": 0.0, "p": 0.0, "n": 0}

log = logging.getLogger("dpp_vllm_plugin")
_OWN = None        # the `own` module when DPP_OWN_EMB=1 (the LM's own states, see own.py)
_RUNNER = [None]


def _runner():
    """The V1 model runner that called the sampler (one per process)."""
    r = _RUNNER[0]
    if r is None:
        f = sys._getframe(1)
        for _ in range(14):
            if f is None:
                break
            s = f.f_locals.get("self")
            if s is not None and hasattr(s, "input_batch") and hasattr(s, "requests"):
                _RUNNER[0] = r = s
                break
            f = f.f_back
    return r


def _lse(logits, temp):
    import torch
    lse1 = torch.empty(logits.shape[0], dtype=torch.float32, device=logits.device)
    lseT = torch.empty_like(lse1)
    for i in range(0, logits.shape[0], CHUNK):
        z = logits[i:i + CHUNK].float()
        lse1[i:i + CHUNK] = torch.logsumexp(z, dim=-1)
        lseT[i:i + CHUNK] = torch.logsumexp(z / temp[i:i + CHUNK].unsqueeze(1), dim=-1)
    return lse1, lseT


def register():
    if os.environ.get("DPP_LOGQ") != "1":
        return
    import torch
    from vllm.v1.sample import sampler as S
    if getattr(S.Sampler.forward, "_dpp_logq", False):
        return
    orig = S.Sampler.forward
    global _OWN
    if os.environ.get("DPP_OWN_EMB") == "1":
        try:
            from . import own as _o
            _o.install_worker_methods()
            _OWN = _o
            log.warning("dpp_vllm_plugin: own-state accumulation enabled (DPP_OWN_EMB=1)")
        except Exception:
            log.exception("dpp_vllm_plugin: own-state feature could not be installed")

    def forward(self, logits, sampling_metadata, predict_bonus_token=False, logprobs_mode_override=None):
        mode = logprobs_mode_override or self.logprobs_mode
        temp = sampling_metadata.temperature
        want = (mode == "raw_logprobs" and sampling_metadata.max_num_logprobs == 1
                and temp is not None and not sampling_metadata.all_greedy)
        if want:
            # Greedy rows in a mixed batch: vLLM divides them by 1.0, so do the same.
            temp = torch.where(temp < 1e-5, torch.ones_like(temp), temp).float()
            lse1, lseT = _lse(logits, temp)  # read before vLLM scales the logits in place
            z_copy = logits.float().clone() if CHECK else None
        octx = None
        if _OWN is not None:
            runner = _runner()
            if runner is not None:
                octx = _OWN.pre(logits, runner)   # accumulate P l, apply teacher forcing (own requests only)
        out = orig(self, logits, sampling_metadata, predict_bonus_token, logprobs_mode_override)
        if octx is not None:
            _OWN.post(octx, out)
        lt = out.logprobs_tensors
        if want and lt is not None and lt.logprobs.dim() == 2 and lt.logprobs.shape[1] == 2:
            ids, lps = lt.logprob_token_ids, lt.logprobs
            if octx is not None and "rf" in octx:
                # teacher-forced rows: vLLM saw a one-hot logit row, so restore the true log p of the forced token
                lps[octx["rf"], 0] = (octx["zy"] - lse1[octx["rf"]]).to(lps.dtype)
            logp = lps[:, 0].float()
            # log q = z_y / T' - lse(z / T'), with z_y = log p + lse(z).
            lps[:, 1] = ((logp + lse1) / temp - lseT).to(lps.dtype)
            sampled = ids[:, 0]
            ids[:, 1] = torch.where(sampled == SENTINEL, SENTINEL + 1, SENTINEL).to(ids.dtype)
            if CHECK:
                direct = (z_copy / temp.unsqueeze(1)).log_softmax(-1).gather(1, sampled.long().unsqueeze(1)).squeeze(1)
                rawlp = z_copy.log_softmax(-1).gather(1, sampled.long().unsqueeze(1)).squeeze(1)
                dq = (direct - lps[:, 1].float()).abs().max().item()
                dp = (rawlp - logp).abs().max().item()
                _STATS["q"] = max(_STATS["q"], dq)
                _STATS["p"] = max(_STATS["p"], dp)
                _STATS["n"] += 1
                if _STATS["n"] % 16 == 0:
                    log.warning("dpp_vllm_plugin check: %d steps, max |log q - direct| %.2e, max |log p - direct| %.2e",
                                _STATS["n"], _STATS["q"], _STATS["p"])
        return out

    forward._dpp_logq = True
    S.Sampler.forward = forward
    log.warning("dpp_vllm_plugin: sampled logprobs=1 now return log q in column 1 (pid %d)", os.getpid())
