"""The LM's own states: per-sequence mean of P l_t, accumulated inside the sampler (DPP_OWN_EMB=1).

For a request whose SamplingParams.extra_args has "dpp_own": TAG, the sampler wrapper in
__init__.py calls `step_pre` with the step's logits (before sampling) and `step_post` with the
sampled tokens. P is the r x V matrix stored by search/build_own_P.py (as Pt, V x r, float16):
P l is the random projection R h of the final hidden state h that produced the logits l.
Per (TAG, sample index) the sum of P l_t and the number of steps are kept on the GPU; the server
reads and clears them with the worker method `dpp_own_take` (collective_rpc).

extra_args keys (all optional except dpp_own):
  dpp_own    str   tag of the request; the sample index is the child index of an n > 1 request
  dpp_stop   list  token ids that end the sequence (steps after the one that sampled one are ignored)
  dpp_max    int   max_tokens of the request (steps after it are ignored; async scheduling may run one extra)
  dpp_force  list  teacher forcing: token id to sample at step t is dpp_force[t]
  dpp_force_eos int token id forced after the list is exhausted (a discarded extra step otherwise)

Nothing here runs unless DPP_OWN_EMB=1 and a request carries dpp_own and the model's P file exists.
Any exception disables the feature for the process and leaves the sampler untouched.
"""
import glob
import json
import logging
import os
import re
import threading
import time

log = logging.getLogger("dpp_vllm_plugin.own")
ROOT = os.environ.get("DPP_OWN_DIR", os.path.join(os.getcwd(), "own_emb"))   # directory with <modelKey>/{P.safetensors,meta.json,mu.npy}
CAP = 1024      # concurrent (tag, index) slots
LMAX = 260      # longest forced list (max_new_tokens 256 plus EOS)
NSTOP = 8
ROW_CHUNK = 16  # rows per P matmul, bounds the fp16 copy of the logits
EXPIRY = 180.0  # seconds a slot may stay unread
IDLE = 60.0     # seconds without own requests after which P is unloaded (GPU memory is tight)

_ST = {"state": None, "off": False, "lock": threading.RLock(), "reaper": None}


def _repo_dirs():
    out = {}
    for f in glob.glob(os.path.join(ROOT, "*", "meta.json")):
        m = json.load(open(f))
        out[m["repo"]] = os.path.dirname(f)
    return out


class Own:
    def __init__(self, repo, d):
        import torch
        from safetensors.torch import load_file
        self.torch = torch
        self.repo = repo
        self.meta = json.load(open(os.path.join(d, "meta.json")))
        self.dev = torch.device("cuda")
        self.Pt = load_file(os.path.join(d, "P.safetensors"), device="cuda")["Pt"].contiguous()
        # final logit soft-capping (Gemma 2): the sampler sees t = c tanh(z / c); the raw logits z = c atanh(t / c) are what W h gives
        self.cap = float((self.meta.get("source") or {}).get("softcap") or 0.0)
        self.V, self.r = self.Pt.shape
        t = torch
        self.accum = t.zeros(CAP, self.r, dtype=t.float32, device=self.dev)
        self.count = t.zeros(CAP, dtype=t.int32, device=self.dev)
        self.done = t.zeros(CAP, dtype=t.bool, device=self.dev)
        self.maxtok = t.full((CAP,), 1 << 30, dtype=t.int32, device=self.dev)
        self.stop = t.full((CAP, NSTOP), -1, dtype=t.int64, device=self.dev)
        self.forced = t.zeros(CAP, LMAX, dtype=t.int64, device=self.dev)
        self.flen = t.zeros(CAP, dtype=t.int32, device=self.dev)
        self.feos = t.zeros(CAP, dtype=t.int64, device=self.dev)
        self.is_forced = [False] * CAP
        self.free = list(range(CAP - 1, -1, -1))
        self.slot_of = {}      # (tag, idx) -> slot
        self.touched = {}      # slot -> time
        self.req_slot = {}     # engine request id -> slot or -1
        self.steps = 0
        self.last_use = time.time()
        log.warning("dpp_vllm_plugin own: loaded P for %s (V=%d, r=%d, %s, %.0f MiB)", repo, self.V, self.r,
                    self.Pt.dtype, self.Pt.numel() * self.Pt.element_size() / 2**20)

    # ---- slots
    def _alloc(self, key, ea, runner_vocab):
        torch = self.torch
        if key in self.slot_of:
            return self.slot_of[key]
        now = time.time()
        if not self.free:
            for s, t0 in list(self.touched.items()):
                if now - t0 > EXPIRY:
                    self._release(s)
        if not self.free:
            return -1
        s = self.free.pop()
        self.slot_of[key] = s
        self.touched[s] = now
        stop = list(ea.get("dpp_stop") or [])[:NSTOP]
        row = torch.full((NSTOP,), -1, dtype=torch.int64)
        if stop:
            row[:len(stop)] = torch.tensor(stop, dtype=torch.int64)
        self.stop[s] = row.to(self.dev)
        self.maxtok[s] = int(ea.get("dpp_max", 1 << 30))
        f = ea.get("dpp_force")
        self.is_forced[s] = f is not None
        if f is not None:
            f = list(f)[:LMAX]
            fr = torch.zeros(LMAX, dtype=torch.int64)
            fr[:len(f)] = torch.tensor(f, dtype=torch.int64)
            self.forced[s] = fr.to(self.dev)
            self.flen[s] = len(f)
            self.feos[s] = int(ea.get("dpp_force_eos", f[-1] if f else 0))
        return s

    def _release(self, s):
        key = next((k for k, v in self.slot_of.items() if v == s), None)
        if key is not None:
            del self.slot_of[key]
        self.touched.pop(s, None)
        self.accum[s] = 0
        self.count[s] = 0
        self.done[s] = False
        self.is_forced[s] = False
        self.free.append(s)

    def _slot_for_request(self, rid, runner):
        s = self.req_slot.get(rid)
        if s is not None:
            return s
        rs = runner.requests.get(rid)
        sp = getattr(rs, "sampling_params", None)
        ea = getattr(sp, "extra_args", None)
        if not ea or "dpp_own" not in ea:
            s = -1
        else:
            m = re.match(r"(\d+)_", rid)
            idx = int(m.group(1)) if m else 0
            s = self._alloc((ea["dpp_own"], idx), ea, self.V)
        if len(self.req_slot) > 8192:
            live = set(runner.input_batch.req_ids)
            self.req_slot = {k: v for k, v in self.req_slot.items() if k in live}
        self.req_slot[rid] = s
        return s

    # ---- per step
    def step_pre(self, logits, runner):
        """Accumulate P l for the own rows and apply teacher forcing. Returns a context for step_post or None."""
        torch = self.torch
        ids = runner.input_batch.req_ids
        B = len(ids)
        if logits.dim() != 2 or logits.shape[0] != B or logits.shape[1] != self.V:
            return None
        discard = runner.discard_request_mask.np[:B]
        rows, slots = [], []
        for i, rid in enumerate(ids):
            s = self.req_slot.get(rid)
            if s is None:
                s = self._slot_for_request(rid, runner)
            if s >= 0 and not discard[i]:
                rows.append(i)
                slots.append(s)
        if not rows:
            return None
        dev = self.dev
        ri = torch.tensor(rows, device=dev)
        si = torch.tensor(slots, device=dev)
        live = ~self.done[si]
        wl = live.float().unsqueeze(1)
        for a in range(0, len(rows), ROW_CHUNK):
            z = logits[ri[a:a + ROW_CHUNK]]
            if self.cap:
                z = self.cap * torch.atanh((z.float() / self.cap).clamp(-0.999999, 0.999999))
            z = z.to(torch.float16)
            pl = (z @ self.Pt).float() * wl[a:a + ROW_CHUNK]
            self.accum.index_add_(0, si[a:a + ROW_CHUNK], pl)
        self.count.index_add_(0, si, live.int())
        self.steps += 1
        self.last_use = time.time()
        ctx = {"ri": ri, "si": si, "live": live}
        fidx = [k for k, s in enumerate(slots) if self.is_forced[s]]
        if fidx:
            fk = torch.tensor(fidx, device=dev)
            rf, sf = ri[fk], si[fk]
            step = (self.count[sf].long() - live[fk].long())     # steps taken before this one
            tok = self.forced[sf, step.clamp(max=LMAX - 1)]
            tok = torch.where(step >= self.flen[sf].long(), self.feos[sf], tok)
            ctx["rf"], ctx["tok"] = rf, tok
            ctx["zy"] = logits[rf, tok].float()
            logits[rf] = float("-inf")
            logits[rf, tok] = 0
        for s in set(slots):
            self.touched[s] = time.time()
        return ctx

    def step_post(self, ctx, sampled):
        """Mark sequences finished by the sampled token (stop id or max_tokens)."""
        torch = self.torch
        si, ri = ctx["si"], ctx["ri"]
        y = sampled.reshape(sampled.shape[0], -1)[:, 0][ri].long()
        hit = (y.unsqueeze(1) == self.stop[si]).any(1)
        full = self.count[si] >= self.maxtok[si]
        self.done[si] = self.done[si] | hit | full

    # ---- results
    def take(self, items):
        """items: [[tag, n], ...]. Returns per item {"sum": bytes (n x r float32), "count": [n ints]}; frees the slots."""
        torch = self.torch
        out = []
        for tag, n in items:
            slots = [self.slot_of.get((tag, i), -1) for i in range(n)]
            ok = [s for s in slots if s >= 0]
            sums = torch.zeros(n, self.r, dtype=torch.float32)
            cnt = [0] * n
            if ok:
                idx = torch.tensor(ok, device=self.dev)
                a = self.accum[idx].cpu()
                c = self.count[idx].cpu().tolist()
                j = 0
                for i, s in enumerate(slots):
                    if s >= 0:
                        sums[i] = a[j]
                        cnt[i] = c[j]
                        j += 1
                for s in ok:
                    self._release(s)
            out.append({"sum": sums.numpy().tobytes(), "count": cnt})
        return out

    def unload(self):
        """Drop P and the tables from the GPU (the next own request reloads them)."""
        torch = self.torch
        self.Pt = self.accum = self.count = self.done = self.maxtok = self.stop = self.forced = self.flen = self.feos = None
        torch.cuda.empty_cache()
        log.warning("dpp_vllm_plugin own: unloaded P for %s after %.0f s idle", self.repo, IDLE)

    def stats(self):
        torch = self.torch
        return {"repo": self.repo, "V": self.V, "r": self.r, "dtype": str(self.Pt.dtype), "slots_used": CAP - len(self.free),
                "steps": self.steps, "allocated_mib": torch.cuda.memory_allocated() / 2**20,
                "P_mib": self.Pt.numel() * self.Pt.element_size() / 2**20}


def _get(runner):
    """The Own state for this process's model, loaded on the first own request; None if unavailable."""
    st = _ST["state"]
    if st is not None:
        return st
    if _ST["off"]:
        return None
    repo = runner.model_config.model
    m = re.search(r"models--([^/]+)--([^/]+)", str(repo))   # vLLM may report the local snapshot path
    if m:
        repo = f"{m.group(1)}/{m.group(2)}"
    d = _repo_dirs().get(repo)
    if d is None or not os.path.exists(os.path.join(d, "P.safetensors")):
        _ST["off"] = True
        log.warning("dpp_vllm_plugin own: no P for %s, feature off in this process", repo)
        return None
    _ST["state"] = Own(repo, d)
    if _ST["reaper"] is None:
        _ST["reaper"] = threading.Thread(target=_reap, daemon=True, name="dpp-own-reaper")
        _ST["reaper"].start()
    return _ST["state"]


def _reap():
    while True:
        time.sleep(10)
        try:
            with _ST["lock"]:
                st = _ST["state"]
                if st is None:
                    continue
                now = time.time()
                with st.torch.inference_mode():
                    for s_, t0 in list(st.touched.items()):
                        if now - t0 > EXPIRY:
                            st._release(s_)
                    if not st.slot_of and now - st.last_use > IDLE:
                        _ST["state"] = None
                        st.unload()
        except Exception:
            log.exception("dpp_vllm_plugin own: reaper failed")


def has_own_request(runner):
    """Cheap test used before loading anything: does any request in the batch ask for own states?"""
    ib = runner.input_batch
    for rid in ib.req_ids:
        s = _ST["state"].req_slot.get(rid) if _ST["state"] is not None else None
        if s is None:
            rs = runner.requests.get(rid)
            ea = getattr(getattr(rs, "sampling_params", None), "extra_args", None)
            if ea and "dpp_own" in ea:
                return True
            if _ST["state"] is not None:
                _ST["state"].req_slot[rid] = -1
        elif s >= 0:
            return True
    return False


def pre(logits, runner):
    if _ST["off"] and _ST["state"] is None:
        return None
    try:
        if _ST["state"] is None and not has_own_request(runner):
            return None
        with _ST["lock"]:
            st = _get(runner)
        if st is None:
            return None
        with _ST["lock"]:
            ctx = st.step_pre(logits, runner)
        if ctx is not None:
            ctx["st"] = st
        return ctx
    except Exception:
        log.exception("dpp_vllm_plugin own: step failed, feature off")
        _ST["off"] = True
        _ST["state"] = None
        return None


def post(ctx, out):
    try:
        ctx["st"].step_post(ctx, out.sampled_token_ids)
    except Exception:
        log.exception("dpp_vllm_plugin own: post-step failed, feature off")
        _ST["off"] = True


def worker_take(self, items):
    st = _ST["state"]
    if st is None:
        return None
    with _ST["lock"], st.torch.inference_mode():   # the tensors were created under inference mode inside the sampler
        r = st.take(items)
        st.last_use = time.time()
        return r


def worker_stats(self):
    st = _ST["state"]
    if st is None:
        return None
    with st.torch.inference_mode():
        return st.stats()


def install_worker_methods():
    from vllm.v1.worker import gpu_worker
    gpu_worker.Worker.dpp_own_take = worker_take
    gpu_worker.Worker.dpp_own_stats = worker_stats
