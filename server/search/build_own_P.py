#!/usr/bin/env python
"""Build the logit-to-projected-state matrix P for one model (CPU only).

Usage: CUDA_VISIBLE_DEVICES= nice -n 19 build_own_P.py MODELKEY [--r 256]

The sampler sees logits l = W h - c*1 (or raw logits W h) for the model's final hidden state h.
With A = [W | 1], h is the first d entries of (A^T A)^-1 A^T l. We store
    P = R X_h = [R | 0] (A^T A)^-1 A^T      (r x V),   R: r x d with orthonormal rows,
so that P l is the random projection R h of the final hidden state. P is stored transposed
(V x r, float16) in own_emb/<key>/P.safetensors with meta.json (seed, r, d, V, cond, dtype, repo).
R is the first r columns of Q from the QR factorization of a Gaussian d x r matrix drawn with
numpy default_rng(seed), seed = first 8 bytes (little endian) of sha256("own_emb:" + key).
"""
import argparse, glob, hashlib, json, os, time
import numpy as np
from huggingface_hub import snapshot_download
from safetensors import safe_open
from safetensors.numpy import save_file

ROOT = os.environ.get("DPP_ROOT", os.getcwd())
os.environ.setdefault("HF_HUB_OFFLINE", "1")


def tensor_slices(d, name):
    """Tensor `name`, or (for checkpoints that keep a text-model prefix such as `language_model.`) the one key ending in `.name`."""
    for f in sorted(glob.glob(d + "/*.safetensors")):
        g = safe_open(f, "pt")
        if name in g.keys():
            return g, name
        hits = [k for k in g.keys() if k.endswith("." + name) and "vision" not in k]
        if hits:
            return g, hits[0]
    return None, None


def find_W(repo):
    d = snapshot_download(repo, local_files_only=True)
    cfg = json.load(open(d + "/config.json")); tc = cfg.get("text_config", cfg)
    emb = tensor_slices(d, "model.embed_tokens.weight")
    head = tensor_slices(d, "lm_head.weight")
    info = {"tie_word_embeddings": tc.get("tie_word_embeddings"), "has_lm_head_tensor": head[0] is not None,
            "softcap": tc.get("final_logit_softcapping"), "logits_scaling": tc.get("logits_scaling"),
            "logit_scale": tc.get("logit_scale"), "quant": cfg.get("quantization_config", {}).get("quant_method")}
    if head[0] is not None:
        g, n = head
        if emb[0] is not None:
            e = emb[0].get_tensor(emb[1]); h = g.get_tensor(n)
            info["lm_head_equals_embed"] = bool((e == h).all())
        info["source"] = "lm_head.weight"
    else:
        g, n = emb; info["source"] = "model.embed_tokens.weight (tied)"
    info["dtype"] = str(g.get_slice(n).get_dtype()); info["shape"] = list(g.get_slice(n).get_shape())
    # a final soft-cap c is undone in the plugin (z = c atanh(t / c)); scaling is not handled
    assert not info["logits_scaling"] and not info["logit_scale"], info
    return g, n, info


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("key"); ap.add_argument("--repo", default=None); ap.add_argument("--r", type=int, default=256); ap.add_argument("--chunk", type=int, default=8192)
    a = ap.parse_args()
    ms = {m["key"]: m for m in json.load(open(ROOT + "/dpp-lm-demo/data/models.json"))["models"]}
    repo = a.repo or ms[a.key]["repo"]
    g, n, info = find_W(repo)
    sl = g.get_slice(n); V, d = sl.get_shape(); r = a.r
    t0 = time.time()
    def chunks():
        for s in range(0, V, a.chunk):
            w = sl[s:min(s + a.chunk, V)]
            W = w.float().numpy().astype(np.float64) if hasattr(w, "float") else np.asarray(w, dtype=np.float64)
            yield s, np.concatenate([W, np.ones((W.shape[0], 1))], 1)
    G = np.zeros((d + 1, d + 1))
    for s, A in chunks():
        G += A.T @ A
    t1 = time.time()
    ev = np.linalg.eigvalsh(G); cond = float(np.sqrt(ev[-1] / ev[0]))
    seed = int.from_bytes(hashlib.sha256(("own_emb:" + a.key).encode()).digest()[:8], "little")
    Q, _ = np.linalg.qr(np.random.default_rng(seed).standard_normal((d, r)))
    R = Q.T
    Raug = np.concatenate([R, np.zeros((r, 1))], 1)
    M = np.linalg.solve(G, Raug.T).T          # r x (d+1), G symmetric
    Pt = np.zeros((V, r), np.float32)
    for s, A in chunks():
        Pt[s:s + A.shape[0]] = (A @ M.T).astype(np.float32)
    # fp16 rounding check on logits of random hidden states (CPU, fp32 accumulation)
    rng = np.random.default_rng(1)
    Wf = sl[0:V].float().numpy() if hasattr(sl[0:2], "float") else np.asarray(sl[0:V], np.float32)
    errs = []
    P16 = Pt.astype(np.float16)
    for _ in range(8):
        h = rng.standard_normal(d).astype(np.float32); h *= 3.0 / (np.linalg.norm(Wf @ h) / np.sqrt(V))
        z = Wf @ h
        l = z - (z.max() + np.log(np.exp(z - z.max()).sum()))
        for lv in (l, z):
            ref = lv.astype(np.float64) @ Pt.astype(np.float64)
            got = lv.astype(np.float64) @ P16.astype(np.float64)
            errs.append(float(np.linalg.norm(got - ref) / np.linalg.norm(ref)))
        errs.append(float(np.linalg.norm(Pt.T.astype(np.float64) @ z - R @ h) / np.linalg.norm(R @ h)))  # recovery of R h from raw logits (fp32 P)
    rec = errs[2::3]; rnd = errs[0::3] + errs[1::3]
    store = "float16" if max(rnd) <= 1e-3 else "float32"
    od = f"{ROOT}/own_emb/{a.key}"; os.makedirs(od, exist_ok=True)
    save_file({"Pt": np.ascontiguousarray(Pt.astype(np.float16 if store == "float16" else np.float32))}, od + "/P.safetensors")
    meta = dict(key=a.key, repo=repo, seed=seed, seed_rule="first 8 bytes LE of sha256('own_emb:'+key)", r=r, d=int(d), V=int(V), cond_A=cond,
                dtype=store, fp16_rel_err_max=max(rnd), recover_rel_err_fp32_max=max(rec), layout="Pt: V x r, P l = l @ Pt", source=info,
                seconds=round(time.time() - t0, 1), gram_seconds=round(t1 - t0, 1))
    json.dump(meta, open(od + "/meta.json", "w"), indent=1)
    print(json.dumps(meta))


if __name__ == "__main__":
    main()
