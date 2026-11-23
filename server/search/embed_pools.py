#!/usr/bin/env python
"""Embed the demo's pools with Qwen3-Embedding-0.6B and build the shared axes.

Usage: embed_pools.py [--data DIR] [--batch 16] [--cpu]
Run with the server venv's python and HF_HUB_OFFLINE=1.

Reads DIR/pools/<model>/<id>_T<tp>.json and writes
  DIR/emb/qwen3-emb/<model>/<id>_T<tp>.bin   float16 little-endian, N x 1024, unit rows
  DIR/basis/qwen3-emb/<id>_T<tp>.json        uncentered PCA axes of all models' rows (as build_model_data.py)
Embedding follows server.py's Embedder for "qwen3-emb": left padding, max_length 512,
last-token pooling, float16 model, empty text replaced by "(empty)", unit norm.
"""
import argparse, glob, json, os, re, sys, time
import numpy as np, torch
from transformers import AutoModel, AutoTokenizer

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import build_model_data as B

REPO = "Qwen/Qwen3-Embedding-0.6B"


class Qwen:
    def __init__(self, cpu=False, batch=16):
        self.dev = "cpu" if cpu else "cuda"
        self.batch = batch
        self.tok = AutoTokenizer.from_pretrained(REPO, padding_side="left")
        self.model = AutoModel.from_pretrained(REPO, dtype=torch.float32 if cpu else torch.float16).to(self.dev).eval()

    @torch.no_grad()
    def __call__(self, texts):
        texts = [t if t.strip() else "(empty)" for t in texts]
        order = sorted(range(len(texts)), key=lambda i: len(texts[i]))
        out = np.zeros((len(texts), 1024), dtype=np.float32)
        for i in range(0, len(order), self.batch):
            idx = order[i:i + self.batch]
            enc = self.tok([texts[j] for j in idx], padding=True, truncation=True, max_length=512,
                           return_tensors="pt").to(self.dev)
            h = self.model(**enc).last_hidden_state[:, -1]
            out[idx] = torch.nn.functional.normalize(h.float(), dim=-1).cpu().numpy()
            del enc, h
            if self.dev == "cuda":
                torch.cuda.empty_cache()
        return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=os.environ.get("DPP_DATA", os.path.join(os.getcwd(), "dpp-lm-demo", "data")))
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--cpu", action="store_true")
    ap.add_argument("--only", default="", help="comma list of model keys to embed; the other models' existing .bin files are reused for the bases")
    a = ap.parse_args()
    only = set(a.only.split(",")) if a.only else None
    emb = Qwen(a.cpu, a.batch)
    t0 = time.time()
    groups = {}  # stem -> [(model, X)]
    files = sorted(glob.glob(os.path.join(a.data, "pools", "*", "*.json")))
    for n, p in enumerate(files):
        model = os.path.basename(os.path.dirname(p))
        stem = os.path.basename(p)[:-5]
        pool = json.load(open(p))
        od = os.path.join(a.data, "emb", "qwen3-emb", model)
        binp = os.path.join(od, stem + ".bin")
        if only is not None and model not in only:
            E16 = np.fromfile(binp, "<f2").reshape(pool["N"], 1024)   # keep the existing file
        else:
            E = emb([s["text"] for s in pool["samples"]])
            assert E.shape == (pool["N"], 1024) and np.isfinite(E).all()
            E16 = E.astype("<f2")
            os.makedirs(od, exist_ok=True)
            open(binp, "wb").write(E16.tobytes())
        X = E16.astype(np.float64)
        X /= np.linalg.norm(X, axis=1, keepdims=True)
        groups.setdefault(stem, []).append((model, X))
        print(f"[{n + 1}/{len(files)}] {model}/{stem} N={pool['N']} {time.time() - t0:.0f}s", flush=True)
    bd = os.path.join(a.data, "basis", "qwen3-emb")
    os.makedirs(bd, exist_ok=True)
    for stem, items in sorted(groups.items()):
        m = re.match(r"(.+)_T([0-9.]+)$", stem)
        id_, tp = m.group(1), m.group(2)
        X = np.vstack([x for _, x in items])
        _, s, Vt = np.linalg.svd(X, full_matrices=False)
        U = Vt[:3].copy()
        B.sign_first(U, X)
        B.sign_rest(U)
        frac = (s[:8] ** 2 / np.sum(X ** 2)).tolist()
        U32 = U.astype("<f4")
        B.check_orth(U32.astype(np.float64), stem)
        B.check_frac(frac, stem)
        rec = {"id": id_, "Tprime": float(tp), "D": 1024, "models": [k for k, _ in items], "rows": int(X.shape[0]),
               "arrows": {"U": B.b64(U32), "frac": [float(x) for x in frac]}, "open": None}
        json.dump(rec, open(os.path.join(bd, stem + ".json"), "w"), separators=(",", ":"))
    print("done", len(files), "pools", len(groups), "bases", f"{time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
