#!/usr/bin/env python
"""Build the multi-model gallery data (pools, shared bases, models.json).

Usage: build_model_data.py --src mapping.json --out DIR

mapping.json:
  {"models": {"<key>": {"name": ..., "params": 8.2, "repo": ..., "dtype": ...,
                        "dirs": {"1.0": "<dir of raw pools>", "1.5": "<dir>"}}, ...}}
Raw pools are server-format files <id>.json (gallery format plus embs, timing,
group; modelKey optional). Optional top-level "ids" overrides the gallery ids.
"""
import argparse, base64, json, os, sys
import numpy as np

IDS = ["story", "city", "animal", "bike", "firstdate", "gift", "dinner"]
TPS = ["1.0", "1.5"]
N, D = 200, 384
KEYS = ["id", "prompt", "model", "modelKey", "modelName", "dtype", "embedder",
        "Tprime", "maxNewTokens", "N", "D", "seed", "samples", "emb"]


def b64(a):
    return base64.b64encode(np.ascontiguousarray(a, dtype="<f4").tobytes()).decode()


def unb64(s, shape):
    a = np.frombuffer(base64.b64decode(s), dtype="<f4")
    assert a.size == int(np.prod(shape)), (a.size, shape)
    return a.reshape(shape).astype(np.float64)


def sign_first(U, X):
    # axis 1: mean(X.u) > 0
    if (X @ U[0]).mean() < 0:
        U[0] = -U[0]


def sign_rest(U):
    for k in range(1, U.shape[0]):
        if U[k][np.argmax(np.abs(U[k]))] < 0:
            U[k] = -U[k]


def sign_open(U):
    for k in range(U.shape[0]):
        if U[k][np.argmax(np.abs(U[k]))] < 0:
            U[k] = -U[k]


def check_frac(frac, what):
    f = np.asarray(frac)
    assert np.all(np.diff(f) <= 1e-12), f"{what}: frac not non-increasing"
    assert f.sum() <= 1 + 1e-6, f"{what}: frac sum {f.sum()}"


def check_orth(U, what):
    err = np.abs(U @ U.T - np.eye(U.shape[0])).max()
    assert err < 1e-5, f"{what}: |UU^T-I|={err}"


def make_basis(X, id_, tp, keys):
    M, Dd = X.shape
    # arrows
    _, s, Vt = np.linalg.svd(X, full_matrices=False)
    ev = s ** 2
    U = Vt[:3].copy()
    sign_first(U, X)
    sign_rest(U)
    frac_a = (ev[:8] / np.sum(X ** 2)).tolist()
    m = U[0].copy()
    # open
    a = X @ m
    pos = a > 0
    g = X[pos] / a[pos, None] - m
    r = np.linalg.norm(g, axis=1)
    thr = max(3 * np.median(r), np.percentile(r, 95))
    bulk = g[r <= thr]
    gbar = bulk.mean(0)
    C = np.cov(bulk - gbar, rowvar=False, bias=False) if False else \
        ((bulk - gbar).T @ (bulk - gbar)) / (len(bulk) - 1)
    w, V = np.linalg.eigh(C)
    order = np.argsort(w)[::-1]
    w, V = w[order], V[:, order]
    Uo = V[:3].T.copy() if False else V[:, :3].T.copy()
    sign_open(Uo)
    frac_o = (w[:8] / np.trace(C)).tolist()
    # float32 round trip then verify
    U32 = U.astype("<f4"); Uo32 = Uo.astype("<f4")
    m32 = m.astype("<f4"); gb32 = gbar.astype("<f4")
    tag = f"{id_} T{tp}"
    check_orth(U32.astype(np.float64), tag + " arrows")
    check_orth(Uo32.astype(np.float64), tag + " open")
    assert abs(np.linalg.norm(m32.astype(np.float64)) - 1) < 1e-5, tag + " m not unit"
    for v in (frac_a, frac_o):
        assert len(v) == 8
    check_frac(frac_a, tag + " arrows"); check_frac(frac_o, tag + " open")
    rec = {"id": id_, "Tprime": float(tp), "D": Dd, "models": keys, "rows": int(M),
           "arrows": {"U": b64(U32), "frac": [float(x) for x in frac_a]},
           "open": {"m": b64(m32), "gbar": b64(gb32), "U": b64(Uo32),
                    "frac": [float(x) for x in frac_o],
                    "bulkRule": "r <= max(3*median, p95)"}}
    return rec, int(len(bulk)), frac_a, frac_o


def load_pool(path, key, cfg, tp, id_):
    raw = json.load(open(path))
    S = raw["samples"]
    assert len(S) == N and raw["N"] == N, f"{path}: N"
    assert raw["D"] == D, f"{path}: D"
    for s in S:
        assert np.isfinite(s["logp"]) and np.isfinite(s["logq"]), f"{path}: non-finite"
    E = unb64(raw["emb"], (N, D))
    assert np.all(np.isfinite(E)), f"{path}: non-finite emb"
    nrm = np.linalg.norm(E, axis=1)
    assert np.abs(nrm - 1).max() < 1e-4, f"{path}: rows not unit ({nrm.min()},{nrm.max()})"
    rec = {
        "id": id_, "prompt": raw["prompt"], "model": cfg["repo"], "modelKey": key,
        "modelName": cfg["name"], "dtype": cfg["dtype"], "embedder": raw["embedder"],
        "Tprime": float(tp), "maxNewTokens": raw["maxNewTokens"], "N": N, "D": D,
        "seed": raw["seed"],
        "samples": [{"text": s["text"], "logp": round(float(s["logp"]), 5),
                     "logq": round(float(s["logq"]), 5), "nTokens": s["nTokens"],
                     "ended": s["ended"]} for s in S],
        "emb": raw["emb"],
    }
    assert list(rec) == KEYS
    return rec, E


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    src = json.load(open(a.src))
    ids = src.get("ids", IDS)
    models = sorted(src["models"].items(), key=lambda kv: kv[1]["params"])
    out = a.out
    os.makedirs(out, exist_ok=True)
    examples = {i: {} for i in ids}
    for id_ in ids:
        for tp in TPS:
            rows, keys = [], []
            for key, cfg in models:
                d = cfg["dirs"].get(tp)
                if not d:
                    continue
                p = os.path.join(os.path.expanduser(d), id_ + ".json")
                if not os.path.exists(p):
                    continue
                rec, E = load_pool(p, key, cfg, tp, id_)
                od = os.path.join(out, "pools", key)
                os.makedirs(od, exist_ok=True)
                with open(os.path.join(od, f"{id_}_T{tp}.json"), "w") as f:
                    json.dump(rec, f, ensure_ascii=False, separators=(",", ":"))
                rows.append(E / np.linalg.norm(E, axis=1, keepdims=True))
                keys.append(key)
                examples[id_].setdefault(key, []).append(tp)
            if not rows:
                continue
            X = np.vstack(rows)
            brec, nb, fa, fo = make_basis(X, id_, tp, keys)
            os.makedirs(os.path.join(out, "basis"), exist_ok=True)
            with open(os.path.join(out, "basis", f"{id_}_T{tp}.json"), "w") as f:
                json.dump(brec, f, separators=(",", ":"))
            fmt = lambda v: "[" + ", ".join(f"{x:.4f}" for x in v[:3]) + "]"
            print(f"{id_:10s} T'={tp}  rows={X.shape[0]:4d} bulk={nb:4d} "
                  f"arrows_frac={fmt(fa)} open_frac={fmt(fo)} models={','.join(keys)}")
    examples = {i: v for i, v in examples.items() if v}
    meta = {"models": [{"key": k, "name": c["name"], "params": c["params"],
                        "repo": c["repo"], "dtype": c["dtype"]} for k, c in models],
            "Tprimes": TPS, "examples": examples}
    with open(os.path.join(out, "models.json"), "w") as f:
        json.dump(meta, f, indent=1)
    print("wrote", out)


if __name__ == "__main__":
    sys.exit(main())
