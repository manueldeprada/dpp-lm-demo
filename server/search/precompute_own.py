#!/usr/bin/env python
"""Own-state vectors for every pool of the demo, by teacher forcing through the running service.

Usage: .venv/bin/python search/precompute_own.py [--models k1,k2] [--data DIR]
For each model: POST /embed_own for every pool (200 texts, ended flags), save the uncentered mean projected
states to own_emb/<key>/raw/<id>_T<tp>.npy, set mu = mean of all rows over the model's pools (duplicates
included) in own_emb/<key>/mu.npy, then write centered, unit-normalized float16 rows to
DATA/emb/own/<key>/<id>_T<tp>.bin (N x 256, little endian, sample order). Needs DPP_OWN_EMB=1 service,
P files from build_own_P.py and the owner key file (DPP_OWN_SECRET_FILE, default $DPP_ROOT/.own_secret).
"""
import argparse, glob, json, os, sys, time
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from own_client import embed_own, ROOT

ap = argparse.ArgumentParser()
ap.add_argument("--models", default="")
ap.add_argument("--data", default=ROOT + "/dpp-lm-demo/data")
a = ap.parse_args()
ms = json.load(open(a.data + "/models.json"))["models"]
order = sorted(ms, key=lambda m: (m["key"] != "qwen3-14b", m["params"]))   # resident first, then by size
keys = [m["key"] for m in order if not a.models or m["key"] in a.models.split(",")]
log = {}
for key in keys:
    files = sorted(glob.glob(f"{a.data}/pools/{key}/*.json"))
    rawdir = f"{ROOT}/own_emb/{key}/raw"; os.makedirs(rawdir, exist_ok=True)
    t0 = time.time(); raws = {}; info = {"pools": 0, "empty_text": 0, "first_step_only": 0, "forced_mismatch": 0, "samples": 0}
    for f in files:
        stem = os.path.basename(f)[:-5]; p = json.load(open(f)); S = p["samples"]
        d = embed_own(key, p["prompt"], [s["text"] for s in S], [s["ended"] for s in S])
        raws[stem] = d["V"].astype(np.float32); np.save(f"{rawdir}/{stem}.npy", raws[stem])
        info["pools"] += 1; info["samples"] += len(S)
        info["empty_text"] += sum(1 for s in S if not s["text"].strip())
        info["first_step_only"] += sum(1 for c in d["count"] if c == 1)
        info["forced_mismatch"] += len(S) - sum(d["forcedOk"])
        assert min(d["count"]) >= 1 and np.isfinite(raws[stem]).all()
    allv = np.concatenate(list(raws.values()))
    mu = allv.mean(0).astype(np.float32)
    np.save(f"{ROOT}/own_emb/{key}/mu.npy", mu)
    tmp = f"{ROOT}/own_emb/{key}/mu.TEMP"
    if os.path.exists(tmp): os.remove(tmp)
    od = f"{a.data}/emb/own/{key}"; os.makedirs(od, exist_ok=True)
    for stem, v in raws.items():
        c = v.astype(np.float64) - mu
        c /= np.maximum(np.linalg.norm(c, axis=1, keepdims=True), 1e-12)
        open(f"{od}/{stem}.bin", "wb").write(c.astype("<f2").tobytes())
    info["seconds"] = round(time.time() - t0, 1); log[key] = info
    print(key, info, flush=True)
json.dump(log, open(ROOT + "/own_emb/precompute_log.json", "w"), indent=1)
