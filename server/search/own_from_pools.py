#!/usr/bin/env python
"""Own-state files from raw pools that carry `ownRaw` (uncentered mean projected states from the generation itself).

Usage: own_from_pools.py MODELKEY [--data DIR]
Reads search/pools/models/<key>/N200_T<tp>/<id>.json, saves own_emb/<key>/raw/<id>_T<tp>.npy, writes mu = mean of all
rows (duplicates included) to own_emb/<key>/mu.npy, and DIR/emb/own/<key>/<id>_T<tp>.bin (centered, unit, float16 LE, N x 256).
"""
import argparse, base64, glob, json, os
import numpy as np
ROOT = os.environ.get("DPP_ROOT", os.getcwd())
ap = argparse.ArgumentParser(); ap.add_argument("key"); ap.add_argument("--data", default=ROOT + "/dpp-lm-demo/data"); a = ap.parse_args()
raws = {}
for f in sorted(glob.glob(f"{ROOT}/search/pools/models/{a.key}/N200_T*/*.json")):
    p = json.load(open(f)); tp = os.path.basename(os.path.dirname(f))[6:]; stem = f"{os.path.basename(f)[:-5]}_T{tp}"
    raws[stem] = np.frombuffer(base64.b64decode(p["ownRaw"]), "<f4").reshape(p["N"], -1).astype(np.float32)
assert len(raws) == 14, len(raws)
os.makedirs(f"{ROOT}/own_emb/{a.key}/raw", exist_ok=True)
for stem, v in raws.items(): np.save(f"{ROOT}/own_emb/{a.key}/raw/{stem}.npy", v)
mu = np.concatenate(list(raws.values())).mean(0).astype(np.float32); np.save(f"{ROOT}/own_emb/{a.key}/mu.npy", mu)
tmp = f"{ROOT}/own_emb/{a.key}/mu.TEMP"
if os.path.exists(tmp): os.remove(tmp)
od = f"{a.data}/emb/own/{a.key}"; os.makedirs(od, exist_ok=True)
for stem, v in raws.items():
    c = v.astype(np.float64) - mu; c /= np.maximum(np.linalg.norm(c, axis=1, keepdims=True), 1e-12)
    open(f"{od}/{stem}.bin", "wb").write(c.astype("<f2").tobytes())
print("wrote", len(raws), "files, mu norm", float(np.linalg.norm(mu)))
