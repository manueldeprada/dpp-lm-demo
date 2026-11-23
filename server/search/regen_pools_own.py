#!/usr/bin/env python
"""Regenerate one model's 14 raw pools with the LM's own states taken from the generation itself.

Usage: regen_pools_own.py MODELKEY   (service up, own available for the model)
Same prompts, seeds and settings as precompute_models.py (N = 200, max_new_tokens 128, seed 1000 + 10 * prompt index
+ (0 for T' 1.0, 1 for 1.5)); the request adds embedders ["own"] and own_raw (uncentered mean projected states, `ownRaw`).
Old raw pools move to pools/models/<key>.before-regen. Used for models whose completions end in whitespace that the
stored text strips (Gemma 2), where teacher forcing the text cannot reproduce the generated token sequence.
"""
import json, os, shutil, sys, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from precompute_models import GALLERY, TPS, POOLS, http
key = sys.argv[1]
old = os.path.join(POOLS, key)
if os.path.exists(old) and not os.path.exists(old + ".before-regen"):
    shutil.move(old, old + ".before-regen")
for gi, (id_, prompt) in enumerate(GALLERY):
    for tp in TPS:
        path = os.path.join(POOLS, key, f"N200_T{tp}", f"{id_}.json")
        if os.path.exists(path):
            continue
        body = {"prompt": prompt, "n": 200, "t_prime": float(tp), "max_new_tokens": 128, "seed": 1000 + 10 * gi + (0 if tp == "1.0" else 1),
                "embedder": "minilm", "embedders": ["own"], "own_raw": True, "model": key}
        t0 = time.time(); rec = http("POST", "/sample", body, timeout=900)
        assert "ownRaw" in rec
        rec.pop("embs", None)
        rec.setdefault("id", id_); rec["modelKey"] = key
        os.makedirs(os.path.dirname(path), exist_ok=True)
        json.dump(rec, open(path + ".tmp", "w")); os.replace(path + ".tmp", path)
        print(f"{key} {id_} T{tp}: {time.time() - t0:.1f}s", flush=True)
