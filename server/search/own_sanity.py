#!/usr/bin/env python
"""Sanity numbers for the own embedder on the finished files (all pools of the demo).

Per model and overall: mean pairwise cosine (all completions), Spearman correlation of pairwise cosines
against MiniLM and Qwen3-Embedding (and MiniLM against Qwen3-Embedding as the reference), and the K = 3
DPP utility of the research study's analyze.py (metric 4): greedy log-det subset chosen under embedder X is
scored under reference embedder Y as (logdet_Y(S_X) - mean random logdet_Y) / (logdet_Y(greedy_Y) - mean random),
on the unique completions of each pool (200 random 3-subsets, seed 0). Mean +- standard error over pools.
Usage: CUDA_VISIBLE_DEVICES= .venv/bin/python search/own_sanity.py [--data DIR]
"""
import argparse, base64, glob, json, os, sys
import numpy as np
os.environ["CUDA_VISIBLE_DEVICES"] = ""
ROOT = os.environ.get("DPP_ROOT", os.getcwd())
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from own_common import spearman, triu, greedy, logdet, random_logdets
ap = argparse.ArgumentParser(); ap.add_argument("--data", default=ROOT + "/dpp-lm-demo/data"); a = ap.parse_args()
ms = [m["key"] for m in json.load(open(a.data + "/models.json"))["models"]]
rng = np.random.default_rng(0)
def unit(X): return X / np.maximum(np.linalg.norm(X, axis=1, keepdims=True), 1e-12)
rows = {}
for key in ms:
    for f in sorted(glob.glob(f"{a.data}/pools/{key}/*.json")):
        stem = os.path.basename(f)[:-5]; p = json.load(open(f)); N = p["N"]
        mini = unit(np.frombuffer(base64.b64decode(p["emb"]), "<f4").reshape(N, p["D"]).astype(np.float64))
        qw = unit(np.fromfile(f"{a.data}/emb/qwen3-emb/{key}/{stem}.bin", "<f2").reshape(N, 1024).astype(np.float64))
        ow = unit(np.fromfile(f"{a.data}/emb/own/{key}/{stem}.bin", "<f2").reshape(N, 256).astype(np.float64))
        G = {"own": ow @ ow.T, "mini": mini @ mini.T, "qwen": qw @ qw.T}
        first = {}
        for i, s in enumerate(p["samples"]): first.setdefault(s["text"], i)
        u = np.array(sorted(first.values())); Gu = {k: g[np.ix_(u, u)] for k, g in G.items()}
        r = {"cos_own": triu(G["own"]).mean(), "cos_mini": triu(G["mini"]).mean(), "cos_qwen": triu(G["qwen"]).mean(),
             "sp_own_mini": spearman(triu(G["own"]), triu(G["mini"])), "sp_own_qwen": spearman(triu(G["own"]), triu(G["qwen"])),
             "sp_mini_qwen": spearman(triu(G["mini"]), triu(G["qwen"])), "uniq": len(u)}
        K = 3; sets = [rng.choice(len(u), K, replace=False) for _ in range(200)]
        rand = {k: random_logdets(Gu[k], K, sets).mean() for k in Gu}
        best = {k: logdet(Gu[k], greedy(Gu[k], K)[0]) for k in Gu}
        sel = {k: greedy(Gu[k], K)[0] for k in Gu}
        def util(x, y): return (logdet(Gu[y], sel[x]) - rand[y]) / (best[y] - rand[y])
        r.update(dpp_own_under_mini=util("own", "mini"), dpp_own_under_qwen=util("own", "qwen"), dpp_mini_under_qwen=util("mini", "qwen"),
                 dpp_qwen_under_mini=util("qwen", "mini"), dpp_mini_under_own=util("mini", "own"), dpp_qwen_under_own=util("qwen", "own"))
        rows.setdefault(key, []).append(r)
cols = list(next(iter(rows.values()))[0])
def agg(rs, c):
    v = np.array([r[c] for r in rs]); return v.mean(), v.std(ddof=1) / np.sqrt(len(v)) if len(v) > 1 else 0.0
out = {"models": {k: {c: agg(rs, c) for c in cols} for k, rs in rows.items()}, "overall": {c: agg([r for rs in rows.values() for r in rs], c) for c in cols}}
json.dump(out, open(ROOT + "/own_emb/sanity.json", "w"), indent=1)
def line(name, d): return f"| {name} | " + " | ".join("%.3f±%.3f" % d[c] for c in ("cos_own", "cos_mini", "cos_qwen", "sp_own_mini", "sp_own_qwen", "sp_mini_qwen", "dpp_own_under_mini", "dpp_own_under_qwen", "dpp_mini_under_qwen", "dpp_qwen_under_mini")) + f" | {d['uniq'][0]:.0f} |"
print("| model | cos own | cos MiniLM | cos Qwen3-Emb | Sp own~MiniLM | Sp own~Qwen3 | Sp MiniLM~Qwen3 | DPP own->MiniLM | DPP own->Qwen3 | DPP MiniLM->Qwen3 | DPP Qwen3->MiniLM | unique/200 |")
for k, d in out["models"].items(): print(line(k, d))
print(line("overall (pools)", out["overall"]))
print("extra: DPP MiniLM->own %.3f, Qwen3->own %.3f (overall)" % (out["overall"]["dpp_mini_under_own"][0], out["overall"]["dpp_qwen_under_own"][0]))
