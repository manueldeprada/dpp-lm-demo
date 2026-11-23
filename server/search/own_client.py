import json, os, base64, urllib.request, numpy as np, hashlib
ROOT = os.environ.get("DPP_ROOT", os.getcwd())
SECRET = open(os.environ.get("DPP_OWN_SECRET_FILE", ROOT + "/.own_secret")).read().strip()
def embed_own(model, prompt, texts, ended, timeout=900):
    b = {"model": model, "prompt": prompt, "texts": texts, "ended": [bool(e) for e in ended]}
    r = urllib.request.Request("http://localhost:8040/embed_own", json.dumps(b).encode(), {"Content-Type": "application/json", "X-Own-Secret": SECRET})
    import time
    for attempt in range(4):   # a failed slot swap leaves the slot empty, the next request starts the model again
        try:
            d = json.load(urllib.request.urlopen(r, timeout=timeout)); break
        except urllib.error.HTTPError as e:
            if e.code not in (500, 429) or attempt == 3: raise
            time.sleep(8)
    d["V"] = np.frombuffer(base64.b64decode(d["vec"]), "<f4").reshape(len(texts), d["D"]).astype(np.float64)
    return d
def R_of(key, d, r=256):
    seed = int.from_bytes(hashlib.sha256(("own_emb:" + key).encode()).digest()[:8], "little")
    Q, _ = np.linalg.qr(np.random.default_rng(seed).standard_normal((d, r)))
    return Q.T
def pool(key, pid, tp="1.5"):
    return json.load(open(f"{ROOT}/dpp-lm-demo/data/pools/{key}/{pid}_T{tp}.json"))
def rank_avg(x):
    u, inv, cnt = np.unique(x, return_inverse=True, return_counts=True)
    cs = np.cumsum(cnt); return (cs - (cnt - 1) / 2.0)[inv]
def spearman(a, b):
    ra, rb = rank_avg(a), rank_avg(b); ra = ra - ra.mean(); rb = rb - rb.mean()
    return float(ra @ rb / np.sqrt((ra @ ra) * (rb @ rb)))
def unit(X): return X / np.maximum(np.linalg.norm(X, axis=1, keepdims=True), 1e-12)
def triu(G): return G[np.triu_indices(G.shape[0], 1)]
