import os, json, base64, numpy as np
ROOT = os.environ.get("DPP_ROOT", os.getcwd())
IDS = ["animal", "bike", "city", "dinner", "firstdate", "gift", "story"]
NMAX = 100
TP = "1.5"

def unit(X):
    return X / np.maximum(np.linalg.norm(X, axis=1, keepdims=True), 1e-12)

def load_pool(key, pid, tp=TP, nmax=NMAX):
    p = json.load(open(f"{ROOT}/dpp-lm-demo/data/pools/{key}/{pid}_T{tp}.json"))
    N, D = p["N"], p["D"]
    mini = np.frombuffer(base64.b64decode(p["emb"]), dtype="<f4").reshape(N, D)[:nmax].astype(np.float64)
    q = np.fromfile(f"{ROOT}/dpp-lm-demo/data/emb/qwen3-emb/{key}/{pid}_T{tp}.bin", dtype="<f2").reshape(N, 1024)[:nmax].astype(np.float64)
    texts = [s["text"] for s in p["samples"][:nmax]]
    return texts, unit(mini), unit(q)

def rank_avg(x):
    u, inv, cnt = np.unique(x, return_inverse=True, return_counts=True)
    cs = np.cumsum(cnt); avg = cs - (cnt - 1) / 2.0
    return avg[inv]

def spearman(a, b):
    ra, rb = rank_avg(a), rank_avg(b)
    ra -= ra.mean(); rb -= rb.mean()
    return float((ra @ rb) / np.sqrt((ra @ ra) * (rb @ rb)))

def triu(G):
    return G[np.triu_indices(G.shape[0], 1)]

def knn(G, k=10):
    S = G.copy(); np.fill_diagonal(S, -np.inf)
    return np.argpartition(-S, k, axis=1)[:, :k]

def overlap(G1, G2, k=10):
    a, b = knn(G1, k), knn(G2, k)
    return float(np.mean([len(set(x) & set(y)) / k for x, y in zip(a, b)]))

def greedy(G, K, tol=1e-12):
    """greedy max-log-det with Cholesky-style updates; returns indices, #steps where the best gain <= tol"""
    n = G.shape[0]; d = np.diag(G).astype(np.float64).copy(); C = np.zeros((K, n)); S = []; bad = 0
    for t in range(K):
        j = int(np.argmax(d))
        if d[j] <= tol: bad += 1
        S.append(j)
        dj = max(d[j], 1e-300)
        e = (G[j] - C[:t].T @ C[:t, j]) / np.sqrt(dj) if t else G[j] / np.sqrt(dj)
        C[t] = e; d = d - e ** 2; d[S] = -np.inf
    return S, bad

def logdet(G, S):
    sign, ld = np.linalg.slogdet(G[np.ix_(S, S)])
    return ld if sign > 0 else -np.inf

def random_logdets(G, K, sets):
    return np.array([logdet(G, s) for s in sets])

def fit_post(X, mode):
    """X: training vectors (raw). Returns function applying post-processing and unit-normalizing."""
    if mode == "P0":
        return lambda Y: unit(Y)
    mu = X.mean(0)
    if mode == "P1":
        return lambda Y: unit(Y - mu)
    Xc = X - mu
    U, s, Vt = np.linalg.svd(Xc, full_matrices=False)
    if mode.startswith("P2k"):
        k = int(mode[3:]); V = Vt[:k]
        return lambda Y: unit((Y - mu) - ((Y - mu) @ V.T) @ V)
    if mode.startswith("P3w"):
        r = int(mode[3:]); r = min(r, len(s)); V = Vt[:r]; sc = s[:r] / np.sqrt(len(X) - 1)
        return lambda Y: unit(((Y - mu) @ V.T) / sc)
    raise ValueError(mode)

CANDS = ["A1", "A2", "A3", "B", "Clast", "Ceos", "D25", "D50", "D75", "Ein", "Eout"]
POSTS = ["P0", "P1", "P2k1", "P2k3", "P3w64", "P3w128"]
