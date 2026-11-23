#!/usr/bin/env python
"""Sample 200-completion pools for every available model, then build the gallery data.

Calls the running server (GET /status, POST /sample), so run it only when the
server is free. Raw pools: $DPP_ROOT/search/pools/models/<key>/N200_T<tp>/<id>.json
Output: $DPP_ROOT/search/model_data/ (via build_model_data.py).
"""
import json, os, subprocess, sys, time, urllib.request

BASE = "http://127.0.0.1:8040"
HERE = os.path.dirname(os.path.abspath(__file__))
POOLS = os.path.join(HERE, "pools", "models")
OUT = os.environ.get("MODEL_DATA_OUT", os.path.join(HERE, "model_data"))
TPS = ["1.0", "1.5"]
# data/index.json entries (id, prompt), in gallery order
GALLERY = [
    ("story", "Write the opening sentence of a short story. Reply with one sentence only."),
    ("city", "Name a city worth visiting and say why, in one sentence."),
    ("animal", "Name an animal and one surprising fact about it, in one sentence."),
    ("bike", "Give one creative use for an old bicycle, in one sentence."),
    ("firstdate", "Suggest an idea for a first date, in one sentence."),
    ("gift", "Suggest a birthday gift for a friend, in one sentence."),
    ("dinner", "Suggest something to cook for dinner tonight, in one sentence."),
]


def dtype_of(key):
    return "awq-int4" if key in ("qwen3-14b", "gemma-3-4b", "gemma-2-9b") else "bf16"


def http(method, path, body=None, timeout=900):
    data = None if body is None else json.dumps(body).encode()
    req = urllib.request.Request(BASE + path, data=data, method=method,
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())


def main():
    status = http("GET", "/status", timeout=30)
    models = status["models"]
    if isinstance(models, dict):
        models = [dict(v, key=v.get("key", k)) for k, v in models.items()]
    models = [m for m in models if m.get("state") != "unavailable"]
    models.sort(key=lambda m: m["params"])
    for m in models:
        key = m["key"]
        for gi, (id_, prompt) in enumerate(GALLERY):
            for tp in TPS:
                path = os.path.join(POOLS, key, f"N200_T{tp}", f"{id_}.json")
                if os.path.exists(path):
                    continue
                body = {"prompt": prompt, "n": 200, "t_prime": float(tp),
                        "max_new_tokens": 128,
                        "seed": 1000 + 10 * gi + (0 if tp == "1.0" else 1),
                        "embedder": "minilm", "model": key}
                for attempt in (0, 1):
                    try:
                        t0 = time.time()
                        rec = http("POST", "/sample", body, timeout=900)
                        break
                    except Exception as e:
                        print(f"error {key} {id_} T{tp} attempt {attempt}: {e}", file=sys.stderr)
                        if attempt:
                            raise
                rec.setdefault("id", id_)
                rec["modelKey"] = key
                os.makedirs(os.path.dirname(path), exist_ok=True)
                with open(path + ".tmp", "w") as f:
                    json.dump(rec, f)
                os.replace(path + ".tmp", path)
                print(f"{key} {id_} T{tp}: {time.time() - t0:.1f}s", flush=True)
    src = {"models": {}}
    for m in models:
        key = m["key"]
        src["models"][key] = {
            "name": m["name"], "params": m["params"], "repo": m["repo"],
            "dtype": dtype_of(key),
            "dirs": {tp: os.path.join(POOLS, key, f"N200_T{tp}") for tp in TPS}}
    src_path = os.path.join(HERE, "model_data_src.json")
    with open(src_path, "w") as f:
        json.dump(src, f, indent=1)
    subprocess.check_call([sys.executable, os.path.join(HERE, "build_model_data.py"),
                           "--src", src_path, "--out", OUT])


if __name__ == "__main__":
    main()
