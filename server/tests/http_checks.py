"""Live checks of the public API hardening against localhost:8040 (acts as a public client; X-Real-IP picks the client key).
Usage: .venv/bin/python tests/http_checks.py [basic|limits]"""
import http.client, json, sys, threading, time
H = "127.0.0.1"
def call(path, body=None, headers=None, raw=None, method="POST", ip=None, ctype="application/json", timeout=300):
    c = http.client.HTTPConnection(H, 8040, timeout=timeout)
    h = {"Content-Type": ctype} if ctype else {}
    if ip: h["X-Real-IP"] = ip
    h.update(headers or {})
    data = raw if raw is not None else (json.dumps(body).encode() if body is not None else None)
    c.request(method, path, body=data, headers=h)
    r = c.getresponse(); t = r.read().decode(errors="replace")
    return r.status, t[:160].replace("\n", " "), dict(r.getheaders())
def show(name, r, want):
    st = r[0]; ok = st == want if isinstance(want, int) else st in want
    print(("PASS " if ok else "FAIL ") + f"{name}: {r[0]} {r[1][:110]}")
    return ok
ok = True
def t(name, r, want):
    global ok; ok &= show(name, r, want)
if sys.argv[1] == "basic":
    base = {"prompt": "Say hi.", "n": 2, "max_new_tokens": 8}
    t("GET /docs", call("/docs", method="GET"), 404); t("GET /openapi.json", call("/openapi.json", method="GET"), 404); t("GET /redoc", call("/redoc", method="GET"), 404)
    t("form content type", call("/sample", raw=b"prompt=hi", ctype="application/x-www-form-urlencoded"), 415)
    t("no content type", call("/sample", raw=b"{}", ctype=None), 415)
    t("invalid json", call("/sample", raw=b"{nope"), 400)
    t("top-level list", call("/sample", raw=b"[1,2]"), 400)
    t("empty body", call("/sample", raw=b""), 400)
    t("NaN t_prime", call("/sample", raw=b'{"prompt":"x","t_prime":NaN}'), 400)
    t("Infinity t_prime", call("/sample", raw=b'{"prompt":"x","t_prime":Infinity}'), 400)
    t("t_prime 1e999", call("/sample", raw=b'{"prompt":"x","t_prime":1e999}'), 400)
    t("n 1e999", call("/sample", raw=b'{"prompt":"x","n":1e999}'), 400)
    t("n string", call("/sample", dict(base, n="abc")), 400)
    t("n list", call("/sample", dict(base, n=[1])), 400)
    t("n nested list", call("/sample", dict(base, n=[[1]])), 400)
    t("seed NaN", call("/sample", raw=b'{"prompt":"x","seed":NaN}'), 400)
    t("max_new_tokens object", call("/sample", dict(base, max_new_tokens={"a": 1})), 400)
    t("prompt number", call("/sample", dict(base, prompt=5)), 400)
    t("prompt list", call("/sample", dict(base, prompt=["a"])), 400)
    t("model list", call("/sample", dict(base, model=["qwen3-14b"])), 400)
    t("model unknown", call("/sample", dict(base, model="nope")), 400)
    t("embedder number", call("/sample", dict(base, embedder=3)), 400)
    t("embedder unknown", call("/sample", dict(base, embedder="nope")), 400)
    t("embedders without owner key", call("/sample", dict(base, embedders=["minilm"])), 403)
    t("own_raw without owner key", call("/sample", dict(base, own_raw=True)), 403)
    t("return_ids without owner key", call("/sample", dict(base, return_ids=True)), 403)
    t("oversize body (content-length)", call("/sample", raw=b'{"prompt":"' + b"a" * 70000 + b'"}'), 413)
    t("embed oversize body", call("/embed", raw=b'{"texts":["' + b"a" * (1536 * 1024 + 10) + b'"]}'), 413)
    t("embed texts missing", call("/embed", {"embedder": "minilm"}), 400)
    t("embed texts not list", call("/embed", {"texts": "abc"}), 400)
    t("embed texts with number", call("/embed", {"texts": [1]}), 400)
    t("embed 201 texts", call("/embed", {"texts": ["a"] * 201}), 400)
    t("embed text too long", call("/embed", {"texts": ["a" * 2001]}), 400)
    t("embed embedder list", call("/embed", {"texts": ["a"], "embedder": ["minilm"]}), 400)
    t("embed ended wrong length", call("/embed", {"texts": ["a"], "embedder": "own", "model": "qwen3-14b", "prompt": "p", "ended": [True, False]}), 400)
    t("embed minilm ok", call("/embed", {"texts": ["hello", "world"], "embedder": "minilm"}), 200)
    # owner key
    sec = open(__import__("os").environ.get("DPP_OWN_SECRET_FILE", ".own_secret")).read().strip()
    o = {"X-Own-Secret": sec}
    t("owner: duplicates deduped", call("/sample", dict(base, embedders=["minilm", "minilm", "minilm"]), headers=o), 200)
    t("owner: 4 distinct embedders", call("/sample", dict(base, embedders=["minilm", "qwen3-emb", "mpnet", "own"]), headers=o), 400)
    t("owner: embedders 100 duplicates", call("/sample", dict(base, embedders=["minilm"] * 100), headers=o), 200)
    t("wrong owner key", call("/sample", dict(base, own_raw=True), headers={"X-Own-Secret": "x" * 40}), 403)
    t("embed_own without key", call("/embed_own", {"model": "qwen3-14b"}), 403)
    t("normal sample ok", call("/sample", base, ip="203.0.113.1"), 200)
    print("ALL PASS" if ok else "SOME FAILED")
if sys.argv[1] == "limits":
    quick = {"prompt": "Say hi.", "n": 200, "max_new_tokens": 256}   # costs 51,200 tokens, answers in a few tokens
    # 1 concurrent request per client
    res = {}
    def go(k, ip, body):
        res[k] = call("/sample", body, ip=ip)
    slow = {"prompt": "Write a long story about a lighthouse.", "n": 30, "max_new_tokens": 128}
    a = threading.Thread(target=go, args=("a1", "198.51.100.10", slow)); a.start(); time.sleep(1.5)
    t("second concurrent request, same client", call("/sample", {"prompt": "Say hi.", "n": 2, "max_new_tokens": 8}, ip="198.51.100.10"), 429)
    print("   Retry-After:", call("/sample", {"prompt": "Say hi.", "n": 2, "max_new_tokens": 8}, ip="198.51.100.10")[2].get("retry-after"))
    t("concurrent request, other client", call("/sample", {"prompt": "Say hi.", "n": 2, "max_new_tokens": 8}, ip="198.51.100.11"), 200)
    a.join(); t("first request still completes", res["a1"], 200)
    t("same client after the first ended", call("/sample", {"prompt": "Say hi.", "n": 2, "max_new_tokens": 8}, ip="198.51.100.10"), 200)
    # token bucket
    r1 = call("/sample", quick, ip="198.51.100.20"); t("bucket: request 1 (51,200 of 150,000)", r1, 200)
    r2 = call("/sample", quick, ip="198.51.100.20"); t("bucket: request 2", r2, 200)
    r3 = call("/sample", quick, ip="198.51.100.20"); t("bucket: request 3 exceeds the hourly share", r3, 429); print("   ", r3[1][:150], "Retry-After:", r3[2].get("retry-after"))
    t("bucket: smaller request from the same client still fits", call("/sample", {"prompt": "Say hi.", "n": 30, "max_new_tokens": 128}, ip="198.51.100.20"), 200)
    t("bucket: other client unaffected", call("/sample", quick, ip="198.51.100.21"), 200)
    # ipv6 /64 shares a key
    call("/sample", quick, ip="2001:db8:5:5::1"); call("/sample", quick, ip="2001:db8:5:5::2")
    t("ipv6 same /64 shares the budget", call("/sample", quick, ip="2001:db8:5:5:ffff::9"), 429)
    # owner bypass
    sec = open(__import__("os").environ.get("DPP_OWN_SECRET_FILE", ".own_secret")).read().strip()
    t("owner key bypasses the budget", call("/sample", quick, ip="198.51.100.20", headers={"X-Own-Secret": sec}), 200)
    # spoofed header from a non-loopback peer cannot be tested locally; no header means the shared key
    print("ALL PASS" if ok else "SOME FAILED")
if sys.argv[1] == "swaps":
    import socket
    tiny = {"prompt": "Say hi.", "n": 2, "max_new_tokens": 8}
    def sw(model, ip): return call("/sample", dict(tiny, model=model), ip=ip)
    t("swap 1 (client S1 -> gemma-3-1b)", sw("gemma-3-1b", "198.51.100.30"), 200)
    r = sw("qwen3-0.6b", "198.51.100.31"); t("swap 2 right after, other client: global 30 s interval", r, 429); print("   ", r[1][:120], "Retry-After:", r[2].get("retry-after"))
    t("request for the loaded model needs no swap", sw("gemma-3-1b", "198.51.100.31"), 200)
    time.sleep(31); t("swap 2 after the interval (client S1)", sw("qwen3-0.6b", "198.51.100.30"), 200)
    time.sleep(31); t("swap 3 (client S1)", sw("qwen2.5-0.5b", "198.51.100.30"), 200)
    time.sleep(31); r = sw("gemma-3-1b", "198.51.100.30"); t("swap 4 (client S1) exceeds 3 swaps per hour", r, 429); print("   ", r[1][:160], "Retry-After:", r[2].get("retry-after"))
    t("client S1 may still use the loaded model", sw("qwen2.5-0.5b", "198.51.100.30"), 200)
    t("another client may swap", sw("gemma-3-1b", "198.51.100.32"), 200)
    # disconnect while a swap is pending: raw socket, close after 3 s
    time.sleep(31)
    body = json.dumps(dict(tiny, model="qwen3-0.6b")).encode()
    s = socket.create_connection((H, 8040)); s.sendall(b"POST /sample HTTP/1.1\r\nHost: x\r\nContent-Type: application/json\r\nX-Real-IP: 198.51.100.40\r\nContent-Length: " + str(len(body)).encode() + b"\r\n\r\n" + body)
    time.sleep(3); s.close(); time.sleep(4)
    st = json.loads(call("/status", method="GET", ctype=None)[1] if False else __import__("urllib.request").request.urlopen("http://127.0.0.1:8040/status").read())
    print("status after the visitor left during the swap: generating", st["generating"], "waiting", st["waiting"])
    t("same client can send a new request (no stuck in-flight slot)", sw("qwen3-0.6b", "198.51.100.40"), 200)
    print("ALL PASS" if ok else "SOME FAILED")
