"""Unit tests of api_guard (no GPU): .venv/bin/python tests/test_api_guard.py"""
import asyncio, json, math, os, sys, tempfile, time
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from fastapi import HTTPException
import api_guard as G

def status(f, *a, **k):
    try:
        r = f(*a, **k); return ("ok", r)
    except HTTPException as e:
        return (e.status_code, e.detail)

ok = True
def check(name, cond, extra=""):
    global ok
    ok &= bool(cond); print(("PASS " if cond else "FAIL ") + name, extra)

# numbers
bad_values = [float("nan"), float("inf"), -float("inf"), 1e999, "nan", "inf", "1e999", "abc", "", [], [1], [[1]], {}, None, True, {"a": 1}]
for v in bad_values:
    for integer in (False, True):
        r = status(G.number, v, "x", integer, 0.2, 2.0)
        check(f"number({v!r}, integer={integer}) rejected with 400", r[0] == 400, r)
check("number clamps high", G.number(10, "x", False, 0.2, 2.0) == 2.0)
check("number clamps low", G.number(-5, "x", False, 0.2, 2.0) == 0.2)
check("integer from float", G.number(30.9, "n", True, 2, 200) == 30)
check("integer from string", G.number("30", "n", True, 2, 200) == 30)
check("huge finite float clamps", G.number(1e300, "n", True, 2, 200) == 200)
check("huge int seed ok", isinstance(G.number(10**30, "seed", True), int))
check("1e999 literal via json is rejected by number()", status(G.number, json.loads("1e999"), "n", True, 2, 200)[0] == 400)
# text
check("text non-string", status(G.text, 5, "prompt", 10)[0] == 400)
check("text empty", status(G.text, "   ", "prompt", 10)[0] == 400)
check("text list", status(G.text, ["a"], "prompt", 10)[0] == 400)
check("text truncated", G.text("abcdef", "p", 3) == "abc")
# embedders
known = {"minilm", "qwen3-emb", "mpnet", "own"}
check("embedders dedupe", G.embedder_list(["minilm", "minilm", "own"], known, 3) == ["minilm", "own"])
check("embedders 100 duplicates ok after dedupe", G.embedder_list(["minilm"] * 100, known, 3) == ["minilm"])
check("embedders 4 distinct rejected", status(G.embedder_list, ["minilm", "qwen3-emb", "mpnet", "own"], known, 3)[0] == 400)
check("embedders unknown rejected", status(G.embedder_list, ["nope"], known, 3)[0] == 400)
check("embedders non-list rejected", status(G.embedder_list, "minilm", known, 3)[0] == 400)
check("embedders nested rejected", status(G.embedder_list, [["minilm"]], known, 3)[0] == 400)
check("embedders none ok", G.embedder_list(None, known, 3) == [])
# client keys
ck = G.client_key
check("loopback without header is shared", ck("127.0.0.1", None) == "shared")
check("loopback with bad header is shared", ck("127.0.0.1", "not an ip") == "shared")
check("loopback with header uses it", ck("127.0.0.1", "203.0.113.7") == "203.0.113.7")
check("ipv6 header keyed on /64", ck("::1", "2001:db8:1:2:aaaa:bbbb:cccc:dddd") == ck("::1", "2001:db8:1:2:1111:2222:3333:4444") == "2001:db8:1:2::/64")
check("non-loopback peer ignores header", ck("198.51.100.4", "203.0.113.7") == "198.51.100.4")
check("ipv4-mapped ipv6 peer", ck("::ffff:198.51.100.4", None) == "198.51.100.4")
# body reading
class FakeReq:
    def __init__(self, body, headers):
        self.body, self.headers = body, headers
    async def stream(self):
        for i in range(0, len(self.body), 1000):
            yield self.body[i:i + 1000]
def rj(body, headers=None, limit=64 * 1024):
    h = {"content-type": "application/json"}; h.update(headers or {})
    return status(lambda: asyncio.run(G.read_json(FakeReq(body, h), limit)))
check("valid object", rj(b'{"a": 1}') == ("ok", {"a": 1}))
check("not json", rj(b"{nope")[0] == 400)
check("NaN literal", rj(b'{"t": NaN}')[0] == 400)
check("Infinity literal", rj(b'{"t": Infinity}')[0] == 400)
check("top-level list", rj(b"[1]")[0] == 400)
check("top-level string", rj(b'"x"')[0] == 400)
check("deep nesting", rj(b"[" * 100000 + b"]" * 100000, limit=10**7)[0] == 400)
check("deep nesting over the default cap", rj(b"[" * 100000 + b"]" * 100000)[0] == 413)
check("bad utf-8", rj(b'{"a": "\xff"}')[0] == 400)
check("oversize by content-length", rj(b"{}", {"content-length": "999999"})[0] == 413)
check("oversize while reading", rj(b'{"a": "' + b"x" * 70000 + b'"}')[0] == 413)
check("wrong content type", status(lambda: asyncio.run(G.read_json(FakeReq(b"{}", {"content-type": "text/plain"}), 100)))[0] == 415)
check("missing content type", status(lambda: asyncio.run(G.read_json(FakeReq(b"{}", {}), 100)))[0] == 415)
check("charset parameter ok", status(lambda: asyncio.run(G.read_json(FakeReq(b"{}", {"content-type": "application/json; charset=utf-8"}), 100)))[0] == "ok")
check("huge integer literal", rj(b'{"n": ' + b"9" * 6000 + b"}")[0] == 400)
# limits
L = G.Limits(1000, 3, 5, 100000, 3, 100, 30)
t = 1000.0
check("first run ok", status(L.check_run, "a", 400, False, t)[0] == "ok")
L.commit_run("a", 400, False, t)
r = status(L.check_run, "a", 100, False, t + 1)
check("second concurrent run from the same client gets 429", r[0] == 429, r)
L.end_run("a", 0, t + 2)
check("after the first ends a second is allowed", status(L.check_run, "a", 400, False, t + 3)[0] == "ok")
L.commit_run("a", 400, False, t + 3); L.end_run("a", 0, t + 4)
r = status(L.check_run, "a", 400, False, t + 5)
check("token bucket exhausted gives 429 with Retry-After", r[0] == 429 and "hour" in r[1], r)
try:
    L.check_run("a", 400, False, t + 5)
except HTTPException as e:
    check("Retry-After header present", int(e.headers["Retry-After"]) > 0, e.headers)
check("another client is unaffected", status(L.check_run, "b", 400, False, t + 5)[0] == "ok")
check("bucket refills", status(L.check_run, "a", 400, False, t + 3600)[0] == "ok")
L2 = G.Limits(10**6, 2, 100, 10**7, 3, 100, 30)
L2.commit_run("c", 1, True, t); L2.end_run("c", 0, t)
r = status(L2.check_run, "d", 1, True, t + 10)
check("global swap interval gives 429", r[0] == 429, r)
check("swap allowed after the interval", status(L2.check_run, "d", 1, True, t + 31)[0] == "ok")
L2.commit_run("c", 1, True, t + 40); L2.end_run("c", 0, t + 41)
r = status(L2.check_run, "c", 1, True, t + 100)
check("per-client swap budget gives 429", r[0] == 429, r)
check("per-client swap budget does not apply without a swap", status(L2.check_run, "c", 1, False, t + 100)[0] == "ok")
L3 = G.Limits(10**6, 3, 2, 10**7, 3, 100, 30)
for i, c in enumerate("xy"):
    L3.commit_run(c, 1, False, t + i); L3.end_run(c, 0, t + i)
check("global runs ceiling gives 429", status(L3.check_run, "z", 1, False, t + 5)[0] == 429)
L4 = G.Limits(10**6, 3, 100, 100, 3, 100, 30)
check("global token ceiling gives 429", status(L4.check_run, "z", 101, False, t)[0] == 429)
L.commit_embed("e", t); L.commit_embed("e", t); L.commit_embed("e", t)
check("embed budget gives 429", status(L.check_embed, "e", t + 1)[0] == 429)
# refund
L5 = G.Limits(1000, 3, 100, 10**7, 3, 100, 30)
L5.commit_run("r", 900, False, t); L5.end_run("r", 900, t)
check("refund restores tokens", status(L5.check_run, "r", 900, False, t)[0] == "ok")
# engines
class Dead:
    async def check_health(self): raise RuntimeError("EngineDeadError stand-in")
class Fine:
    async def check_health(self): return None
class Slow:
    async def check_health(self): await asyncio.sleep(10)
check("healthy engines", asyncio.run(G.first_unhealthy([("resident", Fine()), ("slot", None)])) is None)
r = asyncio.run(G.first_unhealthy([("resident", Fine()), ("slot", Dead())]))
check("dead slot engine found", r and r[0] == "slot", r)
r = asyncio.run(G.first_unhealthy([("resident", Slow())], timeout=0.2))
check("hung engine found by timeout", r and r[0] == "resident", r)
# log rotation
d = tempfile.mkdtemp(); p = os.path.join(d, "server.log")
open(p, "w").write("a" * 100)
check("small log not rotated", G.rotate_log(p, 1000) is False)
open(p, "w").write("b" * 2000)
check("big log rotated", G.rotate_log(p, 1000) is True and os.path.getsize(p) == 0 and open(p + ".1").read() == "b" * 2000)
open(p, "w").write("c" * 2000); G.rotate_log(p, 1000)
check("rotation shifts old files", open(p + ".2").read() == "b" * 2000 and open(p + ".1").read() == "c" * 2000)
print("ALL PASS" if ok else "SOME FAILED"); sys.exit(0 if ok else 1)
