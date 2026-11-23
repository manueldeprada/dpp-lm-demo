"""Exit and recovery paths of server.py without touching the live service: fake engines, os._exit patched. Needs no GPU."""
import asyncio, collections, os, sys, time
os.environ.setdefault("HF_HUB_OFFLINE", "1")
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import server as V
from vllm.v1.engine.exceptions import EngineDeadError
exits = []
class Exit(BaseException): pass
def fake_exit(code): exits.append(code); raise Exit()
V.os._exit = fake_exit
ok = True
def check(name, cond):
    global ok; ok &= bool(cond); print(("PASS " if cond else "FAIL ") + name)
class Dead:
    async def check_health(self): raise EngineDeadError()
    def shutdown(self): pass
class Fine:
    async def check_health(self): return None
    def shutdown(self): pass
# die() exits with a nonzero code
try: V.die("test")
except Exit: pass
check("die() calls os._exit with a nonzero code", exits == [3])
# request-level EngineDeadError on the resident engine exits
V.S["engine"] = Dead(); V.SLOT.update(engine=None, key=None, loading=None); V.S["slot_deaths"] = collections.deque()
exits.clear()
try: V.engine_error(V.S["engine"], EngineDeadError())
except Exit: pass
check("EngineDeadError in a request on the resident engine exits the process", exits == [3])
check("other exceptions are not treated as engine death", V.engine_error(V.S["engine"], ValueError("x")) is False)
# slot engine death resets the slot, three in a window exit
exits.clear(); s1 = Dead(); V.SLOT.update(engine=s1, key="qwen3-4b")
check("EngineDeadError on the slot engine is handled", V.engine_error(s1, EngineDeadError()) is True and V.SLOT["engine"] is None and not exits)
for i in range(2):
    e = Dead(); V.SLOT.update(engine=e, key="qwen3-4b")
    try: V.engine_error(e, EngineDeadError())
    except Exit: pass
check("third slot death within the window exits the process", exits == [3])
# watchdog: dead resident engine, two failed checks then exit
exits.clear(); V.S["engine"] = Dead(); V.SLOT.update(engine=None, key=None, loading=None); V.WATCHDOG_INTERVAL = 0.05
async def run_wd():
    try: await asyncio.wait_for(V.watchdog(), 2)
    except Exit: return "exit"
    except asyncio.TimeoutError: return "timeout"
check("watchdog exits after two failed checks of the resident engine", asyncio.run(run_wd()) == "exit" and exits == [3])
# watchdog: healthy engines keep running; dead slot engine is reset, not fatal
exits.clear(); V.S["engine"] = Fine(); sd = Dead(); V.SLOT.update(engine=sd, key="gemma-3-1b", loading=None); V.S["slot_deaths"].clear()
r = asyncio.run(run_wd())
check("watchdog resets a dead slot engine and keeps running", r == "timeout" and V.SLOT["engine"] is None and not exits)
# no slot check while a swap is loading
V.SLOT.update(engine=Dead(), key="x", loading="y")
check("slot engine is skipped while a swap is loading", [n for n, e in V.engines_to_check() if e is not None] == ["resident"])
print("ALL PASS" if ok else "SOME FAILED")
