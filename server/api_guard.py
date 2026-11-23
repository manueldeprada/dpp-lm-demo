"""Input validation, per-client limits and engine watchdog helpers for the demo server (no GPU imports, unit-testable).

Used by server.py. Everything here raises HTTPException with a short message a visitor can read; nothing raises a bare
exception for bad input.
"""
import asyncio
import collections
import hashlib
import hmac
import ipaddress
import json
import math
import os
import shutil
import time

from fastapi import HTTPException

LOOPBACK = ("127.0.0.1", "::1")


def bad(msg, status=400, headers=None):
    return HTTPException(status, msg, headers=headers)


# ---------------------------------------------------------------- body

def _no_constant(name):
    raise ValueError("non-finite number")


async def read_json(req, limit):
    """Request body as a dict: Content-Type application/json, at most `limit` bytes (checked on Content-Length and while
    reading), valid JSON without NaN/Infinity literals, top level an object."""
    ct = req.headers.get("content-type", "").split(";")[0].strip().lower()
    if ct != "application/json":
        raise bad("Send the request as Content-Type: application/json.", 415)
    cl = req.headers.get("content-length")
    if cl is not None:
        try:
            if int(cl) > limit:
                raise bad(f"The request body is limited to {limit} bytes.", 413)
        except ValueError:
            raise bad("Bad Content-Length header.")
    buf = bytearray()
    async for chunk in req.stream():
        buf.extend(chunk)
        if len(buf) > limit:
            raise bad(f"The request body is limited to {limit} bytes.", 413)
    try:
        obj = json.loads(bytes(buf), parse_constant=_no_constant)
    except (ValueError, UnicodeDecodeError, RecursionError):
        raise bad("The request body is not valid JSON.")
    if not isinstance(obj, dict):
        raise bad("The request body must be a JSON object.")
    return obj


# ---------------------------------------------------------------- values

def number(v, name, integer=False, lo=None, hi=None, clamp=True):
    """A finite number (int, float, or numeric string) as float or int. Out of range: clamped (clamp=True) or 400."""
    if isinstance(v, bool) or v is None or isinstance(v, (list, dict)):
        raise bad(f"{name} must be a number.")
    try:
        if isinstance(v, int) and integer:
            f = v
        else:
            f = float(v)
            if not math.isfinite(f):
                raise bad(f"{name} must be a finite number.")
            if integer:
                f = int(f)
    except HTTPException:
        raise
    except (TypeError, ValueError, OverflowError):
        raise bad(f"{name} must be a number.")
    if lo is not None and f < lo:
        if clamp:
            f = lo
        else:
            raise bad(f"{name} must be at least {lo}.")
    if hi is not None and f > hi:
        if clamp:
            f = hi
        else:
            raise bad(f"{name} must be at most {hi}.")
    return f


def text(v, name, maxlen, required=True, strip=True):
    if v is None:
        v = ""
    if not isinstance(v, str):
        raise bad(f"{name} must be a string.")
    if strip:
        v = v.strip()
    if required and not v:
        raise bad(f"{name} is empty.")
    return v[:maxlen] if maxlen else v


def embedder_list(v, known, max_len):
    """`embedders`: a list of at most `max_len` distinct known keys (duplicates dropped, order kept), or 400."""
    if v is None or v == []:
        return []
    if not isinstance(v, list) or not all(isinstance(k, str) for k in v):
        raise bad("embedders must be a list of embedder names.")
    out = list(dict.fromkeys(v))
    if len(out) > max_len:
        raise bad(f"embedders can list at most {max_len} different embedders.")
    if any(k not in known for k in out):
        raise bad(f"Unknown embedder; use one of {sorted(known)}.")
    return out


# ---------------------------------------------------------------- clients

def client_key(peer, real_ip):
    """Limit key of a request. Behind Caddy (peer is loopback) the visitor address comes from X-Real-IP; without a valid one all
    such traffic shares the key 'shared' (limits then act as global ones). IPv6 is keyed on its /64."""
    if peer in LOOPBACK:
        if not real_ip:
            return "shared"
        try:
            ip = ipaddress.ip_address(real_ip.strip())
        except ValueError:
            return "shared"
    else:
        try:
            ip = ipaddress.ip_address(peer)
        except (ValueError, TypeError):
            return "peer:" + str(peer)
    if isinstance(ip, ipaddress.IPv6Address):
        if ip.ipv4_mapped:
            return str(ip.ipv4_mapped)
        return str(ipaddress.IPv6Network((int(ip) >> 64 << 64, 64)))
    return str(ip)


def short(key):
    """Short hash of a client key for logs."""
    return hashlib.sha256(key.encode()).hexdigest()[:8]


def is_owner(headers, secret_file):
    try:
        secret = open(secret_file).read().strip()
    except OSError:
        return False
    return bool(secret) and hmac.compare_digest(headers.get("x-own-secret", ""), secret)


class Bucket:
    """Token bucket: `capacity` units refilled evenly over `period` seconds."""

    def __init__(self, capacity, period):
        self.cap, self.rate = float(capacity), float(capacity) / period
        self.level, self.t = float(capacity), None

    def _fill(self, now):
        if self.t is None:
            self.t = now
        dt = max(0.0, now - self.t)
        self.level = min(self.cap, self.level + dt * self.rate)
        self.t = max(self.t, now)

    def wait_for(self, cost, now):
        """Seconds until `cost` units are available (0 when they are now)."""
        self._fill(now)
        return 0.0 if self.level >= cost else (cost - self.level) / self.rate

    def take(self, cost, now):
        self._fill(now)
        self.level -= cost

    def give(self, cost, now):
        self._fill(now)
        self.level = min(self.cap, self.level + cost)


class Limits:
    """Per-client and global limits. `check` raises 429 with Retry-After; `commit` records a request."""

    def __init__(self, tokens_per_hour, swaps_per_hour, runs_per_hour_global, tokens_per_hour_global, embeds_per_hour_client,
                 embeds_per_hour_global, swap_min_interval, max_clients=20000):
        self.tph, self.sph = tokens_per_hour, swaps_per_hour
        self.rph_g, self.tph_g = runs_per_hour_global, tokens_per_hour_global
        self.eph_c, self.eph_g = embeds_per_hour_client, embeds_per_hour_global
        self.swap_gap = swap_min_interval
        self.buckets = {}
        self.swaps = collections.defaultdict(collections.deque)
        self.embeds = collections.defaultdict(collections.deque)
        self.runs = collections.deque()
        self.gbucket = Bucket(tokens_per_hour_global, 3600)
        self.eruns = collections.deque()
        self.last_swap = 0.0
        self.inflight = set()
        self.max_clients = max_clients

    @staticmethod
    def _trim(dq, now, window=3600):
        while dq and now - dq[0] > window:
            dq.popleft()

    def _bucket(self, client):
        b = self.buckets.get(client)
        if b is None:
            if len(self.buckets) >= self.max_clients:
                self.buckets.pop(next(iter(self.buckets)))
                self.swaps.pop(next(iter(self.swaps), None), None)
            b = self.buckets[client] = Bucket(self.tph, 3600)
        return b

    @staticmethod
    def _429(msg, wait):
        w = max(1, int(math.ceil(wait)))
        m = max(1, round(w / 60))
        return HTTPException(429, msg.format(w=w, mm="a minute" if m == 1 else f"{m} minutes"), headers={"Retry-After": str(w)})

    def check_run(self, client, cost, needs_swap, now=None):
        now = now or time.time()
        if client in self.inflight:
            raise self._429("You already have a request running. Wait for it to finish, then try again.", 5)
        self._trim(self.runs, now)
        if len(self.runs) >= self.rph_g:
            raise self._429("The server is busy this hour. Please try again in about {mm}.", 300)
        w = self.gbucket.wait_for(cost, now)
        if w > 0:
            raise self._429("The server has produced a lot of text this hour. Please try again in about {mm}.", w)
        w = self._bucket(client).wait_for(cost, now)
        if w > 0:
            raise self._429("You have used your share of generation for this hour. Please try again in about {mm}, or use fewer or shorter completions.", w)
        if needs_swap:
            gap = self.last_swap + self.swap_gap - now
            if gap > 0:
                raise self._429("The server just switched models for someone else. Please try again in {w} seconds.", gap)
            dq = self.swaps[client]
            self._trim(dq, now)
            if len(dq) >= self.sph:
                raise self._429("You have switched models several times this hour. Please try again in about {mm}, or pick the model that is already loaded.", dq[0] + 3600 - now)

    def commit_run(self, client, cost, needs_swap, now=None):
        now = now or time.time()
        self.inflight.add(client)
        self.runs.append(now)
        self._bucket(client).take(cost, now)
        self.gbucket.take(cost, now)
        if needs_swap:
            self.last_swap = now
            self.swaps[client].append(now)

    def end_run(self, client, refund=0, now=None):
        now = now or time.time()
        self.inflight.discard(client)
        if refund:
            self._bucket(client).give(refund, now)
            self.gbucket.give(refund, now)

    def check_embed(self, client, now=None):
        now = now or time.time()
        self._trim(self.eruns, now)
        if len(self.eruns) >= self.eph_g:
            raise self._429("The server is busy this hour. Please try again later.", 300)
        dq = self.embeds[client]
        self._trim(dq, now)
        if len(dq) >= self.eph_c:
            raise self._429("Too many embedding requests from you this hour. Please try again in about {mm}.", dq[0] + 3600 - now)

    def commit_embed(self, client, now=None):
        now = now or time.time()
        self.eruns.append(now)
        self.embeds[client].append(now)

    def snapshot(self, now=None):
        now = now or time.time()
        self._trim(self.runs, now)
        return {"runsLastHour": len(self.runs), "clients": len(self.buckets), "globalTokensLeft": int(self.gbucket.level)}


# ---------------------------------------------------------------- engines and logs

async def first_unhealthy(engines, timeout=8):
    """First (name, error) among [(name, engine)] whose check_health fails, else None. Engines may be None."""
    for name, e in engines:
        if e is None:
            continue
        try:
            await asyncio.wait_for(e.check_health(), timeout)
        except asyncio.TimeoutError:
            return name, "check_health timed out"
        except Exception as ex:  # EngineDeadError or any failure of the engine client
            return name, f"{type(ex).__name__}: {ex}"[:200]
    return None


def rotate_log(path, max_bytes, keep=4):
    """Copy-truncate rotation (the file is appended to by systemd, so truncating in place is safe): path -> path.1 -> ... -> path.<keep>."""
    try:
        if os.path.getsize(path) <= max_bytes:
            return False
    except OSError:
        return False
    for i in range(keep, 0, -1):
        src = path if i == 1 else f"{path}.{i - 1}"
        if os.path.exists(src):
            if i == keep and os.path.exists(f"{path}.{i}"):
                os.remove(f"{path}.{i}")
            shutil.copyfile(src, f"{path}.{i}") if src == path else os.replace(src, f"{path}.{i}")
    with open(path, "r+b") as f:
        f.truncate(0)
    return True
