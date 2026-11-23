// One interface for everything that can produce a pool of completions: our GPU server, a Modal deploy, a Hugging Face Space,
// or a worker in this browser. The page asks each backend for its capabilities and talks to nothing else.
//
// backend = { id, label, kind, url?, caps, status(), sample(req, { signal, onProgress }), embed?(req) }
// caps    = { models: [{ key, name, state? }], embedders: [id], ownModels: [model key] | null, maxN, maxTokens, auth: 'none' | 'hf', continuePool, def?, up }
// sample resolves to { pool, emb } where pool is the pool JSON without `emb` and emb is a Float32Array of N x D (row-major).
// See docs/BACKEND.md for the wire format.

export const DEFAULT_CAPS = { models: [], embedders: ['minilm'], maxN: 200, maxTokens: 256, auth: 'none', continuePool: false, up: false };

/** Backend error with a stable `code`: bad_request (400), unauthorized (401), rate_limited (429), unavailable (503 and network). */
export class BackendError extends Error {
  constructor(code, message, status = 0) { super(message); this.code = code; this.status = status; }
}
const CODES = { 400: 'bad_request', 401: 'unauthorized', 429: 'rate_limited', 503: 'unavailable' };
const FALLBACK = { 400: 'The request was rejected.', 401: 'Sign in to use this backend.', 429: 'The usage limit of this backend is reached.', 503: 'The backend is not available right now.' };

/** "30 seconds", "about 5 minutes", "about 2 hours" from a wait in seconds. */
export function waitWords(sec) {
  const n = Math.max(1, Math.round(sec));
  if (n < 90) return `${n} second${n === 1 ? '' : 's'}`;
  if (n < 5400) return `about ${Math.round(n / 60)} minutes`;
  return `about ${Math.round(n / 3600)} hours`;
}

/**
 * Error for an HTTP status, a parsed body (FastAPI `{detail}` or `{error}`; a string or an object with `message`) and the
 * Retry-After header in seconds (or null). 429 says how long to wait; 503 says the model is restarting.
 */
export function mapError(status, body, retryAfter = null) {
  const d = body?.detail ?? body?.error;
  let msg = typeof d === 'string' ? d : d?.message ?? FALLBACK[status] ?? `The backend answered ${status}.`;
  const err = new BackendError(CODES[status] ?? (status >= 500 ? 'unavailable' : 'bad_request'), msg, status);
  if (status === 429) {
    const sec = Number(retryAfter);
    err.retryAfter = Number.isFinite(sec) && sec > 0 ? sec : null;
    // One instruction: the wait is appended only when the server's message does not already say to try again.
    err.message = err.retryAfter && !/try again/i.test(msg) ? `${msg} Try again in ${waitWords(err.retryAfter)}.` : msg;
  } else if (status === 503) err.message = 'The model is restarting. Try again in a minute.';
  return err;
}

/** Float32Array from base64 (float32, little endian). */
export function decodeF32(b64) {
  const bin = atob(b64), bytes = new Uint8Array(bin.length);
  for (let i = 0; i < bin.length; i++) bytes[i] = bin.charCodeAt(i);
  return new Float32Array(bytes.buffer);
}

/** Pool JSON from a /sample response: { pool: body without emb, emb: Float32Array }. Checks N, D and the length of emb. */
export function decodePool(body) {
  if (body && Array.isArray(body.samples) && body.kernels && typeof body.emb !== 'string') throw new BackendError('bad_request', 'This pool has kernels but no embedding. This page uses embeddings only, so it cannot show it.');
  if (!body || !Array.isArray(body.samples) || typeof body.emb !== 'string') throw new BackendError('bad_request', 'The backend returned no pool (missing samples or emb).');
  const { emb, ...pool } = body;
  const flat = decodeF32(emb);
  const N = pool.N ?? pool.samples.length, D = pool.D ?? Math.round(flat.length / N);
  if (N !== pool.samples.length || flat.length !== N * D) throw new BackendError('bad_request', `The backend returned ${flat.length} embedding values for ${pool.samples.length} samples of dimension ${D}.`);
  return { pool: { ...pool, N, D }, emb: flat };
}

/** Response of POST /embed: { embedder, D, N?, emb: base64 float32 N x D } -> { embedder, D, emb: Float32Array }. One place to adjust. */
export function decodeEmbed(body) {
  if (!body || typeof body.emb !== 'string') throw new BackendError('bad_request', 'The backend returned no embeddings.');
  const flat = decodeF32(body.emb), D = body.D ?? (body.N ? Math.round(flat.length / body.N) : 0);
  if (!D || flat.length % D) throw new BackendError('bad_request', 'The embedding array does not match its dimension.');
  return { embedder: body.embedder, D, emb: flat, approximate: !!body.approximate, note: body.note ?? null };
}

/**
 * Capabilities from a GET /status body. Two shapes: ours ({ models: [{ key, name, state, ... }], defaultModel, embedders?, maxN?, ... })
 * and the minimal Modal one ({ model, modelName?, gpu_minutes_left_this_month }), which means one model. `cfg` is the entry of
 * data/backends.json (it supplies auth and, when /status does not say, embedders).
 */
export function parseStatus(st, cfg = {}) {
  const caps = { ...DEFAULT_CAPS, auth: cfg.auth ?? 'none', up: true };
  let models;
  if (Array.isArray(st?.models)) models = st.models.map((m) => ({ key: m.key, name: m.name ?? m.key, state: m.state, params: m.params, repo: m.repo }));
  else if (st?.model) models = [{ key: st.model, name: st.modelName ?? st.model, state: 'ready' }];
  else models = [];
  caps.models = models;
  caps.multi = Array.isArray(st?.models);
  caps.def = st?.defaultModel ?? models[0]?.key ?? null;
  // embedders: an object { key: repo | null } (our server), an array of ids or { key } (older shape), else the config's list.
  const emb = st?.embedders;
  const keys = Array.isArray(emb) ? emb.map((e) => (typeof e === 'string' ? e : e.key)) : emb && typeof emb === 'object' ? Object.keys(emb) : null;
  caps.embedders = keys?.length ? keys : cfg.embedders ?? ['minilm'];
  caps.ownModels = Array.isArray(st?.ownModels) ? st.ownModels.map((m) => (typeof m === 'string' ? m : m.key)) : null;
  caps.defaultEmbedder = st?.defaultEmbedder ?? null;
  caps.maxN = st?.maxN ?? cfg.maxN ?? caps.maxN;
  caps.maxTokens = st?.maxNewTokens ?? st?.maxTokens ?? cfg.maxTokens ?? caps.maxTokens;
  caps.maxTextChars = st?.maxTextChars ?? 2000;
  caps.continuePool = !!(st?.continuePool ?? cfg.continuePool);
  caps.embed = !!(st?.embed ?? cfg.embed);
  caps.budget = st?.gpu_minutes_left_this_month;
  return caps;
}

/** Embedders of a backend usable with a model: `own` only for the models the backend lists in ownModels (when it lists any). */
export function embeddersFor(caps, model) {
  return caps.embedders.filter((e) => e !== 'own' || !caps.ownModels || caps.ownModels.includes(model));
}

/** Request body of POST /sample. `model` is sent only for a multi-model backend, `embedder` only when the backend offers several. */
export function sampleBody(req, caps) {
  const body = { prompt: req.prompt, n: req.n, t_prime: req.t_prime, max_new_tokens: req.max_new_tokens, seed: req.seed ?? Math.floor(Math.random() * 2 ** 31) };
  if (caps.multi && req.model) body.model = req.model;
  if (req.embedder && (caps.embedders.length > 1 || req.embedder !== caps.embedders[0])) body.embedder = req.embedder;
  return body;
}

/** The url from `?backend=` if it is an http(s) URL: { url, origin }, else null. */
export function parseBackendParam(search) {
  const raw = new URLSearchParams(search).get('backend');
  if (!raw) return null;
  try {
    const u = new URL(raw, 'http://invalid.local');
    if (!/^https?:$/.test(u.protocol) || u.hostname === 'invalid.local') return null;
    return { url: raw.replace(/\/+$/, ''), origin: u.origin };
  } catch { return null; }
}

/** Configs in priority order: the temporary ?backend= first, then the file's entries. */
export function backendConfigs(fileEntries, search = '') {
  const list = Array.isArray(fileEntries) ? fileEntries.filter((c) => c && c.id && c.kind) : [];
  const tmp = parseBackendParam(search);
  return tmp ? [{ id: 'custom', kind: 'http', label: tmp.origin, url: tmp.url, temporary: true, origin: tmp.origin }, ...list] : list;
}

const authHeader = (auth, getToken) => {
  if (auth !== 'hf') return {};
  const t = getToken?.();
  return t ? { Authorization: `Bearer ${t}` } : {};
};

async function readJson(r) { try { return await r.json(); } catch { return {}; } }

/** kind 'http': GET {url}/status, POST {url}/sample, optional POST {url}/embed. */
function httpBackend(cfg, { fetch: f, getToken }) {
  const b = {
    id: cfg.id, label: cfg.label ?? cfg.id, kind: 'http', url: cfg.url, caps: { ...DEFAULT_CAPS, auth: cfg.auth ?? 'none' },
    async status() {
      try {
        const r = await f(`${cfg.url}/status`, { cache: 'no-store', signal: AbortSignal.timeout?.(4000) });
        if (!r.ok) { b.caps = { ...b.caps, up: false }; return b.caps; }
        b.caps = parseStatus(await r.json(), cfg);
      } catch { b.caps = { ...b.caps, up: false }; }
      return b.caps;
    },
    async sample(req, { signal } = {}) {
      let r;
      try {
        r = await f(`${cfg.url}/sample`, { method: 'POST', signal, headers: { 'Content-Type': 'application/json', ...authHeader(cfg.auth, getToken) }, body: JSON.stringify(sampleBody(req, b.caps)) });
      } catch (e) { if (e.name === 'AbortError') throw e; throw new BackendError('unavailable', `Could not reach ${cfg.label ?? cfg.url}.`); }
      const body = await readJson(r);
      if (!r.ok) throw mapError(r.status, body, r.headers?.get?.('Retry-After'));
      return decodePool(body);
    },
  };
  if (cfg.embed !== false) {
    b.embed = async (req, { signal } = {}) => {
      if (!b.caps.embed) throw new BackendError('bad_request', `${b.label} cannot embed existing completions.`);
      const r = await f(`${cfg.url}/embed`, { method: 'POST', signal, headers: { 'Content-Type': 'application/json', ...authHeader(cfg.auth, getToken) }, body: JSON.stringify({ texts: req.texts, embedder: req.embedder, model: req.model, prompt: req.prompt, ended: req.ended }) });
      const body = await readJson(r);
      if (!r.ok) throw mapError(r.status, body, r.headers?.get?.('Retry-After'));
      return decodeEmbed(body);
    };
  }
  return b;
}

/** kind 'space': the Gradio call of a Hugging Face Space (one event, then its server-sent result). */
function spaceBackend(cfg, { fetch: f, getToken }) {
  const caps = { ...DEFAULT_CAPS, auth: 'hf', up: true, models: [{ key: cfg.model ?? 'qwen3-14b', name: cfg.name ?? 'Qwen3 14B', state: 'ready' }], def: cfg.model ?? 'qwen3-14b' };
  return {
    id: cfg.id, label: cfg.label ?? cfg.id, kind: 'space', url: cfg.url, caps,
    async status() { return caps; },
    async sample(req, { signal } = {}) {
      const h = { 'Content-Type': 'application/json', ...authHeader('hf', getToken) };
      const r = await f(`${cfg.url}/gradio_api/call/sample`, { method: 'POST', signal, headers: h, body: JSON.stringify({ data: [req.prompt, cfg.model ?? 'qwen3-14b', req.n, req.t_prime, req.max_new_tokens, req.seed ?? Math.floor(Math.random() * 1e9)] }) });
      if (!r.ok) throw new BackendError(r.status === 404 ? 'unavailable' : CODES[r.status] ?? 'unavailable', `The Space answered ${r.status}${r.status === 404 ? ' (is it running?)' : ''}.`, r.status);
      const { event_id } = await r.json();
      const text = await (await f(`${cfg.url}/gradio_api/call/sample/${event_id}`, { signal, headers: authHeader('hf', getToken) })).text();
      const events = [...text.matchAll(/event: (\w+)\ndata: (.*)/g)].map((m) => [m[1], m[2]]);
      const err = events.find(([e]) => e === 'error'), ok = events.find(([e]) => e === 'complete');
      if (!ok) {
        const msg = err && err[1] !== 'null' ? JSON.parse(err[1]) : null;
        throw new BackendError('rate_limited', msg || 'The Space returned no result. The daily GPU quota on Hugging Face may be used up.');
      }
      return decodePool(JSON.parse(ok[1])[0]);
    },
  };
}

/**
 * kind 'browser': a worker (js/worker.js). `models` are the in-browser models this device can run, [{ key, name }]; the page
 * passes the list because it depends on the device. `makeWorker` creates the Worker.
 */
function browserBackend(cfg, { makeWorker, browserModels }) {
  let worker = null;
  const caps = { ...DEFAULT_CAPS, up: true, models: browserModels?.() ?? [], maxN: 200, embedders: ['minilm'] };
  const b = {
    id: cfg.id, label: cfg.label ?? 'This browser', kind: 'browser', caps,
    async status() { b.caps = { ...caps, models: browserModels?.() ?? caps.models }; return b.caps; },
    terminate() { worker?.terminate(); worker = null; },
    sample(req, { signal, onProgress } = {}) {
      worker ??= makeWorker();
      return new Promise((resolve, reject) => {
        const stop = () => { worker.postMessage({ type: 'stop' }); reject(Object.assign(new Error('aborted'), { name: 'AbortError' })); };
        signal?.addEventListener('abort', stop, { once: true });
        worker.onmessage = ({ data }) => {
          if (data.type === 'status' || data.type === 'progress') onProgress?.(data);
          if (data.type === 'error') reject(new BackendError('unavailable', data.text));
          if (data.type === 'done') { const { pool, emb } = data; resolve({ pool: { ...pool, D: pool.D }, emb }); }
        };
        worker.postMessage({ type: 'run', prompt: req.prompt, N: req.n, Tprime: req.t_prime, maxNewTokens: req.max_new_tokens, model: req.model });
      });
    },
  };
  return b;
}

/** Backend objects for configs. deps: { fetch, getToken, makeWorker, browserModels }. */
export function createBackends(configs, deps = {}) {
  const d = { fetch: (...a) => globalThis.fetch(...a), ...deps };
  return configs.map((c) => (c.kind === 'http' ? httpBackend(c, d) : c.kind === 'space' ? spaceBackend(c, d) : c.kind === 'browser' ? browserBackend(c, d) : null)).filter(Boolean);
}

/** Load data/backends.json (an empty list if it is missing) and apply ?backend=. */
export async function loadBackendConfigs(f = globalThis.fetch, search = globalThis.location?.search ?? '') {
  let file = [];
  try { const r = await f('data/backends.json', { cache: 'no-cache' }); if (r.ok) file = await r.json(); } catch {}
  return backendConfigs(file, search);
}

/**
 * The first backend (in priority order, among `usable`) that offers `model`, or null. If it lacks `embedder`, `embedder` in the
 * result is the one it will use instead, so the page can say so.
 */
export function chooseBackend(backends, { model, embedder }, usable = () => true) {
  for (const b of backends) {
    if (!usable(b) || !b.caps.up || !b.caps.models.some((m) => m.key === model)) continue;
    const list = embeddersFor(b.caps, model), has = list.includes(embedder);
    return { backend: b, embedder: has ? embedder : list[0] ?? b.caps.embedders[0], fellBack: !has };
  }
  return null;
}
