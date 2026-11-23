// Shared display axes across models, the model rail's labels and bars, and the morph helpers.
// Pure functions (no DOM), so Node can run them. Only the display coordinates change with the
// axes: the DPP always runs on the pool's own Gram matrix.
//
// A basis file (data/basis/<id>_T<tp>.json) holds axes fitted on all models' pools of one
// example and T′ together:
//   arrows: U (3×D), the uncentered PCA directions; a point goes to (φ·u_1, φ·u_2, φ·u_3).
//   open:   m (tangent direction), gbar (mean of the tangent-plane points) and U (3×D); a point goes to
//           (g − gbar)·u_k with g = φ/(φ·m) − m, and the tangent point (g = 0) to (−gbar)·u_k.

import { columnLogScale, greedyMAPLog, kdpp, logDetSub, quality, mulberry32, ess } from './dpp.js';

/** The T slider spans 10^T_LOG_MIN to 10^T_LOG_MAX on a log scale. */
export const T_LOG_MIN = -1, T_LOG_MAX = 3;

/** IEEE half floats (little-endian, including subnormals, infinities and NaN) to a Float32Array. */
export function f16ToF32(buf) {
  const n = buf.byteLength >> 1, dv = new DataView(buf), out = new Float32Array(n);
  for (let i = 0; i < n; i++) {
    const h = dv.getUint16(2 * i, true), sg = h & 0x8000 ? -1 : 1, e = (h >> 10) & 0x1f, f = h & 0x3ff;
    out[i] = e === 0 ? sg * f * 2 ** -24 : e === 31 ? (f ? NaN : sg * Infinity) : sg * (1 + f / 1024) * 2 ** (e - 15);
  }
  return out;
}

/** Split a flat N×D array into N rows, each renormalized to unit length (a zero row stays zero). */
export function unitRows(flat, N, D) {
  return Array.from({ length: N }, (_, i) => {
    const r = Float32Array.from(flat.subarray(i * D, (i + 1) * D));
    let n = 0;
    for (let d = 0; d < D; d++) n += r[d] * r[d];
    n = Math.sqrt(n);
    if (n > 0) for (let d = 0; d < D; d++) r[d] /= n;
    return r;
  });
}

export function b64ToF32(b64) {
  const bin = atob(b64);
  const bytes = new Uint8Array(bin.length);
  for (let i = 0; i < bin.length; i++) bytes[i] = bin.charCodeAt(i);
  return new Float32Array(bytes.buffer);
}

/** Basis JSON -> { D, arrows: { U: [3] Float32Array(D), frac }, open: { m, gbar, U, frac } | null }. */
export function parseBasis(j) {
  const D = j.D;
  const rows = (b64) => { const f = b64ToF32(b64); return [0, 1, 2].map((k) => f.subarray(k * D, (k + 1) * D)); };
  return {
    id: j.id, Tprime: j.Tprime, D,
    arrows: { U: rows(j.arrows.U), frac: j.arrows.frac },
    open: j.open ? { m: b64ToF32(j.open.m), gbar: b64ToF32(j.open.gbar), U: rows(j.open.U), frac: j.open.frac } : null,
  };
}

const dot = (a, b) => { let s = 0; for (let d = 0; d < a.length; d++) s += a[d] * b[d]; return s; };

/** Arrows view: coords φ·u_k (N rows of 3) and frac_k = Σ_i (φ_i·u_k)² / N, the share of this pool's spread (Σ|φ|²/N = 1). */
export function sharedArrows(phi, basis) {
  const N = phi.length, U = basis.arrows.U;
  const coords = phi.map((p) => [dot(p, U[0]), dot(p, U[1]), dot(p, U[2])]);
  const frac = [0, 1, 2].map((k) => coords.reduce((s, c) => s + c[k] * c[k], 0) / N);
  return { coords, frac };
}

/**
 * Opened view in the shared frame. a_i = φ_i·m, g_i = φ_i/a_i − m, coords (g_i − gbar)·u_k, origin (−gbar)·u_k.
 * The bulk is the demo's rule on r_i = |g_i|: r ≤ max(3·median r, 95th percentile r).
 * frac_k = Σ_bulk ((g_i − gbar)·u_k)² / Σ_bulk |g_i − gbar|².
 * ok is false when some a_i ≤ 1e-6 (a point at or beyond the horizon of the tangent plane).
 */
export function sharedOpen(phi, basis) {
  const N = phi.length, o = basis.open;
  const a = new Float64Array(N);
  let ok = !!o;
  if (!o) return { ok: false, a };
  for (let i = 0; i < N; i++) { a[i] = dot(phi[i], o.m); if (!(a[i] > 1e-6)) ok = false; }
  if (!ok) return { ok: false, a };
  const D = o.m.length, d = new Float64Array(D);
  const coords = new Array(N), r = new Float64Array(N), dn2 = new Float64Array(N);
  for (let i = 0; i < N; i++) {
    let rr = 0, dd = 0;
    for (let j = 0; j < D; j++) { const g = phi[i][j] / a[i] - o.m[j]; rr += g * g; d[j] = g - o.gbar[j]; dd += d[j] * d[j]; }
    r[i] = Math.sqrt(rr); dn2[i] = dd;
    coords[i] = Float64Array.from([dot(d, o.U[0]), dot(d, o.U[1]), dot(d, o.U[2])]);
  }
  const sorted = Array.from(r).sort((x, y) => x - y);
  const q = (p) => sorted[Math.min(N - 1, Math.floor(p * (N - 1)))];
  const thr = Math.max(3 * q(0.5), q(0.95));
  const bulk = new Uint8Array(N);
  for (let i = 0; i < N; i++) if (r[i] <= thr) bulk[i] = 1;
  let den = 0;
  const num = [0, 0, 0];
  for (let i = 0; i < N; i++) if (bulk[i]) { den += dn2[i]; for (let k = 0; k < 3; k++) num[k] += coords[i][k] ** 2; }
  const origin = Float64Array.from([0, 1, 2].map((k) => -dot(o.gbar, o.U[k])));
  return { ok: true, a, pca: { coords, frac: num.map((v) => v / (den || 1)), bulk, origin, r } };
}

// ---------- morph ----------

export const easeInOut = (t) => (t < 0.5 ? 4 * t * t * t : 1 - ((-2 * t + 2) ** 3) / 2);

/** Indices of `items` sorted by xs[item], ascending (ties by index). */
export const sortByX = (items, xs) => items.slice().sort((p, q) => xs[p] - xs[q] || p - q);

/** Pair the old and new selected sets by rank of their axis-1 coordinate: [[newIdx, oldIdx], ...]. Empty if the sizes differ. */
export function pairSelected(oldSel, oldX, newSel, newX) {
  if (oldSel.length !== newSel.length) return [];
  const o = sortByX(oldSel, oldX), n = sortByX(newSel, newX);
  return n.map((ni, r) => [ni, o[r]]);
}

/** Deterministic seed per model (and draw number), so the rail's sampled sets do not change between redraws. */
export function seedFor(key, draw = 0) {
  let h = 2166136261;
  for (let i = 0; i < key.length; i++) { h ^= key.charCodeAt(i); h = Math.imul(h, 16777619); }
  return ((h >>> 0) + Math.imul(draw, 7919)) >>> 0;
}

/** The paper's DPP weight for a picking mode: p^(1/T) for 'map' (quality), p^(1/T)/q for 'sample' (importance). */
export const paperWeight = (mode) => (mode === 'map' ? 'quality' : 'importance');

/**
 * The weight family the DPP kernel uses: 'quality' (w = p^(1/T)), 'importance' (w = p^(1/T)/q) and 'unit' (w = 1) are themselves;
 * anything else (unknown, missing) behaves as the picking mode's paper weight.
 */
export const resolveWeight = (weight, mode) => (['quality', 'importance', 'unit'].includes(weight) ? weight : paperWeight(mode));

/**
 * The DPP weight after the picking mode is switched by hand: unchanged. Most likely and Sample use the same weight, so the picture
 * stays put and only the selected set changes. Only the paper presets set the weight.
 */
export const weightOnModeSwitch = (weight) => weight;

/** Models offered: all in expert mode, else those with tier "main" (all of them when none has a tier). */
export function visibleModels(models, expert) {
  if (expert || !models.some((m) => m.tier)) return models;
  return models.filter((m) => m.tier === 'main');
}

/** The model to use: `key` if it is allowed, else the default (Qwen3 14B) if allowed, else the first allowed one. No allowed list yet: `key`. */
export function guardModel(key, allowed, def = 'qwen3-14b') {
  if (!allowed || !allowed.length || allowed.includes(key)) return key;
  return allowed.includes(def) ? def : allowed[0];
}

/**
 * The (model, T′) of an example to show for a requested model and T′, among `allowed` models (null: any): the same model at the
 * requested T′, else the same model at its first T′, else Qwen3 14B (T′ 1.5 first), else the first allowed model with data. Null if none.
 */
export function comboFor(examples, id, key, tp, allowed = null) {
  const ex = examples[id];
  if (!ex) return null;
  const ok = (k) => (!allowed || !allowed.length || allowed.includes(k)) && ex[k]?.length;
  if (ok(key)) return { key, tp: ex[key].includes(tp) ? tp : ex[key][0] };
  if (ok('qwen3-14b')) return { key: 'qwen3-14b', tp: ex['qwen3-14b'].includes('1.5') ? '1.5' : ex['qwen3-14b'][0] };
  const k = Object.keys(ex).find(ok);
  return k ? { key: k, tp: ex[k][0] } : null;
}

export const NO_SERVER = 'The demo server is not reachable. The examples still work.';

/**
 * What a run of the typed prompt does. `backends`: [{ id, kind, usable, up, models: [key] }] in priority order, `usable` meaning the
 * backend may be used in the current mode (the browser one only when the user turned on "Run in this browser" in expert mode).
 * Returns { model, backend, switched } or { refuse: message }. It never falls back to a model outside `allowed` or to a backend that is not usable.
 */
export function planRun({ model, allowed, def = 'qwen3-14b', backends }) {
  const m = guardModel(model, allowed, def);
  const live = backends.filter((b) => b.usable && b.up);
  const hit = live.find((b) => b.models.includes(m));
  if (hit) return { model: m, backend: hit.id, switched: m !== model };
  const order = allowed?.length ? [def, ...allowed.filter((k) => k !== def)].filter((k) => allowed.includes(k)) : [];
  for (const k of order) {
    const h = live.find((b) => b.models.includes(k));
    if (h) return { model: k, backend: h.id, switched: true };
  }
  if (backends.some((b) => b.usable && !b.up)) return { refuse: NO_SERVER };
  if (!backends.some((b) => b.usable)) return { refuse: 'No backend is available here. The examples still work.' };
  return { refuse: `No available model can run your prompt right now. The examples still work.` };
}

/**
 * Readout notes: low effective sample size (Kish ESS of the weight in use below K) and few distinct completions (fewer than max(K, N/10)).
 * With `near` ({ size }: the largest group at a tight cosine threshold) it adds a note when half the pool is one answer.
 * Returns { kind: 'ess' | 'distinct' | 'near' | 'one', text }[].
 */
export function readoutNotes({ ess, K, N, distinct, near = null }) {
  const out = [];
  if (Number.isFinite(ess) && ess < K) out.push({ kind: 'ess', text: `Weights are concentrated on about ${Math.max(1, Math.round(ess))} completions, so sets barely vary.` });
  if (distinct < Math.max(K, N / 10)) out.push({ kind: 'distinct', text: `Only ${distinct} distinct completions in this pool. Try a higher sampling temperature T′.` });
  if (near && !out.some((n) => n.kind === 'distinct') && near.size >= 0.5 * N && near.size >= 3) out.push({ kind: 'near', text: `Most completions (${near.size} of ${N}) are near-duplicates of one answer. Try a higher sampling temperature T′.` });
  if (distinct === K && distinct >= 1) out.push({ kind: 'one', text: `Only one set of ${K} distinct completions exists, so Resample and T change nothing.` });
  return out;
}

/** A completion's text with runs of whitespace collapsed and the ends trimmed, so strings that differ only in spacing compare equal. */
export const normText = (t) => String(t).replace(/\s+/g, ' ').trim();

/** Number of distinct completions, whitespace-insensitive. */
export const distinctCount = (samples) => new Set(samples.map((x) => normText(x.text))).size;

/** The K the picker can use: at most the number of distinct completions and N - 1 (at least 1). */
export const effectiveK = (K, distinct, N) => Math.max(1, Math.min(K, distinct, N - 1));

/**
 * log w_i for the kernel L_ij = √w_i φ_i·φ_j √w_j, what columnLogScale returns for the resolved family:
 * quality → the 'map' column scale, importance → the 'sample' column scale, unit → zeros. With the mode's paper weight it equals
 * columnLogScale(mode, ...).
 */
export function weightLogScale(weight, mode, logp, logq, T) {
  const fam = resolveWeight(weight, mode);
  if (fam === 'unit') return logp.map(() => 0);
  return columnLogScale(fam === 'quality' ? 'map' : 'sample', logp, logq, T);
}

/**
 * Kish effective sample size (Σw)²/Σw² of the weights the DPP uses. Every weight is target / q for a pool drawn from q:
 * importance p^(1/T)/q (target p^(1/T)), quality p^(1/T) q/q (target q·p^(1/T)), unit q/q = 1 (target q, so the ESS is N).
 */
export const weightEss = (weight, mode, logp, logq, T) => ess(weightLogScale(weight, mode, logp, logq, T));

/**
 * The paper's Ambrosia operating points (K = 3, N = 200). `Ts` is the sweep of T; `T` is the value a click on the preset sets. `weight` and `lock` are applied on click: the paper's DPP weight (MAP: quality, p^(1/T); Samp: importance, p^(1/T)/q) with the display tied to it. The match requires the resolved DPP weight to be the paper's for the preset's mode.
 * PoolDPP-MAP: pools drawn at T′ = 1.5. PoolDPP-Samp: pools drawn at T′ = 1.0.
 */
export const PRESETS = {
  map: { label: 'MAP, paper', mode: 'map', Tprime: 1.5, K: 3, T: 3, Ts: [1, 3, 7.5, 10], weight: 'quality', lock: true },
  samp: { label: 'Samp, paper', mode: 'sample', Tprime: 1.0, K: 3, T: 0.82, Ts: [0.43, 0.54, 0.67, 0.82, 1], weight: 'importance', lock: true },
};

/**
 * Key of the preset whose picking mode, T′ and K match and whose grid contains T (relative tolerance 1e-6), or null.
 * `weight` (optional) is the DPP weight in use; it must equal the preset's weight.
 */
export function matchPreset({ mode, Tprime, K, T, weight }) {
  for (const [key, p] of Object.entries(PRESETS)) {
    if (weight !== undefined && weight !== p.weight) continue;
    if (p.mode === mode && Number(Tprime) === p.Tprime && K === p.K && p.Ts.some((t) => Math.abs(t - T) <= 1e-6 * t)) return key;
  }
  return null;
}

/**
 * Which settings a preset (or a Paper T chip) changes, in reading order: 'mode' (picking mode), 'T', 'K', 'tp' (the pool's T′),
 * 'weight' (DPP weight choice), 'len' (vector length shown), 'lock'. before and after: { mode, T, K, tp, weight, len, lock }.
 * T compares with relative tolerance 1e-9, tp as numbers.
 */
export function presetDiff(before, after) {
  const same = {
    mode: (a, b) => a === b,
    T: (a, b) => Math.abs(a - b) <= 1e-9 * Math.max(Math.abs(a), Math.abs(b)),
    K: (a, b) => a === b,
    tp: (a, b) => Number(a) === Number(b),
    weight: (a, b) => a === b,
    len: (a, b) => a === b,
    lock: (a, b) => a === b,
  };
  return Object.keys(same).filter((k) => !same[k](before[k], after[k]));
}

/**
 * Display lengths c_i ∈ (0, 1] of the points, a display choice independent of the picking mode.
 * 'unit': all 1. 'quality': √κ_i / max √κ_j with κ = p^(1/T), in every mode. 'importance': √(κ_i/q_i) / max, in every mode.
 * Unknown choices resolve as the picking mode's paper weight. Same convention as dpp.js kernel().
 */
export function displayScale(choice, mode, logp, logq, T) {
  if (choice === 'unit') return logp.map(() => 1);
  const ls = weightLogScale(choice, mode, logp, logq, T);
  const m = Math.max(...ls);
  return ls.map((s) => Math.exp(0.5 * (s - m)));
}

/**
 * The sets the DPP tends to pick in this pool, a function of the pool only: greedy MAP sets of size K at 12 log-spaced T over the slider's range,
 * plus 40 K-DPP samples (sample mode) at each of three temperatures (slider minimum, geometric middle, maximum), one seeded stream.
 * Sets shorter than K are skipped; nothing is deduplicated.
 */
export function relevantSets(G, N, logp, logq, seed = 1, K = 3) {
  const sets = [];
  if (N <= K) return sets;
  const Tof = (e) => 10 ** e;
  for (let k = 0; k < 12; k++) {
    const T = Tof(T_LOG_MIN + ((T_LOG_MAX - T_LOG_MIN) * k) / 11);
    const set = greedyMAPLog(G, N, columnLogScale('map', logp, logq, T), K).set;
    if (set.length === K) sets.push(set.slice());
  }
  const rng = mulberry32(seed);
  for (const e of [T_LOG_MIN, (T_LOG_MIN + T_LOG_MAX) / 2, T_LOG_MAX]) {
    const d = kdpp(G, N, columnLogScale('sample', logp, logq, Tof(e)), K);
    for (let n = 0; n < 40; n++) { const set = d.sample(rng); if (set && set.length === K) sets.push(set.slice()); }
  }
  return sets;
}
