// Pure helpers for the "Matrices" section: the viridis colormap and the pool's primal and dual matrices.
// B has rows sqrt(w_i) phi_i for all N pool rows; primal L = B B^T (N x N); dual C = B^T B (D x D, rank <= min(N, D)).
// The selected set S gives the submatrix L_S (K x K), and det L_S is the squared volume.

import { eigSym } from './dpp.js';
import { clusterLinkage } from './layout.js';

/** Anchors of matplotlib's viridis, dark purple to yellow, evenly spaced in [0, 1]. */
export const VIRIDIS_ANCHORS = ['#440154', '#482878', '#3e4989', '#31688e', '#26828e', '#1f9e89', '#35b779', '#6ece58', '#b5de2b', '#fde725'];
const RGB = VIRIDIS_ANCHORS.map((h) => [1, 3, 5].map((k) => parseInt(h.slice(k, k + 2), 16)));

/** Linear interpolation between anchors. t is clamped to [0, 1]; NaN maps to 0. Returns [r, g, b] in 0..255 (rounded). */
export function viridis(t) {
  const x = Math.min(1, Math.max(0, Number.isFinite(t) ? t : 0)) * (RGB.length - 1);
  const i = Math.min(RGB.length - 2, Math.floor(x)), f = x - i;
  return RGB[i].map((c, k) => Math.round(c + (RGB[i + 1][k] - c) * f));
}
export const viridisCss = (t) => `rgb(${viridis(t).join(',')})`;

/** Black or white text, whichever contrasts more with the [r, g, b] background (WCAG relative luminance). */
export function textOn([r, g, b]) {
  const lin = (c) => { const s = c / 255; return s <= 0.03928 ? s / 12.92 : ((s + 0.055) / 1.055) ** 2.4; };
  const L = 0.2126 * lin(r) + 0.7152 * lin(g) + 0.0722 * lin(b);
  return L > 0.179 ? '#111111' : '#ffffff';
}

/** { min, max } of finite values (a typed array or array). Empty or all non-finite: { min: 0, max: 0 }. */
export function range(values) {
  let min = Infinity, max = -Infinity;
  for (let i = 0; i < values.length; i++) { const v = values[i]; if (Number.isFinite(v)) { if (v < min) min = v; if (v > max) max = v; } }
  return min > max ? { min: 0, max: 0 } : { min, max };
}
/** Position of v in [0, 1] on the linear scale min..max. When max == min every value sits at 0.5 (no division by zero). */
export const unitPos = (v, { min, max }) => (max > min ? (v - min) / (max - min) : 0.5);

/** Three significant digits; exponent form when |x| < 1e-3 or >= 1e4 (weights can be 1e-40). */
export function fmtSig(x, digits = 3) {
  if (!Number.isFinite(x)) return '—';
  if (x === 0) return '0';
  const a = Math.abs(x);
  if (a < 1e-3 || a >= 1e4) return x.toExponential(digits - 1).replace('e+', 'e');
  return String(Number(x.toPrecision(digits)));
}

/** Weights in use from log w (defined up to a constant): w_i = exp(log w_i - max log w), so the largest is 1. This is the scale of S.cDisp² under the weight lock. */
export function weightsFromLog(logw) {
  const m = Math.max(...logw);
  return Array.from(logw, (s) => Math.exp(s - m));
}

/** B_S as a Float64Array of K x D, row k = sqrt(w_{sel[k]}) phi_{sel[k]}. */
export function buildB(phi, sel, w) {
  const K = sel.length, D = K ? phi[sel[0]].length : 0;
  const B = new Float64Array(K * D);
  for (let k = 0; k < K; k++) {
    const r = Math.sqrt(w[sel[k]]), p = phi[sel[k]];
    for (let d = 0; d < D; d++) B[k * D + d] = r * p[d];
  }
  return { B, K, D };
}

/** L_S = B B^T, K x K row-major (symmetric by construction). */
export function primalMatrix({ B, K, D }) {
  const L = new Float64Array(K * K);
  for (let i = 0; i < K; i++) for (let j = i; j < K; j++) {
    let s = 0;
    for (let d = 0; d < D; d++) s += B[i * D + d] * B[j * D + d];
    L[i * K + j] = L[j * K + i] = s;
  }
  return L;
}

/** C = B^T B = sum_k b_k b_k^T, D x D row-major, rank <= K (K = number of rows of B). Upper triangle accumulated, then mirrored. */
export function dualMatrix({ B, K, D }) {
  const C = new Float64Array(D * D);
  for (let k = 0; k < K; k++) {
    const b = B.subarray(k * D, (k + 1) * D);
    for (let i = 0; i < D; i++) {
      const bi = b[i];
      if (bi === 0) continue;
      const row = i * D;
      for (let j = i; j < D; j++) C[row + j] += bi * b[j];
    }
  }
  for (let i = 0; i < D; i++) for (let j = 0; j < i; j++) C[i * D + j] = C[j * D + i];
  return C;
}

/** B for every row of phi: rows sqrt(w_i) phi_i in pool order. */
export const buildAll = (phi, w) => buildB(phi, Array.from({ length: phi.length }, (_, i) => i), w);

/** The rows and columns idx of the n x n matrix A, as a |idx| x |idx| row-major array. */
export function submatrix(A, n, idx) {
  const k = idx.length, out = new Float64Array(k * k);
  for (let r = 0; r < k; r++) for (let c = 0; c < k; c++) out[r * k + c] = A[idx[r] * n + idx[c]];
  return out;
}

/** A with rows and columns permuted: out[r][c] = A[perm[r]][perm[c]]. */
export const permuted = (A, n, perm) => submatrix(A, n, perm);

/**
 * Row and column order of the pool matrix. 'pool': 0..N-1. 'weight': w descending. 'cluster': by cluster id ascending
 * (label < 0, isolated, last), then w descending inside each cluster. 'blocks': the precomputed seriation `blocks`. Ties keep pool order. Returns perm with perm[r] = pool index.
 */
export function rowOrder(mode, w, label = null, blocks = null) {
  if (mode === 'blocks' && blocks) return Array.from(blocks);
  const idx = Array.from({ length: w.length }, (_, i) => i);
  const byW = (a, b) => w[b] - w[a] || a - b;
  if (mode === 'weight') return idx.sort(byW);
  if (mode === 'cluster' && label) {
    const key = (i) => (label[i] >= 0 ? label[i] : Infinity);
    return idx.sort((a, b) => (key(a) === key(b) ? byW(a, b) : key(a) < key(b) ? -1 : 1));
  }
  return idx;
}

/** Number of eigenvalues above rel times the largest (values sorted descending). */
export const countAbove = (values, rel = 1e-9) => values.filter((v) => v > rel * values[0]).length;

export const trace = (A, n) => { let t = 0; for (let i = 0; i < n; i++) t += A[i * n + i]; return t; };

/** RGBA bytes (Uint8ClampedArray, n*n*4) of an n x n matrix on its own min..max viridis scale. */
export function heatmapPixels(A, n, sc = range(A)) {
  const px = new Uint8ClampedArray(n * n * 4);
  for (let i = 0; i < n * n; i++) {
    const [r, g, b] = viridis(unitPos(A[i], sc));
    px[4 * i] = r; px[4 * i + 1] = g; px[4 * i + 2] = b; px[4 * i + 3] = 255;
  }
  return px;
}

/** Distances 1 - cos from the N x N Gram of unit vectors (zero diagonal, clamped at 0). Float32Array. */
export function distCos(G, N) {
  const D = new Float32Array(N * N);
  for (let i = 0; i < N; i++) for (let j = 0; j < N; j++) D[i * N + j] = i === j ? 0 : Math.max(0, 1 - G[i * N + j]);
  return D;
}

/**
 * Seriation by the leaf order of average-linkage clustering on 1 - cos (G is the unit Gram, N x N). At each merge the two child
 * leaf lists are placed in whichever of the four orientations (each child as is or reversed, either side first) makes the two
 * boundary leaves the most similar pair (largest G; ties take the first orientation tried). Depends on G only; deterministic.
 * Returns perm with perm[r] = pool index at row r.
 */
export function seriate(G, N) { return seriation(G, N).order; }

/** Same as seriate, and the average-linkage merges ([{a, b, d, size}], leaves 0..N-1, merge t forms node N + t) behind the order. */
export function seriation(G, N) {
  if (N <= 2) return { order: Array.from({ length: N }, (_, i) => i), merges: N === 2 ? [{ a: 0, b: 1, d: Math.max(0, 1 - G[1]), size: 2 }] : [] };
  const merges = clusterLinkage(distCos(G, N), N, { method: 'average' });
  const leaves = new Array(N + merges.length);
  for (let i = 0; i < N; i++) leaves[i] = [i];
  merges.forEach((m, t) => {
    const A = leaves[m.a], B = leaves[m.b], Ar = A.slice().reverse(), Br = B.slice().reverse();
    let best = null, bs = -Infinity;
    for (const [L, R] of [[A, B], [A, Br], [Ar, B], [Ar, Br]]) {
      const v = G[L[L.length - 1] * N + R[0]];
      if (v > bs + 1e-12) { bs = v; best = [L, R]; }
    }
    leaves[N + t] = best[0].concat(best[1]);
    leaves[m.a] = leaves[m.b] = null;
  });
  return { order: leaves[N + merges.length - 1], merges };
}

/**
 * Segments of the dendrogram drawn with leaves in `order`: leaf r sits at x = r + 0.5, a merge at height d. Each segment has the
 * node it belongs to (the child for a vertical stem, the merged node for the horizontal bar). Returns { segs: [{x1, y1, x2, y2, node}], maxH }.
 */
export function dendrogramLayout(merges, N, order) {
  const pos = new Float64Array(N + merges.length), h = new Float64Array(N + merges.length);
  order.forEach((leaf, r) => { pos[leaf] = r + 0.5; });
  const segs = []; let maxH = 0;
  merges.forEach((m, t) => {
    const id = N + t;
    pos[id] = (pos[m.a] + pos[m.b]) / 2; h[id] = m.d; maxH = Math.max(maxH, m.d);
    segs.push({ x1: pos[m.a], y1: m.d, x2: pos[m.b], y2: m.d, node: id });
    segs.push({ x1: pos[m.a], y1: h[m.a], x2: pos[m.a], y2: m.d, node: m.a });
    segs.push({ x1: pos[m.b], y1: h[m.b], x2: pos[m.b], y2: m.d, node: m.b });
  });
  return { segs, maxH };
}

/** Group of every node: the label of a leaf; an internal node has the group of its children when they agree and are >= 0, else -1. */
export function nodeGroups(merges, N, label) {
  const g = new Int32Array(N + merges.length);
  for (let i = 0; i < N; i++) g[i] = label[i];
  merges.forEach((m, t) => { g[N + t] = g[m.a] === g[m.b] && g[m.a] >= 0 ? g[m.a] : -1; });
  return g;
}

/** Size of the largest group of the average-linkage tree cut at distance maxD (1 - cos): the merges are in non-decreasing height. */
export function largestGroup(merges, maxD) {
  let best = 1;
  for (const m of merges) { if (m.d > maxD + 1e-12) break; if (m.size > best) best = m.size; }
  return best;
}

/** The member with the largest total similarity to the other members (the medoid under 1 - cos); ties take the lowest index. */
export function medoidOf(G, N, members) {
  let best = -1, bs = -Infinity;
  for (const i of members) {
    let t = 0;
    for (const j of members) t += G[i * N + j];
    if (t > bs + 1e-12) { bs = t; best = i; }
  }
  return best;
}

/**
 * The pool's principal axes: eigenvectors of the unit dual Phi^T Phi (D x D), from the N x N unit Gram G = Phi Phi^T.
 * With G u_k = lambda_k u_k, v_k = Phi^T u_k / sqrt(lambda_k) is a unit eigenvector of Phi^T Phi. Returns the first
 * r <= rmax axes with lambda_k > 1e-9 lambda_1: { V: Float64Array[r] of length D, values: lambda_k (descending) }.
 */
export function principalAxes(phi, G, N, rmax = 64) {
  const { values, vectors } = eigSym(G, N);
  const D = phi[0].length, V = [], keep = [];
  for (let k = 0; k < Math.min(rmax, N) && values[k] > 1e-9 * values[0]; k++) {
    const u = vectors[k], v = new Float64Array(D);
    for (let i = 0; i < N; i++) { const ui = u[i], row = phi[i]; for (let d = 0; d < D; d++) v[d] += ui * row[d]; }
    const sc = 1 / Math.sqrt(values[k]);
    for (let d = 0; d < D; d++) v[d] *= sc;
    V.push(v); keep.push(values[k]);
  }
  return { V, values: keep };
}

/** C' = V_r^T C V_r = (B V_r)^T (B V_r), r x r row-major, from the K x D matrix B and r axes V (without forming C). */
export function projectedDual({ B, K, D }, V, r = V.length) {
  r = Math.min(r, V.length);
  const Y = new Float64Array(K * r);
  for (let i = 0; i < K; i++) for (let a = 0; a < r; a++) {
    let s = 0; const va = V[a];
    for (let d = 0; d < D; d++) s += B[i * D + d] * va[d];
    Y[i * r + a] = s;
  }
  const C = new Float64Array(r * r);
  for (let a = 0; a < r; a++) for (let b = a; b < r; b++) {
    let s = 0;
    for (let i = 0; i < K; i++) s += Y[i * r + a] * Y[i * r + b];
    C[a * r + b] = C[b * r + a] = s;
  }
  return C;
}

/** c_ij / sqrt(c_ii c_jj) (0 where a diagonal entry is 0). */
export function normalizeCorr(A, n) {
  const out = new Float64Array(n * n);
  for (let i = 0; i < n; i++) for (let j = 0; j < n; j++) {
    const d = Math.sqrt(A[i * n + i] * A[j * n + j]);
    out[i * n + j] = d > 0 ? A[i * n + j] / d : 0;
  }
  return out;
}
