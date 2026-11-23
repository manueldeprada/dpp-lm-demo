import test from 'node:test';
import assert from 'node:assert/strict';
import { gram, symEig, mulberry32 } from '../js/dpp.js';
import { viridis, viridisCss, textOn, range, unitPos, fmtSig, weightsFromLog, buildB, buildAll, largestGroup, seriate, seriation, dendrogramLayout, nodeGroups, medoidOf, principalAxes, projectedDual, normalizeCorr, submatrix, permuted, rowOrder, countAbove, primalMatrix, dualMatrix, trace, heatmapPixels } from '../js/matrices.js';

const close = (a, b, tol = 1e-9) => assert.ok(Math.abs(a - b) <= tol * Math.max(1, Math.abs(a), Math.abs(b)), `${a} vs ${b}`);
function pool(N, D, seed = 3) {
  const r = mulberry32(seed);
  return Array.from({ length: N }, () => {
    const v = Float64Array.from({ length: D }, () => r() - 0.5);
    const n = Math.hypot(...v);
    return v.map((x) => x / n);
  });
}

test('viridis endpoints, midpoint, clamping', () => {
  assert.deepEqual(viridis(0), [0x44, 0x01, 0x54]);
  assert.deepEqual(viridis(1), [0xfd, 0xe7, 0x25]);
  assert.deepEqual(viridis(-3), viridis(0));
  assert.deepEqual(viridis(9), viridis(1));
  // t = 1/2 lies between anchors 4 (#26828e) and 5 (#1f9e89), halfway.
  assert.deepEqual(viridis(0.5), [Math.round((0x26 + 0x1f) / 2), Math.round((0x82 + 0x9e) / 2), Math.round((0x8e + 0x89) / 2)]);
  assert.equal(viridisCss(0), 'rgb(68,1,84)');
  assert.equal(textOn(viridis(0)), '#ffffff');
  assert.equal(textOn(viridis(1)), '#111111');
});

test('degenerate scale: min == max does not divide by zero', () => {
  const sc = range([2, 2, 2]);
  assert.deepEqual(sc, { min: 2, max: 2 });
  assert.equal(unitPos(2, sc), 0.5);
  assert.equal(unitPos(5, { min: 0, max: 10 }), 0.5);
  const px = heatmapPixels(new Float64Array([7, 7, 7, 7]), 2);
  assert.ok(px.every((v) => Number.isFinite(v)));
  assert.deepEqual(Array.from(px.slice(0, 3)), viridis(0.5));
  assert.deepEqual(range([]), { min: 0, max: 0 });
});

test('fmtSig', () => {
  assert.equal(fmtSig(0.123456), '0.123');
  assert.equal(fmtSig(-1.5), '-1.5');
  assert.equal(fmtSig(0), '0');
  assert.equal(fmtSig(1.234e-40), '1.23e-40');
  assert.equal(fmtSig(123456), '1.23e5');
  assert.equal(fmtSig(NaN), '—');
});

test('weightsFromLog is max-normalized and survives tiny log weights', () => {
  const w = weightsFromLog([-100, -92, -101]);
  close(w[1], 1);
  assert.ok(w[0] > 0 && w[0] < 1e-3);
});

test('L_S is symmetric with diagonal w_i for unit phi', () => {
  const phi = pool(8, 16), w = [0.2, 1, 0.5, 0.05, 0.9, 0.3, 0.7, 0.1], sel = [4, 1, 6];
  const m = buildB(phi, sel, w), L = primalMatrix(m);
  for (let i = 0; i < 3; i++) {
    close(L[i * 3 + i], w[sel[i]]);
    for (let j = 0; j < 3; j++) close(L[i * 3 + j], L[j * 3 + i], 1e-14);
  }
  close(L[1], Math.sqrt(w[4] * w[1]) * phi[4].reduce((a, x, d) => a + x * phi[1][d], 0));
});

test('trace(L_S) == trace(C_S), shared eigenvalues, det L_S == product of eigenvalues', () => {
  const D = 24, phi = pool(10, D), w = phi.map((_, i) => 0.1 + 0.09 * i), sel = [9, 2, 5, 0];
  const m = buildB(phi, sel, w), K = 4;
  const L = primalMatrix(m), C = dualMatrix(m);
  close(trace(L, K), trace(C, D));
  const eL = symEig(L, K).values, eC = symEig(C, D).values;
  for (let k = 0; k < K; k++) close(eL[k], eC[k], 1e-8);
  for (let k = K; k < D; k++) assert.ok(Math.abs(eC[k]) < 1e-10);
  close(eL.reduce((a, x) => a * x, 1), detByElimination(L, K), 1e-8);
  // sum of eigenvalues equals sum of w_i since phi is unit
  close(eL.reduce((a, x) => a + x, 0), sel.reduce((a, i) => a + w[i], 0));
});

test('C_S is symmetric and equals sum w_i phi_i phi_i^T', () => {
  const D = 6, phi = pool(5, D), w = [1, 0.5, 0.25, 0.125, 2], sel = [3, 0, 4];
  const C = dualMatrix(buildB(phi, sel, w));
  for (let a = 0; a < D; a++) for (let b = 0; b < D; b++) {
    close(C[a * D + b], sel.reduce((s, i) => s + w[i] * phi[i][a] * phi[i][b], 0), 1e-12);
    close(C[a * D + b], C[b * D + a], 1e-14);
  }
});

test('full-pool L: L_S is the submatrix, N x N and D x D share eigenvalues', () => {
  const N = 9, D = 14, phi = pool(N, D), w = phi.map((_, i) => 0.2 + 0.1 * ((i * 5) % 7)), sel = [7, 2, 4];
  const all = buildAll(phi, w), L = primalMatrix(all), C = dualMatrix(all);
  assert.equal(all.K, N);
  close(trace(L, N), trace(C, D));
  const eL = symEig(L, N).values, eC = symEig(C, D).values;
  for (let k = 0; k < N; k++) close(eL[k], eC[k], 1e-8);
  assert.equal(countAbove(eC), N);
  const Ls = submatrix(L, N, sel), direct = primalMatrix(buildB(phi, sel, w));
  for (let k = 0; k < 9; k++) close(Ls[k], direct[k], 1e-12);
  for (let a = 0; a < D; a++) for (let b = 0; b < D; b++) close(C[a * D + b], C[b * D + a], 0);
});

test('rowOrder: pool, weight, cluster (weight inside, isolated last) and permuted', () => {
  const w = [0.1, 0.9, 0.5, 0.7, 0.3], label = [1, 0, -1, 0, 1];
  assert.deepEqual(rowOrder('pool', w), [0, 1, 2, 3, 4]);
  assert.deepEqual(rowOrder('weight', w), [1, 3, 2, 4, 0]);
  assert.deepEqual(rowOrder('cluster', w, label), [1, 3, 4, 0, 2]);
  assert.deepEqual(rowOrder('cluster', w, null), [0, 1, 2, 3, 4]);
  const A = Float64Array.from({ length: 25 }, (_, i) => i), P = permuted(A, 5, [4, 0, 1, 2, 3]);
  assert.equal(P[0], 24); assert.equal(P[1], 20); assert.equal(P[5], 4);
});
test('countAbove', () => assert.equal(countAbove([5, 1, 1e-12, 0]), 2));

test('seriate: planted blocks come back contiguous, a permutation, deterministic', () => {
  // 3 blocks of 6 unit vectors around well separated directions in R^12, indices shuffled.
  const r = mulberry32(11), D = 12, per = 6, nb = 3, N = per * nb;
  const items = [];
  for (let b = 0; b < nb; b++) for (let k = 0; k < per; k++) {
    const v = Float64Array.from({ length: D }, (_, d) => (d === 4 * b ? 1 : 0) + 0.15 * (r() - 0.5));
    const n = Math.hypot(...v); items.push({ b, v: v.map((x) => x / n) });
  }
  const sh = mulberry32(5); const perm = items.map((_, i) => i).sort(() => sh() - 0.5);
  const phi = perm.map((i) => items[i].v), block = perm.map((i) => items[i].b);
  const G = gram(phi), o = seriate(G, N);
  assert.deepEqual([...o].sort((a, b) => a - b), Array.from({ length: N }, (_, i) => i));
  const seq = o.map((i) => block[i]);
  let changes = 0; for (let i = 1; i < N; i++) if (seq[i] !== seq[i - 1]) changes++;
  assert.equal(changes, nb - 1, `blocks not contiguous: ${seq}`);
  assert.deepEqual(seriate(G, N), o);
  assert.deepEqual(seriate(new Float64Array(1).fill(1), 1), [0]);
});

test('principalAxes: V orthonormal; C\' diagonal for unit weights; trace(C\') <= trace(C), equal at full rank', () => {
  const N = 7, D = 20, phi = pool(N, D, 8), G = gram(phi);
  const { V, values } = principalAxes(phi, G, N, 64);
  assert.equal(V.length, N);
  for (let a = 0; a < V.length; a++) for (let b = 0; b < V.length; b++) close(V[a].reduce((s, x, d) => s + x * V[b][d], 0), a === b ? 1 : 0, 1e-8);
  const ones = phi.map(() => 1), all = buildAll(phi, ones), C = dualMatrix(all);
  const Cp = projectedDual(all, V);
  for (let a = 0; a < V.length; a++) for (let b = 0; b < V.length; b++) if (a !== b) assert.ok(Math.abs(Cp[a * V.length + b]) < 1e-8);
  for (let a = 0; a < V.length; a++) close(Cp[a * V.length + a], values[a], 1e-8);
  const w = phi.map((_, i) => 0.1 + 0.2 * i), wb = buildAll(phi, w), Cw = dualMatrix(wb);
  close(trace(projectedDual(wb, V), V.length), trace(Cw, D), 1e-8);
  assert.ok(trace(projectedDual(wb, V, 3), 3) <= trace(Cw, D) + 1e-12);
  const Cn = normalizeCorr(projectedDual(wb, V, 4), 4);
  for (let a = 0; a < 4; a++) close(Cn[a * 4 + a], 1);
});

test('dendrogramLayout, nodeGroups and medoidOf on a planted block pool', () => {
  const r = mulberry32(2), D = 10, per = 4, nb = 2, N = per * nb;
  const phi = [];
  for (let b = 0; b < nb; b++) for (let k = 0; k < per; k++) { const v = Float64Array.from({ length: D }, (_, d) => (d === 5 * b ? 1 : 0) + 0.1 * (r() - 0.5)); const n = Math.hypot(...v); phi.push(v.map((x) => x / n)); }
  const G = gram(phi), { order, merges } = seriation(G, N);
  assert.deepEqual(order, seriate(G, N));
  assert.equal(merges.length, N - 1);
  const { segs, maxH } = dendrogramLayout(merges, N, order);
  assert.equal(segs.length, 3 * (N - 1));
  assert.ok(segs.every((s) => s.x1 >= 0.5 && s.x2 <= N - 0.5 && s.y1 <= s.y2 + 1e-12));
  assert.ok(maxH > 0.5);
  const label = phi.map((_, i) => Math.floor(i / per));
  const g = nodeGroups(merges, N, label);
  assert.equal(g[N + N - 2], -1); // the root joins both groups
  assert.equal(new Set([...g].filter((x) => x >= 0)).size, nb);
  const m = medoidOf(G, N, [0, 1, 2, 3]);
  assert.ok([0, 1, 2, 3].includes(m));
  assert.equal(medoidOf(G, N, [6]), 6);
});

test('largestGroup: largest group of the tree at a distance cut', () => {
  const merges = [{ a: 0, b: 1, d: 0.01, size: 2 }, { a: 5, b: 2, d: 0.03, size: 3 }, { a: 6, b: 3, d: 0.2, size: 4 }, { a: 7, b: 4, d: 0.9, size: 5 }];
  assert.equal(largestGroup(merges, 0.05), 3); assert.equal(largestGroup(merges, 0.005), 1); assert.equal(largestGroup(merges, 1), 5); assert.equal(largestGroup([], 0.05), 1);
});

function detByElimination(A, n) {
  const a = Float64Array.from(A); let det = 1;
  for (let c = 0; c < n; c++) {
    let p = c; for (let r = c + 1; r < n; r++) if (Math.abs(a[r * n + c]) > Math.abs(a[p * n + c])) p = r;
    if (a[p * n + c] === 0) return 0;
    if (p !== c) { for (let k = 0; k < n; k++) [a[c * n + k], a[p * n + k]] = [a[p * n + k], a[c * n + k]]; det = -det; }
    det *= a[c * n + c];
    for (let r = c + 1; r < n; r++) { const f = a[r * n + c] / a[c * n + c]; for (let k = c; k < n; k++) a[r * n + k] -= f * a[c * n + k]; }
  }
  return det;
}
