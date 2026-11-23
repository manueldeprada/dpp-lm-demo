import test from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync, existsSync } from 'node:fs';
import { gram, columnLogScale } from '../js/dpp.js';
import { parseBasis, sharedArrows, sharedOpen, pairSelected, seedFor, easeInOut, relevantSets, displayScale, matchPreset, f16ToF32, unitRows, PRESETS, presetDiff, weightLogScale, resolveWeight, paperWeight, weightOnModeSwitch, weightEss, visibleModels, readoutNotes, effectiveK, normText, distinctCount, guardModel, comboFor, planRun, NO_SERVER } from '../js/shared.js';

const dec = (b64, N, D) => { const f = new Float32Array(Buffer.from(b64, 'base64').buffer.slice(0)); return Array.from({ length: N }, (_, i) => f.subarray(i * D, (i + 1) * D)); };
const have = existsSync(new URL('../data/basis/story_T1.5.json', import.meta.url));
const load = (p) => JSON.parse(readFileSync(new URL('../' + p, import.meta.url)));

test('pairing, seeds, easing', () => {
  assert.deepEqual(pairSelected([5, 6, 7], { 5: 3, 6: 1, 7: 2 }, [10, 11, 12], { 10: 0, 11: 9, 12: 4 }), [[10, 6], [12, 7], [11, 5]]);
  assert.deepEqual(pairSelected([1], [0, 0], [1, 2], [0, 0, 0]), []);
  assert.equal(seedFor('a', 0), seedFor('a', 0));
  assert.notEqual(seedFor('a', 0), seedFor('b', 0));
  assert.equal(easeInOut(0), 0); assert.equal(easeInOut(1), 1);
});

test('shared coordinates on the gallery pools', { skip: !have }, () => {
  const b = parseBasis(load('data/basis/story_T1.5.json'));
  for (const key of b.models ?? ['qwen2.5-0.5b', 'gemma-3-1b', 'qwen3-14b']) {
    const pool = load(`data/pools/${key}/story_T1.5.json`);
    const phi = dec(pool.emb, pool.N, pool.D);
    const a = sharedArrows(phi, b);
    assert.equal(a.coords.length, pool.N);
    const s = a.frac.reduce((x, y) => x + y, 0);
    assert.ok(s > 0 && s <= 1 + 1e-6, `frac sum ${s}`);
    const o = sharedOpen(phi, b);
    assert.ok(o.ok);
    assert.equal(o.pca.origin.length, 3);
    assert.ok(o.pca.frac.every((v) => v >= 0 && v <= 1.0001));
  }
});

test('relevantSets is deterministic and a function of the pool only', { skip: !have }, () => {
  const pool = load('data/pools/qwen3-14b/story_T1.5.json');
  const phi = dec(pool.emb, pool.N, pool.D), G = gram(phi);
  const logp = pool.samples.map((x) => x.logp), logq = pool.samples.map((x) => x.logq);
  const a = relevantSets(G, pool.N, logp, logq, 1), b = relevantSets(G, pool.N, logp, logq, 1);
  assert.deepEqual(a, b);
  assert.equal(a.length, 12 + 120);
  assert.ok(a.every((s) => s.length === 3));
  assert.notDeepEqual(a, relevantSets(G, pool.N, logp, logq, 2));
});

test('displayScale: quality is the same in both modes, importance divides by q, paper follows the mode, unit is 1', () => {
  const logp = [-3, -5, -1, -9, -4], logq = [-3.2, -4.1, -1.5, -8, -4.4];
  for (const T of [0.5, 1, 20]) {
    const qm = displayScale('quality', 'map', logp, logq, T), qs = displayScale('quality', 'sample', logp, logq, T);
    assert.deepEqual(qm, qs);
    assert.equal(Math.max(...qm), 1);
    const sm = displayScale('importance', 'map', logp, logq, T), sw = displayScale('importance', 'sample', logp, logq, T);
    assert.deepEqual(sm, sw); // importance is the same in both modes
    assert.notDeepEqual(sw, qs);
    // an unknown choice behaves as the mode's paper weight
    assert.deepEqual(displayScale('paper', 'map', logp, logq, T), qm);
    assert.deepEqual(displayScale('paper', 'sample', logp, logq, T), sw);
    assert.ok(Math.abs(sw[2] / sw[0] - Math.exp(0.5 * ((logp[2] - logp[0]) / T - (logq[2] - logq[0])))) < 1e-12);
  }
  assert.deepEqual(displayScale('unit', 'sample', logp, logq, 3), [1, 1, 1, 1, 1]);
});

test('matchPreset', () => {
  assert.equal(matchPreset({ mode: 'map', Tprime: '1.5', K: 3, T: 3 }), 'map');
  assert.equal(matchPreset({ mode: 'map', Tprime: 1.5, K: 3, T: 7.5 }), 'map');
  assert.equal(matchPreset({ mode: 'sample', Tprime: '1.0', K: 3, T: 0.82 }), 'samp');
  assert.equal(matchPreset({ mode: 'sample', Tprime: 1, K: 3, T: 0.43 }), 'samp');
  assert.equal(matchPreset({ mode: 'map', Tprime: 1.5, K: 4, T: 3 }), null);
  assert.equal(matchPreset({ mode: 'map', Tprime: 1.0, K: 3, T: 3 }), null);
  assert.equal(matchPreset({ mode: 'map', Tprime: 1.5, K: 3, T: 4 }), null);
  assert.equal(matchPreset({ mode: 'sample', Tprime: 1.5, K: 3, T: 0.82 }), null);
  assert.equal(matchPreset({ mode: 'sample', Tprime: 1, K: 3, T: 3 }), null);
});

test('f16ToF32 decodes normal, subnormal, signed, infinite and NaN halves; unitRows renormalizes', () => {
  const h = [0x0000, 0x8000, 0x3c00, 0xbc00, 0x3555, 0x0001, 0x03ff, 0x7bff, 0x7c00, 0xfc00, 0x7e00, 0x4000];
  const buf = new ArrayBuffer(2 * h.length), dv = new DataView(buf);
  h.forEach((x, i) => dv.setUint16(2 * i, x, true));
  const f = f16ToF32(buf);
  assert.equal(f[0], 0); assert.ok(Object.is(f[1], -0));
  assert.equal(f[2], 1); assert.equal(f[3], -1);
  assert.ok(Math.abs(f[4] - 0.333251953125) < 1e-12);
  assert.equal(f[5], 2 ** -24);
  assert.ok(Math.abs(f[6] - 1023 * 2 ** -24) < 1e-15);
  assert.equal(f[7], 65504);
  assert.equal(f[8], Infinity); assert.equal(f[9], -Infinity); assert.ok(Number.isNaN(f[10]));
  assert.equal(f[11], 2);
  const rows = unitRows(Float32Array.from([3, 4, 0, 0, 0, 0]), 2, 3);
  assert.ok(Math.abs(rows[0][0] - 0.6) < 1e-6 && Math.abs(rows[0][1] - 0.8) < 1e-6);
  assert.deepEqual(Array.from(rows[1]), [0, 0, 0]);
});

test('presets use the paper DPP weight with the lock on; the match needs the resolved weight', () => {
  assert.equal(PRESETS.map.weight, 'quality'); assert.equal(PRESETS.map.lock, true);
  assert.equal(PRESETS.samp.weight, 'importance'); assert.equal(PRESETS.samp.lock, true);
  assert.equal(matchPreset({ mode: 'map', Tprime: 1.5, K: 3, T: 3, weight: 'quality' }), 'map');
  assert.equal(matchPreset({ mode: 'map', Tprime: 1.5, K: 3, T: 3, weight: 'importance' }), null);
  assert.equal(matchPreset({ mode: 'sample', Tprime: 1, K: 3, T: 0.82, weight: 'importance' }), 'samp');
  assert.equal(matchPreset({ mode: 'sample', Tprime: 1, K: 3, T: 0.82, weight: 'quality' }), null);
  assert.equal(matchPreset({ mode: 'sample', Tprime: 1, K: 3, T: 0.82, weight: 'unit' }), null);
  // without a weight the match is as before
  assert.equal(matchPreset({ mode: 'map', Tprime: 1.5, K: 3, T: 3 }), 'map');
});

test('weightLogScale: the paper weight of each mode equals columnLogScale; the families do not depend on the mode', () => {
  const logp = [-3, -5, -1, -9, -4], logq = [-3.2, -4.1, -1.5, -8, -4.4];
  for (const T of [0.3, 1, 20]) {
    assert.deepEqual(weightLogScale('quality', 'map', logp, logq, T), columnLogScale('map', logp, logq, T));
    assert.deepEqual(weightLogScale('importance', 'sample', logp, logq, T), columnLogScale('sample', logp, logq, T));
    // tolerant fallback: an unknown value is the mode's paper weight
    assert.deepEqual(weightLogScale('paper', 'map', logp, logq, T), columnLogScale('map', logp, logq, T));
    assert.deepEqual(weightLogScale(undefined, 'sample', logp, logq, T), columnLogScale('sample', logp, logq, T));
    for (const mode of ['map', 'sample']) {
      assert.deepEqual(weightLogScale('quality', mode, logp, logq, T), columnLogScale('map', logp, logq, T));
      assert.deepEqual(weightLogScale('importance', mode, logp, logq, T), columnLogScale('sample', logp, logq, T));
      assert.deepEqual(weightLogScale('unit', mode, logp, logq, T), [0, 0, 0, 0, 0]);
    }
  }
  assert.equal(resolveWeight('paper', 'map'), 'quality'); assert.equal(resolveWeight(null, 'sample'), 'importance'); assert.equal(resolveWeight('unit', 'map'), 'unit'); assert.equal(paperWeight('sample'), 'importance');
});

test('presetDiff lists the changed settings in reading order', () => {
  const a = { mode: 'map', T: 3, K: 3, tp: '1.5', len: 'quality' };
  assert.deepEqual(presetDiff(a, { ...a }), []);
  assert.deepEqual(presetDiff(a, { mode: 'sample', T: 0.82, K: 3, tp: 1.0, len: 'importance' }), ['mode', 'T', 'tp', 'len']);
  const w = { ...a, weight: 'quality', lock: false };
  assert.deepEqual(presetDiff(w, { ...w, lock: true, weight: 'importance', len: 'importance', K: 4 }), ['K', 'weight', 'len', 'lock']);
  assert.deepEqual(presetDiff(a, { ...a, K: 4, T: 3 * (1 + 1e-12), tp: 1.5 }), ['K']);
  assert.deepEqual(presetDiff(a, { ...a, len: 'unit', K: 5, T: 7.5 }), ['T', 'K', 'len']);
});

test('weightOnModeSwitch keeps the weight in every case', () => {
  for (const w of ['quality', 'importance', 'unit']) for (const [f, t] of [['map', 'sample'], ['sample', 'map'], ['map', 'map']]) assert.equal(weightOnModeSwitch(w, f, t), w);
});

test('visibleModels: tier main unless expert; all when no tier is set', () => {
  const ms = [{ key: 'a', tier: 'main' }, { key: 'b' }, { key: 'c', tier: 'main' }];
  assert.deepEqual(visibleModels(ms, false).map((m) => m.key), ['a', 'c']);
  assert.equal(visibleModels(ms, true).length, 3);
  assert.equal(visibleModels([{ key: 'x' }, { key: 'y' }], false).length, 2);
});

test('readoutNotes: low ESS and few distinct completions', () => {
  assert.deepEqual(readoutNotes({ ess: 169, K: 3, N: 200, distinct: 187 }), []);
  const n = readoutNotes({ ess: 1.2, K: 3, N: 200, distinct: 187 });
  assert.equal(n.length, 1); assert.match(n[0].text, /about 1 completions/);
  const d = readoutNotes({ ess: 50, K: 3, N: 200, distinct: 12 });
  assert.equal(d.length, 1); assert.match(d[0].text, /Only 12 distinct/);
  assert.equal(readoutNotes({ ess: 1, K: 5, N: 20, distinct: 3 }).length, 2);
  // pools of 30 completions with d = 1, 2, 3 distinct texts and K = 3
  for (const d of [1, 2, 3]) {
    const kinds = readoutNotes({ ess: 30, K: 3, N: 30, distinct: d }).map((x) => x.kind);
    assert.equal(kinds.includes('distinct'), d < 3, `d=${d}`);
    assert.equal(kinds.includes('one'), d === 3);
    assert.equal(effectiveK(3, d, 30), d);
  }
  assert.equal(effectiveK(10, 200, 30), 10);
  assert.equal(effectiveK(50, 200, 30), 29);
});

test('normText and distinctCount ignore whitespace differences', () => {
  assert.equal(normText('  Why did\n the  chicken\t'), 'Why did the chicken');
  const t = (x) => ({ text: x });
  assert.equal(distinctCount([t('A joke.'), t('A  joke. '), t('\nA joke.'), t('Another joke.'), t('   ')]), 3);
  assert.equal(distinctCount([]), 0);
});

test('readoutNotes: near-duplicates note when half the pool is one answer', () => {
  const n = readoutNotes({ ess: 100, K: 3, N: 200, distinct: 80, near: { size: 150 } });
  assert.deepEqual(n.map((x) => x.kind), ['near']); assert.match(n[0].text, /150 of 200/);
  assert.deepEqual(readoutNotes({ ess: 100, K: 3, N: 200, distinct: 80, near: { size: 60 } }), []);
  // not repeated when the few-distinct note is already shown
  assert.deepEqual(readoutNotes({ ess: 100, K: 3, N: 200, distinct: 12, near: { size: 190 } }).map((x) => x.kind), ['distinct']);
});

test('guardModel maps hidden or unknown models to the default', () => {
  const allowed = ['gemma-2-9b', 'qwen3-14b'];
  assert.equal(guardModel('qwen2.5-0.5b', allowed), 'qwen3-14b');
  assert.equal(guardModel('gemma-2-9b', allowed), 'gemma-2-9b');
  assert.equal(guardModel(null, allowed), 'qwen3-14b');
  assert.equal(guardModel('x', ['a', 'b']), 'a');
  assert.equal(guardModel('x', []), 'x'); assert.equal(guardModel('x', null), 'x');
});

test('comboFor keeps allowed models and the requested model', () => {
  const ex = { story: { 'qwen2.5-0.5b': ['1.0', '1.5'], 'gemma-2-9b': ['1.0'], 'qwen3-14b': ['1.0', '1.5'] }, odd: { 'qwen2.5-0.5b': ['1.0'], 'gemma-2-9b': ['1.5'] } };
  const allowed = ['gemma-2-9b', 'qwen3-14b'];
  assert.deepEqual(comboFor(ex, 'story', 'qwen3-14b', '1.5', allowed), { key: 'qwen3-14b', tp: '1.5' });
  assert.deepEqual(comboFor(ex, 'story', 'qwen2.5-0.5b', '1.5', allowed), { key: 'qwen3-14b', tp: '1.5' }); // hidden model: default
  assert.deepEqual(comboFor(ex, 'story', 'gemma-2-9b', '1.5', allowed), { key: 'gemma-2-9b', tp: '1.0' }); // same model, its T′
  assert.deepEqual(comboFor(ex, 'story', 'qwen2.5-0.5b', '1.5', null), { key: 'qwen2.5-0.5b', tp: '1.5' }); // expert: any
  assert.deepEqual(comboFor(ex, 'odd', 'qwen3-14b', '1.5', allowed), { key: 'gemma-2-9b', tp: '1.5' }); // never the 0.5B
  assert.equal(comboFor(ex, 'nope', 'qwen3-14b', '1.5', allowed), null);
  assert.equal(comboFor({ only: { 'qwen2.5-0.5b': ['1.0'] } }, 'only', 'qwen3-14b', '1.0', allowed), null);
});

test('planRun: no fallback to the browser or to a hidden model', () => {
  const allowed = ['gemma-2-9b', 'qwen3-14b'];
  const server = (up, models = ['gemma-2-9b', 'qwen3-14b'], usable = true) => ({ id: 'server', kind: 'http', usable, up, models });
  const browser = (usable) => ({ id: 'browser', kind: 'browser', usable, up: true, models: ['qwen2.5-0.5b', 'qwen3-0.6b'] });
  // server down, browser off: refuse, whatever the selected model
  for (const model of ['qwen3-14b', 'qwen2.5-0.5b']) assert.deepEqual(planRun({ model, allowed, backends: [server(false), browser(false)] }), { refuse: NO_SERVER });
  // server up: the selected model, or the default for a hidden one
  assert.deepEqual(planRun({ model: 'gemma-2-9b', allowed, backends: [server(true), browser(false)] }), { model: 'gemma-2-9b', backend: 'server', switched: false });
  assert.deepEqual(planRun({ model: 'qwen2.5-0.5b', allowed, backends: [server(true), browser(false)] }), { model: 'qwen3-14b', backend: 'server', switched: true });
  // the server lacks the selected model: another allowed one it serves
  assert.equal(planRun({ model: 'gemma-2-9b', allowed, backends: [server(true, ['qwen3-14b']), browser(false)] }).model, 'qwen3-14b');
  // expert with the browser turned on: the browser runs a small model, the server is not usable then
  assert.deepEqual(planRun({ model: 'qwen2.5-0.5b', allowed: null, backends: [server(true, [], false), browser(true)] }), { model: 'qwen2.5-0.5b', backend: 'browser', switched: false });
  // default mode never reaches the browser model even if the browser backend is marked usable by mistake for a hidden model
  assert.ok(planRun({ model: 'qwen2.5-0.5b', allowed, backends: [server(false), browser(true)] }).refuse);
  assert.match(planRun({ model: 'x', allowed, backends: [server(false, [], false), browser(false)] }).refuse, /No backend/);
});

test('weightEss: unit is N, quality tends to N at large T, importance is smaller when q is uneven', () => {
  const logp = [-3, -5, -1, -9, -4, -7], logq = [-3.2, -4.1, -1.5, -8, -4.4, -6];
  const N = logp.length;
  for (const mode of ['map', 'sample']) {
    assert.ok(Math.abs(weightEss('unit', mode, logp, logq, 3) - N) < 1e-9);
    assert.ok(N - weightEss('quality', mode, logp, logq, 1e6) < 1e-3);
    assert.ok(weightEss('quality', mode, logp, logq, 0.2) < weightEss('quality', mode, logp, logq, 5));
  }
  // same weight, same ESS in both picking modes; Σw²: manual check for importance
  assert.equal(weightEss('importance', 'map', logp, logq, 2), weightEss('importance', 'sample', logp, logq, 2));
  const w = logp.map((lp, i) => Math.exp(lp / 2 - logq[i]));
  const s1 = w.reduce((a, b) => a + b, 0), s2 = w.reduce((a, b) => a + b * b, 0);
  assert.ok(Math.abs(weightEss('importance', 'sample', logp, logq, 2) - (s1 * s1) / s2) < 1e-9);
});
