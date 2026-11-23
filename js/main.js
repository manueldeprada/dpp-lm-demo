import * as d3 from 'd3';
import {
  gram, eigSym, symEig, kernel, greedyMAPLog, kdpp, logDetSub,
  logWeights, ess, quality, diversity, mulberry32,
} from './dpp.js';
import { spanView, openCone, displayedFactor, clusterPool, clusterKeywords, fitView, alignTo, pcaRotate, hull2D, polyArea, tipSheet, clusterLinkage, cutClusters, distFromGram, distFromCoords, clusterDiameters } from './layout.js';
import { Plot3D } from './plot3d.js';
import { initExplainers } from './explain.js';
import { viridis, viridisCss, textOn, range, unitPos, fmtSig, weightsFromLog, buildAll, primalMatrix, dualMatrix, heatmapPixels, submatrix, permuted, rowOrder, countAbove, trace, seriation, dendrogramLayout, nodeGroups, medoidOf, largestGroup, principalAxes, projectedDual, normalizeCorr } from './matrices.js';
import { MODELS, modelByKey } from './lm.js';
import { createBackends, loadBackendConfigs, chooseBackend, embeddersFor } from './backends.js';
import { authEnabled, getAuth, signIn, signOut, finishSignIn } from './hfauth.js';
import {
  parseBasis, sharedArrows, relevantSets, displayScale, weightLogScale, weightEss, resolveWeight, paperWeight, weightOnModeSwitch, visibleModels, readoutNotes, effectiveK, distinctCount, guardModel, comboFor as pickCombo, planRun, PRESETS, matchPreset, presetDiff, f16ToF32, unitRows, T_LOG_MIN, T_LOG_MAX, easeInOut, pairSelected, seedFor,
} from './shared.js';

const $ = (s) => document.querySelector(s);
const css = (v) => getComputedStyle(document.documentElement).getPropertyValue(v).trim();
const fmt = (x, d = 2) => (Number.isFinite(x) ? x.toFixed(d) : '—');
const REDUCED = matchMedia('(prefers-reduced-motion: reduce)');
// For browser tests: how many morphs ran, in which plot, how long the last one took, and how many model switches had no view to morph from.
const DBG = (window.__dppDebug = { morphs: 0, lastMorphMs: 0, lastKind: '', skipped: 0 });
DBG.state = () => S; // for browser tests with a mocked backend
DBG.backends = () => BACKENDS;

// ---------------------------------------------------------------- state
const S = {
  pool: null,        // { prompt, samples, N, D, Tprime, ... }
  phi: null,         // Float32Array[] unit embeddings as returned by the embedder
  G: null,           // Gram ΦᵀΦ
  logp: null, logq: null,
  proj: null,        // { coords: [N][3], frac: number[] }: fixed uncentered PCA of the embeddings (shared basis for examples); the reference frame of the exact view
  view: null,        // { coords, exact, frac, barFrac, s }: the coordinates drawn (spanView, then openCone), recomputed on every render
  shared: null,      // { basis } while a gallery example is drawn in its shared axes (all models' pools, one frame)
  embedder: 'minilm', // key of the embedding model behind the similarity kernel (CATALOG.embedders)
  dims: 3,
  mode: 'map',
  T: { sample: 1, map: 20 },
  K: 3,
  pinned: -1,      // point shown in the inspector
  expert: false,   // expert mode: every control and the extra views
  manual: false,   // the selected set was picked by hand in the matrix
  Kwant: null,     // the K the user asked for (S.K is limited by the number of distinct completions)
  distinct: 0,     // distinct completion strings in the pool
  pinned2: -1,     // second point of a pair pinned in the matrices (-1: none)
  drawNo: 0,       // Resample counter: the sampled set is a function of (model, drawNo), so redraws do not flicker
  sel: [], spec: null, L: null, c: null, pi: null, logP: NaN,
  hover: -1,
  source: 'example', // 'example' (precomputed pools) | 'custom' (the user's own prompt)
  exId: null, model: 'qwen3-14b', tp: '1.5',
  cust: null,        // { prompt, N, Tp, max } of the last run of the user's own prompt
  where: 'server',   // custom runs: 'server' | 'browser' | 'cloud'
  browser: false,    // "Run in this browser"
  pending: false,    // a custom run is in flight
  loadSeq: 0, runSeq: 0,
  morphFrom: null,
};

// T slider: log scale over [0.1, 1000].
const T_MIN = T_LOG_MIN, T_MAX = T_LOG_MAX;
const sliderToT = (v) => 10 ** (T_MIN + ((T_MAX - T_MIN) * v) / 1000);
const tToSlider = (T) => Math.round(((Math.log10(T) - T_MIN) / (T_MAX - T_MIN)) * 1000);

// ---------------------------------------------------------------- data
function decodeEmb(b64, N, D) {
  const bin = atob(b64);
  const bytes = new Uint8Array(bin.length);
  for (let i = 0; i < bin.length; i++) bytes[i] = bin.charCodeAt(i);
  const f = new Float32Array(bytes.buffer);
  return Array.from({ length: N }, (_, i) => f.subarray(i * D, (i + 1) * D));
}

const EXAMPLE_LABELS = { sort: 'Sort a list', story: 'Story opening', bike: 'Old bicycle', bank: 'What is a bank?', language: 'Learn a language' };

// data/models.json drives the model picker: { models: [{ key, name, params, repo, dtype }], Tprimes, examples: { id: { modelKey: [T′, ...] } } }.
// Without it the picker falls back to the in-browser models plus the 14B and the examples load from data/<id>.json.
const CATALOG = { models: [], Tprimes: ['1.0', '1.5'], examples: {}, embedders: [], loaded: false };
// Server option for the embedder (POST body field `embedder`). Off until the GPU server is confirmed to hold the Qwen3-Embedding model.
const SERVER_QWEN_EMB = true;
const DEF_EMB = 'minilm';
// The page's embedder key for what a server reports: a key, a repo id ("Qwen/Qwen3-Embedding-0.6B"), or "own"; null if unknown.
const embKeyOf = (v) => CATALOG.embedders.find((e) => e.key === v || (e.repo && e.repo === v))?.key ?? null;
const embInfo = (k) => CATALOG.embedders.find((e) => e.key === k) ?? { key: k, name: k };
async function loadCatalog() {
  try {
    const r = await fetch('data/models.json', { cache: 'no-cache' }); // revalidate: the catalog changes between deploys
    if (!r.ok) throw new Error(String(r.status));
    const j = await r.json();
    CATALOG.models = j.models.slice();
    CATALOG.Tprimes = j.Tprimes?.length ? j.Tprimes : CATALOG.Tprimes;
    CATALOG.examples = j.examples ?? {};
    CATALOG.embedders = Array.isArray(j.embedders) ? j.embedders : [];
    S.embedder = CATALOG.embedders.find((e) => e.default)?.key ?? DEF_EMB;
    CATALOG.loaded = true;
  } catch {
    const num = (m) => parseFloat(/(\d+(?:\.\d+)?)\s*B\b/i.exec(m.name)?.[1] ?? '0');
    CATALOG.models = MODELS.map((m) => ({ key: m.key, name: m.name, params: num(m), repo: m.repo }))
      .concat([{ key: 'qwen3-14b', name: 'Qwen3 14B', params: 14.8, repo: 'Qwen/Qwen3-14B-AWQ', tier: 'main' }]);
  }
  sortCatalog();
}
const sortCatalog = () => CATALOG.models.sort((a, b) => (a.params ?? 1e9) - (b.params ?? 1e9));
// One place decides which models may be selected, shown or run: all of them in expert mode, else those with tier "main".
// Every assignment to S.model goes through setModel, which maps a disallowed key to the default (Qwen3 14B).
const allowedKeys = () => visibleModels(CATALOG.models, S.expert).map((m) => m.key);
const setModel = (key) => { S.model = guardModel(key, allowedKeys()); return S.model; };
// In-browser running is an expert option the user turns on themselves; a saved flag is ignored in default mode.
const browserOn = () => S.expert && S.browser;
const info = (key) => CATALOG.models.find((m) => m.key === key) ?? { key, name: key };

// Pools of the examples, fetched once each. READY holds the decoded entries, POOLS the in-flight promises.
const POOLS = new Map(), READY = new Map(), BASES = new Map(), BASES_READY = new Map();
// Same for the other embedders: entries keyed `${embedder}|${id}|${model}|${T′}`.
const POOLS_E = new Map(), READY_E = new Map();
function makeEntry(rec, phi) {
  return { rec, phi, N: rec.samples.length, logp: rec.samples.map((s) => s.logp), logq: rec.samples.map((s) => s.logq), G: null, memo: new Map() };
}
const gramOf = (e) => (e.G ??= gram(e.phi));
function fetchPool(id, key, tp) {
  const k = `${id}|${key}|${tp}`;
  if (!POOLS.has(k)) {
    const p = fetch(`data/pools/${key}/${id}_T${tp}.json`).then(async (r) => {
      if (!r.ok) throw new Error(`${r.status}`);
      const rec = await r.json();
      if (typeof rec.emb !== 'string') throw new Error('this pool has no embedding (kernels only, which this page does not use)');
      const e = makeEntry(rec, decodeEmb(rec.emb, rec.N, rec.D));
      READY.set(k, e);
      return e;
    });
    p.catch(() => POOLS.delete(k));
    POOLS.set(k, p);
  }
  return POOLS.get(k);
}
// The pool of an example under an embedder: the same completions and probabilities, new unit embeddings from
// data/emb/<embedder>/<model>/<id>_T<tp>.bin (float16, N×D row-major, rows in sample order). A copy of the record, so that the
// per-pool caches (clusters, fixed view) are not shared with the default embedder.
function fetchEntry(emb, id, key, tp) {
  if (emb === DEF_EMB) return fetchPool(id, key, tp);
  const k = `${emb}|${id}|${key}|${tp}`;
  if (!POOLS_E.has(k)) {
    const p = Promise.all([fetchPool(id, key, tp), fetch(`data/emb/${emb}/${key}/${id}_T${tp}.bin`)]).then(async ([base, r]) => {
      if (!r.ok) throw new Error(`${r.status} for the ${embInfo(emb).name} file`);
      const flat = f16ToF32(await r.arrayBuffer()), N = base.N, D = Math.round(flat.length / N);
      if (!N || D * N !== flat.length) throw new Error(`the ${embInfo(emb).name} file has ${flat.length} values, not a multiple of ${N}`);
      const { emb: _unused, ...rest } = base.rec;
      const e = makeEntry({ ...rest, D, embedder: emb }, unitRows(flat, N, D));
      READY_E.set(k, e);
      return e;
    });
    p.catch(() => POOLS_E.delete(k));
    POOLS_E.set(k, p);
  }
  return POOLS_E.get(k);
}
const readyEntry = (emb, id, key, tp) => (emb === DEF_EMB ? READY.get(`${id}|${key}|${tp}`) : READY_E.get(`${emb}|${id}|${key}|${tp}`));
const basisKey = (emb, id, tp) => (emb === DEF_EMB ? `${id}|${tp}` : `${emb}|${id}|${tp}`);
// Shared axes of one (example, T′, embedder); null when the file is missing (the pool then gets its own axes).
function fetchBasis(id, tp, emb = DEF_EMB) {
  const k = basisKey(emb, id, tp);
  if (!BASES.has(k)) {
    BASES.set(k, fetch(emb === DEF_EMB ? `data/basis/${id}_T${tp}.json` : `data/basis/${emb}/${id}_T${tp}.json`).then(async (r) => {
      if (!r.ok) { BASES_READY.set(k, null); return null; }
      const b = parseBasis(await r.json());
      BASES_READY.set(k, b);
      return b;
    }).catch(() => { BASES_READY.set(k, null); return null; }));
  }
  return BASES.get(k);
}

// The model and T′ to show for an example: those in effect if the example has them, else Qwen3 14B at 1.5, else the first listed.
const comboFor = (id, key, tp) => pickCombo(CATALOG.examples, id, key, tp, allowedKeys());
const exampleTps = (id) => CATALOG.Tprimes.filter((tp) => Object.values(CATALOG.examples[id] ?? {}).some((l) => l.includes(tp)));

async function loadGallery() {
  const [idx] = await Promise.all([(await fetch('data/index.json')).json(), catalogReady]);
  const el = $('#examples');
  el.innerHTML = '<div class="menu-head">Example prompts</div>' + idx.map((g) => `<button type="button" role="menuitem" data-id="${g.id}"><b>${escapeHtml(g.label ?? EXAMPLE_LABELS[g.id] ?? g.id)}</b><span>${escapeHtml(g.prompt)}</span></button>`).join('');
  el.querySelectorAll('button').forEach((b) => (b.onclick = () => { $('#exmenu').open = false; showExample(b.dataset.id); }));
  await showExample(idx.find((g) => g.id === 'story')?.id ?? idx[0].id);
}

function markExample(id) {
  document.querySelectorAll('#examples button').forEach((b) => b.toggleAttribute('aria-current', b.dataset.id === id));
}

// Show an example's precomputed pool for a model and T′ (no generation). On the same example the plot keeps its camera
// and the selected set glides to its new place (morph); a different example starts a fresh plot.
async function showExample(id, { key = S.model, tp = S.tp, morph = false } = {}) {
  const seq = ++S.loadSeq;
  cancelRun();
  const combo = comboFor(id, key, tp);
  if (!combo) return showLegacyExample(id, seq);
  const sameEx = S.source === 'example' && S.exId === id && !!S.pool && !!S.shared;
  S.source = 'example'; S.exId = id; setModel(combo.key); S.tp = combo.tp; S.shownModel = S.model;
  updatePickers();
  const emb = S.embedder, bk = basisKey(emb, id, combo.tp), embNote = emb === DEF_EMB ? '' : ` with ${embInfo(emb).name} embeddings`;
  let entry, basis;
  if (readyEntry(emb, id, combo.key, combo.tp) && BASES_READY.has(bk)) {
    // Cached: everything below runs in this tick, so the caption and the status line change together.
    entry = readyEntry(emb, id, combo.key, combo.tp); basis = BASES_READY.get(bk);
  } else {
    $('#live-status').textContent = `Loading ${info(combo.key).name}${embNote}…`;
    try { [entry, basis] = await Promise.all([fetchEntry(emb, id, combo.key, combo.tp), fetchBasis(id, combo.tp, emb)]); }
    catch (e) {
      if (seq !== S.loadSeq) return;
      if (emb !== DEF_EMB) {
        // A missing embedding file: go back to the default embedder and say why.
        S.embedder = DEF_EMB;
        syncEmbedder();
        await showExample(id, { key: combo.key, tp: combo.tp, morph });
        $('#live-status').textContent = `Could not load the ${embInfo(emb).name} embeddings (${e.message}). Showing MiniLM-L6 instead.`;
        return;
      }
      $('#live-status').textContent = `Could not load the pool of ${info(combo.key).name}: ${e.message}`;
      return;
    }
    if (seq !== S.loadSeq) return;
  }
  if (!sameEx) {
    $('#prompt').value = entry.rec.prompt;
    $('#prompt').dispatchEvent(new Event('input'));
  }
  markExample(id);
  $('#progress').hidden = true;
  const snap = sameEx && morph ? plot.snapshot() : null;
  if (sameEx && morph && !snap) DBG.skipped++;
  $('#live-status').textContent = `Example: ${entry.N} completions from ${info(combo.key).name}, precomputed at T′ ${combo.tp}${embNote}. Edit the prompt to try your own.`;
  setPool(entry.rec, entry.phi, { entry, basis, keepPlot: sameEx, snap });
}

// Examples without a models.json entry: the single pool in data/<id>.json, in its own axes.
async function showLegacyExample(id, seq) {
  let rec;
  try { rec = await (await fetch(`data/${id}.json`)).json(); } catch (e) { $('#live-status').textContent = `Could not load the example: ${e.message}`; return; }
  if (seq !== S.loadSeq) return;
  S.source = 'example'; S.exId = id;
  $('#prompt').value = rec.prompt;
  $('#prompt').dispatchEvent(new Event('input'));
  markExample(id);
  $('#progress').hidden = true;
  $('#live-status').textContent = `Example: ${rec.N} completions from ${rec.modelName ?? modelByKey(MODELS.find((m) => m.repo === rec.model)?.key).name}. Edit the prompt to try your own.`;
  updatePickers();
  setPool(rec, decodeEmb(rec.emb, rec.N, rec.D), {});
}

// opt: entry (cached Gram), basis (shared axes), keepPlot (same example: no camera reset), snap (morph start).
function setPool(pool, phi, opt = {}) {
  $('#labels').innerHTML = '';
  S.pinned = -1; S.pinned2 = -1;
  S.pool = pool;
  S.phi = phi;
  S.distinct = distinctCount(pool.samples);
  S.logp = pool.samples.map((s) => s.logp);
  S.logq = pool.samples.map((s) => s.logq);
  const N = pool.samples.length;
  S.G = opt.entry ? gramOf(opt.entry) : gram(S.phi);
  S.dimCl = null;
  if (opt.basis) {
    // Shared axes: display coordinates only; the DPP uses the pool's own Gram.
    S.shared = { basis: opt.basis };
    S.proj = sharedArrows(phi, opt.basis);
  } else {
    S.shared = null;
    // Uncentered PCA of the unit embeddings via the N×N Gram matrix.
    const eig = eigSym(S.G, N);
    const { values, vectors } = eig;
    const tot = values.reduce((a, b) => a + Math.max(0, b), 0);
    S.proj = {
      frac: values.slice(0, 8).map((v) => Math.max(0, v) / tot),
      coords: Array.from({ length: N }, (_, n) => [0, 1, 2].map((k) => Math.sqrt(Math.max(0, values[k])) * vectors[k][n])),
    };
  }
  // Fixed display coordinates: a function of the pool only (not of the selection, mode, T or K), cached on the pool.
  if (!pool.fixed) {
    let value;
    try {
      const t0 = performance.now();
      const fit = fitView(S.G, N, relevantSets(S.G, N, S.logp, S.logq, 1), S.proj.coords);
      const coords = alignTo(fit.coords, S.proj.coords, 3);
      const sum = (C, k) => C.reduce((a, c) => a + c[k] * c[k], 0) / N;
      const p2 = pcaRotate(coords, 3);
      // Own pools: axes ordered by spread (same as 2D). Gallery examples keep the shared frame, so axis k means the same direction for every model.
      value = S.shared ? { coords, frac: [0, 1, 2].map((k) => sum(coords, k)), coords2: p2.coords, frac2: p2.frac, ms: performance.now() - t0 }
        : { coords: p2.coords, frac: p2.frac, coords2: p2.coords, frac2: p2.frac, ms: performance.now() - t0 };
    } catch (e) {
      console.warn('fitView failed, using the reference coordinates', e);
      value = { coords: S.proj.coords, frac: S.proj.frac.slice(0, 3), coords2: S.proj.coords, frac2: S.proj.frac.slice(0, 3), ms: 0 };
    }
    Object.defineProperty(pool, 'fixed', { value, enumerable: false });
  }
  S.fixed = pool.fixed;
  S.cl = clustersFor();
  S.manual = false;
  S.Kwant ??= S.K;
  S.K = effectiveK(S.Kwant, S.distinct, N);
  $('#K').max = String(Math.max(1, Math.min(20, N - 1, S.distinct)));
  $('#K').value = S.K;
  S.drawNo = 0;
  if (!opt.keepPlot) plot.reset();
  S.morphFrom = opt.snap ?? (opt.keepPlot && REDUCED.matches ? { kind: 'any', instant: true } : null);
  recompute({ newDraw: true });
  S.morphFrom = null;
}

// ---------------------------------------------------------------- view
// Display coordinates. With K <= dims selected vectors the axes are chosen to contain them (spanView), so the drawn
// parallelepiped is the one the DPP scores. Otherwise the fixed coordinates S.proj are drawn. "Open the cone" then
// applies one linear map to all points (openCone), which multiplies every volume by the same factor.
const spanOn = () => document.querySelector('input[name=axes]:checked')?.value === 'span';
function computeView() {
  const dims = S.dims, ref = S.proj.coords, refFrac = S.proj.frac;
  let coords, exact = false, frac, barFrac, s = 1;
  if (spanOn()) {
    const v = spanView(S.phi, S.sel, ref, dims, refFrac);
    const v3 = dims === 3 ? v : spanView(S.phi, S.sel, ref, 3, refFrac);
    ({ coords, exact, frac } = v);
    barFrac = v3.frac;
  } else {
    // Fixed axes: the same coordinates for every selection, mode, T and K. The 2D plot uses the fit's own principal axes.
    const f = S.fixed;
    coords = dims === 3 ? f.coords : f.coords2;
    frac = barFrac = dims === 3 ? f.frac : f.frac2;
  }
  if ($('#cone-toggle').checked) ({ coords, s } = openCone(coords));
  S.view = { coords, exact, frac, barFrac, s, fixed: !spanOn() };
}
const viewCoords = () => S.view.coords;
const sizeQ = () => S.expert && $('#size-toggle').checked;
const lenChoice = () => (sizeQ() ? 'unit' : document.querySelector('input[name=len]:checked')?.value ?? 'quality');
// Display lengths: independent of the picking mode under 'unit' and 'quality'.
function setDisplayScale() {
  syncLen();
  S.cDisp = displayScale(lenChoice(), S.mode, S.logp, S.logq, S.T[S.mode]);
  // Point size shows quality: ball radius ∝ √(quality weight p^(1/T)) relative to the largest, with a floor so every point stays visible.
  S.rsz = sizeQ() ? displayScale('quality', S.mode, S.logp, S.logq, S.T[S.mode]).map((c) => 0.35 + 0.65 * c) : null;
}
// DPP weight: what the kernel uses (a mode switch moves the paper's weight with it, see setMode). Vector length: what is drawn; the lock ties it to the weight.
const weightChoice = () => document.querySelector('input[name=wt]:checked')?.value ?? 'quality';
const lockOn = () => $('#len-lock').getAttribute('aria-pressed') === 'true';
const famNow = () => resolveWeight(weightChoice(), S.mode);
// Every weight is a target over the proposal q: importance p^(1/T)/q, quality p^(1/T) q/q, unit q/q.
const FAM_TEX = { quality: 'w = p^{1/T}\\,q\\,/\\,q', importance: 'w = p^{1/T}\\,/\\,q', unit: 'w = q\\,/\\,q' };
const FAM_TEXT = { quality: 'w = p<sup>1/T</sup> q / q', importance: 'w = p<sup>1/T</sup> / q', unit: 'w = q / q' };
function syncLen() {
  const fam = famNow(), locked = lockOn();
  $('#wt-now').innerHTML = window.katex ? window.katex.renderToString(FAM_TEX[fam], { throwOnError: false }) : FAM_TEXT[fam];
  document.querySelectorAll('input[name=len]').forEach((r) => {
    if (locked) r.checked = r.value === fam;
    r.disabled = locked;
    r.closest('label').classList.toggle('off', locked);
  });
  $('#len-row').classList.toggle('locked', locked);
}
function refreshDisplay() { setDisplayScale(); plot.update(); renderLabels(); renderVolHint(); renderControls(); }
// log w of a completion for the weight in use (up to the constant the kernel drops): p^(1/T)/q, p^(1/T) q/q or q/q.
const weightInfo = () => {
  const fam = famNow(), T = S.T[S.mode];
  return { name: 'log w', f: fam === 'importance' ? (s) => s.logp / T - s.logq : fam === 'quality' ? (s) => s.logp / T : () => 0 };
};
const LEN_WORD = { unit: 'unit', quality: 'quality', importance: 'importance' };
// Name of the volume shown as "true": the kernel's √det L_S when the drawn vectors are the DPP's, else what was drawn.
function volName() {
  if (lenChoice() === 'unit') return '√det G<sub>S</sub>';
  return lenChoice() === famNow() ? '√det L<sub>S</sub>' : `volume of the ${LEN_WORD[lenChoice()]}-scaled vectors`;
}
// Volume of the drawn selected vectors (quality-scaled when scaling is on), K <= dims.
function drawnVolume() {
  const f = (i) => S.cDisp[i];
  const rows = S.sel.map((i) => Array.from(S.view.coords[i], (x) => x * f(i)));
  return displayedFactor(rows, rows.map((_, k) => k));
}

// ---------------------------------------------------------------- inference
function recompute({ newDraw = true } = {}) {
  if (!S.pool) return;
  if (S.manual) S.K = S.sel.length;
  const N = S.logp.length, K = S.K, T = S.T[S.mode];
  const logScale = weightLogScale(weightChoice(), S.mode, S.logp, S.logq, T);
  const { L, c } = kernel(S.G, N, logScale);
  S.L = L;
  S.c = c; // the kernel's lengths (selection math); S.cDisp is what is drawn
  setDisplayScale();
  // Scale-free: greedy and K-DPP work on G with log-scales, so sharp weights neither stop the greedy early nor null the sampler.
  const dpp = kdpp(S.G, N, logScale, K);
  S.pi = dpp.pi;
  if (S.manual) {
    // Hand-picked set: kept as it is; only its numbers follow the weight and T.
  } else if (S.mode === 'map') {
    S.sel = greedyMAPLog(S.G, N, logScale, K).set;
  } else if (newDraw || !S.sel.length) {
    S.sel = dpp.sample(mulberry32(seedFor(S.source === 'example' ? S.model : `custom:${S.model}`, S.drawNo))) ?? [];
  }
  S.logP = S.sel.length === K ? dpp.logProb(S.sel) : NaN;
  render();
}

// ---------------------------------------------------------------- rendering: hero
function render() {
  document.querySelectorAll('input[name=dims]').forEach((r) => (r.checked = +r.value === S.dims));
  const distinct = S.distinct;
  const N = S.pool.samples.length;
  $('#stage-note').textContent = `${N} completions${distinct < N ? ` (${distinct} distinct)` : ''} · ${S.dims === 3 ? 'drag to rotate · tap a point to read it · change dimensions with the bars at right' : 'tap a point to read it · change dimensions with the bars at right'}`;
  $('#subset-meta').textContent = S.manual ? `Your selection · K = ${S.sel.length}` : `K = ${S.sel.length} · ${S.mode === 'map' ? 'greedy MAP' : 'K-DPP sample'} at T = ${fmtT(S.T[S.mode])}`;
  $('#manual-note').hidden = !S.manual;
  $('#manual-back').textContent = `Back to ${S.mode === 'map' ? 'Most likely' : 'Sample'}`;
  if (S.manual) $('#K').value = S.K;
  if (!anim.tweenT) { $('#T-out').textContent = fmtT(S.T[S.mode]); $('#T-slider').value = tToSlider(S.T[S.mode]); }
  renderControls();
  computeView();
  plot.update();
  renderEigbars();
  renderReadout();
  renderCards();
  renderLegend();
  renderVolHint();
  renderInspect();
  renderLabels();
    if ($('.pool-table').open) renderTable();
  renderMatrices();
  updateMore();
}

const fmtT = (T) => (T < 10 ? T.toFixed(2) : T < 100 ? T.toFixed(1) : T.toFixed(0));

// ---------------------------------------------------------------- matrices (advanced section, drawn only while open)
// B has rows √w_i φ_i for all N pool rows and the DPP weight in use (weightChoice), not the display "Vector length". w is max-normalized
// (the kernel drops a constant), so √det L_S equals the page's true volume under the weight lock.
const MAT = { cache: new WeakMap(), fixed: new WeakMap(), tok: 0, timer: 0, hist: [], eigMaxN: 300 };
DBG.matFixed = () => { const f = fixedOf(S.phi); return { tBlocks: f.tBlocks, tAxes: f.tAxes }; };
DBG.matTimes = () => MAT.hist.map((e) => ({ D: e.D, tL: e.tL, tC: e.tC, tCp: e.tCp, tE: e.tE }));
const matOrder = () => document.querySelector('input[name=mord]:checked')?.value ?? 'blocks';
const dualMode = () => document.querySelector('input[name=dmode]:checked')?.value ?? 'axes';
const dualR = () => +(document.querySelector('input[name=dr]:checked')?.value ?? 32);
// Orders and axes depend on the unit embeddings only (never on T, weight, mode, K or the selection), so they are cached per pool and embedder.
const fixedOf = (phi) => MAT.fixed.get(phi) ?? (MAT.fixed.set(phi, {}), MAT.fixed.get(phi));
// The average-linkage tree on 1 - cos: its leaf order is the Blocks order and its cut into groups colors the plot.
function treeOf() {
  const f = fixedOf(S.phi);
  if (!f.tree) { const t = performance.now(); f.tree = seriation(S.G, S.phi.length); f.tBlocks = performance.now() - t; }
  return f.tree;
}
const blocksOrder = () => treeOf().order;
// Linked highlights. Crosshair lines are positioned elements over the canvases, so a hover never redraws a heatmap.
// 'h' lines follow the pointer or the plot's hover; 'p' lines follow the pinned pair (S.pinned, S.pinned2), the page's inspector.
const XH = '<i class="xh hr"></i><i class="xh hc"></i><i class="xh pr"></i><i class="xh pc"></i>';
const MX = { N: 0, pos: null, perm: null, busy: false, raf: 0, ev: null, dpin: null, dhov: null };
function placeCross(root, kind, a, b, n) {
  const r = root.querySelector(`.${kind}r`), c = root.querySelector(`.${kind}c`);
  if (a == null || a < 0) { r.style.display = c.style.display = 'none'; return; }
  r.style.display = c.style.display = 'block';
  r.style.top = `${(100 * (a + 0.5)) / n}%`;
  c.style.left = `${(100 * (b + 0.5)) / n}%`;
}
function crossPrimal(kind, i, j) {
  const root = $('#mat-pcw');
  if (!root || !MX.pos || MX.N !== S.phi?.length) return;
  placeCross(root, kind, i < 0 ? null : MX.pos.get(i), j < 0 ? null : MX.pos.get(j), MX.N);
}
// A point of the plot, a card or a table row is hovered: its row and column in the primal.
function matHover(i) { if (!MX.busy && $('#adv-mat').open) crossPrimal('h', i, i); }
function syncPinCross() {
  if (!$('#adv-mat').open) return;
  crossPrimal('p', S.pinned, S.pinned2 >= 0 ? S.pinned2 : S.pinned);
  const root = $('#mat-pcw');
  if (root && S.mat && S.pinned >= 0) {
    const j = S.pinned2 >= 0 ? S.pinned2 : S.pinned, N = S.mat.N;
    root.setAttribute('aria-label', `Primal matrix, pinned row ${S.pinned}, column ${j}, value ${fmtSig(S.mat.L[S.pinned * N + j])}. Arrow keys move the cell, Escape unpins.`);
  }
}
function pinPair(i, j) {
  inspect(i);
  S.pinned2 = j === i ? -1 : j;
  renderInspect();
  plot.highlight(-1);
}
const rafOnce = (fn) => (e) => { MX.ev = e; if (!MX.raf) MX.raf = requestAnimationFrame(() => { MX.raf = 0; fn(MX.ev); }); };
function colorbar(sc) {
  const stops = Array.from({ length: 10 }, (_, k) => `${viridisCss(k / 9)} ${(100 * k) / 9}%`).join(',');
  return `<div class="cbar"><span class="cb-v">${fmtSig(sc.min)}</span><span class="cb-grad" style="background:linear-gradient(to right,${stops})"></span><span class="cb-v">${fmtSig(sc.max)}</span></div>`;
}
const matCacheOf = (phi) => MAT.cache.get(phi) ?? (MAT.cache.set(phi, new Map()), MAT.cache.get(phi));
function drawCanvas(cv, A, n, sc) {
  cv.getContext('2d').putImageData(new ImageData(heatmapPixels(A, n, sc), n, n), 0, 0);
}
function renderMatrices() {
  if (!$('#adv-mat').open) return;
  clearTimeout(MAT.timer);
  const tok = ++MAT.tok;
  const P = $('#mat-pwrap'), Sb = $('#mat-sub'), Du = $('#mat-dual');
  if (!S.pool || !S.sel.length || S.sel.length !== S.K) {
    P.innerHTML = '<div class="mat-cap">No selected set of size K to show.</div>'; Sb.innerHTML = ''; Du.innerHTML = ''; S.mat = null; return;
  }
  const N = S.phi.length, K = S.sel.length, T = S.T[S.mode], wc = weightChoice();
  const w = weightsFromLog(weightLogScale(wc, S.mode, S.logp, S.logq, T));
  const key = [wc, S.mode, T].join('|'), cache = matCacheOf(S.phi);
  let ent = cache.get(key);
  if (!ent) {
    const t0 = performance.now();
    const m = buildAll(S.phi, w);
    ent = { m, D: m.D, L: primalMatrix(m) };
    ent.tL = performance.now() - t0;
    cache.set(key, ent);
    if (cache.size > 6) cache.delete(cache.keys().next().value);
  }
  const { L, D } = ent;
  // Primal over the pool, in the chosen row order.
  const perm = rowOrder(matOrder(), w, S.cl?.label, matOrder() === 'blocks' ? blocksOrder() : null);
  const scL = range(L), Lp = permuted(L, N, perm);
  const pos = new Map(perm.map((i, r) => [i, r]));
  const strip = (axis) => S.sel.map((i) => `<span class="tick-m" data-i="${i}" title="${escapeHtml(labelText(S.pool.samples[i]))}" style="${axis === 'x' ? 'left' : 'top'}:${(100 * pos.get(i)) / N}%;${axis === 'x' ? 'width' : 'height'}:max(2px,${100 / N}%);background:${selColor(i)}"></span>`).join('');
  const tree = matOrder() === 'blocks' ? treeOf() : null;
  const dend = (axis) => {
    if (!tree) return '';
    const { segs, maxH } = dendrogramLayout(tree.merges, N, perm), ng = S.cl?.label && clOpts().method === 'tree' && byCl() ? nodeGroups(tree.merges, N, S.cl.label) : null;
    const byColor = new Map();
    for (const g of segs) {
      const c = ng && ng[g.node] >= 0 ? clusterColor(ng[g.node]) : css('--muted');
      const f = (h) => 1 - h / (maxH || 1);
      const d = axis === 'x' ? `M${g.x1} ${f(g.y1)}L${g.x2} ${f(g.y2)}` : `M${f(g.y1)} ${g.x1}L${f(g.y2)} ${g.x2}`;
      byColor.set(c, (byColor.get(c) ?? '') + d);
    }
    return `<svg class="dend d${axis}" viewBox="${axis === 'x' ? `0 0 ${N} 1` : `0 0 1 ${N}`}" preserveAspectRatio="none" aria-hidden="true">${[...byColor].map(([c, d]) => `<path d="${d}" stroke="${c}" fill="none" stroke-width="1" vector-effect="non-scaling-stroke"/>`).join('')}</svg>`;
  };
  P.innerHTML = `<div class="pgrid${tree ? ' withd' : ''}"><span></span>${dend('x')}<span></span><div class="strip sx" title="Click to add or remove a completion">${strip('x')}</div>${dend('y')}<div class="strip sy" title="Click to add or remove a completion">${strip('y')}</div><div class="pcw" id="mat-pcw" tabindex="0" role="application" aria-label="Primal matrix, ${N} by ${N}. Arrow keys select a cell, Enter pins it, Escape unpins."><canvas id="mat-pcv" width="${N}" height="${N}"></canvas>${XH}</div></div>`
    + `<div class="mat-hover" id="mat-phover">Hover a pixel for its row, column and value.</div>${colorbar(scL)}`
    + `<div class="mat-cap">L = B B<sup>T</sup> (N × N = ${N} × ${N}), rows and columns in ${{ blocks: 'block order (similar completions next to each other)', cluster: 'group order, then weight', weight: 'weight order', pool: 'pool order' }[matOrder()]}. Colored ticks mark the K = ${K} selected completions. Click a margin position to add or remove that completion.</div>`;
  const pcv = $('#mat-pcv');
  drawCanvas(pcv, Lp, N, scL);
  MX.N = N; MX.pos = pos; MX.perm = perm;
  const phov = $('#mat-phover'), hint = phov.textContent, pcw = $('#mat-pcw');
  const cellAt = (e) => {
    const r = pcv.getBoundingClientRect();
    return [Math.min(N - 1, Math.max(0, Math.floor(((e.clientY - r.top) / r.height) * N))), Math.min(N - 1, Math.max(0, Math.floor(((e.clientX - r.left) / r.width) * N)))];
  };
  const readout = (i, j) => { phov.innerHTML = `row ${i}, column ${j}: ${fmtSig(L[i * N + j])}<br>“${escapeHtml(labelText(S.pool.samples[i]))}” / “${escapeHtml(labelText(S.pool.samples[j]))}”`; };
  const show = (i, j) => { crossPrimal('h', i, j); readout(i, j); MX.busy = true; plot.highlight(i, j === i ? -1 : j); MX.busy = false; };
  pcv.onpointermove = rafOnce((e) => { const [r, c] = cellAt(e); show(perm[r], perm[c]); });
  pcv.onpointerleave = () => { crossPrimal('h', -1, -1); phov.textContent = hint; plot.highlight(-1); };
  pcv.onclick = (e) => {
    const [r, c] = cellAt(e), i = perm[r], j = perm[c];
    if (S.pinned === i && (S.pinned2 === j || (i === j && S.pinned2 < 0))) inspect(-1); else pinPair(i, j);
  };
  pcw.onkeydown = (e) => {
    const d = { ArrowUp: [-1, 0], ArrowDown: [1, 0], ArrowLeft: [0, -1], ArrowRight: [0, 1] }[e.key];
    if (e.key === 'Escape') { inspect(-1); return; }
    if (!d && e.key !== 'Enter' && e.key !== ' ') return;
    e.preventDefault();
    let r = 0, c = 0;
    if (S.pinned >= 0) { r = pos.get(S.pinned); c = pos.get(S.pinned2 >= 0 ? S.pinned2 : S.pinned); }
    if (d) { r = Math.min(N - 1, Math.max(0, r + d[0])); c = Math.min(N - 1, Math.max(0, c + d[1])); }
    pinPair(perm[r], perm[c]);
    readout(perm[r], perm[c]);
  };
  $('#mat-pwrap').querySelectorAll('.tick-m').forEach((t) => {
    const i = +t.dataset.i;
    t.onmouseenter = () => plot.highlight(i);
    t.onmouseleave = () => plot.highlight(-1);
  });
  $('#mat-pwrap').querySelectorAll('.strip').forEach((st) => {
    st.onclick = (e) => {
      const r = st.getBoundingClientRect(), v = st.classList.contains('sx') ? (e.clientX - r.left) / r.width : (e.clientY - r.top) / r.height;
      toggleMember(perm[Math.min(N - 1, Math.max(0, Math.floor(v * N)))]);
    };
  });
  // Selected submatrix, printed.
  const Ls = submatrix(L, N, S.sel), ev0 = symEig(Ls, K).values;
  const det = ev0.reduce((a, x) => a * x, 1), sc = range(Ls);
  const head = S.sel.map((i) => `<span class="mh" title="${escapeHtml(labelText(S.pool.samples[i]))}"><span class="dot" style="background:${selColor(i)}"></span>${i}</span>`);
  let h = `<div class="mat-title">Selected submatrix</div><div class="mgrid" style="grid-template-columns:auto repeat(${K},minmax(0,1fr))"><span></span>${head.join('')}`;
  for (let r = 0; r < K; r++) {
    h += head[r].replace('class="mh"', 'class="mh row"');
    for (let c = 0; c < K; c++) {
      const v = Ls[r * K + c], rgb = viridis(unitPos(v, sc));
      h += `<span class="mc" data-r="${r}" data-c="${c}" style="background:rgb(${rgb.join(',')});color:${textOn(rgb)}">${fmtSig(v)}</span>`;
    }
  }
  h += `</div>${colorbar(sc)}<div class="mat-cap">L<sub>S</sub>, the rows and columns of L for the selected set (K × K = ${K} × ${K}). Headers are pool indices. det L<sub>S</sub> = ${fmtSig(det)}, √det L<sub>S</sub> = ${fmtSig(Math.sqrt(Math.max(0, det)))}</div>`;
  Sb.innerHTML = h;
  S.mat = { L, det, D, N, K, w };
  Sb.querySelectorAll('.mc').forEach((el) => {
    const i = S.sel[+el.dataset.r], j = S.sel[+el.dataset.c];
    el.onmouseenter = () => { crossPrimal('h', i, j); MX.busy = true; plot.highlight(i, j === i ? -1 : j); MX.busy = false; };
    el.onmouseleave = () => { crossPrimal('h', -1, -1); plot.highlight(-1); };
  });
  syncPinCross();
  // Dual over the pool, after a pause that the next change cancels. Principal axes: (B V_r)^T (B V_r), N·D·r operations; coordinates: B^T B, N·D² operations. Both cached.
  const mode = dualMode(), r = dualR();
  $('#dpar').hidden = mode !== 'axes';
  const fx = fixedOf(S.phi);
  const ready = mode === 'axes' ? fx.axes && ent.Cp?.has(r) : !!ent.C;
  if (ready) { drawDual(ent, N, K, Du, tok); return; }
  Du.innerHTML = `<div class="mat-cap">Computing the dual (D = ${D})…</div>`;
  MAT.timer = setTimeout(() => requestAnimationFrame(() => {
    if (tok !== MAT.tok || !$('#adv-mat').open) return;
    if (mode === 'axes') {
      if (!fx.axes) { const t = performance.now(); fx.axes = principalAxes(S.phi, S.G, N, 64); fx.tAxes = performance.now() - t; }
      const t0 = performance.now();
      (ent.Cp ??= new Map()).set(r, projectedDual(ent.m, fx.axes.V, r));
      ent.tCp = performance.now() - t0;
    } else {
      const t0 = performance.now();
      ent.C = dualMatrix(ent.m);
      ent.tC = performance.now() - t0;
    }
    MAT.hist.push(ent);
    drawDual(ent, N, K, Du, tok);
  }), 80);
}
function drawDual(ent, N, K, Du, tok) {
  const { D, L } = ent, mode = dualMode(), fx = fixedOf(S.phi);
  const n = Math.min(N, D), tr = trace(L, N);
  let A, m, norm = false, cap;
  if (mode === 'axes') {
    const r = Math.min(dualR(), fx.axes.V.length);
    norm = $('#mat-norm').checked;
    const Cp = ent.Cp.get(dualR());
    A = norm ? normalizeCorr(Cp, r) : Cp; m = r;
    cap = !S.expert ? `Shown in the pool's principal axes (axis 1 varies most), ${r} of them. With the Unit weight this matrix is diagonal. Other weights add off-diagonal entries. Entries are scaled so the diagonal is 1.` : `C′ = V<sub>r</sub><sup>T</sup> C V<sub>r</sub> (r × r = ${r} × ${r}) for C = B<sup>T</sup> B (D × D = ${D} × ${D}), in the pool's principal axes V (axis 1 has the largest variance of the unit embeddings; it does not need to match the 3D plot's axes). With the Unit weight C′ is diagonal; other weights put mass off the diagonal. It shows no block structure.${norm ? ' Entries are c′<sub>ij</sub> / √(c′<sub>ii</sub> c′<sub>jj</sub>), so the diagonal is 1 and the color scale spans the off-diagonal entries.' : ''}`;
  } else {
    A = ent.C; m = D;
    cap = `C = B<sup>T</sup> B (D × D = ${D} × ${D}) in embedding coordinates.`;
  }
  let sc = range(A);
  if (mode === 'axes' && norm) { const off = []; for (let a = 0; a < m; a++) for (let b = 0; b < m; b++) if (a !== b) off.push(A[a * m + b]); sc = range(off); }
  const evFrom = N <= MAT.eigMaxN ? 'L' : mode === 'coords' && D <= MAT.eigMaxN ? 'C' : null;
  const later = !ent.ev && evFrom;
  const eigLine = () => (ent.ev ? `Top ${Math.min(5, ent.ev.length)} eigenvalues of C: ${ent.ev.slice(0, 5).map((x) => fmtSig(x)).join(', ')}. ${countAbove(ent.ev)} of ${n} are above 1e-9 times the largest.` : evFrom ? `Computing eigenvalues… trace C = ${fmtSig(tr)}.` : `Eigenvalues skipped at this size. trace C = ${fmtSig(tr)}.`);
  Du.innerHTML = `<div class="dcw" id="mat-dcw"><canvas id="mat-cv" width="${m}" height="${m}" aria-label="Dual matrix, ${m} by ${m}"></canvas>${XH}</div><div class="mat-hover" id="mat-hover">Hover a pixel for its row, column and value.</div>${colorbar(sc)}`
    + `<div class="mat-cap">${cap} Rank at most min(N, D) = ${n}, N = ${N}. <span id="mat-eig">${eigLine()}</span> They are the nonzero eigenvalues of L as well.</div>`;
  const cv = $('#mat-cv');
  drawCanvas(cv, A, m, sc);
  const hov = $('#mat-hover'), hint = hov.textContent, dcw = $('#mat-dcw'), axes = mode === 'axes';
  if (MX.dpin && (MX.dpin.m !== m || MX.dpin.mode !== mode)) MX.dpin = null;
  // Loading of completion i on axis k is phi_i . v_k; the diagonal cell of axis k lists the three largest.
  const top3 = (k) => {
    const v = fx.axes.V[k], l = S.phi.map((p, i) => [i, p.reduce((a, x, d) => a + x * v[d], 0)]);
    return l.sort((x, y) => Math.abs(y[1]) - Math.abs(x[1])).slice(0, 3).map(([i, x]) => `“${escapeHtml(labelText(S.pool.samples[i]))}” (${x >= 0 ? '+' : '−'}${Math.abs(x).toFixed(2)})`).join(', ');
  };
  const txt = (r, c) => (axes ? `axis ${r + 1}, axis ${c + 1}: ${fmtSig(A[r * m + c])}${r === c ? `<br>Largest loadings: ${top3(r)}` : ''}` : `row ${r}, column ${c}: ${fmtSig(A[r * m + c])}`);
  const showPin = () => { if (MX.dpin) { placeCross(dcw, 'p', MX.dpin.r, MX.dpin.c, m); hov.innerHTML = txt(MX.dpin.r, MX.dpin.c); } else { placeCross(dcw, 'p', null, null, m); hov.textContent = hint; } };
  const cellAt = (e) => {
    const rc = cv.getBoundingClientRect();
    return [Math.min(m - 1, Math.max(0, Math.floor(((e.clientY - rc.top) / rc.height) * m))), Math.min(m - 1, Math.max(0, Math.floor(((e.clientX - rc.left) / rc.width) * m)))];
  };
  cv.onpointermove = rafOnce((e) => { const [r, c] = cellAt(e); placeCross(dcw, 'h', r, c, m); hov.innerHTML = txt(r, c); });
  cv.onpointerleave = () => { placeCross(dcw, 'h', null, null, m); showPin(); };
  cv.onclick = (e) => {
    const [r, c] = cellAt(e);
    MX.dpin = MX.dpin && MX.dpin.r === r && MX.dpin.c === c ? null : { r, c, m, mode };
    showPin();
  };
  showPin();
  if (later) {
    // Eigenvalues of the smaller of L and C (same nonzero eigenvalues), after the picture is up; cached.
    setTimeout(() => {
      if (tok !== MAT.tok || !$('#adv-mat').open) return;
      const t1 = performance.now();
      ent.ev = (evFrom === 'L' ? eigSym(L, N) : eigSym(ent.C, D)).values;
      ent.tE = performance.now() - t1;
      const el = $('#mat-eig');
      if (el) el.textContent = eigLine();
    }, 30);
  }
}
function renderEigbars() {
  const el = $('#eigbars');
  const f = S.view.barFrac, shared = !!S.shared;
  const max = Math.max(...f) || 1;
  el.innerHTML = `<div class="cap">${shared || S.view.exact || S.view.fixed ? 'Axes' : 'PCA'} <button type="button" class="xbtn" data-x="pca" aria-label="About the projection">?</button></div>` + f.map((v, k) => {
    const click = k < 3;
    return `<div class="bar ${k < S.dims ? 'on' : ''}" data-k="${k + 1}" ${click ? `role="button" tabindex="0" aria-label="Show ${k + 1}D"` : ''}
      title="Axis ${k + 1}: ${(100 * v).toFixed(1)}% of ${S.view.exact ? "this pool's spread, on the axes chosen for the selected set" : S.view.fixed ? (shared ? "this pool's spread. Axes are shared across models, so they are not always in decreasing order" : "this pool's spread, axes ordered by spread") : shared ? "this pool's spread, on a shared axis" : 'variance'}${click ? ` · click for ${k + 1}D` : ''}">
      <span>${k + 1}</span><span class="fill" style="width:${(100 * v) / max}%"></span></div>`;
  }).join('') + `<div class="pct">${(100 * (S.view.exact ? S.view.frac : f.slice(0, S.dims)).reduce((a, b) => a + b, 0)).toFixed(0)}% shown</div>${S.view.exact ? '<div class="sub">fit to selection</div>' : S.view.fixed ? '<div class="sub">fixed axes</div>' : shared ? '<div class="sub">shared axes</div>' : ''}`;
  el.querySelectorAll('.bar[role=button]').forEach((b) => {
    const go = () => setDims(+b.dataset.k);
    b.onclick = go;
    b.onkeydown = (e) => { if (e.key === 'Enter' || e.key === ' ') { e.preventDefault(); go(); } };
  });
}

// Click on a row or column of the matrix: add the completion to the selected set or remove it (at least 1 and at most 20 stay).
function toggleMember(i) {
  if (!S.pool || i < 0 || i >= S.logp.length) return;
  const sel = S.sel.slice(), k = sel.indexOf(i);
  if (k >= 0) { if (sel.length <= 1) return; sel.splice(k, 1); } else { if (sel.length >= 20) return; sel.push(i); }
  S.manual = true; S.sel = sel; S.K = sel.length;
  $('#K').value = S.K;
  recompute({ newDraw: false });
}

// "Add more samples": the same request again with a new seed, appended to the live pool. Selection and block order are recomputed; the view keeps its camera.
const MORE_MAX = 400;
function updateMore() {
  const row = $('#more-row'), btn = $('#add-more'), note = $('#more-note');
  if (!S.pool) { row.hidden = true; return; }
  row.hidden = false;
  const N = S.pool.samples.length, b = S.source === 'custom' ? backendById(S.where) : null;
  let why = '';
  if (S.source !== 'custom') why = 'Precomputed pools have a fixed size. Press ↑ to sample your own prompt, then add more.';
  else if (!b) why = 'No backend is available for this run.';
  else if (S.pending) why = 'A run is in progress.';
  else if (N + (S.cust?.N ?? 0) > MORE_MAX) why = `The pool is limited to ${MORE_MAX} samples.`;
  btn.disabled = !!why;
  btn.title = why;
  note.textContent = why ? (S.source === 'custom' ? why : '') : `${N} samples now, adds ${S.cust.N} more with a new seed.`;
  if (S.source !== 'custom') row.hidden = true;
}
async function addMore() {
  const c = S.cust, b = backendById(S.where);
  if (!c || !b || S.pending || S.source !== 'custom') return;
  const key = S.model, ck = custKey(key, S.where), seq = ++S.runSeq, ctl = new AbortController();
  run = { seq, abort: ctl, kind: b.kind };
  S.pending = true; setBusy(true); updateMore();
  $('#live-status').textContent = `Sampling ${c.N} more from ${info(key).name} on ${whereName(b)}…`;
  const t0 = performance.now();
  try {
    const req = { ...requestFor(c, key), embedder: S.embedder };
    const r = await b.sample(req, { signal: ctl.signal, onProgress: (d) => { if (seq === S.runSeq && d.text) $('#live-status').textContent = d.text; } });
    if (seq !== S.runSeq) return;
    if (r.pool.D !== S.phi[0].length) throw new Error(`the new samples have embedding size ${r.pool.D}, the pool has ${S.phi[0].length}`);
    const add = entryFromResult(r), samples = S.pool.samples.concat(add.rec.samples);
    const rec = { ...S.pool, samples, N: samples.length }, phi = S.phi.concat(add.phi);
    const e = makeEntry(rec, phi);
    remember(ck, e);
    run = null; S.pending = false; setBusy(false);
    $('#live-status').textContent = `${samples.length} completions now (${add.N} added in ${((performance.now() - t0) / 1000).toFixed(1)}s).`;
    setPool(rec, phi, { entry: e, keepPlot: true });
  } catch (err) {
    if (err.name === 'AbortError' || seq !== S.runSeq) return;
    run = null; S.pending = false; setBusy(false);
    $('#live-status').textContent = 'Error: ' + (err.message || err);
    updateMore();
  }
}

function setDims(k) { S.dims = k; plot.reset(); render(); }

function renderReadout() {
  const N = S.logp.length;
  const ok = S.sel.length === S.K;
  const e = weightEss(weightChoice(), S.mode, S.logp, S.logq, S.T[S.mode]); // ESS of the weights in use
  const stat = (k, v, x) => `<span class="stat">${k} <button type="button" class="xbtn" data-x="${x}" aria-label="What is ${k}?">?</button> <span class="v">${v}</span></span>`;
  $('#readout').innerHTML = !ok
    ? `<span class="stat">No subset of size ${S.K}: there are fewer distinct completions than that. Lower K or generate more.</span>`
    : stat('log P(S)', fmt(S.logP), 'logp') +
      stat('quality', fmt(quality(S.logp, S.sel)), 'quality') +
      stat('diversity', fmt(diversity(S.G, N, S.sel)), 'diversity') +
      stat('ESS', `${e.toFixed(0)} of ${N} draws`, 'ess');
  // Near-duplicates: the largest group of the average-linkage tree at cosine 0.95 (1 - cos <= 0.05), cached per pool.
  const fx = fixedOf(S.phi);
  fx.near ??= { size: largestGroup(treeOf().merges, 0.05) };
  const notes = readoutNotes({ ess: ok ? e : NaN, K: S.K, N, distinct: S.distinct, near: fx.near });
  const nbox = $('#rnotes');
  nbox.hidden = !notes.length;
  nbox.innerHTML = notes.map((n) => `<div class="rnote">${escapeHtml(n.text)}${n.kind === 'distinct' || n.kind === 'near' ? ' <button type="button" class="link" data-act="tp">Change T′</button>' : ''}</div>`).join('');
  nbox.querySelectorAll('[data-act=tp]').forEach((b) => (b.onclick = changeTprime));
}
// Reach T′ from the note even when the T′ controls are expert-only: own prompts open the run settings, examples switch to the next higher precomputed T′.
function changeTprime() {
  if (S.source === 'example') {
    const higher = exampleTps(S.exId).map(Number).filter((t) => t > Number(S.tp)).sort((a, b) => a - b)[0];
    if (higher !== undefined) { showExample(S.exId, { key: S.model, tp: higher.toFixed(1), morph: true }); return; }
  }
  $('#adv').open = true;
  $('#live-Tp').focus();
}

function renderVolHint() {
  const el = $('#vol-hint');
  if (S.dims === 1) { el.innerHTML = ''; return; }
  const K = S.sel.length, v = S.view;
  const shown = K >= 2 && K <= 3;
  const mesh = K > 3 && S.sel.length === S.K;
  const scaled = lenChoice() !== 'unit';
  const num = (x) => `<span class="vn">${x < 0.01 ? x.toExponential(1) : x.toFixed(2)}</span>`;
  let txt = '';
  if (shown && S.sel.length === S.K) {
    const lg = 0.5 * logDetSub(S.G, S.logp.length, S.sel);
    const truth = Math.exp(lg + S.sel.reduce((a, i) => a + Math.log(S.cDisp[i]), 0));
    const name = volName();
    const drawn = drawnVolume();
    if (v.exact && v.s === 1) txt = `Volume drawn = ${name} = ${num(drawn)} (exact: the axes span the selected set)`;
    else if (v.exact) txt = `Volume drawn = ${num(drawn)} = ${num(S.dims === 3 ? v.s * v.s : v.s)} × ${name} (${name} = ${num(truth)})`;
    else if (v.fixed) txt = `${Number.isFinite(drawn) ? `Volume drawn ${num(drawn)}, ` : ''}true ${name} ${num(truth)}${v.s !== 1 && Number.isFinite(drawn) ? ` (drawn includes ×${num(S.dims === 3 ? v.s * v.s : v.s)} from the opened cone)` : ''}`;
    else txt = `Volume drawn = ${num(drawn)}, true ${name} = ${num(truth)} (projection: the selected set does not fit in ${S.dims} axes)`;
  }
  if (mesh) {
    const lg = 0.5 * logDetSub(S.G, S.logp.length, S.sel);
    const truth = Math.exp(lg + S.sel.reduce((a, i) => a + Math.log(S.cDisp[i]), 0));
    const name = volName();
    const tips = S.sel.map((i) => Array.from(S.view.coords[i], (x) => x * S.cDisp[i]));
    const area = S.dims === 3 ? tipSheet(tips).area : polyArea(hull2D(tips));
    if (area > 0) txt = `${S.dims === 3 ? 'Mesh area' : 'Polygon area'} ${num(area)} (the ${S.dims === 3 ? 'open surface' : 'outline'} through the ${K} tips in this view), true ${name} ${num(truth)}`;
    else txt = `True ${name} ${num(truth)}`;
  }
  if (sizeQ() && txt) txt += '. Point size shows quality, so the vectors are drawn at unit length and the drawn volume is the unit-vector volume';
  if ($('#cone-toggle').checked && S.dims > 1) {
    txt += `${txt ? '. ' : ''}Cone opened: widths ×${v.s.toFixed(2)}, every ${S.dims === 3 ? 'volume' : 'area'} ×${(S.dims === 3 ? v.s * v.s : v.s).toFixed(2)} (ratios unchanged)`;
  }
  // Why the shape is what it is: a box needs K = 2 or 3 vectors in a view with that many axes.
  let why = '';
  if (K > 3) why = S.dims === 3 ? `K = ${K}: more than three vectors do not form a box in 3D, so the surface through their tips is shown, with its area.` : `K = ${K}: more than ${S.dims} vectors do not form a box in this view, so the outline through their tips is shown, with its area.`;
  else if (K === 3 && S.dims === 2) why = 'K = 3 in the 2D view: the box is drawn flat.';
  // The old "Show the spanned volume (K = 3)" link is gone: for K > 3 the plot shows the mesh through the tips instead.
  el.innerHTML = shown || mesh
    ? `<svg class="sw-vol" width="18" height="12" viewBox="0 0 18 12" aria-hidden="true"><path d="M1 11L5 1H17L13 11Z"/></svg>${mesh ? (S.dims === 3 ? 'surface through the tips' : 'outline of the tips') : 'spanned volume'} <button type="button" class="xbtn" data-x="volume" aria-label="What is the spanned volume?">?</button>${why ? `<span class="vtxt">${why}</span>` : ''}${txt ? `<span class="vtxt">${txt}</span>` : ''}`
    : (txt ? `<span class="vtxt">${txt}</span>` : '');
}

function renderLegend() {
  const on = byCl(), box = $('#cl-legend');
  $('#lg-ramp').hidden = on; $('#lg-sel').hidden = on; box.hidden = !on;
  S.dimCl = null;
  if (!on) { box.innerHTML = ''; return; }
  const cnt = new Array(S.cl.k).fill(0);
  S.cl.label.forEach((c) => { if (c >= 0) cnt[c]++; });
  const chip = (c, name, n, dot, title) => `<button type="button" class="clchip" data-c="${c}" title="${escapeHtml(title)}"><span class="sw" style="${dot}"></span>${escapeHtml(name)} <span class="vn">${n}</span></button>`;
  box.innerHTML = cnt.map((n, c) => chip(c, S.cl.kw[c], n, `background:${clusterColor(c)}`, `${S.cl.kw[c]}: ${n} completions`)).join('')
    + (S.cl.nIso ? chip(-1, 'isolated', S.cl.nIso, `border:1.5px solid ${css('--text-2')};background:transparent`, `${S.cl.nIso} completions far from every cluster, left uncolored`) : '')
    + '<span class="clhint">stronger color = more likely</span>';
  box.querySelectorAll('.clchip').forEach((b) => {
    const on1 = () => plot.dim(+b.dataset.c), off = () => plot.dim(null);
    b.onmouseenter = on1; b.onfocus = on1; b.onmouseleave = off; b.onblur = off;
  });
}

function renderCards() {
  const order = S.sel.slice().sort((a, b) => S.logp[b] - S.logp[a]);
  $('#selected').innerHTML = order.map((i) => {
    const s = S.pool.samples[i];
    return `<li class="card${s.ended ? '' : ' cut'}" data-i="${i}"${byCl() ? ` style="border-left:4px solid ${selColor(i)}"` : ''}>
      <div class="txt">${escapeHtml(s.text)}</div>
      <div class="meta"><span class="dot" style="background:${selColor(i)}"></span>log p ${fmt(s.logp, 1)}<span>·</span>P(in set) ${fmt(S.pi[i], 2)}</div></li>`;
  }).join('');
  $('#selected').querySelectorAll('.card').forEach((el) => {
    el.onmouseenter = () => plot.highlight(+el.dataset.i);
    el.onmouseleave = () => plot.highlight(-1);
    el.onclick = () => inspect(+el.dataset.i);
  });
}

// ---------------------------------------------------------------- tooltip
const tip = $('#tooltip');
function showTip(i, x, y) {
  if (i < 0) { tip.hidden = true; return; }
  const s = S.pool.samples[i];
  const w = weightInfo();
  tip.innerHTML = `${escapeHtml(s.text)}${s.ended ? '' : '…'}<div class="meta">#${i} · log p ${fmt(s.logp, 2)} · log q ${fmt(s.logq, 2)}· ${w.name} ${fmt(w.f(s), 2)}<br>P(i ∈ S) = ${fmt(S.pi[i], 3)}${S.sel.includes(i) ? ' · <b>selected</b>' : ''}</div>`;
  tip.hidden = false;
  const r = tip.getBoundingClientRect();
  tip.style.left = Math.min(window.innerWidth - r.width - 8, x + 14) + 'px';
  tip.style.top = Math.min(window.innerHeight - r.height - 8, y + 14) + 'px';
}

// ---------------------------------------------------------------- projection plot
// With K ≤ 3 each selected vector gets its own colour and we draw the
// parallelogram/parallelepiped it spans (paper, Fig. 1); otherwise one accent.
const VEC = ['--series-1', '--series-2', '--series-3'];
const byCl = () => $('#color-toggle').checked && !!S.cl;
const clusterColor = (c) => (c < 0 ? css('--text-2') : css(`--c${(c % 8) + 1}`));

// ---- clustering of the completions (a coloring aid only; the DPP never sees it)
// Settings live in the View disclosure; results are cached per pool and setting.
const memoOf = (pool, name) => { if (!Object.hasOwn(pool, name)) Object.defineProperty(pool, name, { value: new Map(), enumerable: false }); return pool[name]; };
const clOpts = () => ({
  method: document.querySelector('input[name=clm]:checked')?.value ?? DEFAULTS.clm,
  k: clampInt($('#cl-k').value, 2, 8),
  space: document.querySelector('input[name=cls]:checked')?.value ?? DEFAULTS.cls,
  names: document.querySelector('input[name=cln]:checked')?.value ?? 'num',
  iso: false, // every completion gets a cluster color: with linkage a far point becomes its own small cluster
});
function clustersFor() {
  const pool = S.pool, N = pool.samples.length, o = clOpts(), cache = memoOf(pool, 'clc'), dm = memoOf(pool, 'dm');
  const key = [S.embedder, o.method, o.k, o.space, o.iso, o.names].join('|');
  if (cache.has(key)) return cache.get(key);
  // Distances once per (pool, space): unit embeddings of the active embedder, or the fixed view's unit positions.
  const D = dm.get(o.space) ?? (dm.set(o.space, o.space === 'view' ? distFromCoords(S.fixed.coords) : distFromGram(S.G, N)), dm.get(o.space));
  let label, k, diam;
  if (o.method === 'tree') {
    const De = dm.get('emb') ?? (dm.set('emb', distFromGram(S.G, N)), dm.get('emb'));
    const c = cutClusters(treeOf().merges, N, { k: o.k, minSize: 1, D: De });
    label = c.label; k = c.k; diam = c.diam;
  } else if (o.method === 'kmeans') {
    const unit = (r) => { const n = Math.hypot(...r) || 1; return Float32Array.from(r, (x) => x / n); };
    const r = clusterPool(o.space === 'view' ? S.fixed.coords.map(unit) : S.phi, o.k, 1, { isolate: o.iso });
    label = r.label; k = r.k; diam = clusterDiameters(label, k, D, N);
  } else {
    const dk = `d|${o.space}|${o.method}`;
    if (!dm.has(dk)) dm.set(dk, clusterLinkage(D, N, { method: o.method === 'avg' ? 'average' : 'complete' }));
    const c = cutClusters(dm.get(dk), N, { k: o.k, minSize: o.iso ? 3 : 1, D });
    label = c.label; k = c.k; diam = c.diam;
  }
  if (S.distinct <= 3) { label = new Int32Array(N); k = 1; diam = new Float32Array(1); } // identical strings: one group
  const members = (c) => Array.from(label, (l, i) => (l === c ? i : -1)).filter((i) => i >= 0);
  const kw = o.names === 'kw' ? clusterKeywords(pool.samples.map((x) => x.text), label, k, S.logp)
    : o.names === 'medoid' ? Array.from({ length: k }, (_, c) => labelText(pool.samples[medoidOf(S.G, N, members(c))]))
    : Array.from({ length: k }, (_, c) => `Group ${c + 1}`);
  const res = { label, k, diam, nIso: label.reduce((a, l) => a + (l < 0 ? 1 : 0), 0), kw };
  cache.set(key, res);
  if (cache.size > 40) cache.delete(cache.keys().next().value);
  return res;
}
function onClusterChange() {
  if (!S.pool) return;
  S.cl = clustersFor();
  render();
}
// The clustering settings look disabled while Color by cluster is off.
function syncCl() {
  const on = $('#color-toggle').checked;
  $('#adv-cl').classList.toggle('off', !on);
  $('#adv-cl').querySelectorAll('input, button.btn').forEach((el) => (el.disabled = !on));
}
function selColor(i) {
  if (byCl()) return clusterColor(S.cl.label[i]);
  const k = S.sel.indexOf(i);
  return S.sel.length <= 3 && k >= 0 ? css(VEC[k]) : css('--accent');
}
// Stroke of a selected 2D marker: its colour, or ink when the fill already carries the cluster colour.
const selStroke = (i) => (byCl() ? css('--text-2') : selColor(i));
// Faces (as corner lists) and corner points of the parallelotope spanned by vectors V from origin O.
function parallelotope(O, V) {
  const add = (a, b) => a.map((x, d) => x + b[d] - O[d]);
  const corner = (mask) => V.reduce((acc, v, k) => (mask & (1 << k) ? add(acc, v) : acc), O.slice());
  const faces = [];
  for (let a = 0; a < V.length; a++) for (let b = a + 1; b < V.length; b++) {
    const others = [...Array(V.length).keys()].filter((k) => k !== a && k !== b);
    const offs = others.length ? [0, 1 << others[0]] : [0];
    for (const o of offs) faces.push([o, o | (1 << a), o | (1 << a) | (1 << b), o | (1 << b)].map(corner));
  }
  const full = (1 << V.length) - 1;
  const extra = [];
  for (let mask = 1; mask <= full; mask++) if (mask & (mask - 1)) extra.push({ p: corner(mask), far: mask === full });
  return { faces, corners: extra };
}
const colorFor = () => {
  const lp = S.logp.slice().sort((a, b) => a - b);
  const lo = lp[Math.floor(0.05 * (lp.length - 1))], hi = lp[lp.length - 1];
  const sc = d3.scaleLinear().domain([lo, hi]).range([css('--ramp-lo'), css('--ramp-hi')]).interpolate(d3.interpolateLab).clamp(true);
  if (!byCl()) return (i) => sc(S.logp[i]);
  // Cluster colour mixed toward the background by likelihood: full colour for the most likely, 35% for the least.
  const bg = css('--surface'), t = d3.scaleLinear().domain([lo, hi]).range([0.35, 1]).clamp(true), sel = new Set(S.sel);
  return (i) => {
    const lab = S.cl.label[i];
    const c = lab < 0 ? sc(S.logp[i]) : d3.interpolateRgb(bg, clusterColor(lab))(t(S.logp[i])); // isolated points: the grey likelihood ramp
    return S.dimCl !== null && lab !== S.dimCl && !sel.has(i) ? d3.interpolateRgb(bg, c)(0.15) : c;
  };
};

class Plot2D {
  constructor(el) { this.el = el; this.svg = null; this.raf = 0; this.z = { k: 1, x: 0, y: 0 }; }
  reset() { this.stopAnim(); this.el.innerHTML = ''; this.svg = null; this.z = { k: 1, x: 0, y: 0 }; }
  stopAnim() { if (this.raf) cancelAnimationFrame(this.raf); this.raf = 0; }
  build() {
    const el = this.el;
    this.W = el.clientWidth || 600;
    this.H = el.clientHeight || 450;
    const svg = d3.select(el).append('svg').attr('viewBox', `0 0 ${this.W} ${this.H}`);
    svg.append('defs').append('marker').attr('id', 'arrow').attr('viewBox', '0 0 10 10').attr('refX', 9).attr('refY', 5)
      .attr('markerWidth', 7).attr('markerHeight', 7).attr('orient', 'auto-start-reverse')
      .append('path').attr('d', 'M0,0L10,5L0,10z').attr('fill', css('--accent'));
    const root = (this.root = svg.append('g').attr('class', 'zoomroot').attr('transform', this.ztransform()));
    this.gOrigin = root.append('g');
    this.gOrigin.append('path').attr('class', 'mark').attr('stroke', css('--muted')).attr('stroke-width', 1);
    this.gOrigin.append('text').attr('class', 'direct-label').attr('font-size', 11).attr('fill', css('--muted')).text('origin');
    this.gPara = root.append('g');
    this.gTrail = root.append('g');
    this.gGhost = root.append('g');
    this.gArrow = root.append('g');
    this.gOld = root.append('g'); // the previous model's points, fading out during a morph
    this.gDot = root.append('g');
    this.svg = svg;
    this.jit = S.logp.map((_, i) => ((i * 0.6180339887) % 1) - 0.5);
    const nearest = (ev, r) => {
      const [sx, sy] = d3.pointer(ev), { k, x, y } = this.z;
      const px = (sx - x) / k, py = (sy - y) / k;
      let best = -1, bd = (r / k) ** 2;
      this.pos.forEach(([a, b], i) => { const d = (a - px) ** 2 + (b - py) ** 2; if (d < bd) { bd = d; best = i; } });
      return best;
    };
    // Zoom: wheel, two-finger pinch and drag with the mouse; double click resets. One finger scrolls the page (touch-action pan-x pan-y),
    // so a pinch reaches the plot as two pointers.
    svg.style('touch-action', 'pan-x pan-y');
    const pointers = new Map();
    let drag = null, pinch = null, moved = 0;
    svg.on('wheel', (ev) => {
      ev.preventDefault();
      const [sx, sy] = d3.pointer(ev);
      this.zoomAt(sx, sy, Math.exp(-ev.deltaY * (ev.deltaMode === 1 ? 0.05 : 0.0015)));
    }, { passive: false });
    svg.on('dblclick', () => { this.z = { k: 1, x: 0, y: 0 }; this.applyZoom(); });
    svg.on('pointerdown', (ev) => {
      pointers.set(ev.pointerId, d3.pointer(ev));
      moved = 0;
      if (pointers.size === 2) { const [a, b] = [...pointers.values()]; pinch = { d: Math.hypot(a[0] - b[0], a[1] - b[1]), c: [(a[0] + b[0]) / 2, (a[1] + b[1]) / 2] }; drag = null; }
      else if (ev.pointerType === 'mouse') drag = { x: ev.clientX, y: ev.clientY };
    });
    svg.on('pointermove.zoom', (ev) => {
      if (pointers.has(ev.pointerId)) pointers.set(ev.pointerId, d3.pointer(ev));
      if (pinch && pointers.size === 2) {
        const [a, b] = [...pointers.values()], d = Math.hypot(a[0] - b[0], a[1] - b[1]), c = [(a[0] + b[0]) / 2, (a[1] + b[1]) / 2];
        this.z.x += c[0] - pinch.c[0]; this.z.y += c[1] - pinch.c[1];
        this.zoomAt(c[0], c[1], d / (pinch.d || d));
        pinch = { d, c }; moved = 99;
      } else if (drag && ev.buttons === 1) {
        const dx = ev.clientX - drag.x, dy = ev.clientY - drag.y;
        moved += Math.abs(dx) + Math.abs(dy);
        if (moved > 4 && this.z.k > 1) { this.z.x += dx; this.z.y += dy; this.applyZoom(); }
        drag = { x: ev.clientX, y: ev.clientY };
      }
    });
    const up = (ev) => { pointers.delete(ev.pointerId); if (pointers.size < 2) pinch = null; if (!pointers.size) drag = null; };
    svg.on('pointerup', up).on('pointercancel', up);
    svg.on('pointermove', (ev) => {
      if (ev.pointerType !== 'mouse' || (ev.buttons === 1 && moved > 4)) return;
      const best = nearest(ev, 18);
      this.highlight(best);
      showTip(best, ev.clientX, ev.clientY);
    }).on('pointerleave', () => { this.highlight(-1); showTip(-1); })
      .on('click', (ev) => { if (moved > 4) return; inspect(nearest(ev, ev.pointerType === 'mouse' ? 18 : 32)); });
  }
  ztransform() { return `translate(${this.z.x},${this.z.y}) scale(${this.z.k})`; }
  applyZoom() { this.root?.attr('transform', this.ztransform()); if (S.dims !== 3) placeLabels(); }
  // Zoom by `f` about the screen point (sx, sy); the zoom stays within 1 to 12 and the content keeps covering the plot.
  zoomAt(sx, sy, f) {
    const z = this.z, k = Math.min(12, Math.max(1, z.k * f)), r = k / z.k;
    z.x = sx - (sx - z.x) * r; z.y = sy - (sy - z.y) * r; z.k = k;
    if (k === 1) { z.x = 0; z.y = 0; } else { z.x = Math.min(0, Math.max(this.W * (1 - k), z.x)); z.y = Math.min(0, Math.max(this.H * (1 - k), z.y)); }
    this.applyZoom();
  }
  screenXY(i) { const p = this.pos?.[i]; return p && [p[0] * this.z.k + this.z.x, p[1] * this.z.k + this.z.y]; }
  // Data-space position of item i (scale = 1 for the unit embedding).
  dataXY(i, scale) {
    const c = this.C[i];
    return S.dims === 1 ? [c[0] * scale, this.jit[i] * 1.6] : [c[0] * scale, c[1] * scale];
  }
  // Fit scales to everything drawn, including the far corner of the spanned volume (arrows view: and the origin).
  fit(points, withOrigin = true) {
    const { W, H } = this, m = 16, dims = S.dims;
    const pad = (a) => { const lo = d3.min(a), hi = d3.max(a), p = 0.06 * (hi - lo || 1); return [lo - p, hi + p]; };
    const o = withOrigin ? [0] : [];
    let [x0, x1] = pad(points.map((q) => q[0]).concat(o));
    let [y0, y1] = dims === 1 ? [-1, 1] : pad(points.map((q) => q[1]).concat(o));
    if (dims === 2) {
      const s = Math.min((W - 2 * m) / (x1 - x0), (H - 2 * m) / (y1 - y0));
      const cx = (x0 + x1) / 2, cy = (y0 + y1) / 2;
      x0 = cx - (W - 2 * m) / s / 2; x1 = cx + (W - 2 * m) / s / 2;
      y0 = cy - (H - 2 * m) / s / 2; y1 = cy + (H - 2 * m) / s / 2;
    }
    this.x = d3.scaleLinear().domain([x0, x1]).range([m, W - m]);
    this.y = d3.scaleLinear().domain([y0, y1]).range([H - m, m]);
  }
  // What a morph needs from the view being left: pixel positions, selected set, tangent point, point styles.
  snapshot() {
    if (!this.svg || !this.pos || !this.lastOrg) return null;
    return { kind: '2d', dims: S.dims, pos: this.pos.map((p) => p.slice()), sel: S.sel.slice(), origin: this.lastOrg, r: this.lastR, fill: this.lastFill };
  }
  // mf: a snapshot() of the previous view (morph: the selected set glides, the other points cross-fade), or { instant } for no animation.
  update(mf0 = null) {
    if (!this.svg) this.build();
    this.stopAnim();
    const mf = mf0 && mf0.kind === '2d' && mf0.dims === S.dims && !mf0.instant && !REDUCED.matches ? mf0 : null;
    const instant = !!mf0?.instant || REDUCED.matches;
    const N = S.logp.length, dims = S.dims, H = this.H;
    this.C = viewCoords();
    const scaled = lenChoice() !== 'unit';
    const col = colorFor();
    const sel = new Set(S.sel);
    const idx = d3.range(N);
    const ghostD = idx.map((i) => this.dataXY(i, 1));
    const posD = idx.map((i) => (dims === 1 ? [this.dataXY(i, S.cDisp[i])[0], ghostD[i][1]] : this.dataXY(i, S.cDisp[i])));
    const vol = dims === 2 && S.sel.length >= 2 && S.sel.length <= 3;
    const hullOn = dims === 2 && S.sel.length > 3;
    const volD = vol ? parallelotope([0, 0], S.sel.map((i) => posD[i])) : hullOn ? { faces: [hull2D(S.sel.map((i) => posD[i]))], corners: [] } : { faces: [], corners: [] };
    this.fit([...posD, ...(scaled ? ghostD : []), ...volD.corners.map((c) => c.p)]);
    const px = (q) => [this.x(q[0]), this.y(q[1])];
    const ghostPos = ghostD.map(px);
    this.pos = posD.map(px);
    const ox = this.x(0), oy = this.y(0);
    const t = this.svg.transition().duration(instant ? 0 : 450).ease(d3.easeCubicOut);
    // Plain updates ease to the new state; in a morph, attributes are set at once and the selected set is animated below.
    const tr = (s) => (mf ? s.interrupt() : s.transition(t));
    const MORPH_MS = 500, inOut = d3.easeCubicInOut;

    const to = (s) => (mf ? s.transition().duration(MORPH_MS).ease(inOut) : s.transition(t));
    to(this.gOrigin.select('path.mark').attr('stroke-dasharray', dims === 1 ? '3 3' : null))
      .attr('d', dims === 1 ? `M${ox},16V${H - 16}` : `M${ox - 6},${oy}H${ox + 6}M${ox},${oy - 6}V${oy + 6}`);
    to(this.gOrigin.select('text')).attr('x', ox + 6).attr('y', dims === 1 ? 26 : oy - 6);
    const ghostOn = scaled && S.expert && $('#ghost-toggle').checked;
    const gh = this.gGhost.selectAll('circle').data(idx).join('circle').attr('r', 3).attr('fill', css('--ghost'));
    const tl = this.gTrail.selectAll('line').data(idx).join('line').attr('stroke', css('--ghost')).attr('stroke-width', 1);
    if (mf) {
      gh.interrupt().attr('opacity', 0).attr('cx', (i) => ghostPos[i][0]).attr('cy', (i) => ghostPos[i][1])
        .transition().duration(MORPH_MS).ease(inOut).attr('opacity', ghostOn ? 0.45 : 0);
      tl.interrupt().attr('opacity', 0).attr('x1', (i) => ghostPos[i][0]).attr('y1', (i) => ghostPos[i][1]).attr('x2', (i) => this.pos[i][0]).attr('y2', (i) => this.pos[i][1])
        .transition().duration(MORPH_MS).ease(inOut).attr('opacity', ghostOn ? 0.35 : 0);
    } else {
      gh.attr('opacity', ghostOn ? 0.45 : 0).transition(t).attr('cx', (i) => ghostPos[i][0]).attr('cy', (i) => ghostPos[i][1]);
      tl.attr('opacity', ghostOn ? 0.35 : 0).transition(t).attr('x1', (i) => ghostPos[i][0]).attr('y1', (i) => ghostPos[i][1])
        .attr('x2', (i) => this.pos[i][0]).attr('y2', (i) => this.pos[i][1]);
    }
    const few = S.sel.length <= 3;
    const y0 = (i) => (dims === 1 ? this.pos[i][1] : oy);
    tr(this.gArrow.selectAll('line').data(S.sel, (i) => i).join(
      (enter) => enter.append('line').attr('x1', ox).attr('y1', y0).attr('x2', ox).attr('y2', y0),
    ).attr('stroke', (i) => selColor(i)).attr('stroke-width', few ? 2.5 : 2)
      .attr('marker-end', few ? null : 'url(#arrow)'))
      .attr('x1', ox).attr('y1', y0)
      .attr('x2', (i) => this.pos[i][0]).attr('y2', (i) => this.pos[i][1]);
    // Volume spanned by the selected set (K = 2: parallelogram, K = 3: parallelepiped),
    // from the quality-scaled vectors; det L_S is its squared volume.
    const face = (f) => 'M' + f.map((q) => px(q).join(',')).join('L') + (f.length > 2 ? 'Z' : '');
    tr(this.gPara.selectAll('path').data(volD.faces).join('path')
      .attr('fill', css('--text')).attr('fill-opacity', 0.04)
      .attr('stroke', css('--text-2')).attr('stroke-opacity', 0.7).attr('stroke-width', 1).attr('stroke-linejoin', 'round'))
      .attr('d', face);
    tr(this.gPara.selectAll('circle').data(volD.corners).join('circle')
      .attr('fill', css('--text-2')).attr('r', (c) => (c.far ? 4.5 : 2.5)))
      .attr('cx', (c) => this.x(c.p[0])).attr('cy', (c) => this.y(c.p[1]));
    const radius = (i) => (sel.has(i) ? 6 : 4.5) * (S.rsz ? S.rsz[i] : 1);
    const fillOf = (i) => (byCl() && sel.has(i) ? selColor(i) : col(i));
    const dots = this.gDot.selectAll('circle').data(idx).join('circle')
      .attr('fill', fillOf)
      .attr('stroke', (i) => (sel.has(i) ? selStroke(i) : css('--surface')))
      .attr('stroke-width', (i) => (sel.has(i) ? (byCl() ? 1.5 : 2.5) : 1))
      .sort((a, b) => sel.has(a) - sel.has(b));
    if (mf) {
      dots.interrupt().attr('r', radius).attr('cx', (i) => this.pos[i][0]).attr('cy', (i) => this.pos[i][1]).attr('opacity', (i) => (sel.has(i) ? 1 : 0));
      dots.filter((i) => !sel.has(i)).transition().duration(MORPH_MS).ease(inOut).attr('opacity', 1);
      // The previous model's unselected points fade out where they were.
      const was = new Set(mf.sel);
      this.gOld.selectAll('circle').interrupt().remove();
      this.gOld.selectAll('circle').data(d3.range(mf.pos.length).filter((i) => !was.has(i))).join('circle')
        .attr('cx', (i) => mf.pos[i][0]).attr('cy', (i) => mf.pos[i][1]).attr('r', (i) => mf.r[i]).attr('fill', (i) => mf.fill[i])
        .attr('stroke', css('--surface')).attr('stroke-width', 1).attr('opacity', 1)
        .transition().duration(MORPH_MS).ease(inOut).attr('opacity', 0).remove();
    } else {
      this.gOld.selectAll('circle').interrupt().remove();
      dots.attr('opacity', 1).transition(t).attr('r', radius).attr('cx', (i) => this.pos[i][0]).attr('cy', (i) => this.pos[i][1]);
    }
    this.lastOrg = [ox, oy];
    this.lastR = idx.map(radius);
    this.lastFill = idx.map(fillOf);

    if (mf) {
      // Selected set: pair old and new items by rank of their axis-1 position; arrows, shape and dots follow the interpolated points.
      const from = new Map(pairSelected(mf.sel, mf.pos.map((p) => p[0]), S.sel, this.pos.map((p) => p[0])).map(([ni, oi]) => [ni, mf.pos[oi]]));
      const lerp = (a, b, e) => [a[0] + (b[0] - a[0]) * e, a[1] + (b[1] - a[1]) * e];
      const facePx = (f) => 'M' + f.map((q) => q.join(',')).join('L') + (f.length > 2 ? 'Z' : '');
      const frame = (e) => {
        const cur = (i) => (from.has(i) ? lerp(from.get(i), this.pos[i], e) : this.pos[i]);
        const O = lerp(mf.origin, [ox, oy], e);
        this.gArrow.selectAll('line').each(function (i) {
          const p = cur(i);
          d3.select(this).attr('x1', O[0]).attr('y1', dims === 1 ? p[1] : O[1]).attr('x2', p[0]).attr('y2', p[1]);
        });
        if (vol) {
          const P = S.sel.map(cur), shape = parallelotope(O, P);
          this.gPara.selectAll('path').attr('d', (_, k) => (shape.faces[k] ? facePx(shape.faces[k]) : null));
          this.gPara.selectAll('circle').attr('cx', (_, k) => shape.corners[k]?.p[0]).attr('cy', (_, k) => shape.corners[k]?.p[1]);
        }
        else if (hullOn) this.gPara.selectAll('path').attr('d', (_, k) => (k === 0 ? facePx(hull2D(S.sel.map(cur))) : null));
        this.gDot.selectAll('circle').filter((i) => sel.has(i)).attr('cx', (i) => cur(i)[0]).attr('cy', (i) => cur(i)[1]);
      };
      const t0 = performance.now();
      DBG.morphs++; DBG.lastKind = '2d';
      const step = (now) => {
        const u = Math.min(1, (now - t0) / MORPH_MS);
        frame(easeInOut(u));
        if (u >= 1) DBG.lastMorphMs = Math.round(performance.now() - t0);
        this.raf = u < 1 ? requestAnimationFrame(step) : 0;
      };
      frame(0);
      this.raf = requestAnimationFrame(step);
    }
  }
  // Re-fill the points (hover on a legend chip dims other clusters).
  recolor() {
    if (!this.svg || !this.lastFill) return;
    const col = colorFor(), sel = new Set(S.sel);
    const fill = (i) => (byCl() && sel.has(i) ? selColor(i) : col(i));
    this.gDot.selectAll('circle').attr('fill', fill);
    this.lastFill = this.lastFill.map((_, i) => fill(i));
  }
  highlight(i, i2 = -1) {
    if (!this.svg) return;
    matHover(i);
    if (i < 0) { i = S.pinned; i2 = S.pinned2; }
    S.hover = i;
    this.gDot.selectAll('circle').attr('stroke', (j) => (j === i || j === i2 ? css('--text') : S.sel.includes(j) ? selStroke(j) : css('--surface')));
    document.querySelectorAll('.card').forEach((c) => c.classList.toggle('hl', +c.dataset.i === i || +c.dataset.i === i2));
  }
}

const plot = {
  p2: new Plot2D($('#plot')),
  p3: null,
  reset() {
    this.p2.reset();
    this.p3?.dispose();
    this.p3 = null;
  },
  // The view being left, for a morph: null if nothing is drawn; { instant } under reduced motion.
  snapshot() {
    if (REDUCED.matches) return { kind: 'any', instant: true };
    return (S.dims === 3 ? this.p3?.snapshot() : this.p2.snapshot()) ?? null;
  },
  update() {
    const mf = S.morphFrom;
    S.morphFrom = null;
    if (S.dims === 3) {
      if (!this.p3) {
        $('#plot').innerHTML = '';
        this.p3 = new Plot3D($('#plot'), {
          onHover: (i, x, y) => { showTip(i, x, y); matHover(i); this.highlightCards(i < 0 ? S.pinned : i, i < 0 ? S.pinned2 : -1); },
          onTap: (i) => inspect(i),
        });
      }
      this.p3.update({
        coords: viewCoords(), c: S.cDisp, rs: S.rsz, sel: S.sel, scaled: lenChoice() !== 'unit', ghosts: S.expert && $('#ghost-toggle').checked, fixedFrame: S.view.fixed,
        color: colorFor(), selColor, accent: css('--accent'), ghost: css('--ghost'), text: css('--text-2'), bg: css('--surface'),
      }, mf);
    } else this.p2.update(mf);
  },
  dim(c) {
    S.dimCl = c;
    this.p2.recolor();
    this.p3?.recolor(colorFor(), selColor);
  },
  highlight(i, j = -1) {
    if (S.dims === 3) {
      matHover(i);
      if (i < 0) { i = S.pinned; j = S.pinned2; }
      this.p3?.highlight(i, j); this.highlightCards(i, j);
    } else this.p2.highlight(i, j);
  },
  highlightCards(i, j = -1) { document.querySelectorAll('.card').forEach((c) => c.classList.toggle('hl', +c.dataset.i === i || +c.dataset.i === j)); },
};

// ---------------------------------------------------------------- labels
// Small callouts with the opening words of each selected completion. Shown for
// K ≤ 6 so they never swamp the plot; in 3D they follow the points every frame.
const MAX_LABELS = 6;
// The first few words only: enough to tell the selected completions apart.
function labelText(s) { const w = s.text.replace(/\s+/g, ' ').trim().split(' '); return w.length > 5 ? w.slice(0, 5).join(' ') + '…' : w.join(' '); }
function renderLabels() {
  const layer = $('#labels');
  const show = $('#label-toggle').checked && S.sel.length && S.sel.length <= MAX_LABELS;
  if (!show) { layer.innerHTML = ''; return; }
  const want = new Set(S.sel.map(String));
  layer.querySelectorAll('.lbl').forEach((el) => { if (!want.has(el.dataset.i)) el.remove(); });
  for (const i of S.sel) {
    if (layer.querySelector(`.lbl[data-i="${i}"]`)) continue;
    const el = document.createElement('div');
    el.className = 'lbl';
    el.dataset.i = i;
    el.appendChild(document.createElement('span')).textContent = labelText(S.pool.samples[i]);
    el.onclick = () => inspect(i);
    el.onmouseenter = () => plot.highlight(i);
    el.onmouseleave = () => plot.highlight(-1);
    layer.appendChild(el);
  }
  layer.querySelectorAll('.lbl').forEach((el) => { el.style.color = byCl() ? selColor(+el.dataset.i) : ''; });
  layer.classList.toggle('anim', S.dims !== 3);
  placeLabels();
}
function placeLabels() {
  const layer = $('#labels');
  const els = [...layer.querySelectorAll('.lbl')];
  if (!els.length) return;
  const W = layer.clientWidth, H = layer.clientHeight;
  const items = els.map((el) => {
    const i = +el.dataset.i;
    const p = S.dims === 3 ? plot.p3?.screenXY(i) : plot.p2.screenXY(i);
    return p && { el, x: p[0], y: p[1], w: el.offsetWidth, h: el.offsetHeight };
  }).filter(Boolean).sort((a, b) => a.y - b.y);
  // Right of the point (left if it would leave the plot), then nudge down to avoid overlaps.
  const placed = [];
  for (const it of items) {
    let left = it.x + 8 + it.w > W - 4 ? it.x - 8 - it.w : it.x + 8;
    left = Math.max(4, left);
    let top = Math.min(Math.max(4, it.y - it.h / 2), H - it.h - 4);
    for (const q of placed) {
      const overlapX = left < q.left + q.w && q.left < left + it.w;
      if (overlapX && top < q.top + q.h + 3 && q.top < top + it.h) top = q.top + q.h + 3;
    }
    placed.push({ left, top, w: it.w, h: it.h });
    it.el.style.transform = `translate(${Math.round(left)}px, ${Math.round(top)}px)`;
  }
}
(function follow3D() {
  if (S.dims === 3 && $('#labels').childElementCount) placeLabels();
  requestAnimationFrame(follow3D);
})();

// ---------------------------------------------------------------- inspector
// Tap or click a point (or a card) to pin its text and numbers under the plot.
function inspect(i) {
  S.pinned = i; S.pinned2 = -1;
  showTip(-1);
  renderInspect();
  plot.highlight(i);
}
function renderInspect() {
  const el = $('#inspect');
  const i = S.pinned;
  if (i < 0 || !S.pool || i >= S.pool.samples.length) { el.hidden = true; syncPinCross(); return; }
  const s = S.pool.samples[i];
  const w = weightInfo();
  const inSel = S.sel.includes(i);
  el.hidden = false;
  el.innerHTML = `<div class="ins-txt">${escapeHtml(s.text)}${s.ended ? '' : ' …'}</div>
    <div class="ins-meta">${inSel ? `<span class="pill" style="background:${selColor(i)}">selected</span>` : ''}
      <span>log p ${fmt(s.logp, 2)}</span><span>log q ${fmt(s.logq, 2)}</span><span>${w.name} ${fmt(w.f(s), 2)}</span><span>P(in set) ${fmt(S.pi[i], 3)}</span></div>
    <button class="ins-close" aria-label="Close">×</button>`;
  el.querySelector('.ins-close').onclick = () => inspect(-1);
  const j = S.pinned2;
  if (j >= 0 && j < S.pool.samples.length && S.mat && S.mat.N === S.pool.samples.length) {
    const t = S.pool.samples[j], N = S.mat.N, wi = S.mat.w[i], wj = S.mat.w[j];
    el.insertAdjacentHTML('beforeend', `<div class="ins-txt ins-2">${escapeHtml(t.text)}${t.ended ? '' : ' …'}</div>
      <div class="ins-meta"><span>L<sub>ij</sub> ${fmtSig(S.mat.L[i * N + j])}</span><span>cosine ${fmt(S.G[i * N + j], 3)}</span><span>w<sub>i</sub> ${fmtSig(wi)}</span><span>w<sub>j</sub> ${fmtSig(wj)}</span></div>`);
  }
  syncPinCross();
}

// ---------------------------------------------------------------- pool table
let sortKey = 'logp', sortDir = -1;
function renderTable() {
  const w = weightInfo(); // log w of the weight in use
  const sel = new Set(S.sel);
  const rows = S.pool.samples.map((s, i) => ({ i, text: s.text + (s.ended ? '' : '…'), logp: s.logp, logq: s.logq, logw: w.f(s), pi: S.pi[i] }));
  rows.sort((a, b) => sortDir * (typeof a[sortKey] === 'string' ? a[sortKey].localeCompare(b[sortKey]) : a[sortKey] - b[sortKey]));
  const cols = [['i', '#'], ['text', 'string'], ['logp', 'log p'], ['logq', 'log q'], ['logw', w.name], ['pi', 'P(i∈S)']];
  $('#pool-table').innerHTML = `<div class="wrap"><table><thead><tr>${cols.map(([k, n]) => `<th data-k="${k}">${n}${k === sortKey ? (sortDir > 0 ? ' ▲' : ' ▼') : ''}</th>`).join('')}</tr></thead><tbody>${rows.map((r) => `<tr class="${sel.has(r.i) ? 'sel' : ''}"><td class="num" data-k="#">${r.i}</td><td class="str">${escapeHtml(r.text)}</td><td class="num" data-k="log p">${fmt(r.logp, 2)}</td><td class="num" data-k="log q">${fmt(r.logq, 2)}</td><td class="num" data-k="${w.name}">${fmt(r.logw, 2)}</td><td class="num" data-k="P(in set)">${fmt(r.pi, 3)}</td></tr>`).join('')}</tbody></table></div>`;
  $('#pool-table').querySelectorAll('tbody tr').forEach((tr, k) => {
    const i = rows[k].i;
    tr.onmouseenter = () => plot.highlight(i);
    tr.onmouseleave = () => plot.highlight(-1);
  });
  $('#pool-table').querySelectorAll('th').forEach((th) => (th.onclick = () => {
    const k = th.dataset.k;
    if (k === sortKey) sortDir = -sortDir; else { sortKey = k; sortDir = k === 'text' || k === 'i' ? 1 : -1; }
    renderTable();
  }));
}

// ---------------------------------------------------------------- where models run
// Every place that can produce a pool is a backend (js/backends.js, configured in data/backends.json; ?backend=<url> adds one).
// The page asks each for its capabilities and offers a model only where some usable backend has it.
const showCloud = authEnabled() && (!!getAuth() || /[?&](cloud|code)=?/.test(location.search));
let BACKENDS = [];
const backendById = (id) => BACKENDS.find((b) => b.id === id);
const isCloud = (b) => b.caps.auth === 'hf' || b.kind === 'space';
// A backend can be used now: the browser one when "Run in this browser" is on, the others when it is off, signed-in ones only with ?cloud.
const usableBackend = (b) => (b.kind === 'browser' ? browserOn() : !browserOn() && (!isCloud(b) || showCloud));
const whereName = (b) => (!b ? '' : b.kind === 'browser' ? 'in this browser' : b.temporary ? `server at ${b.origin}` : isCloud(b) ? 'cloud GPU' : 'server GPU');
const serverBackends = () => BACKENDS.filter((b) => b.kind !== 'browser' && usableBackend(b));
const SERVER = { get ok() { return BACKENDS.some((b) => b.kind !== 'browser' && !isCloud(b) && b.caps.up); } };
// The default model of the first reachable server (used when the selected model cannot run).
const serverDefault = () => serverBackends().find((b) => b.caps.up && b.caps.def)?.caps.def ?? null;
async function refreshServer() {
  const list = BACKENDS.filter((b) => b.kind !== 'browser' && (!isCloud(b) || showCloud));
  await Promise.all(list.map((b) => b.status()));
  let added = false;
  for (const b of list) for (const m of b.caps.models) if (!CATALOG.models.some((c) => c.key === m.key)) { CATALOG.models.push({ key: m.key, name: m.name ?? m.key, params: m.params, repo: m.repo }); added = true; }
  if (added) { sortCatalog(); buildPickers(); }
  updatePickers();
  return SERVER.ok;
}
async function initBackends() {
  const cfgs = await loadBackendConfigs(fetch, location.search);
  BACKENDS = createBackends(cfgs, {
    getToken: () => getAuth()?.token,
    makeWorker: () => new Worker(new URL('./worker.js', import.meta.url), { type: 'module' }),
    browserModels: () => MODELS.filter((m) => !browserWhy(m.key)).map((m) => ({ key: m.key, name: m.name })),
  });
}
function terminateWorker() { BACKENDS.find((b) => b.kind === 'browser')?.terminate?.(); }

// Browser side: which models this device can run (WebGPU float16 or a small enough WebAssembly build).
const DEVICE = { hasGPU: false, cached: new Set(), isPhone: navigator.userAgentData?.mobile ?? /Android|iPhone|iPad|Mobile/i.test(navigator.userAgent) };
const sizeMB = (m) => (DEVICE.hasGPU ? m.gpu : m.cpuMB);
const fits = (m) => DEVICE.hasGPU || (m.cpu && m.cpuMB <= (DEVICE.isPhone ? 1000 : 2200));
// '' if the browser can run the model, else why not.
function browserWhy(key) {
  const m = MODELS.find((x) => x.key === key);
  if (!m) return 'Not available in the browser: this model runs on the demo server only.';
  if (fits(m)) return '';
  return !m.cpu ? 'Needs a GPU with float16 support, which this browser does not offer.' : `Too large for this ${DEVICE.isPhone ? 'phone' : 'device'}.`;
}
// Where a custom prompt runs on this model right now: the id of a backend, or null.
function pickWhere(key) {
  if (!allowedKeys().includes(key)) return null;
  if (S.forceCloud) { const c = BACKENDS.find((b) => isCloud(b) && b.caps.models.some((m) => m.key === key)); if (c) return c.id; }
  return chooseBackend(BACKENDS, { model: key, embedder: S.embedder }, usableBackend)?.backend.id ?? null;
}

// ---------------------------------------------------------------- model and embedder pickers
const WHERE_NAME = new Proxy({}, { get: (_, id) => whereName(backendById(id)) });
function chipState(m) {
  if (S.source === 'example') {
    return CATALOG.examples[S.exId]?.[m.key]?.length ? { disabled: false, why: 'precomputed' } : { disabled: true, why: 'no precomputed pool of this model for this example' };
  }
  if (browserOn()) {
    const why = browserWhy(m.key);
    if (why) return { disabled: true, why };
    const mm = MODELS.find((x) => x.key === m.key);
    return { disabled: false, why: `in this browser, ${DEVICE.cached.has(mm.repo) ? 'downloaded' : `${(sizeMB(mm) / 1000).toFixed(1)} GB download`}` };
  }
  const hit = chooseBackend(BACKENDS, { model: m.key, embedder: S.embedder }, usableBackend);
  if (hit) {
    const sm = hit.backend.caps.models.find((x) => x.key === m.key);
    if (sm.state === 'unavailable') return { disabled: true, why: 'the demo server cannot load this model right now' };
    if (sm.state && sm.state !== 'ready') return { disabled: false, cold: true, why: `${whereName(hit.backend)}, loads on first use (about 20 s)` };
    return { disabled: false, why: whereName(hit.backend) };
  }
  return serverBackends().some((b) => b.caps.up)
    ? { disabled: true, why: 'not on the demo server. Turn on Run in this browser to try a small model.' }
    : { disabled: true, why: 'needs the demo server, which is not reachable here. Turn on Run in this browser in the settings.' };
}

// The pills in the prompt box: language model and embedder. Expert mode lists all models, smallest first.
function buildPickers() {
  const sel = $('#lm-select');
  sel.innerHTML = visibleModels(CATALOG.models, S.expert).map((m) => `<option value="${escapeHtml(m.key)}">${escapeHtml(m.name)}</option>`).join('');
  sel.onchange = () => { if (sel.options[sel.selectedIndex]?.disabled) { sel.value = S.model; return; } pickModel(sel.value); };
  // The select must always show S.model: if it is not among the options (it cannot be, after the guard), show it rather than the first option.
  if (!sel.querySelector(`option[value="${S.model}"]`)) sel.insertAdjacentHTML('afterbegin', `<option value="${escapeHtml(S.model)}" disabled>${escapeHtml(info(S.model).name)}</option>`);
  syncPickers();
}
function syncPickers() {
  const sel = $('#lm-select');
  [...sel.options].forEach((o) => {
    const st = chipState(info(o.value));
    o.disabled = st.disabled; o.title = st.why;
    o.textContent = `${info(o.value).name}${st.disabled ? ' (not available here)' : ''}`;
  });
  if ([...sel.options].some((o) => o.value === S.model)) sel.value = S.model;
  syncEmbedder();
}

// Run settings are bounded by the backend that would run them (/status maxN and maxNewTokens).
function liveLimits() {
  const caps = backendById(pickWhere(S.model))?.caps;
  return { maxN: caps?.maxN ?? 200, maxTokens: caps?.maxTokens ?? 256 };
}
function syncLimits() {
  const L = liveLimits(), n = $('#live-N'), t = $('#live-max');
  n.max = String(L.maxN); t.max = String(L.maxTokens);
  if (+n.value > L.maxN) n.value = L.maxN;
  if (+t.value > L.maxTokens) t.value = L.maxTokens;
}
function updatePickers() {
  syncPickers();
  syncLimits();
  renderTp();
}

// T′ switch for the examples.
// The pool's local sampling temperature T′ (examples: the chosen pool; own prompts: the run's setting).
const poolTp = () => Number(S.source === 'example' ? S.tp : S.cust?.Tp ?? S.pool?.Tprime);

// Paper-grid chips under the T slider and the pressed state of the paper-settings buttons.
function renderControls() {
  const box = $('#T-ticks'), grid = PRESETS[S.mode === 'map' ? 'map' : 'samp'].Ts;
  if (box.dataset.mode !== S.mode) {
    box.dataset.mode = S.mode;
    box.innerHTML = '<span>Paper T</span>' + grid.map((t) => `<button type="button" class="tick" data-t="${t}" aria-pressed="false">${t}</button>`).join('');
    box.querySelectorAll('.tick').forEach((b) => (b.onclick = () => setT(+b.dataset.t)));
  }
  box.querySelectorAll('.tick').forEach((b) => b.setAttribute('aria-pressed', String(Math.abs(+b.dataset.t - S.T[S.mode]) <= 1e-6 * +b.dataset.t)));
  const m = matchPreset({ mode: S.mode, Tprime: poolTp(), K: S.K, T: S.T[S.mode], weight: famNow() });
  $('#adv-mat').addEventListener('toggle', renderMatrices);
  document.querySelectorAll('input[name=mord], input[name=dmode], input[name=dr], #mat-norm').forEach((r) => r.addEventListener('change', renderMatrices));
  document.querySelectorAll('#preset-seg button').forEach((b) => b.setAttribute('aria-pressed', String(b.dataset.p === m)));
  const note = $('#preset-note');
  note.hidden = !m;
  if (m) {
    const L = lenChoice(), paperFam = PRESETS[m].weight;
    note.textContent = `Paper setting: ${m === 'map' ? 'PoolDPP-MAP' : 'PoolDPP-Samp'}, T′ ${poolTp().toFixed(1)}${!lockOn() && L !== paperFam ? ` · vectors shown: ${LEN_WORD[L]}` : ''}`;
  }
  syncEmbedder();
  syncChanged();
}

// Defaults of the settings in the two disclosures, in one place.
const DEFAULTS = { lock: true, axes: 'fixed', color: true, label: true, cone: false, link: true, ghost: false, clm: 'tree', clk: 6, cls: 'view' };
function syncChanged() {
  const defEmb = CATALOG.embedders.find((e) => e.default)?.key ?? DEF_EMB;
  const axes = document.querySelector('input[name=axes]:checked')?.value ?? DEFAULTS.axes;
  const pick = weightChoice() !== paperWeight(S.mode) || lockOn() !== DEFAULTS.lock || $('#T-link').checked !== DEFAULTS.link || S.embedder !== defEmb;
  const view = axes !== DEFAULTS.axes || $('#color-toggle').checked !== DEFAULTS.color || $('#label-toggle').checked !== DEFAULTS.label || $('#cone-toggle').checked !== DEFAULTS.cone || $('#ghost-toggle').checked !== DEFAULTS.ghost
    || clOpts().method !== DEFAULTS.clm || clOpts().k !== DEFAULTS.clk || clOpts().space !== DEFAULTS.cls;
  $('#chg-pick').hidden = !pick;
  $('#chg-view').hidden = !view;
}

// ---- embedder: which model's embeddings define the similarity kernel
// Why an embedder cannot be used now ('' if it can).
function embWhy(key) {
  if (S.source === 'custom' && S.pool) {
    if (key === S.embedder) return '';
    const b = backendById(S.where);
    return b?.embed && b.caps.embed && embeddersFor(b.caps, S.model).includes(key) ? '' : `${b ? whereName(b).replace(/^./, (c) => c.toUpperCase()) : 'The backend'} cannot embed an existing pool, so the embedder stays ${embInfo(S.embedder).name} for these completions.`;
  }
  if (key === DEF_EMB) return '';
  if (S.source === 'example') return CATALOG.examples[S.exId] ? '' : 'This example has no precomputed pool for other embedders.';
  return '';
}
function syncEmbedder() {
  const list = CATALOG.embedders, row = $('#emb-row'), sel = $('#emb-select');
  row.hidden = list.length < 2;
  if (row.hidden) return;
  const sig = list.map((e) => e.key).join();
  if (sel.dataset.k !== sig) {
    sel.dataset.k = sig;
    sel.innerHTML = list.map((e) => `<option value="${escapeHtml(e.key)}">${escapeHtml(e.name)}</option>`).join('');
    sel.onchange = () => { setEmbedder(sel.value); sel.value = S.embedder; };
  }
  [...sel.options].forEach((o) => { const why = embWhy(o.value); o.disabled = !!why; o.title = why || embInfo(o.value).note || ''; });
  sel.value = S.embedder;
  sel.title = embInfo(S.embedder).note || '';
}
function setEmbedder(key) {
  const why = embWhy(key);
  if (why) { $('#live-status').textContent = why; syncEmbedder(); return; }
  if (key === S.embedder) return;
  const prev = S.embedder;
  S.embedder = key;
  syncEmbedder();
  if (S.source === 'example' && S.exId) showExample(S.exId, { key: S.model, tp: S.tp, morph: true });
  else if (S.source === 'custom' && S.pool) embedCustom(key, prev);
}
// A live pool under another embedder: the backend embeds the existing completions (POST {url}/embed), nothing is resampled.
async function embedCustom(key, prev) {
  const b = backendById(S.where), samples = S.pool.samples, seq = ++S.runSeq, ctl = new AbortController();
  run = { seq, abort: ctl, kind: 'embed' };
  S.pending = true; setBusy(true);
  $('#live-status').textContent = `Embedding ${samples.length} completions with ${embInfo(key).name}…`;
  try {
    const r = await b.embed({ texts: samples.map((x) => x.text), embedder: key, model: S.model, prompt: S.pool.prompt ?? S.cust?.prompt, ended: samples.map((x) => !!x.ended) }, { signal: ctl.signal });
    if (seq !== S.runSeq) return;
    const N = samples.length;
    if (r.emb.length !== N * r.D) throw new Error('the backend returned a different number of embeddings');
    run = null; S.pending = false; setBusy(false);
    const rec = { ...S.pool, D: r.D, embedder: key };
    const e = makeEntry(rec, unitRows(r.emb, N, r.D));
    setPool(rec, e.phi, { entry: e, keepPlot: false });
    $('#live-status').textContent = `${N} completions embedded with ${embInfo(key).name}.${r.approximate && r.note ? ` ${r.note}` : r.approximate ? ' This embedding is approximate.' : ''}`;
  } catch (err) {
    if (err.name === 'AbortError' || seq !== S.runSeq) { if (seq === S.runSeq || err.name === 'AbortError') { S.embedder = prev ?? DEF_EMB; syncEmbedder(); } return; }
    run = null; S.pending = false; setBusy(false);
    S.embedder = prev ?? DEF_EMB; syncEmbedder();
    $('#live-status').textContent = `Could not embed: ${err.message || err}`;
  }
}
// ---- animated changes (paper presets, Paper T chips)
// The real state changes at once (so the plot morphs once, to the final state). The controls then show it: the T slider glides,
// K counts, and each changed control pulses in reading order. A second click cancels what is running.
const anim = { timers: [], raf: 0, tweenT: false, pulsed: new Set() };
const PULSE_ROW = { mode: '.mode-row', T: '.tfield', K: '.kfield', tp: '#tp-wrap', weight: '#wt-row', len: '#len-row', lock: '.lockline' };
function cancelAnim() {
  anim.timers.forEach(clearTimeout); anim.timers = [];
  cancelAnimationFrame(anim.raf); anim.raf = 0;
  anim.tweenT = false;
  anim.pulsed.forEach((el) => el.classList.remove('pulse', 'pulse-static'));
  anim.pulsed.clear();
}
function pulse(el) {
  el.classList.remove('pulse', 'pulse-static');
  void el.offsetWidth; // restart the animation if it is already running
  const cls = REDUCED.matches ? 'pulse-static' : 'pulse';
  el.classList.add(cls);
  anim.pulsed.add(el);
  const off = () => { el.classList.remove('pulse', 'pulse-static'); anim.pulsed.delete(el); };
  if (REDUCED.matches) anim.timers.push(setTimeout(off, 1200));
  else el.addEventListener('animationend', off, { once: true });
}
// What the controls show now (mid-tween values included).
const shownState = () => ({
  mode: S.mode,
  T: anim.tweenT ? sliderToT(+$('#T-slider').value) : S.T[S.mode],
  K: +$('#K').value || S.K,
  tp: poolTp(),
  weight: weightChoice(),
  len: lenChoice(),
  lock: lockOn(),
});
// Change to `after`: apply() sets the real state and starts the recompute; then the controls animate from what they showed.
function animateChange(after, apply) {
  const before = shownState();
  cancelAnim();
  const diff = presetDiff(before, after);
  if (diff.some((k) => ['tp', 'weight', 'len', 'lock'].includes(k))) $('#adv-pick').open = true;
  const tween = diff.includes('T') && !REDUCED.matches;
  const fromPos = +$('#T-slider').value;
  anim.tweenT = tween;
  apply();
  $('#T-out').textContent = fmtT(tween ? sliderToT(fromPos) : after.T);
  if (!tween) $('#T-slider').value = tToSlider(after.T);
  const step = REDUCED.matches ? 0 : 120;
  diff.forEach((k, i) => anim.timers.push(setTimeout(() => pulse($(PULSE_ROW[k])), i * step)));
  if (tween) {
    const toPos = tToSlider(after.T), t0 = performance.now(), dur = 500;
    const frame = (now) => {
      const u = Math.min(1, (now - t0) / dur), e = easeInOut(u), pos = fromPos + (toPos - fromPos) * e;
      if (u < 1) {
        $('#T-slider').value = pos;
        $('#T-out').textContent = fmtT(sliderToT(pos));
        anim.raf = requestAnimationFrame(frame);
      } else {
        anim.raf = 0; anim.tweenT = false;
        $('#T-slider').value = toPos; // exactly the final value
        $('#T-out').textContent = fmtT(after.T);
      }
    };
    anim.raf = requestAnimationFrame(frame);
  }
  if (diff.includes('K') && !REDUCED.matches) {
    const K0 = before.K, dir = Math.sign(after.K - K0);
    let k = K0;
    $('#K').value = k;
    const tick = () => { k += dir; $('#K').value = k; if (k !== after.K) anim.timers.push(setTimeout(tick, 80)); };
    anim.timers.push(setTimeout(tick, 80));
  }
}

function setT(T) {
  animateChange({ ...shownState(), T }, () => {
    S.T[S.mode] = T;
    if ($('#T-link').checked) S.T.sample = S.T.map = T;
    recompute({ newDraw: true });
  });
}
// Picking mode, K and T at once; T′ through the example's pool switch (or left as is for an own prompt, whose preset then stays unpressed).
function applyPreset(key) {
  const p = PRESETS[key];
  const tp = p.Tprime.toFixed(1), switchTp = S.source === 'example' && S.tp !== tp && exampleTps(S.exId).includes(tp);
  const K = S.pool ? clampInt(p.K, 1, Math.min(20, S.logp.length - 1)) : p.K;
  S.manual = false;
  animateChange({ mode: p.mode, T: p.T, K, tp: switchTp ? p.Tprime : poolTp(), weight: p.weight, len: p.weight, lock: p.lock }, () => {
    S.mode = p.mode;
    document.querySelectorAll('input[name=mode]').forEach((r) => (r.checked = r.value === p.mode));
    $('#resample').disabled = p.mode === 'map';
    document.querySelectorAll('input[name=wt]').forEach((r) => (r.checked = r.value === p.weight));
    $('#len-lock').setAttribute('aria-pressed', String(p.lock)); // recompute() reads the weight and the lock
    if (S.pool) { S.Kwant = K; S.K = effectiveK(K, S.distinct, S.logp.length); }
    S.T[p.mode] = p.T;
    if ($('#T-link').checked) S.T.sample = S.T.map = p.T;
    if (switchTp) showExample(S.exId, { key: S.model, tp, morph: true });
    else if (S.pool) recompute({ newDraw: true });
  });
}

function renderTp() {
  const tps = S.source === 'example' && S.exId ? exampleTps(S.exId) : [];
  const seg = $('#tp-seg'), txt = $('#tp-text'), multi = tps.length >= 2;
  seg.hidden = !multi; txt.hidden = multi;
  if (!multi) { const tp = poolTp(); txt.textContent = Number.isFinite(tp) ? tp.toFixed(1) : '—'; renderControls(); return; }
  if (seg.dataset.tps !== tps.join()) {
    seg.dataset.tps = tps.join();
    seg.innerHTML = tps.map((tp) => `<label><input type="radio" name="tp" value="${tp}"><span>${tp}</span></label>`).join('');
    seg.querySelectorAll('input').forEach((r) => (r.onchange = () => { showExample(S.exId, { key: S.model, tp: r.value, morph: true }); }));
  }
  seg.querySelectorAll('input').forEach((r) => (r.checked = r.value === S.tp));
  renderControls();
}

function pickModel(key, { auto = false } = {}) {
  key = guardModel(key, allowedKeys());
  if (key === S.model) return;
  setModel(key);
  S.forceCloud = false;
  try { localStorage.setItem('dpp-lm-model', key); } catch {}
  updatePickers();
  if (S.source === 'example') {
    const ex = CATALOG.examples[S.exId]?.[key] ?? [];
    showExample(S.exId, { key, tp: ex.includes(S.tp) ? S.tp : ex[0] ?? S.tp, morph: true });
  } else if (S.cust) {
    $('#live-status').textContent = `Running your prompt on ${info(key).name}…`;
    startRun(key);
  }
}

// ---------------------------------------------------------------- live mode
// A finished run of the user's own prompt is kept in memory by (prompt, model, where, N, T′, max tokens), so switching back is instant.
const CUSTOM = new Map();
const custKey = (key, where) => { const c = S.cust; return c ? [c.prompt, key, where, c.N, c.Tp, c.max, S.embedder].join('\u0000') : ''; };
function entryFromBody(body) {
  return makeEntry(body, decodeEmb(body.emb, body.N, body.D));
}
function remember(ck, e) {
  CUSTOM.set(ck, e);
  if (CUSTOM.size > 24) CUSTOM.delete(CUSTOM.keys().next().value);
}

let run = null; // { seq, abort, kind } of the run in flight
// While a request of this client is in flight (a run, an embed, "Add more samples") the pickers and the send button are off: the
// server allows one request per client, and a second one would only fail. The stop button cancels the request.
const setBusy = (b) => {
  $('#live-go').hidden = b; $('#live-stop').hidden = !b;
  $('#lm-select').disabled = b; $('#emb-select').disabled = b;
  updateMore();
};
function cancelRun() {
  if (!run) return;
  run.abort?.abort();
  if (run.kind === 'browser') terminateWorker();
  run = null;
  S.pending = false;
  setBusy(false);
  $('#progress').hidden = true;
}

function showCustom(e, key, where, text) {
  S.source = 'custom'; setModel(key); S.where = where; S.pending = false; S.exId = null;
  S.shownModel = key; S.shownWhere = where;
  markExample(null);
  $('#progress').hidden = true;
  $('#live-status').textContent = text;
  updatePickers();
  setPool(e.rec, e.phi, { entry: e });
}

// Run the stored prompt on `key`: from the memory cache, else on the server, in this browser, or in the cloud.
function startRun(key, { force = false } = {}) {
  const c = S.cust;
  if (!c) return;
  const where = pickWhere(key);
  if (!where) {
    $('#live-status').textContent = `${info(key).name} cannot run here: ${chipState(info(key)).why}`;
    return;
  }
  const ck = custKey(key, where);
  if (force) CUSTOM.delete(ck);
  cancelRun();
  S.loadSeq++;
  const hit = CUSTOM.get(ck);
  if (hit) { showCustom(hit, key, where, `${hit.N} completions from ${info(key).name} (${WHERE_NAME[where]}), kept from your earlier run.`); return; }
  const seq = ++S.runSeq;
  setModel(key); S.where = where; S.pending = true;
  updatePickers();
  const done = (e, text) => {
    remember(ck, e);
    if (seq !== S.runSeq) return;
    run = null; setBusy(false);
    showCustom(e, key, where, text);
    window.__refreshCache?.();
  };
  const fail = (msg) => { if (seq !== S.runSeq) return; run = null; S.pending = false; setBusy(false); $('#progress').hidden = true; $('#live-status').textContent = msg; updatePickers(); };
  return runBackend(backendById(where), key, c, seq, done, fail);
}

// One run on one backend: the same request for every kind (see docs/BACKEND.md). `extra` is what to append to the status line.
function entryFromResult({ pool, emb }) {
  const N = pool.samples.length, D = pool.D;
  return makeEntry(pool, Array.from({ length: N }, (_, i) => emb.subarray(i * D, (i + 1) * D)));
}
function requestFor(c, key) {
  const hit = chooseBackend(BACKENDS, { model: key, embedder: S.embedder }, usableBackend);
  return { prompt: c.prompt, n: c.N, t_prime: c.Tp, max_new_tokens: c.max, seed: Math.floor(Math.random() * 2 ** 31), model: key, embedder: hit?.embedder ?? S.embedder };
}
const embFallbackNote = (want, got) => (want && got && want !== got ? ` ${embInfo(got).name} embeddings were used because the backend has no ${embInfo(want).name}.` : '');
async function runBackend(b, key, c, seq, done, fail) {
  const ctl = new AbortController();
  run = { seq, abort: ctl, kind: b.kind };
  setBusy(true);
  const name = info(key).name, t0 = performance.now();
  const secs = () => ((performance.now() - t0) / 1000).toFixed(0);
  const server = b.kind === 'http';
  $('#progress').hidden = b.kind !== 'browser';
  if (b.kind === 'browser') { navigator.storage?.persist?.().catch(() => {}); $('#progress-fill').style.width = '0%'; $('#progress-last').textContent = ''; }
  if (isCloud(b) && !getAuth()) { run = null; setBusy(false); S.pending = false; return signIn({ prompt: c.prompt, model: key }); }
  const say = () => {
    const st = b.caps.models.find((m) => m.key === key)?.state;
    $('#live-status').textContent = isCloud(b) ? `Waiting for ${name}… ${secs()}s`
      : st && st !== 'ready' ? `Loading ${name} on the server… about 20 s (${secs()} s)` : `Sampling from ${name} on ${whereName(b)}… ${secs()} s`;
  };
  let tick = 0, n = 0;
  if (b.kind !== 'browser') { say(); tick = setInterval(() => { n++; if (server && n % 2 === 0) b.status().then(say); else say(); }, 1000); if (server) b.status().then(say); }
  const onProgress = (data) => {
    if (seq !== S.runSeq) return;
    $('#live-status').textContent = data.text ?? $('#live-status').textContent;
    const dl = data.type === 'status' && /(\d+) \/ (\d+) MB/.exec(data.text);
    if (dl) { $('#progress').hidden = false; $('#progress-fill').style.width = `${(100 * +dl[1]) / Math.max(1, +dl[2])}%`; $('#progress-last').textContent = ''; }
    if (data.type === 'status' && /^Sampling 0\//.test(data.text)) $('#progress-fill').style.width = '0%';
    if (data.type === 'progress') {
      $('#progress').hidden = false;
      $('#progress-fill').style.width = `${(100 * data.n) / data.N}%`;
      if (data.last?.length) $('#progress-last').textContent = '“' + data.last[data.last.length - 1] + '”';
    }
  };
  try {
    const req = requestFor(c, key);
    const r = await b.sample(req, { signal: ctl.signal, onProgress });
    if (server) b.status();
    const e = entryFromResult(r), got = embKeyOf(r.pool.embedder) ?? req.embedder, want = S.embedder;
    if (got && got !== S.embedder) { S.embedder = got; syncEmbedder(); }
    done(e, `${e.N} completions from ${name} on ${whereName(b)}, in ${((performance.now() - t0) / 1000).toFixed(1)}s.${embFallbackNote(want, got)}`);
  } catch (e) {
    if (e.name === 'AbortError') return;
    if (e.code === 'unauthorized' && isCloud(b)) signOut();
    fail('Error: ' + (e.message || e));
  } finally { clearInterval(tick); }
}

// Generate: always a fresh run of the typed prompt on the selected model (or the nearest one that can run here).
async function runLive() {
  const prompt = $('#prompt').value.trim();
  if (!prompt) return;
  // Before the status probe has answered, wait for it (nothing is run on a guess, and nothing falls back to the browser).
  if (!S.serverChecked) { $('#live-status').textContent = 'Checking the server…'; await window.__ready; }
  const plan = planRun({
    model: S.model, allowed: allowedKeys(),
    backends: BACKENDS.map((b) => ({ id: b.id, kind: b.kind, usable: usableBackend(b), up: b.caps.up, models: b.caps.models.map((m) => m.key) })),
  });
  if (plan.refuse) { $('#live-status').textContent = plan.refuse; return; }
  const L = liveLimits();
  S.cust = { prompt, N: clampInt($('#live-N').value, 2, L.maxN), Tp: Math.min(2, Math.max(0.2, +$('#live-Tp').value || 1)), max: clampInt($('#live-max').value, 8, L.maxTokens) };
  if (plan.switched) { setModel(plan.model); updatePickers(); }
  startRun(plan.model, { force: true });
}
const clampInt = (v, lo, hi) => Math.min(hi, Math.max(lo, Math.round(+v || lo)));

// ---------------------------------------------------------------- utils & wiring
function escapeHtml(s) {
  return String(s).replace(/[&<>"']/g, (c) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' })[c]);
}
function setMode(m) {
  // Most likely and Sample use the same weight, vector length, T, T' and axes: only the selected set changes.
  if ($('#T-link').checked) S.T[m] = S.T[S.mode];
  S.mode = m;
  S.manual = false;
  $('#resample').disabled = m === 'map';
  $('#T-slider').value = tToSlider(S.T[m]);
  recompute({ newDraw: true });
}

function setK(v) {
  S.manual = false;
  S.Kwant = clampInt(v, 1, Math.min(20, S.logp.length - 1));
  S.K = effectiveK(S.Kwant, S.distinct, S.logp.length);
  $('#K').value = S.K;
  recompute({ newDraw: true });
}

let catalogReady = null;

// Expert mode: off by default, remembered, ?expert=1 turns it on.
function readExpert() {
  try {
    const q = new URLSearchParams(location.search).get('expert');
    if (q !== null) return q === '1' || q === 'true';
    return localStorage.getItem('dpp-lm-expert') === '1';
  } catch { return false; }
}
// Hidden settings go back to their defaults when expert mode is turned off, so the simple view is always the default view.
function resetExpertSettings() {
  const set = (name, v) => document.querySelectorAll(`input[name=${name}]`).forEach((r) => (r.checked = r.value === v));
  set('axes', DEFAULTS.axes); set('clm', DEFAULTS.clm); set('cls', DEFAULTS.cls); set('cln', 'num'); set('mord', 'blocks'); set('dmode', 'axes'); set('dr', '32');
  set('wt', paperWeight(S.mode)); set('len', paperWeight(S.mode));
  $('#cl-k').value = DEFAULTS.clk;
  for (const [id, v] of [['cone-toggle', DEFAULTS.cone], ['ghost-toggle', DEFAULTS.ghost], ['label-toggle', DEFAULTS.label], ['color-toggle', DEFAULTS.color], ['size-toggle', false], ['mat-norm', true], ['T-link', DEFAULTS.link]]) $('#' + id).checked = v;
  $('#len-lock').setAttribute('aria-pressed', String(DEFAULTS.lock));
  syncCl();
}
// In expert mode the advanced sections open by themselves. One the user closes by hand stays closed for the session.
const AUTO_OPEN = ['#adv-pick', '#adv-view', '#adv-cl', '#adv-mat'];
function openAuto(sel) {
  const d = $(sel);
  if (d.dataset.userClosed === '1' || d.open) return;
  d.dataset.auto = '1';
  d.open = true;
}
function applyExpert(on, { save = true } = {}) {
  const was = S.expert;
  S.expert = on;
  if (on && !was) AUTO_OPEN.forEach(openAuto);
  document.body.classList.toggle('expert', on);
  $('#expert-toggle').checked = on;
  if (save) try { localStorage.setItem('dpp-lm-expert', on ? '1' : '0'); } catch {}
  if (was && !on) resetExpertSettings();
  const g = guardModel(S.model, allowedKeys());
  if (g !== S.model) pickModel(g, { auto: true });
  buildPickers();
  updateBanner();
  if (S.pool && was !== on) { S.cl = clustersFor(); recompute({ newDraw: false }); }
}
function updateBanner() {
  const el = $('.wip');
  const vis = visibleModels(CATALOG.models, S.expert), served = vis.filter((m) => serverBackends().some((b) => b.caps.up && b.caps.models.some((x) => x.key === m.key)));
  const params = vis.map((m) => m.params).filter(Number.isFinite), sz = (x) => `${+x.toFixed(1)}B`;
  let where;
  if (!served.length) where = 'Models run in your browser.';
  else if (S.expert && vis.length > 3) where = `This copy offers ${vis.length} models${params.length ? ` from ${sz(Math.min(...params))} to ${sz(Math.max(...params))}` : ''}${served.length < vis.length ? `, ${served.length} on a GPU server` : ' on a GPU server'}.`;
  else { const names = served.map((m) => m.name); where = `This copy offers ${names.slice(0, -1).join(', ')}${names.length > 1 ? ' and ' : ''}${names[names.length - 1]} on a GPU server.`; }
  el.innerHTML = `<b>Work in progress.</b> This demo is still being built. ${escapeHtml(where)}${S.expert ? '' : ' Expert mode adds smaller models and every setting.'}`;
}
function wire() {
  initExplainers();
  // The settings chip summarizes the sampling settings used for your own prompts.
  const advSum = () => {
    const n = $('#live-N').value.trim(), tp = $('#live-Tp').value.trim();
    $('#adv-sum').textContent = 'Your runs: ' + [n && `${n} samples`, tp && `T′ ${tp}`].filter(Boolean).join(' · ');
  };
  ['#live-N', '#live-Tp'].forEach((id) => $(id).addEventListener('input', advSum));
  advSum();

  // Run in this browser: remembered; defaults to on when no server answers.
  let saved = null;
  try { saved = localStorage.getItem('dpp-lm-browser'); } catch {}
  S.browser = saved === '1'; // only used in expert mode (browserOn)
  $('#browser-toggle').checked = S.browser;
  $('#browser-toggle').onchange = (e) => {
    S.browser = e.target.checked;
    try { localStorage.setItem('dpp-lm-browser', S.browser ? '1' : '0'); } catch {}
    updatePickers();
      };
  let savedModel = null;
  try { savedModel = localStorage.getItem('dpp-lm-model'); } catch {}
  const restoreModel = () => { if (savedModel) setModel(savedModel); };
  const hfRow = () => {
    const el = $('#hf-row');
    el.hidden = !showCloud;
    if (!showCloud) return;
    const a = getAuth();
    el.innerHTML = a ? `Cloud GPU: signed in to Hugging Face as <b>${escapeHtml(a.user)}</b>. <button type="button" class="link" id="hf-signout">Sign out</button>` : 'Cloud GPU: pressing ↑ asks you to sign in with Hugging Face.';
    $('#hf-signout')?.addEventListener('click', (e) => { e.stopPropagation(); signOut(); hfRow(); });
  };
  hfRow();

  // Which models are already in the browser's cache (transformers.js stores files under their URL).
  const refreshCache = async () => {
    try {
      const keys = await (await caches.open('transformers-cache')).keys();
      const file = (m) => `https://huggingface.co/${m.repo}/resolve/main/onnx/model_${DEVICE.hasGPU ? 'q4f16' : m.cpu}.onnx`;
      const urls = new Set(keys.map((r) => r.url));
      DEVICE.cached = new Set(MODELS.filter((m) => fits(m) && urls.has(file(m))).map((m) => m.repo));
      const est = await navigator.storage?.estimate?.();
      const gb = est?.usage ? est.usage / 1e9 : 0;
      $('#cache-info').textContent = DEVICE.cached.size ? `${DEVICE.cached.size} model${DEVICE.cached.size > 1 ? 's' : ''} downloaded (${gb.toFixed(1)} GB).` : '';
      $('#cache-clear').hidden = !DEVICE.cached.size;
    } catch { DEVICE.cached = new Set(); }
    updatePickers();
  };
  // Two-step clear: the first click asks, the second deletes.
  let armed = false;
  $('#cache-clear').onclick = async (e) => {
    e.stopPropagation();
    if (!armed) { armed = true; $('#cache-clear').textContent = 'Click again to delete them'; setTimeout(() => { armed = false; $('#cache-clear').textContent = 'Clear'; }, 4000); return; }
    armed = false; $('#cache-clear').textContent = 'Clear';
    try { await caches.delete('transformers-cache'); } catch {}
    terminateWorker(); // drop models held in memory too
    await refreshCache();
    $('#cache-info').textContent = 'Downloaded models deleted.';
  };
  window.__refreshCache = refreshCache;
  navigator.gpu?.requestAdapter().then((a) => { DEVICE.hasGPU = !!a?.features.has('shader-f16'); refreshCache(); }).catch(() => {});
  refreshCache();

  // Catalog and server probe; the gallery waits only for the catalog.
  for (const sel of AUTO_OPEN) $(sel).addEventListener('toggle', () => { const d = $(sel); if (d.dataset.auto) { delete d.dataset.auto; return; } d.dataset.userClosed = d.open ? '' : '1'; });
  applyExpert(readExpert(), { save: false });
  $('#expert-toggle').onchange = (e) => applyExpert(e.target.checked);
  catalogReady = Promise.all([loadCatalog(), initBackends()]).then(() => { restoreModel(); buildPickers(); });
  const serverProbe = catalogReady.then(() => refreshServer()).then((ok) => {
    S.serverChecked = true;
    updateBanner();
    updatePickers();
    return ok;
  });
  window.__ready = Promise.all([catalogReady, serverProbe]);

  // A small "changed" mark on a closed disclosure whose settings differ from the defaults.
  for (const d of ['#adv-pick', '#adv-view']) $(d).addEventListener('change', syncChanged);
  document.querySelectorAll('#preset-seg button').forEach((b) => (b.onclick = () => applyPreset(b.dataset.p)));

  $('#resample').disabled = S.mode === 'map';
  $('#T-slider').value = tToSlider(S.T[S.mode]);
  let raf = 0;
  $('#T-slider').addEventListener('input', (e) => {
    cancelAnim();
    S.T[S.mode] = sliderToT(+e.target.value);
    if ($('#T-link').checked) S.T.sample = S.T.map = S.T[S.mode];
    $('#T-out').textContent = fmtT(S.T[S.mode]);
    cancelAnimationFrame(raf);
    raf = requestAnimationFrame(() => recompute({ newDraw: true }));
  });
  document.querySelectorAll('input[name=mode]').forEach((r) => (r.onchange = () => setMode(r.value)));
  $('#K').onchange = () => setK($('#K').value);
  $('#K-minus').onclick = () => setK(S.K - 1);
  $('#K-plus').onclick = () => setK(S.K + 1);
  document.querySelectorAll('input[name=dims]').forEach((r) => (r.onchange = () => setDims(+r.value)));
  $('#resample').onclick = () => { S.manual = false; S.drawNo++; recompute({ newDraw: true }); };
  $('#add-more').onclick = addMore;
  $('#manual-back').onclick = () => { S.manual = false; S.K = effectiveK(S.Kwant ?? S.K, S.distinct, S.logp.length); $('#K').value = S.K; recompute({ newDraw: true }); };
  $('#size-toggle').onchange = () => { if (S.pool) refreshDisplay(); };
  $('#cone-toggle').onchange = () => { if (S.pool) render(); };
  document.querySelectorAll('input[name=axes]').forEach((r) => (r.onchange = () => { if (S.pool) render(); }));
  $('#ghost-toggle').onchange = () => { if (S.pool) plot.update(); };
  $('#color-toggle').onchange = () => { syncCl(); if (S.pool) render(); };
  $('#adv-cl').addEventListener('change', onClusterChange);
  $('#cl-minus').onclick = () => { $('#cl-k').value = clampInt(+$('#cl-k').value - 1, 2, 8); onClusterChange(); syncChanged(); };
  $('#cl-plus').onclick = () => { $('#cl-k').value = clampInt(+$('#cl-k').value + 1, 2, 8); onClusterChange(); syncChanged(); };
  syncCl();
  document.querySelectorAll('input[name=len]').forEach((r) => (r.onchange = refreshDisplay));
  document.querySelectorAll('input[name=wt]').forEach((r) => (r.onchange = () => { recompute({ newDraw: true }); }));
  $('#len-lock').onclick = () => { $('#len-lock').setAttribute('aria-pressed', String(!lockOn())); refreshDisplay(); };
  $('#label-toggle').onchange = () => renderLabels();
  document.addEventListener('keydown', (e) => {
    if (e.key !== 'Escape') return;
    for (const m of [$('#adv'), $('#exmenu')]) if (m.open) { m.open = false; if (m.contains(document.activeElement)) m.querySelector('summary').focus(); }
  });
  document.addEventListener('keydown', (e) => { if (e.key === 'Escape' && S.pinned2 >= 0) inspect(-1); });
  document.addEventListener('pointerdown', (e) => { if (e.pointerType === 'touch' && S.pinned2 >= 0 && !e.target.closest('#adv-mat, #inspect')) inspect(-1); });
  $('.pool-table').addEventListener('toggle', () => { if ($('.pool-table').open) renderTable(); });
  $('#live-go').onclick = runLive;
  // Composer: Enter generates, Shift+Enter adds a line; the send button needs a prompt.
  const syncGo = () => ($('#live-go').disabled = !$('#prompt').value.trim());
  $('#prompt').addEventListener('input', syncGo);
  $('#prompt').addEventListener('keydown', (e) => {
    if (e.key === 'Enter' && !e.shiftKey && !e.isComposing) { e.preventDefault(); if (!$('#live-go').hidden && !$('#live-go').disabled) runLive(); }
  });
  syncGo();
  // Close the settings menu on an outside click.
  document.addEventListener('click', (e) => { for (const m of [$('#adv'), $('#exmenu')]) if (m.open && !m.contains(e.target)) m.open = false; });
  // Only one composer menu open at a time.
  for (const [a, b] of [['#adv', '#exmenu'], ['#exmenu', '#adv']]) $(a).addEventListener('toggle', () => { if ($(a).open) $(b).open = false; });
  // Stop right away: shut the worker down (this also cancels a download). Files already
  // downloaded stay in the browser's cache; the next run starts a fresh worker.
  $('#live-stop').onclick = () => {
    S.runSeq++;
    cancelRun();
    // A stopped model change goes back to the model whose pool is on screen.
    if (S.source === 'custom' && S.shownModel && (S.model !== S.shownModel || S.where !== S.shownWhere)) { setModel(S.shownModel); S.where = S.shownWhere; }
    $('#live-status').textContent = S.source === 'custom' ? 'Stopped. The pool on screen is unchanged.' : 'Stopped. Press ↑ to start again.';
    updatePickers();
    window.__refreshCache?.();
  };
  let rt = 0;
  window.addEventListener('resize', () => { clearTimeout(rt); rt = setTimeout(() => { plot.reset(); render(); }, 200); });
  matchMedia('(prefers-color-scheme: dark)').addEventListener('change', () => { plot.reset(); render(); });
}
wire();
// Coming back from "Sign in with Hugging Face": restore the prompt and run it.
finishSignIn().then(async (r) => {
  if (!r) return;
  await window.__ready;
  if (r.stash.prompt) { $('#prompt').value = r.stash.prompt; $('#prompt').dispatchEvent(new Event('input')); }
  if (r.stash.model && BACKENDS.some((b) => isCloud(b) && b.caps.models.some((m) => m.key === r.stash.model))) { setModel(r.stash.model); S.forceCloud = true; updatePickers(); }
  $('#live-status').textContent = `Signed in to Hugging Face as ${r.auth.user}.`;
  if (r.stash.prompt) runLive();
}).catch((e) => { $('#live-status').textContent = 'Sign-in failed: ' + (e.message || e); });
loadGallery().catch((e) => { $('#readout').textContent = 'Could not load gallery: ' + e.message; });
