import test from 'node:test';
import assert from 'node:assert/strict';
import { waitWords, embeddersFor, parseStatus, sampleBody, mapError, BackendError, parseBackendParam, backendConfigs, createBackends, decodePool, decodeEmbed, chooseBackend } from '../js/backends.js';

const b64 = (arr) => Buffer.from(new Float32Array(arr).buffer).toString('base64');
const pool = (N = 2, D = 3) => ({ prompt: 'p', model: 'm', N, D, Tprime: 1, samples: Array.from({ length: N }, (_, i) => ({ text: `s${i}`, logp: -1, logq: -1, ended: true })), emb: b64(Array.from({ length: N * D }, (_, i) => i / 10)) });
const res = (status, body) => ({ ok: status < 400, status, json: async () => body });

test('parseStatus: our shape and the minimal Modal shape', () => {
  const ours = parseStatus({ models: [{ key: 'a', name: 'A', state: 'ready' }, { key: 'b', state: 'cold' }], defaultModel: 'b', embedders: ['minilm', 'qwen3-emb'], maxN: 300 }, { auth: 'none' });
  assert.deepEqual(ours.models.map((m) => m.key), ['a', 'b']);
  assert.equal(ours.def, 'b'); assert.equal(ours.multi, true); assert.deepEqual(ours.embedders, ['minilm', 'qwen3-emb']); assert.equal(ours.maxN, 300);
  const modal = parseStatus({ model: 'qwen3-14b', gpu_minutes_left_this_month: 41 }, { auth: 'hf' });
  assert.deepEqual(modal.models, [{ key: 'qwen3-14b', name: 'qwen3-14b', state: 'ready' }]);
  assert.deepEqual(modal.embedders, ['minilm']); assert.equal(modal.auth, 'hf'); assert.equal(modal.multi, false); assert.equal(modal.budget, 41);
  assert.equal(parseStatus({}, {}).models.length, 0);
  assert.deepEqual(parseStatus({ model: 'x' }, { embedders: ['minilm', 'qwen3-emb'] }).embedders, ['minilm', 'qwen3-emb']);
});

test('sampleBody: model only when multi, embedder only when offered or different', () => {
  const req = { prompt: 'hi', n: 30, t_prime: 1.2, max_new_tokens: 64, seed: 7, model: 'qwen3-14b', embedder: 'qwen3-emb' };
  assert.deepEqual(sampleBody(req, { multi: true, embedders: ['minilm', 'qwen3-emb'] }), { prompt: 'hi', n: 30, t_prime: 1.2, max_new_tokens: 64, seed: 7, model: 'qwen3-14b', embedder: 'qwen3-emb' });
  const m = sampleBody({ ...req, embedder: 'minilm' }, { multi: false, embedders: ['minilm'] });
  assert.equal('model' in m, false); assert.equal('embedder' in m, false);
  assert.ok(Number.isInteger(sampleBody({ ...req, seed: undefined }, { embedders: ['minilm'] }).seed));
});

test('mapError: codes and messages', () => {
  assert.equal(mapError(400, { detail: 'The prompt is empty.' }).code, 'bad_request');
  assert.equal(mapError(401, { detail: 'Sign in' }).code, 'unauthorized');
  assert.equal(mapError(429, { detail: 'used up' }).message, 'used up');
  assert.equal(mapError(503, {}).code, 'unavailable');
  assert.equal(mapError(502, {}).code, 'unavailable');
  assert.match(mapError(429, {}).message, /limit/);
  assert.ok(mapError(400, { error: { message: 'x' } }) instanceof BackendError);
});

test('parseBackendParam: only http(s) URLs', () => {
  assert.deepEqual(parseBackendParam('?backend=https://a.example.com/api/'), { url: 'https://a.example.com/api', origin: 'https://a.example.com' });
  assert.equal(parseBackendParam('?backend=http://localhost:8000').origin, 'http://localhost:8000');
  for (const bad of ['javascript:alert(1)', 'ftp://x.com', '/relative', 'file:///etc/passwd', '']) assert.equal(parseBackendParam(`?backend=${encodeURIComponent(bad)}`), null);
  assert.equal(parseBackendParam('?x=1'), null);
  const cfgs = backendConfigs([{ id: 'server', kind: 'http', url: '/dpp_api' }, { bad: true }], '?backend=http://localhost:9');
  assert.deepEqual(cfgs.map((c) => c.id), ['custom', 'server']);
});

test('http backend: status, sample request and decoding, error mapping, embed', async () => {
  const calls = [];
  const fetch = async (url, opt = {}) => {
    calls.push([url, opt.method ?? 'GET', opt.body ? JSON.parse(opt.body) : null, opt.headers]);
    if (url.endsWith('/status')) return res(200, { models: [{ key: 'q', name: 'Q', state: 'ready' }], embedders: ['minilm', 'qwen3-emb'], embed: true });
    if (url.endsWith('/sample')) return JSON.parse(opt.body).prompt === 'bad' ? res(400, { detail: 'The prompt is empty.' }) : res(200, pool());
    if (url.endsWith('/embed')) return res(200, { embedder: 'qwen3-emb', D: 3, emb: b64([1, 2, 3, 4, 5, 6]) });
    return res(404, {});
  };
  const [b] = createBackends([{ id: 's', kind: 'http', url: 'http://x/api', auth: 'hf' }], { fetch, getToken: () => 'tok' });
  const caps = await b.status();
  assert.equal(caps.up, true); assert.equal(caps.embed, true);
  const out = await b.sample({ prompt: 'hi', n: 2, t_prime: 1, max_new_tokens: 8, seed: 1, model: 'q', embedder: 'qwen3-emb' });
  assert.equal(out.emb.length, 6); assert.equal(out.pool.N, 2); assert.equal('emb' in out.pool, false);
  const sample = calls.find((c) => c[0].endsWith('/sample'));
  assert.equal(sample[1], 'POST'); assert.equal(sample[2].model, 'q'); assert.equal(sample[2].embedder, 'qwen3-emb'); assert.equal(sample[3].Authorization, 'Bearer tok');
  await assert.rejects(b.sample({ prompt: 'bad', n: 2, t_prime: 1, max_new_tokens: 8 }), (e) => e.code === 'bad_request' && /empty/.test(e.message));
  const em = await b.embed({ texts: ['a', 'b'], embedder: 'qwen3-emb', model: 'q', prompt: 'hi' });
  assert.equal(em.D, 3); assert.equal(em.emb.length, 6);
  assert.deepEqual(calls.find((c) => c[0].endsWith('/embed'))[2], { texts: ['a', 'b'], embedder: 'qwen3-emb', model: 'q', prompt: 'hi' });
});

test('http backend: unreachable status and sample', async () => {
  const [b] = createBackends([{ id: 's', kind: 'http', url: 'http://x' }], { fetch: async () => { throw new TypeError('fail'); } });
  assert.equal((await b.status()).up, false);
  await assert.rejects(b.sample({ prompt: 'x', n: 2, t_prime: 1, max_new_tokens: 8 }), (e) => e.code === 'unavailable');
});

test('decodePool and decodeEmbed reject mismatched sizes', () => {
  const bad = pool(2, 3); bad.D = 4;
  assert.throws(() => decodePool(bad), BackendError);
  assert.throws(() => decodePool({ samples: [] }), BackendError);
  assert.throws(() => decodeEmbed({}), BackendError);
  const k = pool(); delete k.emb; k.kernels = [{ layer: 12, gram: [] }];
  assert.throws(() => decodePool(k), /kernels but no embedding/);
});

test('chooseBackend: first backend with the model, embedder fallback is reported', () => {
  const mk = (id, models, embedders, up = true) => ({ id, caps: { up, models: models.map((key) => ({ key })), embedders } });
  const bs = [mk('server', ['a'], ['minilm', 'qwen3-emb'], false), mk('modal', ['a'], ['minilm']), mk('browser', ['z'], ['minilm'])];
  const c = chooseBackend(bs, { model: 'a', embedder: 'qwen3-emb' });
  assert.equal(c.backend.id, 'modal'); assert.equal(c.embedder, 'minilm'); assert.equal(c.fellBack, true);
  assert.equal(chooseBackend(bs, { model: 'a', embedder: 'minilm' }).fellBack, false);
  assert.equal(chooseBackend(bs, { model: 'nope', embedder: 'minilm' }), null);
  assert.equal(chooseBackend(bs, { model: 'a', embedder: 'minilm' }, (b) => b.id !== 'modal'), null);
});

test('parseStatus: the deployed shape (object embedders, ownModels, limits, embed, continuePool)', () => {
  const st = { model: 'qwen3-14b', modelName: 'Qwen3 14B', defaultModel: 'qwen3-14b', models: [{ key: 'qwen3-14b', name: 'Qwen3 14B', state: 'ready' }, { key: 'gemma-2-9b', name: 'Gemma 2 9B', state: 'cold' }],
    embedders: { minilm: 'sentence-transformers/all-MiniLM-L6-v2', 'qwen3-emb': 'Qwen/Qwen3-Embedding-0.6B', mpnet: 'x', own: null }, ownModels: ['qwen3-14b'], defaultEmbedder: 'qwen3-emb',
    maxN: 200, maxNewTokens: 256, maxTextChars: 2000, embed: true, continuePool: true };
  const c = parseStatus(st, {});
  assert.deepEqual(c.embedders, ['minilm', 'qwen3-emb', 'mpnet', 'own']);
  assert.deepEqual(c.ownModels, ['qwen3-14b']); assert.equal(c.defaultEmbedder, 'qwen3-emb');
  assert.equal(c.maxN, 200); assert.equal(c.maxTokens, 256); assert.equal(c.maxTextChars, 2000); assert.equal(c.embed, true); assert.equal(c.continuePool, true);
  assert.deepEqual(embeddersFor(c, 'qwen3-14b'), ['minilm', 'qwen3-emb', 'mpnet', 'own']);
  assert.deepEqual(embeddersFor(c, 'gemma-2-9b'), ['minilm', 'qwen3-emb', 'mpnet']);
  assert.equal(parseStatus({ model: 'x' }, {}).ownModels, null);
  assert.deepEqual(embeddersFor(parseStatus({ model: 'x' }, {}), 'x'), ['minilm']);
});

test('chooseBackend: own is offered only for models in ownModels', () => {
  const caps = { up: true, models: [{ key: 'a' }, { key: 'b' }], embedders: ['minilm', 'qwen3-emb', 'own'], ownModels: ['a'] };
  const bs = [{ id: 's', caps }];
  assert.equal(chooseBackend(bs, { model: 'a', embedder: 'own' }).fellBack, false);
  const c = chooseBackend(bs, { model: 'b', embedder: 'own' });
  assert.equal(c.fellBack, true); assert.equal(c.embedder, 'minilm');
  assert.equal(chooseBackend(bs, { model: 'b', embedder: 'qwen3-emb' }).embedder, 'qwen3-emb');
});

test('errors: 429 with Retry-After in words, 503 restarting, waitWords', () => {
  const e = mapError(429, { detail: 'Too many requests.' }, '125');
  assert.equal(e.code, 'rate_limited'); assert.equal(e.retryAfter, 125); assert.match(e.message, /about 2 minutes/);
  assert.match(mapError(429, { detail: 'Busy.' }, '30').message, /30 seconds/);
  assert.equal(mapError(429, { detail: 'Busy.' }, null).message, 'Busy.');
  const dup = mapError(429, { detail: 'You already have a request running. Wait for it to finish, then try again.' }, '5');
  assert.equal(dup.message, 'You already have a request running. Wait for it to finish, then try again.'); assert.equal(dup.retryAfter, 5);
  assert.equal(mapError(503, { detail: 'engine died' }).message, 'The model is restarting. Try again in a minute.');
  assert.equal(mapError(413, { detail: 'too large' }).code, 'bad_request');
  assert.equal(waitWords(1), '1 second'); assert.equal(waitWords(3600), 'about 60 minutes'); assert.equal(waitWords(7200), 'about 2 hours');
});

test('embed sends ended and returns approximate and note; sample reads Retry-After', async () => {
  const calls = [];
  const fetch = async (url, opt = {}) => {
    calls.push([url, opt.body ? JSON.parse(opt.body) : null]);
    if (url.endsWith('/status')) return res(200, { models: [{ key: 'q' }], embedders: { minilm: 'm', own: null }, ownModels: ['q'], embed: true });
    if (url.endsWith('/embed')) return res(200, { N: 2, embedder: 'own', D: 3, emb: b64([1, 2, 3, 4, 5, 6]), approximate: true, note: 'Teacher-forced states.' });
    return { ok: false, status: 429, headers: { get: (k) => (k === 'Retry-After' ? '90' : null) }, json: async () => ({ detail: 'Slow down.' }) };
  };
  const [b] = createBackends([{ id: 's', kind: 'http', url: 'http://x' }], { fetch });
  await b.status();
  const em = await b.embed({ texts: ['a', 'b'], embedder: 'own', model: 'q', prompt: 'p', ended: [true, false] });
  assert.deepEqual(calls.find((c) => c[0].endsWith('/embed'))[1].ended, [true, false]);
  assert.equal(em.approximate, true); assert.equal(em.note, 'Teacher-forced states.');
  await assert.rejects(b.sample({ prompt: 'x', n: 2, t_prime: 1, max_new_tokens: 8 }), (e) => e.code === 'rate_limited' && /Try again in about 2 minutes/.test(e.message));
});
