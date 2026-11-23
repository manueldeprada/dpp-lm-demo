# PoolDPP demo sampler

A FastAPI server that draws pools of completions from a local language model with exact log p and log q per string,
and returns unit-norm embeddings of each completion. The static demo page (`../index.html`) talks to it through a reverse
proxy under `/dpp_api/`. It runs on one 24 GB GPU: a resident Qwen3 14B AWQ engine and one small model in a swappable slot.

## What it serves

| Route | Use |
|---|---|
| `GET /health` | `{"status":"ok"}`; 503 when an engine fails `check_health()`; `starting` during start-up |
| `GET /status` | models and their state (`ready`, `loading`, `cold`, `unavailable`), embedders, limits the page needs |
| `POST /sample` | draw `n` completions from the annealed proposal q = p_{T'} and return log p, log q, embeddings |
| `POST /embed` | embed existing completions after the fact (`minilm`, `qwen3-emb`, `mpnet`, or the LM's own states `own`) |
| `POST /embed_own` | owner only: teacher-force completions and return uncentered own states |

Requests are JSON (`Content-Type: application/json`). Embeddings are base64 of little-endian float32, N x D.

`POST /sample` body: `prompt` (up to 2000 characters), `n` (2 to 200, default 30), `t_prime` (0.2 to 2, default 1),
`max_new_tokens` (8 to 256, default 128), `seed` (integer, default random), `model` (a key of `/status`), `embedder`.
The completions of one request use the seeds `seed + i`: choose seeds at least `n` apart to append independent draws to a pool.
Response: `samples[{text, logp, logq, nTokens, ended}]`, `emb`, `D`, `N`, `embedder`, `Tprime`, `seed`, `timing`.

`POST /embed` body: `texts` (1 to 200 strings of at most 2000 characters), `embedder`, and for `own` also `model`, `prompt` and optionally
`ended` (one boolean per text). The own states are recomputed by teacher forcing the text, so they are close to, not equal to, the
vectors `/sample` returns from the generation itself (cosine about 0.9999 for Gemma 3 and Qwen3, about 0.99 for Gemma 2 9B, whose stored
text loses trailing whitespace tokens).

Options `embedders`, `own_raw` and `return_ids` of `/sample` need the owner key.

## The LM's own states ("own" embedder)

For a completion, the vector is the mean over generation steps of the model's final hidden state, recovered from the logits the
sampler sees (`plugin/dpp_vllm_plugin/own.py`), projected to 256 dimensions with a fixed random matrix, centered by a per-model mean
and normalized. `search/build_own_P.py` builds the logit-to-projection matrix P from the model's unembedding matrix.
Enable it with `DPP_OWN_EMB=1`. Models whose final logits are soft-capped (Gemma 2) are handled by inverting the cap in the plugin.

## Install and run

```
python -m venv .venv && . .venv/bin/activate
pip install -r requirements.txt            # versions installed on the demo server; requirements-freeze.txt has the full list
pip install -e plugin/                     # registers the vLLM plugin that returns log q next to log p
DPP_LOGQ=1 VLLM_USE_V2_MODEL_RUNNER=0 python -m uvicorn server:app --host 127.0.0.1 --port 8040
```

The model weights come from the Hugging Face hub (`HF_HUB_OFFLINE=1` after the first download). `ninja` must be on `PATH`
(torch.compile and FlashInfer JIT).

Example systemd user unit (`~/.config/systemd/user/dpp-sampler.service`; `%h` is the home directory):

```
[Unit]
Description=PoolDPP demo sampler
After=network-online.target
StartLimitIntervalSec=900
StartLimitBurst=5

[Service]
WorkingDirectory=%h/dpp-sampler
Environment=DPP_LOGQ=1
Environment=VLLM_USE_V2_MODEL_RUNNER=0
Environment=DPP_RESIDENT_UTIL=0.50
Environment=DPP_SLOT_UTIL=0.36
Environment=DPP_OWN_EMB=1
Environment=HF_HUB_OFFLINE=1
Environment=TOKENIZERS_PARALLELISM=false
Environment=PATH=%h/dpp-sampler/.venv/bin:<directory with ninja>:/usr/local/bin:/usr/bin:/bin
ExecStart=%h/dpp-sampler/.venv/bin/python -m uvicorn server:app --host 127.0.0.1 --port 8040 --timeout-graceful-shutdown 20
Restart=always
RestartSec=15
TimeoutStopSec=40
StandardOutput=append:%h/dpp-sampler/logs/server.log
StandardError=append:%h/dpp-sampler/logs/server.log

[Install]
WantedBy=default.target
```

## Environment variables

| Variable | Default | Meaning |
|---|---|---|
| `DPP_LOGQ` | required `1` | the plugin returns log q |
| `VLLM_USE_V2_MODEL_RUNNER` | required `0` | the plugin patches the V1 sampler |
| `DPP_RESIDENT_UTIL` / `DPP_SLOT_UTIL` | 0.50 / 0.36 | GPU fraction of the resident and the default slot engine |
| `DPP_OWN_EMB` | off | `1` enables own states (plugin, `own_emb/<key>/`, `/embed` with `own`) |
| `DPP_OWN_DIR` | `./own_emb` | where the plugin reads `P.safetensors`, `meta.json`, `mu.npy` per model |
| `DPP_PRELOAD` | `qwen3-emb` | embedders loaded at start-up (comma list, empty for none) |
| `DPP_EMBEDDER` | `minilm` | embedder used when a request names none |
| `DPP_RUNS_PER_HOUR` / `DPP_EMBEDS_PER_HOUR` | 300 / 1200 | global ceilings |
| `DPP_LOGQ_CHECK` | off | `1` makes the plugin cross-check log q (slow, for tests) |

`model_util.json` (see `model_util.example.json`) maps a model key to its slot GPU fraction and overrides the table in `server.py`
at the next engine start.


## GPU budget

One 24 GB card (24,125 MiB usable) holds the resident 14B engine (util 0.50, about 12.3 GiB plus cached sampler buffers), one slot engine,
and the main process (MiniLM, Qwen3-Embedding preloaded, the P matrices). The slot fractions in `MODELS` leave at least 300 MiB free
when both engines run a 200 x 256 request with the Qwen3-Embedding embedder (measured warmed up; minimum free over each run):

| Slot model | util | extra engine arguments | KV cache | min free, qwen3-emb | min free, own embedder |
|---|---|---|---|---|---|
| Gemma 2 9B (AWQ int4) | 0.30 | | 1,902 tokens | 1144 to 1298 MiB | 698 to 1613 MiB |
| Gemma 3 4B (QAT int4) | 0.30 | `language_model_only` | 21,202 tokens | 702 MiB | |
| Qwen3 4B (bf16) | 0.35 | `max_num_seqs=32` | 5,024 tokens | 371 MiB | |
| Phi-3.5 mini (bf16) | 0.34 | `max_num_seqs=32` | 1,757 tokens | 776 MiB | |

Lower fractions do not start (Qwen3 4B needs more than 0.34 even with 32 sequences). The owner-only option `embedders` with three
embedders on a 4B slot model leaves less (about 120 MiB), so use it with a smaller `n` there.

## Owner key

A file `.own_secret` next to `server.py` (mode 600, random string; never commit it) enables the owner routes. A request with the header
`X-Own-Secret: <contents>` skips the per-client limits and may use the owner-only options. Without the file `/embed_own` answers 404.

## Limits

Defined at the top of `server.py`; the client key is the visitor address (IPv6 on its /64), taken from `X-Real-IP` when the peer is
the local reverse proxy. Without a real address all traffic shares one key, so the per-client limits act as global ones.

| Limit | Value |
|---|---|
| request body | 64 KB (`/sample`), 1.5 MiB (`/embed`), 4 MiB (owner) |
| generating requests per client | 1 at a time |
| generated tokens per client | 150,000 per hour (n x max_new_tokens, token bucket) |
| slot model swaps | 3 per client per hour, 30 s between any two swaps |
| `/embed` calls per client | 240 per hour |
| global | 300 runs, 1,000,000 tokens, 1,200 embeds per hour |
| queue | 8 waiting pools, 180 s wait |

Violations answer 429 with `Retry-After`. `/docs`, `/redoc` and `/openapi.json` are disabled.

## Health and recovery

`/health` checks both engines. The watchdog checks them every 30 s: two failures of the resident engine, or an `EngineDeadError`
in a request on it, exit the process with code 3 so systemd restarts it. A dead slot engine is reset and rebuilt on the next request;
three slot deaths within 10 minutes also exit.

## Tests

```
python tests/test_api_guard.py     # validation, limits, rotation (no GPU)
python tests/test_exit_paths.py    # exit and recovery paths with fake engines (no GPU)
python tests/http_checks.py basic  # against a running server: malformed input, docs, owner options
python tests/http_checks.py limits
python tests/http_checks.py swaps
python tests/check_plugin_exact.py 1.5 <hf repo>   # plugin log q against vLLM's own, stop the service first (needs a free GPU)
```
`http_checks.py` reads the owner key from `DPP_OWN_SECRET_FILE` (default `.own_secret`).

## Rebuilding the page data

All scripts take the layout root from `DPP_ROOT` (default: the current directory) and the page data directory from `DPP_DATA`
(default `dpp-lm-demo/data`).

1. `search/precompute_models.py` samples the 14 pools of each served model through the running server and runs `search/build_model_data.py`,
   which writes the pool files and refits the shared display bases on all models.
2. `search/embed_pools.py [--only <model>]` writes the Qwen3-Embedding vectors and their bases.
3. `search/build_own_P.py <model>`, then `search/precompute_own.py` (teacher forcing) or `search/regen_pools_own.py` plus
   `search/own_from_pools.py` (own states from the generation itself), then `search/own_sanity.py`.
