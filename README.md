# inference-router

Per-request vLLM template overrides may be supplied as `chat_template_overrides`,
an object keyed by concrete model ID. The proxy selects the resolved model's entry
after alias/fallback routing, forwards it as `chat_template`, and removes the map.
No match leaves normal backend formatting in place. GET
`/api/models/{id}/chat-template` reads the registered model's template without
loading or modifying it; `/tokenize` and `/detokenize` proxy token-only previews.
vLLM must enable `--trust-request-chat-template`. The Clara host's optional
`scripts/clara-vllm-request-templates` launcher adds this flag through `EXTRA_ARGS`
while retaining the existing Qwen runtime launcher and model files.

A VRAM-aware, OpenAI-compatible HTTP proxy for [llama.cpp](https://github.com/ggerganov/llama.cpp) (and vLLM-style safetensors backends). Runs one backend process per loaded model, routes `/v1/*` requests to the right process by the `model` field in the JSON body, and evicts models under VRAM pressure.

Built for a single-user, localhost, multi-GPU AMD workstation (3x Radeon AI PRO R9700). No auth, no rate limiting, no multi-tenant isolation.

## What it does

- **One endpoint, many models.** Clients (Claude Code, aider, continue.dev, raw curl) POST to `http://localhost:8080/v1/chat/completions` with `"model": "<id>"`. The router spawns the backend on demand and proxies the request byte-for-byte — so the full OpenAI surface works, including streaming, tools, and anything llama.cpp adds tomorrow.
- **VRAM-aware admission.** Before spawning, the orchestrator reads GGUF metadata, estimates VRAM, checks free VRAM across all GPUs, and evicts idle models if needed. Eviction prefers long-idle and small models.
- **Smart GPU allocation.** Picks the minimum GPU subset that fits and passes `--tensor-split` explicitly. Never occupies a GPU it doesn't need.
- **Browser dashboard.** Single-page UI at `/` for CRUD on models + binary presets, manual load/stop, live GPU / CPU / RAM stats, and persisted decode/prefill throughput for llama.cpp and vLLM models. Askama HTML templates, vanilla JS, 500 ms poll.
- **Persistence.** Model and preset definitions live in `~/.config/inference-router/{models.json,presets.json}`; throughput history lives in `model_perf.json`. Writes are dirty-flag gated and flushed by the reconcile loop (5 s cadence).

## HTTP surface

Default bind: `0.0.0.0:8080`.

| Path | Methods | Purpose |
| --- | --- | --- |
| `/` | GET | Dashboard (HTML) |
| `/api/status` | GET | Live system + GPU + model snapshot (dashboard polls this) |
| `/api/models` | GET, POST | List / create model definitions |
| `/api/models/validate` | POST | Validate a model definition without saving it |
| `/api/models/{id}` | PUT, DELETE | Update / delete |
| `/api/models/{id}/load` | POST | Load (ensure backend process is running) |
| `/api/models/{id}/stop` | POST | Stop (kill backend process) |
| `/api/presets` | GET, POST | List / create binary presets |
| `/api/presets/{id}` | PUT, DELETE | Update / delete |
| `/api/files?path=...` | GET | Directory browser for the model form |
| `/api/gguf-info?path=...` | GET | Read GGUF metadata (drives context slider + VRAM preview) |
| `/v1/models` | GET | Synthesized OpenAI model list |
| `/v1/*` | POST | Byte-level passthrough to the backend owning the `model` id in the body |
| `/healthz` | GET | Liveness |

## Running

### systemd (user unit)

Installed unit at `~/.config/systemd/user/inference-router.service`:

```bash
systemctl --user start inference-router
systemctl --user enable inference-router     # auto-start on login
journalctl --user -u inference-router -f     # tail logs
```

### Manual

```bash
cargo run --release
# or
./target/release/inference-router
```

Environment:
- `RUST_LOG` — `tracing` filter, e.g. `inference_router=debug`. Defaults to `info`.
- `RADV_DEBUG=nocompute` — forces the graphics queue on AMD RDNA4 (2.4x TG improvement over the compute queue). Set in the systemd unit by default.
- `INFERENCE_ROUTER_MAX_BODY_BYTES` — maximum proxied request body size. Defaults to `1073741824` (1 GiB).
- `INFERENCE_ROUTER_MAX_INSTANCES_PER_MODEL` — maximum concurrent backend processes per model. Defaults to VRAM-limited scale-out; set it to cap replica processes while requests are busy.
- `INFERENCE_ROUTER_VRAM_WAIT_MS` — how long a load waits for active requests to release VRAM before failing. Defaults to `300000` (5 minutes); set `0` to fail immediately.
- `INFERENCE_ROUTER_BACKEND_PORT_RANGE` — optional inclusive range for backend processes, e.g. `9100-9199`. Defaults to OS-assigned ephemeral ports.
- `INFERENCE_ROUTER_LOOP_*` / `INFERENCE_ROUTER_TOOL_LOOP_*` — initial loop-guard defaults when `settings.json` does not exist yet. After that, use Settings → Loop guards in the UI.

The port is `8080`, hardcoded in `AppConfig::default()` (see `src/lifecycle.rs`).

## Configuration

For a persistent replica pool, add `"replica_pools": {"MODEL_ID": 2}` to
`settings.json`. The existing instance cap still applies. The reconcile loop
starts missing replicas in the background while healthy instances continue
serving requests. Launchers must lease disjoint GPU sets and use distinct
container names; the Qwen host launcher reads these from
`/home/aristath/vllm-r9700/qwen38-replicas.json`.

Pool requests prefer the least-busy replica, retaining conversation affinity
when loads tie so the backend can reuse its prefix cache. The affinity key uses
`prompt_cache_key`, session affinity headers, or bounded hashes of increasingly
specific conversation prefixes, and never stores prompt text. A transport failure before response headers gets one attempt on a
surviving replica. A response that has already started is not replayed on another
replica; interrupted streams must be retried by the client. Dead processes are
removed and their replacements warmed automatically.

State files under `~/.config/inference-router/`:

- **`presets.json`** — named binary paths. Lets you rebuild llama.cpp once and have every model pick up the new binary.
- **`settings.json`** — server-level app settings, including streaming and cross-turn loop guard controls exposed in the Settings modal.
- **`models.json`** — one entry per model. Key fields:
  - `binary_preset` (optional): preset id to resolve `binary` from at spawn time.
  - `profile` (optional): workspace label such as `coding`, `long-context`, or `vision` for dashboard filtering.
  - `binary`, `model_path`, `port`, `context`: mandatory.
  - `weights_format`: `gguf` (llama.cpp argv style) or `safetensors` (vLLM style).
  - `cpu_only: true` runs a model entirely on CPU without GPU admission or fit probing. Use a CPU-capable binary preset (such as a CPU build of vLLM for safetensors). For GGUF this also disables draft/MTP and projector offload; normal GPU placement is unchanged when omitted.
  - `igpu_only: true` runs a GGUF model fully on an integrated Vulkan GPU without the discrete-GPU media gate. It fails if no integrated Vulkan GPU exists; it never falls back to a discrete GPU. CPU-only and iGPU-only are mutually exclusive. Draft/MTP and raw GPU-placement overrides are not supported in iGPU-only mode.
  - llama.cpp knobs: `flash_attn`, `n_gpu_layers`, `mlock`, `no_mmap`, `parallel_slots`, `cache_type_{k,v}`, `split_mode` (`none | layer | row | tensor`), `main_gpu`, `tensor_split`.
  - Speculative decoding: `mtp_tokens` for embedded MTP heads, or `draft_model_id` plus `draft_{max,min,p_min}` for an external draft model.
  - `extra_args`: arbitrary argv array — escape hatch for flags not modelled.
  - Sampling: `temperature`, `top_p`, `top_k`, `min_p`.

Easiest path is to edit through the dashboard — the model form validates against GGUF metadata and previews VRAM live.

## How it works

```
client ──POST /v1/chat/completions──▶  inference-router
                                          │
                                          │  peek `model` in body
                                          ▼
                                      Orchestrator
                                      ├─ ensure_loaded(id)
                                      │    ├─ spawn backend (llama-server / vllm)
                                      │    │    kill_on_drop = true
                                      │    └─ wait for ready port
                                      │
                                      └─ proxy bytes → backend:port
```

- **One backend process per loaded model.** No sharing, no hot-swap.
- **Eviction = kill.** No idle timeout; models stay resident until VRAM pressure forces eviction.
- **Graceful shutdown.** The router installs SIGINT and SIGTERM handlers; `tokio::process::Child::kill_on_drop(true)` SIGKILLs every backend as the orchestrator drops. A `systemctl --user stop` or plain `kill` cleans up children too.

## Exclusive media work on a shared GPU

For a host where inference and media share one GPU, register trusted worker
commands in `~/.config/inference-router/media.json`:

```json
{
  "resident_model": "default",
  "workers": {
    "image": {
      "command": ["/absolute/path/to/python", "/absolute/path/to/image-worker.py"],
      "timeout_secs": 1800
    },
    "video": {
      "command": ["/absolute/path/to/python", "/absolute/path/to/video-worker.py"],
      "timeout_secs": 7200
    }
  }
}
```

These are operator-controlled examples, not installed backends. Restart the
router after editing the file. Worker commands are literal executable/argument
arrays, never shell commands supplied by HTTP clients. Each worker reads one
JSON document from stdin, writes its result as JSON to stdout, sends diagnostics
to stderr, and exits. It must keep GPU work in its owned process group: no
independently daemonized servers or detached containers. A long-lived backend
that launches separate GPU processes must be adapted before it can be registered.

`POST /api/gpu/run/image` (or another registered worker name) accepts that JSON
input and returns the worker's JSON result only after cleanup and resident-model
restoration. Existing media job controllers can call this endpoint from their
normal durable job; they should retain their own request IDs and cancellation
policy. This endpoint is synchronous and does not claim durable HTTP delivery or
automatic replay after a router restart.

The router uses fair shared/exclusive admission:

1. A media request joins the admission queue. Later GPU inference cannot jump ahead
   of it, so continued requests from other applications cannot starve media.
2. Already admitted GPU inference and model loads drain, including complete SSE
   responses and their immediate performance-accounting work.
3. GPU-backed inference processes are unloaded. Explicitly CPU-only resident
   processes need not be evicted and their requests can run alongside media.
4. The router starts the worker in its own process group under an exclusive
   permit. GPU requests queue unless their alias has a usable CPU fallback.
5. The worker and any remaining group members are terminated and checked before
   the configured resident model is loaded and passes its normal readiness check.
6. Queued inference is admitted in FIFO lock order, retaining normal inference
   concurrency. Waiting inference gets its turn before later queued media work;
   response completion order can differ when inference runs concurrently.

This covers ordinary `/v1/*` requests and explicit model-load requests. Inference
uses socket backpressure before ordinary admission. During media, routing metadata
is read first and large queued prompts spill to private temporary files (see
[alias fallback](#alias-fallback-during-media)). Unregistered workers are
rejected. Requests that disconnect while queued are cancelled. The backend idle
timeout starts after dispatch, not while waiting for media. External clients may
still impose their own timeouts; the router cannot deliver to a closed connection.
A media worker must not call this router's inference endpoints while holding the
exclusive permit, because that request would wait for the worker itself.

Worker failure, timeout, or client cancellation triggers group cleanup and
restoration. Losing a client does not abandon the cleanup task. If cleanup or
restoration fails, queued inference receives an explicit recovery error; it is
never allowed to overlap uncertain GPU ownership. `POST /api/gpu/recover` retries
verified cleanup and resident-model restoration after the underlying fault is fixed.

`GET /api/gpu` and `gpu_admission` in `/api/status` expose the phase, current media
job, queued inference/media counts, and recovery error. Handoffs also appear in
router events. The admission domain is the router's entire GPU pool; this feature
is intended for exclusive media ownership on a shared GPU, not simultaneous media
and inference on separate GPUs.

The worker saves its boot/PID/start-time identity in
`media-runtime/active-worker.json` **before** executing the configured command.
After restart the router closes admission and cleans up a verified leftover
worker before restoring inference. A potentially reused PID or unverified orphan
group fails closed for operator inspection, rather than killing an unrelated
process or assuming the GPU is free. Router shutdown cancels media and cleans up
its group. The checkpoint contains process identity, not the request body.

The executable normally supervises its own workers using its internal
`--gpu-worker` mode. A library embedding the router can provide the matching
helper binary with `INFERENCE_ROUTER_WORKER_EXECUTABLE`.

## Development

```bash
cargo build
cargo test
cargo clippy --all-targets
```

Integration tests in `tests/proxy_integration.rs` exercise the `/v1/*` proxy against a synthesized upstream — no real llama.cpp spawn required.

### Source layout

```
src/
├── api/            routes, OpenAI passthrough (proxy.rs), body peeking
├── config/         ModelConfig, BinaryPreset, JsonStore
├── orchestrator/   ensure_loaded, eviction, smart GPU allocation
├── process/        spawn/kill wrapper, argv builder
├── system/         CPU % / RAM / CPU temp from /proc and /sys/class/hwmon
├── vram/           GGUF metadata reader (ggus), AMD sysfs tracker
├── ui/             askama template types
├── lifecycle.rs    bootstrap, reconcile loop, signal handling
├── lib.rs, main.rs
templates/          base.html, dashboard.html
tests/              integration tests
```

## Hardware / scope

- **Target:** single workstation, 3x AMD Radeon AI PRO R9700, llama.cpp Vulkan and ROCm builds.

## Examples

### 1. Creating a Basic Model

```bash
# Create a model using the dashboard or API
curl -X POST http://localhost:8080/api/models \
  -H "Content-Type: application/json" \
  -d '{
    "id": "mistral-7b",
    "name": "Mistral 7B",
    "binary": "/path/to/llama-server",
    "model_path": "/path/to/mistral-7b-v0.1.Q4_K_M.gguf",
    "weights_format": "gguf",
    "context": 4096,
    "n_gpu_layers": 32
  }'
```

### 2. Using Speculative Decoding

```bash
# Create a draft model first
curl -X POST http://localhost:8080/api/models \
  -H "Content-Type: application/json" \
  -d '{
    "id": "tiny-draft",
    "name": "Tiny Draft Model",
    "binary": "/path/to/llama-server",
    "model_path": "/path/to/tiny-model.gguf",
    "weights_format": "gguf",
    "context": 2048
  }'

# Then create a target model that uses it for speculative decoding
curl -X POST http://localhost:8080/api/models \
  -H "Content-Type: application/json" \
  -d '{
    "id": "speculative-model",
    "name": "Speculative Model",
    "binary": "/path/to/llama-server",
    "model_path": "/path/to/large-model.gguf",
    "weights_format": "gguf",
    "context": 4096,
    "draft_model_id": "tiny-draft",
    "draft_max": 5,
    "draft_min": 2
  }'
```

### 3. Using Vision Models

```bash
# Create a vision model with mmproj file
curl -X POST http://localhost:8080/api/models \
  -H "Content-Type: application/json" \
  -d '{
    "id": "llava-1.5",
    "name": "LLaVA 1.5",
    "binary": "/path/to/llama-server",
    "model_path": "/path/to/llava-1.5.Q4_K_M.gguf",
    "mmproj_path": "/path/to/mmproj-model-f16.gguf",
    "weights_format": "gguf",
    "context": 2048
  }'

# Use with image input (via OpenAI API)
curl http://localhost:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "llava-1.5",
    "messages": [
      {
        "role": "user",
        "content": [
          {"type": "text", "text": "What is in this image?"},
          {
            "type": "image_url",
            "image_url": {
              "url": "data:image/jpeg;base64,/9j/4AAQSkZJRgABAQ..."
            }
          }
        ]
      }
    ]
  }'
```

### 4. Using the OpenAI-Compatible API

```bash
# Standard chat completion
curl http://localhost:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "mistral-7b",
    "messages": [
      {"role": "user", "content": "Hello!"}
    ]
  }'

# Streaming response
curl http://localhost:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "mistral-7b",
    "messages": [
      {"role": "user", "content": "Tell me a story"}
    ],
    "stream": true
  }'
```

## API Reference

### Management API (`/api/*`)

#### `GET /api/status`
Returns complete system status including:
- CPU/RAM usage and temperature
- GPU metrics (VRAM, utilization, temperature)
- All model configurations and runtime states
- Recent orchestrator events

**Response:** `StatusResponse` (see [src/api/state.rs](#))

#### Model Management
- `GET /api/models` — List all models
- `POST /api/models` — Create a new model
- `POST /api/models/validate` — Validate a model config
- `PUT /api/models/{id}` — Update a model
- `DELETE /api/models/{id}` — Delete a model
- `POST /api/models/{id}/load` — Load a model
- `POST /api/models/{id}/stop` — Stop a model

**Request/Response:** `ModelConfig` (see [src/config/model.rs](#))

#### Preset Management
- `GET /api/presets` — List all binary presets
- `POST /api/presets` — Create a new preset
- `PUT /api/presets/{id}` — Update a preset
- `DELETE /api/presets/{id}` — Delete a preset

**Request/Response:** `BinaryPreset` (see [src/config/preset.rs](#))

#### Settings
- `GET /api/settings` — Get current settings
- `PUT /api/settings` — Update settings

**Request/Response:** `AppSettings` (see [src/config/settings.rs](#))

### OpenAI-Compatible Proxy (`/v1/*`)

#### `GET /v1/models`
Returns a synthesized model list in OpenAI format.

#### `POST /v1/{endpoint}`
Proxies requests to the appropriate backend based on the `model` field.

- Supports all OpenAI endpoints (`/chat/completions`, `/completions`, etc.)
- Preserves streaming responses
- Applies loop guard corrections when enabled

### Utility Endpoints

#### `GET /api/files?path={path}`
Directory browser for model configuration forms.

#### `GET /api/gguf-info?path={path}`
Reads and returns GGUF metadata from a model file.

#### `GET /healthz`
Liveness check (returns "ok").
- **Not a goal:** multi-host, multi-user, auth, quotas, CUDA.

### Shared SYCL GPU admission

`POST /api/backends/{backend}/release-idle` stops idle running models whose binary
preset targets that compute backend. It returns `{"released":["model-id"]}`.
An active request returns HTTP 409 without partial eviction. Other backends and
loading models are preserved. Request guards and the active check share the
process-manager mutex, so a running inference cannot be stopped by a stale idle
observation. Host GPU locks remain the final admission boundary during loads.

For Clara's B60, install the `scripts/llama-sycl-b60` launcher behind the existing
SYCL server and fit-tool wrappers. Set `INFERENCE_LLAMA_SERVER` to the native
`llama-server` or `llama-fit-params` executable in each wrapper. The launcher
sources oneAPI, releases idle SYCL peers, acquires the media workers' two B60
locks, and execs the native program with inherited reservations and parent-death
protection. Introspection commands do not evict models. All model parameters
remain normal router model configuration.

The optional model field `request_idle_timeout_secs` overrides the upstream
watchdog idle timeout for that model. Omit it to inherit the global setting, or
set it to `0` to allow long non-streaming prefills/completions. Engine-reset
watching and idle-instance liveness checks retain their normal policy.

Clara calls the `slm` alias using `/v1/chat/completions`. Reassign that alias in
the dashboard or with `PUT /api/aliases/slm` to change the model without changing
Clara. Its initial `qwen3.5-2b-mtp` target uses the B60/SYCL preset, native 262144
context, one slot, F16 KV, `mtp_tokens: 3`, `threads: 4`, flash attention,
`request_idle_timeout_secs: 0`, and default `enable_thinking: false`.

The [CPU SLM profile](examples/clara-cpu-slm.json) uses:
Gemma4 26B-A4B **IQ2_M weights**, a **262144-token context**, and **Q4_0 K/V
caches**, with one CPU-only slot. The legacy model ID ending in `cpu-32k` is
retained so existing `fast`/`slm` aliases and clients keep working; its display
name describes the current 256K configuration. The context budget includes both
the input and generated output. Q4 here applies to KV cache, not model weights.

After installing the model and `llama-cpu` preset on Clara, apply the profile
when the model has no active requests:

```sh
curl --fail-with-body -X PUT \
  http://127.0.0.1:8080/api/models/gemma4-26b-a4b-iq2m-cpu-32k \
  -H 'Content-Type: application/json' \
  --data-binary @examples/clara-cpu-slm.json
```

Updating spawn settings stops the existing instance. The next `fast`/`slm`
request reloads it; Core and inference-router do not need restarting. Verify
the managed process has `-c 262144 --cache-type-k q4_0 --cache-type-v q4_0`
and that its `/slots` reports `n_ctx: 262144`.

### Clara iGPU fallback

Clara's active fallback uses [this iGPU profile](examples/clara-igpu-slm.json)
and [Vulkan preset](examples/clara-igpu-preset.json). It preserves the IQ2_M
weights, 262144-token context, Q4_0 K/V cache and legacy model ID; `fast`, `slm`
and the default alias's fallback keep their existing targets. All model layers
use the integrated GPU and shared system RAM, outside the RTX 3090's VRAM budget.
The CPU profile above remains available for rollback.

Install the preset with `POST /api/presets`, then apply the iGPU model profile
with `PUT /api/models/gemma4-26b-a4b-iq2m-cpu-32k` when the model is idle.
The Hardware tab's **Compute placement** selector exposes automatic, CPU-only
and integrated-GPU-only modes. The alias fallback selector accepts either
independent mode. The router selects the integrated Vulkan device from the
current inventory, enforces its placement flags, and leaves its processes
running during discrete-GPU media unload/restoration. GPU reset recovery still
tracks the iGPU process.

### Alias fallback during media

The alias editor has an optional **Fallback during media generation** selector.
Choose a CPU-only or iGPU-only GGUF model, or an alias such as `fast` that resolves to one:

```json
{"alias":"default","target":"qwen3.8-27b","fallback":"fast"}
```

New requests use the primary model while GPU admission is available. During
media draining, generation, resident restoration, or a GPU recovery fault, they
use the fallback instead. Aliases pointing at this alias inherit its fallback
unless they set their own. Direct CPU/iGPU requests also run independently of media.
GPU requests with no usable fallback keep the existing queue/recovery behavior.
An edited or unassigned fallback that no longer resolves to a CPU-only or iGPU-only model is
never allowed to load on the reserved GPU.

The entire request is forwarded unchanged except for its resolved `model` field;
the fallback's context limit, modalities and tool support apply. Upstream errors
are returned normally. A request is assigned once: streaming responses never
switch models, and nothing is automatically retried or replayed. Subsequent
requests return to the primary after GPU restoration. The upstream response's
model identifies the model that actually answered.

Routing needs the model from the request body, regardless of JSON field order.
During media, small bodies are read in memory; bodies over 64 KiB spill to
private unnamed files in `request-spool/` beside the router configuration.
Queued GPU prompts stay there until admitted, and their files close on
cancellation. The configured maximum request size still applies. This avoids
holding every queued prompt in RAM while CPU requests continue to arrive.

### Inference request priority

`POST /v1/*` accepts the optional `x-inference-priority` header: `interactive`,
`normal`, or `background`. Missing or unrecognized values use `normal`. The
router consumes this header; it is not forwarded into prompts or backend input.

Admission is shared by the resolved model ID, including aliases and fallbacks.
Queued interactive requests precede normal requests, which precede background
requests; requests of the same priority remain FIFO. Running requests finish
normally, including their complete streaming response. Client retries are new
requests and rejoin the queue. Cancellation drops the waiting admission ticket.
Independent models keep separate queues. Capacity uses the model's configured
`parallel_slots` (one if unset), multiplied by its current running instance count (at least one).
Queued requests do not pin backend instances or GPU readers; media handoffs keep
exclusive access, and chats can overtake background requests waiting for media.
Large queued bodies use the existing private request spool rather than retaining
all prompts in memory. The queue itself stores only priority and request IDs.
