# inference-router

A VRAM-aware, OpenAI-compatible HTTP proxy for [llama.cpp](https://github.com/ggerganov/llama.cpp) (and vLLM-style safetensors backends). Runs one backend process per loaded model, routes `/v1/*` requests to the right process by the `model` field in the JSON body, and evicts models under VRAM pressure.

Built for a single-user, localhost, multi-GPU AMD workstation (3x Radeon AI PRO R9700). No auth, no rate limiting, no multi-tenant isolation.

## What it does

- **One endpoint, many models.** Clients (Claude Code, aider, continue.dev, raw curl) POST to `http://localhost:8080/v1/chat/completions` with `"model": "<id>"`. The router spawns the backend on demand and proxies the request byte-for-byte — so the full OpenAI surface works, including streaming, tools, and anything llama.cpp adds tomorrow.
- **VRAM-aware admission.** Before spawning, the orchestrator reads GGUF metadata, estimates VRAM, checks free VRAM across all GPUs, and evicts idle models if needed. Eviction prefers long-idle and small models.
- **Smart GPU allocation.** Picks the minimum GPU subset that fits and passes `--tensor-split` explicitly. Never occupies a GPU it doesn't need.
- **Browser dashboard.** Single-page UI at `/` for CRUD on models + binary presets, manual load/stop, and live GPU / CPU / RAM stats. Askama HTML templates, vanilla JS, 500 ms poll.
- **Persistence.** Model and preset definitions live in `~/.config/inference-router/{models.json,presets.json}`. Writes are dirty-flag gated and flushed by the reconcile loop (5 s cadence).

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
- `INFERENCE_ROUTER_MAX_INSTANCES_PER_MODEL` — maximum concurrent backend processes per model. Defaults to `1`; raise it if you explicitly want replica scale-out while requests are busy.
- `INFERENCE_ROUTER_VRAM_WAIT_MS` — how long a load waits for active requests to release VRAM before failing. Defaults to `300000` (5 minutes); set `0` to fail immediately.
- `INFERENCE_ROUTER_BACKEND_PORT_RANGE` — optional inclusive range for backend processes, e.g. `9100-9199`. Defaults to OS-assigned ephemeral ports.
- `INFERENCE_ROUTER_LOOP_*` / `INFERENCE_ROUTER_TOOL_LOOP_*` — initial loop-guard defaults when `settings.json` does not exist yet. After that, use Settings → Loop guards in the UI.

The port is `8080`, hardcoded in `AppConfig::default()` (see `src/lifecycle.rs`).

## Configuration

State files under `~/.config/inference-router/`:

- **`presets.json`** — named binary paths. Lets you rebuild llama.cpp once and have every model pick up the new binary.
- **`settings.json`** — server-level app settings, including streaming and cross-turn loop guard controls exposed in the Settings modal.
- **`models.json`** — one entry per model. Key fields:
  - `binary_preset` (optional): preset id to resolve `binary` from at spawn time.
  - `profile` (optional): workspace label such as `coding`, `long-context`, or `vision` for dashboard filtering.
  - `binary`, `model_path`, `port`, `context`: mandatory.
  - `weights_format`: `gguf` (llama.cpp argv style) or `safetensors` (vLLM style).
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
