# Strata backends

Build Strata and its Python environment separately. Register `scripts/strata-server`
as a CUDA binary preset, then add a GGUF model through the normal model API with:

```json
{
  "id": "flash-next",
  "name": "Flash Next",
  "weights_format": "gguf",
  "binary_preset": "strata-cuda",
  "model_path": "/absolute/model-00001-of-00002.gguf",
  "context": 262144,
  "strata": {
    "config": "/absolute/strata-flash-next.json",
    "vram_mib": 23200,
    "gpu": 0
  }
}
```

Use a measured VRAM reservation that includes the model, active KV, expert cache,
scratch and vision encoder. The router admits against that CUDA device's available
budget, evicts idle models as usual, and never runs llama.cpp fitting against
Strata. It waits for `/health` to report loaded before routing requests. Existing
models without `strata` retain their current launch and fitting behavior.

Set the model's `parallel_slots` to the native JSON's `parallel` value. With
multiple Strata slots, one per replica is reserved against requests carrying
`x-inference-priority: background`, so long background generations cannot fill
every slot before a chat arrives. Interactive/normal work can use the full
capacity. Running requests continue; further background requests wait. New
background requests also wait while any foreground request remains active, so
successive large background prefills cannot keep interrupting its generation.
Foreground work arriving during a running background stream still overlaps it.
The existing request-boundary priority behavior is unchanged for other backends.

The native JSON contains Strata's `exe`, `args`, `cwd`, `tokenizer`, `env`, and
optional `vision` configuration. Use absolute paths. `cwd` contains the Strata
checkout, `serve/server.py`, and `.venv/bin/python` (or set `python` explicitly).
The launcher checks that the GGUF and total context match router metadata, copies
the router's sampling defaults, sets the concrete model name and assigned port,
and binds to localhost. Per-request sampling still takes precedence. Context,
expert placement, KV precision, MTP, parallelism, template and vision settings live
in the native JSON; llama.cpp-only model options and `extra_args` are not forwarded.
The dashboard preserves this API-configured `strata` block when editing a model.

The launcher execs into Strata in the router's process group. Engine and vision
children therefore stop with the backend; no extra service, fixed port or runtime
JSON files are needed. Performance uses Strata's response `timings` through the
existing throughput collector. Template overrides for vLLM do not replace Strata's
configured tokenizer template.
