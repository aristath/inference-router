## Notes

### Orchestrator Module Architecture (`src/orchestrator/`)

**Decision/execution split pattern**: `allocation.rs` and `eviction.rs` are pure functions (no shared state, no locking). `engine.rs` holds all mutable state and locking.

**Two-level locking in Orchestrator**:
- Global `admission: Arc<Mutex<()>>` serializes VRAM accounting + eviction + fork/exec. Held only briefly — dropped before health polling (up to 180s) so other loads can proceed in parallel.
- Per-model `load_guards: Arc<Mutex<HashMap<String, Arc<Mutex<()>>>>>` coalesce concurrent `ensure_loaded("same-model")` calls. Guard acquired *inside* the spawned task so task cancellation can't leak the lock.

**Detached load tasks**: `ensure_loaded` spawns the load into a detached `tokio::spawn`. If the HTTP handler's future is cancelled (client disconnect), the state still transitions to `Running` or `Error`. Prevents forever-Loading models.

**Dirty-flag persistence**: `reconcile()` (5s tick) only writes to disk when `dirty` is true. Uses `AtomicBool::swap(false)` for lock-free check-and-clear. On persistence failure, re-sets dirty for retry.

**Draft/target speculative decoding**: The load path folds the draft model's VRAM (weights + KV cache at its own context) into the target's estimate. Admission, eviction, and allocation all key off this single number.

**Eviction score formula**: `ln(idle + 1) + 1 / log2(gib + 1)`. Higher = evict first. Biases toward long-idle, small models. Never-used models get INFINITY score.

**Allocation strategy**: Greedy best-fit. Sorts GPUs by free VRAM descending, picks smallest viable subset. 5% headroom multiplier (`HEADROOM = 1.05`). Emits positional `--tensor-split` string in original device order.

### Key external dependencies (from imports in engine.rs)
- `crate::config` — ModelConfig, ModelRole, ModelState, BinaryPreset, JsonStore, WeightsFormat
- `crate::process::manager` — ProcessManager, SpawnError
- `crate::system::stats` — SystemStats, SystemTracker
- `crate::vram::estimator` — VramEstimate
- `crate::vram::tracker` — GpuInfo, VRAMTracker
