use crate::config::{
    BinaryPreset, ConfigError, JsonStore, ModelConfig, ModelState, WeightsFormat,
};
use crate::orchestrator::allocation::{gpus_used, plan_tensor_split};
use crate::orchestrator::eviction::{decide_eviction, EvictionAction};
use crate::process::manager::{ProcessManager, RequestGuard, SpawnError};
use crate::system::stats::{SystemStats, SystemTracker};
use crate::vram::estimator::VramEstimate;
use crate::vram::tracker::{GpuInfo, VRAMTracker};
use std::collections::HashMap;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use tokio::sync::Mutex;
use tracing::{error, info, warn};

/// Shared runtime data — single source of truth for models + gpus + presets.
#[derive(Default)]
pub struct AppData {
    pub models: HashMap<String, ModelConfig>,
    pub gpus: Vec<GpuInfo>,
    pub presets: HashMap<String, BinaryPreset>,
}

/// Thin handle injected into axum handlers. Cheap to clone.
pub type AppState = Arc<Orchestrator>;

/// Long-lived app controller. Built once in `lifecycle::run`, cloned into
/// handlers, and held by the reconcile task.
pub struct Orchestrator {
    pub data: Arc<Mutex<AppData>>,
    pub process_manager: Arc<Mutex<ProcessManager>>,
    pub vram_tracker: Arc<VRAMTracker>,
    pub system_tracker: Arc<SystemTracker>,
    /// Cross-model serialization for VRAM admission + spawn. See `do_load`.
    pub admission: Arc<Mutex<()>>,
    /// Per-model lock so concurrent `ensure_loaded("m")` calls collapse:
    /// the second waits, then sees `Running` and returns the existing port.
    pub load_guards: Arc<Mutex<HashMap<String, Arc<Mutex<()>>>>>,
    pub store: Arc<JsonStore<Vec<ModelConfig>>>,
    pub presets_store: Arc<JsonStore<Vec<BinaryPreset>>>,
    pub dirty: Arc<AtomicBool>,
    /// Set when presets change so reconcile persists presets.json.
    pub presets_dirty: Arc<AtomicBool>,
    pub server_port: u16,
}

impl Orchestrator {
    pub fn new(
        store: Arc<JsonStore<Vec<ModelConfig>>>,
        presets_store: Arc<JsonStore<Vec<BinaryPreset>>>,
        server_port: u16,
    ) -> Self {
        // One-shot migration of legacy `extra_args` into structured fields.
        // If any model changed, mark dirty so reconcile persists the
        // migrated shape on the next tick.
        let migrated_any = store.with_mut(|list| {
            let mut changed = false;
            for m in list.iter_mut() {
                if m.migrate_extra_args() {
                    changed = true;
                }
            }
            changed
        });

        let models: HashMap<String, ModelConfig> = store
            .snapshot()
            .into_iter()
            .map(|mut m| {
                // On restart, any model we *thought* was running is stale.
                if m.state == ModelState::Running || m.state == ModelState::Loading {
                    m.state = ModelState::Idle;
                    m.pid = None;
                }
                (m.id.clone(), m)
            })
            .collect();
        let presets: HashMap<String, BinaryPreset> = presets_store
            .snapshot()
            .into_iter()
            .map(|p| (p.id.clone(), p))
            .collect();

        Self {
            data: Arc::new(Mutex::new(AppData { models, gpus: Vec::new(), presets })),
            process_manager: Arc::new(Mutex::new(ProcessManager::default())),
            vram_tracker: Arc::new(VRAMTracker),
            system_tracker: Arc::new(SystemTracker::default()),
            admission: Arc::new(Mutex::new(())),
            load_guards: Arc::new(Mutex::new(HashMap::new())),
            store,
            presets_store,
            dirty: Arc::new(AtomicBool::new(migrated_any)),
            presets_dirty: Arc::new(AtomicBool::new(false)),
            server_port,
        }
    }

    // ----- CRUD -----

    /// Lists every configured model, sorted by `id` so the JSON response and
    /// dashboard have a stable deterministic order.
    pub async fn list_models(&self) -> Vec<ModelConfig> {
        let mut list: Vec<ModelConfig> =
            self.data.lock().await.models.values().cloned().collect();
        list.sort_by(|a, b| a.id.cmp(&b.id));
        list
    }

    pub fn system_stats(&self) -> SystemStats {
        self.system_tracker.sample()
    }

    /// Bump `last_used` on the named model. Called by the proxy after a
    /// successful request so the eviction heuristic sees live activity,
    /// not just the initial load timestamp.
    pub async fn mark_used(&self, id: &str) {
        let mut data = self.data.lock().await;
        if let Some(m) = data.models.get_mut(id) {
            m.last_used = Some(unix_now());
            self.dirty.store(true, Ordering::Relaxed);
        }
    }

    pub async fn list_gpus(&self) -> Vec<GpuInfo> {
        // Refresh on each call so polling clients see live numbers
        // (VRAM + GPU busy %) at whatever cadence they're polling at.
        // sysfs reads are a few small text files — cheap enough for 1s.
        let fresh = self.vram_tracker.refresh();
        let mut data = self.data.lock().await;
        data.gpus = fresh.clone();
        fresh
    }

    #[cfg(test)]
    pub async fn get_model(&self, id: &str) -> Option<ModelConfig> {
        self.data.lock().await.models.get(id).cloned()
    }

    // ----- Presets -----

    pub async fn list_presets(&self) -> Vec<BinaryPreset> {
        let mut v: Vec<BinaryPreset> = self.data.lock().await.presets.values().cloned().collect();
        v.sort_by(|a, b| a.id.cmp(&b.id));
        v
    }

    pub async fn add_preset(&self, preset: BinaryPreset) -> Result<(), MutationError> {
        let mut data = self.data.lock().await;
        if data.presets.contains_key(&preset.id) {
            return Err(MutationError::Conflict(preset.id));
        }
        data.presets.insert(preset.id.clone(), preset);
        self.presets_dirty.store(true, Ordering::Relaxed);
        Ok(())
    }

    pub async fn update_preset(&self, preset: BinaryPreset) -> Result<(), MutationError> {
        let mut data = self.data.lock().await;
        if !data.presets.contains_key(&preset.id) {
            return Err(MutationError::NotFound(preset.id));
        }
        data.presets.insert(preset.id.clone(), preset);
        self.presets_dirty.store(true, Ordering::Relaxed);
        Ok(())
    }

    pub async fn remove_preset(&self, id: &str) -> Result<(), MutationError> {
        let mut data = self.data.lock().await;
        if data.presets.remove(id).is_none() {
            return Err(MutationError::NotFound(id.into()));
        }
        self.presets_dirty.store(true, Ordering::Relaxed);
        Ok(())
    }

    pub async fn add_model(&self, model: ModelConfig) -> Result<(), MutationError> {
        let mut data = self.data.lock().await;
        if data.models.contains_key(&model.id) {
            return Err(MutationError::Conflict(model.id));
        }
        if let Some(ref did) = model.draft_model_id {
            validate_draft_reference(&data.models, &model.id, did)?;
        }
        data.models.insert(model.id.clone(), model);
        self.dirty.store(true, Ordering::Relaxed);
        Ok(())
    }

    pub async fn update_model(&self, new: ModelConfig) -> Result<(), MutationError> {
        let mut data = self.data.lock().await;
        if let Some(ref did) = new.draft_model_id {
            validate_draft_reference(&data.models, &new.id, did)?;
        }
        let existing = data.models.get_mut(&new.id)
            .ok_or_else(|| MutationError::NotFound(new.id.clone()))?;

        // Preserve runtime fields if the model is currently running/loading —
        // we don't want the form to wipe state by accident.
        let preserved_state = existing.state.clone();
        let preserved_pid = existing.pid;
        let preserved_last_used = existing.last_used;

        let mut updated = new;
        // Drop the estimate whenever anything that affects KV cache size or
        // weight layout changes — it'll be remeasured on next load.
        let kv_invalidated = existing.model_path != updated.model_path
            || existing.context != updated.context
            || existing.cache_type_k != updated.cache_type_k
            || existing.cache_type_v != updated.cache_type_v
            || existing.n_gpu_layers != updated.n_gpu_layers;
        if kv_invalidated {
            updated.estimated_vram = 0;
        } else {
            updated.estimated_vram = existing.estimated_vram;
        }
        updated.state = preserved_state;
        updated.pid = preserved_pid;
        updated.last_used = preserved_last_used;

        *existing = updated;
        self.dirty.store(true, Ordering::Relaxed);
        Ok(())
    }

    pub async fn remove_model(&self, id: &str) -> Result<(), MutationError> {
        {
            let data = self.data.lock().await;
            if !data.models.contains_key(id) {
                return Err(MutationError::NotFound(id.into()));
            }
            // Refuse to delete a model that any other model uses as a draft —
            // that would leave the referencing model trying to spawn with a
            // stale draft_model_id pointing at nothing.
            let referrers: Vec<String> = data
                .models
                .values()
                .filter(|other| other.draft_model_id.as_deref() == Some(id))
                .map(|other| other.id.clone())
                .collect();
            if !referrers.is_empty() {
                return Err(MutationError::DraftInUse {
                    id: id.into(),
                    targets: referrers,
                });
            }
        };
        // Try a graceful stop first so the child dies cleanly.
        if let Err(e) = self.stop_model_inner(id).await {
            warn!(model = id, error = %e, "failed to stop model during delete");
        }
        // Belt-and-braces: forget any pids still tracked (e.g. signal failed)
        // so kill_on_drop fires on orchestrator shutdown — no orphan processes.
        {
            let mut pm = self.process_manager.lock().await;
            for pid in pm.pids_for_model(id) {
                pm.forget(pid);
            }
        }
        self.data.lock().await.models.remove(id);
        // Drop the per-model load guard so the HashMap doesn't slowly grow
        // as configs come and go.
        self.load_guards.lock().await.remove(id);
        self.dirty.store(true, Ordering::Relaxed);
        Ok(())
    }

    // ----- load / stop -----

    /// Ensures a model has a running instance and returns a `RequestGuard`
    /// that tracks the in-flight request.
    ///
    /// Fast path: if an idle instance already exists, returns immediately.
    /// Scale-up path: if all instances are busy and VRAM permits, a new
    /// instance is spawned in the background; the current request is served
    /// by an existing (busy) instance — no cold-start penalty.
    /// First-spawn path: serialized via a per-model load_guard so concurrent
    /// callers collapse into one spawn. The work runs in a detached task so
    /// caller cancellation (client disconnect) can't leave the model stuck
    /// in Loading forever.
    pub async fn ensure_loaded(self: Arc<Self>, id: &str) -> Result<RequestGuard, LoadError> {
        // Fast path: idle instance available — no serialization needed.
        if let Some(guard) = self.process_manager.lock().await.acquire_idle_instance(id) {
            return Ok(guard);
        }

        // All instances busy (or none). Check if instances exist at all.
        let instance_count = self.process_manager.lock().await.instance_count(id);
        if instance_count > 0 {
            let (estimated_vram, free_vram, model_exists) = {
                let data = self.data.lock().await;
                let m = data.models.get(id);
                let est = m.map(|m| m.estimated_vram).unwrap_or(0);
                let free: u64 = data.gpus.iter().map(|g| g.free_vram()).sum();
                (est, free, m.is_some())
            };
            if !model_exists {
                return Err(LoadError::ModelNotFound(id.into()));
            }
            if estimated_vram > 0 && free_vram >= estimated_vram {
                // Kick off a background spawn; serve this request on an existing instance.
                let me = self.clone();
                let id_owned = id.to_string();
                tokio::spawn(async move { me.spawn_additional_instance(&id_owned).await });
            }
            if let Some(guard) = self.process_manager.lock().await.acquire_any_instance(id) {
                return Ok(guard);
            }
            // Instance pool drained between our checks (all died) — fall through to spawn.
        }

        // No instances: serialize the first spawn via a per-model load_guard.
        let load_guard = {
            let mut guards = self.load_guards.lock().await;
            guards.entry(id.to_string()).or_insert_with(|| Arc::new(Mutex::new(()))).clone()
        };

        let id_owned = id.to_string();
        let me = self.clone();
        let handle = tokio::spawn(async move {
            let _lock = load_guard.lock().await;

            // Re-check after acquiring the lock — another task may have spawned already.
            if let Some(g) = me.process_manager.lock().await.acquire_idle_instance(&id_owned) {
                return Ok(g);
            }

            {
                let data = me.data.lock().await;
                if !data.models.contains_key(&id_owned) {
                    return Err(LoadError::ModelNotFound(id_owned.clone()));
                }
            }

            // Claim Loading state.
            {
                let mut data = me.data.lock().await;
                if let Some(m) = data.models.get_mut(&id_owned) {
                    m.state = ModelState::Loading;
                }
            }

            match me.do_load(&id_owned).await {
                Ok(guard) => Ok(guard),
                Err(e) => {
                    let mut data = me.data.lock().await;
                    if let Some(m) = data.models.get_mut(&id_owned) {
                        m.state = ModelState::Error(e.to_string());
                        m.pid = None;
                    }
                    me.dirty.store(true, Ordering::Relaxed);
                    Err(e)
                }
            }
        });

        match handle.await {
            Ok(result) => result,
            Err(join_err) => {
                let msg = format!("load task panicked: {join_err}");
                {
                    let mut data = self.data.lock().await;
                    if let Some(m) = data.models.get_mut(id) {
                        m.state = ModelState::Error(msg.clone());
                        m.pid = None;
                    }
                }
                self.dirty.store(true, Ordering::Relaxed);
                Err(LoadError::SpawnFailed(
                    crate::process::manager::SpawnError::HealthCheckFailed(
                        crate::process::manager::HealthCheckError::ChildExited(None, msg),
                    ),
                ))
            }
        }
    }

    /// Inner load path.
    ///
    /// The admission lock serializes VRAM accounting + eviction + fork/exec
    /// — the steps where two concurrent loads could step on each other's
    /// VRAM budget. We drop admission before waiting for health; that way
    /// a 180-second health poll can't block a second model from starting
    /// on a different GPU.
    async fn do_load(&self, id: &str) -> Result<RequestGuard, LoadError> {
        // Accumulated weight file sizes (target + draft if present). Set
        // inside the admission block during GGUF reads; used after the health
        // check to recompute estimated_vram from the measured KV bytes.
        let mut weight_file_size: u64 = 0;
        let (mut pending, pid, port) = {
            let _admit = self.admission.lock().await;

            // Refresh VRAM into AppData.
            let gpus = self.vram_tracker.refresh();
            {
                let mut data = self.data.lock().await;
                data.gpus = gpus.clone();
            }

            // Snapshot the model + resolve preset → binary path + draft.
            let (mut model, draft) = {
                let data = self.data.lock().await;
                let mut m = data
                    .models
                    .get(id)
                    .cloned()
                    .ok_or_else(|| LoadError::ModelNotFound(id.into()))?;
                if let Some(ref preset_id) = m.binary_preset {
                    match data.presets.get(preset_id) {
                        Some(p) => m.binary = p.binary.clone(),
                        None => {
                            return Err(LoadError::PresetNotFound(preset_id.clone()));
                        }
                    }
                }
                // Resolve the draft reference now so any missing/
                // role-mismatched draft surfaces as a load-time error
                // rather than a cryptic spawn failure.
                let draft = if let Some(ref did) = m.draft_model_id {
                    let d = data.models.get(did).cloned().ok_or_else(|| {
                        LoadError::DraftNotFound {
                            id: did.clone(),
                            target: id.into(),
                        }
                    })?;
                    Some(d)
                } else {
                    None
                };
                (m, draft)
            };

            if model.weights_format == WeightsFormat::Gguf {
                // Honour the model's configured cache_type_{k,v} so the
                // KV-cache portion of the estimate matches what the
                // backend will actually allocate at run time.
                use crate::config::CacheType;
                use crate::vram::estimator::{GgufInfo, KvPerElement};
                let kv_bytes = KvPerElement::from_types(
                    model.cache_type_k.unwrap_or(CacheType::F16),
                    model.cache_type_v.unwrap_or(CacheType::F16),
                );
                match GgufInfo::read(&model.model_path) {
                    Ok(info) => {
                        weight_file_size = info.file_size;
                        let kv = info.kv_cache_bytes(model.context, kv_bytes);
                        let est = VramEstimate::compute(info.file_size, kv);
                        model.estimated_vram = est.total_vram;
                    }
                    Err(e) => warn!(model = id, error = %e, "gguf parse failed; loading without estimate"),
                }
                // Fold the draft's VRAM (weights + KV cache at its own
                // context, with its own KV quant) into the target's
                // estimate. Admission, eviction, and allocation all key
                // off this single number — without the fold, a 256k
                // target + a 2B draft would slip through admission
                // based on target-only VRAM and then OOM at spawn.
                if let Some(ref d) = draft {
                    let d_kv = KvPerElement::from_types(
                        d.cache_type_k.unwrap_or(CacheType::F16),
                        d.cache_type_v.unwrap_or(CacheType::F16),
                    );
                    match GgufInfo::read(&d.model_path) {
                        Ok(info) => {
                            weight_file_size += info.file_size;
                            let kv = info.kv_cache_bytes(d.context, d_kv);
                            let est = VramEstimate::compute(info.file_size, kv);
                            model.estimated_vram = model.estimated_vram
                                .saturating_add(est.total_vram);
                        }
                        Err(e) => warn!(
                            target = id, draft = d.id,
                            error = %e,
                            "draft gguf parse failed; loading without draft VRAM estimate",
                        ),
                    }
                }
                let mut data = self.data.lock().await;
                if let Some(m) = data.models.get_mut(id) {
                    m.estimated_vram = model.estimated_vram;
                }
            }

            // Evict if needed.
            let free: u64 = gpus.iter().map(|g| g.free_vram()).sum();
            if model.estimated_vram > free {
                let snapshot = self.data.lock().await.models.clone();
                for EvictionAction::Evict(victim) in
                    decide_eviction(&snapshot, &gpus, model.estimated_vram)
                {
                    info!(victim = victim, "evicting to make room");
                    if let Err(e) = self.stop_model_inner(&victim).await {
                        warn!(model = victim, error = %e, "eviction stop failed");
                    }
                }
                // Re-read VRAM after eviction so the allocator sees the freed space.
                let gpus_after = self.vram_tracker.refresh();
                self.data.lock().await.gpus = gpus_after;
            }

            // Auto-allocate across the smallest viable GPU subset, unless the
            // user pinned an explicit tensor_split.
            if model.weights_format == WeightsFormat::Gguf && model.tensor_split.is_none() {
                let snapshot = self.data.lock().await.gpus.clone();
                if snapshot.len() > 1 && model.estimated_vram > 0 {
                    if let Some(split) = plan_tensor_split(&snapshot, model.estimated_vram) {
                        info!(
                            model = id,
                            gpus_used = gpus_used(&split),
                            tensor_split = split,
                            "auto-allocated across minimum GPU subset"
                        );
                        model.tensor_split = Some(split);
                    }
                }
            }

            // Fork + exec (fast). Holding the admission lock across this is
            // still cheap — just until the process exists on disk.
            let pending = self
                .process_manager
                .lock()
                .await
                .spawn_child(&model, draft.as_ref())
                .map_err(LoadError::SpawnFailed)?;
            let pid = pending.pid;
            let port = pending.port;
            (pending, pid, port)
        };
        // Admission and process_manager mutexes are dropped here; other
        // loads can proceed even while we're still waiting for `pending`
        // to report healthy (up to 180 seconds).

        match pending.wait_for_health(std::time::Duration::from_secs(180)).await {
            Ok(kv_bytes) => {
                // register() returns the guard for this request (active starts at 1).
                let guard = self.process_manager.lock().await.register(pending);
                // pm lock dropped before acquiring data lock.
                {
                    let mut data = self.data.lock().await;
                    if let Some(m) = data.models.get_mut(id) {
                        m.state = ModelState::Running;
                        m.pid = Some(pid);
                        m.last_used = Some(unix_now());
                        if kv_bytes > 0 && weight_file_size > 0 {
                            m.estimated_vram =
                                ((weight_file_size + kv_bytes) as f64 * 1.1) as u64;
                            info!(
                                model = id,
                                kv_mib = kv_bytes / 1024 / 1024,
                                total_mib = m.estimated_vram / 1024 / 1024,
                                "updated VRAM estimate from llama.cpp startup logs",
                            );
                        }
                    }
                }
                self.dirty.store(true, Ordering::Relaxed);
                info!(pid, port, model = id, "inference server ready");
                Ok(guard)
            }
            Err(e) => {
                error!(pid, port, error = %e, "health check failed; spawn cancelled");
                Err(LoadError::SpawnFailed(crate::process::manager::SpawnError::HealthCheckFailed(e)))
            }
        }
    }

    /// Spawn an additional instance of `id` in the background. Called when all
    /// existing instances are busy and VRAM headroom permits a new one.
    /// Serialized via the admission lock; aborts silently if an idle instance
    /// appears by the time admission is acquired (another spawn beat us to it).
    async fn spawn_additional_instance(&self, id: &str) {
        let (mut pending, pid, port) = {
            let _admit = self.admission.lock().await;

            // Re-check: did an idle instance appear while we waited?
            if !self.process_manager.lock().await.all_busy(id) {
                return;
            }

            let gpus = self.vram_tracker.refresh();
            { self.data.lock().await.gpus = gpus.clone(); }

            let (mut model, draft) = {
                let data = self.data.lock().await;
                let m = match data.models.get(id) {
                    Some(m) => m.clone(),
                    None => return,
                };
                let mut model = m;
                if let Some(ref preset_id) = model.binary_preset {
                    match data.presets.get(preset_id) {
                        Some(p) => model.binary = p.binary.clone(),
                        None => {
                            warn!(model = id, preset = preset_id, "preset not found for extra instance");
                            return;
                        }
                    }
                }
                let draft = model.draft_model_id.as_ref()
                    .and_then(|did| data.models.get(did).cloned());
                (model, draft)
            };

            let free: u64 = gpus.iter().map(|g| g.free_vram()).sum();
            if model.estimated_vram == 0 || free < model.estimated_vram {
                return;
            }

            if model.weights_format == WeightsFormat::Gguf && model.tensor_split.is_none() {
                let snapshot = self.data.lock().await.gpus.clone();
                if snapshot.len() > 1 && model.estimated_vram > 0 {
                    if let Some(split) = plan_tensor_split(&snapshot, model.estimated_vram) {
                        model.tensor_split = Some(split);
                    }
                }
            }

            let pending = match self.process_manager.lock().await.spawn_child(&model, draft.as_ref()) {
                Ok(p) => p,
                Err(e) => {
                    warn!(model = id, error = %e, "failed to spawn extra instance");
                    return;
                }
            };
            let pid = pending.pid;
            let port = pending.port;
            (pending, pid, port)
        };
        // Admission lock dropped; health check runs without blocking other spawns.

        match pending.wait_for_health(std::time::Duration::from_secs(180)).await {
            Ok(_) => {
                // Drop the guard immediately — no user request is attached.
                drop(self.process_manager.lock().await.register(pending));
                self.dirty.store(true, Ordering::Relaxed);
                info!(pid, port, model = id, "extra instance ready");
            }
            Err(e) => {
                warn!(pid, port, model = id, error = %e, "extra instance health check failed");
            }
        }
    }

    pub async fn stop_model(&self, id: &str) -> Result<(), StopError> {
        self.stop_model_inner(id).await
    }

    /// Stop all instances of `id`. Safe to call from inside `do_load` during
    /// eviction (does not acquire the admission lock).
    async fn stop_model_inner(&self, id: &str) -> Result<(), StopError> {
        {
            let data = self.data.lock().await;
            data.models.get(id).ok_or_else(|| StopError::ModelNotFound(id.into()))?;
        }
        let pids = self.process_manager.lock().await.pids_for_model(id);
        for pid in pids {
            self.process_manager.lock().await.stop(pid).await;
        }
        {
            let mut data = self.data.lock().await;
            if let Some(m) = data.models.get_mut(id) {
                m.state = ModelState::Idle;
                m.pid = None;
            }
        }
        self.dirty.store(true, Ordering::Relaxed);
        Ok(())
    }

    // ----- reconcile -----

    /// Refresh VRAM from sysfs, detect dead processes, persist if dirty.
    /// Called on a 5s timer by `lifecycle::run`.
    pub async fn reconcile(&self) {
        let gpus = self.vram_tracker.refresh();
        { self.data.lock().await.gpus = gpus; }

        // Ask ProcessManager which instances have died since the last tick.
        let dead = self.process_manager.lock().await.dead_instances();

        if !dead.is_empty() {
            let mut data = self.data.lock().await;
            let mut pm = self.process_manager.lock().await;
            for (model_id, pid) in &dead {
                warn!(model = model_id, pid, "process died");
                pm.forget(*pid);
                // Only mark the model Error/Idle when its last instance is gone.
                if pm.instance_count(model_id) == 0 {
                    if let Some(m) = data.models.get_mut(model_id) {
                        m.state = ModelState::Error(format!("process {} died", pid));
                        m.pid = None;
                    }
                }
            }
            self.dirty.store(true, Ordering::Relaxed);
        }

        if self.dirty.swap(false, Ordering::Relaxed) {
            let snapshot: Vec<ModelConfig> =
                self.data.lock().await.models.values().cloned().collect();
            self.store.replace(snapshot);
            if let Err(e) = self.store.save() {
                error!(error = %e, "failed to persist models.json");
                self.dirty.store(true, Ordering::Relaxed);
            }
        }

        if self.presets_dirty.swap(false, Ordering::Relaxed) {
            let snapshot: Vec<BinaryPreset> =
                self.data.lock().await.presets.values().cloned().collect();
            self.presets_store.replace(snapshot);
            if let Err(e) = self.presets_store.save() {
                error!(error = %e, "failed to persist presets.json");
                self.presets_dirty.store(true, Ordering::Relaxed);
            }
        }
    }
}

fn unix_now() -> f64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_secs_f64())
        .unwrap_or(0.0)
}

#[derive(Debug, thiserror::Error)]
pub enum LoadError {
    #[error("model '{0}' not found")]
    ModelNotFound(String),

    #[error("binary preset '{0}' not found — edit the model and pick an existing preset or a custom path")]
    PresetNotFound(String),

    #[error("model references draft '{id}', but no model '{id}' exists (target: '{target}')")]
    DraftNotFound { id: String, target: String },

    #[error("spawn failed: {0}")]
    SpawnFailed(SpawnError),
}

#[derive(Debug, thiserror::Error)]
pub enum StopError {
    #[error("model '{0}' not found")]
    ModelNotFound(String),
}

#[derive(Debug, thiserror::Error)]
pub enum MutationError {
    #[error("model '{0}' not found")]
    NotFound(String),

    #[error("model '{0}' already exists")]
    Conflict(String),

    #[error("invalid config: {0}")]
    InvalidConfig(#[from] ConfigError),

    #[error("cannot delete '{id}': used as a draft by {}", targets.join(", "))]
    DraftInUse { id: String, targets: Vec<String> },
}

/// Validate a target's `draft_model_id` against the current model table.
///
/// Checked conditions:
/// - The referenced id exists in the table.
fn validate_draft_reference(
    models: &HashMap<String, ModelConfig>,
    self_id: &str,
    draft_id: &str,
) -> Result<(), MutationError> {
    if draft_id == self_id {
        return Err(MutationError::InvalidConfig(ConfigError::DraftSelfReference));
    }
    if !models.contains_key(draft_id) {
        return Err(MutationError::InvalidConfig(ConfigError::DraftNotFound {
            id: draft_id.to_string(),
        }));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;
    use tempfile::TempDir;

    fn orch(tmp: &TempDir) -> Arc<Orchestrator> {
        let store = Arc::new(JsonStore::<Vec<ModelConfig>>::new(tmp.path().join("models.json")));
        let presets = Arc::new(JsonStore::<Vec<BinaryPreset>>::new(tmp.path().join("presets.json")));
        Arc::new(Orchestrator::new(store, presets, 8080))
    }

    fn model(id: &str) -> ModelConfig {
        ModelConfig {
            id: id.into(),
            name: id.into(),
            binary: PathBuf::from("/bin/true"),
            model_path: PathBuf::from("/tmp/m.gguf"),
            ..ModelConfig::default()
        }
    }

    #[tokio::test]
    async fn add_list_remove_roundtrip() {
        let tmp = tempfile::tempdir().unwrap();
        let o = orch(&tmp);
        o.add_model(model("a")).await.unwrap();
        assert_eq!(o.list_models().await.len(), 1);
        o.remove_model("a").await.unwrap();
        assert_eq!(o.list_models().await.len(), 0);
    }

    #[tokio::test]
    async fn add_model_duplicate_id_errors() {
        let tmp = tempfile::tempdir().unwrap();
        let o = orch(&tmp);
        o.add_model(model("a")).await.unwrap();
        let err = o.add_model(model("a")).await.unwrap_err();
        assert!(matches!(err, MutationError::Conflict(_)));
    }

    #[tokio::test]
    async fn mark_used_updates_last_used_and_marks_dirty() {
        let tmp = tempfile::tempdir().unwrap();
        let o = orch(&tmp);
        o.add_model(model("a")).await.unwrap();
        // last_used starts as None.
        assert!(o.get_model("a").await.unwrap().last_used.is_none());

        o.dirty.store(false, Ordering::Relaxed);
        o.mark_used("a").await;
        assert!(o.get_model("a").await.unwrap().last_used.is_some());
        assert!(o.dirty.load(Ordering::Relaxed), "mark_used must mark dirty so reconcile persists");
    }

    #[tokio::test]
    async fn mark_used_unknown_model_is_a_noop() {
        let tmp = tempfile::tempdir().unwrap();
        let o = orch(&tmp);
        o.dirty.store(false, Ordering::Relaxed);
        o.mark_used("does-not-exist").await;
        assert!(!o.dirty.load(Ordering::Relaxed));
    }

    #[tokio::test]
    async fn remove_model_clears_per_model_load_guard() {
        let tmp = tempfile::tempdir().unwrap();
        let o = orch(&tmp);
        o.add_model(model("a")).await.unwrap();

        // Prime the load-guard entry (ensure_loaded would insert one).
        o.load_guards.lock().await
            .entry("a".into())
            .or_insert_with(|| Arc::new(Mutex::new(())));
        assert!(o.load_guards.lock().await.contains_key("a"));

        o.remove_model("a").await.unwrap();
        assert!(!o.load_guards.lock().await.contains_key("a"),
            "load_guards must drop entries for removed models so the map doesn't grow forever");
    }

    #[tokio::test]
    async fn ensure_loaded_state_transition_survives_caller_cancellation() {
        // Regression: when the HTTP handler's future is cancelled
        // mid-load (client disconnect, tab reload, etc.), the model
        // must still transition out of `Loading` — either to `Running`
        // or to `Error` — so users don't see a forever-loading row.
        let tmp = tempfile::tempdir().unwrap();
        let o = orch(&tmp);
        o.add_model(model("a")).await.unwrap();

        // Start the load in a spawned task we can abort below.
        // /bin/true exits immediately, so `wait_for_health_or_exit`
        // observes ChildExited and the detached task sets Error.
        let ensure = {
            let o = o.clone();
            tokio::spawn(async move { o.ensure_loaded("a").await })
        };
        // Give the task a moment to hit `tokio::spawn(...)` inside
        // ensure_loaded, then simulate the caller going away.
        tokio::time::sleep(std::time::Duration::from_millis(20)).await;
        ensure.abort();

        // Wait up to ~3s for the detached load task to finish and
        // write the final state.
        for _ in 0..60 {
            tokio::time::sleep(std::time::Duration::from_millis(50)).await;
            let state = o.get_model("a").await.unwrap().state;
            if matches!(state, ModelState::Error(_)) || state == ModelState::Idle {
                return;
            }
        }
        let final_state = o.get_model("a").await.unwrap().state;
        panic!("state never left Loading after caller cancellation: {final_state:?}");
    }

    #[tokio::test]
    async fn list_models_returns_sorted_by_id() {
        let tmp = tempfile::tempdir().unwrap();
        let o = orch(&tmp);
        o.add_model(model("charlie")).await.unwrap();
        o.add_model(model("alpha")).await.unwrap();
        o.add_model(model("bravo")).await.unwrap();
        let ids: Vec<String> = o.list_models().await.into_iter().map(|m| m.id).collect();
        assert_eq!(ids, vec!["alpha", "bravo", "charlie"]);
    }

    #[tokio::test]
    async fn update_model_clears_estimate_when_path_changes() {
        let tmp = tempfile::tempdir().unwrap();
        let o = orch(&tmp);
        let mut m = model("a");
        m.estimated_vram = 7_000_000_000;
        o.add_model(m.clone()).await.unwrap();
        // set vram directly to simulate a load
        {
            let mut d = o.data.lock().await;
            d.models.get_mut("a").unwrap().estimated_vram = 7_000_000_000;
        }
        let mut updated = m.clone();
        updated.model_path = PathBuf::from("/tmp/m2.gguf");
        o.update_model(updated).await.unwrap();
        assert_eq!(o.get_model("a").await.unwrap().estimated_vram, 0);
    }

    #[tokio::test]
    async fn restart_clears_stale_running_state() {
        // Write a models.json with a model in Running state; rebuilding the
        // orchestrator must reset it to Idle (the process is obviously not
        // alive across restarts).
        let tmp = tempfile::tempdir().unwrap();
        let path = tmp.path().join("models.json");
        let mut m = model("a");
        m.state = ModelState::Running;
        m.pid = Some(12345);
        std::fs::write(&path, serde_json::to_string(&vec![m]).unwrap()).unwrap();

        let store = Arc::new(JsonStore::<Vec<ModelConfig>>::new(path));
        let presets = Arc::new(JsonStore::<Vec<BinaryPreset>>::new(tmp.path().join("presets.json")));
        let o = Arc::new(Orchestrator::new(store, presets, 8080));
        let loaded = o.get_model("a").await.unwrap();
        assert_eq!(loaded.state, ModelState::Idle);
        assert_eq!(loaded.pid, None);
    }

    #[tokio::test]
    async fn reconcile_marks_dead_process_as_error() {
        let tmp = tempfile::tempdir().unwrap();
        let o = orch(&tmp);
        let mut m = model("a");
        m.state = ModelState::Running;
        m.pid = Some(999_999);
        o.add_model(m).await.unwrap();

        // Register a synthetic instance with a definitely-dead pid.
        o.process_manager.lock().await.register_test_instance("a", 999_999, 9000);

        o.reconcile().await;
        let after = o.get_model("a").await.unwrap();
        match after.state {
            ModelState::Error(msg) => assert!(msg.contains("999999") || msg.contains("died")),
            other => panic!("expected Error, got {:?}", other),
        }
        assert_eq!(after.pid, None);
    }

    // ----- Speculative decoding -----

    fn draft_model(id: &str) -> ModelConfig {
        ModelConfig {
            id: id.into(),
            name: id.into(),
            model_path: PathBuf::from("/tmp/draft.gguf"),
            context: 16384,
            device: Some("Vulkan1".into()),
            ..ModelConfig::default()
        }
    }

    #[tokio::test]
    async fn add_model_with_draft_reference() {
        let tmp = tempfile::tempdir().unwrap();
        let o = orch(&tmp);
        o.add_model(draft_model("d")).await.unwrap();
        let mut t = model("t");
        t.draft_model_id = Some("d".into());
        t.draft_max = Some(16);
        t.ctx_checkpoints = Some(4);
        o.add_model(t).await.unwrap();
        assert_eq!(o.list_models().await.len(), 2);
    }

    #[tokio::test]
    async fn add_model_referencing_nonexistent_draft_errors() {
        let tmp = tempfile::tempdir().unwrap();
        let o = orch(&tmp);
        let mut t = model("t");
        t.draft_model_id = Some("missing".into());
        let err = o.add_model(t).await.unwrap_err();
        assert!(matches!(
            err,
            MutationError::InvalidConfig(ConfigError::DraftNotFound { .. }),
        ));
    }

    #[tokio::test]
    async fn model_cannot_reference_itself_as_draft() {
        let tmp = tempfile::tempdir().unwrap();
        let o = orch(&tmp);
        let mut t = model("t");
        t.draft_model_id = Some("t".into());
        let err = o.add_model(t).await.unwrap_err();
        assert!(matches!(
            err,
            MutationError::InvalidConfig(ConfigError::DraftSelfReference),
        ));
    }

    #[tokio::test]
    async fn remove_model_in_use_as_draft_errors_with_referrers() {
        let tmp = tempfile::tempdir().unwrap();
        let o = orch(&tmp);
        o.add_model(draft_model("d")).await.unwrap();
        let mut t1 = model("t1");
        t1.draft_model_id = Some("d".into());
        let mut t2 = model("t2");
        t2.draft_model_id = Some("d".into());
        o.add_model(t1).await.unwrap();
        o.add_model(t2).await.unwrap();

        let err = o.remove_model("d").await.unwrap_err();
        match err {
            MutationError::DraftInUse { id, mut targets } => {
                assert_eq!(id, "d");
                targets.sort();
                assert_eq!(targets, vec!["t1".to_string(), "t2".to_string()]);
            }
            other => panic!("expected DraftInUse, got {other:?}"),
        }
    }

    #[tokio::test]
    async fn remove_model_after_unreferencing_succeeds() {
        let tmp = tempfile::tempdir().unwrap();
        let o = orch(&tmp);
        o.add_model(draft_model("d")).await.unwrap();
        let mut t = model("t");
        t.draft_model_id = Some("d".into());
        o.add_model(t.clone()).await.unwrap();

        // Clear the reference, then delete.
        t.draft_model_id = None;
        o.update_model(t).await.unwrap();
        assert!(o.remove_model("d").await.is_ok());
    }

    #[tokio::test]
    async fn reconcile_persists_only_when_dirty() {
        let tmp = tempfile::tempdir().unwrap();
        let path = tmp.path().join("models.json");
        let store = Arc::new(JsonStore::<Vec<ModelConfig>>::new(path.clone()));
        let presets = Arc::new(JsonStore::<Vec<BinaryPreset>>::new(tmp.path().join("presets.json")));
        let o = Arc::new(Orchestrator::new(store, presets, 8080));

        // First reconcile: nothing dirty, no file written.
        o.reconcile().await;
        assert!(!path.exists());

        // Add a model → dirty → reconcile writes the file.
        o.add_model(model("a")).await.unwrap();
        o.reconcile().await;
        assert!(path.exists());
        let first_mtime = std::fs::metadata(&path).unwrap().modified().unwrap();

        // Second idle reconcile: no rewrite.
        std::thread::sleep(std::time::Duration::from_millis(10));
        o.reconcile().await;
        let second_mtime = std::fs::metadata(&path).unwrap().modified().unwrap();
        assert_eq!(first_mtime, second_mtime);
    }
}
