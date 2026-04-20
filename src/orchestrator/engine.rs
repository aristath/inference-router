use crate::config::{BinaryPreset, JsonStore, ModelConfig, ModelState, WeightsFormat};
use crate::orchestrator::allocation::{gpus_used, plan_tensor_split};
use crate::orchestrator::eviction::{decide_eviction, EvictionAction};
use crate::process::manager::{ProcessManager, SpawnError};
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
        if let Some(other) = data.models.values().find(|m| m.port == model.port) {
            return Err(MutationError::PortConflict(model.port, other.id.clone()));
        }
        data.models.insert(model.id.clone(), model);
        self.dirty.store(true, Ordering::Relaxed);
        Ok(())
    }

    pub async fn update_model(&self, new: ModelConfig) -> Result<(), MutationError> {
        let mut data = self.data.lock().await;
        if let Some(other) = data
            .models
            .values()
            .find(|m| m.port == new.port && m.id != new.id)
        {
            return Err(MutationError::PortConflict(new.port, other.id.clone()));
        }
        let existing = data.models.get_mut(&new.id)
            .ok_or_else(|| MutationError::NotFound(new.id.clone()))?;

        // Preserve runtime fields if the model is currently running/loading —
        // we don't want the form to wipe state by accident.
        let preserved_state = existing.state.clone();
        let preserved_pid = existing.pid;
        let preserved_last_used = existing.last_used;

        let mut updated = new;
        // If a running model's model_path or context changed, drop the estimate;
        // it'll recompute on next load.
        if existing.model_path != updated.model_path || existing.context != updated.context {
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
        let pid = {
            let data = self.data.lock().await;
            let m = data.models.get(id).ok_or_else(|| MutationError::NotFound(id.into()))?;
            m.pid
        };
        // Try a graceful stop first so the child dies cleanly.
        if let Err(e) = self.stop_model_inner(id).await {
            warn!(model = id, error = %e, "failed to stop model during delete");
        }
        // Belt-and-braces: if the graceful stop left a tracked pid behind
        // (e.g. signal failed, reconcile hasn't noticed), drop the handle
        // so `kill_on_drop` fires when the ProcessManager Child is dropped
        // on orchestrator shutdown — no orphan llama-server lingering on.
        if let Some(pid) = pid {
            self.process_manager.lock().await.forget(pid);
        }
        self.data.lock().await.models.remove(id);
        // Drop the per-model load guard so the HashMap doesn't slowly grow
        // as configs come and go.
        self.load_guards.lock().await.remove(id);
        self.dirty.store(true, Ordering::Relaxed);
        Ok(())
    }

    // ----- load / stop -----

    /// Ensures a model is `Running` and returns its port. Concurrent callers
    /// for the same model coalesce — first one does the work, the rest wait.
    ///
    /// The load is driven by a detached `tokio::spawn`, so if the caller's
    /// HTTP handler future is cancelled (client disconnect, browser tab
    /// reload, proxy timeout), the state transition to `Running` or
    /// `Error` still happens in the background. Without this, a cancelled
    /// caller would leave the model stuck in `Loading` forever.
    pub async fn ensure_loaded(self: Arc<Self>, id: &str) -> Result<u16, LoadError> {
        // Per-model serialization. The guard is acquired *inside* the
        // spawned task so task cancellation can't leak the lock.
        let guard = {
            let mut guards = self.load_guards.lock().await;
            guards
                .entry(id.to_string())
                .or_insert_with(|| Arc::new(Mutex::new(())))
                .clone()
        };

        let id_owned = id.to_string();
        let me = self.clone();
        let handle = tokio::spawn(async move {
            let _lock = guard.lock().await;

            // Fast path: already running.
            {
                let data = me.data.lock().await;
                let m = data
                    .models
                    .get(&id_owned)
                    .ok_or_else(|| LoadError::ModelNotFound(id_owned.clone()))?;
                if m.state == ModelState::Running {
                    return Ok(m.port);
                }
            }

            // Claim the spawn.
            {
                let mut data = me.data.lock().await;
                if let Some(m) = data.models.get_mut(&id_owned) {
                    m.state = ModelState::Loading;
                }
            }

            match me.do_load(&id_owned).await {
                Ok(port) => Ok(port),
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
                // The detached task panicked. Very unlikely, but clean up
                // so the model doesn't sit in Loading forever.
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
    async fn do_load(&self, id: &str) -> Result<u16, LoadError> {
        let (mut pending, pid, port) = {
            let _admit = self.admission.lock().await;

            // Refresh VRAM into AppData.
            let gpus = self.vram_tracker.refresh();
            {
                let mut data = self.data.lock().await;
                data.gpus = gpus.clone();
            }

            // Snapshot the model + resolve preset → binary path.
            let mut model = {
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
                m
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
                        let est = VramEstimate::compute(
                            info.file_size,
                            model.context,
                            info.n_embd_head(),
                            info.kv_heads_total,
                            kv_bytes,
                        );
                        model.estimated_vram = est.total_vram;
                    }
                    Err(e) => warn!(model = id, error = %e, "gguf parse failed; loading without estimate"),
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
                .spawn_child(&model)
                .map_err(LoadError::SpawnFailed)?;
            let pid = pending.pid;
            let port = pending.port;
            (pending, pid, port)
        };
        // Admission and process_manager mutexes are dropped here; other
        // loads can proceed even while we're still waiting for `pending`
        // to report healthy (up to 180 seconds).

        match pending.wait_for_health(std::time::Duration::from_secs(180)).await {
            Ok(()) => {
                self.process_manager.lock().await.register(pending);
                let mut data = self.data.lock().await;
                if let Some(m) = data.models.get_mut(id) {
                    m.state = ModelState::Running;
                    m.pid = Some(pid);
                    m.last_used = Some(unix_now());
                }
                self.dirty.store(true, Ordering::Relaxed);
                info!(pid, port, model = id, "inference server ready");
                Ok(port)
            }
            Err(e) => {
                // `pending` is dropped here; `kill_on_drop(true)` on the
                // Child fires and the process is SIGKILLed.
                error!(pid, port, error = %e, "health check failed; spawn cancelled");
                Err(LoadError::SpawnFailed(crate::process::manager::SpawnError::HealthCheckFailed(e)))
            }
        }
    }

    pub async fn stop_model(&self, id: &str) -> Result<(), StopError> {
        self.stop_model_inner(id).await
    }

    /// Stop helper that does not acquire the admission lock. Safe to call
    /// from inside `do_load` during eviction.
    async fn stop_model_inner(&self, id: &str) -> Result<(), StopError> {
        let pid = {
            let data = self.data.lock().await;
            let m = data.models.get(id).ok_or_else(|| StopError::ModelNotFound(id.into()))?;
            m.pid
        };
        if let Some(pid) = pid {
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

        let dead: Vec<(String, i32)> = {
            let mut data = self.data.lock().await;
            data.gpus = gpus;

            let pm = self.process_manager.lock().await;
            data.models
                .iter()
                .filter_map(|(id, m)| {
                    if m.state == ModelState::Running {
                        if let Some(pid) = m.pid {
                            if !pm.is_alive(pid) {
                                return Some((id.clone(), pid));
                            }
                        }
                    }
                    None
                })
                .collect()
        };

        if !dead.is_empty() {
            let mut data = self.data.lock().await;
            let mut pm = self.process_manager.lock().await;
            for (id, pid) in &dead {
                warn!(model = id, pid, "process died");
                if let Some(m) = data.models.get_mut(id) {
                    m.state = ModelState::Error(format!("process {} died", pid));
                    m.pid = None;
                }
                pm.forget(*pid);
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

    #[error("port {0} is already in use by model '{1}'")]
    PortConflict(u16, String),
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

    fn model(id: &str, port: u16) -> ModelConfig {
        ModelConfig {
            id: id.into(),
            name: id.into(),
            binary: PathBuf::from("/bin/true"),
            model_path: PathBuf::from("/tmp/m.gguf"),
            port,
            ..ModelConfig::default()
        }
    }

    #[tokio::test]
    async fn add_list_remove_roundtrip() {
        let tmp = tempfile::tempdir().unwrap();
        let o = orch(&tmp);
        o.add_model(model("a", 9001)).await.unwrap();
        assert_eq!(o.list_models().await.len(), 1);
        o.remove_model("a").await.unwrap();
        assert_eq!(o.list_models().await.len(), 0);
    }

    #[tokio::test]
    async fn add_model_duplicate_id_errors() {
        let tmp = tempfile::tempdir().unwrap();
        let o = orch(&tmp);
        o.add_model(model("a", 9001)).await.unwrap();
        let err = o.add_model(model("a", 9002)).await.unwrap_err();
        assert!(matches!(err, MutationError::Conflict(_)));
    }

    #[tokio::test]
    async fn add_model_duplicate_port_errors() {
        let tmp = tempfile::tempdir().unwrap();
        let o = orch(&tmp);
        o.add_model(model("a", 9001)).await.unwrap();
        let err = o.add_model(model("b", 9001)).await.unwrap_err();
        match err {
            MutationError::PortConflict(port, holder) => {
                assert_eq!(port, 9001);
                assert_eq!(holder, "a");
            }
            other => panic!("expected PortConflict, got {other:?}"),
        }
    }

    #[tokio::test]
    async fn update_model_rejects_colliding_port_from_another_model() {
        let tmp = tempfile::tempdir().unwrap();
        let o = orch(&tmp);
        o.add_model(model("a", 9001)).await.unwrap();
        o.add_model(model("b", 9002)).await.unwrap();
        // Try to point `b` at `a`'s port.
        let mut b_on_a_port = model("b", 9001);
        b_on_a_port.name = "b-renamed".into();
        let err = o.update_model(b_on_a_port).await.unwrap_err();
        assert!(matches!(err, MutationError::PortConflict(9001, ref h) if h == "a"));
    }

    #[tokio::test]
    async fn update_model_allows_reusing_its_own_port() {
        // Self-port isn't a collision — `update` may touch any field while
        // leaving the port alone.
        let tmp = tempfile::tempdir().unwrap();
        let o = orch(&tmp);
        o.add_model(model("a", 9001)).await.unwrap();
        let mut a = model("a", 9001);
        a.name = "renamed".into();
        assert!(o.update_model(a).await.is_ok());
    }

    #[tokio::test]
    async fn mark_used_updates_last_used_and_marks_dirty() {
        let tmp = tempfile::tempdir().unwrap();
        let o = orch(&tmp);
        o.add_model(model("a", 9001)).await.unwrap();
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
        o.add_model(model("a", 9001)).await.unwrap();

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
        o.add_model(model("a", 9001)).await.unwrap();

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
        o.add_model(model("charlie", 9001)).await.unwrap();
        o.add_model(model("alpha", 9002)).await.unwrap();
        o.add_model(model("bravo", 9003)).await.unwrap();
        let ids: Vec<String> = o.list_models().await.into_iter().map(|m| m.id).collect();
        assert_eq!(ids, vec!["alpha", "bravo", "charlie"]);
    }

    #[tokio::test]
    async fn update_model_clears_estimate_when_path_changes() {
        let tmp = tempfile::tempdir().unwrap();
        let o = orch(&tmp);
        let mut m = model("a", 9001);
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
        let mut m = model("a", 9001);
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
        let mut m = model("a", 9001);
        m.state = ModelState::Running;
        m.pid = Some(1); // pid 1 (init) exists but we haven't registered it in ProcessManager
        o.add_model(m).await.unwrap();

        // Set a definitely-dead pid directly.
        {
            let mut d = o.data.lock().await;
            d.models.get_mut("a").unwrap().pid = Some(999_999);
        }
        o.reconcile().await;
        let after = o.get_model("a").await.unwrap();
        match after.state {
            ModelState::Error(msg) => assert!(msg.contains("999999") || msg.contains("died")),
            other => panic!("expected Error, got {:?}", other),
        }
        assert_eq!(after.pid, None);
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
        o.add_model(model("a", 9001)).await.unwrap();
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
