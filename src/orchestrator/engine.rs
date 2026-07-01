use crate::config::{
    tag_overrides_by_pci, AppSettings, Backend, BinaryPreset, ConfigError, GpuTagOverride,
    JsonStore, ModelAlias, ModelConfig, ModelPerf, ModelState, WeightsFormat,
};
use crate::orchestrator::allocation::plan_fit_placement;
use crate::orchestrator::eviction::{decide_eviction, EvictionAction};
use crate::process::manager::{ModelRuntime, ProcessManager, RequestGuard, SpawnError};
use crate::system::stats::{SystemStats, SystemTracker};
use crate::vram::estimator::GgufMeta;
use crate::vram::llama_fit::{
    apply_sizing_to_model, fit_binary_for_server, needs_server_owned_fit, run_llama_fit_sizing,
};
use crate::vram::tracker::{GpuInfo, VRAMTracker};
use serde::Serialize;
use std::collections::{HashMap, HashSet, VecDeque};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use tokio::sync::Mutex;
use tracing::{error, info, warn};

pub const DEFAULT_MAX_BODY_BYTES: usize = 1024 * 1024 * 1024;
pub const DEFAULT_MAX_INSTANCES_PER_MODEL: usize = usize::MAX;
pub const DEFAULT_VRAM_WAIT_MS: u64 = 300_000;
const MAX_EVENTS: usize = 200;

/// Shared runtime data — single source of truth for models + gpus + presets.
#[derive(Default)]
pub struct AppData {
    pub models: HashMap<String, ModelConfig>,
    pub gpus: Vec<GpuInfo>,
    pub presets: HashMap<String, BinaryPreset>,
    /// Alias name -> alias definition. Resolved to a target model at request time.
    pub aliases: HashMap<String, ModelAlias>,
}

#[derive(Debug, Clone, Serialize)]
pub struct AppEvent {
    pub ts: f64,
    pub level: &'static str,
    pub message: String,
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
    /// VRAM (bytes) reserved by in-flight loads: processes that have been
    /// fork/exec'd but whose weights haven't yet appeared in sysfs readings.
    /// Subtracted from the sysfs free-VRAM figure inside every admission
    /// window so concurrent loads don't double-book the same headroom.
    pub reserved_vram: Arc<std::sync::atomic::AtomicU64>,
    /// Per-model lock so concurrent `ensure_loaded("m")` calls collapse:
    /// the second waits, then sees `Running` and returns the existing port.
    pub load_guards: Arc<Mutex<HashMap<String, Arc<Mutex<()>>>>>,
    pub store: Arc<JsonStore<Vec<ModelConfig>>>,
    pub presets_store: Arc<JsonStore<Vec<BinaryPreset>>>,
    pub aliases_store: Arc<JsonStore<Vec<ModelAlias>>>,
    /// Operator overrides of per-GPU backend tags (`gpus.json`). `None` in
    /// minimal constructors. Applied to `vram_tracker` at startup and on every
    /// edit.
    pub gpu_tags_store: Option<Arc<JsonStore<Vec<GpuTagOverride>>>>,
    pub settings_store: Option<Arc<JsonStore<AppSettings>>>,
    /// Per-model throughput averages (`model_perf.json`). `None` in minimal
    /// constructors. Updated per request from the upstream response `timings`;
    /// persisted by reconcile when `perf_dirty`.
    pub perf_store: Option<Arc<JsonStore<HashMap<String, ModelPerf>>>>,
    pub dirty: Arc<AtomicBool>,
    /// Set when presets change so reconcile persists presets.json.
    pub presets_dirty: Arc<AtomicBool>,
    /// Set when aliases change so reconcile persists aliases.json.
    pub aliases_dirty: Arc<AtomicBool>,
    /// Set when app settings change so reconcile persists settings.json.
    pub settings_dirty: Arc<AtomicBool>,
    /// Set when per-model perf averages change so reconcile persists
    /// model_perf.json.
    pub perf_dirty: Arc<AtomicBool>,
    pub server_port: u16,
    pub settings: Arc<Mutex<AppSettings>>,
    pub max_body_bytes: usize,
    pub max_instances_per_model: usize,
    pub vram_wait_timeout: std::time::Duration,
    events: Arc<Mutex<VecDeque<AppEvent>>>,
}

impl Orchestrator {
    #[allow(dead_code)]
    pub fn new(
        store: Arc<JsonStore<Vec<ModelConfig>>>,
        presets_store: Arc<JsonStore<Vec<BinaryPreset>>>,
        aliases_store: Arc<JsonStore<Vec<ModelAlias>>>,
        server_port: u16,
    ) -> Self {
        Self::new_inner(
            store,
            presets_store,
            aliases_store,
            None,
            None,
            None,
            AppSettings::from_env(),
            server_port,
        )
    }

    pub fn new_with_settings_store(
        store: Arc<JsonStore<Vec<ModelConfig>>>,
        presets_store: Arc<JsonStore<Vec<BinaryPreset>>>,
        aliases_store: Arc<JsonStore<Vec<ModelAlias>>>,
        settings_store: Arc<JsonStore<AppSettings>>,
        gpu_tags_store: Arc<JsonStore<Vec<GpuTagOverride>>>,
        perf_store: Arc<JsonStore<HashMap<String, ModelPerf>>>,
        server_port: u16,
    ) -> Self {
        let settings = settings_store.snapshot();
        Self::new_inner(
            store,
            presets_store,
            aliases_store,
            Some(settings_store),
            Some(gpu_tags_store),
            Some(perf_store),
            settings.sanitized(),
            server_port,
        )
    }

    #[allow(clippy::too_many_arguments)]
    fn new_inner(
        store: Arc<JsonStore<Vec<ModelConfig>>>,
        presets_store: Arc<JsonStore<Vec<BinaryPreset>>>,
        aliases_store: Arc<JsonStore<Vec<ModelAlias>>>,
        settings_store: Option<Arc<JsonStore<AppSettings>>>,
        gpu_tags_store: Option<Arc<JsonStore<Vec<GpuTagOverride>>>>,
        perf_store: Option<Arc<JsonStore<HashMap<String, ModelPerf>>>>,
        settings: AppSettings,
        server_port: u16,
    ) -> Self {
        let max_body_bytes =
            env_usize("INFERENCE_ROUTER_MAX_BODY_BYTES").unwrap_or(DEFAULT_MAX_BODY_BYTES);
        let max_instances_per_model = env_usize("INFERENCE_ROUTER_MAX_INSTANCES_PER_MODEL")
            .unwrap_or(DEFAULT_MAX_INSTANCES_PER_MODEL)
            .max(1);
        let vram_wait_timeout = std::time::Duration::from_millis(
            env_u64("INFERENCE_ROUTER_VRAM_WAIT_MS").unwrap_or(DEFAULT_VRAM_WAIT_MS),
        );

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
        let aliases: HashMap<String, ModelAlias> = aliases_store
            .snapshot()
            .into_iter()
            .map(|a| (a.alias.clone(), a))
            .collect();

        // Seed the tracker with any persisted per-GPU tag overrides so the very
        // first refresh already reflects the operator's choices.
        let vram_tracker = Arc::new(VRAMTracker::default());
        if let Some(ref store) = gpu_tags_store {
            vram_tracker.set_tag_overrides(tag_overrides_by_pci(&store.snapshot()));
        }

        Self {
            data: Arc::new(Mutex::new(AppData {
                models,
                gpus: Vec::new(),
                presets,
                aliases,
            })),
            process_manager: Arc::new(Mutex::new(ProcessManager::default())),
            vram_tracker,
            system_tracker: Arc::new(SystemTracker::default()),
            admission: Arc::new(Mutex::new(())),
            reserved_vram: Arc::new(std::sync::atomic::AtomicU64::new(0)),
            load_guards: Arc::new(Mutex::new(HashMap::new())),
            store,
            presets_store,
            aliases_store,
            gpu_tags_store,
            settings_store,
            perf_store,
            dirty: Arc::new(AtomicBool::new(migrated_any)),
            presets_dirty: Arc::new(AtomicBool::new(false)),
            aliases_dirty: Arc::new(AtomicBool::new(false)),
            settings_dirty: Arc::new(AtomicBool::new(false)),
            perf_dirty: Arc::new(AtomicBool::new(false)),
            server_port,
            settings: Arc::new(Mutex::new(settings)),
            max_body_bytes,
            max_instances_per_model,
            vram_wait_timeout,
            events: Arc::new(Mutex::new(VecDeque::new())),
        }
    }

    // ----- CRUD -----

    /// Lists every configured model, sorted by `id` so the JSON response and
    /// dashboard have a stable deterministic order.
    pub async fn list_models(&self) -> Vec<ModelConfig> {
        let mut list: Vec<ModelConfig> = self.data.lock().await.models.values().cloned().collect();
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

    pub async fn model_runtimes(&self) -> HashMap<String, ModelRuntime> {
        self.process_manager.lock().await.model_runtimes()
    }

    /// Fold one request's decode/prefill tokens-per-second (from the upstream
    /// `timings`) into the model's running average. Sync + cheap (a HashMap
    /// update under a sync mutex); the disk write happens in reconcile.
    pub fn record_perf(&self, model_id: &str, decode: f64, prefill: f64) {
        let Some(store) = &self.perf_store else {
            return;
        };
        store.with_mut(|map| {
            map.entry(model_id.to_string())
                .or_default()
                .record(decode, prefill);
        });
        self.perf_dirty.store(true, Ordering::Relaxed);
    }

    /// Drop a model's accumulated averages — called when its config changes, so
    /// stale timings don't blend with the new setup.
    pub fn reset_perf(&self, model_id: &str) {
        let Some(store) = &self.perf_store else {
            return;
        };
        let removed = store.with_mut(|map| map.remove(model_id).is_some());
        if removed {
            self.perf_dirty.store(true, Ordering::Relaxed);
        }
    }

    /// Snapshot of every model's persisted throughput averages.
    pub fn perf_snapshot(&self) -> HashMap<String, ModelPerf> {
        self.perf_store
            .as_ref()
            .map(|s| s.snapshot())
            .unwrap_or_default()
    }

    pub async fn recent_events(&self) -> Vec<AppEvent> {
        self.events.lock().await.iter().cloned().collect()
    }

    async fn record_event(&self, level: &'static str, message: impl Into<String>) {
        let mut events = self.events.lock().await;
        events.push_front(AppEvent {
            ts: unix_now(),
            level,
            message: message.into(),
        });
        while events.len() > MAX_EVENTS {
            events.pop_back();
        }
    }

    // ----- GPU capability tags -----

    /// Set (or clear) the operator's backend tags for one GPU, keyed by PCI bus
    /// id. Persists `gpus.json` and re-applies the override set to the tracker
    /// so the next refresh reflects it immediately.
    pub async fn set_gpu_tags(
        &self,
        pci_bus_id: &str,
        tags: std::collections::BTreeSet<Backend>,
    ) -> Result<(), MutationError> {
        let Some(store) = self.gpu_tags_store.clone() else {
            return Err(MutationError::NotFound("gpu tags store".into()));
        };
        store.with_mut(|list| {
            list.retain(|o| o.pci_bus_id != pci_bus_id);
            list.push(GpuTagOverride {
                pci_bus_id: pci_bus_id.to_string(),
                tags,
            });
        });
        let _ = store.save();
        self.vram_tracker
            .set_tag_overrides(tag_overrides_by_pci(&store.snapshot()));
        // Refresh so data.gpus reflects the new tags right away.
        let gpus = self.vram_tracker.refresh();
        self.data.lock().await.gpus = gpus;
        self.record_event("info", format!("updated GPU tags for {pci_bus_id}"))
            .await;
        Ok(())
    }

    // ----- Presets -----

    pub async fn list_presets(&self) -> Vec<BinaryPreset> {
        let mut v: Vec<BinaryPreset> = self.data.lock().await.presets.values().cloned().collect();
        v.sort_by(|a, b| a.id.cmp(&b.id));
        v
    }

    // ----- Aliases -----

    pub async fn list_aliases(&self) -> Vec<ModelAlias> {
        let mut v: Vec<ModelAlias> = self.data.lock().await.aliases.values().cloned().collect();
        v.sort_by(|a, b| a.alias.cmp(&b.alias));
        v
    }

    /// Resolve a requested name to a real model id by following the alias
    /// chain. An alias may target another alias, so `default → planner →
    /// qwen-32b` resolves to `qwen-32b`; repointing `default` propagates to
    /// every alias that references it.
    ///
    /// Returns an empty string when the chain hits an unassigned alias or a
    /// cycle (the proxy turns that into a clear 503). A name that isn't an
    /// alias passes through unchanged. Always resolves regardless of the
    /// `/v1/models` exposure mode.
    pub async fn resolve_model_id(&self, name: &str) -> String {
        let data = self.data.lock().await;
        resolve_alias_chain(&data.aliases, name)
    }

    pub async fn add_alias(&self, alias: ModelAlias) -> Result<(), MutationError> {
        let mut data = self.data.lock().await;
        if data.aliases.contains_key(&alias.alias) {
            return Err(MutationError::AliasConflict(alias.alias));
        }
        validate_alias(&data, &alias)?;
        data.aliases.insert(alias.alias.clone(), alias);
        self.aliases_dirty.store(true, Ordering::Relaxed);
        Ok(())
    }

    pub async fn update_alias(&self, alias: ModelAlias) -> Result<(), MutationError> {
        let mut data = self.data.lock().await;
        if !data.aliases.contains_key(&alias.alias) {
            return Err(MutationError::AliasNotFound(alias.alias));
        }
        validate_alias(&data, &alias)?;
        data.aliases.insert(alias.alias.clone(), alias);
        self.aliases_dirty.store(true, Ordering::Relaxed);
        Ok(())
    }

    pub async fn remove_alias(&self, name: &str) -> Result<(), MutationError> {
        let mut data = self.data.lock().await;
        if data.aliases.remove(name).is_none() {
            return Err(MutationError::AliasNotFound(name.into()));
        }
        // Any alias that targeted this one is now unassigned (mirrors how
        // deleting a model unassigns aliases). The referencing alias stays
        // defined and can be repointed.
        for a in data.aliases.values_mut() {
            if a.target == name {
                a.target.clear();
            }
        }
        self.aliases_dirty.store(true, Ordering::Relaxed);
        Ok(())
    }

    // ----- App settings -----

    pub async fn settings(&self) -> AppSettings {
        let settings: AppSettings = self.settings.lock().await.clone();
        settings
    }

    pub async fn update_settings(&self, settings: AppSettings) {
        *self.settings.lock().await = settings.sanitized();
        self.settings_dirty.store(true, Ordering::Relaxed);
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
        let mut models_to_stop = Vec::new();
        {
            let mut data = self.data.lock().await;
            let existing = data
                .presets
                .get(&preset.id)
                .ok_or_else(|| MutationError::NotFound(preset.id.clone()))?;
            if existing.binary != preset.binary {
                models_to_stop = data
                    .models
                    .values()
                    .filter(|m| {
                        m.binary_preset.as_deref() == Some(preset.id.as_str())
                            && (m.state == ModelState::Running || m.state == ModelState::Loading)
                    })
                    .map(|m| m.id.clone())
                    .collect();
            }
            data.presets.insert(preset.id.clone(), preset);
            self.presets_dirty.store(true, Ordering::Relaxed);
        }
        for id in models_to_stop {
            info!(model = id, "stopping model after binary preset change");
            self.record_event("info", format!("stopping {id}: binary preset changed"))
                .await;
            if let Err(e) = self.stop_model_inner(&id).await {
                warn!(model = id, error = %e, "failed to stop model after preset change");
            }
        }
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
        let stop_after_update;
        let config_changed;
        let id = new.id.clone();
        {
            let mut data = self.data.lock().await;
            if let Some(ref did) = new.draft_model_id {
                validate_draft_reference(&data.models, &new.id, did)?;
            }
            let existing = data
                .models
                .get_mut(&new.id)
                .ok_or_else(|| MutationError::NotFound(new.id.clone()))?;

            let process_live =
                existing.state == ModelState::Running || existing.state == ModelState::Loading;
            // Any change to how the model is spawned invalidates its recorded
            // throughput — reset the perf average for it (whether or not it's
            // currently running).
            config_changed = spawn_config_changed(existing, &new);
            stop_after_update = process_live && config_changed;

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
                || existing.cache_type_v != updated.cache_type_v;
            if kv_invalidated {
                updated.estimated_vram = 0;
            } else {
                updated.estimated_vram = existing.estimated_vram;
            }
            updated.state = preserved_state;
            updated.pid = preserved_pid;
            updated.last_used = preserved_last_used;

            *existing = updated;
        }
        self.dirty.store(true, Ordering::Relaxed);
        if config_changed {
            self.reset_perf(&id);
        }
        if stop_after_update {
            info!(
                model = id,
                "stopping model after spawn-affecting config change"
            );
            self.record_event("info", format!("stopping {id}: configuration changed"))
                .await;
            if let Err(e) = self.stop_model_inner(&id).await {
                warn!(model = id, error = %e, "failed to stop model after config change");
            }
        }
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
        {
            let mut data = self.data.lock().await;
            data.models.remove(id);
            // Aliases are canonical interface names, so we keep them and just
            // unassign the target. The alias stays defined and can be pointed
            // at another model from the UI without being recreated.
            let mut unassigned_any = false;
            for a in data.aliases.values_mut() {
                if a.target == id {
                    a.target.clear();
                    unassigned_any = true;
                }
            }
            if unassigned_any {
                self.aliases_dirty.store(true, Ordering::Relaxed);
            }
        }
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
    /// instance is spawned and returned for the current request.
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
            if let Some(guard) = self.scale_or_reuse_busy_instance(id, false).await? {
                return Ok(guard);
            }
            // Instance pool drained between our checks (all died) — fall through to spawn.
        }

        // No instances: serialize the first spawn via a per-model load_guard.
        let load_guard = {
            let mut guards = self.load_guards.lock().await;
            guards
                .entry(id.to_string())
                .or_insert_with(|| Arc::new(Mutex::new(())))
                .clone()
        };

        let id_owned = id.to_string();
        let me = self.clone();
        let handle = tokio::spawn(async move {
            let _lock = load_guard.lock().await;

            // Re-check after acquiring the lock — another task may have spawned already.
            if let Some(g) = me
                .process_manager
                .lock()
                .await
                .acquire_idle_instance(&id_owned)
            {
                return Ok(g);
            }
            if me.process_manager.lock().await.instance_count(&id_owned) > 0 {
                if let Some(g) = me.scale_or_reuse_busy_instance(&id_owned, true).await? {
                    return Ok(g);
                }
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

            let started = std::time::Instant::now();
            loop {
                match me.do_load(&id_owned).await {
                    Ok(guard) => return Ok(guard),
                    Err(e @ LoadError::InsufficientVram { .. }) => {
                        if !me.should_wait_for_vram(started).await {
                            let mut data = me.data.lock().await;
                            if let Some(m) = data.models.get_mut(&id_owned) {
                                m.state = ModelState::Idle;
                                m.pid = None;
                            }
                            me.dirty.store(true, Ordering::Relaxed);
                            return Err(e);
                        }
                    }
                    Err(e) => {
                        let mut data = me.data.lock().await;
                        if let Some(m) = data.models.get_mut(&id_owned) {
                            m.state = ModelState::Error(e.to_string());
                            m.pid = None;
                        }
                        me.dirty.store(true, Ordering::Relaxed);
                        return Err(e);
                    }
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

    async fn scale_or_reuse_busy_instance(
        &self,
        id: &str,
        after_initial_load: bool,
    ) -> Result<Option<RequestGuard>, LoadError> {
        match self.try_spawn_additional_instance(id).await {
            Ok(Some(guard)) => return Ok(Some(guard)),
            Ok(None) => {}
            Err(e) if is_configuration_load_error(&e) => {
                return Err(e);
            }
            Err(e) => {
                if after_initial_load {
                    warn!(
                        model = id,
                        error = %e,
                        "failed to spawn additional instance after initial load; reusing existing busy instance",
                    );
                } else {
                    warn!(
                        model = id,
                        error = %e,
                        "failed to spawn additional instance; reusing existing busy instance",
                    );
                }
                self.record_event(
                    "warn",
                    format!("failed to scale {id}: {e}; reusing busy instance"),
                )
                .await;
            }
        }

        Ok(self.process_manager.lock().await.acquire_any_instance(id))
    }

    /// Inner load path.
    ///
    /// The admission lock serializes VRAM accounting + eviction + fork/exec
    /// — the steps where two concurrent loads could step on each other's
    /// VRAM budget. We drop admission before waiting for health; that way
    /// a 180-second health poll can't block a second model from starting
    /// on a different GPU.
    async fn do_load(&self, id: &str) -> Result<RequestGuard, LoadError> {
        // Per-GPU VRAM safety caps become the fit margins handed to llama.cpp's
        // own sizing logic.
        let (gpu_cap_pct, display_cap_pct) = {
            let s = self.settings.lock().await;
            (s.gpu_vram_cap_pct, s.display_gpu_vram_cap_pct)
        };
        let (mut pending, pid, port, vram_reservation) = {
            let _admit = self.admission.lock().await;

            // Refresh VRAM into AppData.
            let gpus = self.vram_tracker.refresh();
            {
                let mut data = self.data.lock().await;
                data.gpus = gpus.clone();
            }

            // Snapshot the model + resolve preset → binary path + backends + draft.
            let (mut model, mut draft, target_backends) = {
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
                if m.binary.as_os_str().is_empty() {
                    return Err(LoadError::NoBinary(id.into()));
                }
                let target_backends = resolve_targets(&m, &data.presets);
                // Resolve the draft reference now so any missing/
                // role-mismatched draft surfaces as a load-time error
                // rather than a cryptic spawn failure.
                let draft =
                    if let Some(ref did) = m.draft_model_id {
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
                (m, draft, target_backends)
            };
            normalize_model_device_for_llama(&mut model, &gpus);
            if let Some(ref mut d) = draft {
                normalize_model_device_for_llama(d, &gpus);
            }

            // GGUF sizing and fitted runtime args come from llama.cpp itself.
            // The custom GGUF parser is only catalog metadata; it is not a
            // placement oracle.
            let already_reserved = self.reserved_vram.load(std::sync::atomic::Ordering::SeqCst);
            let caps = (gpu_cap_pct, display_cap_pct);
            match place_model(
                &mut model,
                draft.as_ref(),
                &target_backends,
                &gpus,
                already_reserved,
                caps,
            )
            .map_err(|e| LoadError::FitProbeFailed(e.to_string()))?
            {
                PlaceOutcome::Placed {
                    backend,
                    gpus_used,
                    free,
                    ..
                } => {
                    if free < model.estimated_vram {
                        // Not enough free GPU to hold the whole model — evict
                        // idle models to give llama.cpp's fit probe more room,
                        // then re-plan. The load still proceeds if eviction
                        // frees nothing.
                        let snapshot = self.data.lock().await.models.clone();
                        let idle_models = self.process_manager.lock().await.idle_model_ids();
                        for EvictionAction::Evict(victim) in
                            decide_eviction(&snapshot, free, model.estimated_vram, &idle_models)
                        {
                            info!(victim = victim, "evicting to free GPU for fit probe");
                            self.record_event("info", format!("evicting {victim} to load {id}"))
                                .await;
                            if let Err(e) = self.stop_model_inner(&victim).await {
                                warn!(model = victim, error = %e, "eviction stop failed");
                            }
                        }
                        let gpus_after = self.vram_tracker.refresh();
                        self.data.lock().await.gpus = gpus_after.clone();
                        // Re-plan against the freed GPUs so the probe sees the
                        // extra room.
                        place_model(
                            &mut model,
                            draft.as_ref(),
                            &target_backends,
                            &gpus_after,
                            already_reserved,
                            caps,
                        )
                        .map_err(|e| LoadError::FitProbeFailed(e.to_string()))?;
                    }
                    info!(
                        model = id,
                        ?backend,
                        gpus_used,
                        device = ?model.device,
                        estimated_mib = model.estimated_vram / 1024 / 1024,
                        "auto-placed from llama.cpp fit probe"
                    );
                }
                PlaceOutcome::Fits => {}
                PlaceOutcome::DoesNotFit { free } => {
                    // No eligible GPU has any free VRAM (or a pinned model is too
                    // big). Evict idle models and retry once before giving up.
                    let snapshot = self.data.lock().await.models.clone();
                    let idle_models = self.process_manager.lock().await.idle_model_ids();
                    for EvictionAction::Evict(victim) in
                        decide_eviction(&snapshot, free, model.estimated_vram, &idle_models)
                    {
                        info!(victim = victim, "evicting to make room");
                        self.record_event("info", format!("evicting {victim} to load {id}"))
                            .await;
                        if let Err(e) = self.stop_model_inner(&victim).await {
                            warn!(model = victim, error = %e, "eviction stop failed");
                        }
                    }
                    // Re-read VRAM after eviction so placement sees freed space.
                    let gpus_after = self.vram_tracker.refresh();
                    self.data.lock().await.gpus = gpus_after.clone();
                    match place_model(
                        &mut model,
                        draft.as_ref(),
                        &target_backends,
                        &gpus_after,
                        already_reserved,
                        caps,
                    )
                    .map_err(|e| LoadError::FitProbeFailed(e.to_string()))?
                    {
                        PlaceOutcome::Placed { .. } | PlaceOutcome::Fits => {}
                        PlaceOutcome::DoesNotFit { free } => {
                            return Err(LoadError::InsufficientVram {
                                model: id.into(),
                                needed: model.estimated_vram,
                                free,
                            });
                        }
                    }
                }
            }
            {
                let mut data = self.data.lock().await;
                if let Some(m) = data.models.get_mut(id) {
                    m.estimated_vram = model.estimated_vram;
                }
            }

            // Fork + exec (fast). Holding the admission lock across this is
            // still cheap — just until the process exists on disk.
            let pending = {
                let mut pm = self.process_manager.lock().await;
                pm.spawn_child(&model, draft.as_ref())
                    .map_err(LoadError::SpawnFailed)?
            };
            self.record_event("info", format!("loading {id} on port {}", pending.port))
                .await;
            let pid = pending.pid;
            let port = pending.port;
            // Reserve this model's VRAM before releasing admission so the
            // next concurrent do_load sees the correct remaining budget.
            let vram_reservation = model.estimated_vram;
            if vram_reservation > 0 {
                self.reserved_vram
                    .fetch_add(vram_reservation, std::sync::atomic::Ordering::SeqCst);
            }
            (pending, pid, port, vram_reservation)
        };
        // Admission and process_manager mutexes are dropped here; other
        // loads can proceed even while we're still waiting for `pending`
        // to report healthy (up to 180 seconds).

        match pending
            .wait_for_health(std::time::Duration::from_secs(180))
            .await
        {
            Ok(_kv_bytes) => {
                // Weights are now in VRAM and sysfs reflects reality — release reservation.
                if vram_reservation > 0 {
                    self.reserved_vram
                        .fetch_sub(vram_reservation, std::sync::atomic::Ordering::SeqCst);
                }
                // register() returns the guard for this request (active starts at 1).
                let guard = self.process_manager.lock().await.register(pending);
                // pm lock dropped before acquiring data lock.
                {
                    let mut data = self.data.lock().await;
                    if let Some(m) = data.models.get_mut(id) {
                        m.state = ModelState::Running;
                        m.pid = Some(pid);
                        m.last_used = Some(unix_now());
                    }
                }
                self.dirty.store(true, Ordering::Relaxed);
                info!(pid, port, model = id, "inference server ready");
                self.record_event("info", format!("{id} ready on port {port}"))
                    .await;
                Ok(guard)
            }
            Err(e) => {
                // Spawn failed — release reservation so subsequent loads aren't blocked.
                if vram_reservation > 0 {
                    self.reserved_vram
                        .fetch_sub(vram_reservation, std::sync::atomic::Ordering::SeqCst);
                }
                self.process_manager.lock().await.discard_pending(pending);
                error!(pid, port, error = %e, "health check failed; spawn cancelled");
                self.record_event("error", format!("{id} failed health check: {e}"))
                    .await;
                Err(LoadError::SpawnFailed(
                    crate::process::manager::SpawnError::HealthCheckFailed(e),
                ))
            }
        }
    }

    /// Try to spawn an additional instance for the request that triggered
    /// scale-out. Returns `Ok(None)` when scaling is not currently possible
    /// (cap reached, no estimate, or insufficient free VRAM) so the caller can
    /// fall back to an existing busy instance.
    ///
    /// Serialized via the admission lock; if an idle instance appears by the
    /// time admission is acquired, that idle instance is returned instead.
    async fn try_spawn_additional_instance(
        &self,
        id: &str,
    ) -> Result<Option<RequestGuard>, LoadError> {
        let (mut pending, pid, port, vram_reservation) = {
            let _admit = self.admission.lock().await;

            // Re-check: did an idle instance appear while we waited?
            if let Some(guard) = self.process_manager.lock().await.acquire_idle_instance(id) {
                return Ok(Some(guard));
            }
            if self.process_manager.lock().await.total_instance_count(id)
                >= self.max_instances_per_model
            {
                return Ok(None);
            }

            let gpus = self.vram_tracker.refresh();
            {
                self.data.lock().await.gpus = gpus.clone();
            }

            let (mut model, mut draft, target_backends) = {
                let data = self.data.lock().await;
                let m = data
                    .models
                    .get(id)
                    .cloned()
                    .ok_or_else(|| LoadError::ModelNotFound(id.into()))?;
                let mut model = m;
                if let Some(ref preset_id) = model.binary_preset {
                    match data.presets.get(preset_id) {
                        Some(p) => model.binary = p.binary.clone(),
                        None => return Err(LoadError::PresetNotFound(preset_id.clone())),
                    }
                }
                if model.binary.as_os_str().is_empty() {
                    return Err(LoadError::NoBinary(id.into()));
                }
                let target_backends = resolve_targets(&model, &data.presets);
                let draft =
                    if let Some(ref did) = model.draft_model_id {
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
                (model, draft, target_backends)
            };
            normalize_model_device_for_llama(&mut model, &gpus);
            if let Some(ref mut d) = draft {
                normalize_model_device_for_llama(d, &gpus);
            }

            // Best-effort scale-out: place this extra instance on its backend's
            // GPUs that have room *right now* (e.g. the GPUs the first instance
            // left free). No eviction — if nothing fits, reuse the busy instance.
            let already_reserved = self.reserved_vram.load(std::sync::atomic::Ordering::SeqCst);
            let caps = {
                let s = self.settings.lock().await;
                (s.gpu_vram_cap_pct, s.display_gpu_vram_cap_pct)
            };
            match place_model(
                &mut model,
                draft.as_ref(),
                &target_backends,
                &gpus,
                already_reserved,
                caps,
            )
            .map_err(|e| LoadError::FitProbeFailed(e.to_string()))?
            {
                PlaceOutcome::Placed {
                    backend,
                    gpus_used,
                    fully_on_gpu,
                    ..
                } => {
                    if !scale_out_accepts_placement(fully_on_gpu) {
                        info!(
                            model = id,
                            ?backend,
                            gpus_used,
                            device = ?model.device,
                            estimated_mib = model.estimated_vram / 1024 / 1024,
                            "skipping scale-out because extra instance would spill to CPU"
                        );
                        return Ok(None);
                    }
                    info!(
                        model = id,
                        ?backend,
                        gpus_used,
                        device = ?model.device,
                        estimated_mib = model.estimated_vram / 1024 / 1024,
                        "scaling out from llama.cpp fit probe"
                    );
                }
                PlaceOutcome::Fits => {}
                PlaceOutcome::DoesNotFit { .. } => return Ok(None),
            }

            let pending = match self
                .process_manager
                .lock()
                .await
                .spawn_child(&model, draft.as_ref())
            {
                Ok(p) => p,
                Err(e) => {
                    return Err(LoadError::SpawnFailed(e));
                }
            };
            let pid = pending.pid;
            let port = pending.port;
            self.record_event(
                "info",
                format!("loading extra {id} instance on port {port}"),
            )
            .await;
            let vram_reservation = model.estimated_vram;
            if vram_reservation > 0 {
                self.reserved_vram
                    .fetch_add(vram_reservation, std::sync::atomic::Ordering::SeqCst);
            }
            (pending, pid, port, vram_reservation)
        };
        // Admission lock dropped; health check runs without blocking other spawns.

        match pending
            .wait_for_health(std::time::Duration::from_secs(180))
            .await
        {
            Ok(_) => {
                if vram_reservation > 0 {
                    self.reserved_vram
                        .fetch_sub(vram_reservation, std::sync::atomic::Ordering::SeqCst);
                }
                let guard = self.process_manager.lock().await.register(pending);
                {
                    let mut data = self.data.lock().await;
                    if let Some(m) = data.models.get_mut(id) {
                        m.state = ModelState::Running;
                        m.pid = Some(pid);
                        m.last_used = Some(unix_now());
                    }
                }
                self.dirty.store(true, Ordering::Relaxed);
                info!(pid, port, model = id, "extra instance ready");
                self.record_event("info", format!("extra {id} instance ready on port {port}"))
                    .await;
                Ok(Some(guard))
            }
            Err(e) => {
                if vram_reservation > 0 {
                    self.reserved_vram
                        .fetch_sub(vram_reservation, std::sync::atomic::Ordering::SeqCst);
                }
                self.process_manager.lock().await.discard_pending(pending);
                warn!(pid, port, model = id, error = %e, "extra instance health check failed");
                self.record_event(
                    "warn",
                    format!("extra {id} instance failed health check: {e}"),
                )
                .await;
                Err(LoadError::SpawnFailed(
                    crate::process::manager::SpawnError::HealthCheckFailed(e),
                ))
            }
        }
    }

    async fn should_wait_for_vram(&self, started: std::time::Instant) -> bool {
        if self.vram_wait_timeout.is_zero() {
            return false;
        }
        let elapsed = started.elapsed();
        if elapsed >= self.vram_wait_timeout {
            return false;
        }

        let (has_active, notify) = {
            let pm = self.process_manager.lock().await;
            (pm.has_active_requests(), pm.request_done_notifier())
        };
        if !has_active {
            return false;
        }

        tokio::time::timeout(self.vram_wait_timeout - elapsed, notify.notified())
            .await
            .is_ok()
    }

    pub async fn stop_model(&self, id: &str) -> Result<(), StopError> {
        self.stop_model_inner(id).await
    }

    /// Stop all instances of `id`. Safe to call from inside `do_load` during
    /// eviction (does not acquire the admission lock).
    async fn stop_model_inner(&self, id: &str) -> Result<(), StopError> {
        {
            let data = self.data.lock().await;
            data.models
                .get(id)
                .ok_or_else(|| StopError::ModelNotFound(id.into()))?;
        }
        let pids = self.process_manager.lock().await.pids_for_model(id);
        for pid in &pids {
            self.process_manager.lock().await.stop(*pid).await;
        }
        if !pids.is_empty() {
            self.record_event("info", format!("stopped {id}")).await;
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

    // ----- self-heal / watchdog -----

    /// Kill one wedged instance so the next request respawns it clean. Marks the
    /// model `Idle` (not `Error`) when its last instance is gone, so a follow-up
    /// request reloads it automatically. Returns `false` if the pid is no longer
    /// tracked (already gone). Safe to call from any background task.
    pub async fn recycle_instance(&self, pid: i32, reason: &str) -> bool {
        let model_id = { self.process_manager.lock().await.model_id_for_pid(pid) };
        let Some(model_id) = model_id else {
            return false;
        };
        warn!(pid, model = %model_id, reason, "recycling wedged inference instance");
        self.record_event(
            "warn",
            format!("recycling {model_id} (pid {pid}): {reason}"),
        )
        .await;

        self.process_manager.lock().await.stop(pid).await;

        if self.process_manager.lock().await.instance_count(&model_id) == 0 {
            let mut data = self.data.lock().await;
            if let Some(m) = data.models.get_mut(&model_id) {
                m.state = ModelState::Idle;
                m.pid = None;
            }
        }
        self.dirty.store(true, Ordering::Relaxed);
        self.notify(&format!(
            "inference-router recycled {model_id} (pid {pid}): {reason}"
        ))
        .await;
        true
    }

    /// Recycle every live GPU-backed instance. Called by the engine-reset
    /// watchdog: a GPU hang can wedge any instance resident on the GPU, and
    /// CPU-only instances are unaffected. Returns how many were recycled.
    pub async fn recycle_gpu_instances(&self, reason: &str) -> usize {
        let gpu = { self.process_manager.lock().await.gpu_instances() };
        let mut recycled = 0;
        for (_, pid) in gpu {
            if self.recycle_instance(pid, reason).await {
                recycled += 1;
            }
        }
        recycled
    }

    /// Fire-and-forget notification so an auto-recovery is never silent. POSTs
    /// `{"text": msg}` to the configured webhook; no-op when unset.
    async fn notify(&self, msg: &str) {
        let url = {
            self.settings
                .lock()
                .await
                .watchdog
                .notify_webhook_url
                .clone()
        };
        let url = url.trim().to_string();
        if url.is_empty() {
            return;
        }
        let body = serde_json::json!({ "text": msg });
        tokio::spawn(async move {
            let client = reqwest::Client::new();
            if let Err(e) = client
                .post(&url)
                .json(&body)
                .timeout(std::time::Duration::from_secs(5))
                .send()
                .await
            {
                warn!(error = %e, "watchdog notify webhook failed");
            }
        });
    }

    /// Background task: watch the DRM `devcoredump` node for GPU engine resets
    /// and recycle GPU-backed instances when one fires. Spawned once by
    /// `lifecycle::run`. Runs until the process exits.
    pub async fn run_engine_reset_watchdog(self: Arc<Self>) {
        let (enabled, poll, drm_root, capture_dir) = {
            let s = self.settings.lock().await;
            (
                s.watchdog.enabled && s.watchdog.engine_reset_watch,
                s.watchdog.engine_reset_poll_secs.max(1),
                s.watchdog.drm_root.clone(),
                s.watchdog.devcoredump_capture_dir.clone(),
            )
        };
        if !enabled {
            info!("engine-reset watchdog disabled by settings");
            return;
        }
        let mut watcher = crate::system::gpu_watchdog::CoredumpWatcher::new(&drm_root);
        info!(drm_root = %drm_root, poll_secs = poll, "engine-reset watchdog armed");

        let mut ticker = tokio::time::interval(std::time::Duration::from_secs(poll));
        loop {
            ticker.tick().await;
            let fresh = watcher.poll_new();
            if fresh.is_empty() {
                continue;
            }
            for sig in &fresh {
                warn!(path = %sig.path.display(), "GPU devcoredump detected — engine reset");
                if !capture_dir.trim().is_empty() {
                    match crate::system::gpu_watchdog::capture(&sig.path, &capture_dir) {
                        Ok(saved) => info!(saved = %saved.display(), "captured devcoredump"),
                        Err(e) => warn!(
                            error = %e,
                            "could not capture devcoredump payload (likely root-only); continuing with recycle"
                        ),
                    }
                }
            }
            let n = self
                .recycle_gpu_instances("GPU engine reset (devcoredump detected)")
                .await;
            self.record_event(
                "error",
                format!("GPU engine reset detected; recycled {n} GPU instance(s)"),
            )
            .await;
        }
    }

    /// Background task: probe each live instance on a path a wedged server fails
    /// to answer (`/slots`) and recycle it after repeated timeouts. Catches
    /// wedges that never produced a devcoredump. Spawned once by `lifecycle::run`.
    pub async fn run_liveness_probe(self: Arc<Self>) {
        let (enabled, interval, timeout, path, max_fail) = {
            let s = self.settings.lock().await;
            (
                s.watchdog.enabled && s.watchdog.liveness_enabled,
                s.watchdog.liveness_interval_secs.max(1),
                s.watchdog.liveness_timeout_secs.max(1),
                s.watchdog.liveness_probe_path.clone(),
                s.watchdog.liveness_failures_to_recycle.max(1),
            )
        };
        if !enabled {
            info!("liveness probe disabled by settings");
            return;
        }
        let client = reqwest::Client::builder()
            .timeout(std::time::Duration::from_secs(timeout))
            .build()
            .unwrap_or_else(|_| reqwest::Client::new());
        info!(path = %path, interval_secs = interval, "liveness probe armed");

        let mut fails: HashMap<i32, u32> = HashMap::new();
        let mut ticker = tokio::time::interval(std::time::Duration::from_secs(interval));
        loop {
            ticker.tick().await;
            let instances = { self.process_manager.lock().await.idle_instances() };
            let live_pids: HashSet<i32> = instances.iter().map(|(_, pid, _)| *pid).collect();
            fails.retain(|pid, _| live_pids.contains(pid));

            for (model_id, pid, port) in instances {
                let url = format!("http://127.0.0.1:{}{}", port, path);
                // Any HTTP reply (even 404/501) proves the server loop is alive.
                // Only a *timeout* signals a wedge; connection errors mean the
                // process is gone, which the reconcile reaper handles.
                let timed_out = match client.get(&url).send().await {
                    Ok(_) => false,
                    Err(e) => e.is_timeout(),
                };
                if timed_out {
                    let c = fails.entry(pid).or_insert(0);
                    *c += 1;
                    warn!(model = %model_id, pid, port, path = %path, count = *c, "liveness probe timed out");
                    if *c >= max_fail {
                        self.recycle_instance(
                            pid,
                            &format!("liveness probe timed out {c}x on {path}"),
                        )
                        .await;
                        fails.remove(&pid);
                    }
                } else {
                    fails.remove(&pid);
                }
            }
        }
    }

    // ----- reconcile -----

    /// Refresh VRAM from sysfs, detect dead processes, persist if dirty.
    /// Called on a 5s timer by `lifecycle::run`.
    pub async fn reconcile(&self) {
        let gpus = self.vram_tracker.refresh();
        {
            self.data.lock().await.gpus = gpus;
        }

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
            let store = self.store.clone();
            match tokio::task::spawn_blocking(move || store.save()).await {
                Ok(Ok(())) => {}
                Ok(Err(e)) => {
                    error!(error = %e, "failed to persist models.json");
                    self.dirty.store(true, Ordering::Relaxed);
                }
                Err(e) => {
                    error!(error = %e, "models.json persistence task failed");
                    self.dirty.store(true, Ordering::Relaxed);
                }
            }
        }

        if self.presets_dirty.swap(false, Ordering::Relaxed) {
            let snapshot: Vec<BinaryPreset> =
                self.data.lock().await.presets.values().cloned().collect();
            self.presets_store.replace(snapshot);
            let store = self.presets_store.clone();
            match tokio::task::spawn_blocking(move || store.save()).await {
                Ok(Ok(())) => {}
                Ok(Err(e)) => {
                    error!(error = %e, "failed to persist presets.json");
                    self.presets_dirty.store(true, Ordering::Relaxed);
                }
                Err(e) => {
                    error!(error = %e, "presets.json persistence task failed");
                    self.presets_dirty.store(true, Ordering::Relaxed);
                }
            }
        }

        if self.aliases_dirty.swap(false, Ordering::Relaxed) {
            let snapshot: Vec<ModelAlias> =
                self.data.lock().await.aliases.values().cloned().collect();
            self.aliases_store.replace(snapshot);
            let store = self.aliases_store.clone();
            match tokio::task::spawn_blocking(move || store.save()).await {
                Ok(Ok(())) => {}
                Ok(Err(e)) => {
                    error!(error = %e, "failed to persist aliases.json");
                    self.aliases_dirty.store(true, Ordering::Relaxed);
                }
                Err(e) => {
                    error!(error = %e, "aliases.json persistence task failed");
                    self.aliases_dirty.store(true, Ordering::Relaxed);
                }
            }
        }

        if self.settings_dirty.swap(false, Ordering::Relaxed) {
            if let Some(store) = self.settings_store.clone() {
                let snapshot: AppSettings = self.settings.lock().await.clone();
                store.replace(snapshot);
                match tokio::task::spawn_blocking(move || store.save()).await {
                    Ok(Ok(())) => {}
                    Ok(Err(e)) => {
                        error!(error = %e, "failed to persist settings.json");
                        self.settings_dirty.store(true, Ordering::Relaxed);
                    }
                    Err(e) => {
                        error!(error = %e, "settings.json persistence task failed");
                        self.settings_dirty.store(true, Ordering::Relaxed);
                    }
                }
            } else {
                self.settings_dirty.store(false, Ordering::Relaxed);
            }
        }

        if self.perf_dirty.swap(false, Ordering::Relaxed) {
            if let Some(store) = self.perf_store.clone() {
                // The store already holds the live in-memory averages (updated
                // in-place by record_perf/reset_perf), so just flush to disk.
                match tokio::task::spawn_blocking(move || store.save()).await {
                    Ok(Ok(())) => {}
                    Ok(Err(e)) => {
                        error!(error = %e, "failed to persist model_perf.json");
                        self.perf_dirty.store(true, Ordering::Relaxed);
                    }
                    Err(e) => {
                        error!(error = %e, "model_perf.json persistence task failed");
                        self.perf_dirty.store(true, Ordering::Relaxed);
                    }
                }
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

    #[error(
        "binary preset '{0}' not found — edit the model and pick an existing preset or a custom path"
    )]
    PresetNotFound(String),

    #[error(
        "model '{0}' has no binary configured — edit the model and pick a binary preset or a custom path"
    )]
    NoBinary(String),

    #[error("model references draft '{id}', but no model '{id}' exists (target: '{target}')")]
    DraftNotFound { id: String, target: String },

    #[error("spawn failed: {0}")]
    SpawnFailed(SpawnError),

    #[error("llama.cpp fit probe failed: {0}")]
    FitProbeFailed(String),

    #[error("not enough idle VRAM to load '{model}': need {:.1} GiB, have {:.1} GiB (active requests are not evicted)", bytes_to_gib(*needed), bytes_to_gib(*free))]
    InsufficientVram {
        model: String,
        needed: u64,
        free: u64,
    },
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

    #[error("alias '{0}' already exists")]
    AliasConflict(String),

    #[error("alias '{0}' not found")]
    AliasNotFound(String),

    #[error("alias '{0}' collides with an existing model id")]
    AliasShadowsModel(String),

    #[error("alias '{alias}' points at '{target}', which is not a model or alias")]
    AliasTargetMissing { alias: String, target: String },

    #[error("alias '{alias}' → '{target}' would create a resolution cycle")]
    AliasCycle { alias: String, target: String },

    #[error("invalid alias: {0}")]
    AliasInvalid(String),
}

/// A canonical alias name: 1–64 chars of lowercase ASCII letters, digits, and
/// `.`, `_`, `-`. Kept tight because the name doubles as a URL path segment
/// (`/api/aliases/{alias}`) and the `model` field clients send.
fn is_valid_alias_name(s: &str) -> bool {
    !s.is_empty()
        && s.len() <= 64
        && s.bytes().all(|b| {
            b.is_ascii_lowercase() || b.is_ascii_digit() || matches!(b, b'.' | b'_' | b'-')
        })
}

/// Follow the alias chain from `name` to a concrete model id.
///
/// Stops and returns an empty string when it reaches an unassigned alias or
/// detects a cycle; returns `name` unchanged when it isn't an alias.
fn resolve_alias_chain(aliases: &HashMap<String, ModelAlias>, name: &str) -> String {
    let mut current = name.to_string();
    let mut seen = HashSet::new();
    loop {
        match aliases.get(&current) {
            Some(a) => {
                if !seen.insert(current.clone()) {
                    return String::new(); // cycle
                }
                if a.target.is_empty() {
                    return String::new(); // unassigned alias in the chain
                }
                current = a.target.clone();
            }
            // Not an alias: a model id (or an unknown name that passes through).
            None => return current,
        }
    }
}

/// Would assigning `target` to `alias_name` create a cycle? Walks the chain
/// starting from `target`; revisiting any node (including `alias_name`) means
/// the assignment would loop.
fn target_creates_cycle(
    aliases: &HashMap<String, ModelAlias>,
    alias_name: &str,
    target: &str,
) -> bool {
    let mut current = target.to_string();
    let mut seen = HashSet::new();
    seen.insert(alias_name.to_string());
    while let Some(a) = aliases.get(&current) {
        if !seen.insert(current.clone()) {
            return true;
        }
        if a.target.is_empty() {
            return false;
        }
        current = a.target.clone();
    }
    false
}

/// Validate an alias against the current model + alias tables.
///
/// Aliases are canonical, stable interface names, so an empty `target` is a
/// valid "unassigned" state — the alias exists and can be reassigned from the
/// UI without being recreated.
///
/// Checked conditions:
/// - the alias name matches the canonical charset (see `is_valid_alias_name`)
/// - the alias does not shadow an existing model id
/// - if a target is given, it names an existing model **or** an existing alias
/// - the target does not create a resolution cycle
fn validate_alias(data: &AppData, alias: &ModelAlias) -> Result<(), MutationError> {
    if !is_valid_alias_name(&alias.alias) {
        return Err(MutationError::AliasInvalid(
            "alias must be 1–64 characters of lowercase letters, digits, '.', '_' or '-'".into(),
        ));
    }
    if data.models.contains_key(&alias.alias) {
        return Err(MutationError::AliasShadowsModel(alias.alias.clone()));
    }
    if !alias.target.is_empty() {
        let is_model = data.models.contains_key(&alias.target);
        let is_alias = data.aliases.contains_key(&alias.target);
        if !is_model && !is_alias {
            return Err(MutationError::AliasTargetMissing {
                alias: alias.alias.clone(),
                target: alias.target.clone(),
            });
        }
        if is_alias && target_creates_cycle(&data.aliases, &alias.alias, &alias.target) {
            return Err(MutationError::AliasCycle {
                alias: alias.alias.clone(),
                target: alias.target.clone(),
            });
        }
    }
    Ok(())
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
        return Err(MutationError::InvalidConfig(
            ConfigError::DraftSelfReference,
        ));
    }
    if !models.contains_key(draft_id) {
        return Err(MutationError::InvalidConfig(ConfigError::DraftNotFound {
            id: draft_id.to_string(),
        }));
    }
    Ok(())
}

fn spawn_config_changed(old: &ModelConfig, new: &ModelConfig) -> bool {
    old.weights_format != new.weights_format
        || old.binary_preset != new.binary_preset
        || old.binary != new.binary
        || old.model_path != new.model_path
        || old.mmproj_path != new.mmproj_path
        || old.extra_args != new.extra_args
        || old.context != new.context
        || old.temperature != new.temperature
        || old.top_p != new.top_p
        || old.top_k != new.top_k
        || old.min_p != new.min_p
        || old.presence_penalty != new.presence_penalty
        || old.repeat_penalty != new.repeat_penalty
        || old.flash_attn != new.flash_attn
        || old.mlock != new.mlock
        || old.no_mmap != new.no_mmap
        || old.parallel_slots != new.parallel_slots
        || old.cache_type_k != new.cache_type_k
        || old.cache_type_v != new.cache_type_v
        || old.threads != new.threads
        || old.cache_ram_mib != new.cache_ram_mib
        || old.reasoning_format != new.reasoning_format
        || old.reasoning_budget != new.reasoning_budget
        || old.chat_template_kwargs != new.chat_template_kwargs
        || old.draft_model_id != new.draft_model_id
        || old.mtp_tokens != new.mtp_tokens
        || old.draft_max != new.draft_max
        || old.draft_min != new.draft_min
        || old.draft_p_min != new.draft_p_min
        || old.ctx_checkpoints != new.ctx_checkpoints
        || old.checkpoint_min_step != new.checkpoint_min_step
        || old.checkpoint_every_n_tokens != new.checkpoint_every_n_tokens
}

fn subtract_reserved_from_gpus(mut gpus: Vec<GpuInfo>, mut reserved: u64) -> Vec<GpuInfo> {
    if reserved == 0 {
        return gpus;
    }

    let mut order: Vec<usize> = (0..gpus.len()).collect();
    order.sort_by(|a, b| gpus[*b].free_vram().cmp(&gpus[*a].free_vram()));
    for idx in order {
        if reserved == 0 {
            break;
        }
        let free = gpus[idx].free_vram();
        let take = free.min(reserved);
        gpus[idx].used_vram = gpus[idx].used_vram.saturating_add(take);
        reserved -= take;
    }
    gpus
}

/// The ordered backends a model may run on, from its preset's `targets`
/// (or a single backend inferred from a legacy preset). Falls back to Vulkan —
/// the historical implicit backend — for models with no preset at all.
fn resolve_targets(model: &ModelConfig, presets: &HashMap<String, BinaryPreset>) -> Vec<Backend> {
    model
        .binary_preset
        .as_deref()
        .and_then(|id| presets.get(id))
        .map(BinaryPreset::effective_targets)
        .filter(|t| !t.is_empty())
        .unwrap_or_else(|| vec![Backend::Vulkan])
}

/// Result of attempting to place a model on the current GPUs.
enum PlaceOutcome {
    /// Auto-placed GGUF: `model.device` and llama.cpp-fitted runtime args were set.
    /// `free` is the chosen backend's allocatable VRAM, so the caller can decide
    /// whether to evict idle models for more room (placement succeeds either way
    /// because llama.cpp may spill overflow to CPU).
    Placed {
        backend: Backend,
        gpus_used: usize,
        free: u64,
        fully_on_gpu: bool,
    },
    /// Explicit-device or non-GGUF model that fits by total free VRAM.
    Fits,
    /// No eligible GPU has any free VRAM (auto GGUF), or a pinned/non-GGUF model
    /// exceeds free VRAM. `free` is the best-case free seen (for the message).
    DoesNotFit { free: u64 },
}

fn place_model(
    model: &mut ModelConfig,
    draft: Option<&ModelConfig>,
    targets: &[Backend],
    gpus: &[GpuInfo],
    reserved: u64,
    caps: (u8, u8),
) -> Result<PlaceOutcome, crate::vram::llama_fit::LlamaFitError> {
    if model.weights_format == WeightsFormat::Gguf {
        return place_gguf_with_llama_fit(model, draft, targets, gpus, reserved, caps);
    }
    Ok(try_place(model, targets, gpus, reserved, caps))
}

#[derive(Clone)]
struct FitCandidate {
    backend: Backend,
    device: String,
    fit_target: String,
    free: u64,
    gpus_used: usize,
    sizing: crate::vram::llama_fit::LlamaFitSizing,
}

fn place_gguf_with_llama_fit(
    model: &mut ModelConfig,
    draft: Option<&ModelConfig>,
    targets: &[Backend],
    gpus: &[GpuInfo],
    reserved: u64,
    caps: (u8, u8),
) -> Result<PlaceOutcome, crate::vram::llama_fit::LlamaFitError> {
    let fit_binary = fit_binary_for_server(&model.binary);
    let adjusted = subtract_reserved_from_gpus(gpus.to_vec(), reserved);
    let (gpu_cap_pct, display_cap_pct) = caps;
    let n_layers = model_n_layers(model).unwrap_or(0);

    if let Some(device) = configured_llama_device_value(model) {
        let eligible = model_visible_gpus(model, adjusted);
        let free: u64 = eligible
            .iter()
            .map(|g| g.allocatable_vram(gpu_cap_pct as u64, display_cap_pct as u64))
            .sum();
        if eligible.is_empty() {
            return Ok(PlaceOutcome::DoesNotFit { free });
        }
        let fit_target = eligible
            .iter()
            .map(|g| fit_target_for_probe_mib(g, gpus, gpu_cap_pct, display_cap_pct).to_string())
            .collect::<Vec<_>>()
            .join(",");
        let sizing = run_llama_fit_sizing(&fit_binary, model, draft, &device, &fit_target)?;
        let fully_on_gpu = fitted_fully_on_gpu(&sizing, n_layers);
        apply_fit_selection(
            model,
            &device,
            &fit_target,
            &sizing,
            needs_server_owned_fit(model, draft),
        );
        return Ok(PlaceOutcome::Placed {
            backend: targets.first().copied().unwrap_or(Backend::Vulkan),
            gpus_used: eligible.len(),
            free,
            fully_on_gpu: fully_on_gpu && !needs_server_owned_fit(model, draft),
        });
    }

    let mut best_free = 0u64;
    let mut last_error = None;

    for &backend in targets {
        let mut by_alloc: Vec<GpuInfo> = adjusted
            .iter()
            .filter(|g| !g.integrated && g.supports(backend))
            .cloned()
            .collect();
        if by_alloc.is_empty() {
            continue;
        }
        by_alloc.sort_by_key(|g| {
            std::cmp::Reverse(g.allocatable_vram(gpu_cap_pct as u64, display_cap_pct as u64))
        });

        let mut spill: Option<FitCandidate> = None;
        for count in 1..=by_alloc.len() {
            let mut chosen = by_alloc[..count].to_vec();
            chosen.sort_by_key(|g| g.backend_index(backend).unwrap_or(usize::MAX));
            let free: u64 = chosen
                .iter()
                .map(|g| g.allocatable_vram(gpu_cap_pct as u64, display_cap_pct as u64))
                .sum();
            best_free = best_free.max(free);

            let device = chosen
                .iter()
                .filter_map(|g| g.backend_device_name(backend))
                .collect::<Vec<_>>()
                .join(",");
            if device.is_empty() {
                continue;
            }
            let fit_target = chosen
                .iter()
                .map(|g| {
                    fit_target_for_probe_mib(g, gpus, gpu_cap_pct, display_cap_pct).to_string()
                })
                .collect::<Vec<_>>()
                .join(",");

            match run_llama_fit_sizing(&fit_binary, model, draft, &device, &fit_target) {
                Ok(sizing) => {
                    let candidate = FitCandidate {
                        backend,
                        device,
                        fit_target,
                        free,
                        gpus_used: chosen.len(),
                        sizing,
                    };
                    if fitted_fully_on_gpu(&candidate.sizing, n_layers) {
                        apply_fit_candidate(
                            model,
                            &candidate,
                            &candidate.fit_target,
                            needs_server_owned_fit(model, draft),
                        );
                        return Ok(PlaceOutcome::Placed {
                            backend: candidate.backend,
                            gpus_used: candidate.gpus_used,
                            free: candidate.free,
                            fully_on_gpu: !needs_server_owned_fit(model, draft),
                        });
                    }
                    keep_best_spill_candidate(&mut spill, candidate);
                }
                Err(e) => last_error = Some(e),
            }
        }

        if let Some(candidate) = spill {
            apply_fit_candidate(
                model,
                &candidate,
                &candidate.fit_target,
                needs_server_owned_fit(model, draft),
            );
            return Ok(PlaceOutcome::Placed {
                backend: candidate.backend,
                gpus_used: candidate.gpus_used,
                free: candidate.free,
                fully_on_gpu: false,
            });
        }
    }

    if let Some(e) = last_error {
        Err(e)
    } else {
        Ok(PlaceOutcome::DoesNotFit { free: best_free })
    }
}

fn apply_fit_candidate(
    model: &mut ModelConfig,
    candidate: &FitCandidate,
    fit_target: &str,
    server_owned_fit: bool,
) {
    apply_fit_selection(
        model,
        &candidate.device,
        fit_target,
        &candidate.sizing,
        server_owned_fit,
    );
}

fn apply_fit_selection(
    model: &mut ModelConfig,
    device: &str,
    fit_target: &str,
    sizing: &crate::vram::llama_fit::LlamaFitSizing,
    server_owned_fit: bool,
) {
    model.device = Some(device.to_string());
    if server_owned_fit {
        model.context = sizing.fitted.context.unwrap_or(model.context);
        model.estimated_vram = sizing.device_vram;
        model.n_gpu_layers = None;
        model.tensor_split = None;
        model.override_tensor = None;
        model.fit_target = Some(fit_target.to_string());
    } else {
        apply_sizing_to_model(model, sizing);
    }
}

fn scale_out_accepts_placement(fully_on_gpu: bool) -> bool {
    fully_on_gpu
}

fn keep_best_spill_candidate(best: &mut Option<FitCandidate>, candidate: FitCandidate) {
    let should_replace = best.as_ref().map_or(true, |current| {
        candidate.gpus_used > current.gpus_used
            || (candidate.gpus_used == current.gpus_used
                && candidate.sizing.device_vram > current.sizing.device_vram)
    });
    if should_replace {
        *best = Some(candidate);
    }
}

fn fitted_fully_on_gpu(sizing: &crate::vram::llama_fit::LlamaFitSizing, n_layers: u32) -> bool {
    if fitted_uses_cpu_override(sizing) {
        return false;
    }
    if n_layers == 0 {
        return true;
    }
    match sizing.fitted.n_gpu_layers {
        Some(n) if n < 0 => true,
        Some(n) => n as u32 >= n_layers,
        None => true,
    }
}

fn fitted_uses_cpu_override(sizing: &crate::vram::llama_fit::LlamaFitSizing) -> bool {
    sizing
        .fitted
        .override_tensor
        .as_deref()
        .map(override_tensor_uses_cpu)
        .unwrap_or(false)
}

fn override_tensor_uses_cpu(value: &str) -> bool {
    value.split(',').any(|entry| {
        entry
            .rsplit_once('=')
            .map(|(_, target)| target.trim().eq_ignore_ascii_case("cpu"))
            .unwrap_or(false)
    })
}

fn model_n_layers(model: &ModelConfig) -> Option<u32> {
    model
        .gguf_meta
        .as_ref()
        .map(|m| m.n_layers)
        .or_else(|| GgufMeta::read(&model.model_path).ok().map(|m| m.n_layers))
}

fn fit_target_for_probe_mib(
    adjusted_gpu: &GpuInfo,
    original_gpus: &[GpuInfo],
    gpu_cap_pct: u8,
    display_cap_pct: u8,
) -> u64 {
    let cap = if adjusted_gpu.display_attached {
        display_cap_pct
    } else {
        gpu_cap_pct
    }
    .clamp(1, 100) as u64;
    let base_margin = adjusted_gpu.total_vram.saturating_mul(100 - cap) / 100;
    let original_used = original_gpus
        .iter()
        .find(|g| same_gpu(g, adjusted_gpu))
        .map(|g| g.used_vram)
        .unwrap_or(adjusted_gpu.used_vram);
    let reserved = adjusted_gpu.used_vram.saturating_sub(original_used);
    bytes_to_mib_ceil(base_margin.saturating_add(reserved))
}

fn same_gpu(a: &GpuInfo, b: &GpuInfo) -> bool {
    a.pci_bus_id
        .as_ref()
        .zip(b.pci_bus_id.as_ref())
        .map(|(a, b)| a == b)
        .unwrap_or(false)
        || a.id == b.id
}

fn bytes_to_mib_ceil(bytes: u64) -> u64 {
    bytes.saturating_add(1024 * 1024 - 1) / (1024 * 1024)
}

/// Decide where `model` runs on `gpus` (reserved VRAM subtracted inside).
///
/// GGUF models are handled by `place_gguf_with_llama_fit`, which asks
/// llama.cpp for concrete fitted args before launch. The generic estimator path
/// below remains for non-GGUF models and legacy callers.
fn try_place(
    model: &mut ModelConfig,
    targets: &[Backend],
    gpus: &[GpuInfo],
    reserved: u64,
    caps: (u8, u8),
) -> PlaceOutcome {
    let adjusted = subtract_reserved_from_gpus(gpus.to_vec(), reserved);
    let (gpu_cap_pct, display_cap_pct) = caps;

    // Explicit pin or non-GGUF: trust the config, just check it fits.
    if model.weights_format != WeightsFormat::Gguf || configured_llama_device_value(model).is_some()
    {
        let eligible = model_visible_gpus(model, adjusted);
        let free: u64 = eligible
            .iter()
            .map(|g| g.allocatable_vram(gpu_cap_pct as u64, display_cap_pct as u64))
            .sum();
        return if model.estimated_vram == 0 || free >= model.estimated_vram {
            PlaceOutcome::Fits
        } else {
            PlaceOutcome::DoesNotFit { free }
        };
    }

    let mut best_free = 0u64;
    for &backend in targets {
        let candidates: Vec<GpuInfo> = adjusted
            .iter()
            .filter(|g| !g.integrated && g.supports(backend))
            .cloned()
            .collect();
        let free: u64 = candidates
            .iter()
            .map(|g| g.allocatable_vram(gpu_cap_pct as u64, display_cap_pct as u64))
            .sum();
        best_free = best_free.max(free);
        if let Some(p) = plan_fit_placement(
            backend,
            &candidates,
            model.estimated_vram,
            gpu_cap_pct,
            display_cap_pct,
        ) {
            model.device = Some(p.device);
            model.fit_target = Some(p.fit_target);
            // -fit owns -ngl / tensor-split / expert offload; clear any stale
            // micromanagement so build_command_args doesn't fight it.
            model.tensor_split = None;
            model.n_gpu_layers = None;
            model.n_cpu_moe = None;
            model.override_tensor = None;
            return PlaceOutcome::Placed {
                backend,
                gpus_used: p.gpus_used,
                free,
                fully_on_gpu: free >= model.estimated_vram,
            };
        }
    }
    PlaceOutcome::DoesNotFit { free: best_free }
}

fn model_visible_gpus(model: &ModelConfig, gpus: Vec<GpuInfo>) -> Vec<GpuInfo> {
    let Some(devices) = configured_llama_devices(model) else {
        // No explicit device: automatic placement. Integrated GPUs are kept out
        // of this pool, and raw accounting-only records stay out until the
        // selected backend has a real device name for them.
        return gpus
            .into_iter()
            .filter(|gpu| !gpu.integrated && automatic_backend_candidate(model, gpu))
            .collect();
    };
    if devices.is_empty() {
        return Vec::new();
    }
    devices
        .iter()
        .filter_map(|device| {
            gpus.iter()
                .find(|gpu| gpu_matches_device(gpu, device))
                .cloned()
        })
        .collect()
}

fn automatic_backend_candidate(model: &ModelConfig, gpu: &GpuInfo) -> bool {
    if model_uses_cuda_backend(model) {
        return gpu.cuda_device.is_some();
    }
    gpu.vulkan_index.is_some()
}

fn model_uses_cuda_backend(model: &ModelConfig) -> bool {
    model
        .binary_preset
        .as_deref()
        .map(|preset| preset.to_ascii_lowercase().contains("cuda"))
        .unwrap_or(false)
        || model
            .binary
            .to_string_lossy()
            .to_ascii_lowercase()
            .contains("cuda")
}

fn configured_llama_devices(model: &ModelConfig) -> Option<Vec<String>> {
    configured_llama_device_value(model).map(|device| split_device_list(&device))
}

fn configured_llama_device_value(model: &ModelConfig) -> Option<String> {
    model.device.clone()
}

fn split_device_list(device: &str) -> Vec<String> {
    if device.trim().eq_ignore_ascii_case("none") {
        return Vec::new();
    }
    device
        .split(',')
        .map(str::trim)
        .filter(|part| !part.is_empty())
        .map(str::to_string)
        .collect()
}

fn normalize_model_device_for_llama(model: &mut ModelConfig, gpus: &[GpuInfo]) {
    let Some(device) = configured_llama_device_value(model) else {
        return;
    };
    if device.trim().eq_ignore_ascii_case("none") {
        model.device = Some("none".into());
        return;
    }
    let mut mapped = split_device_list(&device)
        .into_iter()
        .map(|device| resolve_llama_device(&device, gpus).unwrap_or(device))
        .enumerate()
        .collect::<Vec<_>>();
    mapped.sort_by_key(|(idx, device)| {
        (
            device_vulkan_index(device, gpus).unwrap_or(usize::MAX),
            *idx,
        )
    });
    if !mapped.is_empty() {
        let mut devices = mapped
            .into_iter()
            .map(|(_, device)| device)
            .collect::<Vec<_>>();
        devices.dedup_by(|a, b| a.eq_ignore_ascii_case(b));
        model.device = Some(devices.join(","));
    } else {
        model.device = None;
    }
}

fn resolve_llama_device(device: &str, gpus: &[GpuInfo]) -> Option<String> {
    let pci = device.strip_prefix("pci:").unwrap_or(device);
    gpus.iter().find_map(|gpu| {
        if gpu_matches_device(gpu, pci) {
            gpu.vulkan_device
                .clone()
                .or_else(|| gpu.cuda_device.clone())
        } else {
            None
        }
    })
}

fn device_vulkan_index(device: &str, gpus: &[GpuInfo]) -> Option<usize> {
    gpus.iter()
        .find(|gpu| gpu_matches_device(gpu, device))
        .and_then(|gpu| gpu.vulkan_index)
}

fn gpu_matches_device(gpu: &GpuInfo, device: &str) -> bool {
    gpu.vulkan_device
        .as_deref()
        .map(|name| name.eq_ignore_ascii_case(device))
        .unwrap_or(false)
        || gpu
            .cuda_device
            .as_deref()
            .map(|name| name.eq_ignore_ascii_case(device))
            .unwrap_or(false)
        || gpu
            .pci_bus_id
            .as_deref()
            .map(|pci| {
                pci.eq_ignore_ascii_case(device)
                    || format!("pci:{pci}").eq_ignore_ascii_case(device)
            })
            .unwrap_or(false)
}

fn env_usize(name: &str) -> Option<usize> {
    std::env::var(name).ok()?.parse::<usize>().ok()
}

fn env_u64(name: &str) -> Option<u64> {
    std::env::var(name).ok()?.parse::<u64>().ok()
}

fn bytes_to_gib(bytes: u64) -> f64 {
    bytes as f64 / 1_073_741_824.0
}

fn is_configuration_load_error(e: &LoadError) -> bool {
    matches!(
        e,
        LoadError::ModelNotFound(_)
            | LoadError::PresetNotFound(_)
            | LoadError::NoBinary(_)
            | LoadError::DraftNotFound { .. }
    )
}

#[cfg(test)]
#[path = "engine_tests.rs"]
mod tests;
