use crate::config::{
    AppSettings, Backend, BinaryPreset, ConfigError, GpuTagOverride, JsonStore, ModelAlias,
    ModelConfig, ModelState, SplitMode, WeightsFormat, tag_overrides_by_pci,
};
use crate::orchestrator::allocation::plan_backend_split;
use crate::orchestrator::eviction::{EvictionAction, decide_eviction};
use crate::process::manager::{ModelRuntime, ProcessManager, RequestGuard, SpawnError};
use crate::system::stats::{SystemStats, SystemTracker};
use crate::vram::estimator::VramEstimate;
use crate::vram::tracker::{GpuInfo, VRAMTracker};
use serde::Serialize;
use std::collections::{HashMap, HashSet, VecDeque};
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
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
    /// minimal/test constructors. Applied to `vram_tracker` at startup and on
    /// every edit.
    pub gpu_tags_store: Option<Arc<JsonStore<Vec<GpuTagOverride>>>>,
    pub settings_store: Option<Arc<JsonStore<AppSettings>>>,
    pub dirty: Arc<AtomicBool>,
    /// Set when presets change so reconcile persists presets.json.
    pub presets_dirty: Arc<AtomicBool>,
    /// Set when aliases change so reconcile persists aliases.json.
    pub aliases_dirty: Arc<AtomicBool>,
    /// Set when app settings change so reconcile persists settings.json.
    pub settings_dirty: Arc<AtomicBool>,
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
        server_port: u16,
    ) -> Self {
        let settings = settings_store.snapshot();
        Self::new_inner(
            store,
            presets_store,
            aliases_store,
            Some(settings_store),
            Some(gpu_tags_store),
            settings.sanitized(),
            server_port,
        )
    }

    fn new_inner(
        store: Arc<JsonStore<Vec<ModelConfig>>>,
        presets_store: Arc<JsonStore<Vec<BinaryPreset>>>,
        aliases_store: Arc<JsonStore<Vec<ModelAlias>>>,
        settings_store: Option<Arc<JsonStore<AppSettings>>>,
        gpu_tags_store: Option<Arc<JsonStore<Vec<GpuTagOverride>>>>,
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
            dirty: Arc::new(AtomicBool::new(migrated_any)),
            presets_dirty: Arc::new(AtomicBool::new(false)),
            aliases_dirty: Arc::new(AtomicBool::new(false)),
            settings_dirty: Arc::new(AtomicBool::new(false)),
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

    #[cfg(test)]
    pub async fn get_model(&self, id: &str) -> Option<ModelConfig> {
        self.data.lock().await.models.get(id).cloned()
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
            stop_after_update = process_live && spawn_config_changed(existing, &new);

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
        }
        self.dirty.store(true, Ordering::Relaxed);
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
        // Accumulated weight file sizes (target + draft if present). Set
        // inside the admission block during GGUF reads; used after the health
        // check to recompute estimated_vram from the measured KV bytes.
        let mut weight_file_size: u64 = 0;
        // Set in the MoE estimate branch; used after placement to rebuild the
        // expert-offload --override-tensor aligned to the chosen tensor-split.
        let mut moe_n_layers: Option<u32> = None;
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

            if model.weights_format == WeightsFormat::Gguf {
                // Honour the model's configured cache_type_{k,v} so the
                // KV-cache portion of the estimate matches what the
                // backend will actually allocate at run time.
                use crate::config::CacheType;
                use crate::vram::estimator::{auto_n_cpu_moe, GgufInfo, KvPerElement};
                let kv_bytes = KvPerElement::from_types(
                    model.cache_type_k.unwrap_or(CacheType::F16),
                    model.cache_type_v.unwrap_or(CacheType::F16),
                );
                match GgufInfo::read(&model.model_path) {
                    Ok(info) => {
                        weight_file_size = info.file_size;
                        let kv = info.kv_cache_bytes(model.context, kv_bytes);
                        if info.is_moe() {
                            // MoE: dense + KV stay on GPU, experts split via
                            // --n-cpu-moe. Auto-tune N to pack as many experts
                            // into the preferred backend's free VRAM as fit,
                            // unless the operator pinned n_cpu_moe.
                            let reserved =
                                self.reserved_vram.load(std::sync::atomic::Ordering::SeqCst);
                            let backend =
                                target_backends.first().copied().unwrap_or(Backend::Vulkan);
                            let dense = info.dense_weight_bytes();
                            let free: u64 = subtract_reserved_from_gpus(gpus.clone(), reserved)
                                .iter()
                                .filter(|g| !g.integrated && g.supports(backend))
                                .map(|g| g.allocatable_vram())
                                .sum();
                            let n_cpu_moe = model.n_cpu_moe.unwrap_or_else(|| {
                                auto_n_cpu_moe(
                                    dense,
                                    info.expert_weight_bytes,
                                    kv,
                                    info.n_layers,
                                    free,
                                )
                                .unwrap_or(info.n_layers)
                            });
                            // Force all dense/attention onto the GPU; the CPU
                            // experts are pinned via an *interleaved*
                            // --override-tensor so VRAM fills evenly across GPUs
                            // (not clustered like --n-cpu-moe). Applied to this
                            // spawn only.
                            model.n_gpu_layers = Some(99);
                            model.n_cpu_moe = Some(n_cpu_moe);
                            moe_n_layers = Some(info.n_layers);
                            // Globally-spread override as a fallback; replaced
                            // with a split-aligned one after placement below.
                            model.override_tensor = crate::vram::estimator::moe_cpu_override_tensor(
                                info.n_layers,
                                n_cpu_moe,
                            );
                            let est = VramEstimate::compute_moe(
                                dense,
                                info.expert_weight_bytes,
                                kv,
                                info.n_layers,
                                n_cpu_moe,
                            );
                            model.estimated_vram = est.total_vram;
                            info!(
                                model = id,
                                dense_gib = dense >> 30,
                                experts_gib = info.expert_weight_bytes >> 30,
                                n_layers = info.n_layers,
                                n_cpu_moe,
                                est_gib = est.total_vram >> 30,
                                override_tensor = ?model.override_tensor,
                                "MoE offload plan (experts on CPU, interleaved across layers)"
                            );
                        } else {
                            let est = VramEstimate::compute(
                                info.file_size,
                                kv,
                                info.n_layers,
                                model.n_gpu_layers,
                            );
                            model.estimated_vram = est.total_vram;
                        }
                    }
                    Err(e) => {
                        warn!(model = id, error = %e, "gguf parse failed; loading without estimate")
                    }
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
                            let est = VramEstimate::compute(
                                info.file_size,
                                kv,
                                info.n_layers,
                                d.n_gpu_layers,
                            );
                            model.estimated_vram =
                                model.estimated_vram.saturating_add(est.total_vram);
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

            // Place the model on its backend's GPUs, evicting idle models and
            // retrying once if it doesn't fit. `try_place` sets an explicit
            // `--device` + aligned `--tensor-split` for auto-placed GGUF models.
            // Reserved VRAM (concurrent in-flight loads whose weights haven't
            // faulted in yet) is subtracted inside `try_place`.
            let already_reserved = self.reserved_vram.load(std::sync::atomic::Ordering::SeqCst);
            match try_place(&mut model, &target_backends, &gpus, already_reserved) {
                PlaceOutcome::Placed {
                    backend,
                    gpus_used,
                } => {
                    info!(
                        model = id,
                        ?backend,
                        gpus_used,
                        device = ?model.device,
                        tensor_split = ?model.tensor_split,
                        "auto-placed on backend devices"
                    );
                }
                PlaceOutcome::Fits => {}
                PlaceOutcome::DoesNotFit { free } => {
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
                    match try_place(&mut model, &target_backends, &gpus_after, already_reserved) {
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

            // Placement set the tensor-split; rebuild the MoE expert offload so
            // each GPU's layer range gets its proportional share offloaded and
            // VRAM fills to every GPU's cap evenly (not one capping out first).
            if let (Some(nl), Some(ncpu), Some(ts)) = (
                moe_n_layers,
                model.n_cpu_moe,
                model.tensor_split.as_deref(),
            ) {
                let fracs: Vec<f64> = ts.split(',').filter_map(|s| s.trim().parse().ok()).collect();
                let layers =
                    crate::vram::estimator::boundary_aware_cpu_moe_layers(nl, ncpu, &fracs);
                if let Some(ot) = crate::vram::estimator::cpu_moe_override_from_layers(&layers) {
                    model.override_tensor = Some(ot);
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
            Ok(kv_bytes) => {
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
                        if kv_bytes > 0 && weight_file_size > 0 {
                            m.estimated_vram = ((weight_file_size + kv_bytes) as f64 * 1.1) as u64;
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
            match try_place(&mut model, &target_backends, &gpus, already_reserved) {
                PlaceOutcome::Placed {
                    backend,
                    gpus_used,
                } => {
                    info!(
                        model = id,
                        ?backend,
                        gpus_used,
                        device = ?model.device,
                        "scaling out onto free backend devices"
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
        || old.n_gpu_layers != new.n_gpu_layers
        || old.mlock != new.mlock
        || old.no_mmap != new.no_mmap
        || old.parallel_slots != new.parallel_slots
        || old.cache_type_k != new.cache_type_k
        || old.cache_type_v != new.cache_type_v
        || old.split_mode != new.split_mode
        || old.main_gpu != new.main_gpu
        || old.tensor_split != new.tensor_split
        || old.threads != new.threads
        || old.cache_ram_mib != new.cache_ram_mib
        || old.reasoning_format != new.reasoning_format
        || old.reasoning_budget != new.reasoning_budget
        || old.chat_template_kwargs != new.chat_template_kwargs
        || old.device != new.device
        || old.draft_model_id != new.draft_model_id
        || old.mtp_tokens != new.mtp_tokens
        || old.draft_max != new.draft_max
        || old.draft_min != new.draft_min
        || old.draft_p_min != new.draft_p_min
        || old.ctx_checkpoints != new.ctx_checkpoints
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
    /// GGUF auto-placement succeeded; `model.device`/`tensor_split` were set.
    Placed { backend: Backend, gpus_used: usize },
    /// Explicit-device or non-GGUF model that fits by total free VRAM.
    Fits,
    /// Doesn't fit anywhere; `free` is the best-case free VRAM seen across the
    /// model's target backends (for the error message).
    DoesNotFit { free: u64 },
}

/// Decide where `model` runs on `gpus` (reserved VRAM subtracted inside).
///
/// For an auto-placed GGUF model, walks `targets` in priority order and, on the
/// first backend whose tagged GPUs hold the model, sets an explicit `--device`
/// list + a `--tensor-split` aligned to it. Because the device list is explicit,
/// placement is correct even when backends enumerate GPUs in different orders.
/// Explicit-device and non-GGUF models keep their config and are only VRAM-checked.
fn try_place(
    model: &mut ModelConfig,
    targets: &[Backend],
    gpus: &[GpuInfo],
    reserved: u64,
) -> PlaceOutcome {
    let adjusted = subtract_reserved_from_gpus(gpus.to_vec(), reserved);

    // Explicit pin or non-GGUF: trust the config, just check it fits.
    if model.weights_format != WeightsFormat::Gguf
        || configured_llama_device_value(model).is_some()
    {
        let eligible = model_visible_gpus(model, adjusted);
        let free: u64 = eligible.iter().map(|g| g.allocatable_vram()).sum();
        return if model.estimated_vram == 0 || free >= model.estimated_vram {
            PlaceOutcome::Fits
        } else {
            PlaceOutcome::DoesNotFit { free }
        };
    }

    // No estimate (gguf parse failed): can't plan; let the backend default place it.
    if model.estimated_vram == 0 {
        return PlaceOutcome::Fits;
    }

    let mut best_free = 0u64;
    for &backend in targets {
        let candidates: Vec<GpuInfo> = adjusted
            .iter()
            .filter(|g| !g.integrated && g.supports(backend))
            .cloned()
            .collect();
        best_free = best_free.max(candidates.iter().map(|g| g.allocatable_vram()).sum());
        if let Some(p) = plan_backend_split(backend, &candidates, model.estimated_vram) {
            model.device = Some(p.device);
            model.tensor_split = Some(p.tensor_split);
            if p.gpus_used > 1 && model.split_mode.is_none() {
                model.split_mode = Some(SplitMode::Layer);
            }
            return PlaceOutcome::Placed {
                backend,
                gpus_used: p.gpus_used,
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
    let mut configured = model.device.clone();
    let args = &model.extra_args;
    for (idx, arg) in args.iter().enumerate() {
        if arg == "--device" || arg == "-dev" {
            configured = args.get(idx + 1).cloned();
            continue;
        }
        if let Some(value) = arg.strip_prefix("--device=") {
            configured = Some(value.into());
            continue;
        }
        if let Some(value) = arg.strip_prefix("-dev=") {
            configured = Some(value.into());
        }
    }
    configured
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
        strip_target_device_args(&mut model.extra_args);
        return;
    };
    strip_target_device_args(&mut model.extra_args);
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

fn strip_target_device_args(args: &mut Vec<String>) {
    let mut out = Vec::with_capacity(args.len());
    let mut idx = 0;
    while idx < args.len() {
        let arg = &args[idx];
        if arg == "--device" || arg == "-dev" {
            idx += 2;
            continue;
        }
        if arg.starts_with("--device=") || arg.starts_with("-dev=") {
            idx += 1;
            continue;
        }
        out.push(arg.clone());
        idx += 1;
    }
    *args = out;
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
mod tests {
    use super::*;
    use std::path::PathBuf;
    use tempfile::TempDir;

    fn orch(tmp: &TempDir) -> Arc<Orchestrator> {
        let store = Arc::new(JsonStore::<Vec<ModelConfig>>::new(
            tmp.path().join("models.json"),
        ));
        let presets = Arc::new(JsonStore::<Vec<BinaryPreset>>::new(
            tmp.path().join("presets.json"),
        ));
        let aliases = Arc::new(JsonStore::<Vec<ModelAlias>>::new(
            tmp.path().join("aliases.json"),
        ));
        Arc::new(Orchestrator::new(store, presets, aliases, 8080))
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

    #[test]
    fn default_instance_cap_is_vram_limited() {
        assert_eq!(DEFAULT_MAX_INSTANCES_PER_MODEL, usize::MAX);
    }

    fn gpu(id: &str, pci: &str, vulkan_index: usize) -> GpuInfo {
        GpuInfo {
            id: id.into(),
            pci_bus_id: Some(pci.into()),
            vulkan_device: Some(format!("Vulkan{vulkan_index}")),
            vulkan_index: Some(vulkan_index),
            cuda_device: None,
            cuda_index: None,
            rocm_index: None,
            sycl_index: None,
            tags: Default::default(),
            integrated: false,
            total_vram: 32 * 1024 * 1024 * 1024,
            used_vram: 0,
            busy_pct: 0,
            temp_c: None,
            display_attached: false,
        }
    }

    #[test]
    fn model_visible_gpus_filters_by_vulkan_or_pci_device() {
        let gpus = vec![gpu("1", "0000:03:00.0", 1), gpu("4", "0000:1b:00.0", 0)];
        let mut m = model("a");
        m.device = Some("pci:0000:1b:00.0".into());
        normalize_model_device_for_llama(&mut m, &gpus);
        assert_eq!(m.device.as_deref(), Some("Vulkan0"));

        let selected = model_visible_gpus(&m, gpus);
        assert_eq!(selected.len(), 1);
        assert_eq!(selected[0].pci_bus_id.as_deref(), Some("0000:1b:00.0"));
    }

    #[test]
    fn extra_args_device_wins_and_is_normalized_into_structured_device() {
        let gpus = vec![gpu("1", "0000:03:00.0", 1), gpu("4", "0000:1b:00.0", 0)];
        let mut m = model("a");
        m.device = Some("Vulkan1".into());
        m.extra_args = vec![
            "--device".into(),
            "pci:0000:1b:00.0".into(),
            "--threads".into(),
            "16".into(),
        ];

        normalize_model_device_for_llama(&mut m, &gpus);

        assert_eq!(m.device.as_deref(), Some("Vulkan0"));
        assert_eq!(m.extra_args, vec!["--threads", "16"]);
        let selected = model_visible_gpus(&m, gpus);
        assert_eq!(selected[0].vulkan_device.as_deref(), Some("Vulkan0"));
    }

    #[test]
    fn device_list_is_canonicalized_to_vulkan_order() {
        let gpus = vec![
            gpu("1", "0000:03:00.0", 1),
            gpu("3", "0000:0a:00.0", 3),
            gpu("4", "0000:1b:00.0", 0),
        ];
        let mut m = model("a");
        m.device = Some("Vulkan3,pci:0000:1b:00.0,Vulkan1".into());

        normalize_model_device_for_llama(&mut m, &gpus);

        assert_eq!(m.device.as_deref(), Some("Vulkan0,Vulkan1,Vulkan3"));
        let selected = model_visible_gpus(&m, gpus);
        assert_eq!(
            selected
                .iter()
                .map(|g| g.vulkan_device.as_deref().unwrap())
                .collect::<Vec<_>>(),
            vec!["Vulkan0", "Vulkan1", "Vulkan3"]
        );
    }

    #[test]
    fn model_visible_gpus_filters_by_cuda_device() {
        let gpus = vec![GpuInfo {
            id: "cuda0".into(),
            pci_bus_id: Some("0000:1c:00.0".into()),
            vulkan_device: None,
            vulkan_index: None,
            cuda_device: Some("CUDA0".into()),
            cuda_index: Some(0),
            rocm_index: None,
            sycl_index: None,
            tags: Default::default(),
            integrated: false,
            total_vram: 24 * 1024 * 1024 * 1024,
            used_vram: 0,
            busy_pct: 0,
            temp_c: None,
            display_attached: false,
        }];
        let mut m = model("cuda-target");
        m.device = Some("CUDA0".into());

        normalize_model_device_for_llama(&mut m, &gpus);

        assert_eq!(m.device.as_deref(), Some("CUDA0"));
        let selected = model_visible_gpus(&m, gpus);
        assert_eq!(selected.len(), 1);
        assert_eq!(selected[0].cuda_device.as_deref(), Some("CUDA0"));
    }

    #[test]
    fn automatic_vulkan_pool_excludes_accounting_only_gpus() {
        let mut raw = gpu("0", "0000:1e:00.0", 4);
        raw.vulkan_device = None;
        raw.vulkan_index = None;
        let amd = gpu("1", "0000:03:00.0", 0);

        let mut auto = model("auto");
        auto.binary_preset = Some("llama-vulkan".into());

        let visible = model_visible_gpus(&auto, vec![raw, amd]);

        assert_eq!(visible.len(), 1);
        assert_eq!(visible[0].pci_bus_id.as_deref(), Some("0000:03:00.0"));
    }

    #[test]
    fn automatic_cuda_pool_uses_cuda_devices() {
        let cuda = GpuInfo {
            id: "cuda0".into(),
            pci_bus_id: Some("0000:1c:00.0".into()),
            vulkan_device: None,
            vulkan_index: None,
            cuda_device: Some("CUDA0".into()),
            cuda_index: Some(0),
            rocm_index: None,
            sycl_index: None,
            tags: Default::default(),
            integrated: false,
            total_vram: 24 * 1024 * 1024 * 1024,
            used_vram: 0,
            busy_pct: 0,
            temp_c: None,
            display_attached: false,
        };
        let vulkan = gpu("1", "0000:03:00.0", 0);
        let mut auto = model("auto-cuda");
        auto.binary_preset = Some("llama-cuda".into());

        let visible = model_visible_gpus(&auto, vec![cuda, vulkan]);

        assert_eq!(visible.len(), 1);
        assert_eq!(visible[0].cuda_device.as_deref(), Some("CUDA0"));
    }

    #[test]
    fn integrated_gpu_excluded_from_auto_pool_but_selectable_explicitly() {
        let mut igpu = gpu("0", "0000:08:00.0", 2);
        igpu.integrated = true;
        let dgpu = gpu("1", "0000:03:00.0", 0);

        // No device configured → automatic placement: the iGPU is excluded.
        let auto = model("auto");
        let visible = model_visible_gpus(&auto, vec![igpu.clone(), dgpu.clone()]);
        assert_eq!(visible.len(), 1);
        assert_eq!(visible[0].pci_bus_id.as_deref(), Some("0000:03:00.0"));

        // Explicitly targeting the iGPU runs the model there.
        let mut targeted = model("on-igpu");
        targeted.device = Some("Vulkan2".into());
        let visible = model_visible_gpus(&targeted, vec![igpu, dgpu]);
        assert_eq!(visible.len(), 1);
        assert!(visible[0].integrated);
        assert_eq!(visible[0].pci_bus_id.as_deref(), Some("0000:08:00.0"));
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
        assert!(
            o.dirty.load(Ordering::Relaxed),
            "mark_used must mark dirty so reconcile persists"
        );
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
        o.load_guards
            .lock()
            .await
            .entry("a".into())
            .or_insert_with(|| Arc::new(Mutex::new(())));
        assert!(o.load_guards.lock().await.contains_key("a"));

        o.remove_model("a").await.unwrap();
        assert!(
            !o.load_guards.lock().await.contains_key("a"),
            "load_guards must drop entries for removed models so the map doesn't grow forever"
        );
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
    async fn ensure_loaded_reuses_busy_instance_after_waiting_on_load_guard() {
        let tmp = tempfile::tempdir().unwrap();
        let o = orch(&tmp);
        o.add_model(model("a")).await.unwrap();

        let guard = Arc::new(Mutex::new(()));
        let locked = guard.lock().await;
        o.load_guards.lock().await.insert("a".into(), guard.clone());

        let ensure = {
            let o = o.clone();
            tokio::spawn(async move { o.ensure_loaded("a").await })
        };

        tokio::time::sleep(std::time::Duration::from_millis(20)).await;
        o.process_manager
            .lock()
            .await
            .register_test_instance("a", -1, 9000);
        let _busy = o
            .process_manager
            .lock()
            .await
            .acquire_idle_instance("a")
            .unwrap();

        drop(locked);

        let reused = ensure.await.unwrap().unwrap();
        assert_eq!(reused.port, 9000);
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
    async fn update_running_model_name_only_keeps_running() {
        let tmp = tempfile::tempdir().unwrap();
        let o = orch(&tmp);
        let mut m = model("a");
        m.state = ModelState::Running;
        m.pid = Some(123);
        o.add_model(m.clone()).await.unwrap();

        m.name = "renamed".into();
        o.update_model(m).await.unwrap();

        let after = o.get_model("a").await.unwrap();
        assert_eq!(after.name, "renamed");
        assert_eq!(after.state, ModelState::Running);
        assert_eq!(after.pid, Some(123));
    }

    #[tokio::test]
    async fn update_running_model_spawn_change_stops_to_apply_new_config() {
        let tmp = tempfile::tempdir().unwrap();
        let o = orch(&tmp);
        let mut m = model("a");
        m.state = ModelState::Running;
        m.pid = Some(123);
        o.add_model(m.clone()).await.unwrap();

        m.context = 8192;
        o.update_model(m).await.unwrap();

        let after = o.get_model("a").await.unwrap();
        assert_eq!(after.context, 8192);
        assert_eq!(after.state, ModelState::Idle);
        assert_eq!(after.pid, None);
        assert_eq!(after.estimated_vram, 0);
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
        let presets = Arc::new(JsonStore::<Vec<BinaryPreset>>::new(
            tmp.path().join("presets.json"),
        ));
        let aliases = Arc::new(JsonStore::<Vec<ModelAlias>>::new(
            tmp.path().join("aliases.json"),
        ));
        let o = Arc::new(Orchestrator::new(store, presets, aliases, 8080));
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
        o.process_manager
            .lock()
            .await
            .register_test_instance("a", 999_999, 9000);

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
    async fn reserved_vram_prevents_concurrent_overcommit() {
        // Simulates the race: two concurrent do_load calls both see enough
        // free VRAM individually, but together would OOM. The reservation
        // counter must block the second one.
        let tmp = tempfile::tempdir().unwrap();
        let o = orch(&tmp);

        // Inject a fake GPU with 10 GiB free.
        {
            let mut data = o.data.lock().await;
            data.gpus = vec![GpuInfo {
                id: "card0".into(),
                pci_bus_id: None,
                vulkan_device: None,
                vulkan_index: None,
                cuda_device: None,
                cuda_index: None,
                rocm_index: None,
                sycl_index: None,
                tags: Default::default(),
                integrated: false,
                total_vram: 10 * 1024 * 1024 * 1024,
                used_vram: 0,
                busy_pct: 0,
                temp_c: None,
                display_attached: false,
            }];
        }

        // Model A needs 6 GiB; model B needs 6 GiB. Together they need 12 GiB > 10 GiB.
        let six_gib: u64 = 6 * 1024 * 1024 * 1024;

        // Simulate: model A has been fork/exec'd (reservation active) but
        // sysfs hasn't caught up yet (vram_used still 0).
        o.reserved_vram
            .store(six_gib, std::sync::atomic::Ordering::SeqCst);

        // Now the admission logic for model B should see only 4 GiB free.
        let already_reserved = o.reserved_vram.load(std::sync::atomic::Ordering::SeqCst);
        let data = o.data.lock().await;
        let raw_free: u64 = data.gpus.iter().map(|g| g.free_vram()).sum();
        let effective_free = raw_free.saturating_sub(already_reserved);
        drop(data);

        assert_eq!(
            raw_free,
            10 * 1024 * 1024 * 1024,
            "sysfs still shows 10 GiB"
        );
        assert_eq!(
            effective_free,
            4 * 1024 * 1024 * 1024,
            "but effective free is only 4 GiB"
        );
        assert!(
            effective_free < six_gib,
            "model B (6 GiB) must be rejected when only 4 GiB is effectively available"
        );
    }

    #[tokio::test]
    async fn reconcile_persists_only_when_dirty() {
        let tmp = tempfile::tempdir().unwrap();
        let path = tmp.path().join("models.json");
        let store = Arc::new(JsonStore::<Vec<ModelConfig>>::new(path.clone()));
        let presets = Arc::new(JsonStore::<Vec<BinaryPreset>>::new(
            tmp.path().join("presets.json"),
        ));
        let aliases = Arc::new(JsonStore::<Vec<ModelAlias>>::new(
            tmp.path().join("aliases.json"),
        ));
        let o = Arc::new(Orchestrator::new(store, presets, aliases, 8080));

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

    #[tokio::test]
    async fn reconcile_persists_settings_when_dirty() {
        let tmp = tempfile::tempdir().unwrap();
        let settings_path = tmp.path().join("settings.json");
        let store = Arc::new(JsonStore::<Vec<ModelConfig>>::new(
            tmp.path().join("models.json"),
        ));
        let presets = Arc::new(JsonStore::<Vec<BinaryPreset>>::new(
            tmp.path().join("presets.json"),
        ));
        let aliases = Arc::new(JsonStore::<Vec<ModelAlias>>::new(
            tmp.path().join("aliases.json"),
        ));
        let settings = Arc::new(JsonStore::<AppSettings>::new(settings_path.clone()));
        let gpu_tags = Arc::new(JsonStore::<Vec<GpuTagOverride>>::new(
            tmp.path().join("gpus.json"),
        ));
        let o = Arc::new(Orchestrator::new_with_settings_store(
            store, presets, aliases, settings, gpu_tags, 8080,
        ));

        let mut next = o.settings().await;
        next.loop_guards.streaming.repeats = 7;
        next.loop_guards.streaming.action = crate::config::StreamingLoopAction::Log;
        next.loop_guards.tool.window_messages = 24;
        o.update_settings(next).await;
        o.reconcile().await;

        let saved: AppSettings =
            serde_json::from_str(&std::fs::read_to_string(&settings_path).unwrap()).unwrap();
        assert_eq!(saved.loop_guards.streaming.repeats, 7);
        assert_eq!(
            saved.loop_guards.streaming.action,
            crate::config::StreamingLoopAction::Log,
        );
        assert_eq!(saved.loop_guards.tool.window_messages, 24);
    }

    fn alias(name: &str, target: &str) -> ModelAlias {
        ModelAlias {
            alias: name.into(),
            target: target.into(),
        }
    }

    #[tokio::test]
    async fn add_alias_requires_existing_target() {
        let tmp = tempfile::tempdir().unwrap();
        let o = orch(&tmp);
        // Target model does not exist yet.
        let err = o.add_alias(alias("fast", "qwen")).await.unwrap_err();
        assert!(matches!(err, MutationError::AliasTargetMissing { .. }));

        o.add_model(model("qwen")).await.unwrap();
        o.add_alias(alias("fast", "qwen")).await.unwrap();
        assert_eq!(o.list_aliases().await.len(), 1);
    }

    #[tokio::test]
    async fn alias_resolves_to_target_and_passthrough_for_unknown() {
        let tmp = tempfile::tempdir().unwrap();
        let o = orch(&tmp);
        o.add_model(model("qwen")).await.unwrap();
        o.add_alias(alias("fast", "qwen")).await.unwrap();

        assert_eq!(o.resolve_model_id("fast").await, "qwen");
        // A non-alias name passes through unchanged.
        assert_eq!(o.resolve_model_id("qwen").await, "qwen");
        assert_eq!(o.resolve_model_id("unknown").await, "unknown");
    }

    #[tokio::test]
    async fn alias_cannot_shadow_model_id_or_duplicate() {
        let tmp = tempfile::tempdir().unwrap();
        let o = orch(&tmp);
        o.add_model(model("qwen")).await.unwrap();
        o.add_model(model("llama")).await.unwrap();

        // Alias name equal to an existing model id is rejected.
        let err = o.add_alias(alias("llama", "qwen")).await.unwrap_err();
        assert!(matches!(err, MutationError::AliasShadowsModel(_)));

        o.add_alias(alias("fast", "qwen")).await.unwrap();
        // Duplicate alias name is rejected.
        let err = o.add_alias(alias("fast", "llama")).await.unwrap_err();
        assert!(matches!(err, MutationError::AliasConflict(_)));
    }

    #[tokio::test]
    async fn deleting_model_unassigns_aliases_but_keeps_them() {
        let tmp = tempfile::tempdir().unwrap();
        let o = orch(&tmp);
        o.add_model(model("qwen")).await.unwrap();
        o.add_model(model("llama")).await.unwrap();
        o.add_alias(alias("fast", "qwen")).await.unwrap();
        o.add_alias(alias("big", "llama")).await.unwrap();

        o.remove_model("qwen").await.unwrap();

        // Both aliases survive; the one pointing at "qwen" is now unassigned.
        let aliases = o.list_aliases().await;
        assert_eq!(aliases, vec![alias("big", "llama"), alias("fast", "")]);
        // An unassigned alias resolves to an empty target (handled as an error
        // at the proxy layer).
        assert_eq!(o.resolve_model_id("fast").await, "");
    }

    #[tokio::test]
    async fn add_alias_rejects_invalid_names() {
        let tmp = tempfile::tempdir().unwrap();
        let o = orch(&tmp);
        o.add_model(model("qwen")).await.unwrap();
        let too_long = "x".repeat(65);
        for bad in ["Planner", "my alias", "a/b", "café", "", too_long.as_str()] {
            let err = o.add_alias(alias(bad, "qwen")).await.unwrap_err();
            assert!(
                matches!(err, MutationError::AliasInvalid(_)),
                "expected {bad:?} to be rejected as invalid",
            );
        }
        // Canonical names are accepted.
        o.add_alias(alias("gpt-4o.fast_v2", "qwen")).await.unwrap();
    }

    #[tokio::test]
    async fn alias_can_target_another_alias_and_follows_repoints() {
        let tmp = tempfile::tempdir().unwrap();
        let o = orch(&tmp);
        o.add_model(model("qwen")).await.unwrap();
        o.add_model(model("llama")).await.unwrap();
        o.add_alias(alias("default", "qwen")).await.unwrap();
        // Several aliases reference `default` instead of a concrete model.
        o.add_alias(alias("coder", "default")).await.unwrap();
        o.add_alias(alias("planner", "default")).await.unwrap();

        assert_eq!(o.resolve_model_id("coder").await, "qwen");
        assert_eq!(o.resolve_model_id("planner").await, "qwen");

        // Repointing `default` propagates to every alias that references it,
        // without touching those aliases.
        o.update_alias(alias("default", "llama")).await.unwrap();
        assert_eq!(o.resolve_model_id("coder").await, "llama");
        assert_eq!(o.resolve_model_id("planner").await, "llama");
        // The referencing aliases still store the reference, not the model.
        let coder = o
            .list_aliases()
            .await
            .into_iter()
            .find(|a| a.alias == "coder")
            .unwrap();
        assert_eq!(coder.target, "default");
    }

    #[tokio::test]
    async fn alias_chain_through_unassigned_resolves_empty() {
        let tmp = tempfile::tempdir().unwrap();
        let o = orch(&tmp);
        o.add_alias(alias("default", "")).await.unwrap(); // unassigned
        o.add_alias(alias("coder", "default")).await.unwrap();
        assert_eq!(o.resolve_model_id("coder").await, "");
    }

    #[tokio::test]
    async fn alias_cycles_are_rejected() {
        let tmp = tempfile::tempdir().unwrap();
        let o = orch(&tmp);
        o.add_model(model("qwen")).await.unwrap();
        o.add_alias(alias("a", "qwen")).await.unwrap();
        o.add_alias(alias("b", "a")).await.unwrap();
        // a → b would close the loop a → b → a.
        let err = o.update_alias(alias("a", "b")).await.unwrap_err();
        assert!(matches!(err, MutationError::AliasCycle { .. }));
        // Resolution is unaffected (the cyclic update was rejected).
        assert_eq!(o.resolve_model_id("b").await, "qwen");
    }

    #[tokio::test]
    async fn deleting_alias_unassigns_aliases_referencing_it() {
        let tmp = tempfile::tempdir().unwrap();
        let o = orch(&tmp);
        o.add_model(model("qwen")).await.unwrap();
        o.add_alias(alias("default", "qwen")).await.unwrap();
        o.add_alias(alias("coder", "default")).await.unwrap();

        o.remove_alias("default").await.unwrap();
        let coder = o
            .list_aliases()
            .await
            .into_iter()
            .find(|a| a.alias == "coder")
            .unwrap();
        assert_eq!(
            coder.target, "",
            "coder should be unassigned after default is deleted"
        );
    }

    #[tokio::test]
    async fn alias_can_be_created_unassigned_then_pointed_at_a_model() {
        let tmp = tempfile::tempdir().unwrap();
        let o = orch(&tmp);
        // Canonical interface name defined before any model is assigned.
        o.add_alias(alias("planner", "")).await.unwrap();
        assert_eq!(o.resolve_model_id("planner").await, "");

        o.add_model(model("qwen")).await.unwrap();
        o.update_alias(alias("planner", "qwen")).await.unwrap();
        assert_eq!(o.resolve_model_id("planner").await, "qwen");

        // Reassigning to a non-existent model is still rejected.
        let err = o.update_alias(alias("planner", "ghost")).await.unwrap_err();
        assert!(matches!(err, MutationError::AliasTargetMissing { .. }));
    }

    #[tokio::test]
    async fn reconcile_persists_aliases_when_dirty() {
        let tmp = tempfile::tempdir().unwrap();
        let aliases_path = tmp.path().join("aliases.json");
        let o = orch(&tmp);
        o.add_model(model("qwen")).await.unwrap();
        o.add_alias(alias("fast", "qwen")).await.unwrap();
        o.reconcile().await;

        let saved: Vec<ModelAlias> =
            serde_json::from_str(&std::fs::read_to_string(&aliases_path).unwrap()).unwrap();
        assert_eq!(saved, vec![alias("fast", "qwen")]);
    }
}
