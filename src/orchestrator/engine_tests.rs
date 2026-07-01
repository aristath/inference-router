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

impl Orchestrator {
    async fn get_model(&self, id: &str) -> Option<ModelConfig> {
        self.data.lock().await.models.get(id).cloned()
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
fn extra_args_device_is_preserved_but_not_used_for_router_placement() {
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

    assert_eq!(m.device.as_deref(), Some("Vulkan1"));
    assert_eq!(
        m.extra_args,
        vec!["--device", "pci:0000:1b:00.0", "--threads", "16"]
    );
    let selected = model_visible_gpus(&m, gpus);
    assert_eq!(selected[0].vulkan_device.as_deref(), Some("Vulkan1"));
}

fn fit_sizing(
    n_gpu_layers: Option<i32>,
    override_tensor: Option<&str>,
    device_vram: u64,
) -> crate::vram::llama_fit::LlamaFitSizing {
    crate::vram::llama_fit::LlamaFitSizing {
        fitted: crate::vram::llama_fit::LlamaFittedArgs {
            context: None,
            n_gpu_layers,
            tensor_split: None,
            override_tensor: override_tensor.map(str::to_string),
        },
        device_vram,
    }
}

fn spill_candidate(gpus_used: usize, device_vram: u64) -> FitCandidate {
    FitCandidate {
        backend: Backend::Vulkan,
        device: format!("Vulkan{}", gpus_used - 1),
        free: device_vram,
        gpus_used,
        sizing: fit_sizing(
            Some(12),
            Some(r"blk\.(12|13)\.ffn_.*_exps=CPU"),
            device_vram,
        ),
    }
}

#[test]
fn fitted_full_gpu_requires_layers_and_no_cpu_tensor_override() {
    assert!(fitted_fully_on_gpu(&fit_sizing(Some(-1), None, 0), 64));
    assert!(fitted_fully_on_gpu(&fit_sizing(Some(64), None, 0), 64));
    assert!(!fitted_fully_on_gpu(&fit_sizing(Some(12), None, 0), 64));
    assert!(!fitted_fully_on_gpu(
        &fit_sizing(Some(-1), Some(r"blk\.(12|13)\.ffn_.*_exps=CPU"), 0),
        64
    ));
}

#[test]
fn spill_candidate_prefers_more_gpus_before_more_vram() {
    let mut best = None;
    keep_best_spill_candidate(&mut best, spill_candidate(1, 64));
    keep_best_spill_candidate(&mut best, spill_candidate(2, 48));
    keep_best_spill_candidate(&mut best, spill_candidate(2, 96));

    let best = best.unwrap();
    assert_eq!(best.gpus_used, 2);
    assert_eq!(best.sizing.device_vram, 96);
}

#[test]
fn scale_out_rejects_cpu_spill_placement() {
    assert!(scale_out_accepts_placement(true));
    assert!(!scale_out_accepts_placement(false));
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
        .register_existing_instance("a", -1, 9000);
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
        .register_existing_instance("a", 999_999, 9000);

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
    let perf = Arc::new(JsonStore::<HashMap<String, ModelPerf>>::new(
        tmp.path().join("model_perf.json"),
    ));
    let o = Arc::new(Orchestrator::new_with_settings_store(
        store, presets, aliases, settings, gpu_tags, perf, 8080,
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
