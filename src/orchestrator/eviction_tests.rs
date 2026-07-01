use super::*;
use std::path::PathBuf;

fn running(id: &str, vram: u64, idle_secs: f64) -> ModelConfig {
    let now = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap()
        .as_secs_f64();
    ModelConfig {
        id: id.into(),
        name: id.into(),
        binary: PathBuf::from("/b"),
        model_path: PathBuf::from("/m"),
        state: ModelState::Running,
        pid: Some(1),
        estimated_vram: vram,
        last_used: Some(now - idle_secs),
        ..ModelConfig::default()
    }
}

#[test]
fn no_eviction_when_enough_free() {
    let mut models = HashMap::new();
    models.insert("a".into(), running("a", 5_000_000_000, 100.0));
    let idle = HashSet::from(["a".to_string()]);
    assert!(decide_eviction(&models, 64_000_000_000, 1_000_000_000, &idle).is_empty());
}

#[test]
fn evicts_smallest_first_when_idle_equal() {
    let mut models = HashMap::new();
    models.insert("big".into(), running("big", 30_000_000_000, 100.0));
    models.insert("small".into(), running("small", 5_000_000_000, 100.0));
    let idle = HashSet::from(["big".to_string(), "small".to_string()]);
    let actions = decide_eviction(&models, 9_000_000_000, 15_000_000_000, &idle);
    // 9GB free, deficit = 6GB, small alone (5GB) isn't enough so both are evicted.
    assert_eq!(actions[0], EvictionAction::Evict("small".into()));
    assert_eq!(actions[1], EvictionAction::Evict("big".into()));
}

#[test]
fn skips_non_running_models() {
    let mut models = HashMap::new();
    let mut idle = running("idle", 5_000_000_000, 100.0);
    idle.state = ModelState::Idle;
    models.insert("idle".into(), idle);
    models.insert("run".into(), running("run", 5_000_000_000, 100.0));
    let idle_models = HashSet::from(["run".to_string()]);
    let actions = decide_eviction(&models, 4_000_000_000, 10_000_000_000, &idle_models);
    let ids: Vec<&str> = actions
        .iter()
        .map(|EvictionAction::Evict(id)| id.as_str())
        .collect();
    assert_eq!(ids, vec!["run"]);
}

#[test]
fn skips_running_models_without_vram_estimate() {
    // Safetensors models frequently have estimated_vram == 0. A Running
    // model with a zero estimate can't satisfy any deficit, so we must
    // exclude it from eviction candidates.
    let mut models = HashMap::new();
    let mut unknown = running("unknown", 0, 100.0);
    unknown.state = ModelState::Running;
    models.insert("unknown".into(), unknown);
    let idle = HashSet::from(["unknown".to_string()]);
    let actions = decide_eviction(&models, 1_000_000_000, 10_000_000_000, &idle);
    assert!(
        actions.is_empty(),
        "must not evict model with unknown VRAM: {actions:?}"
    );
}

#[test]
fn skips_active_models() {
    let mut models = HashMap::new();
    models.insert("active".into(), running("active", 20_000_000_000, 100.0));
    models.insert("idle".into(), running("idle", 20_000_000_000, 100.0));
    let idle = HashSet::from(["idle".to_string()]);
    let actions = decide_eviction(&models, 1_000_000_000, 10_000_000_000, &idle);
    assert_eq!(actions, vec![EvictionAction::Evict("idle".into())]);
}

#[test]
fn never_used_model_has_infinite_score() {
    let mut m = running("never", 10_000_000_000, 0.0);
    m.last_used = None;
    assert!(eviction_score(&m).is_infinite());
}
