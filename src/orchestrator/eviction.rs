use crate::config::{ModelConfig, ModelState};
use std::collections::{HashMap, HashSet};
use tracing::debug;

/// A Running model can be evicted only if we know how much VRAM it holds.
/// Without an estimate we'd count it as freeing zero bytes, and the
/// eviction loop would burn through victims without ever satisfying the
/// deficit. Safetensors models (no GGUF metadata) are the realistic
/// offenders here.
fn is_evictable(m: &ModelConfig, idle_models: &HashSet<String>) -> bool {
    m.state == ModelState::Running && m.estimated_vram > 0 && idle_models.contains(&m.id)
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum EvictionAction {
    Evict(String),
}

/// Eviction priority. Higher = evict first.
///
/// Bias toward idle, small models: long-idle models are the safest to drop,
/// and all else equal we'd rather evict a small one and keep larger loaded
/// models resident (cheaper to re-spawn the small one later).
pub fn eviction_score(model: &ModelConfig) -> f64 {
    let idle = match model.last_used {
        Some(t) => {
            let now = std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .map(|d| d.as_secs_f64())
                .unwrap_or(0.0);
            (now - t).max(0.0)
        }
        None => return f64::INFINITY,
    };

    let gib = model.estimated_vram as f64 / (1024.0 * 1024.0 * 1024.0);
    (idle + 1.0).ln() + 1.0 / (gib + 1.0).log2()
}

/// Decide which running models to evict to free `needed` bytes of VRAM.
///
/// Returns an empty vec when no eviction is needed.
pub fn decide_eviction(
    models: &HashMap<String, ModelConfig>,
    free_vram: u64,
    needed: u64,
    idle_models: &HashSet<String>,
) -> Vec<EvictionAction> {
    if free_vram >= needed {
        return Vec::new();
    }

    let deficit = needed - free_vram;
    let mut candidates: Vec<&ModelConfig> = models
        .values()
        .filter(|m| is_evictable(m, idle_models))
        .collect();
    candidates.sort_by(|a, b| {
        eviction_score(b)
            .partial_cmp(&eviction_score(a))
            .unwrap_or(std::cmp::Ordering::Equal)
    });

    let mut freed = 0u64;
    let mut actions = Vec::new();
    for c in candidates {
        if freed >= deficit {
            break;
        }
        actions.push(EvictionAction::Evict(c.id.clone()));
        freed += c.estimated_vram;
    }
    if freed < deficit {
        debug!(
            deficit,
            freed, "not enough candidates to satisfy VRAM deficit"
        );
    }
    actions
}

#[cfg(test)]
mod tests {
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
}
