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
#[path = "eviction_tests.rs"]
mod tests;
