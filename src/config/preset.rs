use serde::{Deserialize, Serialize};
use std::path::PathBuf;

use crate::config::Backend;

/// A named inference-server binary. Models reference presets by `id`; changing
/// a preset's `binary` automatically propagates to every model using it.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct BinaryPreset {
    pub id: String,
    pub name: String,
    pub binary: PathBuf,
    /// Backends this binary can drive, in priority order. A model on this preset
    /// is placed on the first target whose tagged GPUs have room (so a
    /// `[Rocm, Vulkan]` preset prefers ROCm and falls back to Vulkan). The
    /// chosen backend also fixes the device *ordering* used to emit
    /// `--device` / `--tensor-split`. Empty = legacy preset; callers fall back
    /// to inferring a single backend from the binary path.
    #[serde(default)]
    pub targets: Vec<Backend>,
}

impl BinaryPreset {
    /// Effective ordered target backends: the explicit `targets`, or a single
    /// backend inferred from the id/name/binary for presets saved before the
    /// field existed.
    pub fn effective_targets(&self) -> Vec<Backend> {
        if !self.targets.is_empty() {
            return self.targets.clone();
        }
        let hint = format!(
            "{} {} {}",
            self.id,
            self.name,
            self.binary.to_string_lossy()
        );
        Backend::infer_from_text(&hint).into_iter().collect()
    }
}

#[cfg(test)]
#[path = "preset_tests.rs"]
mod tests;
