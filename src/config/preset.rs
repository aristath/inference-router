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
        let hint = format!("{} {} {}", self.id, self.name, self.binary.to_string_lossy());
        Backend::infer_from_text(&hint).into_iter().collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn roundtrips_through_serde() {
        let p = BinaryPreset {
            id: "llama-vulkan".into(),
            name: "llama.cpp (Vulkan)".into(),
            binary: PathBuf::from("/home/aristath/llama.cpp/build-vulkan/bin/llama-server"),
            targets: vec![Backend::Vulkan],
        };
        let s = serde_json::to_string(&p).unwrap();
        let back: BinaryPreset = serde_json::from_str(&s).unwrap();
        assert_eq!(p, back);
    }

    #[test]
    fn legacy_preset_without_targets_infers_one_from_path() {
        let json = r#"{"id":"llama-rocm","name":"llama.cpp (ROCm)","binary":"/x/llama.cpp-rocm/extracted/llama-server"}"#;
        let p: BinaryPreset = serde_json::from_str(json).unwrap();
        assert!(p.targets.is_empty());
        assert_eq!(p.effective_targets(), vec![Backend::Rocm]);
    }
}
