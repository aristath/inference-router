use serde::{Deserialize, Serialize};
use std::path::PathBuf;

/// A named inference-server binary. Models reference presets by `id`; changing
/// a preset's `binary` automatically propagates to every model using it.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct BinaryPreset {
    pub id: String,
    pub name: String,
    pub binary: PathBuf,
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
        };
        let s = serde_json::to_string(&p).unwrap();
        let back: BinaryPreset = serde_json::from_str(&s).unwrap();
        assert_eq!(p, back);
    }
}
