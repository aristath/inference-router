use serde::{Deserialize, Serialize};

/// A GGML compute backend (compute API). Every physical GPU is *tagged* with the
/// backends that can drive it (capability), and every binary preset *targets* a
/// set of backends. A model's preset → targets → the GPUs eligible to run it.
///
/// Crucially, each backend has its own device *ordering* (HIP order for ROCm,
/// CUDA order for CUDA, Vulkan order for Vulkan). Positional backend flags like
/// `--tensor-split` / `--device` MUST be expressed in the active backend's order
/// — never a different backend's. This enum is the anchor for that ordering.
#[derive(
    Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize,
)]
#[serde(rename_all = "lowercase")]
pub enum Backend {
    Vulkan,
    Cuda,
    Rocm,
    Sycl,
}

impl Backend {
    /// All backends, in a stable order (used to seed UI tag lists).
    pub const ALL: [Backend; 4] = [Backend::Vulkan, Backend::Cuda, Backend::Rocm, Backend::Sycl];

    /// Lowercase tag string, matching the serde representation.
    pub fn as_str(self) -> &'static str {
        match self {
            Backend::Vulkan => "vulkan",
            Backend::Cuda => "cuda",
            Backend::Rocm => "rocm",
            Backend::Sycl => "sycl",
        }
    }

    /// The prefix llama.cpp uses for this backend's device names, e.g. `ROCm`
    /// in `ROCm0`, `CUDA` in `CUDA0`. Combine with the per-GPU backend index to
    /// build the `--device` value (`ROCm2`).
    pub fn device_prefix(self) -> &'static str {
        match self {
            Backend::Vulkan => "Vulkan",
            Backend::Cuda => "CUDA",
            Backend::Rocm => "ROCm",
            Backend::Sycl => "SYCL",
        }
    }

    /// Best-effort guess of the backend a binary preset targets, from its id /
    /// name / binary path (e.g. `llama-rocm`, `.../build-cuda/...`). Used only to
    /// seed `targets` for presets saved before this field existed.
    pub fn infer_from_text(text: &str) -> Option<Backend> {
        let t = text.to_ascii_lowercase();
        if t.contains("rocm") || t.contains("hip") {
            Some(Backend::Rocm)
        } else if t.contains("cuda") || t.contains("nvidia") {
            Some(Backend::Cuda)
        } else if t.contains("sycl") || t.contains("oneapi") {
            Some(Backend::Sycl)
        } else if t.contains("vulkan") {
            Some(Backend::Vulkan)
        } else {
            None
        }
    }
}
