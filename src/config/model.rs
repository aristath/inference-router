use serde::{Deserialize, Serialize};
use std::path::PathBuf;

fn default_temperature() -> f32 { 0.6 }
fn default_top_p() -> f32 { 0.95 }
fn default_top_k() -> i32 { 40 }
fn default_min_p() -> f32 { 0.0 }

/// Weights file format. Drives the argv style used when spawning the backend.
///
/// - `Gguf` → llama.cpp-style (`-m <file> -c <ctx> --port ...`)
/// - `Safetensors` → vLLM-style (`--model <dir> --port ... --max-model-len <ctx>`)
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "lowercase")]
pub enum WeightsFormat {
    Gguf,
    Safetensors,
}

impl Default for WeightsFormat {
    fn default() -> Self { Self::Gguf }
}

/// Runtime state of a model.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum ModelState {
    Idle,
    Loading,
    Running,
    Error(String),
}

impl Default for ModelState {
    fn default() -> Self { Self::Idle }
}

/// How llama.cpp splits the model across multiple GPUs.
/// Maps 1:1 to `--split-mode {none|layer|row}`.
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "lowercase")]
pub enum SplitMode {
    /// Single GPU only (model must fit).
    None,
    /// Pipeline parallelism — each GPU holds a contiguous range of layers.
    /// Default. Low inter-GPU traffic.
    Layer,
    /// Split each tensor row-wise across GPUs.
    Row,
    /// Tensor parallelism — split each tensor across GPUs and run all of them
    /// in parallel. Generally fastest when supported but bandwidth-sensitive.
    /// In current llama.cpp this is a distinct mode from `row`.
    Tensor,
}

/// KV-cache quantization. Applies to llama.cpp (`--cache-type-{k,v}`).
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "lowercase")]
pub enum CacheType {
    F16,
    Q8_0,
    Q4_0,
}

impl Default for CacheType {
    fn default() -> Self { Self::F16 }
}

impl CacheType {
    /// The literal string llama.cpp expects on the command line.
    pub fn as_arg(&self) -> &'static str {
        match self {
            Self::F16 => "f16",
            Self::Q8_0 => "q8_0",
            Self::Q4_0 => "q4_0",
        }
    }

    /// Approximate bytes per value for VRAM estimates.
    pub fn bytes_per_value(&self) -> f64 {
        match self {
            Self::F16 => 2.0,
            Self::Q8_0 => 1.0,
            Self::Q4_0 => 0.5,
        }
    }
}

/// Self-contained per-model configuration. No external framework entity.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct ModelConfig {
    pub id: String,
    pub name: String,

    pub weights_format: WeightsFormat,
    /// If set, looked up in the presets table at spawn time to get the actual
    /// binary path. If `None`, `binary` below is used verbatim. Lets you
    /// change a binary once (e.g. rebuild llama.cpp) and have every model
    /// pick it up.
    #[serde(default)]
    pub binary_preset: Option<String>,
    pub binary: PathBuf,
    pub model_path: PathBuf,
    pub port: u16,
    #[serde(default)]
    pub extra_args: Vec<String>,

    pub context: u32,
    #[serde(default = "default_temperature")]
    pub temperature: f32,
    #[serde(default = "default_top_p")]
    pub top_p: f32,
    #[serde(default = "default_top_k")]
    pub top_k: i32,
    #[serde(default = "default_min_p")]
    pub min_p: f32,

    // llama.cpp-specific backend flags. All optional; `None`/`false` means
    // "don't emit the flag, let llama.cpp pick its default".
    #[serde(default)]
    pub flash_attn: bool,
    #[serde(default)]
    pub n_gpu_layers: Option<u32>,
    #[serde(default)]
    pub mlock: bool,
    #[serde(default)]
    pub no_mmap: bool,
    #[serde(default)]
    pub parallel_slots: Option<u32>,
    #[serde(default)]
    pub cache_type_k: Option<CacheType>,
    #[serde(default)]
    pub cache_type_v: Option<CacheType>,

    /// `--split-mode`. When `None`, llama.cpp picks its default (`layer` for
    /// multi-GPU). Meaningful only when the model is actually split.
    #[serde(default)]
    pub split_mode: Option<SplitMode>,
    /// `--main-gpu N`. Which GPU holds the small intermediate state.
    #[serde(default)]
    pub main_gpu: Option<u32>,
    /// `--tensor-split F0,F1,…` override. If `None`, the orchestrator will
    /// compute the minimum viable subset at load time based on free VRAM.
    /// If `Some`, the literal string is passed through unchanged.
    #[serde(default)]
    pub tensor_split: Option<String>,

    #[serde(default)]
    pub state: ModelState,
    #[serde(default)]
    pub pid: Option<i32>,
    #[serde(default)]
    pub estimated_vram: u64,
    #[serde(default)]
    pub last_used: Option<f64>,
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample() -> ModelConfig {
        ModelConfig {
            id: "qwen3-30b".into(),
            name: "Qwen3 30B".into(),
            weights_format: WeightsFormat::Gguf,
            binary_preset: Some("llama-vulkan".into()),
            binary: PathBuf::from("/home/u/llama.cpp/build-vulkan/bin/llama-server"),
            model_path: PathBuf::from("/models/qwen3-30b.gguf"),
            port: 9001,
            extra_args: vec!["--override-kv".into(), "something=1".into()],
            context: 32768,
            temperature: 0.6,
            top_p: 0.95,
            top_k: 40,
            min_p: 0.0,
            flash_attn: true,
            n_gpu_layers: Some(99),
            mlock: true,
            no_mmap: false,
            parallel_slots: Some(4),
            cache_type_k: Some(CacheType::Q8_0),
            cache_type_v: Some(CacheType::Q8_0),
            split_mode: Some(SplitMode::Layer),
            main_gpu: Some(0),
            tensor_split: Some("0.5,0.5,0".into()),
            state: ModelState::Idle,
            pid: None,
            estimated_vram: 0,
            last_used: None,
        }
    }

    #[test]
    fn serde_roundtrip_preserves_all_fields() {
        let original = sample();
        let json = serde_json::to_string(&original).unwrap();
        let parsed: ModelConfig = serde_json::from_str(&json).unwrap();
        assert_eq!(original, parsed);
    }

    #[test]
    fn weights_format_serializes_lowercase() {
        let gguf = serde_json::to_string(&WeightsFormat::Gguf).unwrap();
        let safe = serde_json::to_string(&WeightsFormat::Safetensors).unwrap();
        assert_eq!(gguf, "\"gguf\"");
        assert_eq!(safe, "\"safetensors\"");
    }

    #[test]
    fn runtime_fields_default_when_absent() {
        let json = r#"{
            "id": "m", "name": "M",
            "weights_format": "gguf",
            "binary": "/bin/llama", "model_path": "/m.gguf",
            "port": 9001, "context": 4096
        }"#;
        let parsed: ModelConfig = serde_json::from_str(json).unwrap();
        assert_eq!(parsed.state, ModelState::Idle);
        assert_eq!(parsed.pid, None);
        assert_eq!(parsed.estimated_vram, 0);
        assert_eq!(parsed.last_used, None);
        assert_eq!(parsed.extra_args, Vec::<String>::new());
        assert_eq!(parsed.temperature, 0.6);
        assert_eq!(parsed.top_p, 0.95);
        assert_eq!(parsed.top_k, 40);
        assert_eq!(parsed.min_p, 0.0);
        // New llama.cpp flags all default to off/unset.
        assert!(!parsed.flash_attn);
        assert_eq!(parsed.n_gpu_layers, None);
        assert!(!parsed.mlock);
        assert!(!parsed.no_mmap);
        assert_eq!(parsed.parallel_slots, None);
        assert_eq!(parsed.cache_type_k, None);
        assert_eq!(parsed.cache_type_v, None);
        assert_eq!(parsed.split_mode, None);
        assert_eq!(parsed.main_gpu, None);
        assert_eq!(parsed.tensor_split, None);
    }

    #[test]
    fn split_mode_serializes_lowercase() {
        assert_eq!(serde_json::to_string(&SplitMode::None).unwrap(), "\"none\"");
        assert_eq!(serde_json::to_string(&SplitMode::Layer).unwrap(), "\"layer\"");
        assert_eq!(serde_json::to_string(&SplitMode::Row).unwrap(), "\"row\"");
    }

    #[test]
    fn cache_type_serializes_lowercase() {
        assert_eq!(serde_json::to_string(&CacheType::F16).unwrap(), "\"f16\"");
        assert_eq!(serde_json::to_string(&CacheType::Q8_0).unwrap(), "\"q8_0\"");
        assert_eq!(serde_json::to_string(&CacheType::Q4_0).unwrap(), "\"q4_0\"");
    }

    #[test]
    fn error_state_roundtrips_with_message() {
        let mut m = sample();
        m.state = ModelState::Error("process 1234 died".into());
        let json = serde_json::to_string(&m).unwrap();
        let parsed: ModelConfig = serde_json::from_str(&json).unwrap();
        assert_eq!(parsed.state, ModelState::Error("process 1234 died".into()));
    }
}
