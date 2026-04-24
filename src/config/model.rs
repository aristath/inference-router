use serde::{Deserialize, Serialize};
use std::path::PathBuf;

// Default values for sampling params. These are the llama.cpp defaults; we
// keep them in one place so `ModelConfig`, `ModelRequest`, and any future
// DTO all share a single source of truth (visible via `pub` for serde
// `#[serde(default = "...")]` callers in other modules).
pub fn default_temperature() -> f32 { 0.6 }
pub fn default_top_p() -> f32 { 0.95 }
pub fn default_top_k() -> i32 { 40 }
pub fn default_min_p() -> f32 { 0.0 }
pub fn default_presence_penalty() -> f32 { 0.0 }
pub fn default_repeat_penalty() -> f32 { 1.0 }

/// Weights file format. Drives the argv style used when spawning the backend.
///
/// - `Gguf` → llama.cpp-style (`-m <file> -c <ctx> --port ...`)
/// - `Safetensors` → vLLM-style (`--model <dir> --port ... --max-model-len <ctx>`)
#[derive(Debug, Clone, Copy, Default, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "lowercase")]
pub enum WeightsFormat {
    #[default]
    Gguf,
    Safetensors,
}

/// Runtime state of a model.
#[derive(Debug, Clone, Default, Serialize, Deserialize, PartialEq)]
pub enum ModelState {
    #[default]
    Idle,
    Loading,
    Running,
    Error(String),
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

/// `--reasoning-format` enum exactly matching `common_reasoning_format` in
/// llama.cpp (`common/common.h`). The comment there says not to extend the
/// enum "unless you absolutely have to," so we mirror the four values
/// verbatim. Controls how thought tags are returned in the response
/// (orthogonal to `--reasoning on|off|auto`).
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "kebab-case")]
pub enum ReasoningFormat {
    None,
    Auto,
    Deepseek,
    DeepseekLegacy,
}

impl ReasoningFormat {
    /// The literal string llama-server expects on the command line.
    pub fn as_arg(&self) -> &'static str {
        match self {
            Self::None => "none",
            Self::Auto => "auto",
            Self::Deepseek => "deepseek",
            Self::DeepseekLegacy => "deepseek-legacy",
        }
    }

    pub fn from_cli(s: &str) -> Option<Self> {
        match s {
            "none" => Some(Self::None),
            "auto" => Some(Self::Auto),
            "deepseek" => Some(Self::Deepseek),
            "deepseek-legacy" => Some(Self::DeepseekLegacy),
            _ => None,
        }
    }
}

/// KV-cache quantization. Applies to llama.cpp (`--cache-type-{k,v}`).
#[derive(Debug, Clone, Copy, Default, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "lowercase")]
pub enum CacheType {
    #[default]
    F16,
    Q8_0,
    Q4_0,
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

    /// Bytes held per KV-cache element (per head × per token × per K or
    /// V separately), expressed as a rational `(numerator, denominator)`
    /// pair so q4 — which really is half a byte — stays in integer
    /// arithmetic downstream.
    ///
    /// - `f16` → `(2, 1)` = 2 bytes per element
    /// - `q8_0` → `(1, 1)` = 1 byte per element
    /// - `q4_0` → `(1, 2)` = half a byte per element
    pub fn bytes_per_element(&self) -> (u64, u64) {
        match self {
            Self::F16 => (2, 1),
            Self::Q8_0 => (1, 1),
            Self::Q4_0 => (1, 2),
        }
    }
}

/// Self-contained per-model configuration. No external framework entity.
///
/// `Default` produces a blank GGUF model with llama.cpp's sampling defaults
/// and no binary / model / port set. It exists so tests can use
/// `ModelConfig { id: "x".into(), ..Default::default() }` instead of
/// re-listing 30 fields.
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
    #[serde(default = "default_presence_penalty")]
    pub presence_penalty: f32,
    #[serde(default = "default_repeat_penalty")]
    pub repeat_penalty: f32,

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

    /// `--threads N`. Number of CPU threads llama-server uses for generation.
    /// llama.cpp's default is `-1` (auto).
    #[serde(default)]
    pub threads: Option<i32>,
    /// `--cache-ram N` (MiB). Maximum cache size. llama.cpp defaults to 8192;
    /// `-1` = no limit, `0` = disabled.
    #[serde(default)]
    pub cache_ram_mib: Option<i32>,
    /// `--reasoning-format`. Controls how thought tags are returned in the
    /// OpenAI response body.
    #[serde(default)]
    pub reasoning_format: Option<ReasoningFormat>,
    /// `--reasoning-budget N`. `-1` = unrestricted, `0` = immediate end,
    /// `N>0` = token budget.
    #[serde(default)]
    pub reasoning_budget: Option<i32>,
    /// `--chat-template-kwargs STRING`. Raw JSON object passed verbatim to
    /// llama.cpp's Jinja chat template (template-family-specific keys like
    /// `enable_thinking`, `reasoning_effort`, `preserve_thinking`).
    #[serde(default)]
    pub chat_template_kwargs: Option<String>,

    // ===== Speculative decoding =====
    // A model can reference another model by id as its draft. The argv
    // builder pulls the draft's `model_path`, `context`, `n_gpu_layers`,
    // `cache_type_{k,v}`, and `device` to emit `-md / -cd / -ngld /
    // -ctkd / -ctvd / -devd`. Spec-decode policy (how hard this model
    // drives the draft) lives here: `draft_max`, `draft_min`,
    // `draft_p_min`, `ctx_checkpoints`, `checkpoint_every_n_tokens`.

    /// `-devd <dev1,dev2,…>` device list when this model is embedded as
    /// a draft inside another model's spawn. Leave blank to let llama.cpp pick.
    #[serde(default)]
    pub device: Option<String>,

    /// ID of another model to use as a speculative-decoding draft.
    /// Presence enables spec-decode.
    #[serde(default)]
    pub draft_model_id: Option<String>,

    /// `--draft-max N`. Max draft tokens per step. llama.cpp default 16.
    #[serde(default)]
    pub draft_max: Option<u32>,
    /// `--draft-min N`. Min draft tokens before submitting to target.
    #[serde(default)]
    pub draft_min: Option<u32>,
    /// `--draft-p-min P`. Probability floor for greedy draft sampling.
    #[serde(default)]
    pub draft_p_min: Option<f32>,
    /// `--ctx-checkpoints N`. Context-state snapshot slots. Required > 0
    /// for hybrid-recurrent targets (Qwen3.5 dense) so partial-draft
    /// rollback works via snapshot/restore instead of seq_rm.
    #[serde(default)]
    pub ctx_checkpoints: Option<u32>,
    /// `--checkpoint-every-n-tokens N`. Prefill-time checkpoint cadence.
    #[serde(default)]
    pub checkpoint_every_n_tokens: Option<i32>,

    #[serde(default)]
    pub state: ModelState,
    #[serde(default)]
    pub pid: Option<i32>,
    #[serde(default)]
    pub estimated_vram: u64,
    #[serde(default)]
    pub last_used: Option<f64>,
    /// Rich GGUF metadata snapshot stored at model-creation time.
    /// Populated by the dashboard's 2-step "Add model" flow; absent for
    /// models added before this field existed or via the API without it.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub gguf_meta: Option<crate::vram::estimator::GgufMeta>,
}

impl Default for ModelConfig {
    fn default() -> Self {
        Self {
            id: String::new(),
            name: String::new(),
            weights_format: WeightsFormat::default(),
            binary_preset: None,
            binary: PathBuf::new(),
            model_path: PathBuf::new(),
            extra_args: Vec::new(),
            context: 4096,
            temperature: default_temperature(),
            top_p: default_top_p(),
            top_k: default_top_k(),
            min_p: default_min_p(),
            presence_penalty: default_presence_penalty(),
            repeat_penalty: default_repeat_penalty(),
            flash_attn: false,
            n_gpu_layers: None,
            mlock: false,
            no_mmap: false,
            parallel_slots: None,
            cache_type_k: None,
            cache_type_v: None,
            split_mode: None,
            main_gpu: None,
            tensor_split: None,
            threads: None,
            cache_ram_mib: None,
            reasoning_format: None,
            reasoning_budget: None,
            chat_template_kwargs: None,
            device: None,
            draft_model_id: None,
            draft_max: None,
            draft_min: None,
            draft_p_min: None,
            ctx_checkpoints: None,
            checkpoint_every_n_tokens: None,
            state: ModelState::default(),
            pid: None,
            estimated_vram: 0,
            last_used: None,
            gguf_meta: None,
        }
    }
}

/// Validation error returned when adding/updating a model config.
#[derive(Debug, thiserror::Error)]
pub enum ConfigError {
    #[error("model references draft '{id}', but no model with that id exists")]
    DraftNotFound { id: String },
    #[error("model cannot reference itself as a draft")]
    DraftSelfReference,
}

impl ModelConfig {
    /// One-shot migration of raw `extra_args` into the structured fields
    /// added after the fact. Called once per model at orchestrator startup;
    /// returns `true` if anything was moved so the caller can mark the
    /// store dirty.
    ///
    /// The strategy is token-pair: scan flag-value pairs and promote the
    /// seven flags we now model as options. Anything unrecognized stays
    /// in `extra_args`. Structured fields win if already set (we strip the
    /// duplicate from extra_args but don't overwrite).
    pub fn migrate_extra_args(&mut self) -> bool {
        let mut changed = false;
        let mut out: Vec<String> = Vec::with_capacity(self.extra_args.len());
        let args = std::mem::take(&mut self.extra_args);
        let mut i = 0;
        while i < args.len() {
            let flag = args[i].as_str();
            let next = args.get(i + 1);
            let mut consumed = 0;

            match (flag, next) {
                ("--threads" | "-t", Some(v)) => {
                    if let Ok(n) = v.parse::<i32>() {
                        if self.threads.is_none() {
                            self.threads = Some(n);
                        }
                        consumed = 2;
                    }
                }
                ("--cache-ram" | "-cram", Some(v)) => {
                    if let Ok(n) = v.parse::<i32>() {
                        if self.cache_ram_mib.is_none() {
                            self.cache_ram_mib = Some(n);
                        }
                        consumed = 2;
                    }
                }
                ("--reasoning-format", Some(v)) => {
                    if let Some(rf) = ReasoningFormat::from_cli(v) {
                        if self.reasoning_format.is_none() {
                            self.reasoning_format = Some(rf);
                        }
                        consumed = 2;
                    }
                }
                ("--reasoning-budget", Some(v)) => {
                    if let Ok(n) = v.parse::<i32>() {
                        if self.reasoning_budget.is_none() {
                            self.reasoning_budget = Some(n);
                        }
                        consumed = 2;
                    }
                }
                ("--presence-penalty", Some(v)) => {
                    if let Ok(f) = v.parse::<f32>() {
                        self.presence_penalty = f;
                        consumed = 2;
                    }
                }
                ("--repeat-penalty", Some(v)) => {
                    if let Ok(f) = v.parse::<f32>() {
                        self.repeat_penalty = f;
                        consumed = 2;
                    }
                }
                ("--chat-template-kwargs", Some(v)) => {
                    if self.chat_template_kwargs.is_none() {
                        self.chat_template_kwargs = Some(v.clone());
                    }
                    consumed = 2;
                }
                // Speculative decoding policy flags. The draft *path*
                // flags (`-md`, `-ngld`, `-devd`, `-cd`, `-ctkd`, `-ctvd`)
                // are intentionally left alone — they reference a draft
                // model that now has to be a first-class entry in
                // models.json, and we can't synthesise that from a path.
                ("--draft-max" | "--draft" | "--draft-n", Some(v)) => {
                    if let Ok(n) = v.parse::<u32>() {
                        if self.draft_max.is_none() {
                            self.draft_max = Some(n);
                        }
                        consumed = 2;
                    }
                }
                ("--draft-min" | "--draft-n-min", Some(v)) => {
                    if let Ok(n) = v.parse::<u32>() {
                        if self.draft_min.is_none() {
                            self.draft_min = Some(n);
                        }
                        consumed = 2;
                    }
                }
                ("--draft-p-min", Some(v)) => {
                    if let Ok(f) = v.parse::<f32>() {
                        if self.draft_p_min.is_none() {
                            self.draft_p_min = Some(f);
                        }
                        consumed = 2;
                    }
                }
                ("--ctx-checkpoints" | "-ctxcp" | "--swa-checkpoints", Some(v)) => {
                    if let Ok(n) = v.parse::<u32>() {
                        if self.ctx_checkpoints.is_none() {
                            self.ctx_checkpoints = Some(n);
                        }
                        consumed = 2;
                    }
                }
                ("--checkpoint-every-n-tokens" | "-cpent", Some(v)) => {
                    if let Ok(n) = v.parse::<i32>() {
                        if self.checkpoint_every_n_tokens.is_none() {
                            self.checkpoint_every_n_tokens = Some(n);
                        }
                        consumed = 2;
                    }
                }
                _ => {}
            }

            if consumed == 0 {
                out.push(args[i].clone());
                i += 1;
            } else {
                changed = true;
                i += consumed;
            }
        }
        self.extra_args = out;
        changed
    }

}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample() -> ModelConfig {
        ModelConfig {
            id: "qwen3-30b".into(),
            name: "Qwen3 30B".into(),
            binary_preset: Some("llama-vulkan".into()),
            binary: PathBuf::from("/home/u/llama.cpp/build-vulkan/bin/llama-server"),
            model_path: PathBuf::from("/models/qwen3-30b.gguf"),
            extra_args: vec!["--override-kv".into(), "something=1".into()],
            context: 32768,
            flash_attn: true,
            n_gpu_layers: Some(99),
            mlock: true,
            parallel_slots: Some(4),
            cache_type_k: Some(CacheType::Q8_0),
            cache_type_v: Some(CacheType::Q8_0),
            split_mode: Some(SplitMode::Layer),
            main_gpu: Some(0),
            tensor_split: Some("0.5,0.5,0".into()),
            threads: Some(16),
            cache_ram_mib: Some(0),
            reasoning_format: Some(ReasoningFormat::Auto),
            reasoning_budget: Some(-1),
            chat_template_kwargs: Some(r#"{"enable_thinking":true}"#.into()),
            ..ModelConfig::default()
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
        // Post-migration structured fields default to unset / neutral.
        assert_eq!(parsed.threads, None);
        assert_eq!(parsed.cache_ram_mib, None);
        assert_eq!(parsed.reasoning_format, None);
        assert_eq!(parsed.reasoning_budget, None);
        assert_eq!(parsed.chat_template_kwargs, None);
        assert_eq!(parsed.presence_penalty, 0.0);
        assert_eq!(parsed.repeat_penalty, 1.0);
        // Spec-decode fields default to off/unset.
        assert_eq!(parsed.device, None);
        assert_eq!(parsed.draft_model_id, None);
        assert_eq!(parsed.draft_max, None);
        assert_eq!(parsed.draft_min, None);
        assert_eq!(parsed.draft_p_min, None);
        assert_eq!(parsed.ctx_checkpoints, None);
        assert_eq!(parsed.checkpoint_every_n_tokens, None);
    }

    #[test]
    fn reasoning_format_serializes_kebab_case() {
        assert_eq!(serde_json::to_string(&ReasoningFormat::None).unwrap(), "\"none\"");
        assert_eq!(serde_json::to_string(&ReasoningFormat::Auto).unwrap(), "\"auto\"");
        assert_eq!(serde_json::to_string(&ReasoningFormat::Deepseek).unwrap(), "\"deepseek\"");
        assert_eq!(
            serde_json::to_string(&ReasoningFormat::DeepseekLegacy).unwrap(),
            "\"deepseek-legacy\"",
        );
    }

    #[test]
    fn reasoning_format_from_cli_covers_all_values() {
        assert_eq!(ReasoningFormat::from_cli("none"), Some(ReasoningFormat::None));
        assert_eq!(ReasoningFormat::from_cli("auto"), Some(ReasoningFormat::Auto));
        assert_eq!(ReasoningFormat::from_cli("deepseek"), Some(ReasoningFormat::Deepseek));
        assert_eq!(
            ReasoningFormat::from_cli("deepseek-legacy"),
            Some(ReasoningFormat::DeepseekLegacy),
        );
        assert_eq!(ReasoningFormat::from_cli("bogus"), None);
    }

    // ----- Migration -----

    fn bare() -> ModelConfig {
        ModelConfig {
            id: "m".into(),
            name: "M".into(),
            binary: PathBuf::from("/bin/llama"),
            model_path: PathBuf::from("/m.gguf"),
            ..ModelConfig::default()
        }
    }

    #[test]
    fn migrate_extracts_all_seven_flags() {
        let mut m = bare();
        m.extra_args = vec![
            "--threads".into(), "16".into(),
            "--reasoning-format".into(), "auto".into(),
            "--cache-ram".into(), "0".into(),
            "--presence-penalty".into(), "1.5".into(),
            "--repeat-penalty".into(), "1.0".into(),
            "--reasoning-budget".into(), "0".into(),
            "--chat-template-kwargs".into(), r#"{"enable_thinking":false}"#.into(),
        ];
        assert!(m.migrate_extra_args());
        assert_eq!(m.threads, Some(16));
        assert_eq!(m.reasoning_format, Some(ReasoningFormat::Auto));
        assert_eq!(m.cache_ram_mib, Some(0));
        assert_eq!(m.presence_penalty, 1.5);
        assert_eq!(m.repeat_penalty, 1.0);
        assert_eq!(m.reasoning_budget, Some(0));
        assert_eq!(m.chat_template_kwargs.as_deref(), Some(r#"{"enable_thinking":false}"#));
        assert!(m.extra_args.is_empty());
    }

    #[test]
    fn migrate_preserves_unknown_flags() {
        let mut m = bare();
        m.extra_args = vec![
            "--override-kv".into(), "foo=bar".into(),
            "--threads".into(), "16".into(),
            "--custom-flag".into(),
        ];
        assert!(m.migrate_extra_args());
        assert_eq!(m.threads, Some(16));
        assert_eq!(
            m.extra_args,
            vec!["--override-kv", "foo=bar", "--custom-flag"],
        );
    }

    #[test]
    fn migrate_is_idempotent_after_first_pass() {
        let mut m = bare();
        m.extra_args = vec!["--threads".into(), "16".into()];
        assert!(m.migrate_extra_args());
        // Second pass: nothing to migrate, nothing changes.
        assert!(!m.migrate_extra_args());
        assert_eq!(m.threads, Some(16));
        assert!(m.extra_args.is_empty());
    }

    #[test]
    fn migrate_keeps_existing_structured_value_on_conflict() {
        let mut m = bare();
        m.threads = Some(32);
        m.extra_args = vec!["--threads".into(), "16".into()];
        assert!(m.migrate_extra_args()); // changed = dropped from extra_args
        assert_eq!(m.threads, Some(32)); // structured wins
        assert!(m.extra_args.is_empty());
    }

    #[test]
    fn migrate_handles_short_aliases() {
        let mut m = bare();
        m.extra_args = vec!["-t".into(), "8".into(), "-cram".into(), "4096".into()];
        assert!(m.migrate_extra_args());
        assert_eq!(m.threads, Some(8));
        assert_eq!(m.cache_ram_mib, Some(4096));
        assert!(m.extra_args.is_empty());
    }

    #[test]
    fn migrate_ignores_flag_with_unparseable_value() {
        let mut m = bare();
        m.extra_args = vec!["--threads".into(), "not-a-number".into()];
        assert!(!m.migrate_extra_args());
        assert_eq!(m.threads, None);
        assert_eq!(m.extra_args, vec!["--threads", "not-a-number"]);
    }

    #[test]
    fn migrate_rejects_unknown_reasoning_format_value() {
        let mut m = bare();
        m.extra_args = vec!["--reasoning-format".into(), "made-up".into()];
        assert!(!m.migrate_extra_args());
        assert_eq!(m.reasoning_format, None);
        assert_eq!(m.extra_args, vec!["--reasoning-format", "made-up"]);
    }

    #[test]
    fn migrate_real_world_qwen3_args() {
        // Taken verbatim from the user's models.json.
        let mut m = bare();
        m.extra_args = vec![
            "--threads".into(), "16".into(),
            "--reasoning-format".into(), "auto".into(),
            "--cache-ram".into(), "0".into(),
            "--presence-penalty".into(), "1.5".into(),
            "--repeat-penalty".into(), "1.0".into(),
            "--chat-template-kwargs".into(), r#"{"enable_thinking":false}"#.into(),
        ];
        assert!(m.migrate_extra_args());
        assert!(m.extra_args.is_empty());
        assert_eq!(m.threads, Some(16));
        assert_eq!(m.reasoning_format, Some(ReasoningFormat::Auto));
        assert_eq!(m.cache_ram_mib, Some(0));
        assert_eq!(m.presence_penalty, 1.5);
        assert_eq!(m.repeat_penalty, 1.0);
        assert_eq!(m.chat_template_kwargs.as_deref(), Some(r#"{"enable_thinking":false}"#));
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

    // ----- Speculative decoding -----

    #[test]
    fn migrate_extracts_spec_decode_policy_flags() {
        let mut m = bare();
        m.extra_args = vec![
            "--draft-max".into(), "16".into(),
            "--draft-min".into(), "1".into(),
            "--draft-p-min".into(), "0.75".into(),
            "--ctx-checkpoints".into(), "4".into(),
            "--checkpoint-every-n-tokens".into(), "-1".into(),
        ];
        assert!(m.migrate_extra_args());
        assert_eq!(m.draft_max, Some(16));
        assert_eq!(m.draft_min, Some(1));
        assert_eq!(m.draft_p_min, Some(0.75));
        assert_eq!(m.ctx_checkpoints, Some(4));
        assert_eq!(m.checkpoint_every_n_tokens, Some(-1));
        assert!(m.extra_args.is_empty());
    }

    #[test]
    fn migrate_leaves_draft_path_flags_alone() {
        // `-md`, `-ngld`, `-devd`, `-cd`, `-ctkd`, `-ctvd` can't be
        // auto-migrated because they reference a draft GGUF by path —
        // but the new model requires drafts to be ModelConfig entries
        // (addressable by id). Preserve them in extra_args so the user
        // can see what needs to be reconstructed as a draft entry.
        let mut m = bare();
        m.extra_args = vec![
            "-md".into(), "/m/draft.gguf".into(),
            "-ngld".into(), "99".into(),
            "-devd".into(), "Vulkan1".into(),
        ];
        assert!(!m.migrate_extra_args());
        assert_eq!(
            m.extra_args,
            vec!["-md", "/m/draft.gguf", "-ngld", "99", "-devd", "Vulkan1"],
        );
    }

    #[test]
    fn migrate_accepts_draft_max_aliases() {
        // llama.cpp aliases: --draft / --draft-n / --draft-max
        let mut m = bare();
        m.extra_args = vec!["--draft".into(), "8".into()];
        assert!(m.migrate_extra_args());
        assert_eq!(m.draft_max, Some(8));

        let mut m = bare();
        m.extra_args = vec!["--draft-n".into(), "12".into()];
        assert!(m.migrate_extra_args());
        assert_eq!(m.draft_max, Some(12));
    }
}
