use serde::{Deserialize, Serialize};
use std::path::PathBuf;

// Default values for sampling params. These are the llama.cpp defaults; we
// keep them in one place so `ModelConfig`, `ModelRequest`, and any future
// DTO all share a single source of truth (visible via `pub` for serde
// `#[serde(default = "...")]` callers in other modules).
pub fn default_temperature() -> f32 {
    0.6
}
pub fn default_top_p() -> f32 {
    0.95
}
pub fn default_top_k() -> i32 {
    40
}
pub fn default_min_p() -> f32 {
    0.0
}
pub fn default_presence_penalty() -> f32 {
    0.0
}
pub fn default_repeat_penalty() -> f32 {
    1.0
}

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
}

/// Self-contained per-model configuration. No external framework entity.
///
/// `Default` produces a blank GGUF model with llama.cpp's sampling defaults
/// and no binary / model / port set, so callers can use partial struct
/// literals without re-listing every field.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct ModelConfig {
    pub id: String,
    pub name: String,
    #[serde(default)]
    pub profile: Option<String>,

    pub weights_format: WeightsFormat,
    /// If set, looked up in the presets table at spawn time to get the actual
    /// binary path. If `None`, `binary` below is used verbatim. Lets you
    /// change a binary once (e.g. rebuild llama.cpp) and have every model
    /// pick it up.
    #[serde(default)]
    pub binary_preset: Option<String>,
    pub binary: PathBuf,
    pub model_path: PathBuf,
    /// `--mmproj FILE`. Required by llama.cpp for GGUF vision inputs.
    #[serde(default)]
    pub mmproj_path: Option<PathBuf>,
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

    // llama.cpp runtime flags. Placement is intentionally router-owned:
    // llama-fit-params chooses layer/split/expert placement at load time.
    #[serde(default)]
    pub flash_attn: bool,
    #[serde(skip)]
    pub n_gpu_layers: Option<u32>,
    /// Legacy/internal only. Manual MoE expert placement is not accepted from
    /// persisted config; llama-fit-params decides placement for normal loads.
    #[serde(skip)]
    pub n_cpu_moe: Option<u32>,
    /// Fitted at load time from llama-fit-params; not accepted from config.
    #[serde(skip)]
    pub override_tensor: Option<String>,
    /// Legacy/internal only; normal loads probe first and launch fitted args.
    #[serde(skip)]
    pub fit_target: Option<String>,
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

    /// Manual placement knobs are ignored at the config/API boundary.
    #[serde(skip)]
    pub split_mode: Option<SplitMode>,
    #[serde(skip)]
    pub main_gpu: Option<u32>,
    /// Fitted at load time from llama-fit-params; not accepted from config.
    #[serde(skip)]
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
    // builder pulls the draft's `model_path` and `cache_type_{k,v}` to emit
    // `-md / -ctkd / -ctvd`. Placement stays router-owned. Spec-decode policy
    // (how hard this model drives the draft) lives here: `draft_max`,
    // `draft_min`, `draft_p_min`, `ctx_checkpoints`,
    // `checkpoint_every_n_tokens`. MTP speculative decoding uses the target
    // model's own draft heads and is controlled by `mtp_tokens`.
    /// Fitted at load time from the router's chosen GPU subset. Manual target
    /// device selection is ignored at the config/API boundary.
    #[serde(skip)]
    pub device: Option<String>,

    /// ID of another model to use as a speculative-decoding draft.
    /// Presence enables spec-decode.
    #[serde(default)]
    pub draft_model_id: Option<String>,

    /// `--spec-type draft-mtp` + `--spec-draft-n-max N`. Embedded MTP draft
    /// tokens for models with MTP heads. `None` / `0` disables MTP.
    #[serde(default)]
    pub mtp_tokens: Option<u32>,

    /// `--spec-draft-n-max N`. Max external draft tokens per step.
    #[serde(default)]
    pub draft_max: Option<u32>,
    /// `--spec-draft-n-min N`. Min draft tokens before submitting to target.
    #[serde(default)]
    pub draft_min: Option<u32>,
    /// `--spec-draft-p-min P`. Probability floor for greedy draft sampling.
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
            profile: None,
            weights_format: WeightsFormat::default(),
            binary_preset: None,
            binary: PathBuf::new(),
            model_path: PathBuf::new(),
            mmproj_path: None,
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
            n_cpu_moe: None,
            override_tensor: None,
            fit_target: None,
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
            mtp_tokens: None,
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
    /// The strategy is token-pair: scan flag-value pairs and copy recognized
    /// values into structured fields without editing `extra_args`. Raw extra
    /// args are user-owned and still get appended at launch.
    pub fn migrate_extra_args(&mut self) -> bool {
        let mut changed = false;
        let args = self.extra_args.clone();
        let has_mtp_spec_type = args
            .windows(2)
            .any(|w| w[0] == "--spec-type" && w[1] == "draft-mtp")
            || args.iter().any(|a| a == "--spec-type=draft-mtp");
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
                            changed = true;
                        }
                        consumed = 2;
                    }
                }
                ("--cache-ram" | "-cram", Some(v)) => {
                    if let Ok(n) = v.parse::<i32>() {
                        if self.cache_ram_mib.is_none() {
                            self.cache_ram_mib = Some(n);
                            changed = true;
                        }
                        consumed = 2;
                    }
                }
                ("--reasoning-format", Some(v)) => {
                    if let Some(rf) = ReasoningFormat::from_cli(v) {
                        if self.reasoning_format.is_none() {
                            self.reasoning_format = Some(rf);
                            changed = true;
                        }
                        consumed = 2;
                    }
                }
                ("--reasoning-budget", Some(v)) => {
                    if let Ok(n) = v.parse::<i32>() {
                        if self.reasoning_budget.is_none() {
                            self.reasoning_budget = Some(n);
                            changed = true;
                        }
                        consumed = 2;
                    }
                }
                ("--presence-penalty", Some(v)) => {
                    if let Ok(f) = v.parse::<f32>() {
                        if (self.presence_penalty - f).abs() > f32::EPSILON {
                            self.presence_penalty = f;
                            changed = true;
                        }
                        consumed = 2;
                    }
                }
                ("--repeat-penalty", Some(v)) => {
                    if let Ok(f) = v.parse::<f32>() {
                        if (self.repeat_penalty - f).abs() > f32::EPSILON {
                            self.repeat_penalty = f;
                            changed = true;
                        }
                        consumed = 2;
                    }
                }
                ("--chat-template-kwargs", Some(v)) => {
                    if self.chat_template_kwargs.is_none() {
                        self.chat_template_kwargs = Some(v.clone());
                        changed = true;
                    }
                    consumed = 2;
                }
                ("--mmproj" | "-mm", Some(v)) => {
                    if self.mmproj_path.is_none() {
                        self.mmproj_path = Some(PathBuf::from(v));
                        changed = true;
                    }
                    consumed = 2;
                }
                // Speculative decoding policy flags. The draft *path*
                // flags (`-md`, `-ngld`, `-devd`, `-ctkd`, `-ctvd`)
                // are intentionally left alone — they reference a draft
                // model that now has to be a first-class entry in
                // models.json, and we can't synthesise that from a path.
                ("--spec-type", Some(v)) => {
                    if v == "draft-mtp" && self.draft_model_id.is_none() {
                        consumed = 2;
                    }
                }
                _ if flag == "--spec-type=draft-mtp" && self.draft_model_id.is_none() => {
                    consumed = 1;
                }
                ("--spec-draft-n-max", Some(v)) => {
                    if let Ok(n) = v.parse::<u32>() {
                        if has_mtp_spec_type && self.draft_model_id.is_none() {
                            if self.mtp_tokens.is_none() {
                                self.mtp_tokens = Some(n);
                                changed = true;
                            }
                        } else if self.draft_max.is_none() {
                            self.draft_max = Some(n);
                            changed = true;
                        }
                        consumed = 2;
                    }
                }
                ("--draft-max" | "--draft" | "--draft-n", Some(v)) => {
                    if let Ok(n) = v.parse::<u32>() {
                        if self.draft_max.is_none() {
                            self.draft_max = Some(n);
                            changed = true;
                        }
                        consumed = 2;
                    }
                }
                ("--spec-draft-n-min" | "--draft-min" | "--draft-n-min", Some(v)) => {
                    if let Ok(n) = v.parse::<u32>() {
                        if self.draft_min.is_none() {
                            self.draft_min = Some(n);
                            changed = true;
                        }
                        consumed = 2;
                    }
                }
                ("--spec-draft-p-min" | "--draft-p-min", Some(v)) => {
                    if let Ok(f) = v.parse::<f32>() {
                        if self.draft_p_min.is_none() {
                            self.draft_p_min = Some(f);
                            changed = true;
                        }
                        consumed = 2;
                    }
                }
                ("--ctx-checkpoints" | "-ctxcp" | "--swa-checkpoints", Some(v)) => {
                    if let Ok(n) = v.parse::<u32>() {
                        if self.ctx_checkpoints.is_none() {
                            self.ctx_checkpoints = Some(n);
                            changed = true;
                        }
                        consumed = 2;
                    }
                }
                ("--checkpoint-every-n-tokens" | "-cpent", Some(v)) => {
                    if let Ok(n) = v.parse::<i32>() {
                        if self.checkpoint_every_n_tokens.is_none() {
                            self.checkpoint_every_n_tokens = Some(n);
                            changed = true;
                        }
                        consumed = 2;
                    }
                }
                _ => {}
            }

            if consumed == 0 {
                i += 1;
            } else {
                i += consumed;
            }
        }
        if has_mtp_spec_type && self.draft_model_id.is_none() && self.mtp_tokens.is_none() {
            self.mtp_tokens = Some(3);
            changed = true;
        }
        changed
    }
}

#[cfg(test)]
#[path = "model_tests.rs"]
mod tests;
