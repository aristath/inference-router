use serde::{Deserialize, Serialize};

/// Global application settings.
///
/// Configured via environment variables or the dashboard Settings modal.
/// Changes are persisted to `~/.config/inference-router/settings.json`.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct AppSettings {
    /// Folder scanned by the dashboard model-discovery action.
    pub models_folder: String,
    /// Loop guard configuration for both streaming and tool loops
    pub loop_guards: LoopGuardSettings,
    /// Which set of model names is advertised on `/v1/models`.
    pub model_exposure: ModelExposure,
    /// Percent of a GPU's VRAM the router lets llama.cpp fill (`--fit-target`
    /// margin = the remainder). Keeps a safety margin so a packed model can't
    /// OOM the GPU. 1..=100.
    pub gpu_vram_cap_pct: u8,
    /// Same, but for a GPU driving a monitor — lower, to leave headroom for the
    /// desktop/compositor. 1..=100.
    pub display_gpu_vram_cap_pct: u8,
    /// Watchdog / self-heal configuration for wedged inference instances.
    pub watchdog: WatchdogSettings,
}

impl Default for AppSettings {
    fn default() -> Self {
        Self {
            models_folder: default_models_folder(),
            loop_guards: LoopGuardSettings::default(),
            model_exposure: ModelExposure::default(),
            gpu_vram_cap_pct: 98,
            display_gpu_vram_cap_pct: 80,
            watchdog: WatchdogSettings::default(),
        }
    }
}

impl AppSettings {
    pub fn from_env() -> Self {
        Self {
            models_folder: std::env::var("INFERENCE_ROUTER_MODELS_FOLDER")
                .ok()
                .filter(|v| !v.trim().is_empty())
                .unwrap_or_else(default_models_folder),
            loop_guards: LoopGuardSettings::from_env(),
            model_exposure: ModelExposure::from_env(),
            gpu_vram_cap_pct: env_pct("INFERENCE_ROUTER_GPU_VRAM_CAP_PCT").unwrap_or(98),
            display_gpu_vram_cap_pct: env_pct("INFERENCE_ROUTER_DISPLAY_VRAM_CAP_PCT")
                .unwrap_or(80),
            watchdog: WatchdogSettings::from_env(),
        }
    }

    pub fn sanitized(mut self) -> Self {
        if self.models_folder.trim().is_empty() {
            self.models_folder = default_models_folder();
        }
        self.gpu_vram_cap_pct = self.gpu_vram_cap_pct.clamp(1, 100);
        self.display_gpu_vram_cap_pct = self.display_gpu_vram_cap_pct.clamp(1, 100);
        self.loop_guards.sanitize();
        self.watchdog.sanitize();
        self
    }
}

/// Self-heal configuration for wedged inference-server instances.
///
/// A GPU compute-engine hang (e.g. Intel `xe` CCS engine reset on Battlemage)
/// leaves a llama.cpp process *alive* — its `/health` still returns `ok` — but
/// unable to decode: one thread busy-spins, the GPU is idle, and every request
/// is accepted and then silently stalls until the client gives up. None of the
/// pre-existing liveness checks (`kill -0`, spawn-time `/health`) catch this.
///
/// Three independent, layered mechanisms recover from it:
/// 1. **Engine-reset watchdog** — watches the kernel's DRM `devcoredump` node
///    (created on every engine reset) and recycles GPU-backed instances within
///    seconds. Fast path.
/// 2. **Liveness probe** — periodically probes each instance on a path a wedged
///    server fails to answer (`/slots`) and recycles after repeated timeouts.
/// 3. **Idle timeout** — a long backstop (`read_timeout` on the proxy client):
///    if a forwarded request produces no bytes for this long, it is aborted and
///    the instance recycled. Only fires for wedges the first two miss.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct WatchdogSettings {
    /// Master switch for all self-heal behaviour.
    pub enabled: bool,
    /// Backstop: seconds a proxied request may produce no bytes before it is
    /// aborted and its instance recycled. Long by design — the watchdog and
    /// liveness probe handle the common cases far sooner. 0 disables the
    /// idle backstop.
    pub idle_timeout_secs: u64,
    /// Enable the DRM `devcoredump` engine-reset watchdog.
    pub engine_reset_watch: bool,
    /// How often (seconds) to scan for a new `devcoredump`.
    pub engine_reset_poll_secs: u64,
    /// Directory `/sys/class/drm` is scanned under for `card*/device/devcoredump`.
    /// Overridable so tests / non-standard mounts can point elsewhere.
    pub drm_root: String,
    /// Where to copy each captured `devcoredump` for later analysis. Reading the
    /// node also releases it, clearing the kernel's pending report. Empty = don't
    /// capture (the report is left in place for manual inspection).
    pub devcoredump_capture_dir: String,
    /// Enable the periodic per-instance liveness probe.
    pub liveness_enabled: bool,
    /// HTTP path probed on each instance. Must be a path a healthy server answers
    /// quickly but a wedged one stalls on. `/slots` fits (a wedged llama.cpp
    /// still answers `/health` but hangs `/slots`).
    pub liveness_probe_path: String,
    /// Seconds between liveness probe sweeps.
    pub liveness_interval_secs: u64,
    /// Per-probe timeout (seconds). A probe that doesn't answer in this long
    /// counts as one failure.
    pub liveness_timeout_secs: u64,
    /// Consecutive probe failures before an instance is recycled.
    pub liveness_failures_to_recycle: u32,
    /// Optional webhook POSTed a small JSON `{"text": "..."}` whenever an
    /// instance is auto-recycled, so a recovery is never silent. Empty = off.
    pub notify_webhook_url: String,
}

impl Default for WatchdogSettings {
    fn default() -> Self {
        Self {
            enabled: true,
            idle_timeout_secs: 900,
            engine_reset_watch: true,
            engine_reset_poll_secs: 3,
            drm_root: "/sys/class/drm".into(),
            devcoredump_capture_dir: "~/gpu-hangs".into(),
            liveness_enabled: true,
            liveness_probe_path: "/slots".into(),
            liveness_interval_secs: 30,
            liveness_timeout_secs: 10,
            liveness_failures_to_recycle: 3,
            notify_webhook_url: String::new(),
        }
    }
}

impl WatchdogSettings {
    pub fn sanitize(&mut self) {
        if self.drm_root.trim().is_empty() {
            self.drm_root = "/sys/class/drm".into();
        }
        if self.liveness_probe_path.trim().is_empty() {
            self.liveness_probe_path = "/slots".into();
        }
        self.engine_reset_poll_secs = self.engine_reset_poll_secs.max(1);
        self.liveness_interval_secs = self.liveness_interval_secs.max(1);
        self.liveness_timeout_secs = self.liveness_timeout_secs.max(1);
        self.liveness_failures_to_recycle = self.liveness_failures_to_recycle.max(1);
    }

    fn from_env() -> Self {
        let mut cfg = Self::default();
        if let Some(v) = env_bool("INFERENCE_ROUTER_WATCHDOG_ENABLED") {
            cfg.enabled = v;
        }
        if let Some(v) = env_u64("INFERENCE_ROUTER_IDLE_TIMEOUT_SECS") {
            cfg.idle_timeout_secs = v;
        }
        if let Some(v) = env_bool("INFERENCE_ROUTER_ENGINE_RESET_WATCH") {
            cfg.engine_reset_watch = v;
        }
        if let Some(v) = env_u64("INFERENCE_ROUTER_ENGINE_RESET_POLL_SECS") {
            cfg.engine_reset_poll_secs = v;
        }
        if let Ok(v) = std::env::var("INFERENCE_ROUTER_DRM_ROOT") {
            if !v.trim().is_empty() {
                cfg.drm_root = v;
            }
        }
        if let Ok(v) = std::env::var("INFERENCE_ROUTER_DEVCOREDUMP_CAPTURE_DIR") {
            cfg.devcoredump_capture_dir = v;
        }
        if let Some(v) = env_bool("INFERENCE_ROUTER_LIVENESS_ENABLED") {
            cfg.liveness_enabled = v;
        }
        if let Ok(v) = std::env::var("INFERENCE_ROUTER_LIVENESS_PROBE_PATH") {
            if !v.trim().is_empty() {
                cfg.liveness_probe_path = v;
            }
        }
        if let Some(v) = env_u64("INFERENCE_ROUTER_LIVENESS_INTERVAL_SECS") {
            cfg.liveness_interval_secs = v;
        }
        if let Some(v) = env_u64("INFERENCE_ROUTER_LIVENESS_TIMEOUT_SECS") {
            cfg.liveness_timeout_secs = v;
        }
        if let Some(v) = env_usize("INFERENCE_ROUTER_LIVENESS_FAILURES") {
            cfg.liveness_failures_to_recycle = v as u32;
        }
        if let Ok(v) = std::env::var("INFERENCE_ROUTER_NOTIFY_WEBHOOK_URL") {
            cfg.notify_webhook_url = v;
        }
        cfg.sanitize();
        cfg
    }
}

fn env_pct(name: &str) -> Option<u8> {
    std::env::var(name)
        .ok()?
        .trim()
        .parse::<u8>()
        .ok()
        .filter(|p| (1..=100).contains(p))
}

fn default_models_folder() -> String {
    "~/models".into()
}

/// Which set of model names the OpenAI-compatible `/v1/models` endpoint
/// advertises.
///
/// Aliases always *resolve* at request time regardless of this setting — it
/// only controls what is *listed*.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ModelExposure {
    /// List every configured model *and* every alias (default).
    #[default]
    FullList,
    /// List only the defined aliases.
    AliasesOnly,
}

impl ModelExposure {
    fn from_env() -> Self {
        match env_bool("INFERENCE_ROUTER_EXPOSE_ALIASES_ONLY") {
            Some(true) => ModelExposure::AliasesOnly,
            _ => ModelExposure::FullList,
        }
    }
}

/// Configuration for both streaming and cross-turn tool loop guards.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct LoopGuardSettings {
    /// Streaming response loop detection (SSE chunks)
    pub streaming: StreamingLoopSettings,
    /// Cross-turn tool call cycle detection
    pub tool: ToolLoopSettings,
}

impl LoopGuardSettings {
    fn from_env() -> Self {
        Self {
            streaming: StreamingLoopSettings::from_env(),
            tool: ToolLoopSettings::from_env(),
        }
    }

    fn sanitize(&mut self) {
        self.streaming.sanitize();
        self.tool.sanitize();
    }
}

/// Action to take when a streaming loop is detected.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum StreamingLoopAction {
    /// Replay partial output and inject a corrective prompt (default)
    #[default]
    Heal,
    /// Abort the stream with an error message
    Abort,
    /// Log the detection but allow streaming to continue
    Log,
}

/// Configuration for streaming response loop detection.
///
/// Monitors SSE chunks for repeated text patterns using a sliding window.
/// When a loop is detected, takes the configured action (heal, abort, or log).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct StreamingLoopSettings {
    /// Enable/disable streaming loop detection
    pub enabled: bool,
    /// Size of the sliding window in bytes
    pub window_bytes: usize,
    /// Minimum number of repeats to trigger detection
    pub repeats: usize,
    /// How often to check for loops (milliseconds)
    pub check_interval_ms: u64,
    /// Maximum number of retry attempts when healing
    pub max_retries: usize,
    /// Action to take when a loop is detected
    pub action: StreamingLoopAction,
    /// Whether to replay partial output before corrective prompt
    pub replay_partial: bool,
}

impl Default for StreamingLoopSettings {
    fn default() -> Self {
        Self {
            enabled: true,
            window_bytes: 65_536,
            repeats: 10,
            check_interval_ms: 5_000,
            max_retries: 3,
            action: StreamingLoopAction::Heal,
            replay_partial: true,
        }
    }
}

impl StreamingLoopSettings {
    pub fn sanitize(&mut self) {
        self.window_bytes = self.window_bytes.max(1024);
        self.repeats = self.repeats.max(2);
        self.check_interval_ms = self.check_interval_ms.max(1);
    }

    fn from_env() -> Self {
        let mut cfg = Self::default();
        if let Some(v) = env_bool("INFERENCE_ROUTER_LOOP_ENABLED") {
            cfg.enabled = v;
        }
        if let Some(v) = env_usize("INFERENCE_ROUTER_LOOP_WINDOW") {
            cfg.window_bytes = v.max(1024);
        }
        if let Some(v) = env_usize("INFERENCE_ROUTER_LOOP_REPEATS") {
            cfg.repeats = v.max(2);
        }
        if let Some(v) = env_u64("INFERENCE_ROUTER_LOOP_CHECK_INTERVAL_MS") {
            cfg.check_interval_ms = v.max(1);
        }
        if let Some(v) = env_usize("INFERENCE_ROUTER_LOOP_MAX_RETRIES") {
            cfg.max_retries = v;
        }
        if let Some(v) = env_bool("INFERENCE_ROUTER_LOOP_REPLAY_PARTIAL") {
            cfg.replay_partial = v;
        }
        if let Ok(raw) = std::env::var("INFERENCE_ROUTER_LOOP_ACTION") {
            cfg.action = match raw.trim().to_ascii_lowercase().as_str() {
                "" | "heal" => StreamingLoopAction::Heal,
                "abort" => StreamingLoopAction::Abort,
                "log" => StreamingLoopAction::Log,
                _ => StreamingLoopAction::Heal,
            };
        }
        cfg.sanitize();
        cfg
    }
}

/// Configuration for cross-turn tool loop detection.
///
/// Analyzes message history for repeating tool call sequences.
/// When detected, injects a corrective user message.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct ToolLoopSettings {
    /// Enable/disable tool loop detection
    pub enabled: bool,
    /// Minimum number of repeats to trigger detection
    pub repeats: usize,
    /// Number of recent messages to analyze
    pub window_messages: usize,
}

impl Default for ToolLoopSettings {
    fn default() -> Self {
        Self {
            enabled: true,
            repeats: 3,
            window_messages: 80,
        }
    }
}

impl ToolLoopSettings {
    pub fn sanitize(&mut self) {
        self.repeats = self.repeats.max(2);
        self.window_messages = self.window_messages.max(2);
    }

    fn from_env() -> Self {
        let mut cfg = Self::default();
        if let Some(v) = env_bool("INFERENCE_ROUTER_TOOL_LOOP_ENABLED") {
            cfg.enabled = v;
        }
        if let Some(v) = env_usize("INFERENCE_ROUTER_TOOL_LOOP_REPEATS") {
            cfg.repeats = v.max(2);
        }
        if let Some(v) = env_usize("INFERENCE_ROUTER_TOOL_LOOP_WINDOW_MESSAGES") {
            cfg.window_messages = v.max(2);
        }
        cfg.sanitize();
        cfg
    }
}

fn env_usize(name: &str) -> Option<usize> {
    std::env::var(name).ok()?.trim().parse().ok()
}

fn env_u64(name: &str) -> Option<u64> {
    std::env::var(name).ok()?.trim().parse().ok()
}

fn env_bool(name: &str) -> Option<bool> {
    match std::env::var(name)
        .ok()?
        .trim()
        .to_ascii_lowercase()
        .as_str()
    {
        "1" | "true" | "yes" | "on" => Some(true),
        "0" | "false" | "no" | "off" => Some(false),
        _ => None,
    }
}

#[cfg(test)]
#[path = "settings_tests.rs"]
mod tests;
