use serde::{Deserialize, Serialize};

/// Global application settings.
/// 
/// Configured via environment variables or the dashboard Settings modal.
/// Changes are persisted to `~/.config/inference-router/settings.json`.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct AppSettings {
    /// Loop guard configuration for both streaming and tool loops
    pub loop_guards: LoopGuardSettings,
    /// Which set of model names is advertised on `/v1/models`.
    pub model_exposure: ModelExposure,
}

impl AppSettings {
    pub fn from_env() -> Self {
        Self {
            loop_guards: LoopGuardSettings::from_env(),
            model_exposure: ModelExposure::from_env(),
        }
    }

    pub fn sanitized(mut self) -> Self {
        self.loop_guards.sanitize();
        self
    }
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
mod tests {
    use super::*;

    #[test]
    fn defaults_match_loop_guard_runtime_defaults() {
        let settings = AppSettings::default();
        assert!(settings.loop_guards.streaming.enabled);
        assert_eq!(settings.loop_guards.streaming.window_bytes, 65_536);
        assert_eq!(settings.loop_guards.streaming.repeats, 10);
        assert_eq!(settings.loop_guards.streaming.check_interval_ms, 5_000);
        assert_eq!(settings.loop_guards.streaming.max_retries, 3);
        assert_eq!(
            settings.loop_guards.streaming.action,
            StreamingLoopAction::Heal
        );
        assert!(settings.loop_guards.streaming.replay_partial);
        assert!(settings.loop_guards.tool.enabled);
        assert_eq!(settings.loop_guards.tool.repeats, 3);
        assert_eq!(settings.loop_guards.tool.window_messages, 80);
    }

    #[test]
    fn sanitize_clamps_runtime_minimums() {
        let mut settings = AppSettings::default();
        settings.loop_guards.streaming.window_bytes = 1;
        settings.loop_guards.streaming.repeats = 0;
        settings.loop_guards.streaming.check_interval_ms = 0;
        settings.loop_guards.tool.repeats = 0;
        settings.loop_guards.tool.window_messages = 0;

        let settings = settings.sanitized();
        assert_eq!(settings.loop_guards.streaming.window_bytes, 1024);
        assert_eq!(settings.loop_guards.streaming.repeats, 2);
        assert_eq!(settings.loop_guards.streaming.check_interval_ms, 1);
        assert_eq!(settings.loop_guards.tool.repeats, 2);
        assert_eq!(settings.loop_guards.tool.window_messages, 2);
    }
}
