use crate::config::{ModelConfig, ModelRole, SplitMode, WeightsFormat};
use std::collections::HashMap;
use std::path::PathBuf;
use std::time::Duration;
use tracing::{debug, info, warn};

/// A live inference-server process owned by the orchestrator.
///
/// The `Child` is retained for the process's lifetime so that
/// `kill_on_drop(true)` actually fires when the orchestrator shuts down —
/// ensuring llama-server processes die with us instead of orphaning.
struct RunningChild {
    port: u16,
    model_id: String,
    // Kept alive purely so tokio's `kill_on_drop(true)` applies on shutdown:
    // reading the field itself is not meaningful, dropping it is.
    #[allow(dead_code)]
    child: tokio::process::Child,
}

/// Owns running inference-server processes. Single instance per `Orchestrator`.
#[derive(Default)]
pub struct ProcessManager {
    running: HashMap<i32, RunningChild>,
    /// model_id → port, populated on `register`, cleared on `stop`/`forget`.
    /// The only in-memory source of truth for which port a live process
    /// is listening on — ModelConfig no longer carries a port field.
    running_ports: HashMap<String, u16>,
}

/// A freshly spawned but not-yet-healthy child. Returned by `spawn_child`
/// so the caller can await readiness WITHOUT holding the ProcessManager
/// mutex across the 180s health check.
pub struct PendingChild {
    pub pid: i32,
    pub port: u16,
    model_id: String,
    child: tokio::process::Child,
    /// Receives the sum of all measured KV/RS cache bytes parsed from
    /// the child's stderr during startup. Zero if the lines were never
    /// seen (Safetensors servers, crash before init, etc.).
    kv_bytes_rx: tokio::sync::oneshot::Receiver<u64>,
}

impl ProcessManager {
    /// Fork + exec the inference server. Returns immediately — the caller
    /// must drive health via `wait_for_health` and then `register` the PID
    /// back into this manager on success (or drop the `PendingChild` on
    /// failure, which SIGKILLs via `kill_on_drop`).
    ///
    /// `draft` is the resolved draft model config when the target has
    /// `draft_model_id` set; `None` otherwise. Keeping it an explicit
    /// argument (rather than a store reference) keeps the spawn path
    /// testable and makes argument resolution the orchestrator's job.
    pub fn spawn_child(
        &self,
        model: &ModelConfig,
        draft: Option<&ModelConfig>,
    ) -> Result<PendingChild, SpawnError> {
        let port = find_free_port().ok_or(SpawnError::NoFreePort)?;
        let args = build_command_args(model, draft, port);

        info!(
            model = model.id,
            port,
            binary = ?model.binary,
            ?args,
            "spawning inference server",
        );

        // `kill_on_drop(true)` is the safety net for orchestrator shutdown.
        // stderr is piped so we can parse the KV-cache size from llama.cpp's
        // startup output; a background task drains it and forwards each line
        // to stderr so the user experience is unchanged.
        let mut child = tokio::process::Command::new(&model.binary)
            .args(&args)
            .process_group(0)
            .stdout(std::process::Stdio::inherit())
            .stderr(std::process::Stdio::piped())
            .kill_on_drop(true)
            .spawn()
            .map_err(|e| SpawnError::SpawnFailed(e, model.binary.clone()))?;

        let pid = child.id().expect("spawned child has no PID") as i32;

        // Drain stderr: forward every line to our stderr (preserving the
        // existing log experience), and sum up all KV/RS cache size lines.
        let stderr = child.stderr.take().expect("stderr was piped");
        let (kv_tx, kv_rx) = tokio::sync::oneshot::channel::<u64>();
        tokio::spawn(async move {
            use tokio::io::AsyncBufReadExt;
            let mut lines = tokio::io::BufReader::new(stderr).lines();
            let mut total_kv_bytes = 0u64;
            while let Ok(Some(line)) = lines.next_line().await {
                if let Some(mib) = parse_kv_size_mib(&line) {
                    total_kv_bytes += (mib * 1024.0 * 1024.0) as u64;
                }
                eprintln!("{line}");
            }
            let _ = kv_tx.send(total_kv_bytes);
        });

        Ok(PendingChild { pid, port, model_id: model.id.clone(), child, kv_bytes_rx: kv_rx })
    }

    /// Install a healthy child into the running table so its `kill_on_drop`
    /// survives until explicit `stop`/`forget` or orchestrator shutdown.
    pub fn register(&mut self, pending: PendingChild) {
        self.running_ports.insert(pending.model_id.clone(), pending.port);
        self.running.insert(
            pending.pid,
            RunningChild {
                port: pending.port,
                model_id: pending.model_id,
                child: pending.child,
            },
        );
    }

    /// Returns the port the live process for `model_id` is listening on,
    /// or `None` if the model is not currently running.
    pub fn port_for_model(&self, model_id: &str) -> Option<u16> {
        self.running_ports.get(model_id).copied()
    }

    /// Directly records a port for a model without spawning a process.
    /// Used in integration tests to synthesize a Running model.
    pub fn register_port(&mut self, model_id: &str, port: u16) {
        self.running_ports.insert(model_id.into(), port);
    }

    /// SIGTERM, wait briefly, SIGKILL if still alive, then drop our Child
    /// handle so tokio reaps it. Only signals when the PID is tracked —
    /// callers asking us to stop an untracked PID get a warning but no
    /// signal (defends against PID reuse after a crashed/forgotten child).
    pub async fn stop(&mut self, pid: i32) {
        let Some(rc) = self.running.remove(&pid) else {
            warn!(pid, "stopping process not in running table; skipping signal");
            return;
        };

        let _ = kill_process_group(pid, nix::sys::signal::Signal::SIGTERM);
        tokio::time::sleep(Duration::from_millis(500)).await;
        if is_process_alive(pid) {
            warn!(pid, "process didn't exit after SIGTERM, sending SIGKILL");
            let _ = kill_process_group(pid, nix::sys::signal::Signal::SIGKILL);
        }

        self.running_ports.remove(&rc.model_id);
        info!(pid, port = rc.port, model = rc.model_id, "inference server stopped");
        // rc.child drops here; kill_on_drop is a no-op on an already-
        // terminated process but the drop drives tokio's reaper.
    }

    pub fn is_alive(&self, pid: i32) -> bool {
        is_process_alive(pid)
    }

    /// Forget a pid (used when reconcile observes the process has died, or
    /// when delete fails to stop cleanly). Dropping the `RunningChild` lets
    /// tokio reap it and fires `kill_on_drop` as a safety net.
    pub fn forget(&mut self, pid: i32) {
        if let Some(rc) = self.running.remove(&pid) {
            self.running_ports.remove(&rc.model_id);
        }
    }
}

impl PendingChild {
    /// Waits until `/health` returns 200 on the child's port, or until the
    /// child exits, whichever comes first. Returns the total KV+RS cache
    /// bytes parsed from llama.cpp's startup logs (0 if not seen — e.g.
    /// Safetensors backends or a crash before context init).
    pub async fn wait_for_health(&mut self, timeout: Duration) -> Result<u64, HealthCheckError> {
        wait_for_health_or_exit(&mut self.child, self.port, timeout).await?;
        // The KV size lines appear during context init, well before the
        // health endpoint becomes available. Give the stderr reader a brief
        // window to flush any remaining buffered lines before we read.
        let kv_bytes = tokio::time::timeout(
            Duration::from_millis(200),
            &mut self.kv_bytes_rx,
        )
        .await
        .ok()
        .and_then(|r| r.ok())
        .unwrap_or(0);
        Ok(kv_bytes)
    }
}

/// Builds the argv (excluding the binary itself) for a model.
///
/// `draft` is the caller-resolved `ModelConfig` the target's
/// `draft_model_id` points at. When `Some`, the builder emits the
/// speculative-decoding flags (`-md`, `-ngld`, `-devd`, `-cd`, `-ctkd`,
/// `-ctvd`) from the draft's own fields, followed by the policy flags
/// (`--draft-max`, `--draft-min`, `--draft-p-min`, `--ctx-checkpoints`,
/// `--checkpoint-every-n-tokens`) from the target's fields. When `None`,
/// no spec-decode flags are emitted — even if the target has them set —
/// so the argv stays runnable in test code without needing a draft
/// lookup.
pub fn build_command_args(model: &ModelConfig, draft: Option<&ModelConfig>, port: u16) -> Vec<String> {
    let mut args = Vec::new();
    match model.weights_format {
        WeightsFormat::Gguf => {
            args.push("-m".into());
            args.push(model.model_path.to_string_lossy().into_owned());
            args.push("-c".into());
            args.push(model.context.to_string());
            args.push("--port".into());
            args.push(port.to_string());
            args.push("--temp".into());
            args.push(model.temperature.to_string());
            args.push("--top-p".into());
            args.push(model.top_p.to_string());
            args.push("--top-k".into());
            args.push(model.top_k.to_string());
            if model.min_p > 0.0 {
                args.push("--min-p".into());
                args.push(model.min_p.to_string());
            }
            args.push("--presence-penalty".into());
            args.push(model.presence_penalty.to_string());
            args.push("--repeat-penalty".into());
            args.push(model.repeat_penalty.to_string());

            // llama.cpp structured flags. User-typed `extra_args` appends
            // after these so it can override via last-wins.
            //
            // Modern llama-server's `--flash-attn` expects a value (`on|off|
            // auto`); a bare `--flash-attn` consumes the next argv token,
            // which breaks everything. Always emit the value.
            args.push("--flash-attn".into());
            args.push(if model.flash_attn { "on".into() } else { "off".into() });
            if let Some(n) = model.n_gpu_layers {
                args.push("-ngl".into());
                args.push(n.to_string());
            }
            if model.mlock {
                args.push("--mlock".into());
            }
            if model.no_mmap {
                args.push("--no-mmap".into());
            }
            if let Some(n) = model.parallel_slots {
                args.push("--parallel".into());
                args.push(n.to_string());
            }
            if let Some(k) = model.cache_type_k {
                args.push("--cache-type-k".into());
                args.push(k.as_arg().into());
            }
            if let Some(v) = model.cache_type_v {
                args.push("--cache-type-v".into());
                args.push(v.as_arg().into());
            }
            if let Some(mode) = model.split_mode {
                args.push("--split-mode".into());
                args.push(match mode {
                    SplitMode::None => "none".into(),
                    SplitMode::Layer => "layer".into(),
                    SplitMode::Row => "row".into(),
                    SplitMode::Tensor => "tensor".into(),
                });
            }
            if let Some(g) = model.main_gpu {
                args.push("--main-gpu".into());
                args.push(g.to_string());
            }
            if let Some(ref ts) = model.tensor_split {
                args.push("--tensor-split".into());
                args.push(ts.clone());
            }
            if let Some(t) = model.threads {
                args.push("--threads".into());
                args.push(t.to_string());
            }
            if let Some(c) = model.cache_ram_mib {
                args.push("--cache-ram".into());
                args.push(c.to_string());
            }
            if let Some(rf) = model.reasoning_format {
                args.push("--reasoning-format".into());
                args.push(rf.as_arg().into());
            }
            if let Some(rb) = model.reasoning_budget {
                args.push("--reasoning-budget".into());
                args.push(rb.to_string());
            }
            if let Some(ref kw) = model.chat_template_kwargs {
                args.push("--chat-template-kwargs".into());
                args.push(kw.clone());
            }

            // Speculative decoding. Only emitted when the target has a
            // resolved draft — a lone `draft_model_id` without a
            // matching entry is an orchestrator bug, not something to
            // silently degrade from.
            if let (Some(d), ModelRole::Target) = (draft, model.role) {
                args.push("-md".into());
                args.push(d.model_path.to_string_lossy().into_owned());
                if let Some(n) = d.n_gpu_layers {
                    args.push("-ngld".into());
                    args.push(n.to_string());
                }
                if let Some(ref dev) = d.device {
                    args.push("-devd".into());
                    args.push(dev.clone());
                }
                // Always pin the draft's context — defaulting to the
                // target's ctx is wasteful (bench showed 8-30 GB of
                // draft KV at 256k) and surprising.
                args.push("-cd".into());
                args.push(d.context.to_string());
                if let Some(k) = d.cache_type_k {
                    args.push("-ctkd".into());
                    args.push(k.as_arg().into());
                }
                if let Some(v) = d.cache_type_v {
                    args.push("-ctvd".into());
                    args.push(v.as_arg().into());
                }

                // Target-side spec-decode policy knobs. Target owns
                // these because they're about *how hard* the target
                // drives the draft, not about the draft model itself.
                if let Some(n) = model.draft_max {
                    args.push("--draft-max".into());
                    args.push(n.to_string());
                }
                if let Some(n) = model.draft_min {
                    args.push("--draft-min".into());
                    args.push(n.to_string());
                }
                if let Some(p) = model.draft_p_min {
                    args.push("--draft-p-min".into());
                    args.push(p.to_string());
                }
                if let Some(n) = model.ctx_checkpoints {
                    args.push("--ctx-checkpoints".into());
                    args.push(n.to_string());
                }
                if let Some(n) = model.checkpoint_every_n_tokens {
                    args.push("--checkpoint-every-n-tokens".into());
                    args.push(n.to_string());
                }
            }
        }
        WeightsFormat::Safetensors => {
            args.push("--model".into());
            args.push(model.model_path.to_string_lossy().into_owned());
            args.push("--port".into());
            args.push(port.to_string());
            args.push("--max-model-len".into());
            args.push(model.context.to_string());
        }
    }
    args.extend(model.extra_args.iter().cloned());
    args
}

/// Polls `/health` on `port` until success, OR returns an error as soon as
/// the child process exits (so bad argv / OOM / missing model manifest as
/// a concrete error instead of a timeout).
async fn wait_for_health_or_exit(
    child: &mut tokio::process::Child,
    port: u16,
    timeout: Duration,
) -> Result<(), HealthCheckError> {
    let client = reqwest::Client::builder()
        .timeout(Duration::from_secs(2))
        .build()
        .expect("reqwest client");
    let url = format!("http://127.0.0.1:{}/health", port);
    let start = std::time::Instant::now();

    loop {
        // Fast-fail on early child exit.
        match child.try_wait() {
            Ok(Some(status)) => {
                return Err(HealthCheckError::ChildExited(status.code(), status.to_string()));
            }
            Ok(None) => {} // still running
            Err(e) => {
                return Err(HealthCheckError::ChildExited(None, format!("try_wait: {e}")));
            }
        }

        if start.elapsed() > timeout {
            return Err(HealthCheckError::Timeout(timeout));
        }

        match client.get(&url).send().await {
            Ok(response) if response.status().is_success() => {
                debug!(port, "health check passed");
                return Ok(());
            }
            Ok(response) => {
                debug!(port, status = response.status().as_u16(), "health non-2xx, retrying");
            }
            Err(e) => {
                debug!(port, error = %e, "health connect failed, retrying");
            }
        }
        tokio::time::sleep(Duration::from_millis(500)).await;
    }
}

/// Binds to `127.0.0.1:0` and lets the OS assign a free ephemeral port.
/// The listener is dropped immediately; there is a brief window where
/// another process could grab the same port, but in practice this is
/// fine for localhost-only inference servers.
fn find_free_port() -> Option<u16> {
    std::net::TcpListener::bind("127.0.0.1:0")
        .ok()
        .and_then(|l| l.local_addr().ok())
        .map(|a| a.port())
}

/// Parses the MiB value from llama.cpp's KV/RS cache summary log lines:
///
///   `llama_kv_cache_init: size = 1234.56 MiB ( 32768 cells, 64 layers, …)`
///   `llama_memory_recurrent_init: size =   56.78 MiB (  8192 cells, …)`
///
/// Returns `None` for unrelated lines. The `cells,` guard prevents false
/// matches on quantisation progress lines which also contain "size = X MiB"
/// but never have "cells,".
fn parse_kv_size_mib(line: &str) -> Option<f64> {
    let after = line.split("size =").nth(1)?.trim_start();
    let mib_pos = after.find(" MiB")?;
    let mib: f64 = after[..mib_pos].trim().parse().ok()?;
    after[mib_pos..].contains("cells,").then_some(mib)
}

fn is_process_alive(pid: i32) -> bool {
    nix::sys::signal::kill(nix::unistd::Pid::from_raw(pid), None).is_ok()
}

fn kill_process_group(pid: i32, sig: nix::sys::signal::Signal) -> nix::Result<()> {
    nix::sys::signal::kill(nix::unistd::Pid::from_raw(-pid), sig)
}

#[derive(Debug, thiserror::Error)]
pub enum SpawnError {
    #[error("failed to spawn '{}': {}", .1.display(), .0)]
    SpawnFailed(std::io::Error, PathBuf),

    #[error("health check failed: {}", .0)]
    HealthCheckFailed(HealthCheckError),

    #[error("no free port available on 127.0.0.1")]
    NoFreePort,
}

#[derive(Debug, thiserror::Error)]
pub enum HealthCheckError {
    #[error("timed out after {}s", .0.as_secs())]
    Timeout(Duration),

    #[error("http error: {}", .0)]
    Http(#[from] reqwest::Error),

    #[error("inference server exited before becoming ready (exit code {:?}): {}", .0, .1)]
    ChildExited(Option<i32>, String),
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::{CacheType, ReasoningFormat};

    // ----- parse_kv_size_mib -----

    #[test]
    fn parses_standard_kv_cache_line() {
        // Typical output from llama_kv_cache_init for a non-SWA model.
        let line = "llama_kv_cache_init: size = 1024.00 MiB ( 32768 cells, 64 layers, 2/2 seqs), K (q8_0): 512.00 MiB, V (q8_0): 512.00 MiB";
        let mib = parse_kv_size_mib(line).unwrap();
        assert!((mib - 1024.0).abs() < 0.01, "got {mib}");
    }

    #[test]
    fn parses_recurrent_state_line() {
        // llama_memory_recurrent_init uses "RS buffer size" and the same
        // summary format — should be picked up and summed with KV cache.
        let line = "llama_memory_recurrent_init: size =   56.25 MiB (  8192 cells, 28 layers, 2 seqs), R (f16):   28.12 MiB, S (f16):   28.12 MiB";
        let mib = parse_kv_size_mib(line).unwrap();
        assert!((mib - 56.25).abs() < 0.01, "got {mib}");
    }

    #[test]
    fn parses_swa_kv_cache_line() {
        // Hybrid models (e.g. Qwen3.6) emit separate lines for the
        // full-attention and sliding-window KV caches. Both must match.
        let line = "llama_kv_cache_init: size =  128.50 MiB (  4096 cells, 32 layers, 2/2 seqs), K (q8_0):  64.25 MiB, V (q8_0):  64.25 MiB";
        assert!(parse_kv_size_mib(line).is_some());
    }

    #[test]
    fn ignores_quant_progress_lines() {
        // llama-quantize logs "size = X MiB -> Y MiB" with no "cells," —
        // must not be mistaken for a KV cache line.
        let line = "[ 100/ 291]          blk.0.attn_k.weight - [ 4096,  4096,    1,    1], type =    q8_0, size =   16.00 MiB";
        assert!(parse_kv_size_mib(line).is_none());
    }

    #[test]
    fn ignores_model_buffer_lines() {
        let line = "llm_load_tensors:      VULKAN0 model buffer size =  8192.00 MiB";
        assert!(parse_kv_size_mib(line).is_none());
    }

    #[test]
    fn ignores_unrelated_lines() {
        assert!(parse_kv_size_mib("llama_new_context_with_model: n_ctx = 262144").is_none());
        assert!(parse_kv_size_mib("srv  log: HTTP server is listening").is_none());
    }

    fn gguf_model() -> ModelConfig {
        ModelConfig {
            id: "m".into(),
            name: "M".into(),
            binary: PathBuf::from("/usr/local/bin/llama-server"),
            model_path: PathBuf::from("/models/m.gguf"),
            ..ModelConfig::default()
        }
    }

    #[test]
    fn gguf_argv_shape() {
        let args = build_command_args(&gguf_model(), None, 9001);
        assert_eq!(
            args,
            vec![
                "-m", "/models/m.gguf",
                "-c", "4096",
                "--port", "9001",
                "--temp", "0.6",
                "--top-p", "0.95",
                "--top-k", "40",
                "--presence-penalty", "0",
                "--repeat-penalty", "1",
                "--flash-attn", "off",
            ],
        );
    }

    #[test]
    fn gguf_argv_includes_min_p_when_positive() {
        let mut m = gguf_model();
        m.min_p = 0.05;
        let args = build_command_args(&m, None, 9001);
        assert!(args.windows(2).any(|w| w == ["--min-p", "0.05"]));
    }

    #[test]
    fn gguf_argv_omits_min_p_when_zero() {
        let args = build_command_args(&gguf_model(), None, 9001);
        assert!(!args.iter().any(|a| a == "--min-p"));
    }

    #[test]
    fn safetensors_argv_shape() {
        let mut m = gguf_model();
        m.weights_format = WeightsFormat::Safetensors;
        m.model_path = PathBuf::from("/models/m-safetensors");
        let args = build_command_args(&m, None, 9001);
        assert_eq!(
            args,
            vec![
                "--model", "/models/m-safetensors",
                "--port", "9001",
                "--max-model-len", "4096",
            ],
        );
    }

    #[test]
    fn extra_args_appended_last_for_gguf() {
        let mut m = gguf_model();
        m.extra_args = vec!["--flash-attn".into(), "-ngl".into(), "99".into()];
        let args = build_command_args(&m, None, 9001);
        assert_eq!(
            &args[args.len() - 3..],
            &["--flash-attn", "-ngl", "99"],
        );
    }

    #[test]
    fn gguf_argv_emits_structured_llama_flags() {
        let mut m = gguf_model();
        m.flash_attn = true;
        m.n_gpu_layers = Some(99);
        m.mlock = true;
        m.no_mmap = true;
        m.parallel_slots = Some(4);
        m.cache_type_k = Some(CacheType::Q8_0);
        m.cache_type_v = Some(CacheType::Q8_0);
        let args = build_command_args(&m, None, 9001);
        let joined = args.join(" ");
        assert!(joined.contains("--flash-attn on"), "{joined}");
        assert!(joined.contains("-ngl 99"), "{joined}");
        assert!(joined.contains("--mlock"), "{joined}");
        assert!(joined.contains("--no-mmap"), "{joined}");
        assert!(joined.contains("--parallel 4"), "{joined}");
        assert!(joined.contains("--cache-type-k q8_0"), "{joined}");
        assert!(joined.contains("--cache-type-v q8_0"), "{joined}");
    }

    #[test]
    fn gguf_argv_emits_flash_attn_off_when_disabled() {
        let args = build_command_args(&gguf_model(), None, 9001);
        // --flash-attn is always emitted as `off` by default — it used to be
        // a bare flag; modern llama-server requires a value.
        assert_eq!(
            args.windows(2).find(|w| w[0] == "--flash-attn").map(|w| w[1].as_str()),
            Some("off"),
        );
    }

    #[test]
    fn gguf_argv_omits_other_structured_flags_when_unset() {
        let args = build_command_args(&gguf_model(), None, 9001);
        let joined = args.join(" ");
        assert!(!joined.contains("-ngl"));
        assert!(!joined.contains("--mlock"));
        assert!(!joined.contains("--no-mmap"));
        assert!(!joined.contains("--parallel"));
        assert!(!joined.contains("--cache-type-k"));
        assert!(!joined.contains("--cache-type-v"));
        assert!(!joined.contains("--split-mode"));
        assert!(!joined.contains("--main-gpu"));
        assert!(!joined.contains("--tensor-split"));
    }

    #[test]
    fn gguf_argv_emits_penalties_from_structured_fields() {
        let mut m = gguf_model();
        m.presence_penalty = 1.5;
        m.repeat_penalty = 1.1;
        let args = build_command_args(&m, None, 9001);
        let joined = args.join(" ");
        assert!(joined.contains("--presence-penalty 1.5"), "{joined}");
        assert!(joined.contains("--repeat-penalty 1.1"), "{joined}");
    }

    #[test]
    fn gguf_argv_emits_threads_cache_ram_reasoning_when_set() {
        let mut m = gguf_model();
        m.threads = Some(16);
        m.cache_ram_mib = Some(0);
        m.reasoning_format = Some(ReasoningFormat::Deepseek);
        m.reasoning_budget = Some(0);
        m.chat_template_kwargs = Some(r#"{"enable_thinking":false}"#.into());
        let args = build_command_args(&m, None, 9001);
        let joined = args.join(" ");
        assert!(joined.contains("--threads 16"), "{joined}");
        assert!(joined.contains("--cache-ram 0"), "{joined}");
        assert!(joined.contains("--reasoning-format deepseek"), "{joined}");
        assert!(joined.contains("--reasoning-budget 0"), "{joined}");
        assert!(joined.contains(r#"--chat-template-kwargs {"enable_thinking":false}"#), "{joined}");
    }

    #[test]
    fn gguf_argv_omits_new_flags_when_unset() {
        let args = build_command_args(&gguf_model(), None, 9001);
        let joined = args.join(" ");
        assert!(!joined.contains("--threads"));
        assert!(!joined.contains("--cache-ram"));
        assert!(!joined.contains("--reasoning-format"));
        assert!(!joined.contains("--reasoning-budget"));
        assert!(!joined.contains("--chat-template-kwargs"));
    }

    #[test]
    fn gguf_argv_reasoning_format_kebab_for_deepseek_legacy() {
        let mut m = gguf_model();
        m.reasoning_format = Some(ReasoningFormat::DeepseekLegacy);
        let args = build_command_args(&m, None, 9001);
        let joined = args.join(" ");
        assert!(joined.contains("--reasoning-format deepseek-legacy"), "{joined}");
    }

    #[test]
    fn gguf_argv_emits_split_mode_main_gpu_tensor_split() {
        let mut m = gguf_model();
        m.split_mode = Some(SplitMode::Row);
        m.main_gpu = Some(2);
        m.tensor_split = Some("0.5,0.5,0".into());
        let args = build_command_args(&m, None, 9001);
        let j = args.join(" ");
        assert!(j.contains("--split-mode row"), "{j}");
        assert!(j.contains("--main-gpu 2"), "{j}");
        assert!(j.contains("--tensor-split 0.5,0.5,0"), "{j}");
    }

    #[test]
    fn extra_args_appended_after_structured_flags_for_last_wins() {
        let mut m = gguf_model();
        m.n_gpu_layers = Some(50);
        m.extra_args = vec!["-ngl".into(), "99".into()];
        let args = build_command_args(&m, None, 9001);
        // Both present, extra_args version is last → wins on llama.cpp.
        let idx_first = args.iter().position(|a| a == "-ngl").unwrap();
        let idx_last = args.iter().rposition(|a| a == "-ngl").unwrap();
        assert_ne!(idx_first, idx_last);
        assert_eq!(args[idx_first + 1], "50");
        assert_eq!(args[idx_last + 1], "99");
    }

    #[test]
    fn extra_args_appended_last_for_safetensors() {
        let mut m = gguf_model();
        m.weights_format = WeightsFormat::Safetensors;
        m.extra_args = vec!["--tensor-parallel-size".into(), "2".into()];
        let args = build_command_args(&m, None, 9001);
        assert_eq!(
            &args[args.len() - 2..],
            &["--tensor-parallel-size", "2"],
        );
    }

    // ----- Speculative decoding -----

    fn draft_model() -> ModelConfig {
        use crate::config::ModelRole;
        ModelConfig {
            id: "draft".into(),
            name: "D".into(),
            role: ModelRole::Draft,
            model_path: PathBuf::from("/models/draft.gguf"),
            context: 16384,
            n_gpu_layers: Some(99),
            device: Some("Vulkan1".into()),
            cache_type_k: Some(CacheType::Q8_0),
            cache_type_v: Some(CacheType::Q8_0),
            ..ModelConfig::default()
        }
    }

    fn find_flag<'a>(args: &'a [String], flag: &str) -> Option<&'a str> {
        args.windows(2)
            .find(|w| w[0] == flag)
            .map(|w| w[1].as_str())
    }

    #[test]
    fn spec_decode_argv_emits_full_draft_and_policy_flags() {
        let mut t = gguf_model();
        t.draft_model_id = Some("draft".into());
        t.draft_max = Some(16);
        t.draft_min = Some(1);
        t.draft_p_min = Some(0.75);
        t.ctx_checkpoints = Some(4);
        t.checkpoint_every_n_tokens = Some(-1);
        let d = draft_model();
        let args = build_command_args(&t, Some(&d), 9001);

        assert_eq!(find_flag(&args, "-md"), Some("/models/draft.gguf"));
        assert_eq!(find_flag(&args, "-ngld"), Some("99"));
        assert_eq!(find_flag(&args, "-devd"), Some("Vulkan1"));
        assert_eq!(find_flag(&args, "-cd"), Some("16384"));
        assert_eq!(find_flag(&args, "-ctkd"), Some("q8_0"));
        assert_eq!(find_flag(&args, "-ctvd"), Some("q8_0"));
        assert_eq!(find_flag(&args, "--draft-max"), Some("16"));
        assert_eq!(find_flag(&args, "--draft-min"), Some("1"));
        assert_eq!(find_flag(&args, "--draft-p-min"), Some("0.75"));
        assert_eq!(find_flag(&args, "--ctx-checkpoints"), Some("4"));
        assert_eq!(find_flag(&args, "--checkpoint-every-n-tokens"), Some("-1"));
    }

    #[test]
    fn spec_decode_argv_omitted_when_no_draft_resolved() {
        // Target has draft_model_id + policy fields set, but the caller
        // passed draft=None (e.g. draft entry was deleted out from under
        // the config). Don't emit any spec flags — the orchestrator
        // should have failed the spawn upstream, but argv must remain
        // sane.
        let mut t = gguf_model();
        t.draft_model_id = Some("missing".into());
        t.draft_max = Some(16);
        t.ctx_checkpoints = Some(4);
        let args = build_command_args(&t, None, 9001);
        let joined = args.join(" ");
        assert!(!joined.contains("-md"), "{joined}");
        assert!(!joined.contains("--draft-max"), "{joined}");
        assert!(!joined.contains("--ctx-checkpoints"), "{joined}");
    }

    #[test]
    fn spec_decode_argv_omits_unset_policy_flags() {
        // Only draft_max set; --draft-min/--draft-p-min/--ctx-checkpoints
        // stay off. Lets users opt into individual policy knobs without
        // being forced to set all five.
        let mut t = gguf_model();
        t.draft_model_id = Some("draft".into());
        t.draft_max = Some(8);
        let args = build_command_args(&t, Some(&draft_model()), 9001);
        assert_eq!(find_flag(&args, "--draft-max"), Some("8"));
        let joined = args.join(" ");
        assert!(!joined.contains("--draft-min"), "{joined}");
        assert!(!joined.contains("--draft-p-min"), "{joined}");
        assert!(!joined.contains("--ctx-checkpoints"), "{joined}");
        assert!(!joined.contains("--checkpoint-every-n-tokens"), "{joined}");
    }

    #[test]
    fn spec_decode_argv_always_pins_draft_context() {
        // `-cd` is emitted even when `context` on the draft equals the
        // target's, because letting llama-server default it to the
        // target's ctx makes draft KV explode at large target ctx.
        let mut t = gguf_model();
        t.draft_model_id = Some("draft".into());
        let args = build_command_args(&t, Some(&draft_model()), 9001);
        assert_eq!(find_flag(&args, "-cd"), Some("16384"));
    }

    #[test]
    fn spec_decode_argv_skips_draft_device_flag_when_unset() {
        // Draft without a device: orchestrator can still spawn (the
        // draft would land on whatever GPU llama.cpp picks) but we
        // don't emit `-devd ""`.
        let mut d = draft_model();
        d.device = None;
        let mut t = gguf_model();
        t.draft_model_id = Some("draft".into());
        let args = build_command_args(&t, Some(&d), 9001);
        let joined = args.join(" ");
        assert!(!joined.contains("-devd"), "{joined}");
    }
}
