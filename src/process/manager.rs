use crate::config::{ModelConfig, ModelRole, SplitMode, WeightsFormat};
use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;
use std::time::Duration;
use tracing::{debug, info, warn};

/// A live inference-server process owned by the orchestrator.
struct RunningChild {
    port: u16,
    model_id: String,
    #[allow(dead_code)]
    child: tokio::process::Child,
}

/// Per-instance runtime metadata for a live inference-server process.
pub struct InstanceInfo {
    pub pid: i32,
    pub port: u16,
    /// Number of in-flight requests currently routed to this instance.
    pub active: Arc<AtomicUsize>,
}

/// RAII handle returned by `ensure_loaded`. Holds the port to forward to
/// and keeps the instance's active counter incremented for the duration of
/// the request. Dropped when the response body is fully consumed or the
/// connection closes.
#[derive(Debug)]
pub struct RequestGuard {
    pub port: u16,
    active: Arc<AtomicUsize>,
}

impl Drop for RequestGuard {
    fn drop(&mut self) {
        self.active.fetch_sub(1, Ordering::Relaxed);
    }
}

/// Owns running inference-server processes. Single instance per `Orchestrator`.
#[derive(Default)]
pub struct ProcessManager {
    /// pid → child handle (kept for kill_on_drop).
    running: HashMap<i32, RunningChild>,
    /// model_id → pool of live instances.
    instances: HashMap<String, Vec<InstanceInfo>>,
}

/// A freshly spawned but not-yet-healthy child.
pub struct PendingChild {
    pub pid: i32,
    pub port: u16,
    model_id: String,
    child: tokio::process::Child,
    kv_bytes_rx: tokio::sync::oneshot::Receiver<u64>,
}

impl ProcessManager {
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

        let mut child = tokio::process::Command::new(&model.binary)
            .args(&args)
            .process_group(0)
            .stdout(std::process::Stdio::inherit())
            .stderr(std::process::Stdio::piped())
            .kill_on_drop(true)
            .spawn()
            .map_err(|e| SpawnError::SpawnFailed(e, model.binary.clone()))?;

        let pid = child.id().expect("spawned child has no PID") as i32;

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

    /// Install a healthy child into the running table. Returns a `RequestGuard`
    /// for the caller's in-flight request (active starts at 1).
    pub fn register(&mut self, pending: PendingChild) -> RequestGuard {
        let active = Arc::new(AtomicUsize::new(1));
        let port = pending.port;
        self.instances.entry(pending.model_id.clone()).or_default().push(InstanceInfo {
            pid: pending.pid,
            port,
            active: active.clone(),
        });
        self.running.insert(pending.pid, RunningChild {
            port,
            model_id: pending.model_id,
            child: pending.child,
        });
        RequestGuard { port, active }
    }

    /// Find an idle instance (active == 0), increment its counter, and return
    /// a guard. Returns `None` if no instances exist or all are busy.
    pub fn acquire_idle_instance(&self, model_id: &str) -> Option<RequestGuard> {
        for inst in self.instances.get(model_id)?.iter() {
            if inst.active.load(Ordering::Relaxed) == 0 {
                inst.active.fetch_add(1, Ordering::Relaxed);
                return Some(RequestGuard { port: inst.port, active: inst.active.clone() });
            }
        }
        None
    }

    /// Find any instance — idle first, then any busy one. Returns `None` only
    /// if there are no instances at all.
    pub fn acquire_any_instance(&self, model_id: &str) -> Option<RequestGuard> {
        if let Some(g) = self.acquire_idle_instance(model_id) {
            return Some(g);
        }
        let inst = self.instances.get(model_id)?.first()?;
        inst.active.fetch_add(1, Ordering::Relaxed);
        Some(RequestGuard { port: inst.port, active: inst.active.clone() })
    }

    /// Number of live instances for a model.
    pub fn instance_count(&self, model_id: &str) -> usize {
        self.instances.get(model_id).map(|v| v.len()).unwrap_or(0)
    }

    /// True when every instance for the model has active > 0 (or none exist).
    pub fn all_busy(&self, model_id: &str) -> bool {
        match self.instances.get(model_id) {
            None => true,
            Some(v) if v.is_empty() => true,
            Some(v) => v.iter().all(|i| i.active.load(Ordering::Relaxed) > 0),
        }
    }

    /// Returns (model_id, pid) for every instance whose process has died.
    pub fn dead_instances(&self) -> Vec<(String, i32)> {
        let mut dead = Vec::new();
        for (model_id, insts) in &self.instances {
            for inst in insts {
                if !is_process_alive(inst.pid) {
                    dead.push((model_id.clone(), inst.pid));
                }
            }
        }
        dead
    }

    /// Returns all pids for instances of `model_id` (used by stop_model_inner).
    pub fn pids_for_model(&self, model_id: &str) -> Vec<i32> {
        self.instances.get(model_id)
            .map(|v| v.iter().map(|i| i.pid).collect())
            .unwrap_or_default()
    }

    pub fn is_alive(&self, pid: i32) -> bool {
        is_process_alive(pid)
    }

    pub async fn stop(&mut self, pid: i32) {
        let Some(rc) = self.running.remove(&pid) else {
            warn!(pid, "stopping process not in running table; skipping signal");
            return;
        };
        self.remove_instance(&rc.model_id, pid);

        let _ = kill_process_group(pid, nix::sys::signal::Signal::SIGTERM);
        tokio::time::sleep(Duration::from_millis(500)).await;
        if is_process_alive(pid) {
            warn!(pid, "process didn't exit after SIGTERM, sending SIGKILL");
            let _ = kill_process_group(pid, nix::sys::signal::Signal::SIGKILL);
        }
        info!(pid, port = rc.port, model = rc.model_id, "inference server stopped");
    }

    pub fn forget(&mut self, pid: i32) {
        let model_id = self.running.remove(&pid).map(|rc| rc.model_id).or_else(|| {
            // Not in running (e.g. test instances or already-reaped processes):
            // search the instances map by pid so we can still clean up.
            self.instances.iter()
                .find(|(_, insts)| insts.iter().any(|i| i.pid == pid))
                .map(|(id, _)| id.clone())
        });
        if let Some(model_id) = model_id {
            self.remove_instance(&model_id, pid);
        }
    }

    fn remove_instance(&mut self, model_id: &str, pid: i32) {
        if let Some(v) = self.instances.get_mut(model_id) {
            v.retain(|i| i.pid != pid);
            if v.is_empty() {
                self.instances.remove(model_id);
            }
        }
    }

    /// Directly registers an instance without a real process.
    /// Used in tests to synthesize a Running model (integration tests use
    /// pid=-1; unit tests can pass a specific pid to test dead-process detection).
    pub fn register_test_instance(&mut self, model_id: &str, pid: i32, port: u16) {
        let active = Arc::new(AtomicUsize::new(0));
        self.instances.entry(model_id.into()).or_default().push(InstanceInfo {
            pid,
            port,
            active,
        });
    }

    /// Convenience wrapper for integration tests (pid is irrelevant there).
    pub fn register_port(&mut self, model_id: &str, port: u16) {
        self.register_test_instance(model_id, -1, port);
    }
}

impl PendingChild {
    pub async fn wait_for_health(&mut self, timeout: Duration) -> Result<u64, HealthCheckError> {
        wait_for_health_or_exit(&mut self.child, self.port, timeout).await?;
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
        match child.try_wait() {
            Ok(Some(status)) => {
                return Err(HealthCheckError::ChildExited(status.code(), status.to_string()));
            }
            Ok(None) => {}
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

fn find_free_port() -> Option<u16> {
    std::net::TcpListener::bind("127.0.0.1:0")
        .ok()
        .and_then(|l| l.local_addr().ok())
        .map(|a| a.port())
}

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

    #[test]
    fn parses_standard_kv_cache_line() {
        let line = "llama_kv_cache_init: size = 1024.00 MiB ( 32768 cells, 64 layers, 2/2 seqs), K (q8_0): 512.00 MiB, V (q8_0): 512.00 MiB";
        let mib = parse_kv_size_mib(line).unwrap();
        assert!((mib - 1024.0).abs() < 0.01, "got {mib}");
    }

    #[test]
    fn parses_recurrent_state_line() {
        let line = "llama_memory_recurrent_init: size =   56.25 MiB (  8192 cells, 28 layers, 2 seqs), R (f16):   28.12 MiB, S (f16):   28.12 MiB";
        let mib = parse_kv_size_mib(line).unwrap();
        assert!((mib - 56.25).abs() < 0.01, "got {mib}");
    }

    #[test]
    fn parses_swa_kv_cache_line() {
        let line = "llama_kv_cache_init: size =  128.50 MiB (  4096 cells, 32 layers, 2/2 seqs), K (q8_0):  64.25 MiB, V (q8_0):  64.25 MiB";
        assert!(parse_kv_size_mib(line).is_some());
    }

    #[test]
    fn ignores_quant_progress_lines() {
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
        assert_eq!(&args[args.len() - 3..], &["--flash-attn", "-ngl", "99"]);
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
        assert_eq!(&args[args.len() - 2..], &["--tensor-parallel-size", "2"]);
    }

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
        args.windows(2).find(|w| w[0] == flag).map(|w| w[1].as_str())
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
        let mut t = gguf_model();
        t.draft_model_id = Some("draft".into());
        let args = build_command_args(&t, Some(&draft_model()), 9001);
        assert_eq!(find_flag(&args, "-cd"), Some("16384"));
    }

    #[test]
    fn spec_decode_argv_skips_draft_device_flag_when_unset() {
        let mut d = draft_model();
        d.device = None;
        let mut t = gguf_model();
        t.draft_model_id = Some("draft".into());
        let args = build_command_args(&t, Some(&d), 9001);
        let joined = args.join(" ");
        assert!(!joined.contains("-devd"), "{joined}");
    }
}
