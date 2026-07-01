use crate::config::{ModelConfig, WeightsFormat};
use std::collections::{HashMap, HashSet};
use std::path::PathBuf;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;
use std::time::Duration;
use tokio::sync::Notify;
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

#[derive(Debug, Clone, Copy, Default)]
pub struct ModelRuntime {
    pub instances: usize,
    pub pending: usize,
    pub active: usize,
}

/// RAII handle returned by `ensure_loaded`. Holds the port to forward to
/// and keeps the instance's active counter incremented for the duration of
/// the request. Dropped when the response body is fully consumed or the
/// connection closes.
#[derive(Debug)]
pub struct RequestGuard {
    pub port: u16,
    active: Arc<AtomicUsize>,
    request_done: Arc<Notify>,
}

impl Drop for RequestGuard {
    fn drop(&mut self) {
        self.active.fetch_sub(1, Ordering::Relaxed);
        self.request_done.notify_waiters();
    }
}

/// Owns running inference-server processes. Single instance per `Orchestrator`.
pub struct ProcessManager {
    /// pid → child handle (kept for kill_on_drop).
    running: HashMap<i32, RunningChild>,
    /// model_id → pool of live instances.
    instances: HashMap<String, Vec<InstanceInfo>>,
    /// Ports handed to spawned-but-not-yet-registered children.
    reserved_ports: HashSet<u16>,
    /// model_id → spawned-but-not-yet-healthy process count.
    pending_instances: HashMap<String, usize>,
    request_done: Arc<Notify>,
    backend_port_range: Option<(u16, u16)>,
}

impl Default for ProcessManager {
    fn default() -> Self {
        Self {
            running: HashMap::new(),
            instances: HashMap::new(),
            reserved_ports: HashSet::new(),
            pending_instances: HashMap::new(),
            request_done: Arc::new(Notify::new()),
            backend_port_range: parse_port_range_env("INFERENCE_ROUTER_BACKEND_PORT_RANGE"),
        }
    }
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
        &mut self,
        model: &ModelConfig,
        draft: Option<&ModelConfig>,
    ) -> Result<PendingChild, SpawnError> {
        let port = find_free_port(&self.reserved_ports, self.backend_port_range)
            .ok_or(SpawnError::NoFreePort)?;
        self.reserved_ports.insert(port);
        *self.pending_instances.entry(model.id.clone()).or_insert(0) += 1;

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
            .map_err(|e| {
                self.release_pending(&model.id, port);
                SpawnError::SpawnFailed(e, model.binary.clone())
            })?;

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

        Ok(PendingChild {
            pid,
            port,
            model_id: model.id.clone(),
            child,
            kv_bytes_rx: kv_rx,
        })
    }

    /// Install a healthy child into the running table. Returns a `RequestGuard`
    /// for the caller's in-flight request (active starts at 1).
    pub fn register(&mut self, pending: PendingChild) -> RequestGuard {
        let active = Arc::new(AtomicUsize::new(1));
        let port = pending.port;
        self.release_pending(&pending.model_id, pending.port);
        self.instances
            .entry(pending.model_id.clone())
            .or_default()
            .push(InstanceInfo {
                pid: pending.pid,
                port,
                active: active.clone(),
            });
        self.running.insert(
            pending.pid,
            RunningChild {
                port,
                model_id: pending.model_id,
                child: pending.child,
            },
        );
        RequestGuard {
            port,
            active,
            request_done: self.request_done.clone(),
        }
    }

    /// Drop tracking for a pending child that failed health checks. Consuming
    /// the PendingChild lets its `kill_on_drop` child handle clean up too.
    pub fn discard_pending(&mut self, pending: PendingChild) {
        self.release_pending(&pending.model_id, pending.port);
        drop(pending);
    }

    /// Find an idle instance (active == 0), increment its counter, and return
    /// a guard. Returns `None` if no instances exist or all are busy.
    pub fn acquire_idle_instance(&self, model_id: &str) -> Option<RequestGuard> {
        for inst in self.instances.get(model_id)?.iter() {
            if inst.active.load(Ordering::Relaxed) == 0 {
                inst.active.fetch_add(1, Ordering::Relaxed);
                return Some(RequestGuard {
                    port: inst.port,
                    active: inst.active.clone(),
                    request_done: self.request_done.clone(),
                });
            }
        }
        None
    }

    /// Find any instance — idle first, then the least-busy live instance.
    /// Returns `None` only if there are no instances at all.
    pub fn acquire_any_instance(&self, model_id: &str) -> Option<RequestGuard> {
        if let Some(g) = self.acquire_idle_instance(model_id) {
            return Some(g);
        }
        let inst = self
            .instances
            .get(model_id)?
            .iter()
            .min_by_key(|inst| inst.active.load(Ordering::Relaxed))?;
        inst.active.fetch_add(1, Ordering::Relaxed);
        Some(RequestGuard {
            port: inst.port,
            active: inst.active.clone(),
            request_done: self.request_done.clone(),
        })
    }

    /// Number of live instances for a model.
    pub fn instance_count(&self, model_id: &str) -> usize {
        self.instances.get(model_id).map(|v| v.len()).unwrap_or(0)
    }

    /// Number of live plus health-checking instances for a model.
    pub fn total_instance_count(&self, model_id: &str) -> usize {
        self.instance_count(model_id) + self.pending_instances.get(model_id).copied().unwrap_or(0)
    }

    pub fn has_active_requests(&self) -> bool {
        self.instances
            .values()
            .flatten()
            .any(|i| i.active.load(Ordering::Relaxed) > 0)
    }

    pub fn request_done_notifier(&self) -> Arc<Notify> {
        self.request_done.clone()
    }

    pub fn model_runtimes(&self) -> HashMap<String, ModelRuntime> {
        let mut runtimes = HashMap::new();
        for (model_id, insts) in &self.instances {
            let entry = runtimes
                .entry(model_id.clone())
                .or_insert_with(ModelRuntime::default);
            entry.instances = insts.len();
            entry.active = insts.iter().map(|i| i.active.load(Ordering::Relaxed)).sum();
        }
        for (model_id, pending) in &self.pending_instances {
            runtimes
                .entry(model_id.clone())
                .or_insert_with(ModelRuntime::default)
                .pending = *pending;
        }
        runtimes
    }

    /// Model ids whose live instances are all idle.
    ///
    /// Eviction stops every instance for a model, so a model is safe to evict
    /// only when none of its instances are serving a request.
    pub fn idle_model_ids(&self) -> HashSet<String> {
        self.instances
            .iter()
            .filter(|(_, insts)| {
                !insts.is_empty() && insts.iter().all(|i| i.active.load(Ordering::Relaxed) == 0)
            })
            .map(|(id, _)| id.clone())
            .collect()
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
        self.instances
            .get(model_id)
            .map(|v| v.iter().map(|i| i.pid).collect())
            .unwrap_or_default()
    }

    pub async fn stop(&mut self, pid: i32) {
        let Some(rc) = self.running.remove(&pid) else {
            warn!(
                pid,
                "stopping process not in running table; skipping signal"
            );
            return;
        };
        self.remove_instance(&rc.model_id, pid);

        let _ = kill_process_group(pid, nix::sys::signal::Signal::SIGTERM);
        tokio::time::sleep(Duration::from_millis(500)).await;
        if is_process_alive(pid) {
            warn!(pid, "process didn't exit after SIGTERM, sending SIGKILL");
            let _ = kill_process_group(pid, nix::sys::signal::Signal::SIGKILL);
        }
        info!(
            pid,
            port = rc.port,
            model = rc.model_id,
            "inference server stopped"
        );
    }

    pub fn forget(&mut self, pid: i32) {
        let model_id = self.running.remove(&pid).map(|rc| rc.model_id).or_else(|| {
            // Not in running (e.g. externally registered or already-reaped processes):
            // search the instances map by pid so we can still clean up.
            self.instances
                .iter()
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

    fn release_pending(&mut self, model_id: &str, port: u16) {
        self.reserved_ports.remove(&port);
        if let Some(n) = self.pending_instances.get_mut(model_id) {
            *n = n.saturating_sub(1);
            if *n == 0 {
                self.pending_instances.remove(model_id);
            }
        }
    }

    /// Registers an already-known backend instance without spawning it here.
    #[allow(dead_code)]
    pub fn register_existing_instance(&mut self, model_id: &str, pid: i32, port: u16) {
        let active = Arc::new(AtomicUsize::new(0));
        self.instances
            .entry(model_id.into())
            .or_default()
            .push(InstanceInfo { pid, port, active });
    }

    /// Registers an external backend by port when the process id is not tracked.
    #[allow(dead_code)]
    pub fn register_existing_port(&mut self, model_id: &str, port: u16) {
        self.register_existing_instance(model_id, -1, port);
    }
}

impl PendingChild {
    pub async fn wait_for_health(&mut self, timeout: Duration) -> Result<u64, HealthCheckError> {
        wait_for_health_or_exit(&mut self.child, self.port, timeout).await?;
        let kv_bytes = tokio::time::timeout(Duration::from_millis(200), &mut self.kv_bytes_rx)
            .await
            .ok()
            .and_then(|r| r.ok())
            .unwrap_or(0);
        Ok(kv_bytes)
    }
}

pub fn build_command_args(
    model: &ModelConfig,
    draft: Option<&ModelConfig>,
    port: u16,
) -> Vec<String> {
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
            args.push(if model.flash_attn {
                "on".into()
            } else {
                "off".into()
            });
            if let Some(n) = model.n_gpu_layers {
                args.push("-ngl".into());
                args.push(n.to_string());
            }
            if let Some(ref ot) = model.override_tensor {
                args.push("--override-tensor".into());
                args.push(ot.clone());
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
            if let Some(ref ts) = model.tensor_split {
                args.push("--tensor-split".into());
                args.push(ts.clone());
            }
            if let Some(ref dev) = model.device {
                args.push("--device".into());
                args.push(dev.clone());
            }
            if let Some(ref target) = model.fit_target {
                args.push("--fit-target".into());
                args.push(target.clone());
            }
            if gguf_has_fitted_placement(model) {
                args.push("--fit".into());
                args.push("off".into());
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
            if let Some(ref path) = model.mmproj_path {
                args.push("--mmproj".into());
                args.push(path.to_string_lossy().into_owned());
            }

            if draft.is_none() && model.draft_model_id.is_none() {
                if let Some(n) = model.mtp_tokens.filter(|n| *n > 0) {
                    args.push("--spec-type".into());
                    args.push("draft-mtp".into());
                    args.push("--spec-draft-n-max".into());
                    args.push(n.to_string());
                }
            }

            if let Some(d) = draft {
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
                if let Some(k) = d.cache_type_k {
                    args.push("-ctkd".into());
                    args.push(k.as_arg().into());
                }
                if let Some(v) = d.cache_type_v {
                    args.push("-ctvd".into());
                    args.push(v.as_arg().into());
                }
                if let Some(n) = model.draft_max {
                    args.push("--spec-draft-n-max".into());
                    args.push(n.to_string());
                }
                if let Some(n) = model.draft_min {
                    args.push("--spec-draft-n-min".into());
                    args.push(n.to_string());
                }
                if let Some(p) = model.draft_p_min {
                    args.push("--spec-draft-p-min".into());
                    args.push(p.to_string());
                }
                if let Some(n) = model.ctx_checkpoints {
                    args.push("--ctx-checkpoints".into());
                    args.push(n.to_string());
                }
                if let Some(n) = model.checkpoint_min_step {
                    args.push("--checkpoint-min-step".into());
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

fn gguf_has_fitted_placement(model: &ModelConfig) -> bool {
    model.n_gpu_layers.is_some() || model.tensor_split.is_some() || model.override_tensor.is_some()
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
                return Err(HealthCheckError::ChildExited(
                    status.code(),
                    status.to_string(),
                ));
            }
            Ok(None) => {}
            Err(e) => {
                return Err(HealthCheckError::ChildExited(
                    None,
                    format!("try_wait: {e}"),
                ));
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
                debug!(
                    port,
                    status = response.status().as_u16(),
                    "health non-2xx, retrying"
                );
            }
            Err(e) => {
                debug!(port, error = %e, "health connect failed, retrying");
            }
        }
        tokio::time::sleep(Duration::from_millis(500)).await;
    }
}

fn find_free_port(reserved: &HashSet<u16>, range: Option<(u16, u16)>) -> Option<u16> {
    if let Some((start, end)) = range {
        for port in start..=end {
            if reserved.contains(&port) {
                continue;
            }
            if std::net::TcpListener::bind(("127.0.0.1", port)).is_ok() {
                return Some(port);
            }
        }
        return None;
    }

    for _ in 0..128 {
        let port = std::net::TcpListener::bind("127.0.0.1:0")
            .ok()
            .and_then(|l| l.local_addr().ok())
            .map(|a| a.port())?;
        if !reserved.contains(&port) {
            return Some(port);
        }
    }
    None
}

fn parse_port_range_env(name: &str) -> Option<(u16, u16)> {
    parse_port_range(&std::env::var(name).ok()?)
}

fn parse_port_range(raw: &str) -> Option<(u16, u16)> {
    let (a, b) = raw.split_once('-')?;
    let start = a.trim().parse::<u16>().ok()?;
    let end = b.trim().parse::<u16>().ok()?;
    (start <= end).then_some((start, end))
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
#[path = "manager_tests.rs"]
mod tests;
