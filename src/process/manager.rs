use crate::config::{ModelConfig, WeightsFormat};
use std::collections::{HashMap, HashSet};
use std::path::PathBuf;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;
use std::time::Duration;
use tokio::sync::{watch, Notify};
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
    background: Arc<AtomicUsize>,
    /// True when this instance was placed on a GPU (so a GPU engine-reset should
    /// recycle it). Recorded at spawn from the resolved placement, which is the
    /// only reliable source — the persisted `ModelConfig.device` is `serde(skip)`.
    pub uses_gpu: bool,
    /// This instance is isolated from the discrete GPU used by media.
    pub media_independent: bool,
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
    pub(crate) gpu_permit: Option<crate::orchestrator::gpu_gate::InferencePermit>,
    pub port: u16,
    /// PID of the instance this request is routed to, so a caller that detects a
    /// stall (idle timeout) can ask the orchestrator to recycle that exact
    /// process.
    pub pid: i32,
    active: Arc<AtomicUsize>,
    background: Arc<AtomicUsize>,
    is_background: bool,
    request_done: Arc<Notify>,
}

impl RequestGuard {
    /// Keep this exact backend instance active for bounded background work that
    /// must finish immediately after the client-facing request completes.
    pub(crate) fn retain(&self) -> Self {
        self.active.fetch_add(1, Ordering::Relaxed);
        self.background.fetch_add(1, Ordering::Relaxed);
        Self {
            gpu_permit: self.gpu_permit.clone(),
            port: self.port,
            pid: self.pid,
            active: self.active.clone(),
            background: self.background.clone(),
            is_background: true,
            request_done: self.request_done.clone(),
        }
    }
}

impl Drop for RequestGuard {
    fn drop(&mut self) {
        self.active.fetch_sub(1, Ordering::Relaxed);
        if self.is_background {
            self.background.fetch_sub(1, Ordering::Relaxed);
        }
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
    /// pid → cancellable spawned-but-not-yet-healthy process metadata.
    pending: HashMap<i32, PendingInfo>,
    request_done: Arc<Notify>,
    backend_port_range: Option<(u16, u16)>,
    prefix_instances: HashMap<(String, u64), i32>,
}

impl Drop for ProcessManager {
    fn drop(&mut self) {
        // kill_on_drop only kills the API-server leader. Backends such as vLLM
        // also own GPU worker descendants in that leader's process group.
        // Signal only groups we spawned, never externally registered backends.
        for pid in self.running.keys().chain(self.pending.keys()) {
            let _ = kill_process_group(*pid, nix::sys::signal::Signal::SIGKILL);
        }
    }
}

impl Default for ProcessManager {
    fn default() -> Self {
        Self {
            running: HashMap::new(),
            instances: HashMap::new(),
            reserved_ports: HashSet::new(),
            pending_instances: HashMap::new(),
            pending: HashMap::new(),
            request_done: Arc::new(Notify::new()),
            backend_port_range: parse_port_range_env("INFERENCE_ROUTER_BACKEND_PORT_RANGE"),
            prefix_instances: HashMap::new(),
        }
    }
}

struct PendingInfo {
    port: u16,
    model_id: String,
    cancel: watch::Sender<bool>,
}

/// A freshly spawned but not-yet-healthy child.
pub struct PendingChild {
    pub pid: i32,
    pub port: u16,
    model_id: String,
    uses_gpu: bool,
    media_independent: bool,
    child: tokio::process::Child,
    kv_bytes_rx: tokio::sync::oneshot::Receiver<u64>,
    cancel_rx: watch::Receiver<bool>,
}

impl ProcessManager {
    pub fn spawn_child(
        &mut self,
        model: &ModelConfig,
        draft: Option<&ModelConfig>,
    ) -> Result<PendingChild, SpawnError> {
        model
            .validate_placement()
            .map_err(|error| SpawnError::InvalidPlacement(error.to_string()))?;
        if model.igpu_only
            && !model.device.as_deref().is_some_and(|device| {
                device
                    .strip_prefix("Vulkan")
                    .is_some_and(|index| index.parse::<usize>().is_ok())
            })
        {
            return Err(SpawnError::InvalidPlacement(
                "iGPU model has no resolved integrated Vulkan device".into(),
            ));
        }
        let port = find_free_port(&self.reserved_ports, self.backend_port_range)
            .ok_or(SpawnError::NoFreePort)?;
        self.reserved_ports.insert(port);
        *self.pending_instances.entry(model.id.clone()).or_insert(0) += 1;

        let args = build_command_args(model, draft, port);
        let uses_gpu = instance_uses_gpu(model, &args);

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

        let (cancel_tx, cancel_rx) = watch::channel(false);
        self.pending.insert(
            pid,
            PendingInfo {
                port,
                model_id: model.id.clone(),
                cancel: cancel_tx,
            },
        );

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
            uses_gpu,
            media_independent: model.media_independent(),
            child,
            kv_bytes_rx: kv_rx,
            cancel_rx,
        })
    }

    /// Install a healthy child into the running table. Returns a `RequestGuard`
    /// for the caller's in-flight request (active starts at 1).
    pub fn register(&mut self, pending: PendingChild) -> Result<RequestGuard, Box<PendingChild>> {
        if !self.pending.contains_key(&pending.pid) {
            warn!(
                pid = pending.pid,
                port = pending.port,
                model = pending.model_id,
                "refusing to register cancelled pending inference server"
            );
            return Err(Box::new(pending));
        }

        let active = Arc::new(AtomicUsize::new(1));
        let background = Arc::new(AtomicUsize::new(0));
        let port = pending.port;
        let pid = pending.pid;
        self.release_pending(&pending.model_id, pending.port);
        self.pending.remove(&pending.pid);
        self.instances
            .entry(pending.model_id.clone())
            .or_default()
            .push(InstanceInfo {
                pid,
                port,
                active: active.clone(),
                background: background.clone(),
                uses_gpu: pending.uses_gpu,
                media_independent: pending.media_independent,
            });
        self.running.insert(
            pending.pid,
            RunningChild {
                port,
                model_id: pending.model_id,
                child: pending.child,
            },
        );
        Ok(RequestGuard {
            gpu_permit: None,
            port,
            pid,
            active,
            background,
            is_background: false,
            request_done: self.request_done.clone(),
        })
    }

    /// Drop tracking for a pending child that failed health checks. Consuming
    /// the PendingChild lets its `kill_on_drop` child handle clean up too.
    pub fn discard_pending(&mut self, pending: PendingChild) {
        self.release_pending(&pending.model_id, pending.port);
        self.pending.remove(&pending.pid);
        drop(pending);
    }

    /// Find an idle instance (active == 0), increment its counter, and return
    /// a guard. Returns `None` if no instances exist or all are busy.
    pub fn acquire_idle_instance(&self, model_id: &str) -> Option<RequestGuard> {
        for inst in self.instances.get(model_id)?.iter() {
            if inst.active.load(Ordering::Relaxed) == inst.background.load(Ordering::Relaxed) {
                inst.active.fetch_add(1, Ordering::Relaxed);
                return Some(RequestGuard {
                    gpu_permit: None,
                    port: inst.port,
                    pid: inst.pid,
                    active: inst.active.clone(),
                    background: inst.background.clone(),
                    is_background: false,
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
        let inst = self.instances.get(model_id)?.iter().min_by_key(|inst| {
            inst.active
                .load(Ordering::Relaxed)
                .saturating_sub(inst.background.load(Ordering::Relaxed))
        })?;
        inst.active.fetch_add(1, Ordering::Relaxed);
        Some(RequestGuard {
            gpu_permit: None,
            port: inst.port,
            pid: inst.pid,
            active: inst.active.clone(),
            background: inst.background.clone(),
            is_background: false,
            request_done: self.request_done.clone(),
        })
    }

    /// Reuse a conversation's cache when loads tie; an idle or less-busy
    /// replica wins over affinity, and dead replicas cannot be selected.
    pub fn acquire_affine_instance(
        &mut self,
        model_id: &str,
        prefixes: &[u64],
    ) -> Option<RequestGuard> {
        let live = self.instances.get(model_id)?;
        let preferred = prefixes.iter().rev().find_map(|prefix| {
            self.prefix_instances
                .get(&(model_id.to_owned(), *prefix))
                .copied()
                .filter(|pid| live.iter().any(|inst| inst.pid == *pid))
        });
        let inst = self.instances.get(model_id)?.iter().min_by_key(|inst| {
            (
                inst.active
                    .load(Ordering::Relaxed)
                    .saturating_sub(inst.background.load(Ordering::Relaxed)),
                Some(inst.pid) != preferred,
            )
        })?;
        inst.active.fetch_add(1, Ordering::Relaxed);
        let guard = RequestGuard {
            gpu_permit: None,
            port: inst.port,
            pid: inst.pid,
            active: inst.active.clone(),
            background: inst.background.clone(),
            is_background: false,
            request_done: self.request_done.clone(),
        };
        if self.prefix_instances.len() >= 4096 {
            self.prefix_instances.clear();
        }
        for prefix in prefixes {
            self.prefix_instances
                .insert((model_id.to_owned(), *prefix), guard.pid);
        }
        Some(guard)
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

    /// `(model_id, pid)` for every live GPU-backed instance. Used by the
    /// engine-reset watchdog to recycle exactly the instances a GPU hang could
    /// have wedged, leaving CPU-only instances untouched.
    pub fn gpu_instances(&self) -> Vec<(String, i32)> {
        let mut out = Vec::new();
        for (model_id, insts) in &self.instances {
            for inst in insts {
                if inst.uses_gpu {
                    out.push((model_id.clone(), inst.pid));
                }
            }
        }
        out
    }

    /// Discrete/shared GPU instances that must drain before a media handoff.
    pub fn media_gpu_instances(&self) -> Vec<(String, i32)> {
        self.instances
            .iter()
            .flat_map(|(id, instances)| {
                instances
                    .iter()
                    .filter(|instance| instance.uses_gpu && !instance.media_independent)
                    .map(|instance| (id.clone(), instance.pid))
            })
            .collect()
    }

    /// `(model_id, pid, port)` for every idle live instance. Used by the
    /// background liveness probe; active requests rely on the proxy idle-timeout
    /// path so the probe does not interrupt legitimate long generations.
    pub fn idle_instances(&self) -> Vec<(String, i32, u16)> {
        let mut out = Vec::new();
        for (model_id, insts) in &self.instances {
            for inst in insts {
                if inst.active.load(Ordering::Relaxed) == 0 {
                    out.push((model_id.clone(), inst.pid, inst.port));
                }
            }
        }
        out
    }

    /// The model id owning `pid`, if any instance currently maps to it.
    pub fn model_id_for_pid(&self, pid: i32) -> Option<String> {
        self.instances
            .iter()
            .find(|(_, insts)| insts.iter().any(|i| i.pid == pid))
            .map(|(id, _)| id.clone())
    }

    /// Called with the manager mutex held so release cannot race a new guard.
    pub fn model_has_active_requests(&self, model_id: &str) -> bool {
        self.instances.get(model_id).is_some_and(|instances| {
            instances
                .iter()
                .any(|instance| instance.active.load(Ordering::Relaxed) > 0)
        })
    }

    /// Returns all pids for instances of `model_id` (used by stop_model_inner).
    pub fn pids_for_model(&self, model_id: &str) -> Vec<i32> {
        self.instances
            .get(model_id)
            .map(|v| v.iter().map(|i| i.pid).collect())
            .unwrap_or_default()
    }

    /// Returns all live and pending pids for a model. Stop/update/delete operate
    /// at model granularity, so health-checking children must be cancelled too.
    pub fn pids_for_model_runtime(&self, model_id: &str) -> Vec<i32> {
        let mut pids = self.pids_for_model(model_id);
        pids.extend(
            self.pending
                .iter()
                .filter(|(_, pending)| pending.model_id == model_id)
                .map(|(pid, _)| *pid),
        );
        pids
    }

    pub async fn stop(&mut self, pid: i32) {
        if let Some(pending) = self.pending.remove(&pid) {
            self.release_pending(&pending.model_id, pending.port);
            let _ = pending.cancel.send(true);
            info!(
                pid,
                port = pending.port,
                model = pending.model_id,
                "pending inference server cancellation requested"
            );
            return;
        }

        let Some(mut rc) = self.running.remove(&pid) else {
            warn!(
                pid,
                "stopping process not in running table; skipping signal"
            );
            return;
        };
        self.remove_instance(&rc.model_id, pid);

        terminate_child(&mut rc.child).await;
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
        let background = Arc::new(AtomicUsize::new(0));
        self.instances
            .entry(model_id.into())
            .or_default()
            .push(InstanceInfo {
                pid,
                port,
                active,
                background,
                uses_gpu: false,
                media_independent: false,
            });
    }

    /// Registers an external backend by port when the process id is not tracked.
    #[allow(dead_code)]
    pub fn register_existing_port(&mut self, model_id: &str, port: u16) {
        self.register_existing_instance(model_id, -1, port);
    }
}

impl PendingChild {
    pub async fn wait_for_health(&mut self, timeout: Duration) -> Result<u64, HealthCheckError> {
        wait_for_health_or_exit(&mut self.child, self.port, timeout, &mut self.cancel_rx).await?;
        let kv_bytes = tokio::time::timeout(Duration::from_millis(200), &mut self.kv_bytes_rx)
            .await
            .ok()
            .and_then(|r| r.ok())
            .unwrap_or(0);
        Ok(kv_bytes)
    }

    pub async fn terminate(&mut self) {
        terminate_child(&mut self.child).await;
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
            args.push("--min-p".into());
            args.push(model.min_p.to_string());
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
            if let Some(mode) = model.split_mode {
                args.push("--split-mode".into());
                args.push(mode.as_arg().into());
            }
            if let Some(n) = model.main_gpu {
                args.push("--main-gpu".into());
                args.push(n.to_string());
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
    if model.cpu_only && model.weights_format == WeightsFormat::Gguf {
        // Placement policy wins over legacy raw GPU flags, including draft/MTP.
        args.extend(
            [
                "--device",
                "none",
                "--gpu-layers",
                "0",
                "--fit",
                "off",
                "--no-op-offload",
                "--spec-draft-device",
                "none",
                "--spec-draft-ngl",
                "0",
            ]
            .into_iter()
            .map(str::to_owned),
        );
        if model.mmproj_path.is_some() {
            args.push("--no-mmproj-offload".into());
        }
    }
    if model.igpu_only && model.weights_format == WeightsFormat::Gguf {
        // Final placement flags cannot redirect this independent worker to the media GPU.
        args.extend(
            [
                "--device",
                model.device.as_deref().unwrap_or("none"),
                "--gpu-layers",
                "99",
                "--fit",
                "off",
                "--split-mode",
                "none",
                "--main-gpu",
                "0",
                "--tensor-split",
                "1",
                "--spec-draft-device",
                "none",
                "--spec-draft-ngl",
                "0",
            ]
            .into_iter()
            .map(str::to_owned),
        );
        if model.mmproj_path.is_some() {
            args.push("--no-mmproj-offload".into());
        }
    }
    args
}

fn gguf_has_fitted_placement(model: &ModelConfig) -> bool {
    model.n_gpu_layers.is_some() || model.tensor_split.is_some() || model.override_tensor.is_some()
}

fn instance_uses_gpu(model: &ModelConfig, args: &[String]) -> bool {
    match model.weights_format {
        WeightsFormat::Safetensors => !model.cpu_only,
        WeightsFormat::Gguf => gguf_args_use_gpu(args),
    }
}

fn gguf_args_use_gpu(args: &[String]) -> bool {
    let target_ngl = last_flag_value(args, &["-ngl", "--n-gpu-layers", "--gpu-layers"]);
    let draft_ngl = last_flag_value(args, &["-ngld"]);

    if draft_ngl.as_deref().is_some_and(layers_value_uses_gpu) {
        return true;
    }
    if target_ngl.as_deref().is_some_and(layers_value_uses_gpu) {
        return true;
    }

    let explicit_target_cpu = target_ngl
        .as_deref()
        .is_some_and(|value| !layers_value_uses_gpu(value));
    if explicit_target_cpu {
        return false;
    }

    last_flag_value(args, &["--tensor-split", "-ts"])
        .as_deref()
        .is_some_and(split_value_uses_gpu)
        || last_flag_value(args, &["--device", "-dev"])
            .as_deref()
            .is_some_and(device_value_uses_gpu)
        || last_flag_value(args, &["-devd"])
            .as_deref()
            .is_some_and(device_value_uses_gpu)
}

fn layers_value_uses_gpu(value: &str) -> bool {
    value
        .parse::<i32>()
        .map(|n| n > 0 || n == -1)
        .unwrap_or(false)
}

fn split_value_uses_gpu(value: &str) -> bool {
    value
        .split(',')
        .filter_map(|part| part.trim().parse::<f32>().ok())
        .any(|n| n > 0.0)
}

fn device_value_uses_gpu(value: &str) -> bool {
    let value = value.trim();
    !value.is_empty()
        && !value.eq_ignore_ascii_case("cpu")
        && !value.eq_ignore_ascii_case("host")
        && !value.eq_ignore_ascii_case("none")
}

fn last_flag_value(args: &[String], flags: &[&str]) -> Option<String> {
    let mut found = None;
    let mut idx = 0;
    while idx < args.len() {
        let arg = args[idx].as_str();
        if flags.contains(&arg) {
            if let Some(value) = args.get(idx + 1) {
                found = Some(value.clone());
            }
            idx += 2;
            continue;
        }
        for flag in flags {
            if let Some(value) = arg.strip_prefix(&format!("{flag}=")) {
                found = Some(value.to_string());
            }
        }
        idx += 1;
    }
    found
}

async fn wait_for_health_or_exit(
    child: &mut tokio::process::Child,
    port: u16,
    timeout: Duration,
    cancel_rx: &mut watch::Receiver<bool>,
) -> Result<(), HealthCheckError> {
    let client = reqwest::Client::builder()
        .timeout(Duration::from_secs(2))
        // This client only polls a loopback HTTP endpoint. Loading platform TLS
        // roots is unnecessary and can fail in minimal/toolbox containers,
        // which must not prevent an inference backend from starting.
        .tls_certs_only(std::iter::empty::<reqwest::tls::Certificate>())
        .build()?;
    let url = format!("http://127.0.0.1:{}/health", port);
    let start = std::time::Instant::now();

    loop {
        if *cancel_rx.borrow() {
            terminate_child(child).await;
            return Err(HealthCheckError::Cancelled);
        }

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

        let health = async {
            match client.get(&url).send().await {
                Ok(response) if response.status().is_success() => {
                    debug!(port, "health check passed");
                    return true;
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
            false
        };

        tokio::select! {
            changed = cancel_rx.changed() => {
                if changed.is_ok() && *cancel_rx.borrow() {
                    terminate_child(child).await;
                    return Err(HealthCheckError::Cancelled);
                }
            }
            result = health => {
                if result {
                    if *cancel_rx.borrow() {
                        terminate_child(child).await;
                        return Err(HealthCheckError::Cancelled);
                    }
                    return Ok(());
                }
            }
        }
        tokio::time::sleep(Duration::from_millis(500)).await;
    }
}

async fn terminate_child(child: &mut tokio::process::Child) {
    let pid = child.id().map(|p| p as i32);
    if let Some(pid) = pid {
        let _ = kill_process_group(pid, nix::sys::signal::Signal::SIGTERM);
        tokio::time::sleep(Duration::from_millis(500)).await;
        // The leader may exit before its GPU descendants. Keep its child
        // handle unreaped until the entire owned process group is signalled.
        let _ = kill_process_group(pid, nix::sys::signal::Signal::SIGKILL);
        let _ = tokio::time::timeout(Duration::from_secs(5), child.wait()).await;
    } else {
        let _ = child.kill().await;
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
    if process_state(pid) == Some('Z') {
        return false;
    }
    nix::sys::signal::kill(nix::unistd::Pid::from_raw(pid), None).is_ok()
}

fn process_state(pid: i32) -> Option<char> {
    parse_proc_stat_state(&std::fs::read_to_string(format!("/proc/{pid}/stat")).ok()?)
}

fn parse_proc_stat_state(stat: &str) -> Option<char> {
    let after_comm = stat.rsplit_once(") ")?.1;
    after_comm.chars().next()
}

fn kill_process_group(pid: i32, sig: nix::sys::signal::Signal) -> nix::Result<()> {
    nix::sys::signal::kill(nix::unistd::Pid::from_raw(-pid), sig)
}

#[derive(Debug, thiserror::Error)]
pub enum SpawnError {
    #[error("invalid placement: {0}")]
    InvalidPlacement(String),
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

    #[error("load cancelled")]
    Cancelled,
}

#[cfg(test)]
#[path = "manager_tests.rs"]
mod tests;
