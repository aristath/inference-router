//! Router-owned media execution. Commands are operator-registered, never supplied
//! by inference clients. The exclusive permit outlives the worker and restoration.
use super::engine::AppState;
use crate::config::JsonStore;
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::collections::BTreeMap;
use std::path::{Path, PathBuf};
use std::process::Stdio;
use std::sync::{Arc, Mutex};
use std::time::{Duration, SystemTime, UNIX_EPOCH};
use tokio::io::{AsyncRead, AsyncReadExt, AsyncWriteExt};
use tokio::process::Command;
use tokio_util::sync::CancellationToken;

#[derive(Clone, Default, Deserialize, Serialize)]
#[serde(default, deny_unknown_fields)]
pub struct MediaConfig {
    /// Model or alias restored before inference is admitted again.
    pub resident_model: String,
    pub workers: BTreeMap<String, MediaWorker>,
}

#[derive(Clone, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct MediaWorker {
    /// Executable and literal arguments. Input arrives as one JSON document on stdin.
    pub command: Vec<String>,
    pub timeout_secs: u64,
}

#[derive(Default)]
pub struct MediaController {
    config: Mutex<MediaConfig>,
    directory: Mutex<Option<PathBuf>>,
    draining: Mutex<Vec<(String, i32)>>,
    pub shutdown: CancellationToken,
}

impl MediaController {
    pub fn configure(&self, config: MediaConfig, directory: PathBuf) -> Result<(), String> {
        if !config.workers.is_empty() && config.resident_model.trim().is_empty() {
            return Err("media.json requires resident_model when workers are registered".into());
        }
        for (name, worker) in &config.workers {
            if worker.command.is_empty()
                || !Path::new(&worker.command[0]).is_absolute()
                || worker.timeout_secs == 0
            {
                return Err(format!(
                    "media worker '{name}' requires an absolute command and positive timeout_secs"
                ));
            }
        }
        std::fs::create_dir_all(&directory).map_err(|e| e.to_string())?;
        *self.config.lock().unwrap() = config;
        *self.directory.lock().unwrap() = Some(directory);
        Ok(())
    }

    pub(crate) fn marker(&self) -> Result<PathBuf, String> {
        self.directory
            .lock()
            .unwrap()
            .as_ref()
            .map(|dir| dir.join("active-worker.json"))
            .ok_or_else(|| "media execution is not configured".into())
    }

    pub fn worker(&self, name: &str) -> Result<MediaWorker, String> {
        self.config
            .lock()
            .unwrap()
            .workers
            .get(name)
            .cloned()
            .ok_or_else(|| format!("media worker '{name}' is not registered in media.json"))
    }

    pub async fn recover(self: &Arc<Self>, state: AppState) -> Result<(), String> {
        let _exclusive = state.gpu_gate.media().await;
        self.recover_locked(state).await
    }

    pub(crate) async fn recover_locked(&self, state: AppState) -> Result<(), String> {
        state
            .gpu_gate
            .set_error(Some("GPU recovery is in progress".into()));
        let draining = self.draining.lock().unwrap().clone();
        wait_inference_groups(&draining).await?;
        self.draining.lock().unwrap().clear();
        let marker = self.marker()?;
        cleanup_recorded_worker(&marker).await?;
        let resident = self.config.lock().unwrap().resident_model.clone();
        if !resident.is_empty() {
            let job = state.gpu_gate.snapshot().state.media_job;
            state.gpu_gate.set_phase("restoring", job.as_deref());
            let id = state.resolve_model_id(&resident).await;
            drop(
                state
                    .clone()
                    .ensure_loaded_admitted(&id, None)
                    .await
                    .map_err(|e| e.to_string())?,
            );
        }
        state.gpu_gate.set_error(None);
        state.gpu_gate.set_phase("llm", None);
        Ok(())
    }

    pub async fn execute(
        self: Arc<Self>,
        state: AppState,
        worker: MediaWorker,
        name: String,
        input: Value,
        cancelled: CancellationToken,
    ) -> Result<Value, String> {
        let id = format!(
            "{name}-{}-{}",
            std::process::id(),
            SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        );
        let exclusive = tokio::select! {
            biased;
            _ = self.shutdown.cancelled() => return Err("router is shutting down".into()),
            _ = cancelled.cancelled() => return Err("media request cancelled before admission".into()),
            permit = state.gpu_gate.media() => permit,
        };
        // An unexpected unwind must fail closed before releasing GPU admission.
        let mut handoff = Handoff {
            state: state.clone(),
            complete: false,
            _exclusive: exclusive,
        };
        state.gpu_gate.set_phase("unloading", Some(&id));
        state
            .record_event(
                "info",
                format!("media {name} {id}: inference drained; unloading GPU models"),
            )
            .await;
        let mut gpu_instances = state.process_manager.lock().await.gpu_instances();
        gpu_instances.extend(self.draining.lock().unwrap().clone());
        gpu_instances.sort();
        gpu_instances.dedup();
        *self.draining.lock().unwrap() = gpu_instances.clone();
        let run_result = async {
            cleanup_recorded_worker(&self.marker()?).await?;
            let mut models: Vec<_> = gpu_instances.iter().map(|(id, _)| id.clone()).collect();
            models.sort();
            models.dedup();
            for model in models {
                state.stop_model(&model).await.map_err(|e| e.to_string())?;
            }
            wait_inference_groups(&gpu_instances).await?;
            state.gpu_gate.set_phase("media", Some(&id));
            run_worker(
                &worker,
                &self.marker()?,
                input,
                state.max_body_bytes,
                &cancelled,
                &self.shutdown,
            )
            .await
        }
        .await;
        state.gpu_gate.set_phase("restoring", Some(&id));
        let recovery = if let Err(error) = wait_inference_groups(&gpu_instances).await {
            Err(error)
        } else if self.shutdown.is_cancelled() {
            Err("router is shutting down; media cleaned up".into())
        } else {
            self.recover_locked(state.clone()).await
        };
        let result = match recovery {
            Ok(()) => run_result,
            Err(error) => {
                state.gpu_gate.set_error(Some(error.clone()));
                state.gpu_gate.set_phase("recovery_failed", Some(&id));
                Err(format!("GPU restoration failed: {error}"))
            }
        };
        state
            .record_event(
                if result.is_ok() { "info" } else { "error" },
                format!(
                    "media {name} {id}: {}",
                    result
                        .as_ref()
                        .map(|_| "complete".to_string())
                        .unwrap_or_else(|e| e.clone())
                ),
            )
            .await;
        handoff.complete = true;
        result
    }
}

struct Handoff {
    state: AppState,
    complete: bool,
    _exclusive: tokio::sync::OwnedRwLockWriteGuard<()>,
}
impl Drop for Handoff {
    fn drop(&mut self) {
        if !self.complete {
            self.state.gpu_gate.set_error(Some(
                "media supervision interrupted; POST /api/gpu/recover before serving inference"
                    .into(),
            ));
            self.state.gpu_gate.set_phase("recovery_failed", None);
        }
    }
}

#[derive(Clone, Default, Deserialize, Serialize)]
#[serde(default)]
struct WorkerIdentity {
    pid: i32,
    start_ticks: u64,
    boot_id: String,
}

fn process_identity(pid: i32) -> Option<(char, i32, u64)> {
    let stat = std::fs::read_to_string(format!("/proc/{pid}/stat")).ok()?;
    let fields: Vec<_> = stat.rsplit_once(") ")?.1.split_whitespace().collect();
    Some((
        fields[0].chars().next()?,
        fields[2].parse().ok()?,
        fields[19].parse().ok()?,
    ))
}

fn group_alive(pid: i32) -> bool {
    std::fs::read_dir("/proc")
        .into_iter()
        .flatten()
        .flatten()
        .any(|entry| {
            entry
                .file_name()
                .to_str()
                .and_then(|s| s.parse::<i32>().ok())
                .and_then(process_identity)
                .is_some_and(|(state, group, _)| group == pid && state != 'Z' && state != 'X')
        })
}

fn signal_group(pid: i32, signal: nix::sys::signal::Signal) {
    let _ = nix::sys::signal::killpg(nix::unistd::Pid::from_raw(pid), signal);
}

struct WorkerGroup(i32);
impl Drop for WorkerGroup {
    fn drop(&mut self) {
        if self.0 > 0 {
            signal_group(self.0, nix::sys::signal::Signal::SIGKILL);
        }
    }
}

async fn cleanup_recorded_worker(marker: &Path) -> Result<(), String> {
    if !marker.exists() {
        return Ok(());
    }
    let store =
        JsonStore::<WorkerIdentity>::try_new(marker.to_owned()).map_err(|e| e.to_string())?;
    let saved = store.snapshot();
    let boot =
        std::fs::read_to_string("/proc/sys/kernel/random/boot_id").map_err(|e| e.to_string())?;
    if saved.boot_id == boot {
        match process_identity(saved.pid) {
            Some((_, group, ticks)) if ticks == saved.start_ticks && group == saved.pid => {
                signal_group(saved.pid, nix::sys::signal::Signal::SIGKILL);
                for _ in 0..100 {
                    if !group_alive(saved.pid) {break;}
                    tokio::time::sleep(Duration::from_millis(20)).await;
                }
                if group_alive(saved.pid) {return Err("media process group has not exited; GPU remains blocked".into());}
            }
            _ if group_alive(saved.pid) => return Err("unverified leftover media process group; refusing to kill a potentially reused PID or reload inference".into()),
            _ => {},
        }
    }
    std::fs::remove_file(marker).map_err(|e| e.to_string())
}

/// Executed by the dedicated child before the configured command can touch a GPU.
/// Persisting identity in this child closes the spawn-before-checkpoint crash window.
pub fn worker_exec(marker: &Path, command: &[String]) -> std::io::Error {
    use std::os::unix::process::CommandExt;
    let pid = std::process::id() as i32;
    let Some((_, group, ticks)) = process_identity(pid) else {
        return std::io::Error::other("cannot identify media worker");
    };
    if group != pid {
        return std::io::Error::other("media worker requires its own process group");
    }
    let saved = WorkerIdentity {
        pid,
        start_ticks: ticks,
        boot_id: match std::fs::read_to_string("/proc/sys/kernel/random/boot_id") {
            Ok(s) => s,
            Err(e) => return e,
        },
    };
    let store = JsonStore::<WorkerIdentity>::new(marker.to_owned());
    store.replace(saved);
    if let Err(e) = store.save() {
        return std::io::Error::other(e.to_string());
    }
    std::process::Command::new(&command[0])
        .args(&command[1..])
        .exec()
}

async fn capture(
    mut source: impl AsyncRead + Unpin,
    limit: usize,
    tail: bool,
) -> Result<Vec<u8>, String> {
    let mut output = Vec::new();
    let mut buf = [0; 8192];
    loop {
        let count = source.read(&mut buf).await.map_err(|e| e.to_string())?;
        if count == 0 {
            return Ok(output);
        }
        output.extend_from_slice(&buf[..count]);
        if output.len() > limit {
            if tail {
                output.drain(..output.len() - limit);
            } else {
                return Err("media result exceeds router response size limit".into());
            }
        }
    }
}

// Observe exit without reaping the leader. Its PID/process-group identity stays
// reserved until we have signalled all descendants, avoiding a PID reuse race.
async fn wait_worker_exit(pid: i32) -> Result<(), String> {
    loop {
        let exited = {
            let mut info: nix::libc::siginfo_t = unsafe { std::mem::zeroed() };
            let result = unsafe {
                nix::libc::waitid(
                    nix::libc::P_PID,
                    pid as nix::libc::id_t,
                    &mut info,
                    nix::libc::WEXITED | nix::libc::WNOHANG | nix::libc::WNOWAIT,
                )
            };
            if result != 0 {
                return Err(std::io::Error::last_os_error().to_string());
            }
            (unsafe { info.si_pid() }) == pid
        };
        if exited {
            return Ok(());
        }
        tokio::time::sleep(Duration::from_millis(10)).await;
    }
}

async fn run_worker(
    worker: &MediaWorker,
    marker: &Path,
    input: Value,
    max_bytes: usize,
    cancelled: &CancellationToken,
    shutdown: &CancellationToken,
) -> Result<Value, String> {
    let mut command = Command::new(
        std::env::var_os("INFERENCE_ROUTER_WORKER_EXECUTABLE")
            .map(PathBuf::from)
            .unwrap_or(std::env::current_exe().map_err(|e| e.to_string())?),
    );
    command
        .arg("--gpu-worker")
        .arg(marker)
        .arg("--")
        .args(&worker.command)
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .kill_on_drop(true)
        .process_group(0);
    #[cfg(target_os = "linux")]
    unsafe {
        let parent = nix::libc::getpid();
        command.pre_exec(move || {
            if nix::libc::prctl(nix::libc::PR_SET_PDEATHSIG, nix::libc::SIGKILL) != 0 {
                return Err(std::io::Error::last_os_error());
            }
            if nix::libc::getppid() != parent {
                return Err(std::io::Error::from_raw_os_error(nix::libc::ECHILD));
            }
            Ok(())
        });
    }
    let mut child = command.spawn().map_err(|e| e.to_string())?;
    let mut group = WorkerGroup(child.id().unwrap() as i32);
    let mut stdin = child.stdin.take().unwrap();
    let payload = serde_json::to_vec(&input).map_err(|e| e.to_string())?;
    let input_task = tokio::spawn(async move { stdin.write_all(&payload).await });
    let stdout = tokio::spawn(capture(child.stdout.take().unwrap(), max_bytes, false));
    let stderr = tokio::spawn(capture(child.stderr.take().unwrap(), 32768, true));
    let completion = tokio::select! {
        result=wait_worker_exit(group.0) => result,
        _=cancelled.cancelled() => Err("media request cancelled".into()),
        _=shutdown.cancelled() => Err("router is shutting down".into()),
        _=tokio::time::sleep(Duration::from_secs(worker.timeout_secs)) => Err("media worker exceeded its configured runtime".into()),
    };
    // Always terminate the entire owned group, including children outliving the leader.
    signal_group(group.0, nix::sys::signal::Signal::SIGKILL);
    let pid = group.0;
    group.0 = 0;
    let status = tokio::time::timeout(Duration::from_secs(5), child.wait())
        .await
        .map_err(|_| {
            "media leader did not exit after termination; GPU remains blocked".to_string()
        })?
        .map_err(|e| e.to_string())?;
    for _ in 0..100 {
        if !group_alive(pid) {
            break;
        }
        tokio::time::sleep(Duration::from_millis(20)).await;
    }
    if group_alive(pid) {
        return Err("media worker group is still alive; GPU remains reserved".into());
    }
    drop(group);
    input_task.abort();
    let _ = input_task.await;
    let out = stdout.await.map_err(|e| e.to_string())??;
    let err = stderr.await.map_err(|e| e.to_string())??;
    completion?;
    if !status.success() {
        return Err(format!(
            "media worker exited {status}: {}",
            String::from_utf8_lossy(&err)
        ));
    }
    serde_json::from_slice(&out).map_err(|e| format!("media worker did not return JSON: {e}"))
}

async fn wait_inference_groups(instances: &[(String, i32)]) -> Result<(), String> {
    for _ in 0..100 {
        if instances.iter().all(|(_, pid)| !group_alive(*pid)) {
            return Ok(());
        }
        tokio::time::sleep(Duration::from_millis(20)).await;
    }
    Err("inference process group has not exited; GPU remains blocked".into())
}
