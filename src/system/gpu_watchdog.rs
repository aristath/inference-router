//! Detection of GPU compute-engine hangs via the DRM `devcoredump` node.
//!
//! When a GPU engine hangs, the kernel DRM layer resets the engine and creates
//! a coredump under `/sys/class/drm/card<N>/device/devcoredump/` (with a `data`
//! file). We use the *appearance* of that directory as a namespace-independent,
//! privilege-light signal that a hang happened: the directory is world-listable
//! even though `data` itself is `0600 root`, so we can detect the event without
//! reading the (root-only) payload and without correlating host PIDs.
//!
//! Each distinct hang produces a directory with a fresh modification time, so we
//! key "new hang" on `(path, mtime)` rather than mere presence — that way a
//! coredump that lingers (the kernel frees it on a timeout, or root clears it)
//! doesn't re-trigger, while a genuinely new hang does.

use std::collections::HashSet;
use std::path::{Path, PathBuf};
use std::time::{SystemTime, UNIX_EPOCH};

/// A detected coredump: the directory and its modification time (used as a
/// per-event signature).
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct CoredumpSignature {
    pub path: PathBuf,
    /// Directory mtime in whole seconds since the epoch (0 if unreadable).
    pub mtime_secs: u64,
}

/// Scan `drm_root` for `card*/device/devcoredump` directories that currently
/// exist, returning a signature for each. Best-effort: unreadable entries are
/// skipped rather than erroring, so a locked-down or absent sysfs simply yields
/// an empty set.
pub fn scan_devcoredumps(drm_root: &str) -> Vec<CoredumpSignature> {
    let mut out = Vec::new();
    let Ok(entries) = std::fs::read_dir(drm_root) else {
        return out;
    };
    for entry in entries.flatten() {
        let name = entry.file_name();
        let name = name.to_string_lossy();
        // Only real cards (`card0`, `card1`, ...), not connector sub-nodes
        // (`card1-DP-1`) or render nodes (`renderD128`).
        if !name.starts_with("card") || name.contains('-') {
            continue;
        }
        let dump_dir = entry.path().join("device").join("devcoredump");
        match std::fs::metadata(&dump_dir) {
            Ok(meta) if meta.is_dir() => {
                let mtime_secs = meta
                    .modified()
                    .ok()
                    .and_then(|t| t.duration_since(UNIX_EPOCH).ok())
                    .map(|d| d.as_secs())
                    .unwrap_or(0);
                out.push(CoredumpSignature {
                    path: dump_dir,
                    mtime_secs,
                });
            }
            _ => {}
        }
    }
    out
}

/// Tracks which coredumps have already been acted on, so each hang triggers
/// recovery exactly once.
pub struct CoredumpWatcher {
    drm_root: String,
    handled: HashSet<CoredumpSignature>,
}

impl CoredumpWatcher {
    /// Build a watcher and prime it with whatever coredumps already exist, so a
    /// stale report left over from before the router started is treated as
    /// already-handled and never triggers a spurious recycle on boot.
    pub fn new(drm_root: &str) -> Self {
        let handled = scan_devcoredumps(drm_root).into_iter().collect();
        Self {
            drm_root: drm_root.to_string(),
            handled,
        }
    }

    /// Return signatures for coredumps that have appeared (or been refreshed)
    /// since the last poll, marking them handled. Empty when nothing is new.
    pub fn poll_new(&mut self) -> Vec<CoredumpSignature> {
        let current = scan_devcoredumps(&self.drm_root);
        let mut fresh = Vec::new();
        for sig in current {
            if self.handled.insert(sig.clone()) {
                fresh.push(sig);
            }
        }
        fresh
    }
}

/// Best-effort copy of a coredump's `data` payload into `capture_dir` for later
/// analysis, timestamped. Returns the saved path on success. Fails silently
/// (returns `Err`) when the payload is root-only and the router lacks
/// permission — detection does not depend on this succeeding.
pub fn capture(dump_dir: &Path, capture_dir: &str) -> std::io::Result<PathBuf> {
    let expanded = shellexpand::tilde(capture_dir).to_string();
    std::fs::create_dir_all(&expanded)?;
    let ts = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0);
    let data = dump_dir.join("data");
    let bytes = std::fs::read(&data)?;
    let out = Path::new(&expanded).join(format!("devcoredump-{ts}.bin"));
    std::fs::write(&out, bytes)?;
    Ok(out)
}
