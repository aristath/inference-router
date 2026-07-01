//! Hard RAM ceiling for the router **and every llama.cpp instance it spawns**.
//!
//! All spawned inference servers inherit the router service's cgroup, so a
//! single `memory.max` on that cgroup bounds their *combined* RAM. We set it to
//! `total - reserve`, guaranteeing the reserve always stays free for the OS no
//! matter how many models load or how big they are. Swap is left unbounded
//! (`MemorySwapMax=infinity`) so overflow spills to swap / mmap'd files rather
//! than OOM-killing anything.
//!
//! The cap is applied via `systemctl --user set-property` (systemd owns the
//! cgroup) and recomputed from detected RAM on every start, so it adapts to the
//! machine (64 GiB → 40 GiB, 128 GiB → 104 GiB, …). Best-effort: any failure is
//! logged and the service continues.

use std::process::Command;
use tracing::{info, warn};

const GIB: u64 = 1024 * 1024 * 1024;

/// RAM (bytes) always left for the rest of the system. Overridable with
/// `INFERENCE_ROUTER_SYSTEM_RESERVE_GIB`.
pub const DEFAULT_SYSTEM_RESERVE_BYTES: u64 = 24 * GIB;

/// cgroup memory limits derived from total RAM and the reserve.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct MemoryCap {
    /// Hard wall (`memory.max`): the cgroup can never exceed this RAM.
    pub max: u64,
    /// Soft limit (`memory.high`): the kernel reclaims aggressively past here,
    /// a couple GiB below `max`, so we evict/swap smoothly instead of hitting
    /// the hard wall and triggering an in-cgroup OOM kill.
    pub high: u64,
}

/// The configured system reserve in bytes (env override or the 24 GiB default).
pub fn configured_reserve_bytes() -> u64 {
    std::env::var("INFERENCE_ROUTER_SYSTEM_RESERVE_GIB")
        .ok()
        .and_then(|v| v.trim().parse::<u64>().ok())
        .map(|gib| gib * GIB)
        .unwrap_or(DEFAULT_SYSTEM_RESERVE_BYTES)
}

/// Compute the cgroup caps. Returns `None` when the box is too small to leave
/// the reserve plus a usable floor — we'd rather run uncapped than wedge the
/// service into a few hundred MiB.
pub fn compute_cap(total_bytes: u64, reserve_bytes: u64) -> Option<MemoryCap> {
    let max = total_bytes.checked_sub(reserve_bytes)?;
    if max < 2 * GIB {
        return None;
    }
    // Start reclaiming ~5% (capped at 2 GiB) before the hard wall.
    let buffer = (max / 20).min(2 * GIB);
    let high = max.saturating_sub(buffer).max(GIB);
    Some(MemoryCap { max, high })
}

fn read_mem_total_bytes() -> Option<u64> {
    let text = std::fs::read_to_string("/proc/meminfo").ok()?;
    parse_mem_total(&text)
}

fn parse_mem_total(meminfo: &str) -> Option<u64> {
    for line in meminfo.lines() {
        if let Some(rest) = line.strip_prefix("MemTotal:") {
            let kib: u64 = rest.split_whitespace().next()?.parse().ok()?;
            return Some(kib * 1024);
        }
    }
    None
}

/// The systemd unit this process runs in, parsed from the leaf of the cgroup v2
/// path (`…/inference-router.service`). `None` if not under a unit.
fn current_systemd_unit() -> Option<String> {
    let text = std::fs::read_to_string("/proc/self/cgroup").ok()?;
    parse_systemd_unit(&text)
}

fn parse_systemd_unit(proc_self_cgroup: &str) -> Option<String> {
    // cgroup v2 is a single `0::<path>` line.
    let path = proc_self_cgroup.lines().next()?.rsplit("::").next()?;
    let leaf = path.trim_end_matches('/').rsplit('/').next()?;
    (leaf.ends_with(".service") || leaf.ends_with(".scope")).then(|| leaf.to_string())
}

/// Apply the combined RAM cap to the router's cgroup. Best-effort; logs and
/// returns on any failure so a missing cap never stops the service.
pub fn enforce_user_memory_cap(reserve_bytes: u64) {
    let Some(total) = read_mem_total_bytes() else {
        warn!("memory cap: could not read MemTotal; running uncapped");
        return;
    };
    let Some(cap) = compute_cap(total, reserve_bytes) else {
        warn!(
            total_gib = total / GIB,
            reserve_gib = reserve_bytes / GIB,
            "memory cap: too little RAM to hold the reserve; running uncapped"
        );
        return;
    };
    let Some(unit) = current_systemd_unit() else {
        warn!("memory cap: not running under a systemd unit; set MemoryMax manually");
        return;
    };

    let result = Command::new("systemctl")
        .args([
            "--user",
            "set-property",
            &unit,
            &format!("MemoryMax={}", cap.max),
            &format!("MemoryHigh={}", cap.high),
            "MemorySwapMax=infinity",
        ])
        .output();

    match result {
        Ok(o) if o.status.success() => info!(
            unit = %unit,
            total_gib = total / GIB,
            reserve_gib = reserve_bytes / GIB,
            cap_gib = cap.max / GIB,
            "RAM cap applied to router + all spawned llama.cpp instances"
        ),
        Ok(o) => warn!(
            unit = %unit,
            stderr = %String::from_utf8_lossy(&o.stderr).trim(),
            "memory cap: systemctl set-property failed; running uncapped"
        ),
        Err(e) => warn!(error = %e, "memory cap: could not run systemctl; running uncapped"),
    }
}

#[cfg(test)]
#[path = "memcap_tests.rs"]
mod tests;
