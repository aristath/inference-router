//! Host-level metrics: CPU utilization, RAM usage, CPU temperature.
//!
//! All reads go through `/proc` and `/sys` — zero deps, cheap enough to
//! call every second.

use serde::Serialize;
use std::fs;
use std::path::Path;
use std::sync::Mutex;

/// Snapshot returned to API clients and the dashboard.
#[derive(Debug, Clone, Copy, Serialize)]
pub struct SystemStats {
    /// 0..=100 overall CPU busy %. 0.0 on the very first sample.
    pub cpu_pct: f32,
    /// RAM actively used (total − available).
    pub ram_used: u64,
    pub ram_total: u64,
    /// CPU temperature in °C (usually Tctl on AMD / Package id 0 on Intel).
    /// None if no sensor was discoverable.
    pub cpu_temp_c: Option<f32>,
}

/// Holds the previous CPU sample so we can compute a delta for utilization.
pub struct SystemTracker {
    prev_cpu: Mutex<Option<CpuSample>>,
    /// Cached path to the CPU temperature sensor; `None` until first resolved.
    cpu_temp_path: Mutex<Option<Option<std::path::PathBuf>>>,
}

#[derive(Debug, Clone, Copy)]
struct CpuSample {
    total: u64,
    idle: u64,
}

impl Default for SystemTracker {
    fn default() -> Self {
        Self {
            prev_cpu: Mutex::new(None),
            cpu_temp_path: Mutex::new(None),
        }
    }
}

impl SystemTracker {
    /// Reads fresh metrics. Must be called repeatedly — CPU % is derived from
    /// the delta against the previous sample.
    pub fn sample(&self) -> SystemStats {
        self.sample_from("/proc/stat", "/proc/meminfo", "/sys/class/hwmon")
    }

    pub fn sample_from(
        &self,
        stat_path: &str,
        meminfo_path: &str,
        hwmon_root: &str,
    ) -> SystemStats {
        let cpu_pct = self.sample_cpu(stat_path);
        let (ram_used, ram_total) = read_meminfo(meminfo_path);
        let cpu_temp_c = self.read_cpu_temp(hwmon_root);
        SystemStats {
            cpu_pct,
            ram_used,
            ram_total,
            cpu_temp_c,
        }
    }

    fn sample_cpu(&self, stat_path: &str) -> f32 {
        let Some(sample) = read_cpu_sample(stat_path) else {
            return 0.0;
        };
        let mut slot = self.prev_cpu.lock().unwrap();
        let pct = match *slot {
            Some(prev) => {
                let total_delta = sample.total.saturating_sub(prev.total);
                let idle_delta = sample.idle.saturating_sub(prev.idle);
                if total_delta == 0 {
                    0.0
                } else {
                    let busy = total_delta.saturating_sub(idle_delta) as f64;
                    ((busy / total_delta as f64) * 100.0) as f32
                }
            }
            None => 0.0,
        };
        *slot = Some(sample);
        pct.clamp(0.0, 100.0)
    }

    fn read_cpu_temp(&self, hwmon_root: &str) -> Option<f32> {
        // Resolve once, then cache. If resolution fails we cache Some(None).
        let mut cache = self.cpu_temp_path.lock().unwrap();
        if cache.is_none() {
            *cache = Some(resolve_cpu_temp_path(hwmon_root));
        }
        let path = cache.as_ref().unwrap().as_ref()?;
        read_temp_milli(path).map(|m| m as f32 / 1000.0)
    }
}

/// Parse the first "cpu " aggregate line from /proc/stat.
/// Format: cpu user nice system idle iowait irq softirq steal guest guest_nice
fn read_cpu_sample(path: &str) -> Option<CpuSample> {
    let content = fs::read_to_string(path).ok()?;
    let line = content.lines().next()?;
    if !line.starts_with("cpu ") {
        return None;
    }
    let parts: Vec<u64> = line
        .split_ascii_whitespace()
        .skip(1)
        .take(10)
        .filter_map(|s| s.parse::<u64>().ok())
        .collect();
    if parts.len() < 4 {
        return None;
    }
    // idle fields per Linux convention: idle (3) + iowait (4).
    let idle = parts[3] + parts.get(4).copied().unwrap_or(0);
    let total: u64 = parts.iter().sum();
    Some(CpuSample { total, idle })
}

fn read_meminfo(path: &str) -> (u64, u64) {
    let Ok(content) = fs::read_to_string(path) else {
        return (0, 0);
    };
    let mut total_kb = 0u64;
    let mut avail_kb = 0u64;
    for line in content.lines() {
        if let Some(rest) = line.strip_prefix("MemTotal:") {
            total_kb = parse_kb(rest);
        } else if let Some(rest) = line.strip_prefix("MemAvailable:") {
            avail_kb = parse_kb(rest);
        }
        if total_kb != 0 && avail_kb != 0 {
            break;
        }
    }
    let used = total_kb.saturating_sub(avail_kb) * 1024;
    let total = total_kb * 1024;
    (used, total)
}

fn parse_kb(rest: &str) -> u64 {
    rest.split_ascii_whitespace()
        .next()
        .and_then(|s| s.parse::<u64>().ok())
        .unwrap_or(0)
}

/// Walk `/sys/class/hwmon/hwmon*` looking for a CPU temperature sensor.
/// Preference order: `k10temp` (AMD) → `coretemp` (Intel) → first match with
/// a `temp1_input` under a known CPU-ish name.
fn resolve_cpu_temp_path(root: &str) -> Option<std::path::PathBuf> {
    let root = Path::new(root);
    if !root.exists() {
        return None;
    }
    let entries: Vec<_> = fs::read_dir(root).ok()?.flatten().collect();

    // First pass: preferred names.
    for pref in &["k10temp", "coretemp"] {
        for entry in &entries {
            let name_file = entry.path().join("name");
            let Ok(name) = fs::read_to_string(&name_file) else {
                continue;
            };
            if name.trim() == *pref {
                let temp1 = entry.path().join("temp1_input");
                if temp1.exists() {
                    return Some(temp1);
                }
            }
        }
    }

    // Fallback: any hwmon with a temp1_input that *looks* CPU-ish.
    for entry in &entries {
        let name_file = entry.path().join("name");
        let Ok(name) = fs::read_to_string(&name_file) else {
            continue;
        };
        let n = name.trim();
        if n.contains("cpu") || n.contains("pkg") || n.contains("zenpower") {
            let temp1 = entry.path().join("temp1_input");
            if temp1.exists() {
                return Some(temp1);
            }
        }
    }
    None
}

pub fn read_temp_milli(path: &Path) -> Option<i64> {
    fs::read_to_string(path).ok()?.trim().parse::<i64>().ok()
}

#[cfg(test)]
#[path = "stats_tests.rs"]
mod tests;
