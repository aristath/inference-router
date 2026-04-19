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

impl SystemTracker {
    pub fn new() -> Self {
        Self {
            prev_cpu: Mutex::new(None),
            cpu_temp_path: Mutex::new(None),
        }
    }

    /// Reads fresh metrics. Must be called repeatedly — CPU % is derived from
    /// the delta against the previous sample.
    pub fn sample(&self) -> SystemStats {
        self.sample_from("/proc/stat", "/proc/meminfo", "/sys/class/hwmon")
    }

    pub fn sample_from(&self, stat_path: &str, meminfo_path: &str, hwmon_root: &str) -> SystemStats {
        let cpu_pct = self.sample_cpu(stat_path);
        let (ram_used, ram_total) = read_meminfo(meminfo_path);
        let cpu_temp_c = self.read_cpu_temp(hwmon_root);
        SystemStats { cpu_pct, ram_used, ram_total, cpu_temp_c }
    }

    fn sample_cpu(&self, stat_path: &str) -> f32 {
        let Some(sample) = read_cpu_sample(stat_path) else { return 0.0 };
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
    let Ok(content) = fs::read_to_string(path) else { return (0, 0) };
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
            let Ok(name) = fs::read_to_string(&name_file) else { continue };
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
        let Ok(name) = fs::read_to_string(&name_file) else { continue };
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
mod tests {
    use super::*;
    use std::fs;

    #[test]
    fn cpu_sample_first_call_is_zero() {
        let tmp = tempfile::tempdir().unwrap();
        let stat = tmp.path().join("stat");
        fs::write(&stat, "cpu  100 0 50 1000 0 0 0 0 0 0\n").unwrap();
        let tracker = SystemTracker::new();
        let s = tracker.sample_from(
            stat.to_str().unwrap(),
            "/nonexistent",
            "/nonexistent",
        );
        assert_eq!(s.cpu_pct, 0.0);
    }

    #[test]
    fn cpu_sample_computes_delta() {
        let tmp = tempfile::tempdir().unwrap();
        let stat = tmp.path().join("stat");
        let tracker = SystemTracker::new();

        // First sample: 100 busy, 1000 idle → total 1100.
        fs::write(&stat, "cpu  100 0 0 1000 0 0 0 0 0 0\n").unwrap();
        tracker.sample_from(stat.to_str().unwrap(), "/nonexistent", "/nonexistent");

        // Next sample: +100 busy user, +0 idle → 100% busy over the delta.
        fs::write(&stat, "cpu  200 0 0 1000 0 0 0 0 0 0\n").unwrap();
        let s = tracker.sample_from(stat.to_str().unwrap(), "/nonexistent", "/nonexistent");
        assert!((s.cpu_pct - 100.0).abs() < 0.5, "cpu_pct={}", s.cpu_pct);

        // Next: +100 busy, +100 idle → 50%.
        fs::write(&stat, "cpu  300 0 0 1100 0 0 0 0 0 0\n").unwrap();
        let s = tracker.sample_from(stat.to_str().unwrap(), "/nonexistent", "/nonexistent");
        assert!((s.cpu_pct - 50.0).abs() < 0.5, "cpu_pct={}", s.cpu_pct);
    }

    #[test]
    fn meminfo_parses_used_and_total() {
        let tmp = tempfile::tempdir().unwrap();
        let mem = tmp.path().join("meminfo");
        fs::write(&mem, "MemTotal:       1000 kB\nMemFree:  100 kB\nMemAvailable:   400 kB\nBuffers: 0 kB\n").unwrap();
        let tracker = SystemTracker::new();
        let s = tracker.sample_from("/nonexistent", mem.to_str().unwrap(), "/nonexistent");
        assert_eq!(s.ram_total, 1000 * 1024);
        assert_eq!(s.ram_used, (1000 - 400) * 1024);
    }

    #[test]
    fn resolve_cpu_temp_prefers_k10temp() {
        let tmp = tempfile::tempdir().unwrap();
        let root = tmp.path();

        let other = root.join("hwmon0");
        fs::create_dir_all(&other).unwrap();
        fs::write(other.join("name"), "nvme\n").unwrap();
        fs::write(other.join("temp1_input"), "30000\n").unwrap();

        let cpu = root.join("hwmon5");
        fs::create_dir_all(&cpu).unwrap();
        fs::write(cpu.join("name"), "k10temp\n").unwrap();
        fs::write(cpu.join("temp1_input"), "68000\n").unwrap();

        let got = resolve_cpu_temp_path(root.to_str().unwrap()).unwrap();
        assert_eq!(got, cpu.join("temp1_input"));
    }

    #[test]
    fn cpu_temp_is_read_and_scaled_to_celsius() {
        let tmp = tempfile::tempdir().unwrap();
        let root = tmp.path();
        let cpu = root.join("hwmon5");
        fs::create_dir_all(&cpu).unwrap();
        fs::write(cpu.join("name"), "k10temp\n").unwrap();
        fs::write(cpu.join("temp1_input"), "68500\n").unwrap();

        // Supply a meminfo + stat we won't really check.
        let stat = root.join("stat");
        fs::write(&stat, "cpu  0 0 0 0 0 0 0 0 0 0\n").unwrap();
        let meminfo = root.join("meminfo");
        fs::write(&meminfo, "MemTotal: 1 kB\nMemAvailable: 0 kB\n").unwrap();

        let tracker = SystemTracker::new();
        let s = tracker.sample_from(
            stat.to_str().unwrap(),
            meminfo.to_str().unwrap(),
            root.to_str().unwrap(),
        );
        assert_eq!(s.cpu_temp_c, Some(68.5));
    }
}
