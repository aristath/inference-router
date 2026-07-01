use super::*;
use std::fs;

#[test]
fn cpu_sample_first_call_is_zero() {
    let tmp = tempfile::tempdir().unwrap();
    let stat = tmp.path().join("stat");
    fs::write(&stat, "cpu  100 0 50 1000 0 0 0 0 0 0\n").unwrap();
    let tracker = SystemTracker::default();
    let s = tracker.sample_from(stat.to_str().unwrap(), "/nonexistent", "/nonexistent");
    assert_eq!(s.cpu_pct, 0.0);
}

#[test]
fn cpu_sample_computes_delta() {
    let tmp = tempfile::tempdir().unwrap();
    let stat = tmp.path().join("stat");
    let tracker = SystemTracker::default();

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
    fs::write(
        &mem,
        "MemTotal:       1000 kB\nMemFree:  100 kB\nMemAvailable:   400 kB\nBuffers: 0 kB\n",
    )
    .unwrap();
    let tracker = SystemTracker::default();
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

    let tracker = SystemTracker::default();
    let s = tracker.sample_from(
        stat.to_str().unwrap(),
        meminfo.to_str().unwrap(),
        root.to_str().unwrap(),
    );
    assert_eq!(s.cpu_temp_c, Some(68.5));
}
