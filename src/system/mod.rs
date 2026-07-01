//! # System Monitoring
//!
//! Host system metrics collection (CPU, RAM, temperature).
//!
//! ## Key Components
//! - `stats.rs`: `SystemTracker` with delta-based CPU utilization
//!
//! ## Data Sources
//! - CPU usage: `/proc/stat` with delta calculation
//! - RAM usage: `/proc/meminfo`
//! - CPU temp: `/sys/class/hwmon/` or `/sys/class/thermal/`
//! - Zero dependencies, cheap enough to poll every second

pub mod gpu_watchdog;
pub mod memcap;
pub mod stats;
