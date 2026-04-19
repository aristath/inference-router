use serde::{Deserialize, Serialize};
use std::path::Path;

/// One GPU's VRAM + activity + temperature state.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct GpuInfo {
    pub id: String,
    pub total_vram: u64,
    pub used_vram: u64,
    /// GPU compute utilization 0..=100, from
    /// `/sys/class/drm/cardN/device/gpu_busy_percent`.
    #[serde(default)]
    pub busy_pct: u8,
    /// Junction temperature in °C (the "headline" AMD GPU temp). `None` if no
    /// hwmon sensor was found.
    #[serde(default)]
    pub temp_c: Option<f32>,
}

impl GpuInfo {
    pub fn free_vram(&self) -> u64 {
        self.total_vram.saturating_sub(self.used_vram)
    }
}

/// Reads AMD GPU VRAM from sysfs. Pure — holds no state.
#[derive(Default)]
pub struct VRAMTracker;

impl VRAMTracker {
    /// Scans `/sys/class/drm/card*` for real GPU entries (excluding connector
    /// and writeback sub-nodes like `card1-DP-1`) and returns their VRAM stats,
    /// sorted by card id.
    pub fn refresh(&self) -> Vec<GpuInfo> {
        self.refresh_from("/sys/class/drm")
    }

    /// Same as `refresh` but with an injectable sysfs root for tests.
    pub fn refresh_from(&self, sysfs_root: &str) -> Vec<GpuInfo> {
        let root = Path::new(sysfs_root);
        if !root.exists() {
            return Vec::new();
        }

        let Ok(entries) = std::fs::read_dir(root) else {
            return Vec::new();
        };

        let mut gpus = Vec::new();
        for entry in entries.flatten() {
            let file_name = entry.file_name();
            let name = file_name.to_string_lossy();

            // Real cards are `cardN` exactly. Connector/writeback sub-nodes
            // look like `cardN-DP-1`, `cardN-Writeback-1`, etc.
            if !name.starts_with("card") || name.contains('-') {
                continue;
            }

            let device = entry.path().join("device");
            let total = match read_u64(&device.join("mem_info_vram_total")) {
                Some(v) => v,
                None => continue,
            };
            let used = read_u64(&device.join("mem_info_vram_used")).unwrap_or(0);
            let busy_pct = read_u64(&device.join("gpu_busy_percent"))
                .map(|v| v.min(100) as u8)
                .unwrap_or(0);
            let temp_c = read_gpu_junction_temp(&device);

            let id = name.trim_start_matches("card").to_string();
            gpus.push(GpuInfo { id, total_vram: total, used_vram: used, busy_pct, temp_c });
        }

        gpus.sort_by(|a, b| a.id.cmp(&b.id));
        gpus
    }
}

fn read_u64(path: &Path) -> Option<u64> {
    std::fs::read_to_string(path).ok()?.trim().parse::<u64>().ok()
}

/// Reads the AMD junction temperature from `device/hwmon/hwmonN/temp2_input`
/// (milli-°C). Falls back to `temp1_input` (edge) if junction isn't present.
fn read_gpu_junction_temp(device: &Path) -> Option<f32> {
    let hwmon_root = device.join("hwmon");
    let entries = std::fs::read_dir(&hwmon_root).ok()?;
    for entry in entries.flatten() {
        let p = entry.path();
        // Prefer junction; fall back to edge.
        for leaf in &["temp2_input", "temp1_input"] {
            let f = p.join(leaf);
            if let Some(milli) = read_u64(&f) {
                return Some(milli as f32 / 1000.0);
            }
        }
    }
    None
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;

    #[test]
    fn free_vram_saturates() {
        let g = GpuInfo { id: "0".into(), total_vram: 100, used_vram: 150, busy_pct: 0, temp_c: None };
        assert_eq!(g.free_vram(), 0);
    }

    #[test]
    fn refresh_reads_junction_temp() {
        let tmp = tempfile::tempdir().unwrap();
        let root = tmp.path();
        let device = root.join("card1").join("device");
        fs::create_dir_all(&device).unwrap();
        fs::write(device.join("mem_info_vram_total"), "1").unwrap();

        let hwmon = device.join("hwmon").join("hwmon2");
        fs::create_dir_all(&hwmon).unwrap();
        fs::write(hwmon.join("temp1_input"), "41000").unwrap();
        fs::write(hwmon.join("temp2_input"), "43500").unwrap();

        let gpus = VRAMTracker.refresh_from(root.to_str().unwrap());
        assert_eq!(gpus[0].temp_c, Some(43.5));
    }

    #[test]
    fn refresh_reads_gpu_busy_percent() {
        let tmp = tempfile::tempdir().unwrap();
        let root = tmp.path();
        let card1 = root.join("card1").join("device");
        fs::create_dir_all(&card1).unwrap();
        fs::write(card1.join("mem_info_vram_total"), "32061259776").unwrap();
        fs::write(card1.join("mem_info_vram_used"), "1073741824").unwrap();
        fs::write(card1.join("gpu_busy_percent"), "42").unwrap();

        // Card without gpu_busy_percent → defaults to 0.
        let card2 = root.join("card2").join("device");
        fs::create_dir_all(&card2).unwrap();
        fs::write(card2.join("mem_info_vram_total"), "34208743424").unwrap();

        let gpus = VRAMTracker.refresh_from(root.to_str().unwrap());
        assert_eq!(gpus[0].busy_pct, 42);
        assert_eq!(gpus[1].busy_pct, 0);
    }

    #[test]
    fn busy_pct_clamped_at_100() {
        let tmp = tempfile::tempdir().unwrap();
        let root = tmp.path();
        let dev = root.join("card1").join("device");
        fs::create_dir_all(&dev).unwrap();
        fs::write(dev.join("mem_info_vram_total"), "1").unwrap();
        fs::write(dev.join("gpu_busy_percent"), "250").unwrap();
        let gpus = VRAMTracker.refresh_from(root.to_str().unwrap());
        assert_eq!(gpus[0].busy_pct, 100);
    }

    #[test]
    fn refresh_filters_connector_and_writeback_nodes() {
        let tmp = tempfile::tempdir().unwrap();
        let root = tmp.path();

        // Real card with total + used.
        let card1 = root.join("card1").join("device");
        fs::create_dir_all(&card1).unwrap();
        fs::write(card1.join("mem_info_vram_total"), "32061259776").unwrap();
        fs::write(card1.join("mem_info_vram_used"), "1073741824").unwrap();

        // Connector node — looks like a card, no device directory, must be ignored.
        fs::create_dir_all(root.join("card1-DP-1")).unwrap();
        fs::create_dir_all(root.join("card1-Writeback-1")).unwrap();

        // Second real card.
        let card2 = root.join("card2").join("device");
        fs::create_dir_all(&card2).unwrap();
        fs::write(card2.join("mem_info_vram_total"), "34208743424").unwrap();

        let gpus = VRAMTracker.refresh_from(root.to_str().unwrap());
        assert_eq!(gpus.len(), 2);
        assert_eq!(gpus[0].id, "1");
        assert_eq!(gpus[0].total_vram, 32061259776);
        assert_eq!(gpus[0].used_vram, 1073741824);
        assert_eq!(gpus[1].id, "2");
        assert_eq!(gpus[1].total_vram, 34208743424);
        assert_eq!(gpus[1].used_vram, 0);
    }

    #[test]
    fn refresh_skips_cards_without_vram_total() {
        let tmp = tempfile::tempdir().unwrap();
        let root = tmp.path();
        // Non-GPU card (integrated display w/o vram files).
        fs::create_dir_all(root.join("card0").join("device")).unwrap();

        let gpus = VRAMTracker.refresh_from(root.to_str().unwrap());
        assert!(gpus.is_empty());
    }
}
