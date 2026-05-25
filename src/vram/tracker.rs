use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::path::Path;
use std::process::Command;
use std::sync::OnceLock;

const MIN_USABLE_GPU_VRAM_BYTES: u64 = 8 * 1024 * 1024 * 1024;

/// One GPU's VRAM + activity + temperature state.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct GpuInfo {
    /// DRM card number, kept as the stable UI/API id for backwards
    /// compatibility with existing dashboard clients.
    pub id: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub pci_bus_id: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub vulkan_device: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub vulkan_index: Option<usize>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub cuda_device: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub cuda_index: Option<usize>,
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
        let mut gpus = self.refresh_from("/sys/class/drm");
        append_nvidia_gpus(&mut gpus);
        gpus.sort_by(|a, b| a.id.cmp(&b.id));
        gpus
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
            if total < MIN_USABLE_GPU_VRAM_BYTES {
                continue;
            }
            let used = read_u64(&device.join("mem_info_vram_used")).unwrap_or(0);
            let busy_pct = read_u64(&device.join("gpu_busy_percent"))
                .map(|v| v.min(100) as u8)
                .unwrap_or(0);
            let temp_c = read_gpu_junction_temp(&device);

            let id = name.trim_start_matches("card").to_string();
            let pci_bus_id = read_pci_bus_id(&device);
            let vulkan_index = pci_bus_id
                .as_deref()
                .and_then(|pci| vulkan_devices_by_pci().get(pci).copied());
            let vulkan_device = vulkan_index.map(|idx| format!("Vulkan{idx}"));
            gpus.push(GpuInfo {
                id,
                pci_bus_id,
                vulkan_device,
                vulkan_index,
                cuda_device: None,
                cuda_index: None,
                total_vram: total,
                used_vram: used,
                busy_pct,
                temp_c,
            });
        }

        gpus.sort_by(|a, b| a.id.cmp(&b.id));
        gpus
    }
}

fn append_nvidia_gpus(gpus: &mut Vec<GpuInfo>) {
    for gpu in nvidia_smi_gpus() {
        if gpu.pci_bus_id.as_deref().is_some_and(|pci| {
            gpus.iter()
                .any(|existing| existing.pci_bus_id.as_deref() == Some(pci))
        }) {
            continue;
        }
        gpus.push(gpu);
    }
}

fn nvidia_smi_gpus() -> Vec<GpuInfo> {
    let Ok(output) = Command::new("nvidia-smi")
        .args([
            "--query-gpu=index,pci.bus_id,memory.total,memory.used,utilization.gpu,temperature.gpu",
            "--format=csv,noheader,nounits",
        ])
        .output()
    else {
        return Vec::new();
    };
    if !output.status.success() {
        return Vec::new();
    }
    parse_nvidia_smi(&String::from_utf8_lossy(&output.stdout))
}

fn parse_nvidia_smi(output: &str) -> Vec<GpuInfo> {
    output.lines().filter_map(parse_nvidia_smi_line).collect()
}

fn parse_nvidia_smi_line(line: &str) -> Option<GpuInfo> {
    let parts = line.split(',').map(str::trim).collect::<Vec<_>>();
    if parts.len() != 6 {
        return None;
    }
    let cuda_index = parts[0].parse::<usize>().ok()?;
    let pci_bus_id = normalize_pci_bus_id(parts[1]);
    let total_mib = parts[2].parse::<u64>().ok()?;
    if total_mib.saturating_mul(1024 * 1024) < MIN_USABLE_GPU_VRAM_BYTES {
        return None;
    }
    let used_mib = parts[3].parse::<u64>().ok()?;
    let busy_pct = parts[4]
        .parse::<u64>()
        .ok()
        .map(|v| v.min(100) as u8)
        .unwrap_or(0);
    let temp_c = parts[5].parse::<f32>().ok();

    Some(GpuInfo {
        id: format!("cuda{cuda_index}"),
        pci_bus_id,
        vulkan_device: None,
        vulkan_index: None,
        cuda_device: Some(format!("CUDA{cuda_index}")),
        cuda_index: Some(cuda_index),
        total_vram: total_mib.saturating_mul(1024 * 1024),
        used_vram: used_mib.saturating_mul(1024 * 1024),
        busy_pct,
        temp_c,
    })
}

fn read_u64(path: &Path) -> Option<u64> {
    std::fs::read_to_string(path)
        .ok()?
        .trim()
        .parse::<u64>()
        .ok()
}

fn read_pci_bus_id(device: &Path) -> Option<String> {
    let path = std::fs::canonicalize(device).ok()?;
    path.components()
        .rev()
        .filter_map(|component| component.as_os_str().to_str())
        .find(|s| is_pci_bus_id(s))
        .map(str::to_string)
}

fn is_pci_bus_id(s: &str) -> bool {
    let bytes = s.as_bytes();
    bytes.len() == 12
        && bytes[4] == b':'
        && bytes[7] == b':'
        && bytes[10] == b'.'
        && bytes
            .iter()
            .enumerate()
            .all(|(idx, b)| matches!(idx, 4 | 7 | 10) || b.is_ascii_hexdigit())
}

fn normalize_pci_bus_id(raw: &str) -> Option<String> {
    let s = raw.trim().to_ascii_lowercase();
    if is_pci_bus_id(&s) {
        return Some(s);
    }

    let mut parts = s.split(':');
    let domain = parts.next()?;
    let bus = parts.next()?;
    let device_func = parts.next()?;
    if parts.next().is_some() || domain.len() < 4 {
        return None;
    }
    let domain = &domain[domain.len() - 4..];
    let normalized = format!("{domain}:{bus}:{device_func}");
    is_pci_bus_id(&normalized).then_some(normalized)
}

fn vulkan_devices_by_pci() -> &'static HashMap<String, usize> {
    static CACHE: OnceLock<HashMap<String, usize>> = OnceLock::new();
    CACHE.get_or_init(discover_vulkan_devices_by_pci)
}

fn discover_vulkan_devices_by_pci() -> HashMap<String, usize> {
    let Ok(output) = Command::new("vulkaninfo").arg("--summary").output() else {
        return HashMap::new();
    };
    if !output.status.success() {
        return HashMap::new();
    }
    parse_vulkaninfo_summary(&String::from_utf8_lossy(&output.stdout))
}

fn parse_vulkaninfo_summary(summary: &str) -> HashMap<String, usize> {
    let mut out = HashMap::new();
    let mut current_index: Option<usize> = None;
    let mut current_is_discrete_gpu = false;

    for line in summary.lines().map(str::trim) {
        if let Some(rest) = line.strip_prefix("GPU").and_then(|s| s.strip_suffix(':')) {
            current_index = rest.parse::<usize>().ok();
            current_is_discrete_gpu = false;
            continue;
        }

        if let Some(value) = line
            .strip_prefix("deviceType")
            .and_then(|s| s.split('=').nth(1))
        {
            current_is_discrete_gpu = value.trim() == "PHYSICAL_DEVICE_TYPE_DISCRETE_GPU";
            continue;
        }

        if let Some(uuid) = line
            .strip_prefix("deviceUUID")
            .and_then(|s| s.split('=').nth(1))
        {
            let Some(index) = current_index else {
                continue;
            };
            if !current_is_discrete_gpu {
                continue;
            }
            if let Some(pci) = pci_from_radv_device_uuid(uuid.trim()) {
                out.insert(pci, index);
            }
        }
    }

    out
}

fn pci_from_radv_device_uuid(uuid: &str) -> Option<String> {
    let mut parts = uuid.split('-');
    let _domain = parts.next()?;
    let bus_device = parts.next()?;
    if bus_device.len() != 4 || !bus_device.as_bytes().iter().all(u8::is_ascii_hexdigit) {
        return None;
    }
    Some(format!(
        "0000:{}:{}.0",
        &bus_device[0..2],
        &bus_device[2..4]
    ))
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
        let g = GpuInfo {
            id: "0".into(),
            pci_bus_id: None,
            vulkan_device: None,
            vulkan_index: None,
            cuda_device: None,
            cuda_index: None,
            total_vram: 100,
            used_vram: 150,
            busy_pct: 0,
            temp_c: None,
        };
        assert_eq!(g.free_vram(), 0);
    }

    #[test]
    fn refresh_reads_junction_temp() {
        let tmp = tempfile::tempdir().unwrap();
        let root = tmp.path();
        let device = root.join("card1").join("device");
        fs::create_dir_all(&device).unwrap();
        fs::write(
            device.join("mem_info_vram_total"),
            MIN_USABLE_GPU_VRAM_BYTES.to_string(),
        )
        .unwrap();

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
        fs::write(
            dev.join("mem_info_vram_total"),
            MIN_USABLE_GPU_VRAM_BYTES.to_string(),
        )
        .unwrap();
        fs::write(dev.join("gpu_busy_percent"), "250").unwrap();
        let gpus = VRAMTracker.refresh_from(root.to_str().unwrap());
        assert_eq!(gpus[0].busy_pct, 100);
    }

    #[test]
    fn parses_nvidia_smi_gpu_rows() {
        let gpus = parse_nvidia_smi("0, 00000000:1C:00.0, 24576, 31, 12, 42\n");

        assert_eq!(gpus.len(), 1);
        assert_eq!(gpus[0].id, "cuda0");
        assert_eq!(gpus[0].pci_bus_id.as_deref(), Some("0000:1c:00.0"));
        assert_eq!(gpus[0].cuda_device.as_deref(), Some("CUDA0"));
        assert_eq!(gpus[0].cuda_index, Some(0));
        assert_eq!(gpus[0].total_vram, 24576 * 1024 * 1024);
        assert_eq!(gpus[0].used_vram, 31 * 1024 * 1024);
        assert_eq!(gpus[0].busy_pct, 12);
        assert_eq!(gpus[0].temp_c, Some(42.0));
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

    #[test]
    fn refresh_skips_cards_below_minimum_usable_vram() {
        let tmp = tempfile::tempdir().unwrap();
        let root = tmp.path();

        let igpu = root.join("card0").join("device");
        fs::create_dir_all(&igpu).unwrap();
        fs::write(
            igpu.join("mem_info_vram_total"),
            (2 * 1024 * 1024 * 1024u64).to_string(),
        )
        .unwrap();

        let dgpu = root.join("card1").join("device");
        fs::create_dir_all(&dgpu).unwrap();
        fs::write(
            dgpu.join("mem_info_vram_total"),
            MIN_USABLE_GPU_VRAM_BYTES.to_string(),
        )
        .unwrap();

        let gpus = VRAMTracker.refresh_from(root.to_str().unwrap());
        assert_eq!(gpus.len(), 1);
        assert_eq!(gpus[0].id, "1");
    }

    #[test]
    fn parses_vulkaninfo_summary_device_uuid_to_pci_order() {
        let summary = r#"
GPU0:
    deviceType         = PHYSICAL_DEVICE_TYPE_DISCRETE_GPU
    deviceName         = AMD Radeon AI PRO R9700 (RADV GFX1201)
    deviceUUID         = 00000000-1b00-0000-0000-000000000000
GPU1:
    deviceType         = PHYSICAL_DEVICE_TYPE_DISCRETE_GPU
    deviceName         = AMD Radeon AI PRO R9700 (RADV GFX1201)
    deviceUUID         = 00000000-0300-0000-0000-000000000000
GPU2:
    deviceType         = PHYSICAL_DEVICE_TYPE_CPU
    deviceName         = llvmpipe
    deviceUUID         = 6d657361-3236-2e30-2e35-000000000000
"#;
        let map = parse_vulkaninfo_summary(summary);
        assert_eq!(map.get("0000:1b:00.0"), Some(&0));
        assert_eq!(map.get("0000:03:00.0"), Some(&1));
        assert!(!map.contains_key("0000:32:36.0"));
    }
}
