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
    /// True for integrated GPUs (Vulkan `PHYSICAL_DEVICE_TYPE_INTEGRATED_GPU`).
    /// Such devices are enumerated and addressable so a model can target them
    /// explicitly, but they are kept out of the automatic placement pool — they
    /// only run a model when that model's `device` names them.
    #[serde(default)]
    pub integrated: bool,
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
        append_intel_gpus(&mut gpus);
        gpus.sort_by(|a, b| a.id.cmp(&b.id));
        gpus
    }

    /// Same as `refresh` but with an injectable sysfs root for tests.
    pub fn refresh_from(&self, sysfs_root: &str) -> Vec<GpuInfo> {
        self.refresh_from_with_vulkan(sysfs_root, vulkan_devices_by_pci())
    }

    /// Core of `refresh_from` with the Vulkan device map injected so tests can
    /// exercise integrated-GPU classification without invoking `vulkaninfo`.
    fn refresh_from_with_vulkan(
        &self,
        sysfs_root: &str,
        vulkan: &HashMap<String, VulkanDevice>,
    ) -> Vec<GpuInfo> {
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
            let pci_bus_id = read_pci_bus_id(&device);
            let vk = pci_bus_id
                .as_deref()
                .and_then(|pci| vulkan.get(pci).copied());
            let integrated = vk.is_some_and(|d| d.integrated);
            // The minimum-VRAM filter exists to drop integrated/display-only
            // adapters from the pool. Integrated GPUs are kept (so a model can
            // target them explicitly) but flagged; `model_visible_gpus` keeps
            // them out of automatic placement.
            if !integrated && total < MIN_USABLE_GPU_VRAM_BYTES {
                continue;
            }
            let used = read_u64(&device.join("mem_info_vram_used")).unwrap_or(0);
            let busy_pct = read_u64(&device.join("gpu_busy_percent"))
                .map(|v| v.min(100) as u8)
                .unwrap_or(0);
            let temp_c = read_gpu_junction_temp(&device);

            let id = name.trim_start_matches("card").to_string();
            let vulkan_index = vk.map(|d| d.index);
            let vulkan_device = vulkan_index.map(|idx| format!("Vulkan{idx}"));
            gpus.push(GpuInfo {
                id,
                pci_bus_id,
                vulkan_device,
                vulkan_index,
                cuda_device: None,
                cuda_index: None,
                integrated,
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
    let vulkan = vulkan_devices_by_pci();
    for mut gpu in nvidia_smi_gpus() {
        if gpu.pci_bus_id.as_deref().is_some_and(|pci| {
            gpus.iter()
                .any(|existing| existing.pci_bus_id.as_deref() == Some(pci))
        }) {
            continue;
        }
        merge_vulkan_identity(&mut gpu, vulkan);
        gpus.push(gpu);
    }
}

/// Give an NVIDIA GPU (enumerated via `nvidia-smi`) its Vulkan identity too,
/// when the same card is also a Vulkan device. The card keeps its CUDA name for
/// CUDA-only runs, but now also carries a `VulkanN` name + index so it can join
/// a Vulkan tensor-split alongside the AMD cards — a single Vulkan llama.cpp
/// process drives the AMD RADV devices and the NVIDIA ICD together. No-op if the
/// card has no Vulkan device or already has a Vulkan index.
fn merge_vulkan_identity(gpu: &mut GpuInfo, vulkan: &HashMap<String, VulkanDevice>) {
    if gpu.vulkan_index.is_some() {
        return;
    }
    if let Some(vk) = gpu.pci_bus_id.as_deref().and_then(|pci| vulkan.get(pci)) {
        gpu.vulkan_index = Some(vk.index);
        gpu.vulkan_device = Some(format!("Vulkan{}", vk.index));
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
        integrated: false,
        total_vram: total_mib.saturating_mul(1024 * 1024),
        used_vram: used_mib.saturating_mul(1024 * 1024),
        busy_pct,
        temp_c,
    })
}

/// Intel GPUs use the `xe` driver, which (unlike amdgpu) exposes no
/// `mem_info_vram_*` sysfs. We detect xe cards via the `device/driver`
/// symlink, resolve total VRAM by PCI device id (xe exposes no total in
/// unprivileged sysfs — debugfs has it but needs root), and read *used* VRAM
/// from DRM `fdinfo` (`drm-resident-vram0`) summed across processes bound to
/// the card's PCI — the same source `intel_gpu_top` uses. Deduped against
/// existing entries by PCI bus id so a card already seen isn't double-counted.
fn append_intel_gpus(gpus: &mut Vec<GpuInfo>) {
    for gpu in intel_xe_gpus("/sys/class/drm", "/proc") {
        if gpu.pci_bus_id.as_deref().is_some_and(|pci| {
            gpus.iter()
                .any(|existing| existing.pci_bus_id.as_deref() == Some(pci))
        }) {
            continue;
        }
        gpus.push(gpu);
    }
}

/// Total VRAM (MiB) for known Intel GPUs, keyed by PCI device id. The `xe`
/// driver exposes no total-VRAM node in unprivileged sysfs, so we map it per
/// model. Add new GPUs here as we get them.
fn intel_vram_mib_for_device(device_id: u16) -> Option<u64> {
    match device_id {
        0xe211 => Some(24480), // Intel Arc Pro B60 (Battlemage G21) — 24 GB
        _ => None,
    }
}

/// Manual override (MiB) for any Intel card — escape hatch for a model not yet
/// in `intel_vram_mib_for_device`. Takes precedence over the table.
fn intel_vram_mib_override() -> Option<u64> {
    std::env::var("INFERENCE_ROUTER_INTEL_VRAM_MIB")
        .ok()
        .and_then(|v| v.trim().parse::<u64>().ok())
}

/// Reads a `0x….`-style hex id file (e.g. `device/device`) as a u16.
fn read_pci_hex_id(path: &Path) -> Option<u16> {
    let raw = std::fs::read_to_string(path).ok()?;
    let s = raw.trim();
    u16::from_str_radix(s.strip_prefix("0x").unwrap_or(s), 16).ok()
}

/// Basename of the `device/driver` symlink target (e.g. `xe`, `amdgpu`).
fn read_driver_name(device: &Path) -> Option<String> {
    std::fs::read_link(device.join("driver"))
        .ok()?
        .file_name()
        .map(|s| s.to_string_lossy().into_owned())
}

fn intel_xe_gpus(drm_root: &str, proc_root: &str) -> Vec<GpuInfo> {
    let Ok(entries) = std::fs::read_dir(drm_root) else {
        return Vec::new();
    };

    let mut gpus = Vec::new();
    for entry in entries.flatten() {
        let file_name = entry.file_name();
        let name = file_name.to_string_lossy();
        // Real cards only (`cardN`), not connector sub-nodes (`cardN-DP-1`).
        if !name.starts_with("card") || name.contains('-') {
            continue;
        }

        let device = entry.path().join("device");
        if read_driver_name(&device).as_deref() != Some("xe") {
            continue;
        }
        let Some(pci_bus_id) = read_pci_bus_id(&device) else {
            continue;
        };

        // No total-VRAM sysfs for xe — resolve by model (env override wins).
        let device_id = read_pci_hex_id(&device.join("device"));
        let Some(total_mib) =
            intel_vram_mib_override().or_else(|| device_id.and_then(intel_vram_mib_for_device))
        else {
            let id_str = device_id
                .map(|d| format!("{d:#06x}"))
                .unwrap_or_else(|| "unknown".into());
            tracing::warn!(
                card = %name,
                device_id = %id_str,
                "unknown Intel GPU VRAM size — add it to intel_vram_mib_for_device \
                 or set INFERENCE_ROUTER_INTEL_VRAM_MIB; skipping this GPU"
            );
            continue;
        };

        gpus.push(GpuInfo {
            id: name.trim_start_matches("card").to_string(),
            used_vram: intel_used_vram_bytes(proc_root, &pci_bus_id),
            pci_bus_id: Some(pci_bus_id),
            // SYCL/level-zero device selection is by ONEAPI_DEVICE_SELECTOR,
            // not a Vulkan/CUDA index, so those stay unset.
            vulkan_device: None,
            vulkan_index: None,
            cuda_device: None,
            cuda_index: None,
            integrated: false,
            total_vram: total_mib.saturating_mul(1024 * 1024),
            busy_pct: 0,
            temp_c: read_gpu_junction_temp(&device),
        });
    }
    gpus
}

/// Sum resident VRAM (`drm-resident-vram0`, KiB) across all DRM clients bound
/// to `pci`, deduped by `drm-client-id` so several fds of one client aren't
/// counted twice. Returns bytes.
fn intel_used_vram_bytes(proc_root: &str, pci: &str) -> u64 {
    let mut by_client: HashMap<u64, u64> = HashMap::new();
    let Ok(procs) = std::fs::read_dir(proc_root) else {
        return 0;
    };
    for proc in procs.flatten() {
        let Ok(fds) = std::fs::read_dir(proc.path().join("fdinfo")) else {
            continue;
        };
        for fd in fds.flatten() {
            let Ok(content) = std::fs::read_to_string(fd.path()) else {
                continue;
            };
            if let Some((client, resident_kib)) = parse_intel_fdinfo(&content, pci) {
                by_client
                    .entry(client)
                    .and_modify(|v| *v = (*v).max(resident_kib))
                    .or_insert(resident_kib);
            }
        }
    }
    by_client.values().sum::<u64>().saturating_mul(1024)
}

/// Parse a DRM `fdinfo` block. Returns `(drm-client-id, drm-resident-vram0
/// KiB)` when the block belongs to the `xe` driver and matches `pci`.
fn parse_intel_fdinfo(content: &str, pci: &str) -> Option<(u64, u64)> {
    let mut is_xe = false;
    let mut matches_pci = false;
    let mut client_id: Option<u64> = None;
    let mut resident_kib: u64 = 0;
    for line in content.lines() {
        let Some((key, val)) = line.split_once(':') else {
            continue;
        };
        let val = val.trim();
        match key.trim() {
            "drm-driver" => is_xe = val == "xe",
            "drm-pdev" => matches_pci = val == pci,
            "drm-client-id" => client_id = val.parse::<u64>().ok(),
            "drm-resident-vram0" => {
                resident_kib = val
                    .split_whitespace()
                    .next()
                    .and_then(|n| n.parse::<u64>().ok())
                    .unwrap_or(0);
            }
            _ => {}
        }
    }
    (is_xe && matches_pci).then_some((client_id?, resident_kib))
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

/// A Vulkan physical device's enumeration index plus whether it is integrated.
/// Integrated devices are mapped (so a model can target them explicitly) but
/// flagged so they stay out of the automatic placement pool.
#[derive(Debug, Clone, Copy, PartialEq)]
struct VulkanDevice {
    index: usize,
    integrated: bool,
}

fn vulkan_devices_by_pci() -> &'static HashMap<String, VulkanDevice> {
    static CACHE: OnceLock<HashMap<String, VulkanDevice>> = OnceLock::new();
    CACHE.get_or_init(discover_vulkan_devices_by_pci)
}

fn discover_vulkan_devices_by_pci() -> HashMap<String, VulkanDevice> {
    // Full `vulkaninfo` (not `--summary`): the detailed output carries a
    // `VkPhysicalDevicePCIBusInfoPropertiesEXT` block per device, which gives a
    // vendor-neutral PCI address. `--summary` only exposes `deviceUUID`, and the
    // PCI is recoverable from that UUID only for AMD's RADV driver — NVIDIA's
    // UUID is opaque, so a summary-only parse can't map NVIDIA cards.
    let Ok(output) = Command::new("vulkaninfo").output() else {
        return HashMap::new();
    };
    if !output.status.success() {
        return HashMap::new();
    }
    parse_vulkaninfo(&String::from_utf8_lossy(&output.stdout))
}

/// Parse full `vulkaninfo` output into a PCI → `VulkanDevice` map. Reads the
/// "Device Properties and Extensions" section, pairing each `GPUN:` device's
/// `deviceType` with its `VkPhysicalDevicePCIBusInfoPropertiesEXT` PCI address.
/// CPU/virtual devices (no discrete/integrated type) are skipped. `vulkaninfo`
/// prints the PCI fields in decimal; we hex-format them to match sysfs and
/// `nvidia-smi` bus ids (e.g. bus `13` → `0d`, bus `28` → `1c`).
fn parse_vulkaninfo(output: &str) -> HashMap<String, VulkanDevice> {
    let mut out = HashMap::new();
    // Only the detailed section has the PCIBusInfo block; earlier sections
    // repeat `GPUN:` headers without PCI, so gate on the section header.
    let mut in_section = false;
    let mut index: Option<usize> = None;
    // Some(false) = discrete, Some(true) = integrated, None = other (CPU,
    // virtual) which we skip entirely.
    let mut integrated: Option<bool> = None;
    let mut domain: u32 = 0;
    let mut bus: Option<u32> = None;
    let mut device: Option<u32> = None;

    for line in output.lines().map(str::trim) {
        if line.starts_with("Device Properties and Extensions") {
            in_section = true;
            continue;
        }
        if !in_section {
            continue;
        }
        if let Some(rest) = line.strip_prefix("GPU").and_then(|s| s.strip_suffix(':')) {
            if let Ok(idx) = rest.parse::<usize>() {
                index = Some(idx);
                integrated = None;
                domain = 0;
                bus = None;
                device = None;
            }
            continue;
        }
        if let Some(value) = line
            .strip_prefix("deviceType")
            .and_then(|s| s.split('=').nth(1))
        {
            integrated = match value.trim() {
                "PHYSICAL_DEVICE_TYPE_DISCRETE_GPU" => Some(false),
                "PHYSICAL_DEVICE_TYPE_INTEGRATED_GPU" => Some(true),
                _ => None,
            };
            continue;
        }
        if let Some(v) = parse_vulkaninfo_pci_field(line, "pciDomain") {
            domain = v;
        } else if let Some(v) = parse_vulkaninfo_pci_field(line, "pciBus") {
            bus = Some(v);
        } else if let Some(v) = parse_vulkaninfo_pci_field(line, "pciDevice") {
            device = Some(v);
        } else if let Some(func) = parse_vulkaninfo_pci_field(line, "pciFunction") {
            // pciFunction is the last field of the block — emit here.
            if let (Some(idx), Some(integrated), Some(bus), Some(device)) =
                (index, integrated, bus, device)
            {
                let pci = format!("{domain:04x}:{bus:02x}:{device:02x}.{func:x}");
                if is_pci_bus_id(&pci) {
                    out.insert(pci, VulkanDevice { index: idx, integrated });
                }
            }
        }
    }

    out
}

/// Parse a `vulkaninfo` `key = <decimal>` line (e.g. `pciBus = 13`). Returns
/// `None` when the line is a different field, so chained calls can dispatch.
fn parse_vulkaninfo_pci_field(line: &str, key: &str) -> Option<u32> {
    line.strip_prefix(key)
        .filter(|rest| rest.starts_with(char::is_whitespace) || rest.starts_with('='))
        .and_then(|s| s.split('=').nth(1))
        .and_then(|s| s.trim().parse::<u32>().ok())
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
    fn parses_intel_fdinfo_resident_vram() {
        let block = "drm-driver:\txe\n\
            drm-pdev:\t0000:03:00.0\n\
            drm-client-id:\t235\n\
            drm-resident-vram0:\t17398892 KiB\n";
        assert_eq!(
            parse_intel_fdinfo(block, "0000:03:00.0"),
            Some((235, 17398892))
        );
        // Wrong PCI -> not this card.
        assert_eq!(parse_intel_fdinfo(block, "0000:0f:00.0"), None);
        // Non-xe driver -> ignored even if PCI matches.
        let amd = "drm-driver:\tamdgpu\n\
            drm-pdev:\t0000:03:00.0\n\
            drm-client-id:\t1\n\
            drm-resident-vram0:\t100 KiB\n";
        assert_eq!(parse_intel_fdinfo(amd, "0000:03:00.0"), None);
        // xe fd with no VRAM residency yet -> 0 KiB.
        let idle = "drm-driver:\txe\ndrm-pdev:\t0000:03:00.0\ndrm-client-id:\t9\n";
        assert_eq!(parse_intel_fdinfo(idle, "0000:03:00.0"), Some((9, 0)));
    }

    #[test]
    fn intel_vram_table_maps_known_models() {
        assert_eq!(intel_vram_mib_for_device(0xe211), Some(24480)); // Arc Pro B60
        assert_eq!(intel_vram_mib_for_device(0x1234), None); // unknown -> skip/override
    }

    #[test]
    fn reads_pci_hex_id_from_file() {
        let dir = tempfile::tempdir().unwrap();
        let f = dir.path().join("device");
        fs::write(&f, "0xe211\n").unwrap();
        assert_eq!(read_pci_hex_id(&f), Some(0xe211));
        fs::write(&f, "e211").unwrap();
        assert_eq!(read_pci_hex_id(&f), Some(0xe211));
        assert_eq!(read_pci_hex_id(&dir.path().join("missing")), None);
    }

    #[test]
    fn free_vram_saturates() {
        let g = GpuInfo {
            id: "0".into(),
            pci_bus_id: None,
            vulkan_device: None,
            vulkan_index: None,
            cuda_device: None,
            cuda_index: None,
            integrated: false,
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
    fn refresh_keeps_integrated_gpu_below_threshold_and_flags_it() {
        let tmp = tempfile::tempdir().unwrap();
        let root = tmp.path();

        // Real sysfs makes `cardN/device` a symlink into a PCI-named dir;
        // `read_pci_bus_id` canonicalizes it to recover the bus id. Mirror that.
        let mk_card = |card: &str, pci: &str, total: u64| {
            let pci_dir = root.join("pci").join(pci);
            fs::create_dir_all(&pci_dir).unwrap();
            fs::write(pci_dir.join("mem_info_vram_total"), total.to_string()).unwrap();
            let card_dir = root.join(card);
            fs::create_dir_all(&card_dir).unwrap();
            std::os::unix::fs::symlink(&pci_dir, card_dir.join("device")).unwrap();
        };

        // Integrated GPU: small carve-out under the discrete threshold, present
        // in the Vulkan map flagged integrated → kept.
        mk_card("card0", "0000:08:00.0", 2 * 1024 * 1024 * 1024);

        // Display-only adapter under threshold, NOT in the Vulkan map → dropped.
        let display = root.join("card1").join("device");
        fs::create_dir_all(&display).unwrap();
        fs::write(
            display.join("mem_info_vram_total"),
            (1024 * 1024 * 1024u64).to_string(),
        )
        .unwrap();

        // Discrete GPU at/above threshold → kept.
        mk_card("card2", "0000:03:00.0", MIN_USABLE_GPU_VRAM_BYTES);

        let mut vulkan = HashMap::new();
        vulkan.insert(
            "0000:03:00.0".to_string(),
            VulkanDevice {
                index: 0,
                integrated: false,
            },
        );
        vulkan.insert(
            "0000:08:00.0".to_string(),
            VulkanDevice {
                index: 1,
                integrated: true,
            },
        );

        let gpus = VRAMTracker.refresh_from_with_vulkan(root.to_str().unwrap(), &vulkan);
        // card0 (integrated, kept) and card2 (discrete) survive; card1 dropped.
        assert_eq!(gpus.len(), 2);
        let igpu = gpus.iter().find(|g| g.id == "0").unwrap();
        assert!(igpu.integrated);
        assert_eq!(igpu.vulkan_device.as_deref(), Some("Vulkan1"));
        let dgpu = gpus.iter().find(|g| g.id == "2").unwrap();
        assert!(!dgpu.integrated);
    }

    // Mirrors the layout of real `vulkaninfo` output: a noise section up top
    // that repeats `GPUN:` + `deviceType` headers without PCI (must be ignored),
    // then the "Device Properties and Extensions" section with the PCIBusInfo
    // blocks we parse. AMD bus 13 → 0d, NVIDIA bus 28 → 1c (decimal → hex), an
    // integrated card, and a CPU device that must be skipped.
    const VULKANINFO_FIXTURE: &str = r#"
Devices:
========
GPU0:
	apiVersion = 1.4.335
	deviceType = PHYSICAL_DEVICE_TYPE_DISCRETE_GPU
	deviceName = AMD Radeon AI PRO R9700 (RADV GFX1201)

Device Properties and Extensions:
=================================
GPU0:
VkPhysicalDeviceProperties:
---------------------------
	apiVersion        = 1.4.335 (4211023)
	deviceType        = PHYSICAL_DEVICE_TYPE_DISCRETE_GPU
	deviceName        = AMD Radeon AI PRO R9700 (RADV GFX1201)
VkPhysicalDevicePCIBusInfoPropertiesEXT:
	pciDomain   = 0
	pciBus      = 13
	pciDevice   = 0
	pciFunction = 0
GPU1:
VkPhysicalDeviceProperties:
---------------------------
	deviceType        = PHYSICAL_DEVICE_TYPE_DISCRETE_GPU
	deviceName        = NVIDIA GeForce RTX 3090
VkPhysicalDevicePCIBusInfoPropertiesEXT:
	pciDomain   = 0
	pciBus      = 28
	pciDevice   = 0
	pciFunction = 0
GPU2:
VkPhysicalDeviceProperties:
---------------------------
	deviceType        = PHYSICAL_DEVICE_TYPE_INTEGRATED_GPU
	deviceName        = AMD Radeon Graphics (RADV)
VkPhysicalDevicePCIBusInfoPropertiesEXT:
	pciDomain   = 0
	pciBus      = 8
	pciDevice   = 0
	pciFunction = 0
GPU3:
VkPhysicalDeviceProperties:
---------------------------
	deviceType        = PHYSICAL_DEVICE_TYPE_CPU
	deviceName        = llvmpipe (LLVM 22.1.5, 256 bits)
"#;

    #[test]
    fn parses_vulkaninfo_pci_bus_info_to_pci_order() {
        let map = parse_vulkaninfo(VULKANINFO_FIXTURE);
        // AMD discrete: decimal bus 13 → hex 0d.
        assert_eq!(
            map.get("0000:0d:00.0"),
            Some(&VulkanDevice {
                index: 0,
                integrated: false
            })
        );
        // NVIDIA discrete: decimal bus 28 → hex 1c. The whole point — NVIDIA is
        // mapped by PCI from PCIBusInfo, which the RADV-UUID parse could not do.
        assert_eq!(
            map.get("0000:1c:00.0"),
            Some(&VulkanDevice {
                index: 1,
                integrated: false
            })
        );
        // Integrated GPU is mapped (addressable) but flagged.
        assert_eq!(
            map.get("0000:08:00.0"),
            Some(&VulkanDevice {
                index: 2,
                integrated: true
            })
        );
        // CPU/virtual device has no usable PCI mapping and is skipped.
        assert_eq!(map.len(), 3);
    }

    #[test]
    fn merge_vulkan_identity_gives_nvidia_a_vulkan_slot() {
        let map = parse_vulkaninfo(VULKANINFO_FIXTURE);
        let mut nvidia = GpuInfo {
            id: "cuda0".into(),
            pci_bus_id: Some("0000:1c:00.0".into()),
            vulkan_device: None,
            vulkan_index: None,
            cuda_device: Some("CUDA0".into()),
            cuda_index: Some(0),
            integrated: false,
            total_vram: 24 * 1024 * 1024 * 1024,
            used_vram: 0,
            busy_pct: 0,
            temp_c: None,
        };
        merge_vulkan_identity(&mut nvidia, &map);
        // Keeps its CUDA identity and gains the Vulkan one.
        assert_eq!(nvidia.cuda_device.as_deref(), Some("CUDA0"));
        assert_eq!(nvidia.vulkan_index, Some(1));
        assert_eq!(nvidia.vulkan_device.as_deref(), Some("Vulkan1"));
    }

    #[test]
    fn merge_vulkan_identity_noop_without_vulkan_device() {
        let map = parse_vulkaninfo(VULKANINFO_FIXTURE);
        // A card whose PCI isn't a Vulkan device (e.g. NVIDIA Vulkan ICD absent).
        let mut gpu = GpuInfo {
            id: "cuda0".into(),
            pci_bus_id: Some("0000:99:00.0".into()),
            vulkan_device: None,
            vulkan_index: None,
            cuda_device: Some("CUDA0".into()),
            cuda_index: Some(0),
            integrated: false,
            total_vram: 24 * 1024 * 1024 * 1024,
            used_vram: 0,
            busy_pct: 0,
            temp_c: None,
        };
        merge_vulkan_identity(&mut gpu, &map);
        assert_eq!(gpu.vulkan_index, None);
        assert_eq!(gpu.vulkan_device, None);
    }
}
