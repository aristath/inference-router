//! Delegate GPU placement to llama.cpp's `-fit`: pick the backend's eligible
//! GPUs, build the positional `--device` list, and compute the per-device
//! `--fit-target` margin (MiB to leave free) from each GPU's safety cap.
//!
//! The router no longer predicts `-ngl`/`--tensor-split`/expert offload. It
//! hands llama.cpp the device list plus a per-device free-margin and lets `-fit`
//! pack each GPU to ground-truth buffer sizes, spilling whatever doesn't fit to
//! CPU. The only number the router still controls is the safety margin.

use crate::config::Backend;
use crate::vram::tracker::GpuInfo;

/// A `-fit`-delegated placement on one backend: an explicit `--device` list and
/// the `--fit-target` margins (MiB to leave free per device) positionally
/// aligned to it. `-fit` packs each device to `free - margin`.
#[derive(Debug, Clone, PartialEq)]
pub struct FitPlacement {
    pub backend: Backend,
    /// e.g. `Vulkan0,Vulkan1`.
    pub device: String,
    /// e.g. `652,6525`, MiB to leave free, aligned to `device`.
    pub fit_target: String,
    pub gpus_used: usize,
}

/// Plan a `-fit` placement on `backend`: take every eligible GPU (the backend's
/// tagged, non-integrated cards that still have some free VRAM) and emit the
/// device list plus the per-device `--fit-target` margin.
///
/// The margin for a GPU is `(100 - cap%) × total_vram` (in MiB), so `-fit`
/// leaves exactly that much free and the GPU lands at `cap%` of its total —
/// regardless of how much a desktop/compositor is already using. A GPU driving
/// a monitor uses `display_cap_pct` (lower) so the desktop keeps headroom.
///
/// Returns `None` only when the backend has no eligible GPU with free VRAM.
pub fn plan_fit_placement(
    backend: Backend,
    candidates: &[GpuInfo],
    gpu_cap_pct: u8,
    display_cap_pct: u8,
) -> Option<FitPlacement> {
    let mut gpus: Vec<&GpuInfo> = candidates
        .iter()
        .filter(|g| !g.integrated && g.backend_index(backend).is_some() && g.free_vram() > 0)
        .collect();
    if gpus.is_empty() {
        return None;
    }
    // Device order is cosmetic — the fit-target aligns to whatever order we list
    // — but the backend's own index keeps the emitted list deterministic.
    gpus.sort_by_key(|g| g.backend_index(backend).unwrap_or(usize::MAX));

    let device = gpus
        .iter()
        .filter_map(|g| g.backend_device_name(backend))
        .collect::<Vec<_>>()
        .join(",");
    let fit_target = gpus
        .iter()
        .map(|g| fit_target_mib(g, gpu_cap_pct, display_cap_pct).to_string())
        .collect::<Vec<_>>()
        .join(",");

    Some(FitPlacement {
        backend,
        device,
        fit_target,
        gpus_used: gpus.len(),
    })
}

/// The `--fit-target` margin for one GPU, in MiB: the VRAM `-fit` must leave
/// free so the GPU ends at its cap. `(100 - cap%) × total_vram`.
fn fit_target_mib(gpu: &GpuInfo, gpu_cap_pct: u8, display_cap_pct: u8) -> u64 {
    let cap = if gpu.display_attached {
        display_cap_pct
    } else {
        gpu_cap_pct
    }
    .clamp(1, 100) as u64;
    let margin_bytes = gpu.total_vram.saturating_mul(100 - cap) / 100;
    margin_bytes >> 20
}

#[cfg(test)]
mod tests {
    use super::*;

    fn gpu(id: &str, free: u64) -> GpuInfo {
        GpuInfo {
            id: id.into(),
            pci_bus_id: None,
            vulkan_device: None,
            vulkan_index: None,
            cuda_device: None,
            cuda_index: None,
            rocm_index: None,
            sycl_index: None,
            tags: Default::default(),
            integrated: false,
            total_vram: free + 1_000_000_000,
            used_vram: 1_000_000_000,
            busy_pct: 0,
            temp_c: None,
            display_attached: false,
        }
    }

    fn rocm_gpu(rocm_index: usize, free: u64) -> GpuInfo {
        let mut g = gpu(&format!("rocm{rocm_index}"), free);
        g.rocm_index = Some(rocm_index);
        g.tags = [Backend::Rocm].into_iter().collect();
        g
    }

    #[test]
    fn fit_emits_explicit_device_in_index_order() {
        // ROCm0 full, ROCm2 + ROCm3 free → device list names only the free cards,
        // in backend-index order, with one fit-target per device.
        let gpus = vec![
            rocm_gpu(0, 0),
            rocm_gpu(2, 32_000_000_000),
            rocm_gpu(3, 32_000_000_000),
        ];
        let p = plan_fit_placement(Backend::Rocm, &gpus, 98, 80).unwrap();
        assert_eq!(p.device, "ROCm2,ROCm3");
        assert_eq!(p.gpus_used, 2);
        assert_eq!(p.fit_target.split(',').count(), 2);
        assert!(!p.device.contains("ROCm0"), "must skip the full card");
    }

    #[test]
    fn fit_none_when_backend_has_no_devices() {
        // GPUs are Vulkan-only; a ROCm placement finds nothing.
        let gpus = vec![gpu("a", 40_000_000_000)];
        assert!(plan_fit_placement(Backend::Rocm, &gpus, 98, 80).is_none());
    }

    #[test]
    fn fit_target_is_margin_to_leave_free() {
        // A 32 GiB card at 98% leaves 2% free ≈ 640 MiB.
        let gpus = vec![rocm_gpu(0, 30_000_000_000)];
        let p = plan_fit_placement(Backend::Rocm, &gpus, 98, 80).unwrap();
        let total_mib = gpus[0].total_vram >> 20;
        let expect = total_mib * 2 / 100;
        assert_eq!(p.fit_target.parse::<u64>().unwrap(), expect);
    }

    #[test]
    fn display_gpu_gets_a_larger_margin_than_a_normal_gpu() {
        // Two equal GPUs, one driving a monitor → it must be told to leave more
        // free (20% vs 2%) so the desktop keeps headroom.
        let a = rocm_gpu(0, 30_000_000_000);
        let mut b = rocm_gpu(1, 30_000_000_000);
        b.display_attached = true;
        let p = plan_fit_placement(Backend::Rocm, &[a, b], 98, 80).unwrap();
        let parts: Vec<u64> = p.fit_target.split(',').map(|s| s.parse().unwrap()).collect();
        assert!(parts[1] > parts[0], "display GPU should leave more free: {parts:?}");
    }

    #[test]
    fn fit_skips_full_gpus_but_keeps_those_with_any_free() {
        // A card with even a sliver of free VRAM is still offered to -fit.
        let gpus = vec![rocm_gpu(0, 0), rocm_gpu(1, 1_000_000_000)];
        let p = plan_fit_placement(Backend::Rocm, &gpus, 98, 80).unwrap();
        assert_eq!(p.device, "ROCm1");
        assert_eq!(p.gpus_used, 1);
    }
}
