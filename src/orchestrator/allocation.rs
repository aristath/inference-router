//! Pick the smallest subset of GPUs that fits a model + build the backend
//! positional `--tensor-split` string llama.cpp expects.

use crate::config::Backend;
use crate::vram::tracker::GpuInfo;

/// A concrete placement on one backend: an explicit `--device` list and a
/// `--tensor-split` whose values are positionally aligned to that list (verified
/// against the ROCm binary: `--tensor-split` follows `--device` order, not the
/// backend's global enumeration order). Emitting the device list explicitly is
/// what makes placement correct on a box where Vulkan and ROCm enumerate GPUs
/// in different orders.
#[derive(Debug, Clone, PartialEq)]
pub struct BackendPlacement {
    pub backend: Backend,
    /// e.g. `ROCm2,ROCm3`.
    pub device: String,
    /// e.g. `0.514,0.486`, aligned to `device`.
    pub tensor_split: String,
    pub gpus_used: usize,
}

/// Choose GPUs for `backend` from `candidates` (already filtered to GPUs the
/// backend may drive, reserved-adjusted) to fit `needed_vram`, and emit the
/// explicit `--device` + aligned `--tensor-split`.
///
/// Strategy: greedily take the most-free GPU until the cumulative free VRAM
/// covers `needed_vram`, then order the chosen devices by the backend's own
/// index so the emitted list is stable. Returns `None` if even all of the
/// backend's GPUs don't fit it.
///
/// `needed_vram` is the model's `estimated_vram`, which already carries a 10%
/// runtime-overhead margin (see [`crate::vram::estimator::VramEstimate`]), so we
/// fit it directly — no extra multiplier. That keeps the admission gate exactly
/// `free >= need`, matching the number reported to the operator.
pub fn plan_backend_split(
    backend: Backend,
    candidates: &[GpuInfo],
    needed_vram: u64,
) -> Option<BackendPlacement> {
    if candidates.is_empty() || needed_vram == 0 {
        return None;
    }
    let target = needed_vram;

    let mut by_free: Vec<&GpuInfo> = candidates
        .iter()
        .filter(|g| g.free_vram() > 0 && g.backend_index(backend).is_some())
        .collect();
    by_free.sort_by(|a, b| b.free_vram().cmp(&a.free_vram()));

    let mut chosen: Vec<&GpuInfo> = Vec::new();
    let mut acc: u64 = 0;
    for g in by_free {
        chosen.push(g);
        acc = acc.saturating_add(g.free_vram());
        if acc >= target {
            break;
        }
    }
    if acc < target {
        return None;
    }

    // List devices in the backend's index order (cosmetic — the split aligns to
    // whatever order we list — but keeps output deterministic and legible).
    chosen.sort_by_key(|g| g.backend_index(backend).unwrap_or(usize::MAX));
    let total: u64 = chosen.iter().map(|g| g.free_vram()).sum();
    let device = chosen
        .iter()
        .filter_map(|g| g.backend_device_name(backend))
        .collect::<Vec<_>>()
        .join(",");
    let tensor_split = chosen
        .iter()
        .map(|g| format!("{:.3}", g.free_vram() as f64 / total as f64))
        .collect::<Vec<_>>()
        .join(",");
    Some(BackendPlacement {
        backend,
        device,
        tensor_split,
        gpus_used: chosen.len(),
    })
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
        }
    }

    fn rocm_gpu(rocm_index: usize, free: u64) -> GpuInfo {
        let mut g = gpu(&format!("rocm{rocm_index}"), free);
        g.rocm_index = Some(rocm_index);
        g.tags = [Backend::Rocm].into_iter().collect();
        g
    }

    #[test]
    fn backend_split_emits_explicit_device_in_index_order() {
        // ROCm0 nearly full, ROCm2 + ROCm3 free. A 40 GB model must land on the
        // two free cards, named explicitly, with the split aligned to the list.
        let gpus = vec![
            rocm_gpu(0, 2_000_000_000),
            rocm_gpu(2, 32_000_000_000),
            rocm_gpu(3, 32_000_000_000),
        ];
        let p = plan_backend_split(Backend::Rocm, &gpus, 40_000_000_000).unwrap();
        assert_eq!(p.device, "ROCm2,ROCm3");
        assert_eq!(p.gpus_used, 2);
        // Equal free → ~0.5/0.5, and the split has exactly one value per device.
        assert_eq!(p.tensor_split.split(',').count(), 2);
        assert!(!p.device.contains("ROCm0"), "must skip the busy card");
    }

    #[test]
    fn backend_split_none_when_backend_has_no_devices() {
        // GPUs are Vulkan-only; a ROCm placement finds nothing.
        let gpus = vec![gpu("a", 40_000_000_000)];
        assert!(plan_backend_split(Backend::Rocm, &gpus, 10_000_000_000).is_none());
    }

    #[test]
    fn backend_split_single_gpu_when_it_fits_alone() {
        // A 10 GB model fits on one 40 GB card → single device, full share.
        let gpus = vec![rocm_gpu(0, 40_000_000_000), rocm_gpu(1, 40_000_000_000)];
        let p = plan_backend_split(Backend::Rocm, &gpus, 10_000_000_000).unwrap();
        assert_eq!(p.gpus_used, 1);
        assert_eq!(p.tensor_split, "1.000");
    }

    #[test]
    fn backend_split_fits_when_need_equals_free_no_hidden_headroom() {
        // Regression: the estimate already carries overhead, so a model whose
        // need is at-or-just-under total free VRAM must fit — no extra ×1.05
        // gate that rejects "need 136 < have 141".
        let gpus = vec![rocm_gpu(0, 70_000_000_000), rocm_gpu(1, 71_300_000_000)];
        let free: u64 = gpus.iter().map(|g| g.free_vram()).sum(); // 141.3 GB
        // need just below free would previously fail (× 1.05 → above free).
        let need = free - 5_000_000_000; // 136.3 GB
        assert!(plan_backend_split(Backend::Rocm, &gpus, need).is_some());
    }

    #[test]
    fn backend_split_returns_none_when_total_free_too_small() {
        let gpus = vec![rocm_gpu(0, 5_000_000_000), rocm_gpu(1, 5_000_000_000)];
        assert!(plan_backend_split(Backend::Rocm, &gpus, 50_000_000_000).is_none());
    }

    #[test]
    fn backend_split_shares_reflect_free_vram_proportions() {
        // 40 GB + 60 GB free, 80 GB model → both picked, shares ~0.4 / 0.6 in
        // device-index order (ROCm0 then ROCm1).
        let gpus = vec![rocm_gpu(0, 40_000_000_000), rocm_gpu(1, 60_000_000_000)];
        let p = plan_backend_split(Backend::Rocm, &gpus, 80_000_000_000).unwrap();
        assert_eq!(p.device, "ROCm0,ROCm1");
        let parts: Vec<f32> = p.tensor_split.split(',').map(|s| s.parse().unwrap()).collect();
        assert!((parts[0] - 0.4).abs() < 0.01, "{parts:?}");
        assert!((parts[1] - 0.6).abs() < 0.01, "{parts:?}");
    }
}
