//! Pick the smallest subset of GPUs that fits a model + build the backend
//! positional `--tensor-split` string llama.cpp expects.

use crate::vram::tracker::GpuInfo;

/// Headroom multiplier on top of estimated VRAM — smooths over our formula's
/// imprecision and leaves a little room for compute buffers.
const HEADROOM: f64 = 1.05;

/// Returns `Some(tensor_split_string)` if the model fits, or `None` if even
/// using every GPU doesn't cover `needed_vram`.
///
/// Strategy: sort GPUs by free VRAM descending, greedily add the most-free
/// GPU until the cumulative free ≥ `needed_vram * 1.05`. Build the
/// `--tensor-split` string in llama.cpp's Vulkan device order when that
/// mapping is known, putting `0` for GPUs not in the chosen set and
/// proportional shares for the chosen ones.
///
/// If only one GPU is chosen, returns a string with `1.0` in that slot and
/// zeros elsewhere — callers may pair that with `--split-mode none` but it's
/// not required.
pub fn plan_tensor_split(gpus: &[GpuInfo], needed_vram: u64) -> Option<String> {
    if gpus.is_empty() || needed_vram == 0 {
        return None;
    }

    let target = (needed_vram as f64 * HEADROOM) as u64;

    // Index+GPU pairs sorted by free VRAM desc.
    let mut ordered: Vec<(usize, &GpuInfo)> = gpus.iter().enumerate().collect();
    ordered.sort_by(|a, b| b.1.free_vram().cmp(&a.1.free_vram()));

    // Greedy fill.
    let mut chosen: Vec<(usize, u64)> = Vec::new();
    let mut acc: u64 = 0;
    for (idx, g) in ordered {
        if g.free_vram() == 0 {
            continue;
        }
        chosen.push((idx, g.free_vram()));
        acc = acc.saturating_add(g.free_vram());
        if acc >= target {
            break;
        }
    }

    if acc < target {
        return None;
    }

    // Emit fractions in backend device order. llama.cpp's `--tensor-split`
    // is positional: the i-th value applies to the i-th selected/offload
    // device, not Linux DRM card i.
    let mut fracs = vec![0f32; gpus.len()];
    for (idx, free) in &chosen {
        fracs[*idx] = (*free as f64 / acc as f64) as f32;
    }
    let slots = tensor_split_order(gpus);
    Some(
        slots
            .iter()
            .map(|idx| format!("{:.3}", fracs[*idx]))
            .collect::<Vec<_>>()
            .join(","),
    )
}

/// Number of GPUs assigned a non-zero share by `plan_tensor_split`. Useful
/// to decide whether to emit `--split-mode none`.
pub fn gpus_used(split: &str) -> usize {
    split
        .split(',')
        .filter(|s| s.trim().parse::<f32>().map(|v| v > 0.0).unwrap_or(false))
        .count()
}

fn tensor_split_order(gpus: &[GpuInfo]) -> Vec<usize> {
    let mut slots: Vec<usize> = (0..gpus.len()).collect();
    if gpus.iter().all(|g| g.vulkan_index.is_some()) {
        slots.sort_by_key(|idx| gpus[*idx].vulkan_index.unwrap_or(usize::MAX));
    }
    slots
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
            total_vram: free + 1_000_000_000,
            used_vram: 1_000_000_000,
            busy_pct: 0,
            temp_c: None,
        }
    }

    #[test]
    fn single_gpu_fits_alone() {
        let gpus = vec![
            gpu("0", 40_000_000_000),
            gpu("1", 40_000_000_000),
            gpu("2", 40_000_000_000),
        ];
        // 10 GB model fits on first GPU alone (biggest-free wins ties by
        // insertion order via stable sort).
        let s = plan_tensor_split(&gpus, 10_000_000_000).unwrap();
        let parts: Vec<&str> = s.split(',').collect();
        assert_eq!(parts.len(), 3);
        assert_eq!(gpus_used(&s), 1);
        // Exactly one of them is 1.000, the other two 0.000.
        let ones = parts.iter().filter(|p| **p == "1.000").count();
        let zeros = parts.iter().filter(|p| **p == "0.000").count();
        assert_eq!((ones, zeros), (1, 2));
    }

    #[test]
    fn prefers_fewer_gpus_when_two_suffice() {
        // Three GPUs, 30 GB free each. 45 GB model fits on 2.
        let gpus = vec![
            gpu("a", 30_000_000_000),
            gpu("b", 30_000_000_000),
            gpu("c", 30_000_000_000),
        ];
        let s = plan_tensor_split(&gpus, 45_000_000_000).unwrap();
        assert_eq!(gpus_used(&s), 2);
    }

    #[test]
    fn falls_back_to_all_when_needed() {
        let gpus = vec![
            gpu("a", 10_000_000_000),
            gpu("b", 10_000_000_000),
            gpu("c", 10_000_000_000),
        ];
        let s = plan_tensor_split(&gpus, 25_000_000_000).unwrap();
        assert_eq!(gpus_used(&s), 3);
    }

    #[test]
    fn returns_none_when_does_not_fit() {
        let gpus = vec![gpu("a", 5_000_000_000), gpu("b", 5_000_000_000)];
        assert!(plan_tensor_split(&gpus, 50_000_000_000).is_none());
    }

    #[test]
    fn picks_biggest_free_first() {
        // GPU 1 has the most headroom; should be the lone chosen GPU.
        let gpus = vec![
            gpu("a", 5_000_000_000),
            gpu("b", 60_000_000_000),
            gpu("c", 20_000_000_000),
        ];
        let s = plan_tensor_split(&gpus, 30_000_000_000).unwrap();
        let parts: Vec<&str> = s.split(',').collect();
        // Only index 1 has non-zero share.
        assert_eq!(parts[0], "0.000");
        assert_eq!(parts[1], "1.000");
        assert_eq!(parts[2], "0.000");
    }

    #[test]
    fn shares_reflect_free_vram_proportions() {
        // Two chosen GPUs with 40 GB and 60 GB free respectively.
        // Needed 80 GB → both picked, shares ~0.4 / 0.6.
        let gpus = vec![gpu("a", 40_000_000_000), gpu("b", 60_000_000_000)];
        let s = plan_tensor_split(&gpus, 80_000_000_000).unwrap();
        let parts: Vec<f32> = s.split(',').map(|p| p.parse().unwrap()).collect();
        assert!((parts[0] - 0.4).abs() < 0.01, "{parts:?}");
        assert!((parts[1] - 0.6).abs() < 0.01, "{parts:?}");
        let sum: f32 = parts.iter().sum();
        assert!((sum - 1.0).abs() < 0.005, "sum={sum}");
    }

    #[test]
    fn emits_split_in_vulkan_order_when_known() {
        let mut sysfs_first = gpu("card1", 40_000_000_000);
        sysfs_first.vulkan_index = Some(1);
        let mut sysfs_second = gpu("card4", 60_000_000_000);
        sysfs_second.vulkan_index = Some(0);

        let s = plan_tensor_split(&[sysfs_first, sysfs_second], 80_000_000_000).unwrap();
        assert_eq!(s, "0.600,0.400");
    }
}
