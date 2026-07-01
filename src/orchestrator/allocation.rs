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
/// GPU *count* is minimized: greedily take the most-free GPU until the chosen
/// subset's allocatable VRAM covers `needed_vram`, so a model that fits on one
/// GPU runs on one GPU (no layer-split, no PCIe hops between layers). Picking
/// the most-free GPUs first also keeps the count minimal and naturally avoids
/// the display GPU and the smaller cards (less allocatable → sorted last). If no
/// subset can cover the model — e.g. a huge MoE larger than all GPUs combined —
/// we hand `-fit` every eligible GPU and let it spill the overflow to CPU.
///
/// `needed_vram` is the model's `estimated_vram`. A `0` estimate (gguf parse
/// failed) can't be reasoned about, so we fall back to all eligible GPUs.
///
/// Returns `None` only when the backend has no eligible GPU with free VRAM.
pub fn plan_fit_placement(
    backend: Backend,
    candidates: &[GpuInfo],
    needed_vram: u64,
    gpu_cap_pct: u8,
    display_cap_pct: u8,
) -> Option<FitPlacement> {
    let eligible: Vec<&GpuInfo> = candidates
        .iter()
        .filter(|g| !g.integrated && g.backend_index(backend).is_some() && g.free_vram() > 0)
        .collect();
    if eligible.is_empty() {
        return None;
    }

    let alloc = |g: &GpuInfo| g.allocatable_vram(gpu_cap_pct as u64, display_cap_pct as u64);

    // Greedy most-free-first: fewest GPUs that cover the model.
    let mut by_alloc = eligible.clone();
    by_alloc.sort_by_key(|g| std::cmp::Reverse(alloc(g)));
    let mut chosen: Vec<&GpuInfo> = Vec::new();
    let mut acc = 0u64;
    for g in &by_alloc {
        chosen.push(g);
        acc = acc.saturating_add(alloc(g));
        if needed_vram > 0 && acc >= needed_vram {
            break;
        }
    }
    // Couldn't cover it (or no usable estimate) → use every eligible GPU and let
    // -fit spill the remainder to CPU.
    if needed_vram == 0 || acc < needed_vram {
        chosen = eligible;
    }

    // Device order is cosmetic — the fit-target aligns to whatever order we list
    // — but the backend's own index keeps the emitted list deterministic.
    chosen.sort_by_key(|g| g.backend_index(backend).unwrap_or(usize::MAX));

    let device = chosen
        .iter()
        .filter_map(|g| g.backend_device_name(backend))
        .collect::<Vec<_>>()
        .join(",");
    let fit_target = chosen
        .iter()
        .map(|g| fit_target_mib(g, gpu_cap_pct, display_cap_pct).to_string())
        .collect::<Vec<_>>()
        .join(",");

    Some(FitPlacement {
        backend,
        device,
        fit_target,
        gpus_used: chosen.len(),
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
#[path = "allocation_tests.rs"]
mod tests;
