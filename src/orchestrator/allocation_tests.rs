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
    // ROCm0 full, ROCm2 + ROCm3 free → a 50 GB model needs both free cards,
    // named in backend-index order, with one fit-target per device.
    let gpus = vec![
        rocm_gpu(0, 0),
        rocm_gpu(2, 32_000_000_000),
        rocm_gpu(3, 32_000_000_000),
    ];
    let p = plan_fit_placement(Backend::Rocm, &gpus, 50_000_000_000, 98, 80).unwrap();
    assert_eq!(p.device, "ROCm2,ROCm3");
    assert_eq!(p.gpus_used, 2);
    assert_eq!(p.fit_target.split(',').count(), 2);
    assert!(!p.device.contains("ROCm0"), "must skip the full card");
}

#[test]
fn fit_none_when_backend_has_no_devices() {
    // GPUs are Vulkan-only; a ROCm placement finds nothing.
    let gpus = vec![gpu("a", 40_000_000_000)];
    assert!(plan_fit_placement(Backend::Rocm, &gpus, 10_000_000_000, 98, 80).is_none());
}

#[test]
fn fit_target_is_margin_to_leave_free() {
    // A 32 GiB card at 98% leaves 2% free ≈ 640 MiB.
    let gpus = vec![rocm_gpu(0, 30_000_000_000)];
    let p = plan_fit_placement(Backend::Rocm, &gpus, 10_000_000_000, 98, 80).unwrap();
    let total_mib = gpus[0].total_vram >> 20;
    let expect = total_mib * 2 / 100;
    assert_eq!(p.fit_target.parse::<u64>().unwrap(), expect);
}

#[test]
fn display_gpu_gets_a_larger_margin_than_a_normal_gpu() {
    // Two equal GPUs, one driving a monitor; a 45 GB model needs both → the
    // display GPU must be told to leave more free (20% vs 2%).
    let a = rocm_gpu(0, 30_000_000_000);
    let mut b = rocm_gpu(1, 30_000_000_000);
    b.display_attached = true;
    let p = plan_fit_placement(Backend::Rocm, &[a, b], 45_000_000_000, 98, 80).unwrap();
    let parts: Vec<u64> = p
        .fit_target
        .split(',')
        .map(|s| s.parse().unwrap())
        .collect();
    assert!(
        parts[1] > parts[0],
        "display GPU should leave more free: {parts:?}"
    );
}

#[test]
fn fit_skips_full_gpus_but_keeps_those_with_any_free() {
    // A card with even a sliver of free VRAM is still offered to -fit.
    let gpus = vec![rocm_gpu(0, 0), rocm_gpu(1, 1_000_000_000)];
    let p = plan_fit_placement(Backend::Rocm, &gpus, 500_000_000, 98, 80).unwrap();
    assert_eq!(p.device, "ROCm1");
    assert_eq!(p.gpus_used, 1);
}

#[test]
fn fit_uses_one_gpu_when_the_model_fits_on_one() {
    // The core regression fix: a 20 GB model with three 32 GB cards free must
    // land on ONE GPU, not be smeared across all three.
    let gpus = vec![
        rocm_gpu(0, 32_000_000_000),
        rocm_gpu(1, 32_000_000_000),
        rocm_gpu(2, 32_000_000_000),
    ];
    let p = plan_fit_placement(Backend::Rocm, &gpus, 20_000_000_000, 98, 80).unwrap();
    assert_eq!(p.gpus_used, 1, "must not split a 1-GPU model");
    assert_eq!(p.device, "ROCm0");
}

#[test]
fn fit_picks_the_most_free_gpu_not_a_busy_one() {
    // A 20 GB model: skip the nearly-full card, land on the free one.
    let gpus = vec![rocm_gpu(0, 5_000_000_000), rocm_gpu(1, 32_000_000_000)];
    let p = plan_fit_placement(Backend::Rocm, &gpus, 20_000_000_000, 98, 80).unwrap();
    assert_eq!(p.gpus_used, 1);
    assert_eq!(p.device, "ROCm1");
}

#[test]
fn fit_falls_back_to_all_gpus_when_nothing_covers_it() {
    // A huge MoE larger than all GPUs combined → use every eligible GPU and
    // let -fit spill the overflow to CPU (NOT None, NOT a single GPU).
    let gpus = vec![rocm_gpu(0, 32_000_000_000), rocm_gpu(1, 32_000_000_000)];
    let p = plan_fit_placement(Backend::Rocm, &gpus, 500_000_000_000, 98, 80).unwrap();
    assert_eq!(p.gpus_used, 2);
    assert_eq!(p.device, "ROCm0,ROCm1");
}

#[test]
fn fit_zero_estimate_uses_all_eligible_gpus() {
    // No usable estimate (gguf parse failed) → can't size, so use everything.
    let gpus = vec![rocm_gpu(0, 32_000_000_000), rocm_gpu(1, 32_000_000_000)];
    let p = plan_fit_placement(Backend::Rocm, &gpus, 0, 98, 80).unwrap();
    assert_eq!(p.gpus_used, 2);
}
