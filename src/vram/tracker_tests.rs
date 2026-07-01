use super::*;
use std::fs;

#[test]
fn parses_rocm_showbus_to_pci_index_map() {
    let out = "======================================= PCI Bus ID =======================================\n\
            GPU[0]\t\t: PCI Bus: 0000:03:00.0\n\
            GPU[1]\t\t: PCI Bus: 0000:07:00.0\n\
            GPU[2]\t\t: PCI Bus: 0000:0A:00.0\n";
    let map = parse_rocm_showbus(out);
    // HIP index keyed by lowercase-normalised PCI (matches sysfs form).
    assert_eq!(map.get("0000:03:00.0"), Some(&0));
    assert_eq!(map.get("0000:07:00.0"), Some(&1));
    assert_eq!(map.get("0000:0a:00.0"), Some(&2));
    assert_eq!(map.len(), 3);
}

#[test]
fn sycl_indices_assigned_to_intel_gpus_in_pci_order() {
    let tmp = tempfile::tempdir().unwrap();
    let root = tmp.path();
    // An Intel card (0x8086) at 1e and an AMD card (0x1002) at 03. The real
    // device dir is named by PCI so `read_pci_bus_id` (which canonicalises
    // `card*/device`) recovers the bus id, as in actual sysfs.
    for (card, pci, vendor) in [
        ("card0", "0000:1e:00.0", "0x8086"),
        ("card1", "0000:03:00.0", "0x1002"),
    ] {
        let devdir = root.join(pci);
        fs::create_dir_all(&devdir).unwrap();
        fs::write(devdir.join("vendor"), format!("{vendor}\n")).unwrap();
        let card_dir = root.join(card);
        fs::create_dir_all(&card_dir).unwrap();
        std::os::unix::fs::symlink(&devdir, card_dir.join("device")).unwrap();
    }
    let map = discover_sycl_pci_to_index(root.to_str().unwrap());
    assert_eq!(map.get("0000:1e:00.0"), Some(&0)); // Intel GPU → SYCL0
    assert!(!map.contains_key("0000:03:00.0")); // AMD excluded from SYCL
}

#[test]
fn display_detection_maps_connected_connector_to_card_pci() {
    let tmp = tempfile::tempdir().unwrap();
    let root = tmp.path();
    // card4 → PCI 0d with a *connected* monitor; card1 → 03 disconnected.
    for (card, pci, conn, status) in [
        ("card4", "0000:0d:00.0", "card4-DP-13", "connected"),
        ("card1", "0000:03:00.0", "card1-DP-1", "disconnected"),
    ] {
        let dev = root.join(pci);
        fs::create_dir_all(&dev).unwrap();
        fs::create_dir_all(root.join(card)).unwrap();
        std::os::unix::fs::symlink(&dev, root.join(card).join("device")).unwrap();
        let c = root.join(conn);
        fs::create_dir_all(&c).unwrap();
        fs::write(c.join("status"), format!("{status}\n")).unwrap();
    }
    let pcis = display_attached_pcis(root.to_str().unwrap());
    assert!(pcis.contains("0000:0d:00.0"));
    assert!(!pcis.contains("0000:03:00.0"));
}

#[test]
fn allocatable_vram_applies_caps_and_subtracts_used() {
    let gib = 1u64 << 30;
    let mut g = GpuInfo {
        id: "x".into(),
        pci_bus_id: None,
        vulkan_device: None,
        vulkan_index: None,
        cuda_device: None,
        cuda_index: None,
        rocm_index: None,
        sycl_index: None,
        tags: BTreeSet::new(),
        integrated: false,
        total_vram: 32 * gib,
        used_vram: 0,
        busy_pct: 0,
        temp_c: None,
        display_attached: false,
    };
    // 98% cap on a normal GPU.
    assert_eq!(g.allocatable_vram(98, 80), 32 * gib * 98 / 100);
    // Existing usage is subtracted.
    g.used_vram = 10 * gib;
    assert_eq!(g.allocatable_vram(98, 80), 32 * gib * 98 / 100 - 10 * gib);
    // Display GPU drops to the 80% cap.
    g.display_attached = true;
    g.used_vram = 0;
    assert_eq!(g.allocatable_vram(98, 80), 32 * gib * 80 / 100);
}

#[test]
fn discovered_tags_follow_backend_indices() {
    let mut g = GpuInfo {
        id: "x".into(),
        pci_bus_id: Some("0000:03:00.0".into()),
        vulkan_device: None,
        vulkan_index: Some(1),
        cuda_device: None,
        cuda_index: None,
        rocm_index: Some(0),
        sycl_index: None,
        tags: BTreeSet::new(),
        integrated: false,
        total_vram: 32 << 30,
        used_vram: 0,
        busy_pct: 0,
        temp_c: None,
        display_attached: false,
    };
    apply_tags(std::slice::from_mut(&mut g), None);
    // An AMD card seen by both ROCm and Vulkan is tagged for both.
    assert!(g.tags.contains(&Backend::Rocm));
    assert!(g.tags.contains(&Backend::Vulkan));
    assert!(!g.tags.contains(&Backend::Cuda));
    assert!(g.supports(Backend::Rocm));
    assert_eq!(
        g.backend_device_name(Backend::Rocm).as_deref(),
        Some("ROCm0")
    );

    // An operator override replaces the discovered set (ROCm-only).
    let mut over = HashMap::new();
    over.insert(
        "0000:03:00.0".to_string(),
        [Backend::Rocm].into_iter().collect(),
    );
    apply_tags(std::slice::from_mut(&mut g), Some(&over));
    assert!(!g.supports(Backend::Vulkan));
    assert!(g.supports(Backend::Rocm));
}

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
        rocm_index: None,
        sycl_index: None,
        tags: Default::default(),
        integrated: false,
        total_vram: 100,
        used_vram: 150,
        busy_pct: 0,
        temp_c: None,
        display_attached: false,
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

    let gpus = VRAMTracker::default().refresh_from(root.to_str().unwrap());
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

    let gpus = VRAMTracker::default().refresh_from(root.to_str().unwrap());
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
    let gpus = VRAMTracker::default().refresh_from(root.to_str().unwrap());
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

    let gpus = VRAMTracker::default().refresh_from(root.to_str().unwrap());
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

    let gpus = VRAMTracker::default().refresh_from(root.to_str().unwrap());
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

    let gpus = VRAMTracker::default().refresh_from(root.to_str().unwrap());
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
            total_vram: MIN_USABLE_GPU_VRAM_BYTES,
        },
    );
    vulkan.insert(
        "0000:08:00.0".to_string(),
        VulkanDevice {
            index: 1,
            integrated: true,
            total_vram: 2 * 1024 * 1024 * 1024,
        },
    );

    let gpus = VRAMTracker::default().refresh_from_with_vulkan(root.to_str().unwrap(), &vulkan);
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
memoryHeaps: count = 1
	memoryHeaps[0]:
		size   = 34208743424 (0x7f7000000) (31.86 GiB)
		flags: count = 1
			MEMORY_HEAP_DEVICE_LOCAL_BIT
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
	deviceType        = PHYSICAL_DEVICE_TYPE_DISCRETE_GPU
	deviceName        = Intel(R) Arc(tm) Pro B60 Graphics (BMG G21)
VkPhysicalDevicePCIBusInfoPropertiesEXT:
	pciDomain   = 0
	pciBus      = 30
	pciDevice   = 0
	pciFunction = 0
memoryHeaps: count = 2
	memoryHeaps[0]:
		size   = 25669140480 (0x5fa000000) (23.91 GiB)
		flags: count = 1
			MEMORY_HEAP_DEVICE_LOCAL_BIT
	memoryHeaps[1]:
		size   = 50219685888 (0xbb1539800) (46.77 GiB)
		flags:
			None
GPU4:
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
            integrated: false,
            total_vram: 34208743424
        })
    );
    // NVIDIA discrete: decimal bus 28 → hex 1c. The whole point — NVIDIA is
    // mapped by PCI from PCIBusInfo, which the RADV-UUID parse could not do.
    assert_eq!(
        map.get("0000:1c:00.0"),
        Some(&VulkanDevice {
            index: 1,
            integrated: false,
            total_vram: 0
        })
    );
    // Integrated GPU is mapped (addressable) but flagged.
    assert_eq!(
        map.get("0000:08:00.0"),
        Some(&VulkanDevice {
            index: 2,
            integrated: true,
            total_vram: 0
        })
    );
    // Intel discrete: decimal bus 30 -> hex 1e.
    assert_eq!(
        map.get("0000:1e:00.0"),
        Some(&VulkanDevice {
            index: 3,
            integrated: false,
            total_vram: 25669140480
        })
    );
    // CPU/virtual device has no usable PCI mapping and is skipped.
    assert_eq!(map.len(), 4);
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
        rocm_index: None,
        sycl_index: None,
        tags: Default::default(),
        integrated: false,
        total_vram: 24 * 1024 * 1024 * 1024,
        used_vram: 0,
        busy_pct: 0,
        temp_c: None,
        display_attached: false,
    };
    merge_vulkan_identity(&mut nvidia, &map);
    // Keeps its CUDA identity and gains the Vulkan one.
    assert_eq!(nvidia.cuda_device.as_deref(), Some("CUDA0"));
    assert_eq!(nvidia.vulkan_index, Some(1));
    assert_eq!(nvidia.vulkan_device.as_deref(), Some("Vulkan1"));
}

#[test]
fn merge_vulkan_identity_gives_intel_a_vulkan_slot() {
    let map = parse_vulkaninfo(VULKANINFO_FIXTURE);
    let mut intel = GpuInfo {
        id: "0".into(),
        pci_bus_id: Some("0000:1e:00.0".into()),
        vulkan_device: None,
        vulkan_index: None,
        cuda_device: None,
        cuda_index: None,
        rocm_index: None,
        sycl_index: None,
        tags: Default::default(),
        integrated: false,
        total_vram: 24 * 1024 * 1024 * 1024,
        used_vram: 0,
        busy_pct: 0,
        temp_c: None,
        display_attached: false,
    };

    merge_vulkan_identity(&mut intel, &map);

    assert_eq!(intel.vulkan_index, Some(3));
    assert_eq!(intel.vulkan_device.as_deref(), Some("Vulkan3"));
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
        rocm_index: None,
        sycl_index: None,
        tags: Default::default(),
        integrated: false,
        total_vram: 24 * 1024 * 1024 * 1024,
        used_vram: 0,
        busy_pct: 0,
        temp_c: None,
        display_attached: false,
    };
    merge_vulkan_identity(&mut gpu, &map);
    assert_eq!(gpu.vulkan_index, None);
    assert_eq!(gpu.vulkan_device, None);
}
