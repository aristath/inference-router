use serde::{Deserialize, Serialize};
use std::collections::{BTreeSet, HashMap};

use crate::config::Backend;

/// A user override of a single GPU's backend capability tags, persisted in
/// `gpus.json` and keyed by the GPU's PCI bus id (stable across reboots and
/// VRAM refreshes — the DRM card number is not).
///
/// Tags are auto-seeded from hardware discovery (a backend that enumerates the
/// GPU → that tag). An override here *replaces* the discovered set for that GPU,
/// letting the operator e.g. keep an AMD card out of the Vulkan pool while still
/// using it for ROCm. GPUs without an override use their discovered defaults.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct GpuTagOverride {
    pub pci_bus_id: String,
    #[serde(default)]
    pub tags: BTreeSet<Backend>,
}

/// Collapses the persisted overrides into a `pci -> tags` lookup. Later entries
/// win on duplicate PCI ids so a hand-edited file can't produce ambiguity.
pub fn tag_overrides_by_pci(overrides: &[GpuTagOverride]) -> HashMap<String, BTreeSet<Backend>> {
    overrides
        .iter()
        .map(|o| (o.pci_bus_id.clone(), o.tags.clone()))
        .collect()
}
