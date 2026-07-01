use super::*;
use crate::config::{ModelConfig, ModelState};

/// Builds a display straight from a minimal config. File/VRAM sizes resolve
/// to 0 (empty "—") because the default paths don't exist — which is what
/// the empty-cell ordering test relies on.
fn model(id: &str, name: &str, state: ModelState, context: u32) -> ModelDisplay {
    ModelDisplay::from_model(&ModelConfig {
        id: id.into(),
        name: name.into(),
        state,
        context,
        ..ModelConfig::default()
    })
}

fn ids(rows: &[ModelDisplay]) -> Vec<&str> {
    rows.iter().map(|m| m.id.as_str()).collect()
}

#[test]
fn loaded_models_float_to_the_top_regardless_of_column() {
    let rows = vec![
        model("a", "Alpha", ModelState::Idle, 4096),
        model("z", "Zeta", ModelState::Running, 2048),
    ];
    // Name-ascending would put Alpha first, but Running outranks Idle.
    let out = sort_and_filter(rows, ModelSort::new("name", "asc"), "");
    assert_eq!(ids(&out), ["z", "a"]);
}

#[test]
fn name_sort_honours_direction_within_the_loaded_group() {
    let rows = vec![
        model("a", "Alpha", ModelState::Idle, 1),
        model("c", "Charlie", ModelState::Idle, 1),
        model("b", "Bravo", ModelState::Idle, 1),
    ];
    let asc = sort_and_filter(rows.clone(), ModelSort::new("name", "asc"), "");
    assert_eq!(ids(&asc), ["a", "b", "c"]);
    let desc = sort_and_filter(rows, ModelSort::new("name", "desc"), "");
    assert_eq!(ids(&desc), ["c", "b", "a"]);
}

#[test]
fn context_sort_is_numeric() {
    let rows = vec![
        model("a", "A", ModelState::Idle, 8192),
        model("b", "B", ModelState::Idle, 1024),
        model("c", "C", ModelState::Idle, 131072),
    ];
    let out = sort_and_filter(rows, ModelSort::new("context", "asc"), "");
    assert_eq!(ids(&out), ["b", "a", "c"]);
}

#[test]
fn empty_numeric_cells_sort_last_in_both_directions() {
    let mut big = model("big", "Big", ModelState::Idle, 1);
    big.file_size_bytes = 1_000_000;
    big.file_size_gib_str = "0.9".into();
    let mut small = model("small", "Small", ModelState::Idle, 1);
    small.file_size_bytes = 1_000;
    small.file_size_gib_str = "0.1".into();
    let empty = model("empty", "Empty", ModelState::Idle, 1); // 0 bytes -> "—"

    let asc = sort_and_filter(
        vec![empty.clone(), big.clone(), small.clone()],
        ModelSort::new("file-size", "asc"),
        "",
    );
    assert_eq!(ids(&asc), ["small", "big", "empty"]);

    // Reversing the direction flips the populated rows but keeps the empty
    // cell at the bottom.
    let desc = sort_and_filter(
        vec![empty, big, small],
        ModelSort::new("file-size", "desc"),
        "",
    );
    assert_eq!(ids(&desc), ["big", "small", "empty"]);
}

#[test]
fn activity_sort_uses_active_plus_pending() {
    let rows = vec![
        model("idle", "Idle", ModelState::Running, 1).with_runtime(1, 0, 0),
        model("busy", "Busy", ModelState::Running, 1).with_runtime(1, 2, 1),
        model("warm", "Warm", ModelState::Running, 1).with_runtime(1, 1, 0),
    ];
    let out = sort_and_filter(rows, ModelSort::new("activity", "desc"), "");
    assert_eq!(ids(&out), ["busy", "warm", "idle"]);
}

#[test]
fn status_sort_ranks_running_by_requests_then_groups_by_state() {
    let rows = vec![
        model("err", "Err", ModelState::Error("boom".into()), 1),
        model("idle", "Idle", ModelState::Idle, 1),
        model("loading", "Loading", ModelState::Loading, 1).with_runtime(0, 0, 1),
        model("warm", "Warm", ModelState::Running, 1).with_runtime(1, 0, 0),
        model("busy", "Busy", ModelState::Running, 1).with_runtime(1, 3, 0),
    ];
    // desc = busiest first; loaded rows (running/loading) still float above
    // idle/error via loaded_sort_key.
    let out = sort_and_filter(rows, ModelSort::new("status", "desc"), "");
    assert_eq!(ids(&out), ["busy", "warm", "loading", "idle", "err"]);
}

#[test]
fn filter_matches_id_name_and_state_case_insensitively() {
    let rows = vec![
        model("qwen3", "Qwen 3", ModelState::Running, 1),
        model("llama", "Llama", ModelState::Idle, 1),
    ];
    // by name fragment
    let by_name = sort_and_filter(rows.clone(), ModelSort::new("name", "asc"), "QWEN");
    assert_eq!(ids(&by_name), ["qwen3"]);
    // by state
    let by_state = sort_and_filter(rows.clone(), ModelSort::new("name", "asc"), "running");
    assert_eq!(ids(&by_state), ["qwen3"]);
    // no match
    let none = sort_and_filter(rows, ModelSort::new("name", "asc"), "mistral");
    assert!(none.is_empty());
}

#[test]
fn model_sort_parses_direction_and_indicators() {
    assert!(ModelSort::new("name", "asc").ascending);
    assert!(ModelSort::new("name", "").ascending); // default
    assert!(!ModelSort::new("name", "desc").ascending);
    assert!(!ModelSort::new("name", "DESC").ascending); // case-insensitive
    assert_eq!(ModelSort::new("name", "asc").dir_class(), "sort-asc");
    assert_eq!(ModelSort::new("name", "desc").dir_class(), "sort-desc");
    assert_eq!(ModelSort::new("name", "asc").aria(), "ascending");
    assert_eq!(ModelSort::new("name", "desc").aria(), "descending");
}

#[test]
fn event_display_formats_a_timestamp() {
    let ev = EventDisplay::new(0.0, "info", "started".into());
    assert_eq!(ev.level, "info");
    assert_eq!(ev.message, "started");
    // HH:MM:SS, zero-padded.
    assert_eq!(ev.time_str.len(), 8);
    assert_eq!(ev.time_str.matches(':').count(), 2);
}

fn gpu(id: &str, total: u64, used: u64, busy: u8) -> GpuInfo {
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
        total_vram: total,
        used_vram: used,
        busy_pct: busy,
        temp_c: None,
        display_attached: false,
    }
}

#[test]
fn gpu_totals_sum_vram_and_average_activity() {
    const GIB: u64 = 1_073_741_824;
    let gpus = [
        gpu("0", 32 * GIB, 8 * GIB, 40),
        gpu("1", 32 * GIB, 24 * GIB, 60),
    ];
    let t = GpuTotals::from_gpus(&gpus);
    assert_eq!(t.count, 2);
    // VRAM is summed across devices.
    assert_eq!(t.vram_used_gib_str, "32.0"); // 8 + 24
    assert_eq!(t.vram_total_gib_str, "64.0");
    assert_eq!(t.vram_free_gib_str, "32.0");
    assert_eq!(t.vram_pct_str, "50"); // 32 / 64
                                      // Activity is the mean utilization, not a sum.
    assert_eq!(t.busy_pct_str, "50"); // (40 + 60) / 2
}

#[test]
fn gpu_totals_with_no_devices_is_zeroed() {
    let t = GpuTotals::from_gpus(&[]);
    assert_eq!(t.count, 0);
    assert_eq!(t.vram_pct_str, "0");
    assert_eq!(t.busy_pct_str, "0");
}

#[test]
fn fragment_template_wraps_regions_for_oob_morph() {
    let tpl = DashboardFragmentTemplate {
        system: SystemDisplay::from_stats(SystemStats {
            cpu_pct: 0.0,
            ram_used: 0,
            ram_total: 0,
            cpu_temp_c: None,
        }),
        gpus: vec![],
        gpu_totals: GpuTotals::from_gpus(&[]),
        events: vec![],
        models: vec![model("m1", "Model One", ModelState::Idle, 1)],
        has_any_models: true,
        sort_key: "name".into(),
        sort_dir_class: "sort-asc".into(),
        sort_aria: "ascending".into(),
    };
    let html = tpl.render().unwrap();
    // Both live regions are tagged for an out-of-band morph...
    assert!(html.contains(r#"id="live-left" hx-swap-oob="morph""#));
    assert!(html.contains(r#"id="live-models" hx-swap-oob="morph""#));
    // ...and the model row is keyed by id so idiomorph matches it in place.
    assert!(html.contains(r#"id="model-row-m1""#));
    // The active sort column carries its arrow class.
    assert!(html.contains("sort-asc"));
}
