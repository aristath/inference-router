use askama::Template;

use crate::config::{CacheType, ModelConfig, ModelState, WeightsFormat};
use crate::system::stats::SystemStats;
use crate::vram::tracker::GpuInfo;

// The app now renders a single page; the form and settings live inside that
// page as modals, so we no longer template them server-side.

/// Pre-formatted GPU data for templates.
#[derive(Debug, Clone)]
pub struct GpuDisplay {
    pub id: String,
    pub label: String,
    /// Stable PCI bus id, used as the key when editing this GPU's tags.
    pub pci_bus_id: String,
    /// Backend capability tags joined for display, e.g. "vulkan · rocm".
    pub tags_str: String,
    /// True when a monitor is connected (gets the lower VRAM fill cap).
    pub display_attached: bool,
    /// The VRAM fill cap as a label, e.g. "95%" or "75%".
    pub vram_cap_str: String,
    pub used_gib_str: String,
    pub total_gib_str: String,
    pub free_gib_str: String,
    pub usage_pct_str: String,
    pub usage_class: String,
    pub busy_pct_str: String,
    pub busy_class: String,
    pub temp_c_str: String, // "43" or "—"
    pub temp_class: String, // "green" | "yellow" | "red" | "muted"
}

impl GpuDisplay {
    pub fn from_gpu(gpu: &GpuInfo) -> Self {
        let total_gib = gpu.total_vram as f64 / 1_073_741_824.0;
        let used_gib = gpu.used_vram as f64 / 1_073_741_824.0;
        let free_gib = gpu.free_vram() as f64 / 1_073_741_824.0;
        let vram_pct = if gpu.total_vram > 0 {
            gpu.used_vram as f64 / gpu.total_vram as f64 * 100.0
        } else {
            0.0
        };
        let (temp_c_str, temp_class) = match gpu.temp_c {
            Some(t) => (format!("{:.0}", t), temp_class(t)),
            None => ("—".into(), "muted".into()),
        };
        Self {
            id: gpu.id.clone(),
            label: gpu
                .vulkan_device
                .as_ref()
                .map(|dev| match &gpu.pci_bus_id {
                    Some(pci) => format!("{dev} / {pci}"),
                    None => dev.clone(),
                })
                .unwrap_or_else(|| gpu.id.clone()),
            pci_bus_id: gpu.pci_bus_id.clone().unwrap_or_default(),
            tags_str: gpu
                .tags
                .iter()
                .map(|b| b.as_str())
                .collect::<Vec<_>>()
                .join(" · "),
            display_attached: gpu.display_attached,
            vram_cap_str: format!("{}%", gpu.vram_cap_pct()),
            used_gib_str: format!("{:.1}", used_gib),
            total_gib_str: format!("{:.1}", total_gib),
            free_gib_str: format!("{:.1}", free_gib),
            usage_pct_str: format!("{:.0}", vram_pct),
            usage_class: bar_class(vram_pct),
            busy_pct_str: gpu.busy_pct.to_string(),
            busy_class: bar_class(gpu.busy_pct as f64),
            temp_c_str,
            temp_class,
        }
    }
}

/// Aggregate VRAM + activity across all GPUs, shown at the top of the single
/// "GPUs" widget above the per-GPU breakdown. VRAM is additive (sum of used /
/// sum of total); activity is the mean utilization across GPUs, since it's a
/// percentage and summing it would be meaningless.
#[derive(Debug, Clone)]
pub struct GpuTotals {
    pub count: usize,
    pub vram_used_gib_str: String,
    pub vram_total_gib_str: String,
    pub vram_free_gib_str: String,
    pub vram_pct_str: String,
    pub vram_class: String,
    pub busy_pct_str: String,
    pub busy_class: String,
}

impl GpuTotals {
    pub fn from_gpus(gpus: &[GpuInfo]) -> Self {
        let count = gpus.len();
        let used: u64 = gpus.iter().map(|g| g.used_vram).sum();
        let total: u64 = gpus.iter().map(|g| g.total_vram).sum();
        let free = total.saturating_sub(used);
        let vram_pct = if total > 0 {
            used as f64 / total as f64 * 100.0
        } else {
            0.0
        };
        let busy_avg = if count > 0 {
            gpus.iter().map(|g| g.busy_pct as u32).sum::<u32>() as f64 / count as f64
        } else {
            0.0
        };
        Self {
            count,
            vram_used_gib_str: format!("{:.1}", used as f64 / 1_073_741_824.0),
            vram_total_gib_str: format!("{:.1}", total as f64 / 1_073_741_824.0),
            vram_free_gib_str: format!("{:.1}", free as f64 / 1_073_741_824.0),
            vram_pct_str: format!("{:.0}", vram_pct),
            vram_class: bar_class(vram_pct),
            busy_pct_str: format!("{:.0}", busy_avg),
            busy_class: bar_class(busy_avg),
        }
    }
}

/// Pre-formatted host metrics (CPU / RAM / CPU temp) for the dashboard.
#[derive(Debug, Clone)]
pub struct SystemDisplay {
    pub cpu_pct_str: String,
    pub cpu_class: String,
    pub ram_used_gib_str: String,
    pub ram_total_gib_str: String,
    pub ram_pct_str: String,
    pub ram_class: String,
    pub cpu_temp_c_str: String,
    pub cpu_temp_class: String,
}

impl SystemDisplay {
    pub fn from_stats(s: SystemStats) -> Self {
        let ram_used_gib = s.ram_used as f64 / 1_073_741_824.0;
        let ram_total_gib = s.ram_total as f64 / 1_073_741_824.0;
        let ram_pct = if s.ram_total > 0 {
            s.ram_used as f64 / s.ram_total as f64 * 100.0
        } else {
            0.0
        };
        let (cpu_temp_c_str, cpu_temp_class) = match s.cpu_temp_c {
            Some(t) => (format!("{:.0}", t), temp_class(t)),
            None => ("—".into(), "muted".into()),
        };
        Self {
            cpu_pct_str: format!("{:.0}", s.cpu_pct),
            cpu_class: bar_class(s.cpu_pct as f64),
            ram_used_gib_str: format!("{:.1}", ram_used_gib),
            ram_total_gib_str: format!("{:.1}", ram_total_gib),
            ram_pct_str: format!("{:.0}", ram_pct),
            ram_class: bar_class(ram_pct),
            cpu_temp_c_str,
            cpu_temp_class,
        }
    }
}

fn bar_class(pct: f64) -> String {
    if pct > 90.0 {
        "red"
    } else if pct > 70.0 {
        "yellow"
    } else {
        "green"
    }
    .into()
}

/// Rough green/yellow/red temperature bands for CPU and GPU junction temps.
/// 0–70 green, 70–85 yellow, >85 red.
fn temp_class(celsius: f32) -> String {
    if celsius > 85.0 {
        "red"
    } else if celsius > 70.0 {
        "yellow"
    } else {
        "green"
    }
    .into()
}

/// Pre-formatted model data for templates.
#[derive(Debug, Clone)]
pub struct ModelDisplay {
    pub id: String,
    pub name: String,
    pub context_tokens: u32,
    pub context_str: String,
    /// Raw bytes for sorting (not rendered directly).
    pub file_size_bytes: u64,
    pub file_size_gib_str: String,
    pub required_vram_bytes: u64,
    pub required_vram_gib_str: String,
    pub state_display: String,
    pub state_class: String,
    pub state_sort_key: u8,
    pub is_loaded: bool,
    pub loaded_sort_key: u8,
    pub primary_action_sort: String,
    /// Running instances for this model, from the live runtime (not config).
    /// Defaults to 0; set by the handler via [`ModelDisplay::with_runtime`].
    pub running_instances: usize,
    /// In-flight requests for this model, from the live runtime (not config).
    /// Defaults to 0; set by the handler via [`ModelDisplay::with_runtime`].
    pub active_requests: usize,
    /// Instances currently spawning, from the live runtime.
    pub pending_instances: usize,
}

impl ModelDisplay {
    pub fn from_model(m: &ModelConfig) -> Self {
        let (state_display, state_class) = match &m.state {
            ModelState::Idle => ("Idle".into(), "idle".into()),
            ModelState::Loading => ("Loading".into(), "loading".into()),
            ModelState::Running => ("Running".into(), "running".into()),
            ModelState::Error(msg) => (format!("Error: {}", msg), "error".into()),
        };
        let state_sort_key = match &m.state {
            ModelState::Running => 0,
            ModelState::Loading => 1,
            ModelState::Idle => 2,
            ModelState::Error(_) => 3,
        };
        let is_loaded = matches!(m.state, ModelState::Running | ModelState::Loading);
        let primary_action_sort = match &m.state {
            ModelState::Running => "stop",
            ModelState::Idle | ModelState::Error(_) => "load",
            ModelState::Loading => "edit",
        };
        let (file_size_bytes, required_vram_bytes) = compute_sizes(m);

        Self {
            id: m.id.clone(),
            name: m.name.clone(),
            context_tokens: m.context,
            context_str: format_context(m.context),
            file_size_bytes,
            file_size_gib_str: gib_or_dash(file_size_bytes),
            required_vram_bytes,
            required_vram_gib_str: gib_or_dash(required_vram_bytes),
            state_display,
            state_class,
            state_sort_key,
            is_loaded,
            loaded_sort_key: if is_loaded { 0 } else { 1 },
            primary_action_sort: primary_action_sort.into(),
            running_instances: 0,
            active_requests: 0,
            pending_instances: 0,
        }
    }

    /// Overlays live runtime counters (running instances / active requests /
    /// pending instances) onto a display built from static config.
    pub fn with_runtime(
        mut self,
        running_instances: usize,
        active_requests: usize,
        pending_instances: usize,
    ) -> Self {
        self.running_instances = running_instances;
        self.active_requests = active_requests;
        self.pending_instances = pending_instances;
        self
    }

    /// Combined activity used by the activity sort key.
    pub fn total_activity(&self) -> usize {
        self.active_requests + self.pending_instances
    }

    /// Sort weight for the merged "Status" column: higher = busier. Ranks
    /// running models (by in-flight requests, then instance count) above
    /// loading, then idle, then errored. `loaded_sort_key` still floats all
    /// loaded rows to the top first; this orders within those groups.
    pub fn status_sort_key(&self) -> u64 {
        let base = match self.state_sort_key {
            0 => 1_000_000, // Running
            1 => 10_000,    // Loading
            2 => 100,       // Idle
            _ => 0,         // Error
        };
        base + (self.active_requests as u64) * 100 + self.running_instances as u64
    }
}

/// Which model column the dashboard is sorted by, and in which direction.
///
/// The browser holds the user's choice and sends it as `?sort=&dir=` on each
/// poll; the server renders rows already ordered, so morphing never has to
/// fight a client-side re-sort (the old flash). Mirrors the comparator the
/// dashboard JS used to run: loaded models always float to the top, empty
/// numeric cells always sink to the bottom, ties break by name then id.
#[derive(Debug, Clone, Copy)]
pub struct ModelSort<'a> {
    pub key: &'a str,
    /// `true` for ascending. Unknown directions default to ascending.
    pub ascending: bool,
}

impl<'a> ModelSort<'a> {
    pub fn new(key: &'a str, dir: &str) -> Self {
        Self {
            key,
            ascending: !dir.eq_ignore_ascii_case("desc"),
        }
    }

    /// CSS class for the active column header (`sort-asc` / `sort-desc`).
    pub fn dir_class(&self) -> &'static str {
        if self.ascending {
            "sort-asc"
        } else {
            "sort-desc"
        }
    }

    /// `aria-sort` value for the active column header.
    pub fn aria(&self) -> &'static str {
        if self.ascending {
            "ascending"
        } else {
            "descending"
        }
    }
}

/// Filters `displays` by a case-insensitive substring over id/name/state, then
/// sorts in place per `sort`. Returns the rows the table should render in the
/// order they should appear.
///
/// Filtering matches the old `applyModelFilter` haystack (id, name, state) and
/// sorting matches `applyModelTableSort`: loaded-first, then the chosen key
/// with empty cells last, then name and id as stable tie-breakers.
pub fn sort_and_filter(
    mut displays: Vec<ModelDisplay>,
    sort: ModelSort<'_>,
    query: &str,
) -> Vec<ModelDisplay> {
    let needle = query.trim().to_lowercase();
    if !needle.is_empty() {
        displays.retain(|m| {
            m.id.to_lowercase().contains(&needle)
                || m.name.to_lowercase().contains(&needle)
                || m.state_class.to_lowercase().contains(&needle)
        });
    }

    let dir = if sort.ascending { 1 } else { -1 };
    displays.sort_by(|a, b| {
        // Loaded models always come first, regardless of column/direction.
        a.loaded_sort_key
            .cmp(&b.loaded_sort_key)
            .then_with(|| compare_key(a, b, sort.key, dir))
            // Stable tie-breakers so equal rows keep a deterministic order.
            .then_with(|| a.name.to_lowercase().cmp(&b.name.to_lowercase()))
            .then_with(|| a.id.cmp(&b.id))
    });
    displays
}

/// Compares two rows on a single column. `dir` is +1 (asc) or -1 (desc).
/// Numeric columns treat a missing value (rendered as "—") as empty and sort
/// it last in *both* directions, matching the old JS comparator.
fn compare_key(a: &ModelDisplay, b: &ModelDisplay, key: &str, dir: i32) -> std::cmp::Ordering {
    use std::cmp::Ordering;

    let apply = |ord: Ordering| if dir < 0 { ord.reverse() } else { ord };
    // (is_empty, value): empties always sink, then compare values with `dir`.
    let num = |empty_a: bool, va: u64, empty_b: bool, vb: u64| match (empty_a, empty_b) {
        (true, true) => Ordering::Equal,
        (true, false) => Ordering::Greater, // a empty -> after b
        (false, true) => Ordering::Less,
        (false, false) => apply(va.cmp(&vb)),
    };

    match key {
        "state" => apply((a.state_sort_key).cmp(&b.state_sort_key)),
        "status" => apply((a.status_sort_key()).cmp(&b.status_sort_key())),
        "context" => apply((a.context_tokens).cmp(&b.context_tokens)),
        "file-size" => num(
            a.file_size_bytes == 0,
            a.file_size_bytes,
            b.file_size_bytes == 0,
            b.file_size_bytes,
        ),
        "vram" => num(
            a.required_vram_bytes == 0,
            a.required_vram_bytes,
            b.required_vram_bytes == 0,
            b.required_vram_bytes,
        ),
        "activity" => apply((a.total_activity() as u64).cmp(&(b.total_activity() as u64))),
        "actions" => apply(a.primary_action_sort.cmp(&b.primary_action_sort)),
        // "name" and any unknown key fall back to name (text).
        _ => apply(a.name.to_lowercase().cmp(&b.name.to_lowercase())),
    }
}

fn format_context(ctx: u32) -> String {
    if ctx >= 1024 {
        format!("{}K", ctx / 1024)
    } else {
        format!("{}", ctx)
    }
}

fn gib_or_dash(bytes: u64) -> String {
    if bytes == 0 {
        "—".into()
    } else {
        format!("{:.1}", bytes as f64 / 1_073_741_824.0)
    }
}

/// Compute on-disk weights size and estimated required VRAM (weights + KV
/// cache at `m.context` + 10% overhead). Best-effort — returns `0` when the
/// file is missing or the GGUF header is unreadable.
///
/// For GGUF models we go through `GgufInfo::read`, which sums sibling
/// shards for multi-file models (`foo.gguf-00001-of-00005.gguf`) so the
/// displayed size reflects the whole model rather than only the file
/// the config points at.
fn compute_sizes(m: &ModelConfig) -> (u64, u64) {
    use crate::vram::estimator::{GgufInfo, KvPerElement, VramEstimate};

    match m.weights_format {
        WeightsFormat::Gguf => {
            let info = m
                .gguf_meta
                .as_ref()
                .map(GgufInfo::from)
                .or_else(|| GgufInfo::read(&m.model_path).ok());
            match info {
                Some(info) => {
                    // Honour the model's configured KV cache quantization so
                    // q8_0/q4_0 shrink the Required VRAM column, matching
                    // what llama-server actually allocates at run time.
                    // Unset falls back to f16, which is also llama.cpp's
                    // default.
                    let kv_bytes = KvPerElement::from_types(
                        m.cache_type_k.unwrap_or(CacheType::F16),
                        m.cache_type_v.unwrap_or(CacheType::F16),
                    );
                    let kv = info.kv_cache_bytes(m.context, kv_bytes);
                    let estimate =
                        VramEstimate::compute(info.file_size, kv, info.n_layers, m.n_gpu_layers);
                    (info.file_size, estimate.total_vram)
                }
                // Header unreadable (missing file, non-GGUF, etc.) — show a
                // dash (the gib_or_dash formatter handles 0).
                None => (0, 0),
            }
        }
        // Safetensors: sum the directory's regular files for size, and
        // leave required VRAM blank — estimating vLLM's memory without
        // instantiating the model is too tangled for a quick stat.
        WeightsFormat::Safetensors => (safetensors_dir_size(&m.model_path), 0),
    }
}

/// Sum of regular files inside a safetensors model directory. Returns 0
/// for missing or non-directory paths so the UI shows a dash.
fn safetensors_dir_size(path: &std::path::Path) -> u64 {
    let Ok(meta) = std::fs::metadata(path) else {
        return 0;
    };
    if !meta.is_dir() {
        return 0;
    }
    let Ok(rd) = std::fs::read_dir(path) else {
        return 0;
    };
    let mut total = 0u64;
    for entry in rd.flatten() {
        if let Ok(em) = entry.metadata() {
            if em.is_file() {
                total = total.saturating_add(em.len());
            }
        }
    }
    total
}

/// One pre-formatted orchestrator event for the dashboard's event log.
///
/// Rendered server-side now (it used to be built in JS from `/api/status`).
/// The timestamp is formatted in the server's local time — fine for a
/// localhost dashboard where server and browser share a clock.
#[derive(Debug, Clone)]
pub struct EventDisplay {
    pub time_str: String,
    pub level: String,
    pub message: String,
}

impl EventDisplay {
    pub fn new(ts: f64, level: &str, message: String) -> Self {
        use chrono::{Local, TimeZone};
        let time_str = Local
            .timestamp_opt(ts.trunc() as i64, 0)
            .single()
            .map(|dt| dt.format("%H:%M:%S").to_string())
            .unwrap_or_else(|| "--:--:--".into());
        Self {
            time_str,
            level: level.to_string(),
            message,
        }
    }
}

/// Full single-page dashboard (first paint). Shares the live-region partials
/// (`_live_left_inner.html`, `_live_models_inner.html`) with
/// [`DashboardFragmentTemplate`] so the initial HTML and every poll render from
/// the same markup.
#[derive(Template)]
#[template(path = "dashboard.html")]
pub struct DashboardTemplate {
    pub title: String,
    pub system: SystemDisplay,
    pub gpus: Vec<GpuDisplay>,
    pub gpu_totals: GpuTotals,
    pub events: Vec<EventDisplay>,
    /// Already filtered + sorted for display.
    pub models: Vec<ModelDisplay>,
    /// Whether any models exist at all (before filtering) — distinguishes the
    /// "No models configured" empty state from "No models match this filter".
    pub has_any_models: bool,
    /// Active sort column key (e.g. `"name"`, `"state"`).
    pub sort_key: String,
    /// `sort-asc` / `sort-desc` class for the active header.
    pub sort_dir_class: String,
    /// `ascending` / `descending` for the active header's `aria-sort`.
    pub sort_aria: String,
    pub server_port: u16,
}

/// The two live regions only, each wrapped for an idiomorph out-of-band swap.
/// Returned by `GET /fragment/dashboard` on every poll; htmx morphs each region
/// into place so unchanged rows/cards are never re-created (no flash).
#[derive(Template)]
#[template(path = "dashboard_fragment.html")]
pub struct DashboardFragmentTemplate {
    pub system: SystemDisplay,
    pub gpus: Vec<GpuDisplay>,
    pub gpu_totals: GpuTotals,
    pub events: Vec<EventDisplay>,
    pub models: Vec<ModelDisplay>,
    pub has_any_models: bool,
    pub sort_key: String,
    pub sort_dir_class: String,
    pub sort_aria: String,
}

#[cfg(test)]
mod tests {
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
}
