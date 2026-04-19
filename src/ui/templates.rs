use askama::Template;

use crate::config::{ModelConfig, ModelState, WeightsFormat};
use crate::system::stats::SystemStats;
use crate::vram::tracker::GpuInfo;

// The app now renders a single page; the form and settings live inside that
// page as modals, so we no longer template them server-side.

/// Pre-formatted GPU data for templates.
#[derive(Debug, Clone)]
pub struct GpuDisplay {
    pub id: String,
    pub used_gib_str: String,
    pub total_gib_str: String,
    pub free_gib_str: String,
    pub usage_pct_str: String,
    pub usage_class: String,
    pub busy_pct_str: String,
    pub busy_class: String,
    pub temp_c_str: String,    // "43" or "—"
    pub temp_class: String,    // "green" | "yellow" | "red" | "muted"
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
        let ram_pct = if s.ram_total > 0 { s.ram_used as f64 / s.ram_total as f64 * 100.0 } else { 0.0 };
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
    if pct > 90.0 { "red" } else if pct > 70.0 { "yellow" } else { "green" }.into()
}

/// Rough green/yellow/red temperature bands for CPU and GPU junction temps.
/// 0–70 green, 70–85 yellow, >85 red.
fn temp_class(celsius: f32) -> String {
    if celsius > 85.0 { "red" } else if celsius > 70.0 { "yellow" } else { "green" }.into()
}

/// Pre-formatted model data for templates.
#[derive(Debug, Clone)]
pub struct ModelDisplay {
    pub id: String,
    pub name: String,
    pub format_str: String,
    pub port: u16,
    pub context: u32,
    /// Raw bytes for client-side numeric sort (via `data-sort` attribute).
    pub file_size_bytes: u64,
    pub file_size_gib_str: String,
    pub required_vram_bytes: u64,
    pub required_vram_gib_str: String,
    pub state_display: String,
    pub state_class: String,
}

impl ModelDisplay {
    pub fn from_model(m: &ModelConfig) -> Self {
        let (state_display, state_class) = match &m.state {
            ModelState::Idle => ("Idle".into(), "idle".into()),
            ModelState::Loading => ("Loading".into(), "loading".into()),
            ModelState::Running => ("Running".into(), "running".into()),
            ModelState::Error(msg) => (format!("Error: {}", msg), "error".into()),
        };
        let format_str = match m.weights_format {
            WeightsFormat::Gguf => "GGUF".into(),
            WeightsFormat::Safetensors => "Safetensors".into(),
        };

        let (file_size_bytes, required_vram_bytes) = compute_sizes(m);

        Self {
            id: m.id.clone(),
            name: m.name.clone(),
            format_str,
            port: m.port,
            context: m.context,
            file_size_bytes,
            file_size_gib_str: gib_or_dash(file_size_bytes),
            required_vram_bytes,
            required_vram_gib_str: gib_or_dash(required_vram_bytes),
            state_display,
            state_class,
        }
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
fn compute_sizes(m: &ModelConfig) -> (u64, u64) {
    use crate::vram::estimator::VramEstimate;

    let file_size = size_of_path(&m.model_path);

    let required = match m.weights_format {
        WeightsFormat::Gguf => VramEstimate::from_gguf(&m.model_path, m.context)
            .map(|e| e.total_vram)
            .unwrap_or(0),
        // Estimating vLLM's memory without instantiating the model is too
        // tangled for a quick server-side stat; leave blank.
        WeightsFormat::Safetensors => 0,
    };

    (file_size, required)
}

/// For a GGUF model: plain file stat.
/// For a Safetensors directory: sum of contained regular files.
fn size_of_path(path: &std::path::Path) -> u64 {
    let Ok(meta) = std::fs::metadata(path) else { return 0 };
    if meta.is_file() {
        return meta.len();
    }
    if meta.is_dir() {
        let mut total = 0u64;
        if let Ok(rd) = std::fs::read_dir(path) {
            for entry in rd.flatten() {
                if let Ok(em) = entry.metadata() {
                    if em.is_file() {
                        total = total.saturating_add(em.len());
                    }
                }
            }
        }
        return total;
    }
    0
}

#[derive(Template)]
#[template(path = "dashboard.html")]
pub struct DashboardTemplate {
    pub title: String,
    pub system: SystemDisplay,
    pub gpus: Vec<GpuDisplay>,
    pub models: Vec<ModelDisplay>,
    pub server_port: u16,
}
