use axum::extract::State;
use axum::response::IntoResponse;
use axum::Json;
use serde::Serialize;

use crate::config::{ModelState, WeightsFormat};
use crate::orchestrator::AppState;

/// Complete system status snapshot returned by the `/api/status` endpoint.
/// Used by the dashboard to display live GPU/CPU/RAM metrics and model states.
#[derive(Serialize)]
struct StatusResponse {
    /// Host-level metrics (CPU %, RAM usage, temperature)
    system: SystemResponse,
    /// GPU information including VRAM usage and temperature
    gpus: Vec<GpuResponse>,
    /// All configured models with their runtime state
    models: Vec<ModelResponse>,
    /// Recent orchestrator events (loads, stops, errors)
    events: Vec<crate::orchestrator::engine::AppEvent>,
    /// Port the server is listening on
    server_port: u16,
}

/// Host system metrics snapshot.
#[derive(Serialize)]
struct SystemResponse {
    /// CPU utilization percentage (0-100)
    cpu_pct: f32,
    /// Used RAM in bytes
    ram_used: u64,
    /// Total RAM in bytes
    ram_total: u64,
    /// CPU temperature in Celsius (None if unavailable)
    cpu_temp_c: Option<f32>,
}

/// GPU metrics including VRAM usage and temperature.
#[derive(Serialize)]
struct GpuResponse {
    /// GPU identifier (e.g., "gfx1100")
    id: String,
    /// PCI bus ID (e.g., "0000:03:00.0")
    pci_bus_id: Option<String>,
    /// Vulkan device name if available
    vulkan_device: Option<String>,
    /// Vulkan device index if available
    vulkan_index: Option<usize>,
    /// CUDA device name if available
    cuda_device: Option<String>,
    /// CUDA device index if available
    cuda_index: Option<usize>,
    /// Total VRAM in bytes
    total_vram: u64,
    /// Used VRAM in bytes
    used_vram: u64,
    /// GPU utilization percentage (0-100)
    busy_pct: u8,
    /// GPU temperature in Celsius (None if unavailable)
    temp_c: Option<f32>,
}

/// Model configuration and runtime state.
#[derive(Serialize)]
struct ModelResponse {
    /// Unique model identifier
    id: String,
    /// Display name
    name: String,
    /// Optional profile tag for filtering
    profile: Option<String>,
    /// Weights format ("gguf" or "safetensors")
    weights_format: &'static str,
    /// Path to the inference server binary
    binary: String,
    /// Path to model weights
    model_path: String,
    /// Context length
    context: u32,
    /// Additional CLI arguments
    extra_args: Vec<String>,
    /// Current state ("idle", "loading", "running", "error")
    state: &'static str,
    /// Error message if state is "error"
    state_message: Option<String>,
    /// Process ID if running
    pid: Option<i32>,
    /// Estimated VRAM usage in bytes
    estimated_vram: u64,
    /// Timestamp of last use (None if never used)
    last_used: Option<f64>,
    /// Number of running instances
    instances: usize,
    /// Number of instances being loaded
    pending_instances: usize,
    /// Number of active requests
    active_requests: usize,
}

/// Returns a complete system status snapshot.
/// 
/// # Endpoint
/// `GET /api/status`
/// 
/// # Response
/// Returns a `StatusResponse` containing:
/// - System metrics (CPU, RAM, temperature)
/// - GPU information (VRAM, utilization, temperature)
/// - All model configurations and runtime states
/// - Recent orchestrator events
/// - Server port
/// 
/// # Used by
/// - Dashboard polling (every 500ms)
/// - CLI status commands
/// - Health checks
pub async fn get_app_state(State(state): State<AppState>) -> impl IntoResponse {
    let gpus = state.list_gpus().await;
    let models = state.list_models().await;
    let runtimes = state.model_runtimes().await;
    let events = state.recent_events().await;
    let sys = state.system_stats();

    let response = StatusResponse {
        server_port: state.server_port,
        events,
        system: SystemResponse {
            cpu_pct: sys.cpu_pct,
            ram_used: sys.ram_used,
            ram_total: sys.ram_total,
            cpu_temp_c: sys.cpu_temp_c,
        },
        gpus: gpus
            .into_iter()
            .map(|g| GpuResponse {
                id: g.id,
                pci_bus_id: g.pci_bus_id,
                vulkan_device: g.vulkan_device,
                vulkan_index: g.vulkan_index,
                cuda_device: g.cuda_device,
                cuda_index: g.cuda_index,
                total_vram: g.total_vram,
                used_vram: g.used_vram,
                busy_pct: g.busy_pct,
                temp_c: g.temp_c,
            })
            .collect(),
        models: models
            .into_iter()
            .map(|m| {
                let runtime = runtimes.get(&m.id).copied().unwrap_or_default();
                let (state_name, state_message) = match &m.state {
                    ModelState::Idle => ("idle", None),
                    ModelState::Loading => ("loading", None),
                    ModelState::Running => ("running", None),
                    ModelState::Error(msg) => ("error", Some(msg.clone())),
                };
                ModelResponse {
                    id: m.id,
                    name: m.name,
                    profile: m.profile,
                    weights_format: match m.weights_format {
                        WeightsFormat::Gguf => "gguf",
                        WeightsFormat::Safetensors => "safetensors",
                    },
                    binary: m.binary.to_string_lossy().into_owned(),
                    model_path: m.model_path.to_string_lossy().into_owned(),
                    context: m.context,
                    extra_args: m.extra_args,
                    state: state_name,
                    state_message,
                    pid: m.pid,
                    estimated_vram: m.estimated_vram,
                    last_used: m.last_used,
                    instances: runtime.instances,
                    pending_instances: runtime.pending,
                    active_requests: runtime.active,
                }
            })
            .collect(),
    };

    Json(response)
}
