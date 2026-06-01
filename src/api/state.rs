use axum::extract::State;
use axum::response::IntoResponse;
use axum::Json;
use serde::Serialize;

use crate::config::{ModelState, WeightsFormat};
use crate::orchestrator::AppState;

#[derive(Serialize)]
struct StatusResponse {
    system: SystemResponse,
    gpus: Vec<GpuResponse>,
    models: Vec<ModelResponse>,
    events: Vec<crate::orchestrator::engine::AppEvent>,
    server_port: u16,
}

#[derive(Serialize)]
struct SystemResponse {
    cpu_pct: f32,
    ram_used: u64,
    ram_total: u64,
    cpu_temp_c: Option<f32>,
}

#[derive(Serialize)]
struct GpuResponse {
    id: String,
    pci_bus_id: Option<String>,
    vulkan_device: Option<String>,
    vulkan_index: Option<usize>,
    cuda_device: Option<String>,
    cuda_index: Option<usize>,
    total_vram: u64,
    used_vram: u64,
    busy_pct: u8,
    temp_c: Option<f32>,
}

#[derive(Serialize)]
struct ModelResponse {
    id: String,
    name: String,
    profile: Option<String>,
    weights_format: &'static str,
    binary: String,
    model_path: String,
    context: u32,
    extra_args: Vec<String>,
    state: &'static str,
    state_message: Option<String>,
    pid: Option<i32>,
    estimated_vram: u64,
    last_used: Option<f64>,
    instances: usize,
    pending_instances: usize,
    active_requests: usize,
}

/// Full system snapshot used by the dashboard poller.
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
