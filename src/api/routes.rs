use axum::extract::{Path, Query, State};
use axum::http::StatusCode;
use axum::response::IntoResponse;
use axum::Json;
use serde::Deserialize;
use std::path::PathBuf;

use crate::config::{
    BinaryPreset, CacheType, ModelConfig, ModelState, ReasoningFormat, SplitMode, WeightsFormat,
};
use crate::orchestrator::{AppState, LoadError, MutationError, StopError};
use crate::vram::estimator::GgufInfo;

fn default_temperature() -> f32 { 0.6 }
fn default_top_p() -> f32 { 0.95 }
fn default_top_k() -> i32 { 40 }
fn default_min_p() -> f32 { 0.0 }
fn default_presence_penalty() -> f32 { 0.0 }
fn default_repeat_penalty() -> f32 { 1.0 }

#[derive(Deserialize)]
pub struct ModelRequest {
    pub id: String,
    pub name: String,
    pub weights_format: WeightsFormat,
    #[serde(default)]
    pub binary_preset: Option<String>,
    pub binary: String,
    pub model_path: String,
    pub port: u16,
    pub context: u32,
    #[serde(default = "default_temperature")]
    pub temperature: f32,
    #[serde(default = "default_top_p")]
    pub top_p: f32,
    #[serde(default = "default_top_k")]
    pub top_k: i32,
    #[serde(default = "default_min_p")]
    pub min_p: f32,
    #[serde(default = "default_presence_penalty")]
    pub presence_penalty: f32,
    #[serde(default = "default_repeat_penalty")]
    pub repeat_penalty: f32,
    #[serde(default)]
    pub extra_args: Vec<String>,
    #[serde(default)]
    pub flash_attn: bool,
    #[serde(default)]
    pub n_gpu_layers: Option<u32>,
    #[serde(default)]
    pub mlock: bool,
    #[serde(default)]
    pub no_mmap: bool,
    #[serde(default)]
    pub parallel_slots: Option<u32>,
    #[serde(default)]
    pub cache_type_k: Option<CacheType>,
    #[serde(default)]
    pub cache_type_v: Option<CacheType>,
    #[serde(default)]
    pub split_mode: Option<SplitMode>,
    #[serde(default)]
    pub main_gpu: Option<u32>,
    #[serde(default)]
    pub tensor_split: Option<String>,
    #[serde(default)]
    pub threads: Option<i32>,
    #[serde(default)]
    pub cache_ram_mib: Option<i32>,
    #[serde(default)]
    pub reasoning_format: Option<ReasoningFormat>,
    #[serde(default)]
    pub reasoning_budget: Option<i32>,
    #[serde(default)]
    pub chat_template_kwargs: Option<String>,
}

impl ModelRequest {
    fn into_config(self) -> ModelConfig {
        ModelConfig {
            id: self.id,
            name: self.name,
            weights_format: self.weights_format,
            binary_preset: self.binary_preset,
            binary: PathBuf::from(self.binary),
            model_path: PathBuf::from(self.model_path),
            port: self.port,
            extra_args: self.extra_args,
            context: self.context,
            temperature: self.temperature,
            top_p: self.top_p,
            top_k: self.top_k,
            min_p: self.min_p,
            presence_penalty: self.presence_penalty,
            repeat_penalty: self.repeat_penalty,
            flash_attn: self.flash_attn,
            n_gpu_layers: self.n_gpu_layers,
            mlock: self.mlock,
            no_mmap: self.no_mmap,
            parallel_slots: self.parallel_slots,
            cache_type_k: self.cache_type_k,
            cache_type_v: self.cache_type_v,
            split_mode: self.split_mode,
            main_gpu: self.main_gpu,
            tensor_split: self.tensor_split,
            threads: self.threads,
            cache_ram_mib: self.cache_ram_mib,
            reasoning_format: self.reasoning_format,
            reasoning_budget: self.reasoning_budget,
            chat_template_kwargs: self.chat_template_kwargs,
            state: ModelState::Idle,
            pid: None,
            estimated_vram: 0,
            last_used: None,
        }
    }
}

pub async fn list_models(State(state): State<AppState>) -> impl IntoResponse {
    Json(state.list_models().await)
}

pub async fn create_model(
    State(state): State<AppState>,
    Json(req): Json<ModelRequest>,
) -> impl IntoResponse {
    let id = req.id.clone();
    let model = req.into_config();
    match state.add_model(model).await {
        Ok(()) => (StatusCode::CREATED, Json(serde_json::json!({"id": id}))).into_response(),
        Err(MutationError::Conflict(id)) => (
            StatusCode::CONFLICT,
            Json(serde_json::json!({"error": format!("model '{id}' already exists")})),
        )
            .into_response(),
        Err(MutationError::NotFound(_)) => unreachable!("add_model never returns NotFound"),
    }
}

pub async fn update_model(
    State(state): State<AppState>,
    Path(id): Path<String>,
    Json(req): Json<ModelRequest>,
) -> impl IntoResponse {
    // Enforce URL id == body id to avoid renaming confusion.
    if req.id != id {
        return (
            StatusCode::BAD_REQUEST,
            Json(serde_json::json!({"error": "id in URL and body must match"})),
        )
            .into_response();
    }
    let model = req.into_config();
    match state.update_model(model).await {
        Ok(()) => (StatusCode::OK, Json(serde_json::json!({"ok": true}))).into_response(),
        Err(MutationError::NotFound(id)) => (
            StatusCode::NOT_FOUND,
            Json(serde_json::json!({"error": format!("model '{id}' not found")})),
        )
            .into_response(),
        Err(MutationError::Conflict(_)) => unreachable!("update_model never returns Conflict"),
    }
}

pub async fn delete_model(
    State(state): State<AppState>,
    Path(id): Path<String>,
) -> impl IntoResponse {
    match state.remove_model(&id).await {
        Ok(()) => (StatusCode::OK, Json(serde_json::json!({"ok": true}))).into_response(),
        Err(MutationError::NotFound(id)) => (
            StatusCode::NOT_FOUND,
            Json(serde_json::json!({"error": format!("model '{id}' not found")})),
        )
            .into_response(),
        Err(MutationError::Conflict(_)) => unreachable!(),
    }
}

pub async fn load_model(
    State(state): State<AppState>,
    Path(id): Path<String>,
) -> impl IntoResponse {
    match state.ensure_loaded(&id).await {
        Ok(port) => (
            StatusCode::OK,
            Json(serde_json::json!({"ok": true, "port": port})),
        )
            .into_response(),
        Err(LoadError::ModelNotFound(id)) => (
            StatusCode::NOT_FOUND,
            Json(serde_json::json!({"error": format!("model '{id}' not found")})),
        )
            .into_response(),
        Err(e @ LoadError::PresetNotFound(_)) => (
            StatusCode::UNPROCESSABLE_ENTITY,
            Json(serde_json::json!({"error": e.to_string()})),
        )
            .into_response(),
        Err(LoadError::SpawnFailed(e)) => (
            StatusCode::SERVICE_UNAVAILABLE,
            Json(serde_json::json!({"error": format!("spawn failed: {e}")})),
        )
            .into_response(),
    }
}

// ===== Binary presets =====

#[derive(Deserialize)]
pub struct PresetRequest {
    pub id: String,
    pub name: String,
    pub binary: String,
}

impl PresetRequest {
    fn into_preset(self) -> BinaryPreset {
        BinaryPreset {
            id: self.id,
            name: self.name,
            binary: PathBuf::from(self.binary),
        }
    }
}

pub async fn list_presets(State(state): State<AppState>) -> impl IntoResponse {
    Json(state.list_presets().await)
}

pub async fn create_preset(
    State(state): State<AppState>,
    Json(req): Json<PresetRequest>,
) -> impl IntoResponse {
    let id = req.id.clone();
    match state.add_preset(req.into_preset()).await {
        Ok(()) => (StatusCode::CREATED, Json(serde_json::json!({"id": id}))).into_response(),
        Err(MutationError::Conflict(id)) => (
            StatusCode::CONFLICT,
            Json(serde_json::json!({"error": format!("preset '{id}' already exists")})),
        )
            .into_response(),
        Err(MutationError::NotFound(_)) => unreachable!(),
    }
}

pub async fn update_preset(
    State(state): State<AppState>,
    Path(id): Path<String>,
    Json(req): Json<PresetRequest>,
) -> impl IntoResponse {
    if req.id != id {
        return (
            StatusCode::BAD_REQUEST,
            Json(serde_json::json!({"error": "id in URL and body must match"})),
        )
            .into_response();
    }
    match state.update_preset(req.into_preset()).await {
        Ok(()) => (StatusCode::OK, Json(serde_json::json!({"ok": true}))).into_response(),
        Err(MutationError::NotFound(id)) => (
            StatusCode::NOT_FOUND,
            Json(serde_json::json!({"error": format!("preset '{id}' not found")})),
        )
            .into_response(),
        Err(MutationError::Conflict(_)) => unreachable!(),
    }
}

pub async fn delete_preset(
    State(state): State<AppState>,
    Path(id): Path<String>,
) -> impl IntoResponse {
    match state.remove_preset(&id).await {
        Ok(()) => (StatusCode::OK, Json(serde_json::json!({"ok": true}))).into_response(),
        Err(MutationError::NotFound(id)) => (
            StatusCode::NOT_FOUND,
            Json(serde_json::json!({"error": format!("preset '{id}' not found")})),
        )
            .into_response(),
        Err(MutationError::Conflict(_)) => unreachable!(),
    }
}

pub async fn stop_model(
    State(state): State<AppState>,
    Path(id): Path<String>,
) -> impl IntoResponse {
    match state.stop_model(&id).await {
        Ok(()) => (StatusCode::OK, Json(serde_json::json!({"ok": true}))).into_response(),
        Err(StopError::ModelNotFound(id)) => (
            StatusCode::NOT_FOUND,
            Json(serde_json::json!({"error": format!("model '{id}' not found")})),
        )
            .into_response(),
    }
}

#[derive(Deserialize)]
pub struct GgufInfoQuery {
    pub path: String,
}

/// Reads GGUF metadata for a file on disk. Used by the model form to drive
/// the context slider's upper bound and the live VRAM preview.
pub async fn gguf_info(Query(q): Query<GgufInfoQuery>) -> impl IntoResponse {
    let expanded = shellexpand::tilde(&q.path).to_string();
    match GgufInfo::read(std::path::Path::new(&expanded)) {
        Ok(info) => (StatusCode::OK, Json(info)).into_response(),
        Err(e) => (
            StatusCode::BAD_REQUEST,
            Json(serde_json::json!({"error": e.to_string()})),
        )
            .into_response(),
    }
}

#[derive(Deserialize)]
pub struct FileBrowserQuery {
    pub path: String,
}

pub async fn list_files(Query(req): Query<FileBrowserQuery>) -> impl IntoResponse {
    let expanded = shellexpand::tilde(&req.path).to_string();
    let path = std::path::Path::new(&expanded);
    if !path.exists() || !path.is_dir() {
        return (
            StatusCode::NOT_FOUND,
            Json(serde_json::json!({"error": "not a directory"})),
        )
            .into_response();
    }

    let mut entries = Vec::new();
    if let Ok(read_dir) = std::fs::read_dir(path) {
        for entry in read_dir.flatten() {
            let name = entry.file_name().to_string_lossy().into_owned();
            let is_dir = entry.path().is_dir();
            let full = entry.path().to_string_lossy().into_owned();
            entries.push(serde_json::json!({
                "name": name,
                "path": full,
                "is_dir": is_dir,
            }));
        }
    }
    entries.sort_by(|a, b| {
        let a_dir = a["is_dir"].as_bool().unwrap_or(false);
        let b_dir = b["is_dir"].as_bool().unwrap_or(false);
        match (a_dir, b_dir) {
            (true, false) => std::cmp::Ordering::Less,
            (false, true) => std::cmp::Ordering::Greater,
            _ => a["name"].as_str().unwrap_or("").cmp(b["name"].as_str().unwrap_or("")),
        }
    });
    Json(entries).into_response()
}
