use axum::extract::{Path, Query, State};
use axum::http::StatusCode;
use axum::response::IntoResponse;
use axum::Json;
use serde::Deserialize;
use std::path::Path as StdPath;

use crate::config::{BinaryPreset, ModelConfig};
use crate::orchestrator::{AppState, LoadError, MutationError, StopError};
use crate::vram::estimator::GgufMeta;

pub async fn list_models(State(state): State<AppState>) -> impl IntoResponse {
    Json(state.list_models().await)
}

/// Convert the orchestrator's mutation errors into an HTTP response. Used by
/// every create/update/delete handler so the mapping lives in one place.
fn mutation_response(e: MutationError) -> axum::response::Response {
    match e {
        MutationError::NotFound(id) => (
            StatusCode::NOT_FOUND,
            Json(serde_json::json!({"error": format!("'{id}' not found")})),
        )
            .into_response(),
        MutationError::Conflict(id) => (
            StatusCode::CONFLICT,
            Json(serde_json::json!({"error": format!("'{id}' already exists")})),
        )
            .into_response(),
        MutationError::InvalidConfig(err) => (
            StatusCode::UNPROCESSABLE_ENTITY,
            Json(serde_json::json!({"error": err.to_string()})),
        )
            .into_response(),
        MutationError::DraftInUse { id, targets } => (
            StatusCode::CONFLICT,
            Json(serde_json::json!({
                "error": format!(
                    "cannot delete draft '{id}': still referenced by {}. \
                     Unset draft_model_id on those targets first.",
                    targets.join(", "),
                ),
            })),
        )
            .into_response(),
    }
}

pub async fn create_model(
    State(state): State<AppState>,
    Json(model): Json<ModelConfig>,
) -> impl IntoResponse {
    let id = model.id.clone();
    match state.add_model(model).await {
        Ok(()) => (StatusCode::CREATED, Json(serde_json::json!({"id": id}))).into_response(),
        Err(e) => mutation_response(e),
    }
}

pub async fn update_model(
    State(state): State<AppState>,
    Path(id): Path<String>,
    Json(model): Json<ModelConfig>,
) -> impl IntoResponse {
    // Enforce URL id == body id to avoid renaming confusion.
    if model.id != id {
        return (
            StatusCode::BAD_REQUEST,
            Json(serde_json::json!({"error": "id in URL and body must match"})),
        )
            .into_response();
    }
    match state.update_model(model).await {
        Ok(()) => (StatusCode::OK, Json(serde_json::json!({"ok": true}))).into_response(),
        Err(e) => mutation_response(e),
    }
}

pub async fn delete_model(
    State(state): State<AppState>,
    Path(id): Path<String>,
) -> impl IntoResponse {
    match state.remove_model(&id).await {
        Ok(()) => (StatusCode::OK, Json(serde_json::json!({"ok": true}))).into_response(),
        Err(e) => mutation_response(e),
    }
}

pub async fn load_model(
    State(state): State<AppState>,
    Path(id): Path<String>,
) -> impl IntoResponse {
    match state.clone().ensure_loaded(&id).await {
        Ok(guard) => (
            StatusCode::OK,
            Json(serde_json::json!({"ok": true, "port": guard.port})),
        )
            .into_response(),
        Err(LoadError::ModelNotFound(id)) => (
            StatusCode::NOT_FOUND,
            Json(serde_json::json!({"error": format!("model '{id}' not found")})),
        )
            .into_response(),
        Err(e @ (LoadError::PresetNotFound(_) | LoadError::DraftNotFound { .. })) => (
            StatusCode::UNPROCESSABLE_ENTITY,
            Json(serde_json::json!({"error": e.to_string()})),
        )
            .into_response(),
        Err(LoadError::SpawnFailed(e)) => (
            StatusCode::SERVICE_UNAVAILABLE,
            Json(serde_json::json!({"error": format!("spawn failed: {e}")})),
        )
            .into_response(),
        Err(e @ LoadError::InsufficientVram { .. }) => (
            StatusCode::SERVICE_UNAVAILABLE,
            Json(serde_json::json!({"error": e.to_string()})),
        )
            .into_response(),
    }
}

// ===== Binary presets =====

pub async fn list_presets(State(state): State<AppState>) -> impl IntoResponse {
    Json(state.list_presets().await)
}

pub async fn create_preset(
    State(state): State<AppState>,
    Json(preset): Json<BinaryPreset>,
) -> impl IntoResponse {
    let id = preset.id.clone();
    match state.add_preset(preset).await {
        Ok(()) => (StatusCode::CREATED, Json(serde_json::json!({"id": id}))).into_response(),
        Err(e) => mutation_response(e),
    }
}

pub async fn update_preset(
    State(state): State<AppState>,
    Path(id): Path<String>,
    Json(preset): Json<BinaryPreset>,
) -> impl IntoResponse {
    if preset.id != id {
        return (
            StatusCode::BAD_REQUEST,
            Json(serde_json::json!({"error": "id in URL and body must match"})),
        )
            .into_response();
    }
    match state.update_preset(preset).await {
        Ok(()) => (StatusCode::OK, Json(serde_json::json!({"ok": true}))).into_response(),
        Err(e) => mutation_response(e),
    }
}

pub async fn delete_preset(
    State(state): State<AppState>,
    Path(id): Path<String>,
) -> impl IntoResponse {
    match state.remove_preset(&id).await {
        Ok(()) => (StatusCode::OK, Json(serde_json::json!({"ok": true}))).into_response(),
        Err(e) => mutation_response(e),
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

// ===== File-browser + GGUF metadata =====
//
// Both endpoints accept arbitrary paths. The dashboard is localhost-only and
// single-user, and the user already has shell access to the whole
// filesystem, so restricting the browser to a subtree would only frustrate
// legitimate uses (e.g. picking a GGUF from /mnt).

#[derive(Deserialize)]
pub struct GgufInfoQuery {
    pub path: String,
}

/// Reads GGUF metadata for a file on disk. Used by the model form to drive
/// the context slider's upper bound and the live VRAM preview.
pub async fn gguf_info(Query(q): Query<GgufInfoQuery>) -> impl IntoResponse {
    let expanded = shellexpand::tilde(&q.path).to_string();
    match GgufMeta::read(StdPath::new(&expanded)) {
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
    let path = StdPath::new(&expanded);
    if !path.is_dir() {
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
