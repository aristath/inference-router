use axum::extract::{Path, Query, State};
use axum::http::StatusCode;
use axum::response::IntoResponse;
use axum::Json;
use serde::Deserialize;
use std::path::{Path as StdPath, PathBuf};

use crate::config::{BinaryPreset, ModelConfig};
use crate::orchestrator::{AppState, LoadError, MutationError, StopError};
use crate::vram::estimator::GgufInfo;

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
        MutationError::PortConflict(port, holder) => (
            StatusCode::CONFLICT,
            Json(serde_json::json!({
                "error": format!("port {port} is already in use by model '{holder}'"),
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
// Both endpoints accept arbitrary paths from the browser. The dashboard is
// localhost-only and single-user, but the endpoints still sandbox to `$HOME`
// so a stray script or browser redirection can't list `/etc`, `/root`, etc.

fn home_root() -> Option<PathBuf> {
    std::env::var_os("HOME")
        .map(PathBuf::from)
        .and_then(|p| std::fs::canonicalize(p).ok())
}

/// Resolve `raw` against `$HOME` and return the canonicalized path only if
/// it lives under `$HOME`. Returns `None` on any form of escape (symlinks,
/// `..`, absolute paths pointing outside the sandbox).
fn sandboxed(raw: &str) -> Option<PathBuf> {
    let expanded = shellexpand::tilde(raw).to_string();
    let resolved = std::fs::canonicalize(StdPath::new(&expanded)).ok()?;
    let home = home_root()?;
    resolved.starts_with(&home).then_some(resolved)
}

#[derive(Deserialize)]
pub struct GgufInfoQuery {
    pub path: String,
}

/// Reads GGUF metadata for a file on disk. Used by the model form to drive
/// the context slider's upper bound and the live VRAM preview.
pub async fn gguf_info(Query(q): Query<GgufInfoQuery>) -> impl IntoResponse {
    let Some(path) = sandboxed(&q.path) else {
        return (
            StatusCode::FORBIDDEN,
            Json(serde_json::json!({"error": "path must be inside $HOME"})),
        )
            .into_response();
    };
    match GgufInfo::read(&path) {
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
    let Some(path) = sandboxed(&req.path) else {
        return (
            StatusCode::FORBIDDEN,
            Json(serde_json::json!({"error": "path must be inside $HOME"})),
        )
            .into_response();
    };
    if !path.is_dir() {
        return (
            StatusCode::NOT_FOUND,
            Json(serde_json::json!({"error": "not a directory"})),
        )
            .into_response();
    }

    let mut entries = Vec::new();
    if let Ok(read_dir) = std::fs::read_dir(&path) {
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
