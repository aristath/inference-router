use axum::extract::{Path, Query, State};
use axum::http::StatusCode;
use axum::response::IntoResponse;
use axum::Json;
use serde::{Deserialize, Serialize};
use std::path::Path as StdPath;
use std::time::Duration;
use tokio::process::Command;

use crate::config::{AppSettings, BinaryPreset, CacheType, ModelAlias, ModelConfig, WeightsFormat};
use crate::orchestrator::{AppState, LoadError, MutationError, StopError};
use crate::vram::estimator::GgufMeta;

pub async fn list_models(State(state): State<AppState>) -> impl IntoResponse {
    Json(state.list_models().await)
}

pub async fn get_settings(State(state): State<AppState>) -> impl IntoResponse {
    Json(state.settings().await)
}

pub async fn update_settings(
    State(state): State<AppState>,
    Json(settings): Json<AppSettings>,
) -> impl IntoResponse {
    state.update_settings(settings).await;
    (StatusCode::OK, Json(serde_json::json!({"ok": true})))
}

/// Response from model validation endpoint (`/api/models/validate`).
/// 
/// Returned when validating a model configuration before saving.
#[derive(Serialize)]
struct ValidationResponse {
    /// Whether the configuration is valid
    valid: bool,
    /// Validation errors (empty if valid)
    errors: Vec<String>,
    /// Non-blocking warnings
    warnings: Vec<String>,
}

pub async fn validate_model(
    State(state): State<AppState>,
    Json(model): Json<ModelConfig>,
) -> impl IntoResponse {
    let presets = state.list_presets().await;
    let gpus = state.list_gpus().await;
    let mut errors = Vec::new();
    let mut warnings = Vec::new();

    let binary = if let Some(preset_id) = model.binary_preset.as_deref() {
        match presets.iter().find(|p| p.id == preset_id) {
            Some(p) => Some(p.binary.as_path()),
            None => {
                errors.push(format!("binary preset '{preset_id}' does not exist"));
                None
            }
        }
    } else {
        Some(model.binary.as_path())
    };

    if let Some(path) = binary {
        validate_binary_path(path, &mut errors);
    }
    validate_model_path(&model, &mut errors);
    validate_tensor_split(&model, gpus.len(), &mut errors, &mut warnings);
    validate_context(&model, &mut warnings);
    validate_cache_quantization(&model, &mut warnings);

    let status = if errors.is_empty() {
        StatusCode::OK
    } else {
        StatusCode::UNPROCESSABLE_ENTITY
    };
    (
        status,
        Json(ValidationResponse {
            valid: errors.is_empty(),
            errors,
            warnings,
        }),
    )
        .into_response()
}

/// Converts mutation errors into appropriate HTTP responses.
/// 
/// # Purpose
/// Centralizes error handling for all CRUD operations to ensure consistent
/// error responses across the API.
/// 
/// # Error Mapping
/// | Error Type | HTTP Status | Response Body |
/// |------------|-------------|----------------|
/// | NotFound   | 404          | {"error": "'id' not found"} |
/// | Conflict   | 409          | {"error": "'id' already exists"} |
/// | InvalidConfig | 422      | {"error": "validation message"} |
/// 
/// # Used by
/// All model/preset create/update/delete handlers
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
        MutationError::AliasConflict(alias) => (
            StatusCode::CONFLICT,
            Json(serde_json::json!({"error": format!("alias '{alias}' already exists")})),
        )
            .into_response(),
        MutationError::AliasNotFound(alias) => (
            StatusCode::NOT_FOUND,
            Json(serde_json::json!({"error": format!("alias '{alias}' not found")})),
        )
            .into_response(),
        e @ (MutationError::AliasShadowsModel(_)
        | MutationError::AliasTargetMissing { .. }
        | MutationError::AliasCycle { .. }
        | MutationError::AliasInvalid(_)) => (
            StatusCode::UNPROCESSABLE_ENTITY,
            Json(serde_json::json!({"error": e.to_string()})),
        )
            .into_response(),
    }
}

fn validate_binary_path(path: &StdPath, errors: &mut Vec<String>) {
    if path.as_os_str().is_empty() {
        errors.push("binary path is empty".into());
        return;
    }
    let Ok(meta) = std::fs::metadata(path) else {
        errors.push(format!("binary '{}' does not exist", path.display()));
        return;
    };
    if !meta.is_file() {
        errors.push(format!("binary '{}' is not a file", path.display()));
    }
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        if meta.permissions().mode() & 0o111 == 0 {
            errors.push(format!("binary '{}' is not executable", path.display()));
        }
    }
}

fn validate_model_path(model: &ModelConfig, errors: &mut Vec<String>) {
    if model.model_path.as_os_str().is_empty() {
        errors.push("model path is empty".into());
        return;
    }
    let Ok(meta) = std::fs::metadata(&model.model_path) else {
        errors.push(format!(
            "model path '{}' does not exist",
            model.model_path.display()
        ));
        return;
    };
    match model.weights_format {
        WeightsFormat::Gguf if !meta.is_file() => {
            errors.push(format!(
                "GGUF model path '{}' is not a file",
                model.model_path.display()
            ));
        }
        WeightsFormat::Safetensors if !meta.is_dir() => {
            errors.push(format!(
                "safetensors model path '{}' is not a directory",
                model.model_path.display(),
            ));
        }
        _ => {}
    }
}

fn validate_tensor_split(
    model: &ModelConfig,
    gpu_count: usize,
    errors: &mut Vec<String>,
    warnings: &mut Vec<String>,
) {
    let Some(split) = model.tensor_split.as_deref() else {
        return;
    };
    let parts: Vec<&str> = split.split(',').map(str::trim).collect();
    if gpu_count > 0 && parts.len() != gpu_count {
        warnings.push(format!(
            "tensor_split has {} entries but {} GPUs were detected",
            parts.len(),
            gpu_count,
        ));
    }
    let mut non_zero = false;
    for part in parts {
        match part.parse::<f32>() {
            Ok(v) if v > 0.0 => non_zero = true,
            Ok(_) => {}
            Err(_) => errors.push(format!("tensor_split entry '{part}' is not a number")),
        }
    }
    if !non_zero {
        errors.push("tensor_split must contain at least one positive entry".into());
    }
}

fn validate_context(model: &ModelConfig, warnings: &mut Vec<String>) {
    if let Some(meta) = &model.gguf_meta {
        if model.context > meta.max_context {
            warnings.push(format!(
                "context {} exceeds GGUF metadata max_context {}",
                model.context, meta.max_context,
            ));
        }
    }
}

fn validate_cache_quantization(model: &ModelConfig, warnings: &mut Vec<String>) {
    let quantized = matches!(model.cache_type_k, Some(CacheType::Q8_0 | CacheType::Q4_0))
        || matches!(model.cache_type_v, Some(CacheType::Q8_0 | CacheType::Q4_0));
    if quantized && !model.flash_attn {
        warnings.push("KV cache quantization usually requires flash attention".into());
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

// ===== Model aliases =====

pub async fn list_aliases(State(state): State<AppState>) -> impl IntoResponse {
    Json(state.list_aliases().await)
}

pub async fn create_alias(
    State(state): State<AppState>,
    Json(alias): Json<ModelAlias>,
) -> impl IntoResponse {
    let name = alias.alias.clone();
    match state.add_alias(alias).await {
        Ok(()) => (StatusCode::CREATED, Json(serde_json::json!({"alias": name}))).into_response(),
        Err(e) => mutation_response(e),
    }
}

pub async fn update_alias(
    State(state): State<AppState>,
    Path(alias_name): Path<String>,
    Json(alias): Json<ModelAlias>,
) -> impl IntoResponse {
    if alias.alias != alias_name {
        return (
            StatusCode::BAD_REQUEST,
            Json(serde_json::json!({"error": "alias in URL and body must match"})),
        )
            .into_response();
    }
    match state.update_alias(alias).await {
        Ok(()) => (StatusCode::OK, Json(serde_json::json!({"ok": true}))).into_response(),
        Err(e) => mutation_response(e),
    }
}

pub async fn delete_alias(
    State(state): State<AppState>,
    Path(alias_name): Path<String>,
) -> impl IntoResponse {
    match state.remove_alias(&alias_name).await {
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

pub async fn restart_service() -> impl IntoResponse {
    tokio::spawn(async {
        tokio::time::sleep(Duration::from_millis(250)).await;
        match Command::new("systemctl")
            .args(["--user", "restart", "inference-router.service"])
            .status()
            .await
        {
            Ok(status) if status.success() => {}
            Ok(status) => {
                tracing::error!(?status, "failed to restart inference-router.service");
            }
            Err(error) => {
                tracing::error!(%error, "failed to spawn systemctl restart");
            }
        }
    });

    (
        StatusCode::ACCEPTED,
        Json(serde_json::json!({"ok": true, "message": "restart scheduled"})),
    )
        .into_response()
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
            _ => a["name"]
                .as_str()
                .unwrap_or("")
                .cmp(b["name"].as_str().unwrap_or("")),
        }
    });
    Json(entries).into_response()
}
