use crate::orchestrator::AppState;
use axum::extract::{Path, State};
use axum::http::StatusCode;
use axum::response::{IntoResponse, Response};
use axum::Json;
use serde_json::{json, Value};
use tokio_util::sync::CancellationToken;

pub async fn status(State(state): State<AppState>) -> impl IntoResponse {
    Json(state.gpu_gate.snapshot())
}

pub async fn run(
    State(state): State<AppState>,
    Path(name): Path<String>,
    Json(input): Json<Value>,
) -> Response {
    let worker = match state.media.worker(&name) {
        Ok(worker) => worker,
        Err(error) => return (StatusCode::NOT_FOUND, Json(json!({"error":error}))).into_response(),
    };
    let cancelled = CancellationToken::new();
    let _cancel_on_disconnect = cancelled.clone().drop_guard();
    // Do not drop cleanup/restoration if the caller disappears while awaiting it.
    let controller = state.media.clone();
    let task = tokio::spawn(controller.execute(state, worker, name, input, cancelled));
    match task.await {
        Ok(Ok(result)) => Json(result).into_response(),
        Ok(Err(error)) => (
            StatusCode::SERVICE_UNAVAILABLE,
            Json(json!({"error":error})),
        )
            .into_response(),
        Err(error) => (
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(json!({"error":format!("media supervision failed: {error}")})),
        )
            .into_response(),
    }
}

pub async fn recover(State(state): State<AppState>) -> Response {
    let controller = state.media.clone();
    match tokio::spawn(async move { controller.recover(state).await }).await {
        Ok(Ok(())) => Json(json!({"ok":true})).into_response(),
        Ok(Err(error)) => (
            StatusCode::SERVICE_UNAVAILABLE,
            Json(json!({"error":error})),
        )
            .into_response(),
        Err(error) => (
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(json!({"error":error.to_string()})),
        )
            .into_response(),
    }
}
