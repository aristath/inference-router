use axum::body::Body;
use axum::extract::{Request, State};
use axum::http::{HeaderName, HeaderValue, Method, StatusCode};
use axum::response::{IntoResponse, Response};
use axum::Json;
use futures_util::Stream;
use serde_json::json;
use std::pin::Pin;
use std::task::{Context, Poll};
use tracing::{error, warn};

use crate::api::body_peek;
use crate::api::loop_guard::{self, StreamSession};
use crate::orchestrator::AppState;
use crate::process::manager::RequestGuard;

/// Hop-by-hop headers per RFC 7230 §6.1. Must be stripped when proxying.
const HOP_BY_HOP: &[&str] = &[
    "connection",
    "keep-alive",
    "proxy-authenticate",
    "proxy-authorization",
    "te",
    "trailer",
    "transfer-encoding",
    "upgrade",
];

/// Wraps a byte stream and keeps a `RequestGuard` alive until the stream
/// is exhausted or dropped. This ensures the instance's active counter
/// stays incremented for the full duration of a streaming response.
struct GuardedStream<S> {
    inner: S,
    _guard: RequestGuard,
}

impl<S: Stream + Unpin> Stream for GuardedStream<S> {
    type Item = <S as Stream>::Item;
    fn poll_next(
        mut self: Pin<&mut Self>,
        cx: &mut Context<'_>,
    ) -> Poll<Option<<S as Stream>::Item>> {
        Pin::new(&mut self.inner).poll_next(cx)
    }
}

/// Byte-level passthrough handler for `/v1/*`. Peeks `model` from the JSON
/// body, calls `ensure_loaded`, then proxies request/response unchanged.
pub async fn proxy_handler(State(state): State<AppState>, req: Request) -> Response {
    if req.method() != Method::POST {
        return (
            StatusCode::METHOD_NOT_ALLOWED,
            Json(json!({"error": "this path only supports POST with a JSON body containing 'model'"})),
        )
            .into_response();
    }

    let (parts, body) = req.into_parts();

    let body_bytes = match axum::body::to_bytes(body, state.max_body_bytes).await {
        Ok(b) => b,
        Err(e) => {
            return (
                StatusCode::PAYLOAD_TOO_LARGE,
                Json(json!({
                    "error": format!(
                        "request body exceeds max size of {} bytes or could not be read: {e}",
                        state.max_body_bytes,
                    ),
                })),
            )
                .into_response();
        }
    };

    let model_id = match body_peek::extract_model(&body_bytes) {
        Some(id) => id,
        None => {
            return (
                StatusCode::BAD_REQUEST,
                Json(json!({"error": "request body must be JSON with a 'model' field"})),
            )
                .into_response();
        }
    };

    let guard = match state.clone().ensure_loaded(&model_id).await {
        Ok(g) => g,
        Err(e) => {
            warn!(model = model_id, error = %e, "ensure_loaded failed");
            return (
                StatusCode::SERVICE_UNAVAILABLE,
                Json(json!({"error": e.to_string(), "model": model_id})),
            )
                .into_response();
        }
    };

    let port = guard.port;

    state.mark_used(&model_id).await;

    let path_and_query = parts
        .uri
        .path_and_query()
        .map(|p| p.as_str())
        .unwrap_or("/");
    let upstream_url = format!("http://127.0.0.1:{}{}", port, path_and_query);

    let client = reqwest::Client::new();
    let upstream_method = reqwest::Method::from_bytes(parts.method.as_str().as_bytes())
        .unwrap_or(reqwest::Method::POST);

    let settings = state.settings().await;
    let outbound_body =
        loop_guard::guard_request(parts.uri.path(), &body_bytes, &settings.loop_guards.tool)
            .unwrap_or_else(|| body_bytes.to_vec());

    if let Some(session) = StreamSession::new(
        client.clone(),
        upstream_method.clone(),
        upstream_url.clone(),
        parts.headers.clone(),
        parts.uri.path(),
        &outbound_body,
        &settings.loop_guards.streaming,
    ) {
        return session.into_response(guard).await;
    }

    let mut builder = client
        .request(upstream_method, &upstream_url)
        .body(outbound_body);

    for (name, value) in parts.headers.iter() {
        let lower = name.as_str().to_ascii_lowercase();
        if HOP_BY_HOP.contains(&lower.as_str()) || lower == "host" || lower == "content-length" {
            continue;
        }
        builder = builder.header(name.as_str(), value);
    }

    let upstream = match builder.send().await {
        Ok(r) => r,
        Err(e) => {
            error!(upstream = upstream_url, error = %e, "upstream request failed");
            return (
                StatusCode::BAD_GATEWAY,
                Json(json!({"error": format!("upstream error: {e}")})),
            )
                .into_response();
        }
    };

    let status = StatusCode::from_u16(upstream.status().as_u16())
        .unwrap_or(StatusCode::INTERNAL_SERVER_ERROR);

    let mut resp_builder = Response::builder().status(status);
    for (name, value) in upstream.headers().iter() {
        let lower = name.as_str().to_ascii_lowercase();
        if HOP_BY_HOP.contains(&lower.as_str()) || lower == "content-length" {
            continue;
        }
        if let (Ok(hn), Ok(hv)) = (
            HeaderName::from_bytes(name.as_str().as_bytes()),
            HeaderValue::from_bytes(value.as_bytes()),
        ) {
            resp_builder = resp_builder.header(hn, hv);
        }
    }

    // Wrap the byte stream so the guard stays alive until the body is fully
    // consumed or the connection is dropped.
    let stream = GuardedStream {
        inner: upstream.bytes_stream(),
        _guard: guard,
    };
    match resp_builder.body(Body::from_stream(stream)) {
        Ok(r) => r,
        Err(e) => {
            error!(error = %e, "failed to build proxy response");
            (
                StatusCode::INTERNAL_SERVER_ERROR,
                "proxy response build failed",
            )
                .into_response()
        }
    }
}

/// Synthesized OpenAI-style model list from the current config.
///
/// Drafts are filtered out: they have no standalone process and shouldn't
/// appear as selectable models to OpenAI clients.
pub async fn list_v1_models(State(state): State<AppState>) -> impl IntoResponse {
    let models = state.list_models().await;
    let data: Vec<_> = models
        .into_iter()
        .map(|m| {
            json!({
                "id": m.id,
                "object": "model",
                "created": 0,
                "owned_by": "local",
            })
        })
        .collect();
    Json(json!({"object": "list", "data": data}))
}

pub async fn healthz() -> impl IntoResponse {
    (StatusCode::OK, "ok")
}
