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

use crate::api::loop_guard::{self, StreamSession};
use crate::api::routing_body::RoutingBody;
use crate::orchestrator::request_queue::{Priority, RequestPermit, PRIORITY_HEADER};
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
    vllm_perf: Option<VllmPerfCollection>,
}

struct VllmPerfCollection {
    state: AppState,
    model_id: String,
    pid: i32,
    port: u16,
    generation: u64,
}

impl VllmPerfCollection {
    fn spawn(self, guard: RequestGuard) {
        crate::api::perf::spawn_vllm_perf_collection(
            self.state,
            self.model_id,
            self.pid,
            self.port,
            self.generation,
            guard,
        );
    }
}

impl<S: Stream + Unpin> Stream for GuardedStream<S> {
    type Item = <S as Stream>::Item;
    fn poll_next(
        mut self: Pin<&mut Self>,
        cx: &mut Context<'_>,
    ) -> Poll<Option<<S as Stream>::Item>> {
        let result = Pin::new(&mut self.inner).poll_next(cx);
        if matches!(result, Poll::Ready(None)) {
            if let Some(collection) = self.vllm_perf.take() {
                collection.spawn(self._guard.retain());
            }
        }
        result
    }
}

/// Loud, actionable error returned when an instance was found unresponsive and
/// recycled, so the client shows "restarting, retry" instead of hanging or a
/// generic gateway error.
fn unresponsive_response(model_id: &str) -> Response {
    (
        StatusCode::SERVICE_UNAVAILABLE,
        Json(json!({
            "error": format!("model '{model_id}' was unresponsive and is being restarted; please retry"),
            "model": model_id,
            "retry": true,
        })),
    )
        .into_response()
}

fn rewrite_model_field(body: &[u8], model_id: &str) -> Result<Vec<u8>, serde_json::Error> {
    let mut doc: serde_json::Value = serde_json::from_slice(body)?;
    if let Some(model) = doc.get_mut("model") {
        *model = serde_json::Value::String(model_id.to_owned());
    }
    // Select only after routing: an alias may have been sent to its fallback.
    if let Some(overrides) = doc
        .as_object_mut()
        .and_then(|o| o.remove("chat_template_overrides"))
    {
        if let Some(template) = overrides.get(model_id).and_then(|v| v.as_str()) {
            doc["chat_template"] = serde_json::Value::String(template.to_owned());
        }
    }
    serde_json::to_vec(&doc)
}

/// Byte-level passthrough handler for `/v1/*`. Peeks `model` from the JSON
/// body, calls `ensure_loaded`, then proxies request/response. Alias requests
/// are forwarded with the resolved model id because upstream OpenAI-compatible
/// servers validate the `model` field themselves.
pub async fn proxy_handler(State(state): State<AppState>, req: Request) -> Response {
    if req.method() != Method::POST {
        return (
            StatusCode::METHOD_NOT_ALLOWED,
            Json(json!({"error": "this path only supports POST with a JSON body containing 'model'"})),
        )
            .into_response();
    }

    let (mut parts, body) = req.into_parts();
    let priority = Priority::from_header(
        parts
            .headers
            .remove(PRIORITY_HEADER)
            .as_ref()
            .and_then(|v| v.to_str().ok()),
    );
    let directory = state
        .store
        .path()
        .parent()
        .unwrap_or(std::path::Path::new("."))
        .join("request-spool");
    let body = match RoutingBody::read(body, state.max_body_bytes, Some(directory.as_path())).await
    {
        Ok(body) => body,
        Err(error) => {
            return (StatusCode::PAYLOAD_TOO_LARGE, Json(json!({"error":error}))).into_response()
        }
    };

    let requested_model = match body.model() {
        Some(id) => id,
        None => {
            return (
                StatusCode::BAD_REQUEST,
                Json(json!({"error": "request body must be JSON with a 'model' field"})),
            )
                .into_response();
        }
    };

    // Resolve aliases to their target model. Non-alias names pass through
    // unchanged.
    let media_blocked = state.gpu_gate.try_inference().is_none();
    let (model_id, independent) = state
        .inference_target(&requested_model, media_blocked)
        .await;

    // A defined-but-unassigned alias resolves to an empty target. Real model
    // ids are never empty, so this unambiguously means "alias not pointed at a
    // model yet" — surface a clear error instead of a confusing "not found".
    if model_id.is_empty() {
        return (
            StatusCode::SERVICE_UNAVAILABLE,
            Json(json!({
                "error": format!("alias '{requested_model}' is not assigned to a model"),
                "model": requested_model,
            })),
        )
            .into_response();
    }

    let slots = state
        .data
        .lock()
        .await
        .models
        .get(&model_id)
        .and_then(|model| model.parallel_slots)
        .unwrap_or(1)
        .max(1) as usize;
    let replicas = state
        .process_manager
        .lock()
        .await
        .instance_count(&model_id)
        .max(1);
    let admission = match state
        .request_queue
        .acquire(
            &model_id,
            priority,
            slots.saturating_mul(replicas),
            (!independent).then(|| state.gpu_gate.clone()),
        )
        .await
    {
        Ok(permit) => permit,
        Err(error) => {
            return (
                StatusCode::SERVICE_UNAVAILABLE,
                Json(json!({"error":error})),
            )
                .into_response()
        }
    };
    let response = proxy_admitted(
        state,
        parts,
        body,
        requested_model,
        model_id,
        independent,
        &admission,
    )
    .await;
    response.map(|body| {
        Body::from_stream(AdmittedStream {
            inner: body.into_data_stream(),
            permit: Some(admission),
        })
    })
}

/// The queue's permit follows the complete response body, including disconnects.
struct AdmittedStream<S> {
    inner: S,
    permit: Option<RequestPermit>,
}
impl<S: Stream + Unpin> Stream for AdmittedStream<S> {
    type Item = S::Item;
    fn poll_next(mut self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Option<Self::Item>> {
        let next = Pin::new(&mut self.inner).poll_next(cx);
        if matches!(next, Poll::Ready(None)) {
            self.permit.take();
        }
        next
    }
}

async fn proxy_admitted(
    state: AppState,
    parts: axum::http::request::Parts,
    body: RoutingBody,
    requested_model: String,
    model_id: String,
    independent: bool,
    admission: &RequestPermit,
) -> Response {
    let loaded = if independent {
        state.clone().ensure_loaded_independent(&model_id).await
    } else {
        state
            .clone()
            .ensure_loaded_admitted(&model_id, admission.gpu.clone())
            .await
    };
    let guard = match loaded {
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
    let body_bytes = match body.into_bytes().await {
        Ok(bytes) => bytes,
        Err(error) => {
            return (
                StatusCode::BAD_REQUEST,
                Json(json!({"error":error.to_string()})),
            )
                .into_response()
        }
    };

    let port = guard.port;
    let pid = guard.pid;
    let vllm_perf_generation = state.vllm_perf_generation(&model_id).await;

    state.mark_used(&model_id).await;

    let path_and_query = parts
        .uri
        .path_and_query()
        .map(|p| p.as_str())
        .unwrap_or("/");
    let upstream_url = format!("http://127.0.0.1:{}{}", port, path_and_query);

    let settings = state.settings().await;

    // Idle-timeout backstop: `read_timeout` fires when the upstream produces no
    // bytes (no response headers, or no further stream chunk) for this long — the
    // signature of a wedged instance. Long by design; the watchdog and liveness
    // probe catch the common cases far sooner. The same client is cloned into the
    // streaming path, so both non-streaming and SSE inherit the backstop.
    let model_timeout = state
        .data
        .lock()
        .await
        .models
        .get(&model_id)
        .and_then(|model| model.request_idle_timeout_secs);
    let idle_timeout = proxy_idle_timeout(
        settings.watchdog.enabled,
        settings.watchdog.idle_timeout_secs,
        model_timeout,
    );
    let client = {
        let mut b = reqwest::Client::builder();
        if let Some(d) = idle_timeout {
            b = b.read_timeout(d);
        }
        b.build().unwrap_or_else(|_| reqwest::Client::new())
    };

    let upstream_method = reqwest::Method::from_bytes(parts.method.as_str().as_bytes())
        .unwrap_or(reqwest::Method::POST);

    let routed_body = if requested_model == model_id
        && !body_bytes
            .windows(25)
            .any(|w| w == b"\"chat_template_overrides\"")
    {
        body_bytes.to_vec()
    } else {
        match rewrite_model_field(&body_bytes, &model_id) {
            Ok(body) => body,
            Err(e) => {
                return (
                    StatusCode::BAD_REQUEST,
                    Json(json!({"error": format!("request body could not be rewritten with resolved model id: {e}")})),
                )
                    .into_response();
            }
        }
    };

    let outbound_body =
        loop_guard::guard_request(parts.uri.path(), &routed_body, &settings.loop_guards.tool)
            .unwrap_or(routed_body);

    if let Some(session) = StreamSession::new(
        client.clone(),
        upstream_method.clone(),
        upstream_url.clone(),
        parts.headers.clone(),
        parts.uri.path(),
        &outbound_body,
        &settings.loop_guards.streaming,
        state.clone(),
        model_id.clone(),
        vllm_perf_generation,
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
        Err(e) if e.is_timeout() => {
            warn!(
                upstream = upstream_url,
                "no response before idle timeout; recycling wedged instance"
            );
            let pid = guard.pid;
            drop(guard);
            state
                .recycle_instance(pid, "idle timeout before response headers")
                .await;
            return unresponsive_response(&model_id);
        }
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

    // A non-streaming response is a single JSON document carrying the `timings`
    // block — buffer it, fold the throughput into the model's average, then
    // hand the buffered body back. Streaming (SSE) responses can't be buffered
    // here; their stats are captured in the StreamSession tap instead.
    let is_sse = upstream
        .headers()
        .get(reqwest::header::CONTENT_TYPE)
        .and_then(|v| v.to_str().ok())
        .map(|v| v.contains("text/event-stream"))
        .unwrap_or(false);

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

    if !is_sse {
        let body_bytes = match upstream.bytes().await {
            Ok(b) => b,
            Err(e) if e.is_timeout() => {
                warn!(
                    upstream = upstream_url,
                    "stalled mid-body before idle timeout; recycling wedged instance"
                );
                let pid = guard.pid;
                drop(guard);
                state
                    .recycle_instance(pid, "idle timeout mid-response")
                    .await;
                return unresponsive_response(&model_id);
            }
            Err(e) => {
                error!(upstream = upstream_url, error = %e, "reading upstream body failed");
                return (
                    StatusCode::BAD_GATEWAY,
                    Json(json!({"error": format!("upstream body error: {e}")})),
                )
                    .into_response();
            }
        };
        if status == StatusCode::OK {
            if let Ok(v) = serde_json::from_slice::<serde_json::Value>(&body_bytes) {
                if let Some((decode, prefill)) = crate::config::timings_from_json(&v) {
                    state.record_perf(&model_id, decode, prefill);
                }
            }
        }
        if let Some(generation) = vllm_perf_generation {
            VllmPerfCollection {
                state: state.clone(),
                model_id: model_id.clone(),
                pid,
                port,
                generation,
            }
            .spawn(guard.retain());
        }
        // The whole response is materialized, so the request is complete here.
        drop(guard);
        return match resp_builder.body(Body::from(body_bytes)) {
            Ok(r) => r,
            Err(e) => {
                error!(error = %e, "failed to build proxy response");
                (
                    StatusCode::INTERNAL_SERVER_ERROR,
                    "proxy response build failed",
                )
                    .into_response()
            }
        };
    }

    // Wrap the byte stream so the guard stays alive until the body is fully
    // consumed or the connection is dropped.
    let stream = GuardedStream {
        inner: upstream.bytes_stream(),
        _guard: guard,
        vllm_perf: vllm_perf_generation.map(|generation| VllmPerfCollection {
            state: state.clone(),
            model_id: model_id.clone(),
            pid,
            port,
            generation,
        }),
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
/// The exposure mode controls which names are advertised:
/// - `FullList` (default): every configured model *and* every alias.
/// - `AliasesOnly`: only the defined aliases.
///
/// Aliases always resolve at request time regardless of this setting.
pub async fn list_v1_models(State(state): State<AppState>) -> impl IntoResponse {
    use crate::config::ModelExposure;

    fn entry(id: &str) -> serde_json::Value {
        json!({
            "id": id,
            "object": "model",
            "created": 0,
            "owned_by": "local",
        })
    }

    let exposure = state.settings().await.model_exposure;
    let mut data: Vec<serde_json::Value> = Vec::new();

    if exposure == ModelExposure::FullList {
        for m in state.list_models().await {
            data.push(entry(&m.id));
        }
    }
    for a in state.list_aliases().await {
        data.push(entry(&a.alias));
    }

    Json(json!({"object": "list", "data": data}))
}

pub async fn healthz() -> impl IntoResponse {
    (StatusCode::OK, "ok")
}

fn proxy_idle_timeout(
    enabled: bool,
    global: u64,
    model: Option<u64>,
) -> Option<std::time::Duration> {
    enabled
        .then_some(model.unwrap_or(global))
        .filter(|seconds| *seconds > 0)
        .map(std::time::Duration::from_secs)
}

#[cfg(test)]
mod idle_timeout_tests {
    use super::proxy_idle_timeout;
    use std::time::Duration;
    #[test]
    fn templates_follow_the_resolved_model_including_fallbacks() {
        let body = br#"{"model":"default","messages":[],"chat_template_overrides":{"gpu":"GPU template","cpu":"CPU template"}}"#;
        for (target, expected) in [
            ("gpu", Some("GPU template")),
            ("cpu", Some("CPU template")),
            ("other", None),
        ] {
            let bytes = super::rewrite_model_field(body, target).unwrap();
            let doc: serde_json::Value = serde_json::from_slice(&bytes).unwrap();
            assert_eq!(doc["model"], target);
            assert_eq!(doc["chat_template"].as_str(), expected);
            assert!(doc.get("chat_template_overrides").is_none());
            assert_eq!(doc["messages"], serde_json::json!([]));
        }
    }
    #[test]
    fn model_timeout_override_preserves_other_models_and_watchdogs() {
        assert_eq!(
            proxy_idle_timeout(true, 900, None),
            Some(Duration::from_secs(900))
        );
        assert_eq!(proxy_idle_timeout(true, 900, Some(0)), None);
        assert_eq!(
            proxy_idle_timeout(true, 900, Some(2400)),
            Some(Duration::from_secs(2400))
        );
        assert_eq!(proxy_idle_timeout(false, 900, Some(2400)), None);
    }
}
