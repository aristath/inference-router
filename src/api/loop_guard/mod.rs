//! # Loop Guard System
//!
//! Detects and corrects infinite generation loops in streaming responses and cross-turn tool
//! cycles. The system has two independent subsystems:
//!
//! 1. **Streaming Loop Guard** — Monitors SSE chunks for repeated text patterns in real-time.
//!    - Uses a Z-function based tandem repeat detector with configurable window size
//!    - Can heal (replay partial output + inject corrective prompt), abort, or just log
//!    - Supports OpenAI-style chat/completions and llama.cpp's native `/completion` endpoints
//!
//! 2. **Cross-Turn Tool Loop Guard** — Detects repeated tool call sequences across multiple
//!    turns in chat histories.
//!    - Analyzes the last N messages for repeating tool → result → tool patterns
//!    - Injects a corrective user message when detected
//!    - Handles both OpenAI tool_calls and legacy function_call formats
//!
//! Both systems are opt-in via `AppSettings` and can be configured independently.
//!
//! ## Key Components
//! - `StreamSession`: Manages a single streaming request with loop detection
//! - `cross_turn`: Detects tool cycles in message histories
//! - `detector`: Z-function based repeat detection algorithm
//! - `sse`: Parses streaming chunks from OpenAI/llama.cpp formats
//! - `endpoint`: Handles endpoint-specific payload formats and corrective injection

mod cross_turn;
mod detector;
mod endpoint;
mod sse;

use std::collections::HashMap;
use std::io;
use std::pin::Pin;
use std::task::{Context, Poll};
use std::time::Duration;

use axum::body::{Body, Bytes};
use axum::http::{HeaderMap, HeaderName, HeaderValue, StatusCode};
use axum::response::{IntoResponse, Response};
use futures_util::{Stream, StreamExt};
use serde_json::{json, Value};
use tokio::sync::mpsc;
use tokio::time::{interval_at, Instant};
use tracing::{error, warn};

use crate::config::{StreamingLoopAction, StreamingLoopSettings, ToolLoopSettings};
use crate::orchestrator::engine::AppState;
use crate::process::manager::RequestGuard;

use self::detector::Detector;
use self::endpoint::{ChoiceSnapshot, EndpointKind};
use self::sse::EventParser;

pub(super) fn guard_request(path: &str, body: &[u8], cfg: &ToolLoopSettings) -> Option<Vec<u8>> {
    cross_turn::guard_request(path, body, cfg)
}

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

#[derive(Clone, Debug)]
struct Config {
    enabled: bool,
    window: usize,
    repeats: usize,
    check_interval: Duration,
    max_retries: usize,
    action: StreamingLoopAction,
    replay_partial: bool,
}

impl Default for Config {
    fn default() -> Self {
        Self {
            enabled: true,
            window: 65_536,
            repeats: 10,
            check_interval: Duration::from_secs(5),
            max_retries: 3,
            action: StreamingLoopAction::Heal,
            replay_partial: true,
        }
    }
}

impl Config {
    fn from_settings(settings: &StreamingLoopSettings) -> Self {
        Self {
            enabled: settings.enabled,
            window: settings.window_bytes.max(1024),
            repeats: settings.repeats.max(2),
            check_interval: Duration::from_millis(settings.check_interval_ms.max(1)),
            max_retries: settings.max_retries,
            action: settings.action,
            replay_partial: settings.replay_partial,
        }
    }
}

pub(super) struct StreamSession {
    cfg: Config,
    client: reqwest::Client,
    method: reqwest::Method,
    upstream_url: String,
    headers: HeaderMap,
    req_doc: Value,
    spec: EndpointKind,
    choices: HashMap<i64, ChoiceState>,
    /// For folding the response `timings` into the model's throughput average.
    state: AppState,
    model_id: String,
    /// PID of the instance serving this request, so a mid-stream stall (idle
    /// timeout) can recycle exactly that process. Set in `into_response` from
    /// the request guard.
    pid: i32,
    /// One perf record per request — the timings ride in the final SSE event.
    perf_recorded: bool,
}

impl StreamSession {
    #[allow(clippy::too_many_arguments)]
    pub(super) fn new(
        client: reqwest::Client,
        method: reqwest::Method,
        upstream_url: String,
        headers: HeaderMap,
        path: &str,
        body: &[u8],
        cfg: &StreamingLoopSettings,
        state: AppState,
        model_id: String,
    ) -> Option<Self> {
        let cfg = Config::from_settings(cfg);
        if !cfg.enabled {
            return None;
        }
        let spec = EndpointKind::detect(path)?;
        let req_doc: Value = serde_json::from_slice(body).ok()?;
        let streaming = req_doc
            .get("stream")
            .and_then(Value::as_bool)
            .unwrap_or(false);
        if !streaming {
            return None;
        }
        Some(Self {
            cfg,
            client,
            method,
            upstream_url,
            headers,
            req_doc,
            spec,
            choices: HashMap::new(),
            state,
            model_id,
            pid: 0,
            perf_recorded: false,
        })
    }

    /// Fold this SSE event's `timings` (if any) into the model's running
    /// throughput average. Only the final event carries timings, so this records
    /// at most once per request.
    fn note_timings(&mut self, payload: &[u8]) {
        if self.perf_recorded {
            return;
        }
        if let Ok(v) = serde_json::from_slice::<Value>(payload) {
            if let Some((decode, prefill)) = crate::config::timings_from_json(&v) {
                self.state.record_perf(&self.model_id, decode, prefill);
                self.perf_recorded = true;
            }
        }
    }

    pub(super) async fn into_response(mut self, guard: RequestGuard) -> Response {
        self.pid = guard.pid;
        let first = match self.send_upstream().await {
            Ok(response) => response,
            Err(e) if e.is_timeout() => {
                warn!(
                    upstream = self.upstream_url,
                    "no response before idle timeout; recycling wedged instance"
                );
                let pid = guard.pid;
                drop(guard);
                self.state
                    .recycle_instance(pid, "idle timeout before response headers")
                    .await;
                return (
                    StatusCode::SERVICE_UNAVAILABLE,
                    axum::Json(json!({
                        "error": format!("model '{}' was unresponsive and is being restarted; please retry", self.model_id),
                        "model": self.model_id,
                        "retry": true,
                    })),
                )
                    .into_response();
            }
            Err(e) => {
                error!(upstream = self.upstream_url, error = %e, "upstream request failed");
                return (
                    StatusCode::BAD_GATEWAY,
                    axum::Json(json!({"error": format!("upstream error: {e}")})),
                )
                    .into_response();
            }
        };

        let status = StatusCode::from_u16(first.status().as_u16())
            .unwrap_or(StatusCode::INTERNAL_SERVER_ERROR);
        let (tx, rx) = mpsc::channel::<Result<Bytes, io::Error>>(16);
        let response = response_from_upstream(status, first.headers(), rx);

        if status != StatusCode::OK {
            tokio::spawn(async move {
                let _guard = guard;
                forward_response_body(first, tx).await;
            });
            return response;
        }

        tokio::spawn(async move {
            let _guard = guard;
            self.run(first, tx).await;
        });
        response
    }

    async fn run(&mut self, first: reqwest::Response, tx: mpsc::Sender<Result<Bytes, io::Error>>) {
        let mut first = Some(first);
        for attempt in 0usize.. {
            if tx.is_closed() {
                return;
            }

            let upstream = if let Some(response) = first.take() {
                response
            } else {
                match self.send_upstream().await {
                    Ok(response) => response,
                    Err(e) => {
                        error!(
                            endpoint = self.spec.name(),
                            attempt,
                            error = %e,
                            "loop-guard upstream request failed",
                        );
                        return;
                    }
                }
            };

            match self.attempt_once(upstream, &tx).await {
                Outcome::FinishedNaturally => return,
                Outcome::ClientDisconnected => return,
                Outcome::IdleTimeout => {
                    warn!(
                        endpoint = self.spec.name(),
                        attempt, "stream stalled past idle timeout; recycling wedged instance",
                    );
                    self.state
                        .recycle_instance(self.pid, "idle timeout mid-stream")
                        .await;
                    self.write_stream_error(
                        &tx,
                        "model was unresponsive and is being restarted; please retry",
                    )
                    .await;
                    return;
                }
                Outcome::UpstreamError(e) => {
                    error!(
                        endpoint = self.spec.name(),
                        attempt,
                        error = %e,
                        "loop-guard upstream error",
                    );
                    return;
                }
                Outcome::LoopDetected {
                    choice_index,
                    period,
                    snippet,
                } => {
                    warn!(
                        endpoint = self.spec.name(),
                        attempt,
                        choice = choice_index,
                        period,
                        snippet = %truncate(&snippet, 120),
                        "loop detected",
                    );

                    match self.cfg.action {
                        StreamingLoopAction::Log | StreamingLoopAction::Abort => {
                            self.write_final_halt_delta(&tx, attempt + 1, period).await;
                            return;
                        }
                        StreamingLoopAction::Heal => {
                            if attempt >= self.cfg.max_retries {
                                warn!(
                                    endpoint = self.spec.name(),
                                    retries = self.cfg.max_retries,
                                    "loop-guard exhausted retries; halting",
                                );
                                self.write_final_halt_delta(&tx, attempt + 1, period).await;
                                return;
                            }
                            let partials = self.build_partials(choice_index, period);
                            self.spec.inject_corrective(
                                &mut self.req_doc,
                                attempt,
                                partials.as_ref(),
                            );
                            self.reset_choices();
                        }
                    }
                }
            }
        }
    }

    async fn send_upstream(&self) -> Result<reqwest::Response, reqwest::Error> {
        let body = serde_json::to_vec(&self.req_doc).expect("serde_json::Value serializes");
        let mut builder = self
            .client
            .request(self.method.clone(), &self.upstream_url)
            .body(body);
        for (name, value) in self.headers.iter() {
            let lower = name.as_str().to_ascii_lowercase();
            if HOP_BY_HOP.contains(&lower.as_str()) || lower == "host" || lower == "content-length"
            {
                continue;
            }
            builder = builder.header(name.as_str(), value);
        }
        builder.send().await
    }

    async fn attempt_once(
        &mut self,
        upstream: reqwest::Response,
        tx: &mpsc::Sender<Result<Bytes, io::Error>>,
    ) -> Outcome {
        if upstream.status() != reqwest::StatusCode::OK {
            let status = upstream.status();
            forward_response_body(upstream, tx.clone()).await;
            return Outcome::UpstreamError(format!("upstream HTTP {status}"));
        }

        let mut stream = upstream.bytes_stream();
        let mut parser = EventParser::new();
        let mut ticker = interval_at(
            Instant::now() + self.cfg.check_interval,
            self.cfg.check_interval,
        );
        let mut log_mode_fired = false;

        loop {
            tokio::select! {
                maybe_chunk = stream.next() => {
                    match maybe_chunk {
                        Some(Ok(chunk)) => {
                            if tx.send(Ok(chunk.clone())).await.is_err() {
                                return Outcome::ClientDisconnected;
                            }
                            for payload in parser.push(&chunk) {
                                if !self.spec.is_done(&payload) {
                                    self.feed_detectors(&payload);
                                    self.note_timings(&payload);
                                }
                            }
                        }
                        Some(Err(e)) => {
                            // A read timeout mid-stream is the idle-timeout
                            // backstop firing on a wedged instance; other errors
                            // are ordinary upstream failures.
                            if e.is_timeout() {
                                return Outcome::IdleTimeout;
                            }
                            return Outcome::UpstreamError(e.to_string());
                        }
                        None => {
                            for payload in parser.finish() {
                                if !self.spec.is_done(&payload) {
                                    self.feed_detectors(&payload);
                                    self.note_timings(&payload);
                                }
                            }
                            return Outcome::FinishedNaturally;
                        }
                    }
                }
                _ = tx.closed() => {
                    return Outcome::ClientDisconnected;
                }
                _ = ticker.tick() => {
                    let (idx, period) = self.scan_all();
                    if period == 0 {
                        continue;
                    }
                    if self.cfg.action == StreamingLoopAction::Log {
                        if !log_mode_fired {
                            log_mode_fired = true;
                            warn!(
                                endpoint = self.spec.name(),
                                choice = idx,
                                period,
                                "loop detected in log mode",
                            );
                        }
                        continue;
                    }
                    let snippet = self.snapshot_snippet(idx, period, 256);
                    return Outcome::LoopDetected {
                        choice_index: idx,
                        period,
                        snippet,
                    };
                }
            }
        }
    }

    fn feed_detectors(&mut self, payload: &[u8]) {
        for delta in self.spec.parse_chunk(payload) {
            let choice = self
                .choices
                .entry(delta.index)
                .or_insert_with(|| ChoiceState::new(self.cfg.window, self.cfg.repeats));
            if !delta.reasoning_content.is_empty() {
                choice.append_tracked(FieldKind::Reasoning, delta.reasoning_content.as_bytes());
            }
            if !delta.content.is_empty() {
                choice.append_tracked(FieldKind::Content, delta.content.as_bytes());
            }
            if !delta.tool_calls.is_empty() {
                choice.append_untracked(&delta.tool_calls);
            }
        }
    }

    fn scan_all(&self) -> (i64, usize) {
        let mut indices: Vec<_> = self.choices.keys().copied().collect();
        indices.sort_unstable();
        for idx in indices {
            if let Some(choice) = self.choices.get(&idx) {
                let period = choice.detector.scan();
                if period > 0 {
                    return (idx, period);
                }
            }
        }
        (0, 0)
    }

    fn snapshot_snippet(&self, idx: i64, period: usize, max: usize) -> Vec<u8> {
        self.choices
            .get(&idx)
            .map(|choice| choice.detector.snippet(period, max))
            .unwrap_or_default()
    }

    fn build_partials(
        &self,
        looping_idx: i64,
        period: usize,
    ) -> Option<HashMap<i64, ChoiceSnapshot>> {
        if !self.cfg.replay_partial {
            return None;
        }
        let mut out = HashMap::with_capacity(self.choices.len());
        for (&idx, choice) in &self.choices {
            let p = if idx == looping_idx { period } else { 0 };
            let mut snap = choice.snapshot(self.cfg.repeats, p);
            snap.index = idx;
            if snap.content.is_empty() && snap.reasoning_content.is_empty() {
                continue;
            }
            out.insert(idx, snap);
        }
        Some(out)
    }

    fn reset_choices(&mut self) {
        self.choices.clear();
    }

    async fn write_final_halt_delta(
        &self,
        tx: &mpsc::Sender<Result<Bytes, io::Error>>,
        attempts: usize,
        period: usize,
    ) {
        let out = self.spec.format_halt_delta(attempts, period);
        let _ = tx.send(Ok(Bytes::from(out))).await;
    }

    /// Emit a terminal SSE error event followed by `[DONE]` so a client whose
    /// stream we're aborting (wedged instance recycled) sees a clear error and
    /// stops waiting, rather than the connection just going quiet.
    async fn write_stream_error(&self, tx: &mpsc::Sender<Result<Bytes, io::Error>>, msg: &str) {
        let payload = json!({
            "error": { "message": msg, "type": "router_recycle", "model": self.model_id },
        });
        let out = format!("data: {payload}\n\ndata: [DONE]\n\n");
        let _ = tx.send(Ok(Bytes::from(out))).await;
    }
}

enum Outcome {
    FinishedNaturally,
    ClientDisconnected,
    /// Upstream produced no further bytes within the idle-timeout window — the
    /// signature of a wedged instance.
    IdleTimeout,
    LoopDetected {
        choice_index: i64,
        period: usize,
        snippet: Vec<u8>,
    },
    UpstreamError(String),
}

#[derive(Clone, Copy)]
enum FieldKind {
    Reasoning,
    Content,
}

struct Segment {
    det_start: usize,
    det_end: usize,
    field: FieldKind,
    field_start: usize,
}

struct ChoiceState {
    detector: Detector,
    det_total: usize,
    content: Vec<u8>,
    reasoning: Vec<u8>,
    segments: Vec<Segment>,
}

impl ChoiceState {
    fn new(window: usize, repeats: usize) -> Self {
        Self {
            detector: Detector::new(window, repeats),
            det_total: 0,
            content: Vec::new(),
            reasoning: Vec::new(),
            segments: Vec::new(),
        }
    }

    fn append_tracked(&mut self, field: FieldKind, p: &[u8]) {
        if p.is_empty() {
            return;
        }
        let field_start = match field {
            FieldKind::Reasoning => {
                let start = self.reasoning.len();
                self.reasoning.extend_from_slice(p);
                start
            }
            FieldKind::Content => {
                let start = self.content.len();
                self.content.extend_from_slice(p);
                start
            }
        };
        self.segments.push(Segment {
            det_start: self.det_total,
            det_end: self.det_total + p.len(),
            field,
            field_start,
        });
        self.detector.append(p);
        self.det_total += p.len();
    }

    fn append_untracked(&mut self, p: &[u8]) {
        if p.is_empty() {
            return;
        }
        self.detector.append(p);
        self.det_total += p.len();
    }

    fn snapshot(&self, repeats: usize, period: usize) -> ChoiceSnapshot {
        if period == 0 {
            return ChoiceSnapshot {
                index: 0,
                content: String::from_utf8_lossy(&self.content).into_owned(),
                reasoning_content: String::from_utf8_lossy(&self.reasoning).into_owned(),
            };
        }

        let mut extent = self.detector.loop_extent(period);
        if extent < repeats {
            extent = repeats;
        }
        let trunc_point = self.det_total.saturating_sub(extent * period);
        let mut content = Vec::new();
        let mut reasoning = Vec::new();
        for segment in &self.segments {
            if segment.det_start >= trunc_point {
                break;
            }
            let mut keep = segment.det_end - segment.det_start;
            if segment.det_end > trunc_point {
                keep = trunc_point - segment.det_start;
            }
            match segment.field {
                FieldKind::Reasoning => reasoning.extend_from_slice(
                    &self.reasoning[segment.field_start..segment.field_start + keep],
                ),
                FieldKind::Content => content.extend_from_slice(
                    &self.content[segment.field_start..segment.field_start + keep],
                ),
            }
        }
        ChoiceSnapshot {
            index: 0,
            content: String::from_utf8_lossy(&content).into_owned(),
            reasoning_content: String::from_utf8_lossy(&reasoning).into_owned(),
        }
    }
}

struct ReceiverBodyStream {
    rx: mpsc::Receiver<Result<Bytes, io::Error>>,
}

impl Stream for ReceiverBodyStream {
    type Item = Result<Bytes, io::Error>;

    fn poll_next(mut self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Option<Self::Item>> {
        self.rx.poll_recv(cx)
    }
}

fn response_from_upstream(
    status: StatusCode,
    headers: &reqwest::header::HeaderMap,
    rx: mpsc::Receiver<Result<Bytes, io::Error>>,
) -> Response {
    let mut builder = Response::builder().status(status);
    for (name, value) in headers.iter() {
        let lower = name.as_str().to_ascii_lowercase();
        if HOP_BY_HOP.contains(&lower.as_str()) || lower == "content-length" {
            continue;
        }
        if let (Ok(hn), Ok(hv)) = (
            HeaderName::from_bytes(name.as_str().as_bytes()),
            HeaderValue::from_bytes(value.as_bytes()),
        ) {
            builder = builder.header(hn, hv);
        }
    }
    if !headers.contains_key(reqwest::header::CONTENT_TYPE) {
        builder = builder.header("content-type", "text/event-stream");
    }
    builder
        .body(Body::from_stream(ReceiverBodyStream { rx }))
        .unwrap_or_else(|e| {
            error!(error = %e, "failed to build loop-guard response");
            (
                StatusCode::INTERNAL_SERVER_ERROR,
                "proxy response build failed",
            )
                .into_response()
        })
}

async fn forward_response_body(
    upstream: reqwest::Response,
    tx: mpsc::Sender<Result<Bytes, io::Error>>,
) {
    let mut stream = upstream.bytes_stream();
    while let Some(chunk) = stream.next().await {
        match chunk {
            Ok(bytes) => {
                if tx.send(Ok(bytes)).await.is_err() {
                    return;
                }
            }
            Err(e) => {
                let _ = tx.send(Err(io::Error::other(e.to_string()))).await;
                return;
            }
        }
    }
}

fn truncate(bytes: &[u8], n: usize) -> String {
    if bytes.len() <= n {
        return String::from_utf8_lossy(bytes).into_owned();
    }
    format!("{}...", String::from_utf8_lossy(&bytes[..n]))
}

#[cfg(test)]
#[path = "mod_tests.rs"]
mod tests;
