//! End-to-end test that the proxy forwards bytes unchanged between a client
//! and an upstream llama-server-like process — including SSE multi-line events
//! and comment keepalives.

use std::ffi::OsString;
use std::net::SocketAddr;
use std::pin::Pin;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, OnceLock};
use std::task::{Context, Poll};
use std::time::Duration;

use axum::body::{Body, Bytes};
use axum::response::IntoResponse;
use axum::routing::{any, get, post};
use axum::Json;
use futures_util::{stream, Stream};
use inference_router::api::proxy;
use inference_router::config::{JsonStore, ModelAlias, ModelConfig, ModelExposure, ModelState};
use inference_router::orchestrator::{AppState, Orchestrator};
use tokio::net::TcpListener;
use tokio::sync::oneshot;

struct EnvVarGuard {
    saved: Vec<(&'static str, Option<OsString>)>,
}

impl EnvVarGuard {
    fn set(vars: &[(&'static str, &'static str)]) -> Self {
        let saved = vars
            .iter()
            .map(|(name, _)| (*name, std::env::var_os(name)))
            .collect();
        for (name, value) in vars {
            std::env::set_var(name, value);
        }
        Self { saved }
    }
}

impl Drop for EnvVarGuard {
    fn drop(&mut self) {
        for (name, value) in self.saved.iter().rev() {
            match value {
                Some(value) => std::env::set_var(name, value),
                None => std::env::remove_var(name),
            }
        }
    }
}

fn loop_env_lock() -> &'static tokio::sync::Mutex<()> {
    static LOCK: OnceLock<tokio::sync::Mutex<()>> = OnceLock::new();
    LOCK.get_or_init(|| tokio::sync::Mutex::new(()))
}

/// JSON response body we expect to see round-trip unchanged.
const JSON_RESPONSE: &str = r#"{"id":"cmpl-1","object":"chat.completion","choices":[{"message":{"role":"assistant","content":"hi"}}]}"#;

/// SSE body with a multi-line event, a comment keepalive, and a terminating
/// `[DONE]` — the tricky cases that line-buffering proxies corrupt.
const SSE_BODY: &[u8] =
    b": keepalive\n\ndata: {\"delta\":\"he\"}\ndata: {\"delta\":\"llo\"}\n\ndata: [DONE]\n\n";

async fn fake_chat_completions(body: Json<serde_json::Value>) -> impl IntoResponse {
    let stream = body
        .get("stream")
        .and_then(|v| v.as_bool())
        .unwrap_or(false);
    if stream {
        ([("content-type", "text/event-stream")], SSE_BODY.to_vec()).into_response()
    } else {
        (
            [("content-type", "application/json")],
            JSON_RESPONSE.to_string(),
        )
            .into_response()
    }
}

async fn fake_health() -> &'static str {
    "ok"
}

async fn spawn_fake_llama() -> u16 {
    let app = axum::Router::new()
        .route("/health", get(fake_health))
        .route("/v1/chat/completions", post(fake_chat_completions));

    let listener = TcpListener::bind(SocketAddr::from(([127, 0, 0, 1], 0)))
        .await
        .unwrap();
    let port = listener.local_addr().unwrap().port();
    tokio::spawn(async move {
        axum::serve(listener, app).await.unwrap();
    });
    // Give the server a moment to actually accept connections.
    tokio::time::sleep(Duration::from_millis(50)).await;
    port
}

fn sse_delta(content: &str) -> String {
    let chunk = serde_json::json!({
        "choices": [{
            "index": 0,
            "delta": {"content": content},
        }],
    });
    format!("data: {chunk}\n\n")
}

fn sse_done() -> &'static str {
    "data: [DONE]\n\n"
}

fn looping_script(prefix: &str, unit: &str, reps: usize) -> Vec<u8> {
    let mut body = String::new();
    body.push_str(&sse_delta(prefix));
    for _ in 0..reps {
        body.push_str(&sse_delta(unit));
    }
    body.push_str(sse_done());
    body.into_bytes()
}

fn clean_script(text: &str) -> Vec<u8> {
    let mut body = String::new();
    body.push_str(&sse_delta(text));
    body.push_str(sse_done());
    body.into_bytes()
}

fn delayed_sse_response(body: Vec<u8>) -> axum::response::Response {
    let chunks: Vec<Bytes> = body.chunks(64).map(Bytes::copy_from_slice).collect();
    let s = stream::unfold((chunks, 0usize), |(chunks, idx)| async move {
        if idx >= chunks.len() {
            return None;
        }
        tokio::time::sleep(Duration::from_millis(10)).await;
        Some((
            Ok::<Bytes, std::io::Error>(chunks[idx].clone()),
            (chunks, idx + 1),
        ))
    });
    (
        [("content-type", "text/event-stream")],
        Body::from_stream(s),
    )
        .into_response()
}

async fn spawn_looping_fake_llama() -> (u16, Arc<AtomicUsize>) {
    let calls = Arc::new(AtomicUsize::new(0));
    let handler_calls = calls.clone();
    let app = axum::Router::new()
        .route("/health", get(fake_health))
        .route(
            "/v1/chat/completions",
            post(move |Json(_body): Json<serde_json::Value>| {
                let calls = handler_calls.clone();
                async move {
                    let call = calls.fetch_add(1, Ordering::SeqCst);
                    if call == 0 {
                        delayed_sse_response(looping_script("starting...", &"ABCDE".repeat(12), 12))
                    } else {
                        delayed_sse_response(clean_script("recovered"))
                    }
                }
            }),
        );

    let listener = TcpListener::bind(SocketAddr::from(([127, 0, 0, 1], 0)))
        .await
        .unwrap();
    let port = listener.local_addr().unwrap().port();
    tokio::spawn(async move {
        axum::serve(listener, app).await.unwrap();
    });
    tokio::time::sleep(Duration::from_millis(50)).await;
    (port, calls)
}

struct DropNotifyStream {
    dropped: Option<oneshot::Sender<()>>,
}

impl Stream for DropNotifyStream {
    type Item = Result<Bytes, std::io::Error>;

    fn poll_next(self: Pin<&mut Self>, _cx: &mut Context<'_>) -> Poll<Option<Self::Item>> {
        Poll::Pending
    }
}

impl Drop for DropNotifyStream {
    fn drop(&mut self) {
        if let Some(tx) = self.dropped.take() {
            let _ = tx.send(());
        }
    }
}

async fn spawn_stalling_fake_llama() -> (u16, oneshot::Receiver<()>) {
    let (drop_tx, drop_rx) = oneshot::channel();
    let drop_tx = Arc::new(std::sync::Mutex::new(Some(drop_tx)));
    let handler_drop_tx = drop_tx.clone();
    let app = axum::Router::new()
        .route("/health", get(fake_health))
        .route(
            "/v1/chat/completions",
            post(move |Json(_body): Json<serde_json::Value>| {
                let dropped = handler_drop_tx.lock().unwrap().take();
                async move {
                    (
                        [("content-type", "text/event-stream")],
                        Body::from_stream(DropNotifyStream { dropped }),
                    )
                        .into_response()
                }
            }),
        );

    let listener = TcpListener::bind(SocketAddr::from(([127, 0, 0, 1], 0)))
        .await
        .unwrap();
    let port = listener.local_addr().unwrap().port();
    tokio::spawn(async move {
        axum::serve(listener, app).await.unwrap();
    });
    tokio::time::sleep(Duration::from_millis(50)).await;
    (port, drop_rx)
}

async fn spawn_capturing_fake_llama() -> (u16, Arc<tokio::sync::Mutex<Option<serde_json::Value>>>) {
    let seen = Arc::new(tokio::sync::Mutex::new(None));
    let handler_seen = seen.clone();
    let app = axum::Router::new()
        .route("/health", get(fake_health))
        .route(
            "/v1/chat/completions",
            post(move |Json(body): Json<serde_json::Value>| {
                let seen = handler_seen.clone();
                async move {
                    *seen.lock().await = Some(body);
                    (
                        [("content-type", "application/json")],
                        JSON_RESPONSE.to_string(),
                    )
                        .into_response()
                }
            }),
        );

    let listener = TcpListener::bind(SocketAddr::from(([127, 0, 0, 1], 0)))
        .await
        .unwrap();
    let port = listener.local_addr().unwrap().port();
    tokio::spawn(async move {
        axum::serve(listener, app).await.unwrap();
    });
    tokio::time::sleep(Duration::from_millis(50)).await;
    (port, seen)
}

/// Builds an Orchestrator + axum Router with a "loaded" model pointed at the
/// given upstream port. No real spawn — we synthesize `Running` state.
async fn build_proxy_app(upstream_port: u16) -> axum::Router {
    router_for(build_proxy_orchestrator(upstream_port).await)
}

/// Wraps an orchestrator in the proxy router (so tests can pre-configure the
/// orchestrator — e.g. add aliases — before serving).
fn router_for(orchestrator: Arc<Orchestrator>) -> axum::Router {
    let state: AppState = orchestrator;
    axum::Router::new()
        .route("/v1/models", get(proxy::list_v1_models))
        .route("/v1/{*rest}", any(proxy::proxy_handler))
        .with_state(state)
}

/// Builds the orchestrator with a single "loaded" model pointed at `upstream_port`.
async fn build_proxy_orchestrator(upstream_port: u16) -> Arc<Orchestrator> {
    let tmp = tempfile::tempdir().unwrap();
    let store = Arc::new(JsonStore::<Vec<ModelConfig>>::new(
        tmp.path().join("models.json"),
    ));
    let presets = Arc::new(inference_router::config::JsonStore::<
        Vec<inference_router::config::BinaryPreset>,
    >::new(tmp.path().join("presets.json")));
    let aliases = Arc::new(inference_router::config::JsonStore::<
        Vec<inference_router::config::ModelAlias>,
    >::new(tmp.path().join("aliases.json")));
    // Leak the tempdir so it outlives the test — dropping mid-test deletes the
    // config dir out from under the store.
    std::mem::forget(tmp);

    let orchestrator = Arc::new(Orchestrator::new(store, presets, aliases, 0));
    orchestrator
        .add_model(ModelConfig {
            id: "fake".into(),
            name: "fake".into(),
            binary: std::path::PathBuf::from("/bin/true"),
            model_path: std::path::PathBuf::from("/dev/null"),
            ..ModelConfig::default()
        })
        .await
        .unwrap();

    // Synthesize Running state so ensure_loaded is a no-op.
    {
        let mut data = orchestrator.data.lock().await;
        let m = data.models.get_mut("fake").unwrap();
        m.state = ModelState::Running;
        m.pid = Some(1);
    }
    orchestrator
        .process_manager
        .lock()
        .await
        .register_existing_port("fake", upstream_port);

    orchestrator
}

async fn start_proxy_server(app: axum::Router) -> u16 {
    let listener = TcpListener::bind(SocketAddr::from(([127, 0, 0, 1], 0)))
        .await
        .unwrap();
    let port = listener.local_addr().unwrap().port();
    tokio::spawn(async move {
        axum::serve(listener, app).await.unwrap();
    });
    tokio::time::sleep(Duration::from_millis(50)).await;
    port
}

#[tokio::test]
async fn proxy_forwards_json_body_identically() {
    let upstream = spawn_fake_llama().await;
    let app = build_proxy_app(upstream).await;
    let proxy_port = start_proxy_server(app).await;

    let client = reqwest::Client::new();
    let resp = client
        .post(format!(
            "http://127.0.0.1:{}/v1/chat/completions",
            proxy_port
        ))
        .header("content-type", "application/json")
        .body(r#"{"model":"fake","messages":[{"role":"user","content":"hi"}]}"#)
        .send()
        .await
        .unwrap();

    assert_eq!(resp.status().as_u16(), 200);
    let body = resp.text().await.unwrap();
    assert_eq!(body, JSON_RESPONSE);
}

#[tokio::test]
async fn proxy_forwards_sse_bytes_identically() {
    let upstream = spawn_fake_llama().await;
    let app = build_proxy_app(upstream).await;
    let proxy_port = start_proxy_server(app).await;

    let client = reqwest::Client::new();
    let resp = client
        .post(format!(
            "http://127.0.0.1:{}/v1/chat/completions",
            proxy_port
        ))
        .header("content-type", "application/json")
        .body(r#"{"model":"fake","stream":true,"messages":[]}"#)
        .send()
        .await
        .unwrap();

    assert_eq!(resp.status().as_u16(), 200);
    let ct = resp
        .headers()
        .get("content-type")
        .unwrap()
        .to_str()
        .unwrap()
        .to_string();
    assert!(ct.starts_with("text/event-stream"), "content-type was {ct}");
    let body = resp.bytes().await.unwrap();
    // Must be byte-identical — comment `:` lines, multi-line events, `[DONE]`
    // terminator, all untouched.
    assert_eq!(&body[..], SSE_BODY);
}

#[tokio::test]
async fn proxy_heals_streaming_chat_loop_by_reissuing() {
    let _env_lock = loop_env_lock().lock().await;
    let _env = EnvVarGuard::set(&[
        ("INFERENCE_ROUTER_LOOP_REPEATS", "4"),
        ("INFERENCE_ROUTER_LOOP_CHECK_INTERVAL_MS", "25"),
        ("INFERENCE_ROUTER_LOOP_MAX_RETRIES", "2"),
    ]);

    let (upstream, calls) = spawn_looping_fake_llama().await;
    let app = build_proxy_app(upstream).await;
    let proxy_port = start_proxy_server(app).await;

    let client = reqwest::Client::new();
    let resp = client
        .post(format!(
            "http://127.0.0.1:{}/v1/chat/completions",
            proxy_port
        ))
        .header("content-type", "application/json")
        .body(r#"{"model":"fake","stream":true,"messages":[{"role":"user","content":"hi"}]}"#)
        .send()
        .await
        .unwrap();

    assert_eq!(resp.status().as_u16(), 200);
    let body = resp.text().await.unwrap();
    assert!(body.contains("recovered"), "body was {body}");
    assert!(
        calls.load(Ordering::SeqCst) >= 2,
        "loop should have triggered a reissue"
    );
}

#[tokio::test]
async fn proxy_cancels_monitored_upstream_when_client_drops_response() {
    let (upstream, upstream_body_dropped) = spawn_stalling_fake_llama().await;
    let app = build_proxy_app(upstream).await;
    let proxy_port = start_proxy_server(app).await;

    let client = reqwest::Client::new();
    let resp = client
        .post(format!(
            "http://127.0.0.1:{}/v1/chat/completions",
            proxy_port
        ))
        .header("content-type", "application/json")
        .body(r#"{"model":"fake","stream":true,"messages":[]}"#)
        .send()
        .await
        .unwrap();

    assert_eq!(resp.status().as_u16(), 200);
    drop(resp);

    tokio::time::timeout(Duration::from_secs(1), upstream_body_dropped)
        .await
        .expect("upstream body should be dropped promptly after client disconnect")
        .expect("upstream drop notification should be delivered");
}

#[tokio::test]
async fn proxy_injects_cross_turn_tool_loop_corrective() {
    let _env_lock = loop_env_lock().lock().await;
    let _env = EnvVarGuard::set(&[("INFERENCE_ROUTER_TOOL_LOOP_REPEATS", "2")]);

    let (upstream, seen_body) = spawn_capturing_fake_llama().await;
    let app = build_proxy_app(upstream).await;
    let proxy_port = start_proxy_server(app).await;

    let body = serde_json::json!({
        "model": "fake",
        "messages": [
            {"role": "user", "content": "fix it"},
            {
                "role": "assistant",
                "content": null,
                "tool_calls": [{
                    "id": "call_1",
                    "type": "function",
                    "function": {
                        "name": "edit_file",
                        "arguments": "{\"path\":\"apps/avatar-kiosk/server.mjs\",\"oldString\":\"x\",\"newString\":\"x\"}"
                    }
                }]
            },
            {
                "role": "tool",
                "tool_call_id": "call_1",
                "content": "No changes to apply: oldString and newString are identical."
            },
            {
                "role": "assistant",
                "content": null,
                "tool_calls": [{
                    "id": "call_2",
                    "type": "function",
                    "function": {
                        "name": "edit_file",
                        "arguments": "{\"path\":\"apps/avatar-kiosk/server.mjs\",\"oldString\":\"x\",\"newString\":\"x\"}"
                    }
                }]
            },
            {
                "role": "tool",
                "tool_call_id": "call_2",
                "content": "No changes to apply: oldString and newString are identical."
            }
        ]
    });

    let client = reqwest::Client::new();
    let resp = client
        .post(format!(
            "http://127.0.0.1:{}/v1/chat/completions",
            proxy_port
        ))
        .json(&body)
        .send()
        .await
        .unwrap();

    assert_eq!(resp.status().as_u16(), 200);
    let upstream_body = seen_body.lock().await.clone().unwrap();
    let messages = upstream_body["messages"].as_array().unwrap();
    let last = messages.last().unwrap();
    assert_eq!(last["role"], "user");
    let content = last["content"].as_str().unwrap();
    assert!(content.contains("automated proxy notice"));
    assert!(content.contains("edit_file"));
    assert!(content.contains("oldString and newString are identical"));
}

#[tokio::test]
async fn proxy_rejects_non_post() {
    let upstream = spawn_fake_llama().await;
    let app = build_proxy_app(upstream).await;
    let proxy_port = start_proxy_server(app).await;

    let client = reqwest::Client::new();
    let resp = client
        .get(format!(
            "http://127.0.0.1:{}/v1/chat/completions",
            proxy_port
        ))
        .send()
        .await
        .unwrap();
    assert_eq!(resp.status().as_u16(), 405);
}

#[tokio::test]
async fn proxy_rejects_missing_model_field() {
    let upstream = spawn_fake_llama().await;
    let app = build_proxy_app(upstream).await;
    let proxy_port = start_proxy_server(app).await;

    let client = reqwest::Client::new();
    let resp = client
        .post(format!(
            "http://127.0.0.1:{}/v1/chat/completions",
            proxy_port
        ))
        .header("content-type", "application/json")
        .body(r#"{"messages":[]}"#)
        .send()
        .await
        .unwrap();
    assert_eq!(resp.status().as_u16(), 400);
}

#[tokio::test]
async fn proxy_returns_503_for_unknown_model() {
    let upstream = spawn_fake_llama().await;
    let app = build_proxy_app(upstream).await;
    let proxy_port = start_proxy_server(app).await;

    let client = reqwest::Client::new();
    let resp = client
        .post(format!(
            "http://127.0.0.1:{}/v1/chat/completions",
            proxy_port
        ))
        .header("content-type", "application/json")
        .body(r#"{"model":"does-not-exist"}"#)
        .send()
        .await
        .unwrap();
    assert_eq!(resp.status().as_u16(), 503);
}

#[tokio::test]
async fn v1_models_synthesizes_list_without_upstream() {
    let upstream = spawn_fake_llama().await;
    let app = build_proxy_app(upstream).await;
    let proxy_port = start_proxy_server(app).await;

    let client = reqwest::Client::new();
    let resp = client
        .get(format!("http://127.0.0.1:{}/v1/models", proxy_port))
        .send()
        .await
        .unwrap();
    assert_eq!(resp.status().as_u16(), 200);
    let body: serde_json::Value = resp.json().await.unwrap();
    assert_eq!(body["object"], "list");
    let data = body["data"].as_array().unwrap();
    assert!(data.iter().any(|m| m["id"] == "fake"));
}

#[tokio::test]
async fn proxy_routes_alias_to_target_model() {
    let upstream = spawn_fake_llama().await;
    let orchestrator = build_proxy_orchestrator(upstream).await;
    orchestrator
        .add_alias(ModelAlias {
            alias: "gpt-4o".into(),
            target: "fake".into(),
        })
        .await
        .unwrap();
    let proxy_port = start_proxy_server(router_for(orchestrator)).await;

    let client = reqwest::Client::new();
    let resp = client
        .post(format!(
            "http://127.0.0.1:{}/v1/chat/completions",
            proxy_port
        ))
        .header("content-type", "application/json")
        .body(r#"{"model":"gpt-4o","messages":[{"role":"user","content":"hi"}]}"#)
        .send()
        .await
        .unwrap();

    // The alias resolved to the running "fake" model and served its response.
    assert_eq!(resp.status().as_u16(), 200);
    assert_eq!(resp.text().await.unwrap(), JSON_RESPONSE);
}

#[tokio::test]
async fn proxy_routes_alias_chain_to_target_model() {
    let upstream = spawn_fake_llama().await;
    let orchestrator = build_proxy_orchestrator(upstream).await;
    // coder -> default -> fake (a real, running model).
    orchestrator
        .add_alias(ModelAlias {
            alias: "default".into(),
            target: "fake".into(),
        })
        .await
        .unwrap();
    orchestrator
        .add_alias(ModelAlias {
            alias: "coder".into(),
            target: "default".into(),
        })
        .await
        .unwrap();
    let proxy_port = start_proxy_server(router_for(orchestrator)).await;

    let client = reqwest::Client::new();
    let resp = client
        .post(format!(
            "http://127.0.0.1:{}/v1/chat/completions",
            proxy_port
        ))
        .header("content-type", "application/json")
        .body(r#"{"model":"coder","messages":[{"role":"user","content":"hi"}]}"#)
        .send()
        .await
        .unwrap();
    assert_eq!(resp.status().as_u16(), 200);
    assert_eq!(resp.text().await.unwrap(), JSON_RESPONSE);
}

#[tokio::test]
async fn proxy_returns_503_for_unassigned_alias() {
    let upstream = spawn_fake_llama().await;
    let orchestrator = build_proxy_orchestrator(upstream).await;
    // A canonical alias defined but not yet pointed at a model.
    orchestrator
        .add_alias(ModelAlias {
            alias: "planner".into(),
            target: String::new(),
        })
        .await
        .unwrap();
    let proxy_port = start_proxy_server(router_for(orchestrator)).await;

    let client = reqwest::Client::new();
    let resp = client
        .post(format!(
            "http://127.0.0.1:{}/v1/chat/completions",
            proxy_port
        ))
        .header("content-type", "application/json")
        .body(r#"{"model":"planner","messages":[]}"#)
        .send()
        .await
        .unwrap();
    assert_eq!(resp.status().as_u16(), 503);
    let body: serde_json::Value = resp.json().await.unwrap();
    assert!(
        body["error"].as_str().unwrap().contains("not assigned"),
        "expected a clear unassigned-alias error, got: {body}"
    );
}

#[tokio::test]
async fn v1_models_full_list_includes_models_and_aliases() {
    let upstream = spawn_fake_llama().await;
    let orchestrator = build_proxy_orchestrator(upstream).await;
    orchestrator
        .add_alias(ModelAlias {
            alias: "gpt-4o".into(),
            target: "fake".into(),
        })
        .await
        .unwrap();
    let proxy_port = start_proxy_server(router_for(orchestrator)).await;

    let client = reqwest::Client::new();
    let body: serde_json::Value = client
        .get(format!("http://127.0.0.1:{}/v1/models", proxy_port))
        .send()
        .await
        .unwrap()
        .json()
        .await
        .unwrap();
    let ids: Vec<&str> = body["data"]
        .as_array()
        .unwrap()
        .iter()
        .map(|m| m["id"].as_str().unwrap())
        .collect();
    assert!(ids.contains(&"fake"), "full list must include the model");
    assert!(ids.contains(&"gpt-4o"), "full list must include the alias");
}

#[tokio::test]
async fn v1_models_aliases_only_hides_real_models() {
    let upstream = spawn_fake_llama().await;
    let orchestrator = build_proxy_orchestrator(upstream).await;
    orchestrator
        .add_alias(ModelAlias {
            alias: "gpt-4o".into(),
            target: "fake".into(),
        })
        .await
        .unwrap();
    let mut settings = orchestrator.settings().await;
    settings.model_exposure = ModelExposure::AliasesOnly;
    orchestrator.update_settings(settings).await;
    let proxy_port = start_proxy_server(router_for(orchestrator)).await;

    let client = reqwest::Client::new();
    let body: serde_json::Value = client
        .get(format!("http://127.0.0.1:{}/v1/models", proxy_port))
        .send()
        .await
        .unwrap()
        .json()
        .await
        .unwrap();
    let ids: Vec<&str> = body["data"]
        .as_array()
        .unwrap()
        .iter()
        .map(|m| m["id"].as_str().unwrap())
        .collect();
    assert_eq!(ids, vec!["gpt-4o"], "aliases-only must list only aliases");
}
