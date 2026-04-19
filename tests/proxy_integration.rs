//! End-to-end test that the proxy forwards bytes unchanged between a client
//! and an upstream llama-server-like process — including SSE multi-line events
//! and comment keepalives.

use std::net::SocketAddr;
use std::sync::Arc;
use std::time::Duration;

use axum::response::IntoResponse;
use axum::routing::{any, get, post};
use axum::Json;
use inference_router::api::proxy;
use inference_router::config::{JsonStore, ModelConfig, ModelState, WeightsFormat};
use inference_router::orchestrator::{AppState, Orchestrator};
use tokio::net::TcpListener;

/// JSON response body we expect to see round-trip unchanged.
const JSON_RESPONSE: &str = r#"{"id":"cmpl-1","object":"chat.completion","choices":[{"message":{"role":"assistant","content":"hi"}}]}"#;

/// SSE body with a multi-line event, a comment keepalive, and a terminating
/// `[DONE]` — the tricky cases that line-buffering proxies corrupt.
const SSE_BODY: &[u8] = b": keepalive\n\ndata: {\"delta\":\"he\"}\ndata: {\"delta\":\"llo\"}\n\ndata: [DONE]\n\n";

async fn fake_chat_completions(body: Json<serde_json::Value>) -> impl IntoResponse {
    let stream = body.get("stream").and_then(|v| v.as_bool()).unwrap_or(false);
    if stream {
        (
            [("content-type", "text/event-stream")],
            SSE_BODY.to_vec(),
        )
            .into_response()
    } else {
        (
            [("content-type", "application/json")],
            JSON_RESPONSE.to_string(),
        )
            .into_response()
    }
}

async fn fake_health() -> &'static str { "ok" }

async fn spawn_fake_llama() -> u16 {
    let app = axum::Router::new()
        .route("/health", get(fake_health))
        .route("/v1/chat/completions", post(fake_chat_completions));

    let listener = TcpListener::bind(SocketAddr::from(([127, 0, 0, 1], 0))).await.unwrap();
    let port = listener.local_addr().unwrap().port();
    tokio::spawn(async move {
        axum::serve(listener, app).await.unwrap();
    });
    // Give the server a moment to actually accept connections.
    tokio::time::sleep(Duration::from_millis(50)).await;
    port
}

/// Builds an Orchestrator + axum Router with a "loaded" model pointed at the
/// given upstream port. No real spawn — we synthesize `Running` state.
async fn build_proxy_app(upstream_port: u16) -> axum::Router {
    let tmp = tempfile::tempdir().unwrap();
    let store = Arc::new(JsonStore::<Vec<ModelConfig>>::new(
        tmp.path().join("models.json"),
    ));
    let presets = Arc::new(inference_router::config::JsonStore::<Vec<inference_router::config::BinaryPreset>>::new(
        tmp.path().join("presets.json"),
    ));
    // Leak the tempdir so it outlives the test — dropping mid-test deletes the
    // config dir out from under the store.
    std::mem::forget(tmp);

    let orchestrator = Arc::new(Orchestrator::new(store, presets, 0));
    orchestrator
        .add_model(ModelConfig {
            id: "fake".into(),
            name: "fake".into(),
            weights_format: WeightsFormat::Gguf,
            binary_preset: None,
            binary: std::path::PathBuf::from("/bin/true"),
            model_path: std::path::PathBuf::from("/dev/null"),
            port: upstream_port,
            extra_args: vec![],
            context: 4096,
            temperature: 0.6,
            top_p: 0.95,
            top_k: 40,
            min_p: 0.0,
            flash_attn: false,
            n_gpu_layers: None,
            mlock: false,
            no_mmap: false,
            parallel_slots: None,
            cache_type_k: None,
            cache_type_v: None,
            split_mode: None,
            main_gpu: None,
            tensor_split: None,
            threads: None,
            cache_ram_mib: None,
            reasoning_format: None,
            reasoning_budget: None,
            chat_template_kwargs: None,
            presence_penalty: 0.0,
            repeat_penalty: 1.0,
            state: ModelState::Idle,
            pid: None,
            estimated_vram: 0,
            last_used: None,
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

    let state: AppState = orchestrator;
    axum::Router::new()
        .route("/v1/models", get(proxy::list_v1_models))
        .route("/v1/{*rest}", any(proxy::proxy_handler))
        .with_state(state)
}

async fn start_proxy_server(app: axum::Router) -> u16 {
    let listener = TcpListener::bind(SocketAddr::from(([127, 0, 0, 1], 0))).await.unwrap();
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
        .post(format!("http://127.0.0.1:{}/v1/chat/completions", proxy_port))
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
        .post(format!("http://127.0.0.1:{}/v1/chat/completions", proxy_port))
        .header("content-type", "application/json")
        .body(r#"{"model":"fake","stream":true,"messages":[]}"#)
        .send()
        .await
        .unwrap();

    assert_eq!(resp.status().as_u16(), 200);
    let ct = resp.headers().get("content-type").unwrap().to_str().unwrap().to_string();
    assert!(ct.starts_with("text/event-stream"), "content-type was {ct}");
    let body = resp.bytes().await.unwrap();
    // Must be byte-identical — comment `:` lines, multi-line events, `[DONE]`
    // terminator, all untouched.
    assert_eq!(&body[..], SSE_BODY);
}

#[tokio::test]
async fn proxy_rejects_non_post() {
    let upstream = spawn_fake_llama().await;
    let app = build_proxy_app(upstream).await;
    let proxy_port = start_proxy_server(app).await;

    let client = reqwest::Client::new();
    let resp = client
        .get(format!("http://127.0.0.1:{}/v1/chat/completions", proxy_port))
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
        .post(format!("http://127.0.0.1:{}/v1/chat/completions", proxy_port))
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
        .post(format!("http://127.0.0.1:{}/v1/chat/completions", proxy_port))
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

