use axum::{
    body::Body,
    extract::State,
    http::HeaderMap,
    response::{IntoResponse, Response},
    routing::{any, get, post},
    Json,
};
use futures_util::stream;
use inference_router::{
    api::proxy,
    config::{JsonStore, ModelAlias, ModelConfig, ModelState},
    orchestrator::Orchestrator,
};
use std::sync::{Arc, Mutex};
use tokio::{
    net::TcpListener,
    sync::Notify,
    time::{timeout, Duration},
};

#[derive(Clone, Default)]
struct Upstream {
    order: Arc<Mutex<Vec<String>>>,
    finish: Arc<Notify>,
}
async fn generate(
    State(upstream): State<Upstream>,
    headers: HeaderMap,
    Json(body): Json<serde_json::Value>,
) -> Response {
    assert!(!headers.contains_key("x-inference-priority"));
    let label = body["messages"][0]["content"].as_str().unwrap().to_owned();
    upstream.order.lock().unwrap().push(label.clone());
    if label == "first" {
        let bytes = stream::unfold(0, move |step| {
            let finish = upstream.finish.clone();
            async move {
                match step {
                    0 => Some((
                        Ok::<_, std::io::Error>(
                            "data: {\"choices\":[{\"delta\":{\"content\":\"start\"}}]}\n\n",
                        ),
                        1,
                    )),
                    1 => {
                        finish.notified().await;
                        Some((Ok("data: [DONE]\n\n"), 2))
                    }
                    _ => None,
                }
            }
        });
        return Response::builder()
            .header("content-type", "text/event-stream")
            .body(Body::from_stream(bytes))
            .unwrap();
    }
    Json(serde_json::json!({"choices":[{"message":{"content":label}}]})).into_response()
}
async fn server(app: axum::Router) -> (u16, tokio::task::JoinHandle<()>) {
    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let port = listener.local_addr().unwrap().port();
    (
        port,
        tokio::spawn(async move { axum::serve(listener, app).await.unwrap() }),
    )
}
async fn depth(state: &Orchestrator, count: usize) {
    timeout(Duration::from_secs(3), async {
        while state.request_queue.queued("fake") != count {
            tokio::task::yield_now().await;
        }
    })
    .await
    .unwrap();
}

#[tokio::test]
async fn queued_aliases_share_priority_and_running_stream_finishes_before_next_request() {
    let upstream = Upstream::default();
    let (upstream_port, backend) = server(
        axum::Router::new()
            .route("/health", get(|| async { "ok" }))
            .route("/v1/chat/completions", post(generate))
            .with_state(upstream.clone()),
    )
    .await;
    let directory = tempfile::tempdir().unwrap();
    let state = Arc::new(Orchestrator::new(
        Arc::new(JsonStore::new(directory.path().join("models.json"))),
        Arc::new(JsonStore::new(directory.path().join("presets.json"))),
        Arc::new(JsonStore::new(directory.path().join("aliases.json"))),
        0,
    ));
    state
        .add_model(ModelConfig {
            id: "fake".into(),
            name: "fake".into(),
            binary: "/bin/true".into(),
            model_path: "/dev/null".into(),
            ..Default::default()
        })
        .await
        .unwrap();
    {
        let mut data = state.data.lock().await;
        let model = data.models.get_mut("fake").unwrap();
        model.state = ModelState::Running;
        model.pid = Some(1);
        data.aliases.insert(
            "default".into(),
            ModelAlias {
                alias: "default".into(),
                target: "fake".into(),
                fallback: None,
            },
        );
    }
    state
        .process_manager
        .lock()
        .await
        .register_existing_port("fake", upstream_port);
    let (port, router) = server(
        axum::Router::new()
            .route("/v1/{*rest}", any(proxy::proxy_handler))
            .with_state(state.clone()),
    )
    .await;
    let client = reqwest::Client::new();
    let send = |label: &'static str, priority: Option<&'static str>| {
        let mut request = client.post(format!("http://127.0.0.1:{port}/v1/chat/completions"))
            .json(&serde_json::json!({"model":"default", "messages":[{"role":"user", "content":label}], "stream": label == "first"}));
        if let Some(priority) = priority {
            request = request.header("x-inference-priority", priority);
        }
        tokio::spawn(async move { request.send().await.unwrap().text().await.unwrap() })
    };
    let first = send("first", Some("background"));
    timeout(Duration::from_secs(3), async {
        while upstream.order.lock().unwrap().is_empty() {
            tokio::task::yield_now().await;
        }
    })
    .await
    .unwrap();
    let background = send("background", Some("background"));
    depth(&state, 1).await;
    let cancelled = send("cancelled", Some("interactive"));
    depth(&state, 2).await;
    cancelled.abort();
    let _ = cancelled.await;
    depth(&state, 1).await;
    let normal = send("normal", None);
    depth(&state, 2).await;
    let chat = send("chat", Some("interactive"));
    depth(&state, 3).await;
    assert_eq!(*upstream.order.lock().unwrap(), vec!["first"]);
    upstream.finish.notify_one();
    timeout(Duration::from_secs(3), async {
        tokio::try_join!(first, background, normal, chat).unwrap();
    })
    .await
    .unwrap();
    assert_eq!(
        *upstream.order.lock().unwrap(),
        vec!["first", "chat", "normal", "background"]
    );
    assert_eq!(state.request_queue.queued("fake"), 0);
    router.abort();
    backend.abort();
}
