//! End-to-end tests for the live dashboard fragment endpoint and embedded
//! assets that back the htmx + idiomorph live updates.
//!
//! These pin the contract the browser depends on: `/fragment/dashboard` returns
//! the two live regions tagged for an out-of-band morph, honours the sort/filter
//! query params server-side, and `/assets/*` serves the vendored JS.

use std::net::SocketAddr;
use std::sync::Arc;
use std::time::Duration;

use inference_router::config::{JsonStore, ModelConfig, ModelState};
use inference_router::lifecycle::build_router;
use inference_router::orchestrator::{AppState, Orchestrator};
use tokio::net::TcpListener;

/// Builds an orchestrator with two models: one Running ("zulu"), one Idle
/// ("alpha"). No real processes are spawned.
async fn orchestrator_with_two_models() -> Arc<Orchestrator> {
    let tmp = tempfile::tempdir().unwrap();
    let store = Arc::new(JsonStore::<Vec<ModelConfig>>::new(
        tmp.path().join("models.json"),
    ));
    let presets = Arc::new(
        JsonStore::<Vec<inference_router::config::BinaryPreset>>::new(
            tmp.path().join("presets.json"),
        ),
    );
    let aliases = Arc::new(JsonStore::<Vec<inference_router::config::ModelAlias>>::new(
        tmp.path().join("aliases.json"),
    ));
    // Leak the tempdir so it outlives the test (dropping it deletes the config
    // dir out from under the stores).
    std::mem::forget(tmp);

    let orchestrator = Arc::new(Orchestrator::new(store, presets, aliases, 0));
    for (id, name) in [("alpha", "Alpha"), ("zulu", "Zulu")] {
        orchestrator
            .add_model(ModelConfig {
                id: id.into(),
                name: name.into(),
                binary: std::path::PathBuf::from("/bin/true"),
                model_path: std::path::PathBuf::from("/dev/null"),
                ..ModelConfig::default()
            })
            .await
            .unwrap();
    }
    // Mark "zulu" Running so it should float to the top of the table.
    {
        let mut data = orchestrator.data.lock().await;
        data.models.get_mut("zulu").unwrap().state = ModelState::Running;
    }
    orchestrator
}

async fn serve(orchestrator: Arc<Orchestrator>) -> u16 {
    let state: AppState = orchestrator;
    let app = build_router(state);
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

async fn get(port: u16, path: &str) -> (reqwest::StatusCode, String) {
    let resp = reqwest::get(format!("http://127.0.0.1:{port}{path}"))
        .await
        .unwrap();
    let status = resp.status();
    let body = resp.text().await.unwrap();
    (status, body)
}

#[tokio::test]
async fn fragment_returns_oob_morph_regions_with_rows() {
    let port = serve(orchestrator_with_two_models().await).await;
    let (status, body) = get(port, "/fragment/dashboard").await;

    assert_eq!(status, reqwest::StatusCode::OK);
    // Both live regions are present and tagged for an out-of-band morph.
    assert!(body.contains(r#"id="live-left" hx-swap-oob="morph""#));
    assert!(body.contains(r#"id="live-models" hx-swap-oob="morph""#));
    // Both models render, each keyed by id so idiomorph matches them in place.
    assert!(body.contains(r#"id="model-row-alpha""#));
    assert!(body.contains(r#"id="model-row-zulu""#));
}

#[tokio::test]
async fn fragment_sorts_loaded_models_first() {
    let port = serve(orchestrator_with_two_models().await).await;
    // Default sort is name-ascending, which alone would put Alpha before Zulu,
    // but the Running model must come first.
    let (_status, body) = get(port, "/fragment/dashboard?sort=name&dir=asc").await;

    let zulu = body.find("model-row-zulu").expect("zulu row present");
    let alpha = body.find("model-row-alpha").expect("alpha row present");
    assert!(
        zulu < alpha,
        "running 'zulu' should sort above idle 'alpha'"
    );
}

#[tokio::test]
async fn fragment_filters_server_side() {
    let port = serve(orchestrator_with_two_models().await).await;

    // Matching filter keeps only Alpha.
    let (_s, body) = get(port, "/fragment/dashboard?q=alpha").await;
    assert!(body.contains("model-row-alpha"));
    assert!(!body.contains("model-row-zulu"));

    // A filter that matches nothing shows the empty-filter row, not the models.
    let (_s, body) = get(port, "/fragment/dashboard?q=nonsense").await;
    assert!(!body.contains("model-row-alpha"));
    assert!(!body.contains("model-row-zulu"));
    assert!(body.contains("No models match this filter."));
}

#[tokio::test]
async fn assets_are_served_with_js_content_type() {
    let port = serve(orchestrator_with_two_models().await).await;

    let resp = reqwest::get(format!("http://127.0.0.1:{port}/assets/htmx.min.js"))
        .await
        .unwrap();
    assert_eq!(resp.status(), reqwest::StatusCode::OK);
    let ctype = resp
        .headers()
        .get("content-type")
        .and_then(|v| v.to_str().ok())
        .unwrap_or("")
        .to_string();
    assert!(ctype.contains("javascript"), "got content-type {ctype}");
    let body = resp.text().await.unwrap();
    assert!(body.contains("htmx"));

    // The idiomorph extension is served too, and registers the morph swap.
    let (status, body) = get(port, "/assets/idiomorph-ext.min.js").await;
    assert_eq!(status, reqwest::StatusCode::OK);
    assert!(body.contains("defineExtension"));

    // Unknown asset names are rejected (no filesystem path traversal surface).
    let (status, _) = get(port, "/assets/secrets.js").await;
    assert_eq!(status, reqwest::StatusCode::NOT_FOUND);
}

#[tokio::test]
async fn index_page_loads_libraries_and_live_regions() {
    let port = serve(orchestrator_with_two_models().await).await;
    let (status, body) = get(port, "/").await;

    assert_eq!(status, reqwest::StatusCode::OK);
    // Libraries are referenced from the embedded asset routes...
    assert!(body.contains(r#"src="/assets/htmx.min.js""#));
    assert!(body.contains(r#"src="/assets/idiomorph-ext.min.js""#));
    // ...the body opts into the morph extension...
    assert!(body.contains(r#"hx-ext="morph""#));
    // ...and the live regions exist for the poll loop to morph into.
    assert!(body.contains(r#"id="live-left""#));
    assert!(body.contains(r#"id="live-models""#));
}
