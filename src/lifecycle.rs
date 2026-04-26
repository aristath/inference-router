use std::net::SocketAddr;
use std::path::PathBuf;
use std::sync::Arc;
use std::time::Duration;

use axum::extract::State;
use axum::response::{Html, IntoResponse};
use axum::routing::{any, get, post, put};
use askama::Template;
use tracing::{error, info};

use crate::api::proxy;
use crate::api::routes::*;
use crate::api::state::get_app_state;
use crate::config::{BinaryPreset, JsonStore, ModelConfig};
use crate::orchestrator::{AppState, Orchestrator};
use crate::ui::templates::{DashboardTemplate, GpuDisplay, ModelDisplay, ModelGroup, SystemDisplay};

const DEFAULT_CONFIG_DIR: &str = "~/.config/inference-router";

pub struct AppConfig {
    pub port: u16,
    pub config_dir: PathBuf,
}

impl Default for AppConfig {
    fn default() -> Self {
        Self {
            port: 8080,
            config_dir: PathBuf::from(DEFAULT_CONFIG_DIR),
        }
    }
}

pub async fn run(config: AppConfig) -> anyhow::Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| "info".into()),
        )
        .init();

    let expanded = shellexpand::tilde(&config.config_dir.to_string_lossy()).to_string();
    let config_dir = PathBuf::from(expanded);
    std::fs::create_dir_all(&config_dir)?;

    let models_store: Arc<JsonStore<Vec<ModelConfig>>> =
        Arc::new(JsonStore::new(config_dir.join("models.json")));
    let presets_store: Arc<JsonStore<Vec<BinaryPreset>>> =
        Arc::new(JsonStore::new(config_dir.join("presets.json")));

    let orchestrator = Arc::new(Orchestrator::new(
        models_store.clone(),
        presets_store.clone(),
        config.port,
    ));

    // Kick off reconcile loop.
    let reconcile_orch = orchestrator.clone();
    let reconcile_handle = tokio::spawn(async move {
        let mut ticker = tokio::time::interval(Duration::from_secs(5));
        ticker.tick().await;
        reconcile_orch.reconcile().await;
        loop {
            ticker.tick().await;
            reconcile_orch.reconcile().await;
        }
    });

    let app_state: AppState = orchestrator.clone();

    let router = axum::Router::new()
        // Single-page UI
        .route("/", get(index_page))
        // REST API
        .route("/api/status", get(get_app_state))
        .route("/api/models", get(list_models).post(create_model))
        .route("/api/models/{id}", put(update_model).delete(delete_model))
        .route("/api/models/{id}/load", post(load_model))
        .route("/api/models/{id}/stop", post(stop_model))
        .route("/api/files", get(list_files))
        .route("/api/gguf-info", get(gguf_info))
        .route("/api/presets", get(list_presets).post(create_preset))
        .route("/api/presets/{id}", put(update_preset).delete(delete_preset))
        // OpenAI-compatible surface
        .route("/v1/models", get(proxy::list_v1_models))
        .route("/v1/{*rest}", any(proxy::proxy_handler))
        // Liveness
        .route("/healthz", get(proxy::healthz))
        .with_state(app_state);

    let addr = SocketAddr::from(([0, 0, 0, 0], config.port));
    let listener = tokio::net::TcpListener::bind(addr).await?;
    info!(%addr, "inference-router listening");

    let server_result = axum::serve(listener, router)
        .with_graceful_shutdown(shutdown_signal())
        .await;

    // Stop reconcile AND wait for its task to actually finish so it drops its
    // Arc<Orchestrator> clone before we fall out of scope. Without the await
    // the abort is just scheduled and the Arc may linger past `run`'s return.
    reconcile_handle.abort();
    let _ = reconcile_handle.await;

    info!("orchestrator shutting down; killing any running inference servers");
    // `drop(orchestrator)` fires as we fall out of scope: Arc refcount → 0,
    // ProcessManager drops, each RunningChild drops, kill_on_drop(true) on the
    // tokio::process::Child issues SIGKILL synchronously.

    if let Err(e) = server_result {
        error!(error = %e, "server exited with error");
        return Err(anyhow::anyhow!(e));
    }
    Ok(())
}

/// Waits for Ctrl-C (SIGINT) *or* SIGTERM. Catching SIGTERM is essential so
/// a normal `kill <pid>` or systemd stop runs our Drop impls instead of the
/// default terminate-immediately handler.
async fn shutdown_signal() {
    let ctrl_c = async {
        let _ = tokio::signal::ctrl_c().await;
    };
    #[cfg(unix)]
    let terminate = async {
        use tokio::signal::unix::{signal, SignalKind};
        let mut s = signal(SignalKind::terminate())
            .expect("install SIGTERM handler");
        s.recv().await;
    };
    #[cfg(not(unix))]
    let terminate = std::future::pending::<()>();

    tokio::select! {
        _ = ctrl_c => info!("SIGINT received; shutting down"),
        _ = terminate => info!("SIGTERM received; shutting down"),
    }
}

async fn index_page(State(state): State<AppState>) -> impl IntoResponse {
    let gpus = state.list_gpus().await;
    let models = state.list_models().await;
    let sys = state.system_stats();

    let displays: Vec<ModelDisplay> = models.iter().map(ModelDisplay::from_model).collect();

    // Group by architecture; unknown arch → "Other" (sorted last).
    let mut arch_groups: std::collections::HashMap<String, Vec<ModelDisplay>> = Default::default();
    for m in displays {
        let key = if m.architecture.is_empty() { "Other".to_string() } else { m.architecture.clone() };
        arch_groups.entry(key).or_default().push(m);
    }
    let mut groups: Vec<ModelGroup> = arch_groups
        .into_iter()
        .map(|(arch, mut ms)| {
            ms.sort_by_key(|m| m.file_size_bytes);
            let display_name = capitalize_first(&arch);
            ModelGroup { display_name, models: ms }
        })
        .collect();
    groups.sort_by(|a, b| {
        if a.display_name == "Other" { return std::cmp::Ordering::Greater; }
        if b.display_name == "Other" { return std::cmp::Ordering::Less; }
        a.display_name.cmp(&b.display_name)
    });

    let tpl = DashboardTemplate {
        title: "Dashboard".into(),
        system: SystemDisplay::from_stats(sys),
        gpus: gpus.iter().map(GpuDisplay::from_gpu).collect(),
        groups,
        server_port: state.server_port,
    };
    Html(tpl.render().unwrap())
}

fn capitalize_first(s: &str) -> String {
    let mut c = s.chars();
    match c.next() {
        None => String::new(),
        Some(f) => f.to_uppercase().collect::<String>() + c.as_str(),
    }
}
