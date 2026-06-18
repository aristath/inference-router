use std::net::SocketAddr;
use std::path::PathBuf;
use std::sync::Arc;
use std::time::Duration;

use askama::Template;
use axum::extract::{Query, State};
use axum::response::{Html, IntoResponse};
use axum::routing::{any, get, post, put};
use tracing::{error, info};

use crate::api::proxy;
use crate::api::routes::*;
use crate::api::state::get_app_state;
use crate::config::{AppSettings, BinaryPreset, JsonStore, ModelAlias, ModelConfig};
use crate::orchestrator::{AppState, Orchestrator};
use crate::ui::templates::{
    sort_and_filter, DashboardFragmentTemplate, DashboardTemplate, EventDisplay, GpuDisplay,
    ModelDisplay, ModelSort, SystemDisplay,
};

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
            tracing_subscriber::EnvFilter::try_from_default_env().unwrap_or_else(|_| "info".into()),
        )
        .init();

    let expanded = shellexpand::tilde(&config.config_dir.to_string_lossy()).to_string();
    let config_dir = PathBuf::from(expanded);
    std::fs::create_dir_all(&config_dir)?;

    let models_store: Arc<JsonStore<Vec<ModelConfig>>> =
        Arc::new(JsonStore::new(config_dir.join("models.json")));
    let presets_store: Arc<JsonStore<Vec<BinaryPreset>>> =
        Arc::new(JsonStore::new(config_dir.join("presets.json")));
    let aliases_store: Arc<JsonStore<Vec<ModelAlias>>> =
        Arc::new(JsonStore::new(config_dir.join("aliases.json")));
    let settings_path = config_dir.join("settings.json");
    let settings_exists = settings_path.exists();
    let settings_store: Arc<JsonStore<AppSettings>> = Arc::new(JsonStore::new(settings_path));
    if !settings_exists {
        settings_store.replace(AppSettings::from_env());
    }

    let orchestrator = Arc::new(Orchestrator::new_with_settings_store(
        models_store.clone(),
        presets_store.clone(),
        aliases_store.clone(),
        settings_store.clone(),
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

    let router = build_router(app_state);

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
        let mut s = signal(SignalKind::terminate()).expect("install SIGTERM handler");
        s.recv().await;
    };
    #[cfg(not(unix))]
    let terminate = std::future::pending::<()>();

    tokio::select! {
        _ = ctrl_c => info!("SIGINT received; shutting down"),
        _ = terminate => info!("SIGTERM received; shutting down"),
    }
}

/// Builds the full application router. Extracted from [`run`] so tests can
/// mount every route against a pre-configured orchestrator without binding a
/// socket or starting the reconcile loop.
pub fn build_router(app_state: AppState) -> axum::Router {
    axum::Router::new()
        // Single-page UI
        .route("/", get(index_page))
        // Live dashboard regions, morphed in on each poll (htmx + idiomorph)
        .route("/fragment/dashboard", get(dashboard_fragment))
        // Embedded front-end assets (htmx, idiomorph)
        .route("/assets/{file}", get(crate::ui::assets::serve_asset))
        // REST API
        .route("/api/status", get(get_app_state))
        .route("/api/settings", get(get_settings).put(update_settings))
        .route("/api/models", get(list_models).post(create_model))
        .route("/api/models/validate", post(validate_model))
        .route("/api/models/{id}", put(update_model).delete(delete_model))
        .route("/api/models/{id}/load", post(load_model))
        .route("/api/models/{id}/stop", post(stop_model))
        .route("/api/service/restart", post(restart_service))
        .route("/api/files", get(list_files))
        .route("/api/gguf-info", get(gguf_info))
        .route("/api/presets", get(list_presets).post(create_preset))
        .route(
            "/api/presets/{id}",
            put(update_preset).delete(delete_preset),
        )
        .route("/api/aliases", get(list_aliases).post(create_alias))
        .route(
            "/api/aliases/{alias}",
            put(update_alias).delete(delete_alias),
        )
        // OpenAI-compatible surface
        .route("/v1/models", get(proxy::list_v1_models))
        .route("/v1/{*rest}", any(proxy::proxy_handler))
        // Liveness
        .route("/healthz", get(proxy::healthz))
        .with_state(app_state)
}

/// Sort/filter the browser replays on every poll. Absent fields fall back to
/// the first-paint defaults (name ascending, no filter).
#[derive(serde::Deserialize, Default)]
struct DashboardQuery {
    sort: Option<String>,
    dir: Option<String>,
    q: Option<String>,
}

/// The four live dashboard regions (host, GPUs, events, models), pre-formatted
/// and ready to drop into either the full page or the poll fragment. Built once
/// per request so first paint and every poll render from identical data.
struct LiveData {
    system: SystemDisplay,
    gpus: Vec<GpuDisplay>,
    events: Vec<EventDisplay>,
    models: Vec<ModelDisplay>,
    has_any_models: bool,
    sort_key: String,
    sort_dir_class: String,
    sort_aria: String,
}

/// Snapshots host/GPU/event/model state and applies the requested sort+filter.
/// Activity counters come from the live process runtimes, overlaid onto the
/// per-model display built from static config.
async fn collect_live_data(state: &AppState, sort: &str, dir: &str, query: &str) -> LiveData {
    let gpus = state.list_gpus().await;
    let models = state.list_models().await;
    let runtimes = state.model_runtimes().await;
    let events = state.recent_events().await;
    let sys = state.system_stats();

    let has_any_models = !models.is_empty();
    let displays: Vec<ModelDisplay> = models
        .iter()
        .map(|m| {
            let rt = runtimes.get(&m.id).copied().unwrap_or_default();
            ModelDisplay::from_model(m).with_runtime(rt.active, rt.pending)
        })
        .collect();

    let model_sort = ModelSort::new(sort, dir);
    let models = sort_and_filter(displays, model_sort, query);

    LiveData {
        system: SystemDisplay::from_stats(sys),
        gpus: gpus.iter().map(GpuDisplay::from_gpu).collect(),
        events: events
            .into_iter()
            .map(|e| EventDisplay::new(e.ts, e.level, e.message))
            .collect(),
        models,
        has_any_models,
        sort_key: model_sort.key.to_string(),
        sort_dir_class: model_sort.dir_class().to_string(),
        sort_aria: model_sort.aria().to_string(),
    }
}

async fn index_page(State(state): State<AppState>) -> impl IntoResponse {
    // First paint uses the default ordering; the browser then polls
    // /fragment/dashboard with its own sort/filter choices.
    let live = collect_live_data(&state, "name", "asc", "").await;
    let tpl = DashboardTemplate {
        title: "Dashboard".into(),
        system: live.system,
        gpus: live.gpus,
        events: live.events,
        models: live.models,
        has_any_models: live.has_any_models,
        sort_key: live.sort_key,
        sort_dir_class: live.sort_dir_class,
        sort_aria: live.sort_aria,
        server_port: state.server_port,
    };
    Html(tpl.render().unwrap())
}

/// Poll endpoint: renders only the live regions, each wrapped for an idiomorph
/// out-of-band swap. The browser calls this on its polling interval with the
/// active sort column/direction and filter text.
async fn dashboard_fragment(
    State(state): State<AppState>,
    Query(query): Query<DashboardQuery>,
) -> impl IntoResponse {
    let sort = query.sort.as_deref().unwrap_or("name");
    let dir = query.dir.as_deref().unwrap_or("asc");
    let q = query.q.as_deref().unwrap_or("");
    let live = collect_live_data(&state, sort, dir, q).await;
    let tpl = DashboardFragmentTemplate {
        system: live.system,
        gpus: live.gpus,
        events: live.events,
        models: live.models,
        has_any_models: live.has_any_models,
        sort_key: live.sort_key,
        sort_dir_class: live.sort_dir_class,
        sort_aria: live.sort_aria,
    };
    Html(tpl.render().unwrap())
}
