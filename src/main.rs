mod api;
mod config;
mod lifecycle;
mod orchestrator;
mod process;
mod system;
mod ui;
mod vram;

use lifecycle::AppConfig;

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let mut config = AppConfig::default();
    // Optional env overrides (default behavior unchanged when unset). Useful for
    // running an isolated second instance without clobbering the real config.
    if let Some(port) = std::env::var("INFERENCE_ROUTER_PORT")
        .ok()
        .and_then(|p| p.trim().parse().ok())
    {
        config.port = port;
    }
    if let Ok(dir) = std::env::var("INFERENCE_ROUTER_CONFIG_DIR") {
        config.config_dir = dir.into();
    }
    lifecycle::run(config).await
}
