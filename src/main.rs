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
    lifecycle::run(AppConfig::default()).await
}
