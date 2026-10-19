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
    let args: Vec<String> = std::env::args().collect();
    if args.get(1).is_some_and(|s| s == "--gpu-worker") {
        if args.len() < 5 || args[3] != "--" {
            anyhow::bail!("invalid media worker invocation");
        }
        return Err(
            orchestrator::media::worker_exec(std::path::Path::new(&args[2]), &args[4..]).into(),
        );
    }
    lifecycle::run(AppConfig::default()).await
}
