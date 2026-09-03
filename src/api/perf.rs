//! Background collection of vLLM's per-process Prometheus throughput totals.

use std::collections::HashMap;
use std::sync::{Mutex, OnceLock};
use std::time::Duration;

use tracing::debug;

use crate::config::vllm_metrics_from_prometheus;
use crate::orchestrator::AppState;
use crate::process::manager::RequestGuard;

const MAX_METRICS_BYTES: usize = 8 * 1024 * 1024;

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
struct CollectionKey {
    model_id: String,
    pid: i32,
    port: u16,
    generation: u64,
}

/// One active scrape per backend generation. A request that completes during
/// a scrape sets the value to `true`, guaranteeing one trailing refresh without
/// issuing an unbounded burst of concurrent `/metrics` requests.
fn collection_queue() -> &'static Mutex<HashMap<CollectionKey, bool>> {
    static QUEUE: OnceLock<Mutex<HashMap<CollectionKey, bool>>> = OnceLock::new();
    QUEUE.get_or_init(|| Mutex::new(HashMap::new()))
}

fn metrics_client() -> &'static reqwest::Client {
    static CLIENT: OnceLock<reqwest::Client> = OnceLock::new();
    CLIENT.get_or_init(|| {
        reqwest::Client::builder()
            .no_proxy()
            .connect_timeout(Duration::from_secs(1))
            .timeout(Duration::from_secs(3))
            .build()
            .unwrap_or_else(|_| reqwest::Client::new())
    })
}

/// Fetch one metrics snapshot after a proxied vLLM request completes. The work
/// is detached so metrics collection never extends client response latency.
pub(super) fn spawn_vllm_perf_collection(
    state: AppState,
    model_id: String,
    pid: i32,
    port: u16,
    generation: u64,
    guard: RequestGuard,
) {
    let key = CollectionKey {
        model_id,
        pid,
        port,
        generation,
    };
    {
        let mut queue = collection_queue()
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        if let Some(pending) = queue.get_mut(&key) {
            *pending = true;
            return;
        }
        queue.insert(key.clone(), false);
    }

    tokio::spawn(async move {
        // Prevent the exact serving instance from being evicted between the
        // response completing and its bounded metrics scrape finishing.
        let _guard = guard;
        loop {
            if let Err(error) =
                collect_vllm_perf(&state, &key.model_id, key.pid, key.port, key.generation).await
            {
                // Older/custom OpenAI-compatible backends may not expose
                // vLLM's metric set. Throughput remains unavailable for them.
                debug!(model = key.model_id, pid = key.pid, port = key.port, %error, "vLLM metrics collection skipped");
            }

            let run_again = {
                let mut queue = collection_queue()
                    .lock()
                    .unwrap_or_else(|poisoned| poisoned.into_inner());
                match queue.get_mut(&key) {
                    Some(pending) if *pending => {
                        *pending = false;
                        true
                    }
                    _ => {
                        queue.remove(&key);
                        false
                    }
                }
            };
            if !run_again {
                break;
            }
        }
    });
}

async fn collect_vllm_perf(
    state: &AppState,
    model_id: &str,
    pid: i32,
    port: u16,
    generation: u64,
) -> anyhow::Result<()> {
    let url = format!("http://127.0.0.1:{port}/metrics");
    let response = metrics_client().get(url).send().await?.error_for_status()?;
    if response
        .content_length()
        .is_some_and(|len| len > MAX_METRICS_BYTES as u64)
    {
        anyhow::bail!("metrics response exceeded {MAX_METRICS_BYTES} bytes");
    }
    let body = response.bytes().await?;
    if body.len() > MAX_METRICS_BYTES {
        anyhow::bail!("metrics response exceeded {MAX_METRICS_BYTES} bytes");
    }
    let text = std::str::from_utf8(&body)?;
    let snapshot = vllm_metrics_from_prometheus(text)
        .ok_or_else(|| anyhow::anyhow!("required vLLM throughput counters were absent"))?;
    state.record_vllm_perf_snapshot(model_id, pid, generation, snapshot);
    Ok(())
}
