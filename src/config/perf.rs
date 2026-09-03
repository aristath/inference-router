//! Per-model throughput stats, persisted to `model_perf.json`.
//!
//! llama.cpp reports per-request throughput in each response's `timings` block.
//! vLLM exposes cumulative token and phase-time counters from `/metrics`; the
//! router folds counter deltas into weighted totals. Both paths persist their
//! result per model, so the numbers survive a model unload or router restart.
//! A model's entry is reset whenever its config changes, because old timings no
//! longer describe the new setup.

use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Rolling throughput averages for one model. `samples` is the request count
/// represented by the means. vLLM records also retain optional token/time
/// totals so their weighted average remains exact after reloading from disk.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct ModelPerf {
    /// Mean decode (generation) tokens/sec.
    pub decode: f64,
    /// Mean prefill (prompt-processing) tokens/sec.
    pub prefill: f64,
    /// Completed requests folded into the means.
    pub samples: u64,
    /// Weighted vLLM generation-token total. Zero for llama.cpp records and old
    /// persisted files.
    #[serde(default, skip_serializing_if = "is_zero")]
    pub vllm_decode_tokens: f64,
    /// Weighted vLLM decode-time total in seconds.
    #[serde(default, skip_serializing_if = "is_zero")]
    pub vllm_decode_seconds: f64,
    /// Weighted vLLM computed-prefill-token total (cache hits excluded).
    #[serde(default, skip_serializing_if = "is_zero")]
    pub vllm_prefill_tokens: f64,
    /// Weighted vLLM prefill-time total in seconds.
    #[serde(default, skip_serializing_if = "is_zero")]
    pub vllm_prefill_seconds: f64,
}

fn is_zero(value: &f64) -> bool {
    *value == 0.0
}

impl ModelPerf {
    /// Fold one request's throughput into the running means. A metric is only
    /// folded when actually reported (`> 0` and finite), so e.g. a full
    /// prompt-cache hit (no prefill work) doesn't drag the prefill average down.
    pub fn record(&mut self, decode: f64, prefill: f64) {
        let k = (self.samples + 1) as f64;
        if decode > 0.0 && decode.is_finite() {
            self.decode += (decode - self.decode) / k;
        }
        if prefill > 0.0 && prefill.is_finite() {
            self.prefill += (prefill - self.prefill) / k;
        }
        self.samples += 1;
    }

    /// Fold deltas from a vLLM Prometheus snapshot into time-weighted model
    /// throughput. Keeping the underlying totals makes aggregation exact across
    /// concurrent requests, replicas, backend restarts, and router restarts.
    pub(crate) fn record_vllm(&mut self, delta: &VllmPerfDelta) -> bool {
        let mut recorded = false;
        if valid_ratio(delta.decode_tokens, delta.decode_seconds) {
            self.vllm_decode_tokens += delta.decode_tokens;
            self.vllm_decode_seconds += delta.decode_seconds;
            self.decode = self.vllm_decode_tokens / self.vllm_decode_seconds;
            recorded = true;
        }
        if valid_ratio(delta.prefill_tokens, delta.prefill_seconds) {
            self.vllm_prefill_tokens += delta.prefill_tokens;
            self.vllm_prefill_seconds += delta.prefill_seconds;
            self.prefill = self.vllm_prefill_tokens / self.vllm_prefill_seconds;
            recorded = true;
        }
        if recorded {
            self.samples = self
                .samples
                .saturating_add(delta.decode_samples.max(delta.prefill_samples).max(1));
        }
        recorded
    }
}

fn valid_ratio(tokens: f64, seconds: f64) -> bool {
    tokens > 0.0 && tokens.is_finite() && seconds > 0.0 && seconds.is_finite()
}

/// Pull `(decode_tps, prefill_tps)` from a llama.cpp response (or streaming SSE
/// event) JSON's `timings` object. `None` when there's no usable timing in this
/// document — only the final non-streaming body / final SSE event carries it.
pub fn timings_from_json(v: &serde_json::Value) -> Option<(f64, f64)> {
    let t = v.get("timings")?;
    let decode = t
        .get("predicted_per_second")
        .and_then(serde_json::Value::as_f64)
        .unwrap_or(0.0);
    let prefill = t
        .get("prompt_per_second")
        .and_then(serde_json::Value::as_f64)
        .unwrap_or(0.0);
    if decode <= 0.0 && prefill <= 0.0 {
        return None;
    }
    Some((decode, prefill))
}

/// Cumulative vLLM counters from one backend process's Prometheus endpoint.
#[derive(Debug, Clone, PartialEq)]
pub(crate) struct VllmPerfSnapshot {
    pub(crate) process_start_seconds: f64,
    pub(crate) decode_tokens: f64,
    pub(crate) decode_seconds: f64,
    pub(crate) decode_samples: u64,
    pub(crate) prefill_tokens: f64,
    pub(crate) prefill_seconds: f64,
    pub(crate) prefill_samples: u64,
}

/// New work observed since the previous scrape of a vLLM process.
#[derive(Debug, Clone, Default, PartialEq)]
pub(crate) struct VllmPerfDelta {
    pub(crate) decode_tokens: f64,
    pub(crate) decode_seconds: f64,
    pub(crate) decode_samples: u64,
    pub(crate) prefill_tokens: f64,
    pub(crate) prefill_seconds: f64,
    pub(crate) prefill_samples: u64,
}

/// Parse and aggregate the exact counters needed from vLLM's Prometheus text
/// exposition. Multiple labeled series are summed, which covers data-parallel
/// engines exposed by one backend process.
pub(crate) fn vllm_metrics_from_prometheus(text: &str) -> Option<VllmPerfSnapshot> {
    Some(VllmPerfSnapshot {
        process_start_seconds: metric_max(text, "process_start_time_seconds")?,
        decode_tokens: metric_sum(text, "vllm:request_generation_tokens_sum")?,
        decode_seconds: metric_sum(text, "vllm:request_decode_time_seconds_sum")?,
        decode_samples: metric_count(text, "vllm:request_decode_time_seconds_count")?,
        prefill_tokens: metric_sum(text, "vllm:request_prefill_kv_computed_tokens_sum")?,
        prefill_seconds: metric_sum(text, "vllm:request_prefill_time_seconds_sum")?,
        prefill_samples: metric_count(text, "vllm:request_prefill_time_seconds_count")?,
    })
}

fn metric_values<'a>(text: &'a str, name: &'a str) -> impl Iterator<Item = f64> + 'a {
    text.lines().filter_map(move |line| {
        let line = line.trim();
        if line.is_empty() || line.starts_with('#') {
            return None;
        }
        let mut fields = line.split_whitespace();
        let series = fields.next()?;
        let series_name = series.split_once('{').map_or(series, |(n, _)| n);
        if series_name != name {
            return None;
        }
        let value = fields.next()?.parse::<f64>().ok()?;
        (value.is_finite() && value >= 0.0).then_some(value)
    })
}

fn metric_sum(text: &str, name: &str) -> Option<f64> {
    let values: Vec<_> = metric_values(text, name).collect();
    if values.is_empty() {
        return None;
    }
    let total = values.into_iter().sum::<f64>();
    total.is_finite().then_some(total)
}

fn metric_max(text: &str, name: &str) -> Option<f64> {
    metric_values(text, name).reduce(f64::max)
}

fn metric_count(text: &str, name: &str) -> Option<u64> {
    let value = metric_sum(text, name)?;
    (value.fract() == 0.0 && value <= u64::MAX as f64).then_some(value as u64)
}

#[derive(Debug)]
struct TrackedVllmSnapshot {
    model_id: String,
    snapshot: VllmPerfSnapshot,
}

/// Deduplicates concurrent/out-of-order scrapes and converts monotonically
/// increasing process counters into deltas. Model generations reject late
/// scrapes from requests that began before a configuration reset.
#[derive(Debug, Default)]
pub(crate) struct VllmPerfTracker {
    instances: HashMap<i32, TrackedVllmSnapshot>,
    generations: HashMap<String, u64>,
}

impl VllmPerfTracker {
    pub(crate) fn generation(&self, model_id: &str) -> u64 {
        self.generations.get(model_id).copied().unwrap_or(0)
    }

    pub(crate) fn reset_model(&mut self, model_id: &str) {
        let generation = self.generations.entry(model_id.to_string()).or_default();
        *generation = generation.saturating_add(1);
        self.instances.retain(|_, entry| entry.model_id != model_id);
    }

    pub(crate) fn observe(
        &mut self,
        model_id: &str,
        pid: i32,
        generation: u64,
        next: VllmPerfSnapshot,
    ) -> Option<VllmPerfDelta> {
        if generation != self.generation(model_id) {
            return None;
        }

        let previous = self.instances.get(&pid);
        let delta = match previous {
            None => delta_from_zero(&next),
            Some(previous)
                if previous.model_id != model_id
                    || next.process_start_seconds > previous.snapshot.process_start_seconds =>
            {
                delta_from_zero(&next)
            }
            Some(previous)
                if next.process_start_seconds < previous.snapshot.process_start_seconds
                    || snapshot_is_older(&next, &previous.snapshot) =>
            {
                return None;
            }
            Some(previous) => delta_between(&next, &previous.snapshot),
        };

        self.instances.insert(
            pid,
            TrackedVllmSnapshot {
                model_id: model_id.to_string(),
                snapshot: next,
            },
        );

        (delta.decode_samples > 0 || delta.prefill_samples > 0).then_some(delta)
    }
}

fn snapshot_is_older(next: &VllmPerfSnapshot, previous: &VllmPerfSnapshot) -> bool {
    next.decode_samples < previous.decode_samples
        || next.prefill_samples < previous.prefill_samples
        || next.decode_tokens < previous.decode_tokens
        || next.decode_seconds < previous.decode_seconds
        || next.prefill_tokens < previous.prefill_tokens
        || next.prefill_seconds < previous.prefill_seconds
}

fn delta_from_zero(next: &VllmPerfSnapshot) -> VllmPerfDelta {
    VllmPerfDelta {
        decode_tokens: next.decode_tokens,
        decode_seconds: next.decode_seconds,
        decode_samples: next.decode_samples,
        prefill_tokens: next.prefill_tokens,
        prefill_seconds: next.prefill_seconds,
        prefill_samples: next.prefill_samples,
    }
}

fn delta_between(next: &VllmPerfSnapshot, previous: &VllmPerfSnapshot) -> VllmPerfDelta {
    VllmPerfDelta {
        decode_tokens: next.decode_tokens - previous.decode_tokens,
        decode_seconds: next.decode_seconds - previous.decode_seconds,
        decode_samples: next.decode_samples - previous.decode_samples,
        prefill_tokens: next.prefill_tokens - previous.prefill_tokens,
        prefill_seconds: next.prefill_seconds - previous.prefill_seconds,
        prefill_samples: next.prefill_samples - previous.prefill_samples,
    }
}

#[cfg(test)]
#[path = "perf_tests.rs"]
mod tests;
