//! Per-model throughput stats, persisted to `model_perf.json`.
//!
//! The router is a transparent proxy, so every llama.cpp response carries a
//! `timings` block. We fold each request's decode/prefill tokens-per-second into
//! a simple running average per model and persist it, so the numbers survive a
//! model being unloaded (or the router restarting). A model's entry is reset
//! whenever its config changes — old timings no longer describe the new setup.

use serde::{Deserialize, Serialize};

/// Rolling throughput averages for one model. The persisted file is a plain
/// `{ "<model-id>": { decode, prefill, samples }, … }` object. `samples` is the
/// request count folded into the means; it's what keeps the *average* correct
/// after the value is reloaded from disk.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct ModelPerf {
    /// Mean decode (generation) tokens/sec — llama.cpp `predicted_per_second`.
    pub decode: f64,
    /// Mean prefill (prompt-processing) tokens/sec — llama.cpp `prompt_per_second`.
    pub prefill: f64,
    /// Completed requests folded into the means.
    pub samples: u64,
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

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn running_average_folds_samples() {
        let mut p = ModelPerf::default();
        p.record(10.0, 100.0);
        p.record(20.0, 200.0);
        assert_eq!(p.samples, 2);
        assert!((p.decode - 15.0).abs() < 1e-9);
        assert!((p.prefill - 150.0).abs() < 1e-9);
    }

    #[test]
    fn zero_or_missing_metric_is_not_folded() {
        let mut p = ModelPerf::default();
        p.record(10.0, 100.0);
        // A cache-hit request: no prefill work reported.
        p.record(20.0, 0.0);
        assert_eq!(p.samples, 2);
        assert!((p.decode - 15.0).abs() < 1e-9, "decode still averages");
        assert!((p.prefill - 100.0).abs() < 1e-9, "prefill untouched by the 0");
    }

    #[test]
    fn timings_parsed_from_response_body() {
        let v = json!({
            "timings": { "predicted_per_second": 16.44, "prompt_per_second": 25.45 }
        });
        assert_eq!(timings_from_json(&v), Some((16.44, 25.45)));
    }

    #[test]
    fn no_timings_block_yields_none() {
        assert_eq!(timings_from_json(&json!({"choices": []})), None);
        assert_eq!(
            timings_from_json(&json!({"timings": {"predicted_per_second": 0}})),
            None
        );
    }
}
