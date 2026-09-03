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
    assert!(
        (p.prefill - 100.0).abs() < 1e-9,
        "prefill untouched by the 0"
    );
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

fn vllm_metrics(
    start: f64,
    decode_tokens: f64,
    decode_seconds: f64,
    samples: u64,
    prefill_tokens: f64,
    prefill_seconds: f64,
) -> String {
    format!(
        r#"
# HELP ignored metadata
process_start_time_seconds {start}
vllm:request_generation_tokens_sum{{engine="0",model_name="model"}} {decode_tokens}
vllm:request_decode_time_seconds_sum{{engine="0",model_name="model"}} {decode_seconds}
vllm:request_decode_time_seconds_count{{engine="0",model_name="model"}} {samples}
vllm:request_prefill_kv_computed_tokens_sum{{engine="0",model_name="model"}} {prefill_tokens}
vllm:request_prefill_time_seconds_sum{{engine="0",model_name="model"}} {prefill_seconds}
vllm:request_prefill_time_seconds_count{{engine="0",model_name="model"}} {samples}
vllm:request_decode_time_seconds_bucket{{engine="0",le="+Inf"}} 999
"#
    )
}

#[test]
fn vllm_prometheus_metrics_are_parsed() {
    let text = format!(
        "{}{}",
        vllm_metrics(100.0, 200.0, 4.0, 2, 1_000.0, 0.5),
        // A second engine in the same process must be aggregated. The process
        // start value is duplicated in real multiprocess exposition.
        vllm_metrics(100.0, 300.0, 6.0, 3, 2_000.0, 1.0),
    );
    let snapshot = vllm_metrics_from_prometheus(&text).unwrap();
    assert_eq!(snapshot.process_start_seconds, 100.0);
    assert_eq!(snapshot.decode_tokens, 500.0);
    assert_eq!(snapshot.decode_seconds, 10.0);
    assert_eq!(snapshot.decode_samples, 5);
    assert_eq!(snapshot.prefill_tokens, 3_000.0);
    assert_eq!(snapshot.prefill_seconds, 1.5);
    assert_eq!(snapshot.prefill_samples, 5);
}

#[test]
fn incomplete_vllm_metrics_are_rejected() {
    assert_eq!(
        vllm_metrics_from_prometheus("process_start_time_seconds 1\n"),
        None
    );
}

#[test]
fn vllm_totals_produce_time_weighted_throughput() {
    let mut perf = ModelPerf::default();
    perf.record_vllm(&VllmPerfDelta {
        decode_tokens: 200.0,
        decode_seconds: 4.0,
        decode_samples: 2,
        prefill_tokens: 1_000.0,
        prefill_seconds: 0.5,
        prefill_samples: 2,
    });
    perf.record_vllm(&VllmPerfDelta {
        decode_tokens: 300.0,
        decode_seconds: 6.0,
        decode_samples: 3,
        prefill_tokens: 2_000.0,
        prefill_seconds: 1.0,
        prefill_samples: 3,
    });

    assert_eq!(perf.samples, 5);
    assert_eq!(perf.decode, 50.0);
    assert_eq!(perf.prefill, 2_000.0);
    assert_eq!(perf.vllm_decode_tokens, 500.0);
    assert_eq!(perf.vllm_decode_seconds, 10.0);
    assert_eq!(perf.vllm_prefill_tokens, 3_000.0);
    assert_eq!(perf.vllm_prefill_seconds, 1.5);
}

#[test]
fn vllm_tracker_deduplicates_and_rejects_stale_scrapes() {
    let mut tracker = VllmPerfTracker::default();
    let first =
        vllm_metrics_from_prometheus(&vllm_metrics(100.0, 200.0, 4.0, 2, 1_000.0, 0.5)).unwrap();
    let second =
        vllm_metrics_from_prometheus(&vllm_metrics(100.0, 500.0, 10.0, 5, 3_000.0, 1.5)).unwrap();

    let initial = tracker.observe("model", 42, 0, first.clone()).unwrap();
    assert_eq!(initial.decode_tokens, 200.0);
    assert!(tracker.observe("model", 42, 0, first.clone()).is_none());

    let delta = tracker.observe("model", 42, 0, second).unwrap();
    assert_eq!(delta.decode_tokens, 300.0);
    assert_eq!(delta.decode_samples, 3);
    assert_eq!(delta.prefill_tokens, 2_000.0);
    assert!(tracker.observe("model", 42, 0, first).is_none());
}

#[test]
fn vllm_tracker_handles_process_restart_and_model_reset() {
    let mut tracker = VllmPerfTracker::default();
    let old =
        vllm_metrics_from_prometheus(&vllm_metrics(100.0, 200.0, 4.0, 2, 1_000.0, 0.5)).unwrap();
    let restarted =
        vllm_metrics_from_prometheus(&vllm_metrics(200.0, 50.0, 1.0, 1, 500.0, 0.25)).unwrap();

    tracker.observe("model", 42, 0, old.clone()).unwrap();
    let delta = tracker.observe("model", 42, 0, restarted.clone()).unwrap();
    assert_eq!(delta.decode_tokens, 50.0, "new process starts from zero");

    tracker.reset_model("model");
    assert_eq!(tracker.generation("model"), 1);
    assert!(
        tracker.observe("model", 42, 0, old).is_none(),
        "late pre-reset scrape must be ignored"
    );
    assert!(tracker.observe("model", 42, 1, restarted).is_some());
}

#[test]
fn old_model_perf_json_remains_compatible() {
    let perf: ModelPerf =
        serde_json::from_str(r#"{"decode":12.5,"prefill":800.0,"samples":4}"#).unwrap();
    assert_eq!(perf.decode, 12.5);
    assert_eq!(perf.samples, 4);
    assert_eq!(perf.vllm_decode_tokens, 0.0);
    assert_eq!(perf.vllm_prefill_seconds, 0.0);
}
