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
