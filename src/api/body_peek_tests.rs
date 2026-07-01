use super::*;

#[test]
fn extracts_model_from_chat_body() {
    let body = br#"{"model":"qwen3-30b","messages":[{"role":"user","content":"hi"}]}"#;
    assert_eq!(extract_model(body), Some("qwen3-30b".into()));
}

#[test]
fn extracts_model_from_completion_body() {
    let body = br#"{"model":"llama-3","prompt":"Hello","max_tokens":10}"#;
    assert_eq!(extract_model(body), Some("llama-3".into()));
}

#[test]
fn returns_none_for_non_json() {
    assert_eq!(extract_model(b"not json at all"), None);
    assert_eq!(extract_model(b""), None);
}

#[test]
fn returns_none_when_model_missing() {
    let body = br#"{"messages":[{"role":"user","content":"hi"}]}"#;
    assert_eq!(extract_model(body), None);
}

#[test]
fn returns_none_when_model_not_string() {
    let body = br#"{"model":123}"#;
    assert_eq!(extract_model(body), None);
}
