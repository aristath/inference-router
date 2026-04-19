/// Peeks at an OpenAI-style request body and returns the `model` field.
///
/// Returns `None` if the body is not JSON, or if the JSON has no top-level
/// string `model`. Chat/completion/embedding bodies are small enough that
/// parsing the whole thing is fine.
pub fn extract_model(body: &[u8]) -> Option<String> {
    let value: serde_json::Value = serde_json::from_slice(body).ok()?;
    value.get("model")?.as_str().map(|s| s.to_owned())
}

#[cfg(test)]
mod tests {
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
}
