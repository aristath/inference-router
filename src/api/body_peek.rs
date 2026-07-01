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
#[path = "body_peek_tests.rs"]
mod tests;
