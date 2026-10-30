//! Stable conversation identity without retaining prompts in router state.
use std::hash::{Hash, Hasher};

pub fn prefix_keys(body: &[u8], headers: &axum::http::HeaderMap) -> Option<Vec<u64>> {
    let doc: serde_json::Value = serde_json::from_slice(body).ok()?;
    let mut hash = std::collections::hash_map::DefaultHasher::new();
    let session = doc
        .get("prompt_cache_key")
        .and_then(|v| v.as_str())
        .or_else(|| {
            ["x-session-id", "session_id", "x-session-affinity"]
                .iter()
                .find_map(|name| headers.get(*name).and_then(|v| v.to_str().ok()))
        });
    if let Some(session) = session {
        session.hash(&mut hash);
        return Some(vec![hash.finish()]);
    }
    // Increasingly specific prefixes distinguish agents with the same initial goal.
    let mut keys = Vec::new();
    for message in doc
        .get("messages")?
        .as_array()?
        .iter()
        .filter(|m| {
            !matches!(
                m.get("role").and_then(|r| r.as_str()),
                Some("system" | "developer")
            )
        })
        .take(12)
    {
        let encoded = serde_json::to_vec(message).ok()?;
        encoded[..encoded.len().min(8192)].hash(&mut hash);
        keys.push(hash.finish());
    }
    (!keys.is_empty()).then_some(keys)
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn growing_conversations_keep_prefixes_and_diverging_agents_differ() {
        let headers = axum::http::HeaderMap::new();
        let first = br#"{"messages":[{"role":"user","content":"same goal"}]}"#;
        let a = br#"{"messages":[{"role":"user","content":"same goal"},{"role":"assistant","content":"issue A"}]}"#;
        let b = br#"{"messages":[{"role":"user","content":"same goal"},{"role":"assistant","content":"issue B"}]}"#;
        assert!(prefix_keys(a, &headers)
            .unwrap()
            .starts_with(&prefix_keys(first, &headers).unwrap()));
        assert_ne!(prefix_keys(a, &headers), prefix_keys(b, &headers));
        assert_eq!(prefix_keys(b"invalid", &headers), None);
    }
}
