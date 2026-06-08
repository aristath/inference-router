//! # Endpoint-Specific Handling
//!
//! Adapts the loop guard system to different upstream API formats:
//! - OpenAI chat/completions
//! - OpenAI legacy completions
//! - llama.cpp's native `/completion` endpoint
//!
//! Each variant knows how to:
//! 1. Parse streaming chunks into choice deltas
//! 2. Detect when a response is complete
//! 3. Format halt messages
//! 4. Inject corrective prompts
//!
//! ## Key Differences
//! | Endpoint          | Chunk Format       | Delta Field | Done Marker       |
//! |-------------------|--------------------|-------------|-------------------|
//! | chat/completions  | SSE (data: ...)    | delta       | [DONE]            |
//! | completions       | SSE (data: ...)    | text        | [DONE]            |
//! | llama.cpp          | SSE (data: ...)    | content     | stop: true        |

use std::collections::HashMap;

use serde_json::{json, Value};

use super::sse::{self, ChoiceDelta};

/// Adapter for endpoint-specific streaming formats.
/// 
/// # Responsibilities
/// 1. Parses streaming chunks into choice deltas
/// 2. Detects when a response is complete
/// 3. Formats halt messages for loop detection
/// 4. Injects corrective prompts when loops are detected
/// 
/// # Supported Endpoints
/// - OpenAI chat/completions (SSE with `delta` field)
/// - OpenAI legacy completions (SSE with `text` field)
/// - llama.cpp native `/completion` (SSE with `content` field)
/// 
/// # Key Methods
/// - `detect()`: Identifies endpoint from path
/// - `parse_chunk()`: Extracts choice deltas from chunk bytes
/// - `is_done()`: Checks if response is complete
/// - `format_halt_delta()`: Creates loop detection message
/// - `inject_corrective()`: Adds corrective prompt to request
#[derive(Debug, Clone, Copy)]
pub(super) enum EndpointKind {
    Chat,
    Completion,
    LlamaCpp,
}

#[derive(Debug, Clone)]
pub(super) struct ChoiceSnapshot {
    pub(super) index: i64,
    pub(super) content: String,
    pub(super) reasoning_content: String,
}

impl EndpointKind {
    pub(super) fn detect(path: &str) -> Option<Self> {
        if path.ends_with("/chat/completions") {
            Some(Self::Chat)
        } else if path.ends_with("/completions") {
            Some(Self::Completion)
        } else if path.ends_with("/completion") {
            Some(Self::LlamaCpp)
        } else {
            None
        }
    }

    pub(super) fn name(&self) -> &'static str {
        match self {
            Self::Chat => "chat/completions",
            Self::Completion => "completions",
            Self::LlamaCpp => "llama.cpp /completion",
        }
    }

    pub(super) fn parse_chunk(&self, payload: &[u8]) -> Vec<ChoiceDelta> {
        match self {
            Self::Chat => sse::extract_chat(payload),
            Self::Completion => sse::extract_completion(payload),
            Self::LlamaCpp => sse::extract_llama_cpp(payload),
        }
    }

    pub(super) fn is_done(&self, payload: &[u8]) -> bool {
        sse::is_done(payload)
    }

    pub(super) fn format_halt_delta(&self, attempts: usize, period: usize) -> Vec<u8> {
        match self {
            Self::Chat => chat_halt_delta(attempts, period),
            Self::Completion => completion_halt_delta(attempts, period),
            Self::LlamaCpp => llama_cpp_halt_delta(attempts, period),
        }
    }

    pub(super) fn inject_corrective(
        &self,
        req_doc: &mut Value,
        attempt: usize,
        partial: Option<&HashMap<i64, ChoiceSnapshot>>,
    ) {
        match self {
            Self::Chat => chat_inject_corrective(req_doc, attempt, partial),
            Self::Completion | Self::LlamaCpp => {
                completion_inject_corrective(req_doc, attempt, partial)
            }
        }
    }
}

fn corrective_text(attempt: usize) -> &'static str {
    match attempt {
        0 => "[INTERRUPT \u{2014} automated proxy notice] The previous attempt entered a thinking/output loop. Step back, identify the real next concrete action, and proceed without restating analysis you've already done.",
        1 => "[INTERRUPT \u{2014} automated proxy notice] Two attempts have looped. Skip all deliberation and produce the next tool call or concrete output directly.",
        _ => "[INTERRUPT \u{2014} automated proxy notice] Final attempt \u{2014} emit only the next required action, no thinking, no commentary.",
    }
}

fn chat_halt_delta(attempts: usize, period: usize) -> Vec<u8> {
    let chunk = json!({
        "choices": [{
            "index": 0,
            "delta": {
                "content": format!("\n\n[loop detected after {attempts} attempt(s) (period={period}), generation halted]\n"),
            },
            "finish_reason": "stop",
        }],
    });
    format!("data: {}\n\ndata: [DONE]\n\n", chunk).into_bytes()
}

fn completion_halt_delta(attempts: usize, period: usize) -> Vec<u8> {
    let chunk = json!({
        "choices": [{
            "index": 0,
            "text": format!("\n\n[loop detected after {attempts} attempt(s) (period={period}), halted]\n"),
            "finish_reason": "stop",
        }],
    });
    format!("data: {}\n\ndata: [DONE]\n\n", chunk).into_bytes()
}

fn llama_cpp_halt_delta(attempts: usize, period: usize) -> Vec<u8> {
    let chunk = json!({
        "content": format!("\n\n[loop detected after {attempts} attempt(s) (period={period}), halted]\n"),
        "stop": true,
        "stopped_eos": false,
        "stopped_word": false,
    });
    format!("data: {}\n\n", chunk).into_bytes()
}

fn chat_inject_corrective(
    req_doc: &mut Value,
    attempt: usize,
    partial: Option<&HashMap<i64, ChoiceSnapshot>>,
) {
    let Some(msgs) = req_doc.get_mut("messages").and_then(Value::as_array_mut) else {
        return;
    };

    if let Some(snap) = pick_partial(partial) {
        let body = build_assistant_body(snap);
        if !body.is_empty() {
            msgs.push(json!({
                "role": "assistant",
                "content": body,
            }));
        }
    }

    msgs.push(json!({
        "role": "user",
        "content": corrective_text(attempt),
    }));
}

fn build_assistant_body(snap: &ChoiceSnapshot) -> String {
    let mut out = String::new();
    if !snap.reasoning_content.is_empty() {
        out.push_str("<think>");
        out.push_str(&snap.reasoning_content);
        out.push_str("</think>");
    }
    out.push_str(&snap.content);
    out
}

fn completion_inject_corrective(
    req_doc: &mut Value,
    attempt: usize,
    partial: Option<&HashMap<i64, ChoiceSnapshot>>,
) {
    let suffix = build_completion_suffix(attempt, partial);
    match req_doc.get_mut("prompt") {
        Some(Value::String(prompt)) => {
            prompt.push_str(&suffix);
        }
        Some(Value::Array(items)) => {
            for item in items {
                if let Value::String(prompt) = item {
                    prompt.push_str(&suffix);
                }
            }
        }
        _ => {}
    }
}

fn build_completion_suffix(
    attempt: usize,
    partial: Option<&HashMap<i64, ChoiceSnapshot>>,
) -> String {
    let mut out = String::new();
    if let Some(snap) = pick_partial(partial) {
        out.push_str(&snap.content);
    }
    out.push_str("\n\n[");
    out.push_str(corrective_text(attempt));
    out.push_str("]\n\n");
    out
}

fn pick_partial(partial: Option<&HashMap<i64, ChoiceSnapshot>>) -> Option<&ChoiceSnapshot> {
    partial?
        .values()
        .filter(|snap| !snap.content.is_empty() || !snap.reasoning_content.is_empty())
        .min_by_key(|snap| snap.index)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn chat_corrective_appends_partial_and_user_message() {
        let mut doc = json!({"messages":[{"role":"user","content":"hi"}]});
        let mut partial = HashMap::new();
        partial.insert(
            0,
            ChoiceSnapshot {
                index: 0,
                content: "answer".into(),
                reasoning_content: "thinking".into(),
            },
        );
        EndpointKind::Chat.inject_corrective(&mut doc, 0, Some(&partial));
        let msgs = doc["messages"].as_array().unwrap();
        assert_eq!(msgs.len(), 3);
        assert_eq!(msgs[1]["role"], "assistant");
        assert_eq!(msgs[1]["content"], "<think>thinking</think>answer");
        assert!(msgs[2]["content"].as_str().unwrap().contains("INTERRUPT"));
    }

    #[test]
    fn completion_corrective_wraps_interrupt_like_go_version() {
        let mut doc = json!({"prompt":"Once"});
        EndpointKind::Completion.inject_corrective(&mut doc, 0, None);
        let prompt = doc["prompt"].as_str().unwrap();
        assert!(prompt.starts_with("Once\n\n[[INTERRUPT"));
        assert!(prompt.ends_with("]\n\n"));
    }
}
