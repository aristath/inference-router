//! Parses streaming chunks from OpenAI-style chat-completions, OpenAI-style
//! legacy completions, and llama.cpp's native `/completion` endpoint into
//! per-choice deltas suitable for loop detection.
//!
//! The detector consumes every byte the assistant emits: content, reasoning,
//! and serialized tool-call deltas. Tool-call loops are real failure modes, and
//! excluding tool calls would let those through.

use serde::Deserialize;
use serde_json::value::RawValue;

/// Reports whether an SSE data payload is the OpenAI `[DONE]` sentinel.
pub(super) fn is_done(payload: &[u8]) -> bool {
    trim_ascii(payload) == b"[DONE]"
}

/// One choice's contribution to a single streamed chunk.
#[derive(Debug, Default, PartialEq, Eq)]
pub(super) struct ChoiceDelta {
    pub(super) index: i64,
    pub(super) content: String,
    pub(super) reasoning_content: String,
    /// Raw JSON of `choices[i].delta.tool_calls`, kept byte-stable so
    /// identical upstream chunks produce identical detector input.
    pub(super) tool_calls: Vec<u8>,
    #[allow(dead_code)]
    pub(super) stop: bool,
}

impl ChoiceDelta {
    /// Returns the bytes this delta contributes to the loop detector's rolling
    /// buffer, in fixed order: reasoning || content || tool_calls.
    #[allow(dead_code)]
    pub(super) fn detector_bytes(&self) -> Vec<u8> {
        let mut out = Vec::with_capacity(
            self.reasoning_content.len() + self.content.len() + self.tool_calls.len(),
        );
        out.extend_from_slice(self.reasoning_content.as_bytes());
        out.extend_from_slice(self.content.as_bytes());
        out.extend_from_slice(&self.tool_calls);
        out
    }
}

pub(super) fn extract_chat(payload: &[u8]) -> Vec<ChoiceDelta> {
    let payload = trim_ascii(payload);
    if payload.first() != Some(&b'{') {
        return Vec::new();
    }

    #[derive(Deserialize)]
    struct Event {
        choices: Vec<Choice>,
    }
    #[derive(Deserialize)]
    struct Choice {
        #[serde(default)]
        index: i64,
        #[serde(default)]
        delta: Delta,
    }
    #[derive(Default, Deserialize)]
    struct Delta {
        #[serde(default)]
        content: String,
        #[serde(default)]
        reasoning_content: String,
        #[serde(default)]
        thinking: String,
        #[serde(default)]
        tool_calls: Vec<Box<RawValue>>,
    }

    let Ok(ev) = serde_json::from_slice::<Event>(payload) else {
        return Vec::new();
    };

    let mut out = Vec::with_capacity(ev.choices.len());
    for choice in ev.choices {
        let mut tool_calls = Vec::new();
        for tc in choice.delta.tool_calls {
            tool_calls.extend_from_slice(tc.get().as_bytes());
        }
        let reasoning_content = choice.delta.reasoning_content + &choice.delta.thinking;
        if choice.delta.content.is_empty() && reasoning_content.is_empty() && tool_calls.is_empty()
        {
            continue;
        }
        out.push(ChoiceDelta {
            index: choice.index,
            content: choice.delta.content,
            reasoning_content,
            tool_calls,
            stop: false,
        });
    }
    out
}

pub(super) fn extract_completion(payload: &[u8]) -> Vec<ChoiceDelta> {
    let payload = trim_ascii(payload);
    if payload.first() != Some(&b'{') {
        return Vec::new();
    }

    #[derive(Deserialize)]
    struct Event {
        choices: Vec<Choice>,
    }
    #[derive(Deserialize)]
    struct Choice {
        #[serde(default)]
        index: i64,
        #[serde(default)]
        text: String,
    }

    let Ok(ev) = serde_json::from_slice::<Event>(payload) else {
        return Vec::new();
    };

    ev.choices
        .into_iter()
        .filter(|c| !c.text.is_empty())
        .map(|c| ChoiceDelta {
            index: c.index,
            content: c.text,
            ..ChoiceDelta::default()
        })
        .collect()
}

pub(super) fn extract_llama_cpp(payload: &[u8]) -> Vec<ChoiceDelta> {
    let payload = trim_ascii(payload);
    if payload.first() != Some(&b'{') {
        return Vec::new();
    }

    #[derive(Deserialize)]
    struct Event {
        #[serde(default)]
        content: String,
        #[serde(default)]
        stop: bool,
    }

    let Ok(ev) = serde_json::from_slice::<Event>(payload) else {
        return Vec::new();
    };
    if ev.content.is_empty() && !ev.stop {
        return Vec::new();
    }
    vec![ChoiceDelta {
        index: 0,
        content: ev.content,
        stop: ev.stop,
        ..ChoiceDelta::default()
    }]
}

fn trim_ascii(mut bytes: &[u8]) -> &[u8] {
    while bytes.first().is_some_and(u8::is_ascii_whitespace) {
        bytes = &bytes[1..];
    }
    while bytes.last().is_some_and(u8::is_ascii_whitespace) {
        bytes = &bytes[..bytes.len() - 1];
    }
    bytes
}

pub(super) struct EventParser {
    event_buf: Vec<u8>,
    line_buf: Vec<u8>,
}

impl EventParser {
    pub(super) fn new() -> Self {
        Self {
            event_buf: Vec::new(),
            line_buf: Vec::new(),
        }
    }

    pub(super) fn push(&mut self, chunk: &[u8]) -> Vec<Vec<u8>> {
        let mut events = Vec::new();
        for &b in chunk {
            self.line_buf.push(b);
            if b == b'\n' {
                if let Some(payload) = self.process_line() {
                    events.push(payload);
                }
            }
        }
        events
    }

    pub(super) fn finish(&mut self) -> Vec<Vec<u8>> {
        let mut events = Vec::new();
        if !self.line_buf.is_empty() {
            if let Some(payload) = self.process_line() {
                events.push(payload);
            }
        }
        if !self.event_buf.is_empty() {
            events.push(std::mem::take(&mut self.event_buf));
        }
        events
    }

    fn process_line(&mut self) -> Option<Vec<u8>> {
        let mut line = std::mem::take(&mut self.line_buf);
        while matches!(line.last(), Some(b'\n' | b'\r')) {
            line.pop();
        }
        if line.is_empty() {
            if self.event_buf.is_empty() {
                return None;
            }
            return Some(std::mem::take(&mut self.event_buf));
        }
        if let Some(data) = line.strip_prefix(b"data:") {
            if !self.event_buf.is_empty() {
                self.event_buf.push(b'\n');
            }
            self.event_buf.extend_from_slice(trim_ascii(data));
        }
        None
    }
}

#[cfg(test)]
#[path = "sse_tests.rs"]
mod tests;
