use std::collections::HashMap;

use serde_json::{json, Value};
use tracing::warn;

use crate::config::ToolLoopSettings;

#[derive(Clone, Debug, PartialEq, Eq)]
struct ToolCall {
    name: String,
    arguments: String,
}

#[derive(Clone, Debug, PartialEq, Eq)]
struct ToolEvent {
    name: String,
    arguments: String,
    output: String,
}

#[derive(Clone, Debug, PartialEq, Eq)]
struct Detection {
    period: usize,
    repeats: usize,
    block: Vec<ToolEvent>,
}

pub(super) fn guard_request(path: &str, body: &[u8], cfg: &ToolLoopSettings) -> Option<Vec<u8>> {
    if !path.ends_with("/chat/completions") {
        return None;
    }

    if !cfg.enabled {
        return None;
    }

    let mut doc: Value = serde_json::from_slice(body).ok()?;
    let messages = doc.get("messages").and_then(Value::as_array)?;
    let events = collect_tool_events(messages, cfg.window_messages.max(2));
    let detection = detect_repeated_suffix(&events, cfg.repeats.max(2))?;

    warn!(
        tools = %tool_sequence(&detection.block),
        period = detection.period,
        repeats = detection.repeats,
        "cross-turn tool loop detected; injecting corrective message",
    );

    inject_corrective(&mut doc, &detection)?;
    serde_json::to_vec(&doc).ok()
}

fn collect_tool_events(messages: &[Value], window_messages: usize) -> Vec<ToolEvent> {
    let start = messages.len().saturating_sub(window_messages);
    let messages = &messages[start..];
    let tail_start = messages
        .iter()
        .rposition(is_conversation_boundary)
        .map(|idx| idx + 1)
        .unwrap_or(0);
    let messages = &messages[tail_start..];
    let mut events = Vec::new();

    for (idx, msg) in messages.iter().enumerate() {
        if role(msg) != Some("assistant") {
            continue;
        }

        let calls = assistant_tool_calls(msg);
        if !calls.is_empty() {
            collect_openai_tool_results(messages, idx + 1, &calls, &mut events);
        }

        if let Some(call) = assistant_function_call(msg) {
            collect_legacy_function_result(messages, idx + 1, &call, &mut events);
        }
    }

    events
}

fn assistant_tool_calls(msg: &Value) -> HashMap<String, ToolCall> {
    let Some(items) = msg.get("tool_calls").and_then(Value::as_array) else {
        return HashMap::new();
    };
    let mut out = HashMap::new();
    for item in items {
        let Some(id) = item.get("id").and_then(Value::as_str) else {
            continue;
        };
        let Some(function) = item.get("function") else {
            continue;
        };
        let Some(name) = function.get("name").and_then(Value::as_str) else {
            continue;
        };
        let arguments = canonical_arguments(function.get("arguments"));
        out.insert(
            id.to_string(),
            ToolCall {
                name: name.to_string(),
                arguments,
            },
        );
    }
    out
}

fn assistant_function_call(msg: &Value) -> Option<ToolCall> {
    let function = msg.get("function_call")?;
    let name = function.get("name").and_then(Value::as_str)?;
    Some(ToolCall {
        name: name.to_string(),
        arguments: canonical_arguments(function.get("arguments")),
    })
}

fn collect_openai_tool_results(
    messages: &[Value],
    start: usize,
    calls: &HashMap<String, ToolCall>,
    events: &mut Vec<ToolEvent>,
) {
    for msg in &messages[start..] {
        match role(msg) {
            Some("tool") => {
                let Some(id) = msg.get("tool_call_id").and_then(Value::as_str) else {
                    continue;
                };
                let Some(call) = calls.get(id) else {
                    continue;
                };
                events.push(ToolEvent {
                    name: call.name.clone(),
                    arguments: call.arguments.clone(),
                    output: canonical_content(msg.get("content")),
                });
            }
            Some("function") => {}
            _ => break,
        }
    }
}

fn collect_legacy_function_result(
    messages: &[Value],
    start: usize,
    call: &ToolCall,
    events: &mut Vec<ToolEvent>,
) {
    for msg in &messages[start..] {
        match role(msg) {
            Some("function") => {
                let Some(name) = msg.get("name").and_then(Value::as_str) else {
                    continue;
                };
                if name != call.name {
                    continue;
                }
                events.push(ToolEvent {
                    name: call.name.clone(),
                    arguments: call.arguments.clone(),
                    output: canonical_content(msg.get("content")),
                });
                return;
            }
            Some("tool") => {}
            _ => break,
        }
    }
}

fn detect_repeated_suffix(events: &[ToolEvent], repeats: usize) -> Option<Detection> {
    if events.len() < repeats {
        return None;
    }

    for period in 1..=events.len() / repeats {
        let suffix_start = events.len() - period * repeats;
        let block = &events[events.len() - period..];
        let repeated = (0..repeats).all(|copy| {
            let start = suffix_start + copy * period;
            &events[start..start + period] == block
        });
        if !repeated {
            continue;
        }

        let mut count = repeats;
        while events.len() >= (count + 1) * period {
            let start = events.len() - (count + 1) * period;
            if &events[start..start + period] != block {
                break;
            }
            count += 1;
        }

        return Some(Detection {
            period,
            repeats: count,
            block: block.to_vec(),
        });
    }

    None
}

fn inject_corrective(doc: &mut Value, detection: &Detection) -> Option<()> {
    let messages = doc.get_mut("messages").and_then(Value::as_array_mut)?;
    messages.push(json!({
        "role": "user",
        "content": corrective_text(detection),
    }));
    Some(())
}

fn corrective_text(detection: &Detection) -> String {
    format!(
        "[INTERRUPT \u{2014} automated proxy notice] The recent conversation repeated the same tool cycle {repeats} times. Do not repeat this tool sequence with the same arguments: {tools}. Use a different concrete action, explain the blocker, or ask for the missing information. Last tool result(s): {outputs}",
        repeats = detection.repeats,
        tools = tool_sequence(&detection.block),
        outputs = tool_outputs(&detection.block),
    )
}

fn canonical_arguments(value: Option<&Value>) -> String {
    let Some(value) = value else {
        return String::new();
    };
    match value {
        Value::String(s) => canonical_json_string(s).unwrap_or_else(|| s.trim().to_string()),
        other => serde_json::to_string(other).unwrap_or_default(),
    }
}

fn canonical_content(value: Option<&Value>) -> String {
    let Some(value) = value else {
        return String::new();
    };
    match value {
        Value::String(s) => s.trim().to_string(),
        other => serde_json::to_string(other).unwrap_or_default(),
    }
}

fn canonical_json_string(s: &str) -> Option<String> {
    let parsed: Value = serde_json::from_str(s).ok()?;
    serde_json::to_string(&parsed).ok()
}

fn role(msg: &Value) -> Option<&str> {
    msg.get("role").and_then(Value::as_str)
}

fn is_conversation_boundary(msg: &Value) -> bool {
    matches!(role(msg), Some("user" | "system" | "developer"))
}

fn tool_sequence(events: &[ToolEvent]) -> String {
    events
        .iter()
        .map(|event| event.name.as_str())
        .collect::<Vec<_>>()
        .join(" -> ")
}

fn tool_outputs(events: &[ToolEvent]) -> String {
    events
        .iter()
        .map(|event| format!("{}: {}", event.name, truncate(&event.output, 240)))
        .collect::<Vec<_>>()
        .join(" | ")
}

fn truncate(s: &str, max: usize) -> String {
    if s.len() <= max {
        return s.to_string();
    }
    let mut end = max;
    while !s.is_char_boundary(end) {
        end -= 1;
    }
    format!("{}...", &s[..end])
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tool_cycle(call_id: &str, arguments: &str, output: &str) -> Vec<Value> {
        tool_cycle_named(call_id, "edit_file", arguments, output)
    }

    fn tool_cycle_named(call_id: &str, name: &str, arguments: &str, output: &str) -> Vec<Value> {
        vec![
            json!({
                "role": "assistant",
                "content": null,
                "tool_calls": [{
                    "id": call_id,
                    "type": "function",
                    "function": {
                        "name": name,
                        "arguments": arguments,
                    },
                }],
            }),
            json!({
                "role": "tool",
                "tool_call_id": call_id,
                "content": output,
            }),
        ]
    }

    #[test]
    fn detects_repeated_openai_tool_cycles_at_suffix() {
        let mut messages = vec![json!({"role":"user","content":"fix it"})];
        messages.extend(tool_cycle(
            "a",
            r#"{"path":"/tmp/a","old":"x"}"#,
            "no changes",
        ));
        messages.extend(tool_cycle(
            "b",
            r#"{ "old" : "x", "path" : "/tmp/a" }"#,
            "no changes",
        ));
        messages.extend(tool_cycle(
            "c",
            r#"{"path":"/tmp/a","old":"x"}"#,
            "no changes",
        ));

        let events = collect_tool_events(&messages, 80);
        let detection = detect_repeated_suffix(&events, 3).unwrap();
        assert_eq!(detection.period, 1);
        assert_eq!(detection.repeats, 3);
        assert_eq!(detection.block[0].name, "edit_file");
    }

    #[test]
    fn user_message_resets_autonomous_tail() {
        let mut messages = vec![json!({"role":"user","content":"fix it"})];
        messages.extend(tool_cycle("a", r#"{"path":"/tmp/a"}"#, "no changes"));
        messages.push(json!({"role":"user","content":"try that same edit again"}));
        messages.extend(tool_cycle("b", r#"{"path":"/tmp/a"}"#, "no changes"));
        messages.push(json!({"role":"user","content":"try that same edit a third time"}));
        messages.extend(tool_cycle("c", r#"{"path":"/tmp/a"}"#, "no changes"));

        let events = collect_tool_events(&messages, 80);
        assert_eq!(events.len(), 1);
        assert!(detect_repeated_suffix(&events, 3).is_none());
    }

    #[test]
    fn repeated_multi_tool_cycle_names_full_sequence() {
        let mut messages = vec![json!({"role":"user","content":"fix it"})];
        messages.extend(tool_cycle_named(
            "a",
            "read_file",
            r#"{"path":"/tmp/a"}"#,
            "old contents",
        ));
        messages.extend(tool_cycle_named(
            "b",
            "edit_file",
            r#"{"path":"/tmp/a","old":"x","new":"x"}"#,
            "no changes",
        ));
        messages.extend(tool_cycle_named(
            "c",
            "read_file",
            r#"{"path":"/tmp/a"}"#,
            "old contents",
        ));
        messages.extend(tool_cycle_named(
            "d",
            "edit_file",
            r#"{"path":"/tmp/a","old":"x","new":"x"}"#,
            "no changes",
        ));

        let events = collect_tool_events(&messages, 80);
        let detection = detect_repeated_suffix(&events, 2).unwrap();
        assert_eq!(detection.period, 2);
        assert_eq!(tool_sequence(&detection.block), "read_file -> edit_file");
        let text = corrective_text(&detection);
        assert!(text.contains("read_file -> edit_file"));
        assert!(text.contains("read_file: old contents"));
        assert!(text.contains("edit_file: no changes"));
    }

    #[test]
    fn different_tool_outputs_do_not_loop() {
        let mut messages = vec![json!({"role":"user","content":"fix it"})];
        messages.extend(tool_cycle("a", r#"{"path":"/tmp/a"}"#, "first error"));
        messages.extend(tool_cycle("b", r#"{"path":"/tmp/a"}"#, "second error"));
        messages.extend(tool_cycle("c", r#"{"path":"/tmp/a"}"#, "third error"));

        let events = collect_tool_events(&messages, 80);
        assert!(detect_repeated_suffix(&events, 3).is_none());
    }

    #[test]
    fn guard_injects_corrective_message() {
        let mut messages = vec![json!({"role":"user","content":"fix it"})];
        messages.extend(tool_cycle(
            "a",
            r#"{"path":"/tmp/a"}"#,
            "oldString and newString are identical",
        ));
        messages.extend(tool_cycle(
            "b",
            r#"{"path":"/tmp/a"}"#,
            "oldString and newString are identical",
        ));
        messages.extend(tool_cycle(
            "c",
            r#"{"path":"/tmp/a"}"#,
            "oldString and newString are identical",
        ));

        let body = json!({
            "model": "fake",
            "messages": messages,
        });
        let guarded = guard_request(
            "/v1/chat/completions",
            &serde_json::to_vec(&body).unwrap(),
            &ToolLoopSettings::default(),
        )
        .unwrap();
        let guarded: Value = serde_json::from_slice(&guarded).unwrap();
        let messages = guarded["messages"].as_array().unwrap();
        let last = messages.last().unwrap();
        assert_eq!(last["role"], "user");
        let content = last["content"].as_str().unwrap();
        assert!(content.contains("automated proxy notice"));
        assert!(content.contains("oldString and newString are identical"));
    }

    #[test]
    fn ignores_non_chat_paths() {
        let body = json!({"model":"fake","messages":[]});
        assert!(guard_request(
            "/v1/completions",
            &serde_json::to_vec(&body).unwrap(),
            &ToolLoopSettings::default(),
        )
        .is_none());
    }
}
