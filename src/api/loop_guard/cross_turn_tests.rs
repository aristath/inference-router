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
