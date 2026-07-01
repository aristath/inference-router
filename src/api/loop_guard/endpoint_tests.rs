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
