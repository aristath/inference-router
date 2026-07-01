use super::*;

#[test]
fn extracts_chat_content_reasoning_thinking_and_tools() {
    let got = extract_chat(
            br#"{"choices":[{"index":0,"delta":{"content":"C","reasoning_content":"R","thinking":"T","tool_calls":[{"index":0,"function":{"name":"read","arguments":"{\"path\":\"/x\"}"}}]}}]}"#,
        );
    assert_eq!(got.len(), 1);
    assert_eq!(got[0].index, 0);
    assert_eq!(got[0].content, "C");
    assert_eq!(got[0].reasoning_content, "RT");
    assert!(String::from_utf8_lossy(&got[0].tool_calls).contains(r#""name":"read""#));
    assert_eq!(
        String::from_utf8_lossy(&got[0].detector_bytes()),
        format!("RTC{}", String::from_utf8_lossy(&got[0].tool_calls))
    );
}

#[test]
fn parses_multiline_sse_event() {
    let mut p = EventParser::new();
    let events = p.push(b": keepalive\n\ndata: {\"a\":1}\ndata: {\"b\":2}\n\n");
    assert_eq!(events, vec![b"{\"a\":1}\n{\"b\":2}".to_vec()]);
}

#[test]
fn done_sentinel() {
    assert!(is_done(b" [DONE]\n"));
    assert!(!is_done(b"[done]"));
}
