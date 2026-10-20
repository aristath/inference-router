use super::*;

#[test]
fn roundtrips_through_serde() {
    let a = ModelAlias {
        fallback: None,
        alias: "gpt-4o".into(),
        target: "qwen3-32b".into(),
    };
    let s = serde_json::to_string(&a).unwrap();
    let back: ModelAlias = serde_json::from_str(&s).unwrap();
    assert_eq!(a, back);
}

#[test]
fn old_alias_files_have_no_fallback_and_new_files_preserve_it() {
    let old: ModelAlias = serde_json::from_str(r#"{"alias":"default","target":"gpu"}"#).unwrap();
    assert_eq!(old.fallback, None);
    let mut new = old;
    new.fallback = Some("fast".into());
    let saved = serde_json::to_string(&new).unwrap();
    assert_eq!(serde_json::from_str::<ModelAlias>(&saved).unwrap(), new);
}
