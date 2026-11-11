use super::*;

#[test]
fn roundtrips_through_serde() {
    let a = ModelAlias {
        slot: None,
        slots: vec![],
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

#[test]
fn slot_zero_and_explicit_slot_survive_storage() {
    for slot in [0, 1] {
        let input = format!(r#"{{"alias":"compaction","target":"default","slot":{slot}}}"#);
        let value: ModelAlias = serde_json::from_str(&input).unwrap();
        assert_eq!(value.slot, Some(slot));
        let restored: ModelAlias =
            serde_json::from_str(&serde_json::to_string(&value).unwrap()).unwrap();
        assert_eq!(restored.slot, Some(slot));
    }
    let old: ModelAlias = serde_json::from_str(r#"{"alias":"default","target":"gpu"}"#).unwrap();
    assert_eq!(old.slot, None);
}

#[test]
fn configured_slot_pool_survives_storage() {
    let value: ModelAlias =
        serde_json::from_str(r#"{"alias":"default","target":"gpu","slots":[0,1]}"#).unwrap();
    assert_eq!(value.slots, vec![0, 1]);
    assert_eq!(
        serde_json::from_str::<ModelAlias>(&serde_json::to_string(&value).unwrap()).unwrap(),
        value
    );
}
