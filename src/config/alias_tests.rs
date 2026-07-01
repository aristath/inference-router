use super::*;

#[test]
fn roundtrips_through_serde() {
    let a = ModelAlias {
        alias: "gpt-4o".into(),
        target: "qwen3-32b".into(),
    };
    let s = serde_json::to_string(&a).unwrap();
    let back: ModelAlias = serde_json::from_str(&s).unwrap();
    assert_eq!(a, back);
}
