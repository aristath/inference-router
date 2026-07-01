use super::*;

#[test]
fn roundtrips_through_serde() {
    let p = BinaryPreset {
        id: "llama-vulkan".into(),
        name: "llama.cpp (Vulkan)".into(),
        binary: PathBuf::from("/home/aristath/llama.cpp/build-vulkan/bin/llama-server"),
        targets: vec![Backend::Vulkan],
    };
    let s = serde_json::to_string(&p).unwrap();
    let back: BinaryPreset = serde_json::from_str(&s).unwrap();
    assert_eq!(p, back);
}

#[test]
fn legacy_preset_without_targets_infers_one_from_path() {
    let json = r#"{"id":"llama-rocm","name":"llama.cpp (ROCm)","binary":"/x/llama.cpp-rocm/extracted/llama-server"}"#;
    let p: BinaryPreset = serde_json::from_str(json).unwrap();
    assert!(p.targets.is_empty());
    assert_eq!(p.effective_targets(), vec![Backend::Rocm]);
}
