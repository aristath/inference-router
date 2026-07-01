use super::*;

#[test]
fn finds_gguf_files_recursively_and_follows_symlinked_dirs_once() {
    let tmp = tempfile::tempdir().unwrap();
    let nested = tmp.path().join("nested");
    std::fs::create_dir(&nested).unwrap();
    std::fs::write(nested.join("a.gguf"), b"not really gguf").unwrap();
    std::fs::write(tmp.path().join("b.GGUF"), b"not really gguf").unwrap();
    std::fs::write(tmp.path().join("model-00001-of-00003.gguf"), b"1").unwrap();
    std::fs::write(tmp.path().join("model-00002-of-00003.gguf"), b"2").unwrap();
    std::fs::write(tmp.path().join("model-00003-of-00003.gguf"), b"3").unwrap();
    std::fs::write(tmp.path().join("c.txt"), b"nope").unwrap();

    #[cfg(unix)]
    std::os::unix::fs::symlink(&nested, tmp.path().join("nested-link")).unwrap();

    let mut errors = Vec::new();
    let files = find_gguf_files(tmp.path(), &mut errors);
    assert!(errors.is_empty());
    assert_eq!(files.len(), 3);
    assert!(files.iter().any(|p| p.ends_with("a.gguf")));
    assert!(files.iter().any(|p| p.ends_with("b.GGUF")));
    assert!(files
        .iter()
        .any(|p| p.ends_with("model-00001-of-00003.gguf")));
    assert!(!files
        .iter()
        .any(|p| p.ends_with("model-00002-of-00003.gguf")));
    assert!(!files
        .iter()
        .any(|p| p.ends_with("model-00003-of-00003.gguf")));
}

fn gguf_model(id: &str, path: &str, preset: Option<&str>) -> ModelConfig {
    let mut m = ModelConfig::default();
    m.id = id.to_string();
    m.weights_format = WeightsFormat::Gguf;
    m.model_path = PathBuf::from(path);
    m.binary_preset = preset.map(str::to_string);
    m
}

#[test]
fn nearest_template_matches_sibling_quant_dir() {
    let existing = vec![
        gguf_model(
            "qwen35-9b",
            "/models/qwen3.5/9b/UD-Q4_K_XL/Qwen3.5-9B.gguf",
            Some("llama-vulkan"),
        ),
        // Cousin under a different size tree — must NOT win.
        gguf_model(
            "qwen35-4b",
            "/models/qwen3.5/4b/UD-Q8_K_XL/Qwen3.5-4B.gguf",
            Some("llama-rocm"),
        ),
    ];
    let newcomer = Path::new("/models/qwen3.5/9b/QWOPUS-Q8_K_XL/Qwopus3.5-9B.gguf");
    let template = nearest_template(newcomer, &existing).expect("sibling template");
    assert_eq!(template.id, "qwen35-9b");
    assert_eq!(template.binary_preset.as_deref(), Some("llama-vulkan"));
}

#[test]
fn nearest_template_ignores_unrelated_subtree() {
    let existing = vec![gguf_model(
        "other",
        "/models/gemma/4b/Q8/Gemma.gguf",
        Some("llama-vulkan"),
    )];
    let newcomer = Path::new("/models/qwen3.5/9b/QWOPUS-Q8_K_XL/Qwopus3.5-9B.gguf");
    assert!(nearest_template(newcomer, &existing).is_none());
}

#[test]
fn missing_file_under_symlinked_root_still_appears_under_root() {
    // Reproduces the real bug: the scan folder is a symlink (~/models ->
    // /mnt/models/models) and a model's file (and its whole quant directory)
    // has been deleted. The under-root check must still say "yes, this was in
    // the folder" so it lands in the `missing` list.
    let tmp = tempfile::tempdir().unwrap();
    let real = tmp.path().join("real_models");
    std::fs::create_dir(&real).unwrap();
    let link = tmp.path().join("link_models");
    #[cfg(unix)]
    std::os::unix::fs::symlink(&real, &link).unwrap();

    // Path stored as the real target, with its directory gone (never created).
    let gone_real = real.join("Q8_0/model.gguf");
    assert!(!gone_real.exists());
    assert!(path_appears_under_root(&gone_real, &link));

    // Path stored via the symlink form (how a scan-added model looks).
    let gone_link = link.join("Q8_0/model.gguf");
    assert!(path_appears_under_root(&gone_link, &link));

    // A path in an unrelated tree is still correctly rejected.
    let outside = tmp.path().join("elsewhere/model.gguf");
    assert!(!path_appears_under_root(&outside, &link));
}

#[test]
fn unique_id_appends_numeric_suffixes() {
    let mut used = HashSet::from(["qwen".to_string(), "qwen-2".to_string()]);
    assert_eq!(unique_id("qwen", &mut used), "qwen-3");
    assert!(used.contains("qwen-3"));
}

#[test]
fn mmproj_files_are_detected_and_matched_only_when_unambiguous() {
    let tmp = tempfile::tempdir().unwrap();
    let model = tmp.path().join("vision.gguf");
    let mmproj = tmp.path().join("mmproj-vision.gguf");
    let other = tmp.path().join("other");
    std::fs::create_dir(&other).unwrap();
    let other_mmproj = other.join("mmproj-other.gguf");

    assert!(!is_mmproj_file(&model));
    assert!(is_mmproj_file(&mmproj));
    assert_eq!(
        matching_mmproj(&model, &[mmproj.clone(), other_mmproj.clone()]),
        Some(mmproj.clone())
    );

    let second = tmp.path().join("projector-extra.gguf");
    assert_eq!(matching_mmproj(&model, &[mmproj, second]), None);
}
