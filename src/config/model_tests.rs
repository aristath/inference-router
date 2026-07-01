use super::*;

fn sample() -> ModelConfig {
    ModelConfig {
        id: "qwen3-30b".into(),
        name: "Qwen3 30B".into(),
        profile: Some("coding".into()),
        binary_preset: Some("llama-vulkan".into()),
        binary: PathBuf::from("/home/u/llama.cpp/build-vulkan/bin/llama-server"),
        model_path: PathBuf::from("/models/qwen3-30b.gguf"),
        mmproj_path: Some(PathBuf::from("/models/mmproj.gguf")),
        extra_args: vec!["--override-kv".into(), "something=1".into()],
        context: 32768,
        flash_attn: true,
        n_gpu_layers: Some(99),
        mlock: true,
        parallel_slots: Some(4),
        cache_type_k: Some(CacheType::Q8_0),
        cache_type_v: Some(CacheType::Q8_0),
        split_mode: Some(SplitMode::Layer),
        main_gpu: Some(0),
        tensor_split: Some("0.5,0.5,0".into()),
        threads: Some(16),
        cache_ram_mib: Some(0),
        reasoning_format: Some(ReasoningFormat::Auto),
        reasoning_budget: Some(-1),
        chat_template_kwargs: Some(r#"{"enable_thinking":true}"#.into()),
        mtp_tokens: Some(4),
        ..ModelConfig::default()
    }
}

#[test]
fn serde_roundtrip_preserves_public_fields_and_drops_internal_placement() {
    let original = sample();
    let json = serde_json::to_string(&original).unwrap();
    let parsed: ModelConfig = serde_json::from_str(&json).unwrap();
    let mut expected = original;
    expected.n_gpu_layers = None;
    expected.n_cpu_moe = None;
    expected.override_tensor = None;
    expected.fit_target = None;
    expected.split_mode = None;
    expected.main_gpu = None;
    expected.tensor_split = None;
    expected.device = None;
    assert_eq!(expected, parsed);
}

#[test]
fn weights_format_serializes_lowercase() {
    let gguf = serde_json::to_string(&WeightsFormat::Gguf).unwrap();
    let safe = serde_json::to_string(&WeightsFormat::Safetensors).unwrap();
    assert_eq!(gguf, "\"gguf\"");
    assert_eq!(safe, "\"safetensors\"");
}

#[test]
fn runtime_fields_default_when_absent() {
    let json = r#"{
            "id": "m", "name": "M",
            "weights_format": "gguf",
            "binary": "/bin/llama", "model_path": "/m.gguf",
            "port": 9001, "context": 4096
        }"#;
    let parsed: ModelConfig = serde_json::from_str(json).unwrap();
    assert_eq!(parsed.state, ModelState::Idle);
    assert_eq!(parsed.pid, None);
    assert_eq!(parsed.estimated_vram, 0);
    assert_eq!(parsed.last_used, None);
    assert_eq!(parsed.profile, None);
    assert_eq!(parsed.extra_args, Vec::<String>::new());
    assert_eq!(parsed.temperature, 0.6);
    assert_eq!(parsed.top_p, 0.95);
    assert_eq!(parsed.top_k, 40);
    assert_eq!(parsed.min_p, 0.0);
    // New llama.cpp flags all default to off/unset.
    assert!(!parsed.flash_attn);
    assert_eq!(parsed.n_gpu_layers, None);
    assert!(!parsed.mlock);
    assert!(!parsed.no_mmap);
    assert_eq!(parsed.parallel_slots, None);
    assert_eq!(parsed.cache_type_k, None);
    assert_eq!(parsed.cache_type_v, None);
    assert_eq!(parsed.split_mode, None);
    assert_eq!(parsed.main_gpu, None);
    assert_eq!(parsed.tensor_split, None);
    // Post-migration structured fields default to unset / neutral.
    assert_eq!(parsed.threads, None);
    assert_eq!(parsed.cache_ram_mib, None);
    assert_eq!(parsed.reasoning_format, None);
    assert_eq!(parsed.reasoning_budget, None);
    assert_eq!(parsed.chat_template_kwargs, None);
    assert_eq!(parsed.presence_penalty, 0.0);
    assert_eq!(parsed.repeat_penalty, 1.0);
    // Spec-decode fields default to off/unset.
    assert_eq!(parsed.device, None);
    assert_eq!(parsed.draft_model_id, None);
    assert_eq!(parsed.mtp_tokens, None);
    assert_eq!(parsed.draft_max, None);
    assert_eq!(parsed.draft_min, None);
    assert_eq!(parsed.draft_p_min, None);
    assert_eq!(parsed.ctx_checkpoints, None);
    assert_eq!(parsed.checkpoint_every_n_tokens, None);
}

#[test]
fn manual_placement_fields_are_ignored_from_config() {
    let json = r#"{
            "id": "m", "name": "M",
            "weights_format": "gguf",
            "binary": "/bin/llama", "model_path": "/m.gguf",
            "context": 4096,
            "n_gpu_layers": 99,
            "n_cpu_moe": 13,
            "override_tensor": "blk\\..*=CPU",
            "fit_target": "1024",
            "split_mode": "row",
            "main_gpu": 1,
            "tensor_split": "1,1",
            "device": "Vulkan1"
        }"#;
    let parsed: ModelConfig = serde_json::from_str(json).unwrap();
    assert_eq!(parsed.n_gpu_layers, None);
    assert_eq!(parsed.n_cpu_moe, None);
    assert_eq!(parsed.override_tensor, None);
    assert_eq!(parsed.fit_target, None);
    assert_eq!(parsed.split_mode, None);
    assert_eq!(parsed.main_gpu, None);
    assert_eq!(parsed.tensor_split, None);
    assert_eq!(parsed.device, None);
}

#[test]
fn reasoning_format_serializes_kebab_case() {
    assert_eq!(
        serde_json::to_string(&ReasoningFormat::None).unwrap(),
        "\"none\""
    );
    assert_eq!(
        serde_json::to_string(&ReasoningFormat::Auto).unwrap(),
        "\"auto\""
    );
    assert_eq!(
        serde_json::to_string(&ReasoningFormat::Deepseek).unwrap(),
        "\"deepseek\""
    );
    assert_eq!(
        serde_json::to_string(&ReasoningFormat::DeepseekLegacy).unwrap(),
        "\"deepseek-legacy\"",
    );
}

#[test]
fn reasoning_format_from_cli_covers_all_values() {
    assert_eq!(
        ReasoningFormat::from_cli("none"),
        Some(ReasoningFormat::None)
    );
    assert_eq!(
        ReasoningFormat::from_cli("auto"),
        Some(ReasoningFormat::Auto)
    );
    assert_eq!(
        ReasoningFormat::from_cli("deepseek"),
        Some(ReasoningFormat::Deepseek)
    );
    assert_eq!(
        ReasoningFormat::from_cli("deepseek-legacy"),
        Some(ReasoningFormat::DeepseekLegacy),
    );
    assert_eq!(ReasoningFormat::from_cli("bogus"), None);
}

// ----- Migration -----

fn bare() -> ModelConfig {
    ModelConfig {
        id: "m".into(),
        name: "M".into(),
        binary: PathBuf::from("/bin/llama"),
        model_path: PathBuf::from("/m.gguf"),
        ..ModelConfig::default()
    }
}

#[test]
fn migrate_extracts_all_seven_flags() {
    let mut m = bare();
    m.extra_args = vec![
        "--threads".into(),
        "16".into(),
        "--reasoning-format".into(),
        "auto".into(),
        "--cache-ram".into(),
        "0".into(),
        "--presence-penalty".into(),
        "1.5".into(),
        "--repeat-penalty".into(),
        "1.0".into(),
        "--reasoning-budget".into(),
        "0".into(),
        "--chat-template-kwargs".into(),
        r#"{"enable_thinking":false}"#.into(),
        "--mmproj".into(),
        "/models/mmproj.gguf".into(),
    ];
    let original_extra_args = m.extra_args.clone();
    assert!(m.migrate_extra_args());
    assert_eq!(m.threads, Some(16));
    assert_eq!(m.reasoning_format, Some(ReasoningFormat::Auto));
    assert_eq!(m.cache_ram_mib, Some(0));
    assert_eq!(m.presence_penalty, 1.5);
    assert_eq!(m.repeat_penalty, 1.0);
    assert_eq!(m.reasoning_budget, Some(0));
    assert_eq!(
        m.chat_template_kwargs.as_deref(),
        Some(r#"{"enable_thinking":false}"#)
    );
    assert_eq!(
        m.mmproj_path.as_deref(),
        Some(std::path::Path::new("/models/mmproj.gguf"))
    );
    assert_eq!(m.extra_args, original_extra_args);
}

#[test]
fn migrate_preserves_unknown_flags() {
    let mut m = bare();
    m.extra_args = vec![
        "--override-kv".into(),
        "foo=bar".into(),
        "--threads".into(),
        "16".into(),
        "--custom-flag".into(),
    ];
    let original_extra_args = m.extra_args.clone();
    assert!(m.migrate_extra_args());
    assert_eq!(m.threads, Some(16));
    assert_eq!(m.extra_args, original_extra_args);
}

#[test]
fn migrate_preserves_manual_placement_flags_in_extra_args() {
    let mut m = bare();
    m.extra_args = vec![
        "-ngl".into(),
        "99".into(),
        "--tensor-split=1,1".into(),
        "--device".into(),
        "Vulkan1".into(),
        "--override-tensor".into(),
        "blk\\..*=CPU".into(),
        "--custom".into(),
        "value".into(),
    ];
    assert!(!m.migrate_extra_args());
    assert_eq!(
        m.extra_args,
        vec![
            "-ngl",
            "99",
            "--tensor-split=1,1",
            "--device",
            "Vulkan1",
            "--override-tensor",
            "blk\\..*=CPU",
            "--custom",
            "value",
        ]
    );
}

#[test]
fn migrate_is_idempotent_after_first_pass() {
    let mut m = bare();
    m.extra_args = vec!["--threads".into(), "16".into()];
    let original_extra_args = m.extra_args.clone();
    assert!(m.migrate_extra_args());
    // Second pass: nothing to migrate, nothing changes.
    assert!(!m.migrate_extra_args());
    assert_eq!(m.threads, Some(16));
    assert_eq!(m.extra_args, original_extra_args);
}

#[test]
fn migrate_keeps_existing_structured_value_on_conflict() {
    let mut m = bare();
    m.threads = Some(32);
    m.extra_args = vec!["--threads".into(), "16".into()];
    assert!(!m.migrate_extra_args());
    assert_eq!(m.threads, Some(32));
    assert_eq!(m.extra_args, vec!["--threads", "16"]);
}

#[test]
fn migrate_handles_short_aliases() {
    let mut m = bare();
    m.extra_args = vec!["-t".into(), "8".into(), "-cram".into(), "4096".into()];
    let original_extra_args = m.extra_args.clone();
    assert!(m.migrate_extra_args());
    assert_eq!(m.threads, Some(8));
    assert_eq!(m.cache_ram_mib, Some(4096));
    assert_eq!(m.extra_args, original_extra_args);
}

#[test]
fn migrate_ignores_flag_with_unparseable_value() {
    let mut m = bare();
    m.extra_args = vec!["--threads".into(), "not-a-number".into()];
    assert!(!m.migrate_extra_args());
    assert_eq!(m.threads, None);
    assert_eq!(m.extra_args, vec!["--threads", "not-a-number"]);
}

#[test]
fn migrate_rejects_unknown_reasoning_format_value() {
    let mut m = bare();
    m.extra_args = vec!["--reasoning-format".into(), "made-up".into()];
    assert!(!m.migrate_extra_args());
    assert_eq!(m.reasoning_format, None);
    assert_eq!(m.extra_args, vec!["--reasoning-format", "made-up"]);
}

#[test]
fn migrate_real_world_qwen3_args() {
    // Taken verbatim from the user's models.json.
    let mut m = bare();
    m.extra_args = vec![
        "--threads".into(),
        "16".into(),
        "--reasoning-format".into(),
        "auto".into(),
        "--cache-ram".into(),
        "0".into(),
        "--presence-penalty".into(),
        "1.5".into(),
        "--repeat-penalty".into(),
        "1.0".into(),
        "--chat-template-kwargs".into(),
        r#"{"enable_thinking":false}"#.into(),
    ];
    let original_extra_args = m.extra_args.clone();
    assert!(m.migrate_extra_args());
    assert_eq!(m.extra_args, original_extra_args);
    assert_eq!(m.threads, Some(16));
    assert_eq!(m.reasoning_format, Some(ReasoningFormat::Auto));
    assert_eq!(m.cache_ram_mib, Some(0));
    assert_eq!(m.presence_penalty, 1.5);
    assert_eq!(m.repeat_penalty, 1.0);
    assert_eq!(
        m.chat_template_kwargs.as_deref(),
        Some(r#"{"enable_thinking":false}"#)
    );
}

#[test]
fn split_mode_serializes_lowercase() {
    assert_eq!(serde_json::to_string(&SplitMode::None).unwrap(), "\"none\"");
    assert_eq!(
        serde_json::to_string(&SplitMode::Layer).unwrap(),
        "\"layer\""
    );
    assert_eq!(serde_json::to_string(&SplitMode::Row).unwrap(), "\"row\"");
}

#[test]
fn cache_type_serializes_lowercase() {
    assert_eq!(serde_json::to_string(&CacheType::F16).unwrap(), "\"f16\"");
    assert_eq!(serde_json::to_string(&CacheType::Q8_0).unwrap(), "\"q8_0\"");
    assert_eq!(serde_json::to_string(&CacheType::Q4_0).unwrap(), "\"q4_0\"");
}

#[test]
fn error_state_roundtrips_with_message() {
    let mut m = sample();
    m.state = ModelState::Error("process 1234 died".into());
    let json = serde_json::to_string(&m).unwrap();
    let parsed: ModelConfig = serde_json::from_str(&json).unwrap();
    assert_eq!(parsed.state, ModelState::Error("process 1234 died".into()));
}

// ----- Speculative decoding -----

#[test]
fn migrate_extracts_spec_decode_policy_flags() {
    let mut m = bare();
    m.extra_args = vec![
        "--spec-draft-n-max".into(),
        "16".into(),
        "--spec-draft-n-min".into(),
        "1".into(),
        "--spec-draft-p-min".into(),
        "0.75".into(),
        "--ctx-checkpoints".into(),
        "4".into(),
        "--checkpoint-every-n-tokens".into(),
        "-1".into(),
    ];
    let original_extra_args = m.extra_args.clone();
    assert!(m.migrate_extra_args());
    assert_eq!(m.draft_max, Some(16));
    assert_eq!(m.draft_min, Some(1));
    assert_eq!(m.draft_p_min, Some(0.75));
    assert_eq!(m.ctx_checkpoints, Some(4));
    assert_eq!(m.checkpoint_every_n_tokens, Some(-1));
    assert_eq!(m.extra_args, original_extra_args);
}

#[test]
fn migrate_extracts_mtp_spec_decode_flags() {
    let mut m = bare();
    m.extra_args = vec![
        "--spec-draft-n-max".into(),
        "4".into(),
        "--spec-type=draft-mtp".into(),
    ];
    let original_extra_args = m.extra_args.clone();
    assert!(m.migrate_extra_args());
    assert_eq!(m.mtp_tokens, Some(4));
    assert_eq!(m.draft_max, None);
    assert_eq!(m.extra_args, original_extra_args);
}

#[test]
fn migrate_mtp_spec_type_without_count_preserves_llama_default() {
    let mut m = bare();
    m.extra_args = vec!["--spec-type".into(), "draft-mtp".into()];
    let original_extra_args = m.extra_args.clone();
    assert!(m.migrate_extra_args());
    assert_eq!(m.mtp_tokens, Some(3));
    assert_eq!(m.extra_args, original_extra_args);
}

#[test]
fn migrate_leaves_draft_path_flags_alone() {
    // `-md`, `-ngld`, `-devd`, `-ctkd`, `-ctvd` can't be
    // auto-migrated because they reference a draft GGUF by path —
    // but the new model requires drafts to be ModelConfig entries
    // (addressable by id). Preserve them in extra_args so the user
    // can see what needs to be reconstructed as a draft entry.
    let mut m = bare();
    m.extra_args = vec![
        "-md".into(),
        "/m/draft.gguf".into(),
        "-ngld".into(),
        "99".into(),
        "-devd".into(),
        "Vulkan1".into(),
    ];
    assert!(!m.migrate_extra_args());
    assert_eq!(
        m.extra_args,
        vec!["-md", "/m/draft.gguf", "-ngld", "99", "-devd", "Vulkan1"],
    );
}

#[test]
fn migrate_accepts_draft_max_aliases() {
    // llama.cpp aliases: --draft / --draft-n / --draft-max
    let mut m = bare();
    m.extra_args = vec!["--draft".into(), "8".into()];
    assert!(m.migrate_extra_args());
    assert_eq!(m.draft_max, Some(8));

    let mut m = bare();
    m.extra_args = vec!["--draft-n".into(), "12".into()];
    assert!(m.migrate_extra_args());
    assert_eq!(m.draft_max, Some(12));
}
