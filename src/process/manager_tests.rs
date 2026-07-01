use super::*;
use crate::config::{CacheType, ReasoningFormat};

impl ProcessManager {
    fn reserve_test_pending(&mut self, model_id: &str, port: u16) {
        self.reserved_ports.insert(port);
        *self.pending_instances.entry(model_id.into()).or_insert(0) += 1;
    }
}

#[test]
fn parses_standard_kv_cache_line() {
    let line = "llama_kv_cache_init: size = 1024.00 MiB ( 32768 cells, 64 layers, 2/2 seqs), K (q8_0): 512.00 MiB, V (q8_0): 512.00 MiB";
    let mib = parse_kv_size_mib(line).unwrap();
    assert!((mib - 1024.0).abs() < 0.01, "got {mib}");
}

#[test]
fn parses_recurrent_state_line() {
    let line = "llama_memory_recurrent_init: size =   56.25 MiB (  8192 cells, 28 layers, 2 seqs), R (f16):   28.12 MiB, S (f16):   28.12 MiB";
    let mib = parse_kv_size_mib(line).unwrap();
    assert!((mib - 56.25).abs() < 0.01, "got {mib}");
}

#[test]
fn parses_swa_kv_cache_line() {
    let line = "llama_kv_cache_init: size =  128.50 MiB (  4096 cells, 32 layers, 2/2 seqs), K (q8_0):  64.25 MiB, V (q8_0):  64.25 MiB";
    assert!(parse_kv_size_mib(line).is_some());
}

#[test]
fn ignores_quant_progress_lines() {
    let line = "[ 100/ 291]          blk.0.attn_k.weight - [ 4096,  4096,    1,    1], type =    q8_0, size =   16.00 MiB";
    assert!(parse_kv_size_mib(line).is_none());
}

#[test]
fn ignores_model_buffer_lines() {
    let line = "llm_load_tensors:      VULKAN0 model buffer size =  8192.00 MiB";
    assert!(parse_kv_size_mib(line).is_none());
}

#[test]
fn ignores_unrelated_lines() {
    assert!(parse_kv_size_mib("llama_new_context_with_model: n_ctx = 262144").is_none());
    assert!(parse_kv_size_mib("srv  log: HTTP server is listening").is_none());
}

fn gguf_model() -> ModelConfig {
    ModelConfig {
        id: "m".into(),
        name: "M".into(),
        binary: PathBuf::from("/usr/local/bin/llama-server"),
        model_path: PathBuf::from("/models/m.gguf"),
        ..ModelConfig::default()
    }
}

#[test]
fn gguf_argv_shape() {
    let args = build_command_args(&gguf_model(), None, 9001);
    assert_eq!(
        args,
        vec![
            "-m",
            "/models/m.gguf",
            "-c",
            "4096",
            "--port",
            "9001",
            "--temp",
            "0.6",
            "--top-p",
            "0.95",
            "--top-k",
            "40",
            "--presence-penalty",
            "0",
            "--repeat-penalty",
            "1",
            "--flash-attn",
            "off",
        ],
    );
}

#[test]
fn gguf_argv_includes_min_p_when_positive() {
    let mut m = gguf_model();
    m.min_p = 0.05;
    let args = build_command_args(&m, None, 9001);
    assert!(args.windows(2).any(|w| w == ["--min-p", "0.05"]));
}

#[test]
fn gguf_argv_omits_min_p_when_zero() {
    let args = build_command_args(&gguf_model(), None, 9001);
    assert!(!args.iter().any(|a| a == "--min-p"));
}

#[test]
fn safetensors_argv_shape() {
    let mut m = gguf_model();
    m.weights_format = WeightsFormat::Safetensors;
    m.model_path = PathBuf::from("/models/m-safetensors");
    let args = build_command_args(&m, None, 9001);
    assert_eq!(
        args,
        vec![
            "--model",
            "/models/m-safetensors",
            "--port",
            "9001",
            "--max-model-len",
            "4096",
        ],
    );
}

#[test]
fn extra_args_appended_last_for_gguf_without_filtering() {
    let mut m = gguf_model();
    m.extra_args = vec![
        "--flash-attn".into(),
        "-ngl".into(),
        "99".into(),
        "--tensor-split=1,1".into(),
        "--custom".into(),
        "value".into(),
    ];
    let args = build_command_args(&m, None, 9001);
    assert_eq!(
        &args[args.len() - 6..],
        &[
            "--flash-attn",
            "-ngl",
            "99",
            "--tensor-split=1,1",
            "--custom",
            "value",
        ]
    );
}

#[test]
fn gguf_argv_emits_structured_llama_flags() {
    let mut m = gguf_model();
    m.flash_attn = true;
    m.n_gpu_layers = Some(99);
    m.mlock = true;
    m.no_mmap = true;
    m.parallel_slots = Some(4);
    m.cache_type_k = Some(CacheType::Q8_0);
    m.cache_type_v = Some(CacheType::Q8_0);
    let args = build_command_args(&m, None, 9001);
    let joined = args.join(" ");
    assert!(joined.contains("--flash-attn on"), "{joined}");
    assert!(joined.contains("-ngl 99"), "{joined}");
    assert!(joined.contains("--mlock"), "{joined}");
    assert!(joined.contains("--no-mmap"), "{joined}");
    assert!(joined.contains("--parallel 4"), "{joined}");
    assert!(joined.contains("--cache-type-k q8_0"), "{joined}");
    assert!(joined.contains("--cache-type-v q8_0"), "{joined}");
}

#[test]
fn gguf_argv_emits_flash_attn_off_when_disabled() {
    let args = build_command_args(&gguf_model(), None, 9001);
    assert_eq!(
        args.windows(2)
            .find(|w| w[0] == "--flash-attn")
            .map(|w| w[1].as_str()),
        Some("off"),
    );
}

#[test]
fn gguf_argv_omits_other_structured_flags_when_unset() {
    let args = build_command_args(&gguf_model(), None, 9001);
    let joined = args.join(" ");
    assert!(!joined.contains("-ngl"));
    assert!(!joined.contains("--mlock"));
    assert!(!joined.contains("--no-mmap"));
    assert!(!joined.contains("--parallel"));
    assert!(!joined.contains("--cache-type-k"));
    assert!(!joined.contains("--cache-type-v"));
    assert!(!joined.contains("--split-mode"));
    assert!(!joined.contains("--main-gpu"));
    assert!(!joined.contains("--tensor-split"));
    assert!(!joined.contains("--device"));
}

#[test]
fn gguf_argv_emits_penalties_from_structured_fields() {
    let mut m = gguf_model();
    m.presence_penalty = 1.5;
    m.repeat_penalty = 1.1;
    let args = build_command_args(&m, None, 9001);
    let joined = args.join(" ");
    assert!(joined.contains("--presence-penalty 1.5"), "{joined}");
    assert!(joined.contains("--repeat-penalty 1.1"), "{joined}");
}

#[test]
fn gguf_argv_emits_threads_cache_ram_reasoning_when_set() {
    let mut m = gguf_model();
    m.threads = Some(16);
    m.cache_ram_mib = Some(0);
    m.reasoning_format = Some(ReasoningFormat::Deepseek);
    m.reasoning_budget = Some(0);
    m.chat_template_kwargs = Some(r#"{"enable_thinking":false}"#.into());
    let args = build_command_args(&m, None, 9001);
    let joined = args.join(" ");
    assert!(joined.contains("--threads 16"), "{joined}");
    assert!(joined.contains("--cache-ram 0"), "{joined}");
    assert!(joined.contains("--reasoning-format deepseek"), "{joined}");
    assert!(joined.contains("--reasoning-budget 0"), "{joined}");
    assert!(
        joined.contains(r#"--chat-template-kwargs {"enable_thinking":false}"#),
        "{joined}"
    );
}

#[test]
fn gguf_argv_omits_new_flags_when_unset() {
    let args = build_command_args(&gguf_model(), None, 9001);
    let joined = args.join(" ");
    assert!(!joined.contains("--threads"));
    assert!(!joined.contains("--cache-ram"));
    assert!(!joined.contains("--reasoning-format"));
    assert!(!joined.contains("--reasoning-budget"));
    assert!(!joined.contains("--chat-template-kwargs"));
}

#[test]
fn gguf_argv_reasoning_format_kebab_for_deepseek_legacy() {
    let mut m = gguf_model();
    m.reasoning_format = Some(ReasoningFormat::DeepseekLegacy);
    let args = build_command_args(&m, None, 9001);
    let joined = args.join(" ");
    assert!(
        joined.contains("--reasoning-format deepseek-legacy"),
        "{joined}"
    );
}

#[test]
fn gguf_argv_emits_fitted_tensor_split_but_not_manual_split_knobs() {
    let mut m = gguf_model();
    m.split_mode = Some(crate::config::SplitMode::Row);
    m.main_gpu = Some(2);
    m.tensor_split = Some("0.5,0.5,0".into());
    let args = build_command_args(&m, None, 9001);
    let j = args.join(" ");
    assert!(!j.contains("--split-mode"), "{j}");
    assert!(!j.contains("--main-gpu"), "{j}");
    assert!(j.contains("--tensor-split 0.5,0.5,0"), "{j}");
}

#[test]
fn gguf_argv_emits_target_device_when_set() {
    let mut m = gguf_model();
    m.device = Some("Vulkan1,Vulkan2".into());
    let args = build_command_args(&m, None, 9001);
    assert_eq!(find_flag(&args, "--device"), Some("Vulkan1,Vulkan2"));
}

#[test]
fn gguf_argv_prefers_override_tensor_over_n_cpu_moe() {
    let mut m = gguf_model();
    m.n_cpu_moe = Some(23);
    m.override_tensor = Some(r"blk\.(0|2)\.ffn_.*_exps=CPU".into());
    let args = build_command_args(&m, None, 9001);
    assert_eq!(
        find_flag(&args, "--override-tensor"),
        Some(r"blk\.(0|2)\.ffn_.*_exps=CPU")
    );
    // Must NOT also emit --n-cpu-moe (would double-offload).
    assert!(!args.join(" ").contains("--n-cpu-moe"));
}

#[test]
fn gguf_argv_ignores_legacy_n_cpu_moe() {
    let mut m = gguf_model();
    m.n_cpu_moe = Some(13);
    assert_eq!(
        find_flag(&build_command_args(&m, None, 9001), "--n-cpu-moe"),
        None
    );
    m.n_cpu_moe = Some(0);
    assert!(!build_command_args(&m, None, 9001)
        .join(" ")
        .contains("--n-cpu-moe"));
    m.n_cpu_moe = None;
    assert!(!build_command_args(&m, None, 9001)
        .join(" ")
        .contains("--n-cpu-moe"));
}

#[test]
fn gguf_argv_emits_mmproj() {
    let mut m = gguf_model();
    m.mmproj_path = Some(PathBuf::from("/models/mmproj.gguf"));
    let args = build_command_args(&m, None, 9001);
    let j = args.join(" ");
    assert!(j.contains("--mmproj /models/mmproj.gguf"), "{j}");
}

#[test]
fn extra_args_can_override_fitted_args_when_user_asks_for_it() {
    let mut m = gguf_model();
    m.n_gpu_layers = Some(50);
    m.extra_args = vec!["-ngl".into(), "99".into()];
    let args = build_command_args(&m, None, 9001);
    let idx_first = args.iter().position(|a| a == "-ngl").unwrap();
    let idx_last = args.iter().rposition(|a| a == "-ngl").unwrap();
    assert_ne!(idx_first, idx_last);
    assert_eq!(args[idx_first + 1], "50");
    assert_eq!(args[idx_last + 1], "99");
}

#[test]
fn gpu_classification_uses_final_extra_args() {
    let mut m = gguf_model();
    m.n_gpu_layers = Some(50);
    m.extra_args = vec!["-ngl".into(), "0".into()];
    assert!(!instance_uses_gpu(&m, &build_command_args(&m, None, 9001)));

    let mut m = gguf_model();
    m.extra_args = vec!["-ngl".into(), "99".into()];
    assert!(instance_uses_gpu(&m, &build_command_args(&m, None, 9001)));

    let mut m = gguf_model();
    m.extra_args = vec!["--device=Vulkan0".into()];
    assert!(instance_uses_gpu(&m, &build_command_args(&m, None, 9001)));

    let mut m = gguf_model();
    m.weights_format = WeightsFormat::Safetensors;
    assert!(instance_uses_gpu(&m, &build_command_args(&m, None, 9001)));
}

#[test]
fn extra_args_can_override_router_fit_policy_when_user_asks_for_it() {
    let mut m = gguf_model();
    m.n_gpu_layers = Some(50);
    m.extra_args = vec!["--fit".into(), "on".into()];
    let args = build_command_args(&m, None, 9001);
    let idx_first = args.iter().position(|a| a == "--fit").unwrap();
    let idx_last = args.iter().rposition(|a| a == "--fit").unwrap();
    assert_ne!(idx_first, idx_last);
    assert_eq!(args[idx_first + 1], "off");
    assert_eq!(args[idx_last + 1], "on");
    assert_eq!(&args[args.len() - 2..], &["--fit", "on"]);
}

#[test]
fn extra_args_appended_last_for_safetensors() {
    let mut m = gguf_model();
    m.weights_format = WeightsFormat::Safetensors;
    m.extra_args = vec!["--tensor-parallel-size".into(), "2".into()];
    let args = build_command_args(&m, None, 9001);
    assert_eq!(&args[args.len() - 2..], &["--tensor-parallel-size", "2"]);
}

fn draft_model() -> ModelConfig {
    ModelConfig {
        id: "draft".into(),
        name: "D".into(),
        model_path: PathBuf::from("/models/draft.gguf"),
        context: 16384,
        n_gpu_layers: Some(99),
        device: Some("Vulkan1".into()),
        cache_type_k: Some(CacheType::Q8_0),
        cache_type_v: Some(CacheType::Q8_0),
        ..ModelConfig::default()
    }
}

fn find_flag<'a>(args: &'a [String], flag: &str) -> Option<&'a str> {
    args.windows(2)
        .find(|w| w[0] == flag)
        .map(|w| w[1].as_str())
}

#[test]
fn spec_decode_argv_emits_full_draft_and_policy_flags() {
    let mut t = gguf_model();
    t.draft_model_id = Some("draft".into());
    t.draft_max = Some(16);
    t.draft_min = Some(1);
    t.draft_p_min = Some(0.75);
    t.ctx_checkpoints = Some(4);
    t.checkpoint_min_step = Some(0);
    let d = draft_model();
    let args = build_command_args(&t, Some(&d), 9001);
    assert_eq!(find_flag(&args, "-md"), Some("/models/draft.gguf"));
    assert_eq!(find_flag(&args, "-ngld"), Some("99"));
    assert_eq!(find_flag(&args, "-devd"), Some("Vulkan1"));
    assert_eq!(find_flag(&args, "-ctkd"), Some("q8_0"));
    assert_eq!(find_flag(&args, "-ctvd"), Some("q8_0"));
    assert_eq!(find_flag(&args, "--spec-draft-n-max"), Some("16"));
    assert_eq!(find_flag(&args, "--spec-draft-n-min"), Some("1"));
    assert_eq!(find_flag(&args, "--spec-draft-p-min"), Some("0.75"));
    assert_eq!(find_flag(&args, "--ctx-checkpoints"), Some("4"));
    assert_eq!(find_flag(&args, "--checkpoint-min-step"), Some("0"));
    assert_eq!(find_flag(&args, "--checkpoint-every-n-tokens"), None);
}

#[test]
fn spec_decode_argv_omitted_when_no_draft_resolved() {
    let mut t = gguf_model();
    t.draft_model_id = Some("missing".into());
    t.draft_max = Some(16);
    t.ctx_checkpoints = Some(4);
    let args = build_command_args(&t, None, 9001);
    let joined = args.join(" ");
    assert!(!joined.contains("-md"), "{joined}");
    assert!(!joined.contains("--spec-draft-n-max"), "{joined}");
    assert!(!joined.contains("--ctx-checkpoints"), "{joined}");
}

#[test]
fn spec_decode_argv_omits_unset_policy_flags() {
    let mut t = gguf_model();
    t.draft_model_id = Some("draft".into());
    t.draft_max = Some(8);
    let args = build_command_args(&t, Some(&draft_model()), 9001);
    assert_eq!(find_flag(&args, "--spec-draft-n-max"), Some("8"));
    let joined = args.join(" ");
    assert!(!joined.contains("--spec-draft-n-min"), "{joined}");
    assert!(!joined.contains("--spec-draft-p-min"), "{joined}");
    assert!(!joined.contains("--ctx-checkpoints"), "{joined}");
    assert!(!joined.contains("--checkpoint-every-n-tokens"), "{joined}");
}

#[test]
fn gguf_argv_emits_fit_target_without_disabling_server_fit() {
    let mut m = gguf_model();
    m.device = Some("Vulkan0,Vulkan1".into());
    m.fit_target = Some("1024,2048".into());
    let args = build_command_args(&m, None, 9001);
    assert_eq!(find_flag(&args, "--device"), Some("Vulkan0,Vulkan1"));
    assert_eq!(find_flag(&args, "--fit-target"), Some("1024,2048"));
    assert_eq!(find_flag(&args, "--fit"), None);
}

#[test]
fn legacy_checkpoint_every_n_tokens_is_not_emitted_as_structured_arg() {
    let mut t = gguf_model();
    t.draft_model_id = Some("draft".into());
    t.checkpoint_every_n_tokens = Some(-1);
    let args = build_command_args(&t, Some(&draft_model()), 9001);
    assert_eq!(find_flag(&args, "--checkpoint-every-n-tokens"), None);
}

#[test]
fn spec_decode_argv_does_not_emit_removed_draft_context_flag() {
    let mut t = gguf_model();
    t.draft_model_id = Some("draft".into());
    let args = build_command_args(&t, Some(&draft_model()), 9001);
    let joined = args.join(" ");
    assert!(!joined.contains("-cd"), "{joined}");
}

#[test]
fn spec_decode_argv_skips_draft_device_flag_when_unset() {
    let mut d = draft_model();
    d.device = None;
    let mut t = gguf_model();
    t.draft_model_id = Some("draft".into());
    let args = build_command_args(&t, Some(&d), 9001);
    let joined = args.join(" ");
    assert!(!joined.contains("-devd"), "{joined}");
}

#[test]
fn mtp_argv_emits_spec_type_and_token_count() {
    let mut t = gguf_model();
    t.mtp_tokens = Some(4);
    let args = build_command_args(&t, None, 9001);
    assert_eq!(find_flag(&args, "--spec-type"), Some("draft-mtp"));
    assert_eq!(find_flag(&args, "--spec-draft-n-max"), Some("4"));
}

#[test]
fn mtp_argv_omitted_when_zero() {
    let mut t = gguf_model();
    t.mtp_tokens = Some(0);
    let args = build_command_args(&t, None, 9001);
    let joined = args.join(" ");
    assert!(!joined.contains("--spec-type"), "{joined}");
    assert!(!joined.contains("--spec-draft-n-max"), "{joined}");
}

#[test]
fn mtp_argv_does_not_mix_with_external_draft() {
    let mut t = gguf_model();
    t.draft_model_id = Some("draft".into());
    t.draft_max = Some(8);
    t.mtp_tokens = Some(4);
    let args = build_command_args(&t, Some(&draft_model()), 9001);
    assert_eq!(find_flag(&args, "--spec-draft-n-max"), Some("8"));
    let joined = args.join(" ");
    assert!(!joined.contains("--spec-type draft-mtp"), "{joined}");
}

#[test]
fn pending_instances_count_toward_total() {
    let mut pm = ProcessManager::default();
    pm.register_existing_instance("m", 123, 9001);
    pm.reserve_test_pending("m", 9002);

    assert_eq!(pm.instance_count("m"), 1);
    assert_eq!(pm.total_instance_count("m"), 2);
}

#[test]
fn acquire_any_instance_picks_idle_then_least_busy() {
    let mut pm = ProcessManager::default();
    pm.register_existing_instance("m", 123, 9001);
    pm.register_existing_instance("m", 124, 9002);

    let g1 = pm.acquire_any_instance("m").unwrap();
    assert_eq!(g1.port, 9001);

    let g2 = pm.acquire_any_instance("m").unwrap();
    assert_eq!(g2.port, 9002);

    let g3 = pm.acquire_any_instance("m").unwrap();
    assert_eq!(g3.port, 9001);

    let g4 = pm.acquire_any_instance("m").unwrap();
    assert_eq!(g4.port, 9002);

    drop((g1, g2, g3, g4));
}

#[test]
fn idle_instances_excludes_active_requests() {
    let mut pm = ProcessManager::default();
    pm.register_existing_instance("m", 123, 9001);
    pm.register_existing_instance("m", 124, 9002);

    let active = pm.acquire_idle_instance("m").unwrap();
    assert_eq!(active.pid, 123);
    assert_eq!(pm.idle_instances(), vec![("m".into(), 124, 9002)]);

    drop(active);
    assert_eq!(pm.idle_instances().len(), 2);
}

#[test]
fn parses_backend_port_range() {
    assert_eq!(parse_port_range("9100-9200"), Some((9100, 9200)));
    assert_eq!(parse_port_range("9200-9100"), None);
    assert_eq!(parse_port_range("nope"), None);
}
