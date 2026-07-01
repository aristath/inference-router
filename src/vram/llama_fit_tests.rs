use super::*;
use crate::config::CacheType;

#[test]
fn parses_fitted_args_with_quoted_override_tensor() {
    let fitted = parse_fitted_args(
            r#"-c 4096 -ngl 48 -ts 1,2 -ot "blk\.14\.ffn_(up|down)_(ch|)exps=CPU,blk\.15\.ffn_(up|down)_(ch|)exps=CPU""#,
        )
        .unwrap();
    assert_eq!(fitted.context, Some(4096));
    assert_eq!(fitted.n_gpu_layers, Some(48));
    assert_eq!(fitted.tensor_split.as_deref(), Some("1,2"));
    assert!(fitted.override_tensor.unwrap().contains(r"blk\.14\.ffn_"));
}

#[test]
fn base_args_include_only_fit_tool_supported_memory_options() {
    let model = ModelConfig {
        model_path: "/models/target.gguf".into(),
        context: 8192,
        mmproj_path: Some("/models/mmproj.gguf".into()),
        draft_model_id: Some("draft".into()),
        draft_max: Some(16),
        draft_min: Some(4),
        draft_p_min: Some(0.7),
        ctx_checkpoints: Some(3),
        checkpoint_every_n_tokens: Some(-1),
        ..ModelConfig::default()
    };
    let draft = ModelConfig {
        model_path: "/models/draft.gguf".into(),
        n_gpu_layers: Some(12),
        device: Some("Vulkan2".into()),
        cache_type_k: Some(CacheType::Q8_0),
        cache_type_v: Some(CacheType::Q8_0),
        ..ModelConfig::default()
    };
    let args = base_args(&model, "Vulkan0,Vulkan1").join(" ");
    assert!(args.contains("-m /models/target.gguf"), "{args}");
    assert!(args.contains("-c 8192"), "{args}");
    assert!(args.contains("--device Vulkan0,Vulkan1"), "{args}");
    assert!(!args.contains("--mmproj"), "{args}");
    assert!(!args.contains("-md"), "{args}");
    assert!(!args.contains("-ngld"), "{args}");
    assert!(!args.contains("-devd"), "{args}");
    assert!(!args.contains("-ctkd"), "{args}");
    assert!(!args.contains("-ctvd"), "{args}");
    assert!(!args.contains("--spec-draft-n-max"), "{args}");
    assert!(!args.contains("--ctx-checkpoints"), "{args}");
    assert!(!args.contains("--checkpoint-every-n-tokens"), "{args}");

    assert!(needs_server_owned_fit(&model, Some(&draft)));
}

#[test]
fn parses_fit_print_device_rows_and_ignores_host() {
    let bytes =
        parse_fit_print_device_vram("Vulkan2 1267 67 493\nHost 397 0 12\nVulkan3 100 20 30\n")
            .unwrap();
    assert_eq!(bytes, (1267 + 67 + 493 + 100 + 20 + 30) * MIB);
}

#[test]
fn applies_negative_ngl_as_all_layers_convention() {
    let mut model = ModelConfig {
        cache_type_k: Some(CacheType::Q8_0),
        ..ModelConfig::default()
    };
    apply_sizing_to_model(
        &mut model,
        &LlamaFitSizing {
            fitted: LlamaFittedArgs {
                context: Some(8192),
                n_gpu_layers: Some(-1),
                tensor_split: None,
                override_tensor: None,
            },
            device_vram: 123,
        },
    );
    assert_eq!(model.context, 8192);
    assert_eq!(model.n_gpu_layers, Some(999));
    assert_eq!(model.estimated_vram, 123);
    assert_eq!(model.cache_type_k, Some(CacheType::Q8_0));
}
