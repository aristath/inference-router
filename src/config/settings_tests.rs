use super::*;

#[test]
fn defaults_match_loop_guard_runtime_defaults() {
    let settings = AppSettings::default();
    assert_eq!(settings.models_folder, "~/models");
    assert!(settings.loop_guards.streaming.enabled);
    assert_eq!(settings.loop_guards.streaming.window_bytes, 65_536);
    assert_eq!(settings.loop_guards.streaming.repeats, 10);
    assert_eq!(settings.loop_guards.streaming.check_interval_ms, 5_000);
    assert_eq!(settings.loop_guards.streaming.max_retries, 3);
    assert_eq!(
        settings.loop_guards.streaming.action,
        StreamingLoopAction::Heal
    );
    assert!(settings.loop_guards.streaming.replay_partial);
    assert!(settings.loop_guards.tool.enabled);
    assert_eq!(settings.loop_guards.tool.repeats, 3);
    assert_eq!(settings.loop_guards.tool.window_messages, 80);
}

#[test]
fn sanitize_clamps_runtime_minimums() {
    let mut settings = AppSettings::default();
    settings.loop_guards.streaming.window_bytes = 1;
    settings.loop_guards.streaming.repeats = 0;
    settings.loop_guards.streaming.check_interval_ms = 0;
    settings.loop_guards.tool.repeats = 0;
    settings.loop_guards.tool.window_messages = 0;

    let settings = settings.sanitized();
    assert_eq!(settings.loop_guards.streaming.window_bytes, 1024);
    assert_eq!(settings.loop_guards.streaming.repeats, 2);
    assert_eq!(settings.loop_guards.streaming.check_interval_ms, 1);
    assert_eq!(settings.loop_guards.tool.repeats, 2);
    assert_eq!(settings.loop_guards.tool.window_messages, 2);
}
