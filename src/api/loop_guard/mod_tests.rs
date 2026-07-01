use super::*;

#[test]
fn choice_snapshot_truncates_loop_region() {
    let mut choice = ChoiceState::new(64 * 1024, 4);
    choice.append_tracked(FieldKind::Content, b"go");
    for _ in 0..12 {
        choice.append_tracked(FieldKind::Content, b"XYZABXYZAB");
    }
    let period = choice.detector.scan();
    assert_eq!(period, 5);
    let snap = choice.snapshot(4, period);
    assert_eq!(snap.content, "go");
}
