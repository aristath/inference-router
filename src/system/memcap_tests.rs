use super::*;

#[test]
fn cap_leaves_exactly_the_reserve() {
    // 64 GiB → 40 GiB cap; 128 GiB → 104 GiB cap.
    let cap = compute_cap(64 * GIB, 24 * GIB).unwrap();
    assert_eq!(cap.max, 40 * GIB);
    assert!(cap.high < cap.max && cap.high >= 38 * GIB);

    assert_eq!(compute_cap(128 * GIB, 24 * GIB).unwrap().max, 104 * GIB);
}

#[test]
fn no_cap_when_ram_below_reserve_plus_floor() {
    assert!(compute_cap(24 * GIB, 24 * GIB).is_none());
    assert!(compute_cap(25 * GIB, 24 * GIB).is_none()); // only 1 GiB left
}

#[test]
fn parses_mem_total_and_unit() {
    assert_eq!(
        parse_mem_total("MemFree: 100 kB\nMemTotal:   65390216 kB\n"),
        Some(65390216 * 1024)
    );
    assert_eq!(
        parse_systemd_unit(
            "0::/user.slice/user-1000.slice/user@1000.service/app.slice/inference-router.service\n"
        )
        .as_deref(),
        Some("inference-router.service")
    );
    assert_eq!(parse_systemd_unit("0::/user.slice/foo\n"), None);
}
