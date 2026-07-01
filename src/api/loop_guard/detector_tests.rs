use super::*;

#[test]
fn exact_k_fold_tandem() {
    let cases = [
        ("aaa", 3, 1),
        ("ababab", 3, 2),
        ("abcabcabc", 3, 3),
        ("aaaaaaaaaa", 10, 1),
        ("xyzxyzxyzxyzxyzxyzxyzxyzxyzxyz", 10, 3),
    ];
    for (input, repeats, want) in cases {
        let mut d = Detector::new(64 * 1024, repeats);
        d.append(input.as_bytes());
        assert_eq!(d.scan(), want, "{input}");
    }
}

#[test]
fn not_at_suffix() {
    let mut d = Detector::new(1024, 3);
    d.append(b"abcabcabc-tail-different");
    assert_eq!(d.scan(), 0);
}

#[test]
fn similar_prose_is_not_a_loop() {
    let prose = "I'm picturing the layout. I'm realizing the core issue is alignment. \
            I'm settling on a vertical flow. The key insight is that branches nest. \
            I'm picturing how loops fit. I'm realizing branches need their own block.";
    let mut d = Detector::new(64 * 1024, 3);
    d.append(prose.as_bytes());
    assert_eq!(d.scan(), 0);
}

#[test]
fn loop_extent() {
    let mut d = Detector::new(64 * 1024, 3);
    d.append("preamble: ".as_bytes());
    d.append("XYZAB".repeat(25).as_bytes());
    assert_eq!(d.loop_extent(5), 25);
}

#[test]
fn z_function_spot_check() {
    let got = z_function(b"aabaabxaab");
    assert_eq!(got, vec![10, 1, 0, 3, 1, 0, 0, 3, 1, 0]);
}
