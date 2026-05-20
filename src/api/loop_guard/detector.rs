//! Finds K-fold tandem byte repetition at the suffix of a rolling buffer.
//!
//! A "loop" here means: the last K*p bytes of the buffer are exactly K
//! byte-identical copies of some length-p block, for some p >= 1. Detection
//! is exact (no fuzzy matching). The detector is fed the assistant's emitted
//! text/tool-call bytes from a streaming completion and scanned periodically
//! (for example every few seconds), so the per-byte ingest cost is just a
//! slice append.
//!
//! Algorithm: Z-function of the reversed buffer in O(n) per scan. The
//! reversal is the key trick: a tandem repeat at the suffix of the original
//! buffer is a tandem repeat at the prefix of the reversed buffer, and the
//! Z-function directly answers prefix-match-length queries.
//!
//! For a candidate period p, the suffix of the original buffer is K-fold
//! tandem with period p iff `z[p] >= (K-1)*p` in the reversed buffer.

/// Accumulates bytes into a capped rolling buffer and exposes a scan that
/// returns the smallest tandem-repeat period at the suffix.
pub(super) struct Detector {
    buf: Vec<u8>,
    max_bytes: usize,
    repeats: usize,
}

impl Detector {
    /// Returns a detector that keeps at most `max_bytes` of history and reports
    /// a loop when it sees `repeats` consecutive byte-identical copies of any
    /// block at the suffix.
    pub(super) fn new(max_bytes: usize, repeats: usize) -> Self {
        let max_bytes = max_bytes.max(1);
        let repeats = repeats.max(2);
        Self {
            buf: Vec::with_capacity(max_bytes),
            max_bytes,
            repeats,
        }
    }

    /// Adds bytes to the rolling buffer, dropping the oldest bytes if the cap
    /// is exceeded.
    pub(super) fn append(&mut self, p: &[u8]) {
        if p.is_empty() {
            return;
        }
        if self.buf.len() + p.len() <= self.max_bytes {
            self.buf.extend_from_slice(p);
            return;
        }
        if p.len() >= self.max_bytes {
            self.buf.clear();
            self.buf
                .extend_from_slice(&p[p.len().saturating_sub(self.max_bytes)..]);
            return;
        }
        let excess = self.buf.len() + p.len() - self.max_bytes;
        self.buf.copy_within(excess.., 0);
        self.buf.truncate(self.buf.len() - excess);
        self.buf.extend_from_slice(p);
    }

    /// Empties the buffer.
    #[allow(dead_code)]
    pub(super) fn reset(&mut self) {
        self.buf.clear();
    }

    /// Returns the current buffer length in bytes.
    #[allow(dead_code)]
    pub(super) fn len(&self) -> usize {
        self.buf.len()
    }

    /// Returns the smallest period p >= 1 such that the last K*p bytes of the
    /// buffer are K consecutive byte-identical p-length blocks.
    pub(super) fn scan(&self) -> usize {
        let n = self.buf.len();
        let k = self.repeats;
        if n < k {
            return 0;
        }

        let mut rev = Vec::with_capacity(n);
        rev.extend(self.buf.iter().rev().copied());

        let z = z_function(&rev);
        let max_p = n / k;
        let needed = k - 1;
        for (p, &match_len) in z.iter().enumerate().take(max_p + 1).skip(1) {
            if match_len >= needed * p {
                return p;
            }
        }
        0
    }

    /// Extends a known-period tandem repeat backward to its full run length at
    /// the buffer's suffix.
    pub(super) fn loop_extent(&self, period: usize) -> usize {
        if period == 0 {
            return 0;
        }
        let n = self.buf.len();
        if n < period {
            return 0;
        }
        let unit = &self.buf[n - period..];
        let mut k = 0;
        while (k + 1) * period <= n {
            let start = n - (k + 1) * period;
            if self.buf[start..start + period] != *unit {
                break;
            }
            k += 1;
        }
        k
    }

    /// Returns up to `max_bytes` from the start of the most recently repeated
    /// unit, for logging.
    pub(super) fn snippet(&self, period: usize, max_bytes: usize) -> Vec<u8> {
        let n = self.buf.len();
        if period == 0 || period > n {
            return Vec::new();
        }
        let start = n - period;
        let end = if max_bytes > 0 && period > max_bytes {
            start + max_bytes
        } else {
            n
        };
        self.buf[start..end].to_vec()
    }
}

fn z_function(s: &[u8]) -> Vec<usize> {
    let n = s.len();
    let mut z = vec![0; n];
    if n == 0 {
        return z;
    }
    z[0] = n;
    let (mut l, mut r) = (0, 0);
    for i in 1..n {
        if i < r {
            z[i] = z[i - l].min(r - i);
        }
        while i + z[i] < n && s[z[i]] == s[i + z[i]] {
            z[i] += 1;
        }
        if i + z[i] > r {
            l = i;
            r = i + z[i];
        }
    }
    z
}

#[cfg(test)]
mod tests {
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
}
