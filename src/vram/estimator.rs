use crate::config::CacheType;
use ggus::{
    GGufMetaDataValueType as Ty, GGufMetaError, GGufMetaMap, GGufMetaMapExt,
    GGufMetaValueArray, GGufReader,
};
use memmap2::Mmap;
use std::collections::HashMap;
use std::fs::File;
use std::path::{Path, PathBuf};

/// Result of a VRAM estimation.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct VramEstimate {
    /// VRAM occupied by model weights (file size).
    pub weight_vram: u64,
    /// VRAM for the KV cache at the chosen context.
    pub kv_cache_vram: u64,
    /// Total VRAM with 10% runtime overhead.
    pub total_vram: u64,
}

/// Bytes-per-element for the K and V caches, expressed as a rational
/// `(numerator, denominator)` pair so q4_0 (half a byte) avoids
/// floating-point drift.
///
/// Constructed via `From<CacheType>`; defaults to fp16 (2 bytes).
#[derive(Debug, Clone, Copy)]
pub struct KvPerElement {
    pub k: (u64, u64),
    pub v: (u64, u64),
}

impl KvPerElement {
    /// Both caches held at fp16 — llama.cpp's default when
    /// `--cache-type-{k,v}` are unset. Exposed mainly for tests and as
    /// a documented worst-case baseline.
    #[allow(dead_code)] // used by tests and available for callers
    pub const FP16_BOTH: Self = Self { k: (2, 1), v: (2, 1) };

    pub fn from_types(k: CacheType, v: CacheType) -> Self {
        Self { k: k.bytes_per_element(), v: v.bytes_per_element() }
    }
}

impl VramEstimate {
    /// Pure formula, independent of GGUF parsing. Unit-tested directly.
    ///
    /// ```text
    /// per_token_bytes = n_embd_head * kv_heads_total * (k_bytes + v_bytes)
    /// kv_cache        = context * per_token_bytes
    /// total           = (file_size + kv_cache) * 1.1                  // 10% overhead
    /// ```
    ///
    /// `kv_heads_total` is the sum of `n_head_kv` across every layer —
    /// for a uniform model this equals `n_layers * n_head_kv`; for
    /// hybrid models (linear-attention layers mixed with full-attention
    /// layers, as in kimi-linear) it correctly excludes layers with
    /// zero KV heads, which `max * n_layers` would over-count.
    pub fn compute(
        file_size: u64,
        context: u32,
        n_embd_head: u64,
        kv_heads_total: u64,
        kv_bytes: KvPerElement,
    ) -> Self {
        let elements = context as u64 * n_embd_head.max(1) * kv_heads_total;
        // K + V stored separately; each may have its own quantization.
        // `(num/den)` is multiplied into the element count to keep q4_0
        // (half-a-byte) exact instead of rounding early.
        let k_bytes = elements * kv_bytes.k.0 / kv_bytes.k.1;
        let v_bytes = elements * kv_bytes.v.0 / kv_bytes.v.1;
        let kv_cache = k_bytes + v_bytes;
        let total = (file_size + kv_cache).saturating_mul(11) / 10;
        Self { weight_vram: file_size, kv_cache_vram: kv_cache, total_vram: total }
    }

}

/// All the GGUF metadata fields the orchestrator + form UI care about.
///
/// The fields split cleanly into two groups:
/// - **Display** (`n_head`, `n_head_kv`): one representative integer
///   per field; arrays are reduced to their max element. These exist
///   so the UI can show "this model has 96 heads, 8 KV heads" without
///   cluttering the form.
/// - **Math** (`kv_heads_total`, `n_embd`, `key_length`): the
///   primitives used by `VramEstimate::compute`. For uniform models
///   these equal the obvious products; for hybrid and per-layer MoE
///   architectures they carry the summed-over-layers shape directly
///   so the KV-cache estimate stays correct.
#[derive(Debug, Clone, Copy, serde::Serialize)]
pub struct GgufInfo {
    pub max_context: u32,
    pub n_layers: u32,
    pub n_embd: u32,
    pub n_head: u32,
    pub n_head_kv: u32,
    /// `<arch>.attention.key_length` when the GGUF exposes it, else 0.
    /// When present, KV-cache math uses this directly instead of
    /// `n_embd / n_head` — which is wrong for models like Step-3.5
    /// where the head dim is 128 but `n_embd / n_head` = 42.
    pub key_length: u32,
    /// `sum over layers of n_head_kv[layer]`. Equals `n_layers *
    /// n_head_kv` for uniform models; for hybrid models (e.g.
    /// kimi-linear) it excludes layers with 0 KV heads so the estimate
    /// doesn't double-count linear-attention layers.
    pub kv_heads_total: u64,
    /// Total bytes across all on-disk shards of this model. Equals the
    /// file size for a single-file GGUF, but sums all sibling shards for
    /// multi-file GGUFs named like `foo.gguf-00001-of-00005.gguf`.
    pub file_size: u64,
}

impl GgufInfo {
    pub fn read(path: &Path) -> Result<Self, EstimateError> {
        let file = File::open(path)?;
        // Multi-file GGUFs point at shard 1/N; the rest of the weights
        // live in sibling files. Sum them all so VRAM estimates reflect
        // the full model, not just the first shard.
        let file_size = sharded_total_size(path).unwrap_or_else(|| {
            file.metadata().map(|m| m.len()).unwrap_or(0)
        });
        // Safety: we only read from the mmap for the duration of this call;
        // the file is not mutated concurrently in our workflow.
        let mmap = unsafe { Mmap::map(&file) }?;

        let kvs = read_metadata_kvs(&mmap)?;
        let map = MetaMap(kvs);

        let max_context = map.llm_context_length().map_err(meta_err)? as u32;
        let n_layers = map.llm_block_count().map_err(meta_err)? as u32;
        let n_embd = map.llm_embedding_length().map_err(meta_err)? as u32;

        // Head counts may be scalars OR per-layer integer arrays of any
        // int element type (Step-3.5 uses i32, not u32). We parse both
        // shapes uniformly and carry forward the correct summed total,
        // not just a single representative.
        let head_stats = read_int_field(&map, &attention_key(&map, "head_count")?)?;
        let kv_stats = read_int_field(&map, &attention_key(&map, "head_count_kv")?)?;

        // Optional — only some models expose it. When present we use it
        // verbatim as the per-head dim; when absent we fall back to
        // n_embd / n_head (see `n_embd_head`).
        let key_length = match map.llm_attention_key_length() {
            Ok(v) => v as u32,
            Err(GGufMetaError::NotExist) => 0,
            Err(GGufMetaError::TypeMismatch(Ty::Array)) => 0,
            Err(e) => return Err(meta_err(e)),
        };

        Ok(Self {
            max_context,
            n_layers,
            n_embd,
            n_head: head_stats.max,
            n_head_kv: kv_stats.max,
            kv_heads_total: kv_stats.total_over_layers(n_layers),
            key_length,
            file_size,
        })
    }

    /// Per-head dimension used by KV-cache math. Prefers the explicit
    /// `attention.key_length` metadata; falls back to `n_embd / n_head`;
    /// then to 128 (the default for most modern transformer backbones).
    pub fn n_embd_head(&self) -> u64 {
        if self.key_length > 0 {
            self.key_length as u64
        } else if self.n_head > 0 {
            self.n_embd as u64 / self.n_head as u64
        } else {
            128
        }
    }
}

/// Scalar-or-array stats: whichever shape the GGUF field came in as,
/// we record the representative (`max`) value and the real sum when
/// the field was a per-layer array. For pure scalars `sum` is `None`
/// and callers synthesize it as `max * n_layers`.
#[derive(Debug, Clone, Copy)]
struct IntFieldStats {
    max: u32,
    sum: Option<u64>,
}

impl IntFieldStats {
    /// The KV-math-correct total. If the field was a per-layer array,
    /// use the sum directly; if a scalar, expand it across all layers.
    fn total_over_layers(&self, n_layers: u32) -> u64 {
        self.sum.unwrap_or_else(|| self.max as u64 * n_layers as u64)
    }
}

/// Resolve `<general.architecture>.attention.<suffix>` into a full key.
fn attention_key(map: &MetaMap, suffix: &str) -> Result<String, EstimateError> {
    let arch = map.general_architecture().map_err(meta_err)?;
    Ok(format!("{arch}.attention.{suffix}"))
}

/// Read an integer-shaped metadata field that may be a scalar (any int
/// type) or an array of any int type. Missing key → zeros.
fn read_int_field(map: &MetaMap, key: &str) -> Result<IntFieldStats, EstimateError> {
    let (ty, bytes) = match map.get(key) {
        Some(v) => v,
        None => return Ok(IntFieldStats { max: 0, sum: None }),
    };
    if ty == Ty::Array {
        let (values, _len) = read_int_array(bytes)?;
        let max = values.iter().copied().max().unwrap_or(0);
        let sum: u64 = values.iter().copied().sum();
        // Clamp to u32 for the display field. Sums larger than u32::MAX
        // would mean billions of KV heads — impossible in practice.
        let max_u32 = max.min(u32::MAX as u64) as u32;
        Ok(IntFieldStats { max: max_u32, sum: Some(sum) })
    } else {
        // Scalar: let ggus's get_usize handle sign extension / width.
        let v = map.get_usize(key).map_err(meta_err)?;
        Ok(IntFieldStats { max: v as u32, sum: None })
    }
}

/// Parse a GGUF array of any integer element type into a `Vec<u64>`.
/// Non-integer element types yield `(vec![], len)` so the caller can
/// still inspect the length if it needs to.
fn read_int_array(bytes: &[u8]) -> Result<(Vec<u64>, u64), EstimateError> {
    let mut r = GGufReader::new(bytes);
    let (elem_ty, len_usize) = r.read_arr_header().map_err(read_err)?;
    let len = len_usize as u64;

    macro_rules! collect_as {
        ($t:ty, $lift:expr) => {{
            let arr = GGufMetaValueArray::<$t>::new(r, len_usize);
            let vals: Vec<u64> = arr
                .filter_map(Result::ok)
                .map(|v| $lift(v))
                .collect();
            Ok((vals, len))
        }};
    }

    // Negative values (e.g. a `-1` sentinel in i32 arrays) collapse to 0
    // — they're not valid head counts. Positive values cast losslessly.
    let neg_to_zero = |v: i64| -> u64 {
        if v < 0 { 0 } else { v as u64 }
    };

    match elem_ty {
        Ty::U8 => collect_as!(u8, |v| v as u64),
        Ty::I8 => collect_as!(i8, |v: i8| neg_to_zero(v as i64)),
        Ty::U16 => collect_as!(u16, |v| v as u64),
        Ty::I16 => collect_as!(i16, |v: i16| neg_to_zero(v as i64)),
        Ty::U32 => collect_as!(u32, |v| v as u64),
        Ty::I32 => collect_as!(i32, |v: i32| neg_to_zero(v as i64)),
        Ty::U64 => collect_as!(u64, |v| v),
        Ty::I64 => collect_as!(i64, neg_to_zero),
        _ => Ok((Vec::new(), len)),
    }
}

/// If `path` looks like a sharded-GGUF member (ends with
/// `-NNNNN-of-MMMMM.gguf`), sum the sizes of all siblings and return the
/// total. Returns `None` when the pattern doesn't match or any sibling
/// is missing — callers fall back to the single-file size.
fn sharded_total_size(path: &Path) -> Option<u64> {
    let fname = path.file_name()?.to_str()?;
    let rest = fname.strip_suffix(".gguf")?;
    let (stem_and_idx, total_str) = rest.rsplit_once("-of-")?;
    let total: u32 = total_str.parse().ok()?;
    if total == 0 {
        return None;
    }
    let (stem, idx_str) = stem_and_idx.rsplit_once('-')?;
    let width = idx_str.len();
    idx_str.parse::<u32>().ok()?;

    let parent: PathBuf = path.parent().unwrap_or_else(|| Path::new(".")).to_path_buf();
    let mut sum = 0u64;
    for i in 1..=total {
        let shard = parent.join(format!("{stem}-{i:0width$}-of-{total:0width$}.gguf"));
        let size = std::fs::metadata(&shard).ok()?.len();
        sum = sum.checked_add(size)?;
    }
    Some(sum)
}

/// Reads the GGUF header + metadata KV block into a (key → (ty, value_bytes))
/// map. Ignores tensor descriptors and data.
fn read_metadata_kvs(data: &[u8]) -> Result<HashMap<String, (Ty, Vec<u8>)>, EstimateError> {
    let mut reader = GGufReader::new(data);
    let header = reader.read_header().map_err(read_err)?;
    if !header.is_magic_correct() {
        return Err(EstimateError::Gguf("bad magic".into()));
    }
    if !header.is_native_endian() {
        return Err(EstimateError::Gguf("non-native endian not supported".into()));
    }

    let mut map = HashMap::with_capacity(header.metadata_kv_count as usize);
    for _ in 0..header.metadata_kv_count {
        let kv = reader.read_meta_kv().map_err(read_err)?;
        map.insert(kv.key().to_string(), (kv.ty(), kv.value_bytes().to_vec()));
    }
    Ok(map)
}

/// Adapter that lets us call the `GGufMetaMapExt` helpers on our parsed map.
struct MetaMap(HashMap<String, (Ty, Vec<u8>)>);

impl GGufMetaMap for MetaMap {
    fn get(&self, key: &str) -> Option<(Ty, &[u8])> {
        self.0.get(key).map(|(ty, v)| (*ty, v.as_slice()))
    }
}

fn read_err(e: ggus::GGufReadError) -> EstimateError {
    EstimateError::Gguf(format!("{e:?}"))
}

fn meta_err(e: GGufMetaError) -> EstimateError {
    EstimateError::Gguf(format!("{e:?}"))
}

#[derive(Debug, thiserror::Error)]
pub enum EstimateError {
    #[error("io error: {0}")]
    Io(#[from] std::io::Error),

    #[error("gguf error: {0}")]
    Gguf(String),
}

#[cfg(test)]
mod tests {
    use super::*;

    // Canonical llama-7b-ish params used for arithmetic checks.
    const FILE_SIZE: u64 = 6_442_450_944; // ~6 GiB
    const N_LAYERS: u64 = 32;
    const N_EMBD_HEAD: u64 = 128; // 4096 / 32
    const N_HEAD_KV: u64 = 32;

    #[test]
    fn kv_cache_scales_linearly_with_context() {
        let kv_heads_total = N_LAYERS * N_HEAD_KV;
        let a = VramEstimate::compute(FILE_SIZE, 4096, N_EMBD_HEAD, kv_heads_total, KvPerElement::FP16_BOTH);
        let b = VramEstimate::compute(FILE_SIZE, 16384, N_EMBD_HEAD, kv_heads_total, KvPerElement::FP16_BOTH);
        let ratio = b.kv_cache_vram as f64 / a.kv_cache_vram as f64;
        assert!((3.9..=4.1).contains(&ratio), "ratio={ratio}");
    }

    #[test]
    fn gqa_shrinks_kv_cache_proportionally() {
        let mha = VramEstimate::compute(FILE_SIZE, 4096, N_EMBD_HEAD, N_LAYERS * 32, KvPerElement::FP16_BOTH);
        let gqa = VramEstimate::compute(FILE_SIZE, 4096, N_EMBD_HEAD, N_LAYERS * 4, KvPerElement::FP16_BOTH);
        let ratio = mha.kv_cache_vram as f64 / gqa.kv_cache_vram as f64;
        assert!((7.9..=8.1).contains(&ratio), "ratio={ratio}");
    }

    #[test]
    fn zero_context_yields_no_kv_cache() {
        let e = VramEstimate::compute(FILE_SIZE, 0, N_EMBD_HEAD, N_LAYERS * N_HEAD_KV, KvPerElement::FP16_BOTH);
        assert_eq!(e.kv_cache_vram, 0);
        assert_eq!(e.total_vram, (FILE_SIZE * 11) / 10);
    }

    #[test]
    fn overhead_is_ten_percent() {
        let e = VramEstimate::compute(FILE_SIZE, 4096, N_EMBD_HEAD, N_LAYERS * N_HEAD_KV, KvPerElement::FP16_BOTH);
        let base = FILE_SIZE + e.kv_cache_vram;
        assert_eq!(e.total_vram, base * 11 / 10);
    }

    #[test]
    fn q8_kv_quantization_halves_kv_cache_vs_fp16() {
        let fp16 = VramEstimate::compute(0, 4096, N_EMBD_HEAD, N_LAYERS * N_HEAD_KV, KvPerElement::FP16_BOTH);
        let q8 = VramEstimate::compute(
            0, 4096, N_EMBD_HEAD, N_LAYERS * N_HEAD_KV,
            KvPerElement::from_types(CacheType::Q8_0, CacheType::Q8_0),
        );
        assert_eq!(q8.kv_cache_vram * 2, fp16.kv_cache_vram);
    }

    #[test]
    fn q4_kv_quantization_quarters_kv_cache_vs_fp16() {
        let fp16 = VramEstimate::compute(0, 4096, N_EMBD_HEAD, N_LAYERS * N_HEAD_KV, KvPerElement::FP16_BOTH);
        let q4 = VramEstimate::compute(
            0, 4096, N_EMBD_HEAD, N_LAYERS * N_HEAD_KV,
            KvPerElement::from_types(CacheType::Q4_0, CacheType::Q4_0),
        );
        assert_eq!(q4.kv_cache_vram * 4, fp16.kv_cache_vram);
    }

    #[test]
    fn mixed_kv_quantization_sums_k_and_v_independently() {
        let mixed = VramEstimate::compute(
            0, 4096, N_EMBD_HEAD, N_LAYERS * N_HEAD_KV,
            KvPerElement::from_types(CacheType::Q4_0, CacheType::F16),
        );
        // K at 0.5 bytes/elem + V at 2.0 bytes/elem = 2.5 bytes/elem,
        // versus 4.0 for f16 both. Ratio 2.5/4 = 5/8.
        let fp16 = VramEstimate::compute(
            0, 4096, N_EMBD_HEAD, N_LAYERS * N_HEAD_KV,
            KvPerElement::FP16_BOTH,
        );
        assert_eq!(mixed.kv_cache_vram * 8, fp16.kv_cache_vram * 5);
    }

    fn info_with(n_embd: u32, n_head: u32, n_head_kv: u32, key_length: u32) -> GgufInfo {
        GgufInfo {
            max_context: 0, n_layers: 0, n_embd, n_head, n_head_kv,
            key_length, kv_heads_total: 0, file_size: 0,
        }
    }

    #[test]
    fn n_embd_head_prefers_key_length_when_present() {
        // 128 wins even though n_embd / n_head = 42.
        assert_eq!(info_with(4096, 96, 8, 128).n_embd_head(), 128);
    }

    #[test]
    fn n_embd_head_falls_back_to_n_embd_over_n_head() {
        assert_eq!(info_with(4096, 32, 8, 0).n_embd_head(), 128);
    }

    #[test]
    fn n_embd_head_defaults_to_128_when_head_count_unknown() {
        assert_eq!(info_with(4096, 0, 0, 0).n_embd_head(), 128);
    }

    #[test]
    fn int_field_stats_scalar_expands_across_layers() {
        let stats = IntFieldStats { max: 8, sum: None };
        assert_eq!(stats.total_over_layers(45), 8 * 45);
    }

    #[test]
    fn int_field_stats_array_uses_summed_total() {
        // Hybrid model: 45 layers with n_head_kv = [8, 0, 8, 0, ...].
        // Total = sum directly, NOT max * n_layers.
        let stats = IntFieldStats { max: 8, sum: Some(8 * 23) };
        // Even if the caller wrongly passes a layer count, sum wins.
        assert_eq!(stats.total_over_layers(45), 8 * 23);
    }

    // ---- Array reader: all supported integer element types ----

    /// Build a GGUF Array value body: <elem_ty: u32><len: u64><values…>.
    fn encode_array(elem_ty: Ty, values_le: &[u8], len: u64) -> Vec<u8> {
        let mut out = Vec::new();
        out.extend_from_slice(&(elem_ty as u32).to_le_bytes());
        out.extend_from_slice(&len.to_le_bytes());
        out.extend_from_slice(values_le);
        out
    }

    #[test]
    fn read_int_array_parses_i32_shape_like_step35() {
        // Step-3.5 writes attention.head_count_kv as an i32 array of 45
        // entries, all equal to 8. We must decode that correctly — u32
        // parsing alone would have returned an empty array.
        let mut body = Vec::new();
        for _ in 0..45u32 {
            body.extend_from_slice(&8i32.to_le_bytes());
        }
        let arr = encode_array(Ty::I32, &body, 45);
        let (values, len) = read_int_array(&arr).unwrap();
        assert_eq!(len, 45);
        assert_eq!(values.len(), 45);
        assert!(values.iter().all(|&v| v == 8));
    }

    #[test]
    fn read_int_array_parses_u32() {
        let mut body = Vec::new();
        for v in [1u32, 2, 3] {
            body.extend_from_slice(&v.to_le_bytes());
        }
        let (values, len) = read_int_array(&encode_array(Ty::U32, &body, 3)).unwrap();
        assert_eq!(len, 3);
        assert_eq!(values, vec![1, 2, 3]);
    }

    #[test]
    fn read_int_array_clamps_negative_i32_values_to_zero() {
        // -1 sentinels appear in some GGUFs to mean "no attention on
        // this layer". Treat them as 0 KV heads so the estimate
        // correctly ignores them instead of wrapping to ~4 billion.
        let mut body = Vec::new();
        body.extend_from_slice(&(-1i32).to_le_bytes());
        body.extend_from_slice(&8i32.to_le_bytes());
        let (values, _) = read_int_array(&encode_array(Ty::I32, &body, 2)).unwrap();
        assert_eq!(values, vec![0, 8]);
    }

    #[test]
    fn read_int_array_rejects_non_integer_elements() {
        // A float array is legitimate for other keys but not a head
        // count. Yield empty so the caller falls back cleanly instead
        // of returning garbage.
        let (values, len) = read_int_array(&encode_array(Ty::F32, &[], 0)).unwrap();
        assert!(values.is_empty());
        assert_eq!(len, 0);
    }

    // ---- Sharded-total-size ----

    #[test]
    fn sharded_total_size_sums_all_shards() {
        let tmp = tempfile::tempdir().unwrap();
        let base = tmp.path().join("model.gguf");
        // Three shards of 10/20/30 bytes.
        for (i, size) in [(1u32, 10), (2, 20), (3, 30)] {
            let name = format!("model.gguf-{i:05}-of-00003.gguf");
            std::fs::write(tmp.path().join(&name), vec![0u8; size]).unwrap();
        }
        let shard1 = tmp.path().join("model.gguf-00001-of-00003.gguf");
        assert_eq!(sharded_total_size(&shard1), Some(60));
        // Non-sharded filename returns None (caller falls back to its
        // single-file size).
        assert_eq!(sharded_total_size(&base), None);
    }

    #[test]
    fn sharded_total_size_returns_none_if_a_shard_is_missing() {
        let tmp = tempfile::tempdir().unwrap();
        // Only write shard 1/3 — caller should fall back.
        let name = "model.gguf-00001-of-00003.gguf";
        std::fs::write(tmp.path().join(name), vec![0u8; 10]).unwrap();
        let shard1 = tmp.path().join(name);
        assert_eq!(sharded_total_size(&shard1), None);
    }

    #[test]
    fn gguf_info_read_fails_on_missing_file() {
        let err = GgufInfo::read(Path::new("/nonexistent.gguf")).unwrap_err();
        assert!(matches!(err, EstimateError::Io(_)));
    }

    #[test]
    fn gguf_info_read_rejects_non_gguf_file() {
        let tmp = tempfile::NamedTempFile::new().unwrap();
        std::fs::write(tmp.path(), b"not a gguf file at all, just some bytes").unwrap();
        let err = GgufInfo::read(tmp.path()).unwrap_err();
        assert!(matches!(err, EstimateError::Gguf(_)));
    }

}
