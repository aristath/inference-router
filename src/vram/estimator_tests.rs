use super::*;
use crate::config::CacheType;
use memmap2::Mmap;
use serde::{Deserialize, Serialize};
use std::fs::File;
use std::path::Path;

/// Result of a VRAM estimation.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct VramEstimate {
    /// VRAM occupied by the GPU-resident model weights (the offloaded layers).
    pub weight_vram: u64,
    /// VRAM for the KV cache of the GPU-resident layers.
    pub kv_cache_vram: u64,
    /// Total VRAM with 10% runtime overhead.
    pub total_vram: u64,
    /// Anonymous CPU-side RAM for the *non*-offloaded layers' KV cache. The
    /// non-offloaded *weights* stay mmap'd (file-backed, reclaimable) so they
    /// don't count here. Zero when the model is fully GPU-resident.
    pub cpu_kv_ram: u64,
}

/// Fraction (0.0..=1.0) of the model's repeating layers that live on the GPU,
/// given the configured `-ngl`. llama.cpp's `-ngl N` puts N layers on the GPU
/// and the rest on the CPU. `None` (no `-ngl` passed) or `N >= n_layers` (e.g.
/// the `99` "all" convention) means fully offloaded.
pub fn gpu_layer_fraction(n_layers: u32, n_gpu_layers: Option<u32>) -> f64 {
    if n_layers == 0 {
        return 1.0;
    }
    match n_gpu_layers {
        None => 1.0,
        Some(n) if n >= n_layers => 1.0,
        Some(n) => n as f64 / n_layers as f64,
    }
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
    pub const FP16_BOTH: Self = Self {
        k: (2, 1),
        v: (2, 1),
    };

    pub fn from_types(k: CacheType, v: CacheType) -> Self {
        Self {
            k: cache_type_bytes(k),
            v: cache_type_bytes(v),
        }
    }
}

fn cache_type_bytes(cache_type: CacheType) -> (u64, u64) {
    match cache_type {
        CacheType::F16 => (2, 1),
        CacheType::Q8_0 => (1, 1),
        CacheType::Q4_0 => (1, 2),
    }
}

impl VramEstimate {
    /// Combine weight size and pre-computed KV cache bytes into a VRAM total
    /// (with 10% runtime overhead), accounting for partial GPU offload.
    ///
    /// Only the offloaded fraction of weights + KV counts as VRAM; the
    /// remainder's KV becomes CPU RAM (`cpu_kv_ram`) while its weights stay
    /// mmap'd (not counted). With full offload (`n_gpu_layers` ≥ `n_layers` or
    /// `None`) this reduces to "everything is VRAM", matching the old behaviour.
    /// The KV-cache computation itself lives on [`GgufInfo::kv_cache_bytes`].
    pub fn compute(
        file_size: u64,
        kv_cache_bytes: u64,
        n_layers: u32,
        n_gpu_layers: Option<u32>,
    ) -> Self {
        let frac = gpu_layer_fraction(n_layers, n_gpu_layers);
        let weight_vram = scale_bytes(file_size, frac);
        let kv_vram = scale_bytes(kv_cache_bytes, frac);
        let total = (weight_vram + kv_vram).saturating_mul(11) / 10;
        Self {
            weight_vram,
            kv_cache_vram: kv_vram,
            total_vram: total,
            cpu_kv_ram: kv_cache_bytes.saturating_sub(kv_vram),
        }
    }
}

fn scale_bytes(bytes: u64, fraction: f64) -> u64 {
    (bytes as f64 * fraction.clamp(0.0, 1.0)).round() as u64
}

/// All the GGUF metadata fields the orchestrator + form UI care about.
///
/// The fields split cleanly into two groups:
/// - **Display** (`n_head`, `n_head_kv`, `kv_heads_total`): convenience
///   aggregates for the form UI and model list.
/// - **KV-cache math** (`full_kv_heads`, `swa_kv_heads`,
///   `sliding_window`, `key_length`, `key_length_swa`): the exact
///   primitives needed by `kv_cache_bytes`. For uniform models these
///   collapse to the scalar case; for hybrid architectures (Step-3.5
///   per-layer MoE, kimi-linear with linear-attention mixed in,
///   gemma with sliding-window attention on most layers) they carry
///   the per-layer shape so the estimate stays grounded in what
///   llama.cpp actually allocates.
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct GgufInfo {
    pub max_context: u32,
    pub n_layers: u32,
    pub n_embd: u32,
    pub n_head: u32,
    pub n_head_kv: u32,
    /// `<arch>.attention.key_length` when the GGUF exposes it, else 0.
    /// The per-head dim for **full-context** attention layers. Step-3.5
    /// publishes 128 here even though `n_embd / n_head = 42`; gemma-4
    /// publishes 512 for its global-attention layers.
    pub key_length: u32,
    /// Per-head dim for **sliding-window** attention layers, when the
    /// model has separate SWA layers. 0 when the model has no SWA.
    pub key_length_swa: u32,
    /// Sum of `n_head_kv` across **full-context** layers. Equals
    /// `kv_heads_total` for a pure full-attention model; for models
    /// with SWA, this counts only the subset that caches full context.
    pub full_kv_heads: u64,
    /// Sum of `n_head_kv` across **sliding-window** layers only.
    /// Their effective context is capped at `sliding_window` tokens.
    /// Zero for models without SWA.
    pub swa_kv_heads: u64,
    /// Sum of `n_head_kv` across layers that allocate KV — purely for
    /// display. `full_kv_heads + swa_kv_heads`. Kept as a separate field so
    /// the JS form can show it without recomputing.
    pub kv_heads_total: u64,
    /// Sliding-window size in tokens. 0 when the model doesn't use SWA.
    pub sliding_window: u32,
    /// For recurrent/hybrid models that only allocate full KV on every Nth
    /// layer, this is `<arch>.full_attention_interval`.
    #[serde(default)]
    pub full_attention_interval: Option<u32>,
    /// Total bytes across all on-disk shards of this model. Equals the
    /// file size for a single-file GGUF, but sums all sibling shards for
    /// multi-file GGUFs named like `foo.gguf-00001-of-00005.gguf`.
    pub file_size: u64,
    /// Total bytes of the Mixture-of-Experts weight tensors (`*_exps*`),
    /// summed across shards. `> 0` iff the model is MoE. These weights can be
    /// offloaded to CPU independently of the dense/attention weights via
    /// llama.cpp's `--n-cpu-moe`, which is the key to running huge MoE models:
    /// the experts are most of the bytes but only a few are active per token.
    #[serde(default)]
    pub expert_weight_bytes: u64,
}

impl GgufInfo {
    pub fn read(path: &Path) -> Result<Self, EstimateError> {
        let file = File::open(path)?;
        // Multi-file GGUFs point at shard 1/N; the rest of the weights
        // live in sibling files. Sum them all so VRAM estimates reflect
        // the full model, not just the first shard.
        let file_size = sharded_total_size(path)
            .unwrap_or_else(|| file.metadata().map(|m| m.len()).unwrap_or(0));
        // Safety: we only read from the mmap for the duration of this call;
        // the file is not mutated concurrently in our workflow.
        let mmap = unsafe { Mmap::map(&file) }?;

        let kvs = read_metadata_kvs(&mmap)?;
        let map = MetaMap(kvs);

        let max_context = map.llm_context_length().map_err(meta_err)? as u32;
        let n_layers = map.llm_block_count().map_err(meta_err)? as u32;
        let n_embd = map.llm_embedding_length().map_err(meta_err)? as u32;
        let arch = map.general_architecture().ok().map(|s| s.to_string());
        let arch_ref = arch.as_deref().unwrap_or("");

        // Head counts may be scalars OR per-layer integer arrays of any
        // int element type (Step-3.5 uses i32, not u32). We parse both
        // shapes uniformly and carry forward the correct summed total,
        // not just a single representative.
        let head_stats = read_int_field(&map, &attention_key(&map, "head_count")?)?;
        let kv_heads_per_layer =
            read_int_field_per_layer(&map, &attention_key(&map, "head_count_kv")?, n_layers)?;

        // Optional — only some models expose it. When present we use it
        // verbatim as the per-head dim; when absent we fall back to
        // n_embd / n_head (see `n_embd_head`).
        let key_length = read_optional_u32(&map, &attention_key(&map, "key_length")?)?;
        let key_length_swa = read_optional_u32(&map, &attention_key(&map, "key_length_swa")?)?;
        let sliding_window = read_optional_u32(&map, &attention_key(&map, "sliding_window")?)?;
        let full_attention_interval = map
            .get_usize(&format!("{arch_ref}.full_attention_interval"))
            .ok()
            .map(|v| v as u32);

        // Which layers use sliding-window attention vs full? Gemma-3/4
        // publish `<arch>.attention.sliding_window_pattern` as a bool
        // array with one entry per layer. Other models either set every
        // layer to the same pattern (empty / absent array) or have no
        // SWA at all (sliding_window == 0).
        let swa_pattern =
            read_bool_array_field(&map, &attention_key(&map, "sliding_window_pattern")?)?;

        let (full_kv_heads, swa_kv_heads) = split_kv_heads(
            &kv_heads_per_layer,
            &swa_pattern,
            sliding_window,
            n_layers,
            full_attention_interval,
        );
        let kv_heads_total = full_kv_heads + swa_kv_heads;
        let n_head_kv_max = kv_heads_per_layer.iter().copied().max().unwrap_or(0) as u32;

        Ok(Self {
            max_context,
            n_layers,
            n_embd,
            n_head: head_stats.max,
            n_head_kv: n_head_kv_max,
            key_length,
            key_length_swa,
            full_kv_heads,
            swa_kv_heads,
            kv_heads_total,
            sliding_window,
            full_attention_interval,
            file_size,
            expert_weight_bytes: read_expert_weight_bytes(path, file_size),
        })
    }

    /// Whether this is a Mixture-of-Experts model (has expert weight tensors).
    pub fn is_moe(&self) -> bool {
        self.expert_weight_bytes > 0
    }

    /// Non-expert ("dense") weight bytes: attention, norms, embeddings, output,
    /// router, and shared experts — everything that isn't a routed expert. For
    /// a MoE model these stay GPU-resident while experts can spill to CPU.
    pub fn dense_weight_bytes(&self) -> u64 {
        self.file_size.saturating_sub(self.expert_weight_bytes)
    }

    /// Per-head dimension for full-context attention layers. Prefers
    /// the explicit `attention.key_length`; falls back to `n_embd /
    /// n_head`; then to 128 as a last resort.
    pub fn n_embd_head(&self) -> u64 {
        if self.key_length > 0 {
            self.key_length as u64
        } else if self.n_head > 0 {
            self.n_embd as u64 / self.n_head as u64
        } else {
            128
        }
    }

    /// Per-head dimension for sliding-window attention layers. Gemma
    /// uses a smaller dim here (256 vs 512 for full) — if the GGUF
    /// doesn't expose a separate value, we reuse the full-attention
    /// key length.
    pub fn n_embd_head_swa(&self) -> u64 {
        if self.key_length_swa > 0 {
            self.key_length_swa as u64
        } else {
            self.n_embd_head()
        }
    }

    /// Bytes the KV cache will occupy at the given runtime `context`,
    /// honouring sliding-window attention when present. K and V caches
    /// are accounted for separately because `cache_type_k` and
    /// `cache_type_v` can be configured independently.
    ///
    /// For a non-SWA model this collapses to:
    /// ```text
    /// context * kv_heads_total * n_embd_head * (k_bytes + v_bytes)
    /// ```
    ///
    /// For a gemma-style SWA model, the window-bounded layers only
    /// cache `min(context, sliding_window)` tokens, with a smaller
    /// per-head dim — so the estimate drops from ~115 GiB to ~5 GiB
    /// at 131 072 / q8.
    pub fn kv_cache_bytes(&self, context: u32, kv_bytes: KvPerElement) -> u64 {
        let ctx = context as u64;
        let full_elements = ctx * self.full_kv_heads * self.n_embd_head();
        let swa_elements = if self.sliding_window > 0 {
            let eff = ctx.min(self.sliding_window as u64);
            eff * self.swa_kv_heads * self.n_embd_head_swa()
        } else {
            0
        };
        let elements = full_elements + swa_elements;
        elements * kv_bytes.k.0 / kv_bytes.k.1 + elements * kv_bytes.v.0 / kv_bytes.v.1
    }
}

impl From<&GgufMeta> for GgufInfo {
    fn from(m: &GgufMeta) -> Self {
        let (full_kv_heads, swa_kv_heads) = normalize_cached_kv_heads(
            m.full_kv_heads,
            m.swa_kv_heads,
            m.sliding_window,
            m.n_layers,
            m.n_head_kv,
            m.full_attention_interval,
        );
        Self {
            max_context: m.max_context,
            n_layers: m.n_layers,
            n_embd: m.n_embd,
            n_head: m.n_head,
            n_head_kv: m.n_head_kv,
            key_length: m.key_length,
            key_length_swa: m.key_length_swa,
            full_kv_heads,
            swa_kv_heads,
            kv_heads_total: full_kv_heads + swa_kv_heads,
            sliding_window: m.sliding_window,
            full_attention_interval: m.full_attention_interval,
            file_size: m.file_size,
            expert_weight_bytes: m.expert_weight_bytes,
        }
    }
}

fn normalize_cached_kv_heads(
    full_kv_heads: u64,
    swa_kv_heads: u64,
    sliding_window: u32,
    n_layers: u32,
    n_head_kv: u32,
    full_attention_interval: Option<u32>,
) -> (u64, u64) {
    if sliding_window == 0 {
        if let Some(interval) = full_attention_interval.filter(|&v| v > 1) {
            if n_layers > 0 && n_head_kv > 0 {
                return ((n_layers / interval) as u64 * n_head_kv as u64, 0);
            }
        }
    }
    (full_kv_heads, swa_kv_heads)
}

// Canonical llama-7b-ish params used for arithmetic checks.
const FILE_SIZE: u64 = 6_442_450_944; // ~6 GiB

/// Build a uniform (non-SWA) GgufInfo: every layer full-context
/// with `n_head_kv` KV heads and head_dim = `key_length`.
fn uniform_info(n_layers: u32, n_head_kv: u32, key_length: u32, file_size: u64) -> GgufInfo {
    GgufInfo {
        max_context: 0,
        n_layers,
        n_embd: 0,
        n_head: n_head_kv,
        n_head_kv,
        key_length,
        key_length_swa: 0,
        full_kv_heads: n_layers as u64 * n_head_kv as u64,
        swa_kv_heads: 0,
        kv_heads_total: n_layers as u64 * n_head_kv as u64,
        sliding_window: 0,
        full_attention_interval: None,
        file_size,
        expert_weight_bytes: 0,
    }
}

#[test]
fn kv_cache_scales_linearly_with_context() {
    let info = uniform_info(32, 32, 128, FILE_SIZE);
    let a = info.kv_cache_bytes(4096, KvPerElement::FP16_BOTH);
    let b = info.kv_cache_bytes(16384, KvPerElement::FP16_BOTH);
    let ratio = b as f64 / a as f64;
    assert!((3.9..=4.1).contains(&ratio), "ratio={ratio}");
}

#[test]
fn gqa_shrinks_kv_cache_proportionally() {
    let mha = uniform_info(32, 32, 128, 0).kv_cache_bytes(4096, KvPerElement::FP16_BOTH);
    let gqa = uniform_info(32, 4, 128, 0).kv_cache_bytes(4096, KvPerElement::FP16_BOTH);
    let ratio = mha as f64 / gqa as f64;
    assert!((7.9..=8.1).contains(&ratio), "ratio={ratio}");
}

#[test]
fn zero_context_yields_no_kv_cache() {
    let info = uniform_info(32, 32, 128, FILE_SIZE);
    let kv = info.kv_cache_bytes(0, KvPerElement::FP16_BOTH);
    assert_eq!(kv, 0);
    let e = VramEstimate::compute(FILE_SIZE, kv, 32, None);
    assert_eq!(e.total_vram, (FILE_SIZE * 11) / 10);
}

#[test]
fn overhead_is_ten_percent() {
    let info = uniform_info(32, 32, 128, FILE_SIZE);
    let kv = info.kv_cache_bytes(4096, KvPerElement::FP16_BOTH);
    let e = VramEstimate::compute(FILE_SIZE, kv, 32, None);
    assert_eq!(e.total_vram, (FILE_SIZE + kv) * 11 / 10);
}

#[test]
#[ignore = "reads a real multi-GB sharded GGUF; run explicitly with --ignored"]
fn real_mimo_expert_split_is_sane() {
    let path = Path::new(
        "/home/aristath/models/mimo-v2.5/UD-Q4_K_XL/MiMo-V2.5-UD-Q4_K_XL-00001-of-00005.gguf",
    );
    let info = GgufInfo::read(path).unwrap();
    let gib = 1u64 << 30;
    eprintln!(
        "file={} GiB  experts={} GiB  dense={} GiB  moe={}",
        info.file_size / gib,
        info.expert_weight_bytes / gib,
        info.dense_weight_bytes() / gib,
        info.is_moe()
    );
    assert!(info.is_moe());
    // Experts dominate but are a strict subset of the file.
    assert!(info.expert_weight_bytes > 100 * gib);
    assert!(info.expert_weight_bytes < info.file_size);
    assert!(info.dense_weight_bytes() > 0);
}

#[test]
#[ignore = "reads a real multi-GB sharded GGUF; run explicitly with --ignored"]
fn real_glm52_expert_split_never_exceeds_file() {
    let path = Path::new(
        "/home/aristath/models/glm-5.2/UD-Q2_K_XL/GLM-5.2-UD-Q2_K_XL-00001-of-00007.gguf",
    );
    let info = GgufInfo::read(path).unwrap();
    let gib = 1u64 << 30;
    eprintln!(
        "file={} GiB  experts={} GiB  dense={} GiB",
        info.file_size / gib,
        info.expert_weight_bytes / gib,
        info.dense_weight_bytes() / gib
    );
    // The bug: experts (309) once exceeded the file (236). Must not anymore.
    assert!(info.expert_weight_bytes < info.file_size);
    assert!(info.dense_weight_bytes() > 0);
}

#[test]
#[ignore = "reads a real multi-GB sharded GGUF; run explicitly with --ignored"]
fn real_deepseek_mxfp4_meta_reads() {
    let path = Path::new(
        "/mnt/models/models/deepseek-v4/flash/MXFP4/DeepSeek-V4-Flash-MXFP4-00001-of-00004.gguf",
    );
    let meta = GgufMeta::read(path).unwrap();
    eprintln!(
        "name={:?} quant={:?} file={} experts={}",
        meta.name, meta.quant_label, meta.file_size, meta.expert_weight_bytes
    );
    assert_eq!(meta.quant_label.as_deref(), Some("MXFP4_MOE"));
    assert!(meta.file_size > 100 * (1u64 << 30));
}

#[test]
fn expert_tensor_classifier_matches_llama_naming() {
    assert!(is_expert_tensor("blk.5.ffn_gate_exps.weight"));
    assert!(is_expert_tensor("blk.12.ffn_down_exps.weight"));
    assert!(!is_expert_tensor("blk.5.attn_q.weight"));
    assert!(!is_expert_tensor("blk.5.ffn_gate_inp.weight")); // router stays on GPU
    assert!(!is_expert_tensor("token_embd.weight"));
}

#[test]
fn ggml_type_layout_includes_current_llama_types() {
    assert_eq!(ggml_type_layout(34), Some((256, 54))); // TQ1_0
    assert_eq!(ggml_type_layout(35), Some((256, 66))); // TQ2_0
    assert_eq!(ggml_type_layout(39), Some((32, 17))); // MXFP4
    assert_eq!(ggml_type_layout(40), Some((64, 36))); // NVFP4
    assert_eq!(ggml_type_layout(41), Some((128, 18))); // Q1_0
    assert_eq!(tensor_nbytes(39, &[32, 4]).unwrap(), 68);
}

#[test]
fn partial_offload_scales_vram_by_gpu_layer_fraction() {
    // 51-layer model, only 14 layers on GPU → ~27% of weights+KV is VRAM,
    // and the other 37 layers' KV becomes CPU RAM (weights stay mmap'd).
    let info = uniform_info(51, 8, 128, FILE_SIZE);
    let kv = info.kv_cache_bytes(8192, KvPerElement::FP16_BOTH);
    let full = VramEstimate::compute(FILE_SIZE, kv, 51, Some(99));
    let partial = VramEstimate::compute(FILE_SIZE, kv, 51, Some(14));

    // Full offload is unchanged from the old behaviour.
    assert_eq!(full.total_vram, (FILE_SIZE + kv) * 11 / 10);
    assert_eq!(full.cpu_kv_ram, 0);

    // Partial offload is far smaller and frees most of the KV to CPU RAM.
    assert!(partial.total_vram < full.total_vram / 3, "{partial:?}");
    let frac = 14.0 / 51.0;
    let expect_cpu_kv = kv - (kv as f64 * frac).round() as u64;
    assert_eq!(partial.cpu_kv_ram, expect_cpu_kv);
}

#[test]
fn q8_kv_quantization_halves_kv_cache_vs_fp16() {
    let info = uniform_info(32, 32, 128, 0);
    let fp16 = info.kv_cache_bytes(4096, KvPerElement::FP16_BOTH);
    let q8 = info.kv_cache_bytes(
        4096,
        KvPerElement::from_types(CacheType::Q8_0, CacheType::Q8_0),
    );
    assert_eq!(q8 * 2, fp16);
}

#[test]
fn q4_kv_quantization_quarters_kv_cache_vs_fp16() {
    let info = uniform_info(32, 32, 128, 0);
    let fp16 = info.kv_cache_bytes(4096, KvPerElement::FP16_BOTH);
    let q4 = info.kv_cache_bytes(
        4096,
        KvPerElement::from_types(CacheType::Q4_0, CacheType::Q4_0),
    );
    assert_eq!(q4 * 4, fp16);
}

#[test]
fn mixed_kv_quantization_sums_k_and_v_independently() {
    let info = uniform_info(32, 32, 128, 0);
    let mixed = info.kv_cache_bytes(
        4096,
        KvPerElement::from_types(CacheType::Q4_0, CacheType::F16),
    );
    // K at 0.5 bytes/elem + V at 2.0 bytes/elem = 2.5 bytes/elem
    // vs. 4.0 for f16 both → 5/8 ratio.
    let fp16 = info.kv_cache_bytes(4096, KvPerElement::FP16_BOTH);
    assert_eq!(mixed * 8, fp16 * 5);
}

fn info_head_dim(n_embd: u32, n_head: u32, n_head_kv: u32, key_length: u32) -> GgufInfo {
    GgufInfo {
        max_context: 0,
        n_layers: 0,
        n_embd,
        n_head,
        n_head_kv,
        key_length,
        key_length_swa: 0,
        full_kv_heads: 0,
        swa_kv_heads: 0,
        kv_heads_total: 0,
        sliding_window: 0,
        full_attention_interval: None,
        file_size: 0,
        expert_weight_bytes: 0,
    }
}

#[test]
fn n_embd_head_prefers_key_length_when_present() {
    // 128 wins even though n_embd / n_head = 42.
    assert_eq!(info_head_dim(4096, 96, 8, 128).n_embd_head(), 128);
}

#[test]
fn n_embd_head_falls_back_to_n_embd_over_n_head() {
    assert_eq!(info_head_dim(4096, 32, 8, 0).n_embd_head(), 128);
}

#[test]
fn n_embd_head_defaults_to_128_when_head_count_unknown() {
    assert_eq!(info_head_dim(4096, 0, 0, 0).n_embd_head(), 128);
}

#[test]
fn n_embd_head_swa_falls_back_to_full_head_dim_when_absent() {
    let mut info = info_head_dim(4096, 32, 8, 128);
    info.key_length_swa = 0;
    assert_eq!(info.n_embd_head_swa(), 128);
}

#[test]
fn n_embd_head_swa_uses_its_own_key_length_when_set() {
    let mut info = info_head_dim(4096, 32, 8, 512);
    info.key_length_swa = 256;
    assert_eq!(info.n_embd_head_swa(), 256);
}

// ---- Sliding-window / hybrid attention ----

/// A gemma-4-style info. 10 global layers with 4 KV heads each,
/// 50 sliding-window layers with 16 KV heads each. Global head_dim
/// 512, SWA head_dim 256, window 1024.
fn gemma_like_info() -> GgufInfo {
    GgufInfo {
        max_context: 262_144,
        n_layers: 60,
        n_embd: 5376,
        n_head: 32,
        n_head_kv: 16,
        key_length: 512,
        key_length_swa: 256,
        full_kv_heads: 40, // 10 layers × 4 heads
        swa_kv_heads: 800, // 50 layers × 16 heads
        kv_heads_total: 840,
        sliding_window: 1024,
        full_attention_interval: None,
        file_size: 0,
        expert_weight_bytes: 0,
    }
}

#[test]
fn swa_layers_cap_at_sliding_window_size() {
    // Run at a very large context. SWA contribution should NOT
    // scale with context once context exceeds the window.
    let info = gemma_like_info();
    let small_ctx = info.kv_cache_bytes(1024, KvPerElement::FP16_BOTH);
    let huge_ctx = info.kv_cache_bytes(262_144, KvPerElement::FP16_BOTH);

    // At 1024 ctx: full = 1024*40*512*4 ; swa = 1024*800*256*4
    // At 262144 ctx: full = 262144*40*512*4 ; swa = 1024*800*256*4
    // (SWA unchanged.) So the delta is purely the full-context
    // contribution, and the ratio of SWA bytes to full bytes at
    // 262144 ctx is 256*800 / (40*512*262144/1024) which is way
    // smaller than 1 — sanity-check it's < 10%.
    let full_only = 4 * 262_144u64 * 40 * 512;
    let total_elements = full_only + (4 * 1024 * 800 * 256);
    assert_eq!(huge_ctx, total_elements);
    assert!(small_ctx < huge_ctx);
    // SWA contribution at small ctx is the same as at huge ctx.
    let small_full = 4 * 1024u64 * 40 * 512;
    let small_swa = 4 * 1024u64 * 800 * 256;
    assert_eq!(small_ctx, small_full + small_swa);
}

#[test]
fn gemma_like_estimate_is_dominated_by_the_10_full_layers() {
    // At 131072 ctx / q8 KV, gemma-4-31b should be nowhere near
    // 100+ GiB of KV cache — that was the bug this path fixes.
    let info = gemma_like_info();
    let kv = info.kv_cache_bytes(
        131_072,
        KvPerElement::from_types(CacheType::Q8_0, CacheType::Q8_0),
    );
    let gib = kv as f64 / 1_073_741_824.0;
    // Expected ~5.4 GiB; give a generous window.
    assert!((4.0..=7.0).contains(&gib), "got {gib} GiB, expected ~5 GiB");
}

#[test]
fn non_swa_model_ignores_sliding_window_fields() {
    // A plain full-attention model with sliding_window = 0.
    // swa_kv_heads = 0, so no SWA contribution.
    let info = uniform_info(32, 32, 128, 0);
    let kv = info.kv_cache_bytes(4096, KvPerElement::FP16_BOTH);
    // 32 layers * 32 heads * 128 dim * 4096 ctx * 4 bytes = reference.
    let ref_bytes = 32u64 * 32 * 128 * 4096 * 4;
    assert_eq!(kv, ref_bytes);
}

#[test]
fn qwen35_like_recurrent_model_only_charges_full_attention_layers() {
    // Qwen3.5/Qwen3.6 publish full_attention_interval = 4: every
    // 4th layer allocates full KV; the recurrent layers do not.
    let info = GgufInfo {
        max_context: 262_144,
        n_layers: 64,
        n_embd: 5120,
        n_head: 40,
        n_head_kv: 4,
        key_length: 256,
        key_length_swa: 0,
        full_kv_heads: 64, // 16 full-attention layers * 4 KV heads
        swa_kv_heads: 0,
        kv_heads_total: 64,
        sliding_window: 0,
        full_attention_interval: Some(4),
        file_size: 0,
        expert_weight_bytes: 0,
    };

    let kv = info.kv_cache_bytes(262_144, KvPerElement::FP16_BOTH);
    assert_eq!(kv, 16 * 1024 * 1024 * 1024);
}

// ---- split_kv_heads ----

#[test]
fn split_kv_heads_no_sliding_window_folds_all_into_full() {
    let (full, swa) = split_kv_heads(&[16, 16, 16], &[true, true, false], 0, 3, None);
    assert_eq!(full, 48);
    assert_eq!(swa, 0);
}

#[test]
fn split_kv_heads_uses_full_attention_interval_for_recurrent_models() {
    let (full, swa) = split_kv_heads(&[4, 4, 4, 4, 4, 4, 4, 4], &[], 0, 8, Some(4));
    assert_eq!(full, 8);
    assert_eq!(swa, 0);
}

#[test]
fn split_kv_heads_zips_pattern_with_per_layer_counts() {
    // Typical gemma shape: a few global layers sprinkled among
    // many SWA ones, each with different KV-head counts.
    let kv = vec![4, 16, 16, 16, 16, 16]; // 1 global + 5 SWA
    let pat = vec![false, true, true, true, true, true];
    let (full, swa) = split_kv_heads(&kv, &pat, 1024, 6, None);
    assert_eq!(full, 4);
    assert_eq!(swa, 16 * 5);
}

#[test]
fn split_kv_heads_falls_back_to_all_full_on_mismatched_lengths() {
    // n_layers = 4 but we only got 3 entries — conservatively
    // treat the whole model as full-context.
    let (full, swa) = split_kv_heads(&[8, 8, 8], &[true, false, true], 1024, 4, None);
    assert_eq!(full, 24);
    assert_eq!(swa, 0);
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
