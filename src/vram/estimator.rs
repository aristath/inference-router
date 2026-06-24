use crate::config::CacheType;
use ggus::{
    GGufMetaDataValueType as Ty, GGufMetaError, GGufMetaMap, GGufMetaMapExt, GGufMetaValueArray,
    GGufReader,
};
use memmap2::Mmap;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::fs::File;
use std::path::{Path, PathBuf};

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
    #[allow(dead_code)] // used by tests and available for callers
    pub const FP16_BOTH: Self = Self {
        k: (2, 1),
        v: (2, 1),
    };

    pub fn from_types(k: CacheType, v: CacheType) -> Self {
        Self {
            k: k.bytes_per_element(),
            v: v.bytes_per_element(),
        }
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

    /// MoE-aware estimate. Dense/attention weights and the full KV cache are
    /// GPU-resident (all attention runs on GPU every token); the routed experts
    /// are split by `n_cpu_moe` — the first `n_cpu_moe` of `n_layers` layers'
    /// experts go to CPU (mmap'd, file-backed) and the rest stay on GPU.
    pub fn compute_moe(
        dense_bytes: u64,
        expert_bytes: u64,
        kv_cache_bytes: u64,
        n_layers: u32,
        n_cpu_moe: u32,
    ) -> Self {
        let n_cpu = n_cpu_moe.min(n_layers);
        let gpu_expert_frac = if n_layers == 0 {
            0.0
        } else {
            (n_layers - n_cpu) as f64 / n_layers as f64
        };
        let experts_on_gpu = scale_bytes(expert_bytes, gpu_expert_frac);
        let weight_vram = dense_bytes.saturating_add(experts_on_gpu);
        let total = (weight_vram + kv_cache_bytes).saturating_mul(11) / 10;
        Self {
            weight_vram,
            kv_cache_vram: kv_cache_bytes,
            total_vram: total,
            // CPU-resident experts are mmap'd weights (file-backed, reclaimable),
            // not anonymous RAM, so they don't count against the admission RAM.
            cpu_kv_ram: 0,
        }
    }
}

/// Smallest `n_cpu_moe` (fewest CPU-expert layers → most experts packed into
/// VRAM) whose MoE estimate fits `free_vram`. `None` if even all-experts-on-CPU
/// (`n_cpu_moe = n_layers`, i.e. only dense + KV on GPU) doesn't fit.
pub fn auto_n_cpu_moe(
    dense_bytes: u64,
    expert_bytes: u64,
    kv_cache_bytes: u64,
    n_layers: u32,
    free_vram: u64,
) -> Option<u32> {
    (0..=n_layers).find(|&n| {
        VramEstimate::compute_moe(dense_bytes, expert_bytes, kv_cache_bytes, n_layers, n).total_vram
            <= free_vram
    })
}

/// The `n_cpu` layer indices whose experts go to CPU, spread *evenly* across
/// `[0, n_layers)` rather than taking the first N.
///
/// `--n-cpu-moe N` offloads the first N layers; combined with `--split-mode
/// layer` (which assigns GPUs contiguous low→high layer ranges) that clusters
/// all the light CPU-expert layers onto the first GPUs, leaving them half-empty
/// while the GPU-expert layers fill the rest. Interleaving the offloaded layers
/// gives every GPU's range a uniform light/heavy mix, so VRAM fills evenly and
/// more experts fit on GPU.
pub fn cpu_moe_layers(n_layers: u32, n_cpu: u32) -> Vec<u32> {
    let n_cpu = n_cpu.min(n_layers);
    if n_cpu == 0 {
        return Vec::new();
    }
    (0..n_cpu)
        .map(|j| ((j as u64 * n_layers as u64) / n_cpu as u64) as u32)
        .collect()
}

/// Boundary-aware version of [`cpu_moe_layers`]: given the per-GPU
/// `--tensor-split` fractions (which determine each GPU's *contiguous* layer
/// range under `--split-mode layer`), offload each GPU's *proportional share*
/// of layers, spread within that GPU's range. Every GPU then keeps the same
/// fraction of GPU-resident experts, so VRAM fills to each GPU's cap together
/// instead of one capping out while others sit half-empty.
pub fn boundary_aware_cpu_moe_layers(n_layers: u32, n_cpu: u32, split_fracs: &[f64]) -> Vec<u32> {
    let n_cpu = n_cpu.min(n_layers);
    if n_cpu == 0 || n_layers == 0 {
        return Vec::new();
    }
    let total: f64 = split_fracs
        .iter()
        .filter(|f| f.is_finite())
        .map(|f| f.max(0.0))
        .sum();
    if split_fracs.is_empty() || total <= 0.0 {
        return cpu_moe_layers(n_layers, n_cpu); // no split info → global spread
    }
    let offload_frac = n_cpu as f64 / n_layers as f64;
    let mut result = Vec::new();
    let mut cum = 0.0f64;
    let mut start = 0u32;
    let last = split_fracs.len() - 1;
    for (i, f) in split_fracs.iter().enumerate() {
        cum += f.max(0.0);
        let end = if i == last {
            n_layers
        } else {
            (((cum / total) * n_layers as f64).round() as u32).clamp(start, n_layers)
        };
        let range = end.saturating_sub(start);
        if range > 0 {
            // Offload this GPU's proportional share, spread within [start, end).
            let n_off = (offload_frac * range as f64).round() as u32;
            for j in 0..n_off {
                let layer = start + ((j as u64 * range as u64) / n_off.max(1) as u64) as u32;
                result.push(layer.min(end - 1));
            }
        }
        start = end;
    }
    result.sort_unstable();
    result.dedup();
    result
}

/// Build the `--override-tensor` value pinning these layers' expert tensors to
/// CPU: `blk\.(L1|L2|…)\.ffn_.*_exps=CPU`. `None` for an empty layer set.
pub fn cpu_moe_override_from_layers(layers: &[u32]) -> Option<String> {
    if layers.is_empty() {
        return None;
    }
    let alt = layers
        .iter()
        .map(u32::to_string)
        .collect::<Vec<_>>()
        .join("|");
    // The trailing `\.` keeps `2` from matching `blk.21.…`.
    Some(format!(r"blk\.({alt})\.ffn_.*_exps=CPU"))
}

/// Globally-spread `--override-tensor` (used as a fallback before the split is
/// known). [`boundary_aware_cpu_moe_layers`] is preferred once it is.
pub fn moe_cpu_override_tensor(n_layers: u32, n_cpu: u32) -> Option<String> {
    cpu_moe_override_from_layers(&cpu_moe_layers(n_layers, n_cpu))
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

/// Maps `general.file_type` integer to its canonical quant label string.
pub fn file_type_label(ft: u32) -> &'static str {
    match ft {
        0 => "F32",
        1 => "F16",
        2 => "Q4_0",
        3 => "Q4_1",
        4 => "Q4_1_F16",
        7 => "Q8_0",
        8 => "Q5_0",
        9 => "Q5_1",
        10 => "Q2_K",
        11 => "Q3_K_S",
        12 => "Q3_K_M",
        13 => "Q3_K_L",
        14 => "Q4_K_S",
        15 => "Q4_K_M",
        16 => "Q5_K_S",
        17 => "Q5_K_M",
        18 => "Q6_K",
        19 => "IQ2_XXS",
        20 => "IQ2_XS",
        21 => "Q2_K_S",
        22 => "IQ3_XS",
        23 => "IQ3_XXS",
        24 => "IQ1_S",
        25 => "IQ4_NL",
        26 => "IQ3_S",
        27 => "IQ3_M",
        28 => "IQ2_S",
        29 => "IQ2_M",
        30 => "IQ4_XS",
        31 => "IQ1_M",
        32 => "BF16",
        33 => "Q4_0_4_4",
        34 => "Q4_0_4_8",
        35 => "Q4_0_8_8",
        _ => "Unknown",
    }
}

/// Richer GGUF metadata — a superset of `GgufInfo` — used for the
/// dashboard's 2-step "Add model" flow and stored as `gguf_meta` in the
/// model JSON so the UI can render rich labels without re-reading the file.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct GgufMeta {
    // VRAM-estimation fields (same names as GgufInfo for front-end compat)
    pub max_context: u32,
    pub n_layers: u32,
    pub n_embd: u32,
    pub n_head: u32,
    pub n_head_kv: u32,
    pub key_length: u32,
    pub key_length_swa: u32,
    pub full_kv_heads: u64,
    pub swa_kv_heads: u64,
    pub kv_heads_total: u64,
    pub sliding_window: u32,
    pub file_size: u64,
    /// MoE expert weight bytes across shards (`> 0` iff MoE). See
    /// [`GgufInfo::expert_weight_bytes`].
    #[serde(default)]
    pub expert_weight_bytes: u64,
    // Identity
    pub architecture: Option<String>,
    pub name: Option<String>,
    pub basename: Option<String>,
    pub size_label: Option<String>,
    pub file_type: Option<u32>,
    pub quant_label: Option<String>,
    pub quantized_by: Option<String>,
    pub license: Option<String>,
    pub tags: Vec<String>,
    // Provenance
    pub base_model_name: Option<String>,
    pub base_model_org: Option<String>,
    pub base_model_repo: Option<String>,
    // Architecture detail
    pub feed_forward_length: Option<u32>,
    pub expert_count: Option<u32>,
    pub expert_used_count: Option<u32>,
    pub rope_freq_base: Option<f32>,
    pub ssm_inner_size: Option<u32>,
    #[serde(default)]
    pub full_attention_interval: Option<u32>,
    // Tokenizer
    pub chat_template: Option<String>,
    pub bos_token_id: Option<u32>,
    pub eos_token_id: Option<u32>,
    // Derived suggestions for the "Add model" form
    pub suggested_id: String,
    pub suggested_name: String,
}

impl GgufMeta {
    pub fn read(path: &Path) -> Result<Self, EstimateError> {
        let file = File::open(path)?;
        let file_size = sharded_total_size(path)
            .unwrap_or_else(|| file.metadata().map(|m| m.len()).unwrap_or(0));
        let mmap = unsafe { Mmap::map(&file) }?;
        let kvs = read_metadata_kvs(&mmap)?;
        let map = MetaMap(kvs);

        // --- GgufInfo fields (same logic as GgufInfo::read) ---
        let max_context = map.llm_context_length().map_err(meta_err)? as u32;
        let n_layers = map.llm_block_count().map_err(meta_err)? as u32;
        let n_embd = map.llm_embedding_length().map_err(meta_err)? as u32;
        let arch = map.general_architecture().ok().map(|s| s.to_string());
        let arch_ref = arch.as_deref().unwrap_or("");

        let head_stats = read_int_field(&map, &attention_key(&map, "head_count")?)?;
        let kv_heads_per_layer =
            read_int_field_per_layer(&map, &attention_key(&map, "head_count_kv")?, n_layers)?;

        let key_length = read_optional_u32(&map, &attention_key(&map, "key_length")?)?;
        let key_length_swa = read_optional_u32(&map, &attention_key(&map, "key_length_swa")?)?;
        let sliding_window = read_optional_u32(&map, &attention_key(&map, "sliding_window")?)?;
        let full_attention_interval = map
            .get_usize(&format!("{arch_ref}.full_attention_interval"))
            .ok()
            .map(|v| v as u32);

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

        // --- Identity ---
        let name = meta_read_str(&map, "general.name");
        let basename = meta_read_str(&map, "general.basename");
        let size_label = meta_read_str(&map, "general.size_label");
        let quantized_by = meta_read_str(&map, "general.quantized_by");
        let license = meta_read_str(&map, "general.license");
        let tags = meta_read_str_array(&map, "general.tags");

        let file_type: Option<u32> = map.get_usize("general.file_type").ok().map(|v| v as u32);
        let quant_label = file_type.map(|ft| file_type_label(ft).to_string());

        // --- Provenance ---
        let base_model_name = meta_read_str(&map, "general.base_model.0.name");
        let base_model_org = meta_read_str(&map, "general.base_model.0.organization");
        let base_model_repo = meta_read_str(&map, "general.base_model.0.repo_url");

        // --- Architecture detail ---
        let feed_forward_length = map
            .get_usize(&format!("{arch_ref}.feed_forward_length"))
            .ok()
            .map(|v| v as u32);
        let expert_count = map
            .get_usize(&format!("{arch_ref}.expert_count"))
            .ok()
            .map(|v| v as u32);
        let expert_used_count = map
            .get_usize(&format!("{arch_ref}.expert_used_count"))
            .ok()
            .map(|v| v as u32);
        let rope_freq_base = meta_read_f32(&map, &format!("{arch_ref}.rope.freq_base"));
        let ssm_inner_size = map
            .get_usize(&format!("{arch_ref}.ssm.inner_size"))
            .ok()
            .map(|v| v as u32);

        // --- Tokenizer ---
        let chat_template = meta_read_str(&map, "tokenizer.chat_template");
        let bos_token_id = map
            .get_usize("tokenizer.ggml.bos_token_id")
            .ok()
            .map(|v| v as u32);
        let eos_token_id = map
            .get_usize("tokenizer.ggml.eos_token_id")
            .ok()
            .map(|v| v as u32);

        // --- Derived suggestions ---
        let quant_lower = quant_label.as_deref().unwrap_or("").to_ascii_lowercase();
        let base_slug = {
            let raw = basename
                .as_deref()
                .or(name.as_deref())
                .unwrap_or_else(|| path.file_stem().and_then(|s| s.to_str()).unwrap_or("model"));
            slug(raw)
        };
        let suggested_id = if quant_lower.is_empty() {
            base_slug.clone()
        } else {
            format!("{base_slug}-{quant_lower}")
        };
        let display_name = name.as_deref().or(basename.as_deref()).unwrap_or("Model");
        let suggested_name = if quant_label.as_deref().is_none_or(|q| q == "Unknown") {
            display_name.to_owned()
        } else {
            format!("{} {}", display_name, quant_label.as_deref().unwrap_or(""))
        };

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
            file_size,
            expert_weight_bytes: read_expert_weight_bytes(path, file_size),
            architecture: arch,
            name,
            basename,
            size_label,
            file_type,
            quant_label,
            quantized_by,
            license,
            tags,
            base_model_name,
            base_model_org,
            base_model_repo,
            feed_forward_length,
            expert_count,
            expert_used_count,
            rope_freq_base,
            ssm_inner_size,
            full_attention_interval,
            chat_template,
            bos_token_id,
            eos_token_id,
            suggested_id,
            suggested_name,
        })
    }
}

/// Produce a dash-separated, lowercase ID-safe slug from a display string.
/// Keeps alphanumeric, `-`, and `.`; collapses runs of invalid chars to `-`.
fn slug(s: &str) -> String {
    let mut out = String::with_capacity(s.len());
    let mut last_was_sep = true;
    for c in s.chars() {
        if c.is_ascii_alphanumeric() || c == '.' {
            out.push(c.to_ascii_lowercase());
            last_was_sep = false;
        } else if c == '-' || c == '_' || c == ' ' {
            if !last_was_sep {
                out.push('-');
            }
            last_was_sep = true;
        }
        // other chars skipped
    }
    if out.ends_with('-') {
        out.pop();
    }
    out
}

/// Read an optional string from the GGUF metadata map.
/// GGUF string value format: `u64_len (LE) + UTF-8 bytes`.
fn meta_read_str(map: &MetaMap, key: &str) -> Option<String> {
    let (ty, bytes) = map.get(key)?;
    if ty != Ty::String || bytes.len() < 8 {
        return None;
    }
    let len = u64::from_le_bytes(bytes[..8].try_into().ok()?) as usize;
    let end = 8usize.checked_add(len)?;
    std::str::from_utf8(bytes.get(8..end)?)
        .ok()
        .map(|s| s.to_owned())
}

/// Read an optional f32 from the GGUF metadata map.
fn meta_read_f32(map: &MetaMap, key: &str) -> Option<f32> {
    let (ty, bytes) = map.get(key)?;
    if ty != Ty::F32 || bytes.len() < 4 {
        return None;
    }
    Some(f32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]))
}

/// Read a GGUF string array from the metadata map.
/// Array header: `elem_type (u32 LE) + count (u64 LE)`.
/// Each element: `u64_len (LE) + UTF-8 bytes`.
fn meta_read_str_array(map: &MetaMap, key: &str) -> Vec<String> {
    let (ty, bytes) = match map.get(key) {
        Some(v) => v,
        None => return Vec::new(),
    };
    if ty != Ty::Array || bytes.len() < 12 {
        return Vec::new();
    }
    let elem_ty = u32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]);
    if elem_ty != 8 {
        // 8 = STRING in GGUF spec
        return Vec::new();
    }
    let count = u64::from_le_bytes(bytes[4..12].try_into().unwrap_or([0; 8])) as usize;
    let mut pos = 12usize;
    let mut result = Vec::with_capacity(count.min(64));
    for _ in 0..count {
        if pos + 8 > bytes.len() {
            break;
        }
        let str_len = u64::from_le_bytes(bytes[pos..pos + 8].try_into().unwrap_or([0; 8])) as usize;
        pos += 8;
        if pos + str_len > bytes.len() {
            break;
        }
        if let Ok(s) = std::str::from_utf8(&bytes[pos..pos + str_len]) {
            result.push(s.to_owned());
        }
        pos += str_len;
    }
    result
}

fn read_optional_u32(map: &MetaMap, key: &str) -> Result<u32, EstimateError> {
    match map.get_usize(key) {
        Ok(v) => Ok(v as u32),
        Err(GGufMetaError::NotExist) => Ok(0),
        Err(GGufMetaError::TypeMismatch(_)) => Ok(0),
        Err(e) => Err(meta_err(e)),
    }
}

/// Read `head_count_kv` as a per-layer `Vec<u32>`. If the GGUF stores
/// a scalar, broadcast it across `n_layers`. Missing key → empty vec.
fn read_int_field_per_layer(
    map: &MetaMap,
    key: &str,
    n_layers: u32,
) -> Result<Vec<u32>, EstimateError> {
    let (ty, bytes) = match map.get(key) {
        Some(v) => v,
        None => return Ok(Vec::new()),
    };
    if ty == Ty::Array {
        let (values, _len) = read_int_array(bytes)?;
        Ok(values
            .into_iter()
            .map(|v| v.min(u32::MAX as u64) as u32)
            .collect())
    } else {
        let scalar = map.get_usize(key).map_err(meta_err)? as u32;
        Ok(vec![scalar; n_layers as usize])
    }
}

/// Read a bool array. Missing / unsupported-typed fields yield an empty
/// vec (caller treats this as "no SWA pattern").
fn read_bool_array_field(map: &MetaMap, key: &str) -> Result<Vec<bool>, EstimateError> {
    let (ty, bytes) = match map.get(key) {
        Some(v) => v,
        None => return Ok(Vec::new()),
    };
    if ty != Ty::Array {
        return Ok(Vec::new());
    }
    let mut r = GGufReader::new(bytes);
    let (elem_ty, len) = r.read_arr_header().map_err(read_err)?;
    if elem_ty != Ty::Bool {
        return Ok(Vec::new());
    }
    let arr = GGufMetaValueArray::<bool>::new(r, len);
    Ok(arr.filter_map(Result::ok).collect())
}

/// Partition the per-layer KV-head counts into (full-context sum,
/// sliding-window sum) using the layer pattern from the GGUF.
///
/// - If the model publishes no `sliding_window` size, every layer is
///   full-context (swa sum is 0), unless the architecture publishes
///   `full_attention_interval`, in which case only those interval layers
///   allocate full KV.
/// - If the model has SWA but publishes no per-layer pattern,
///   conservatively treat every layer as full-context — safer to
///   overestimate than under-estimate VRAM.
/// - If the pattern length doesn't match `n_layers`, same conservative
///   fallback.
///
/// The pattern convention matches llama.cpp: `true` at index `i` means
/// layer `i` uses sliding-window attention.
fn split_kv_heads(
    kv_per_layer: &[u32],
    swa_pattern: &[bool],
    sliding_window: u32,
    n_layers: u32,
    full_attention_interval: Option<u32>,
) -> (u64, u64) {
    let total: u64 = kv_per_layer.iter().map(|&v| v as u64).sum();
    if sliding_window == 0 {
        if let Some(interval) = full_attention_interval.filter(|&v| v > 1) {
            if kv_per_layer.len() == n_layers as usize {
                let full = kv_per_layer
                    .iter()
                    .enumerate()
                    .filter(|(idx, _)| ((idx + 1) as u32).is_multiple_of(interval))
                    .map(|(_, &heads)| heads as u64)
                    .sum();
                return (full, 0);
            }
        }
        return (total, 0);
    }
    if swa_pattern.len() != n_layers as usize || kv_per_layer.len() != n_layers as usize {
        // Without a reliable per-layer breakdown, overestimate rather
        // than silently halve the KV cache — treat all layers as full.
        return (total, 0);
    }
    let mut full = 0u64;
    let mut swa = 0u64;
    for (heads, is_swa) in kv_per_layer.iter().zip(swa_pattern.iter()) {
        if *is_swa {
            swa += *heads as u64;
        } else {
            full += *heads as u64;
        }
    }
    (full, swa)
}

/// Representative stats for an integer-shaped metadata field. Only
/// `max` is used these days — it's what the form UI displays. The
/// per-layer sum is computed by `read_int_field_per_layer` where
/// needed.
#[derive(Debug, Clone, Copy)]
struct IntFieldStats {
    max: u32,
}

/// Resolve `<general.architecture>.attention.<suffix>` into a full key.
fn attention_key(map: &MetaMap, suffix: &str) -> Result<String, EstimateError> {
    let arch = map.general_architecture().map_err(meta_err)?;
    Ok(format!("{arch}.attention.{suffix}"))
}

/// Read an integer-shaped metadata field that may be a scalar (any int
/// type) or an array of any int type. Missing key → zero max.
fn read_int_field(map: &MetaMap, key: &str) -> Result<IntFieldStats, EstimateError> {
    let (ty, bytes) = match map.get(key) {
        Some(v) => v,
        None => return Ok(IntFieldStats { max: 0 }),
    };
    if ty == Ty::Array {
        let (values, _len) = read_int_array(bytes)?;
        let max = values.iter().copied().max().unwrap_or(0);
        Ok(IntFieldStats {
            max: max.min(u32::MAX as u64) as u32,
        })
    } else {
        // Scalar: let ggus's get_usize handle sign extension / width.
        let v = map.get_usize(key).map_err(meta_err)?;
        Ok(IntFieldStats { max: v as u32 })
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
            let vals: Vec<u64> = arr.filter_map(Result::ok).map(|v| $lift(v)).collect();
            Ok((vals, len))
        }};
    }

    // Negative values (e.g. a `-1` sentinel in i32 arrays) collapse to 0
    // — they're not valid head counts. Positive values cast losslessly.
    let neg_to_zero = |v: i64| -> u64 {
        if v < 0 {
            0
        } else {
            v as u64
        }
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

    let parent: PathBuf = path
        .parent()
        .unwrap_or_else(|| Path::new("."))
        .to_path_buf();
    let mut sum = 0u64;
    for i in 1..=total {
        let shard = parent.join(format!("{stem}-{i:0width$}-of-{total:0width$}.gguf"));
        let size = std::fs::metadata(&shard).ok()?.len();
        sum = sum.checked_add(size)?;
    }
    Some(sum)
}

/// All on-disk shard paths for a model (just `[path]` for a single-file GGUF).
fn shard_paths(path: &Path) -> Vec<PathBuf> {
    let single = || vec![path.to_path_buf()];
    let Some(fname) = path.file_name().and_then(|s| s.to_str()) else {
        return single();
    };
    let Some(rest) = fname.strip_suffix(".gguf") else {
        return single();
    };
    let Some((stem_and_idx, total_str)) = rest.rsplit_once("-of-") else {
        return single();
    };
    let Ok(total) = total_str.parse::<u32>() else {
        return single();
    };
    let Some((stem, idx_str)) = stem_and_idx.rsplit_once('-') else {
        return single();
    };
    if total == 0 || idx_str.parse::<u32>().is_err() {
        return single();
    }
    let width = idx_str.len();
    let parent = path.parent().unwrap_or_else(|| Path::new("."));
    (1..=total)
        .map(|i| parent.join(format!("{stem}-{i:0width$}-of-{total:0width$}.gguf")))
        .collect()
}

/// True for a routed Mixture-of-Experts weight tensor — the exact set
/// llama.cpp's `--n-cpu-moe` / `--cpu-moe` move to CPU. These are named
/// `blk.<i>.ffn_{gate,up,down}_exps.weight`; matching the `_exps` marker covers
/// them while excluding the small router (`ffn_gate_inp`) and shared-expert
/// (`*_shexp`) tensors that stay GPU-resident.
fn is_expert_tensor(name: &str) -> bool {
    name.contains("_exps")
}

/// MoE expert-weight bytes for a model, as the experts' **share of the actual
/// on-disk size**. Returns 0 for a dense model.
///
/// We don't trust the absolute per-tensor `nbytes()`: the GGUF reader's quant
/// size table can be wrong for newer/mixed quant types (e.g. "UD-Q2_K_XL"),
/// which made the raw expert sum exceed the whole file. Instead we take the
/// expert *fraction* of the summed tensor bytes and apply it to `file_size`, so
/// a proportional size error cancels and the result is always ≤ `file_size`.
fn read_expert_weight_bytes(path: &Path, file_size: u64) -> u64 {
    let mut expert = 0u128;
    let mut all = 0u128;
    for shard in shard_paths(path) {
        let Ok(file) = File::open(&shard) else { continue };
        // Safety: read-only for the duration of this call; not mutated concurrently.
        let Ok(mmap) = (unsafe { Mmap::map(&file) }) else {
            continue;
        };
        if let Ok((e, a)) = sum_shard_tensor_bytes(&mmap) {
            expert += e as u128;
            all += a as u128;
        }
    }
    if all == 0 || expert == 0 {
        return 0;
    }
    ((file_size as u128 * expert / all).min(file_size as u128)) as u64
}

/// `(expert_tensor_bytes, all_tensor_bytes)` in one shard: read the header, skip
/// the metadata KVs, then walk the tensor descriptors.
fn sum_shard_tensor_bytes(data: &[u8]) -> Result<(u64, u64), EstimateError> {
    let mut reader = GGufReader::new(data);
    let header = reader.read_header().map_err(read_err)?;
    if !header.is_magic_correct() || !header.is_native_endian() {
        return Ok((0, 0));
    }
    for _ in 0..header.metadata_kv_count {
        reader.read_meta_kv().map_err(read_err)?;
    }
    let (mut expert, mut all) = (0u64, 0u64);
    for _ in 0..header.tensor_count {
        let meta = reader.read_tensor_meta().map_err(read_err)?;
        let nbytes = meta.to_info().nbytes() as u64;
        all = all.saturating_add(nbytes);
        if is_expert_tensor(meta.name()) {
            expert = expert.saturating_add(nbytes);
        }
    }
    Ok((expert, all))
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
        return Err(EstimateError::Gguf(
            "non-native endian not supported".into(),
        ));
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
    fn moe_estimate_keeps_dense_and_kv_on_gpu_and_splits_experts() {
        let dense = 20 << 30; // 20 GiB attention/dense
        let experts = 160 << 30; // 160 GiB routed experts
        let kv = 10 << 30; // 10 GiB KV
        let layers = 51;

        // All experts on GPU → dense + experts + KV, ×1.1.
        let all_gpu = VramEstimate::compute_moe(dense, experts, kv, layers, 0);
        assert_eq!(all_gpu.total_vram, (dense + experts + kv) * 11 / 10);
        // All experts on CPU → only dense + KV count as VRAM (experts mmap'd).
        let all_cpu = VramEstimate::compute_moe(dense, experts, kv, layers, layers);
        assert_eq!(all_cpu.total_vram, (dense + kv) * 11 / 10);
        assert_eq!(all_cpu.cpu_kv_ram, 0); // experts are file-backed, not anon RAM
        // More CPU-expert layers → less VRAM.
        let half = VramEstimate::compute_moe(dense, experts, kv, layers, 25);
        assert!(half.total_vram < all_gpu.total_vram && half.total_vram > all_cpu.total_vram);
    }

    #[test]
    fn auto_n_cpu_moe_packs_max_experts_into_free_vram() {
        let dense = 20 << 30;
        let experts = 160 << 30;
        let kv = 10 << 30;
        let layers = 51;
        // 128 GiB free: dense(20)+kv(10)=30 must stay, leaving ~98 GiB for
        // experts → a chunk of layers' experts spill to CPU, but not all.
        let free = 128u64 << 30;
        let n = auto_n_cpu_moe(dense, experts, kv, layers, free).unwrap();
        assert!(n > 0 && n < layers, "n_cpu_moe={n}");
        // The chosen plan actually fits, and one fewer CPU layer would not.
        assert!(VramEstimate::compute_moe(dense, experts, kv, layers, n).total_vram <= free);
        assert!(VramEstimate::compute_moe(dense, experts, kv, layers, n - 1).total_vram > free);

        // Too little VRAM even for dense+KV → None.
        assert!(auto_n_cpu_moe(dense, experts, kv, layers, 5 << 30).is_none());
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
    fn cpu_moe_layers_spread_evenly_not_clustered() {
        // 23 of 51 layers → evenly distributed across the whole range, distinct,
        // not the first 23 (which would cluster on the first GPUs).
        let layers = cpu_moe_layers(51, 23);
        assert_eq!(layers.len(), 23);
        assert!(layers.iter().all(|&l| l < 51));
        let mut sorted = layers.clone();
        sorted.dedup();
        assert_eq!(sorted.len(), 23, "all distinct");
        assert!(*layers.last().unwrap() > 40, "must reach high layers, not just 0..22");
        assert_eq!(cpu_moe_layers(51, 0), Vec::<u32>::new());
        assert_eq!(cpu_moe_layers(51, 51).len(), 51);
    }

    #[test]
    fn boundary_aware_offload_gives_each_gpu_its_proportional_share() {
        // 4 GPUs, last one (index 3) gets a 2x bigger split → its layer range is
        // bigger, so it must receive proportionally MORE offloaded layers.
        let fracs = vec![0.2, 0.2, 0.2, 0.4];
        let layers = boundary_aware_cpu_moe_layers(50, 25, &fracs); // offload ~half
        assert!(!layers.is_empty());
        assert!(layers.iter().all(|&l| l < 50));
        // Count offloaded layers in the big GPU's range (last 40% ≈ layers 30..50).
        let in_big = layers.iter().filter(|&&l| l >= 30).count();
        let in_first = layers.iter().filter(|&&l| l < 10).count();
        assert!(in_big > in_first, "big-split GPU should offload more: big={in_big} first={in_first}");
        // Empty split → falls back to the global spread (non-empty).
        assert!(!boundary_aware_cpu_moe_layers(50, 25, &[]).is_empty());
        assert!(boundary_aware_cpu_moe_layers(50, 0, &fracs).is_empty());
    }

    #[test]
    fn moe_override_tensor_builds_anchored_regex() {
        let ot = moe_cpu_override_tensor(4, 2).unwrap();
        assert!(ot.ends_with("=CPU"));
        assert!(ot.contains("ffn_.*_exps"));
        assert!(ot.contains(r"blk\."));
        assert!(moe_cpu_override_tensor(4, 0).is_none());
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
}
