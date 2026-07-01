use memmap2::Mmap;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::fs::File;
use std::path::{Path, PathBuf};

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
        36 => "TQ1_0",
        37 => "TQ2_0",
        38 => "MXFP4_MOE",
        39 => "NVFP4",
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
        Err(GGufMetaError::TypeMismatch) => Ok(0),
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
    let Some((elem_ty, len, mut pos)) = array_header(bytes) else {
        return Ok(Vec::new());
    };
    if elem_ty != Ty::Bool {
        return Ok(Vec::new());
    }
    let mut out = Vec::with_capacity(len.min(usize::MAX as u64) as usize);
    for _ in 0..len {
        let Some(value) = bytes.get(pos).copied() else {
            break;
        };
        if value <= 1 {
            out.push(value == 1);
        }
        pos += 1;
    }
    Ok(out)
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
        // Scalar: use the metadata helper for sign extension / width.
        let v = map.get_usize(key).map_err(meta_err)?;
        Ok(IntFieldStats { max: v as u32 })
    }
}

/// Parse a GGUF array of any integer element type into a `Vec<u64>`.
/// Non-integer element types yield `(vec![], len)` so the caller can
/// still inspect the length if it needs to.
fn read_int_array(bytes: &[u8]) -> Result<(Vec<u64>, u64), EstimateError> {
    let Some((elem_ty, len, mut pos)) = array_header(bytes) else {
        return Err(EstimateError::Gguf("bad array metadata".into()));
    };

    // Negative values (e.g. a `-1` sentinel in i32 arrays) collapse to 0
    // — they're not valid head counts. Positive values cast losslessly.
    let neg_to_zero = |v: i64| -> u64 {
        if v < 0 {
            0
        } else {
            v as u64
        }
    };

    macro_rules! collect_as {
        ($size:literal, $read:expr) => {{
            let mut values = Vec::with_capacity(len.min(usize::MAX as u64) as usize);
            for _ in 0..len {
                if pos + $size > bytes.len() {
                    break;
                }
                values.push($read(&bytes[pos..pos + $size]));
                pos += $size;
            }
            Ok((values, len))
        }};
    }

    match elem_ty {
        Ty::U8 => collect_as!(1, |b: &[u8]| b[0] as u64),
        Ty::I8 => collect_as!(1, |b: &[u8]| neg_to_zero(i8::from_le_bytes([b[0]]) as i64)),
        Ty::U16 => collect_as!(2, |b: &[u8]| u16::from_le_bytes([b[0], b[1]]) as u64),
        Ty::I16 => collect_as!(2, |b: &[u8]| neg_to_zero(
            i16::from_le_bytes([b[0], b[1]]) as i64
        )),
        Ty::U32 => collect_as!(4, |b: &[u8]| u32::from_le_bytes(b.try_into().unwrap())
            as u64),
        Ty::I32 => collect_as!(4, |b: &[u8]| neg_to_zero(
            i32::from_le_bytes(b.try_into().unwrap()) as i64
        )),
        Ty::U64 => collect_as!(8, |b: &[u8]| u64::from_le_bytes(b.try_into().unwrap())),
        Ty::I64 => collect_as!(8, |b: &[u8]| neg_to_zero(i64::from_le_bytes(
            b.try_into().unwrap()
        ))),
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
        let Ok(file) = File::open(&shard) else {
            continue;
        };
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
        let nbytes = tensor_nbytes(meta.kind, &meta.shape)?;
        all = all.saturating_add(nbytes);
        if is_expert_tensor(&meta.name) {
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
        map.insert(kv.key, (kv.ty, kv.value));
    }
    Ok(map)
}

struct MetaMap(HashMap<String, (Ty, Vec<u8>)>);

impl MetaMap {
    fn get(&self, key: &str) -> Option<(Ty, &[u8])> {
        self.0.get(key).map(|(ty, v)| (*ty, v.as_slice()))
    }

    fn get_usize(&self, key: &str) -> Result<usize, GGufMetaError> {
        let (ty, bytes) = self.get(key).ok_or(GGufMetaError::NotExist)?;
        let value = match ty {
            Ty::U8 => bytes.first().copied().map(u64::from),
            Ty::I8 => bytes
                .first()
                .map(|v| i8::from_le_bytes([*v]) as i64)
                .and_then(nonnegative),
            Ty::U16 => read_array::<2>(bytes)
                .map(u16::from_le_bytes)
                .map(u64::from),
            Ty::I16 => read_array::<2>(bytes)
                .map(i16::from_le_bytes)
                .map(|v| v as i64)
                .and_then(nonnegative),
            Ty::U32 => read_array::<4>(bytes)
                .map(u32::from_le_bytes)
                .map(u64::from),
            Ty::I32 => read_array::<4>(bytes)
                .map(i32::from_le_bytes)
                .map(|v| v as i64)
                .and_then(nonnegative),
            Ty::U64 => read_array::<8>(bytes).map(u64::from_le_bytes),
            Ty::I64 => read_array::<8>(bytes)
                .map(i64::from_le_bytes)
                .and_then(nonnegative),
            _ => return Err(GGufMetaError::TypeMismatch),
        }
        .ok_or(GGufMetaError::Invalid)?;
        usize::try_from(value).map_err(|_| GGufMetaError::Invalid)
    }

    fn general_architecture(&self) -> Result<String, GGufMetaError> {
        meta_read_str(self, "general.architecture").ok_or(GGufMetaError::NotExist)
    }

    fn llm_context_length(&self) -> Result<usize, GGufMetaError> {
        self.get_usize(&format!("{}.context_length", self.general_architecture()?))
    }

    fn llm_block_count(&self) -> Result<usize, GGufMetaError> {
        self.get_usize(&format!("{}.block_count", self.general_architecture()?))
    }

    fn llm_embedding_length(&self) -> Result<usize, GGufMetaError> {
        self.get_usize(&format!(
            "{}.embedding_length",
            self.general_architecture()?
        ))
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u32)]
enum Ty {
    U8 = 0,
    I8 = 1,
    U16 = 2,
    I16 = 3,
    U32 = 4,
    I32 = 5,
    F32 = 6,
    Bool = 7,
    String = 8,
    Array = 9,
    U64 = 10,
    I64 = 11,
    F64 = 12,
}

impl TryFrom<u32> for Ty {
    type Error = GgufReadError;

    fn try_from(value: u32) -> Result<Self, Self::Error> {
        match value {
            0 => Ok(Self::U8),
            1 => Ok(Self::I8),
            2 => Ok(Self::U16),
            3 => Ok(Self::I16),
            4 => Ok(Self::U32),
            5 => Ok(Self::I32),
            6 => Ok(Self::F32),
            7 => Ok(Self::Bool),
            8 => Ok(Self::String),
            9 => Ok(Self::Array),
            10 => Ok(Self::U64),
            11 => Ok(Self::I64),
            12 => Ok(Self::F64),
            _ => {
                let _ = value;
                Err(GgufReadError::Invalid)
            }
        }
    }
}

#[derive(Debug)]
enum GGufMetaError {
    NotExist,
    TypeMismatch,
    Invalid,
}

#[derive(Debug)]
enum GgufReadError {
    Eos,
    Utf8,
    Invalid,
}

#[derive(Debug)]
struct GgufHeader {
    magic: [u8; 4],
    version: u32,
    tensor_count: u64,
    metadata_kv_count: u64,
}

impl GgufHeader {
    fn is_magic_correct(&self) -> bool {
        self.magic == *b"GGUF"
    }

    fn is_native_endian(&self) -> bool {
        (2..=3).contains(&self.version)
    }
}

#[derive(Debug)]
struct MetaKv {
    key: String,
    ty: Ty,
    value: Vec<u8>,
}

#[derive(Debug)]
struct TensorMeta {
    name: String,
    shape: Vec<u64>,
    kind: u32,
}

struct GGufReader<'a> {
    data: &'a [u8],
    pos: usize,
}

impl<'a> GGufReader<'a> {
    fn new(data: &'a [u8]) -> Self {
        Self { data, pos: 0 }
    }

    fn read_header(&mut self) -> Result<GgufHeader, GgufReadError> {
        let magic = self.read_array::<4>()?;
        let version = self.read_u32()?;
        let tensor_count = self.read_u64()?;
        let metadata_kv_count = self.read_u64()?;
        Ok(GgufHeader {
            magic,
            version,
            tensor_count,
            metadata_kv_count,
        })
    }

    fn read_meta_kv(&mut self) -> Result<MetaKv, GgufReadError> {
        let key = self.read_string()?;
        let ty = Ty::try_from(self.read_u32()?)?;
        let start = self.pos;
        self.skip_value(ty)?;
        let value = self.data[start..self.pos].to_vec();
        Ok(MetaKv { key, ty, value })
    }

    fn read_tensor_meta(&mut self) -> Result<TensorMeta, GgufReadError> {
        let name = self.read_string()?;
        let dims = self.read_u32()?;
        if dims > 4 {
            let _ = dims;
            return Err(GgufReadError::Invalid);
        }
        let mut shape = Vec::with_capacity(dims as usize);
        for _ in 0..dims {
            shape.push(self.read_u64()?);
        }
        let kind = self.read_u32()?;
        let _offset = self.read_u64()?;
        Ok(TensorMeta { name, shape, kind })
    }

    fn read_string(&mut self) -> Result<String, GgufReadError> {
        let len = self.read_u64()?;
        let bytes = self.read_bytes(len as usize)?;
        std::str::from_utf8(bytes)
            .map(str::to_owned)
            .map_err(|_| GgufReadError::Utf8)
    }

    fn skip_value(&mut self, ty: Ty) -> Result<(), GgufReadError> {
        match ty {
            Ty::U8 | Ty::I8 | Ty::Bool => self.skip(1),
            Ty::U16 | Ty::I16 => self.skip(2),
            Ty::U32 | Ty::I32 | Ty::F32 => self.skip(4),
            Ty::U64 | Ty::I64 | Ty::F64 => self.skip(8),
            Ty::String => {
                let len = self.read_u64()?;
                self.skip(len as usize)
            }
            Ty::Array => {
                let elem_ty = Ty::try_from(self.read_u32()?)?;
                let len = self.read_u64()?;
                if let Some(size) = fixed_value_size(elem_ty) {
                    return self.skip(size.saturating_mul(len as usize));
                }
                for _ in 0..len {
                    self.skip_value(elem_ty)?;
                }
                Ok(())
            }
        }
    }

    fn read_u32(&mut self) -> Result<u32, GgufReadError> {
        self.read_array::<4>().map(u32::from_le_bytes)
    }

    fn read_u64(&mut self) -> Result<u64, GgufReadError> {
        self.read_array::<8>().map(u64::from_le_bytes)
    }

    fn read_array<const N: usize>(&mut self) -> Result<[u8; N], GgufReadError> {
        let bytes = self.read_bytes(N)?;
        bytes.try_into().map_err(|_| GgufReadError::Eos)
    }

    fn read_bytes(&mut self, len: usize) -> Result<&'a [u8], GgufReadError> {
        let end = self.pos.checked_add(len).ok_or(GgufReadError::Eos)?;
        let bytes = self.data.get(self.pos..end).ok_or(GgufReadError::Eos)?;
        self.pos = end;
        Ok(bytes)
    }

    fn skip(&mut self, len: usize) -> Result<(), GgufReadError> {
        self.read_bytes(len).map(|_| ())
    }
}

fn fixed_value_size(ty: Ty) -> Option<usize> {
    match ty {
        Ty::U8 | Ty::I8 | Ty::Bool => Some(1),
        Ty::U16 | Ty::I16 => Some(2),
        Ty::U32 | Ty::I32 | Ty::F32 => Some(4),
        Ty::U64 | Ty::I64 | Ty::F64 => Some(8),
        Ty::String | Ty::Array => None,
    }
}

fn array_header(bytes: &[u8]) -> Option<(Ty, u64, usize)> {
    if bytes.len() < 12 {
        return None;
    }
    let elem_ty = Ty::try_from(u32::from_le_bytes(bytes[0..4].try_into().ok()?)).ok()?;
    let len = u64::from_le_bytes(bytes[4..12].try_into().ok()?);
    Some((elem_ty, len, 12))
}

fn read_array<const N: usize>(bytes: &[u8]) -> Option<[u8; N]> {
    bytes.get(..N)?.try_into().ok()
}

fn nonnegative(value: i64) -> Option<u64> {
    (value >= 0).then_some(value as u64)
}

fn tensor_nbytes(kind: u32, shape: &[u64]) -> Result<u64, EstimateError> {
    let (block_size, type_size) = ggml_type_layout(kind)
        .ok_or_else(|| EstimateError::Gguf(format!("unsupported GGML tensor type {kind}")))?;
    let elements = shape
        .iter()
        .try_fold(1u64, |acc, dim| acc.checked_mul(*dim))
        .ok_or_else(|| EstimateError::Gguf("tensor element count overflow".into()))?;
    if block_size == 0 || type_size == 0 {
        return Err(EstimateError::Gguf(format!(
            "removed GGML tensor type {kind}"
        )));
    }
    elements
        .checked_mul(type_size)
        .map(|bytes| bytes / block_size)
        .ok_or_else(|| EstimateError::Gguf("tensor byte count overflow".into()))
}

fn ggml_type_layout(kind: u32) -> Option<(u64, u64)> {
    const QK_K: u64 = 256;
    match kind {
        0 => Some((1, 4)),                           // F32
        1 => Some((1, 2)),                           // F16
        2 => Some((32, 18)),                         // Q4_0
        3 => Some((32, 20)),                         // Q4_1
        4 | 5 => Some((0, 0)),                       // removed Q4_2/Q4_3
        6 => Some((32, 22)),                         // Q5_0
        7 => Some((32, 24)),                         // Q5_1
        8 => Some((32, 34)),                         // Q8_0
        9 => Some((32, 36)),                         // Q8_1
        10 => Some((QK_K, 84)),                      // Q2_K
        11 => Some((QK_K, 110)),                     // Q3_K
        12 => Some((QK_K, 144)),                     // Q4_K
        13 => Some((QK_K, 176)),                     // Q5_K
        14 => Some((QK_K, 210)),                     // Q6_K
        15 => Some((QK_K, 292)),                     // Q8_K
        16 => Some((QK_K, 66)),                      // IQ2_XXS
        17 => Some((QK_K, 74)),                      // IQ2_XS
        18 => Some((QK_K, 98)),                      // IQ3_XXS
        19 => Some((QK_K, 50)),                      // IQ1_S
        20 => Some((32, 18)),                        // IQ4_NL
        21 => Some((QK_K, 110)),                     // IQ3_S
        22 => Some((QK_K, 82)),                      // IQ2_S
        23 => Some((QK_K, 136)),                     // IQ4_XS
        24 => Some((1, 1)),                          // I8
        25 => Some((1, 2)),                          // I16
        26 => Some((1, 4)),                          // I32
        27 => Some((1, 8)),                          // I64
        28 => Some((1, 8)),                          // F64
        29 => Some((QK_K, 56)),                      // IQ1_M
        30 => Some((1, 2)),                          // BF16
        31 | 32 | 33 | 36 | 37 | 38 => Some((0, 0)), // removed repack types
        34 => Some((QK_K, 54)),                      // TQ1_0
        35 => Some((QK_K, 66)),                      // TQ2_0
        39 => Some((32, 17)),                        // MXFP4
        40 => Some((64, 36)),                        // NVFP4
        41 => Some((128, 18)),                       // Q1_0
        _ => None,
    }
}

fn read_err(e: GgufReadError) -> EstimateError {
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
#[path = "estimator_tests.rs"]
mod tests;
