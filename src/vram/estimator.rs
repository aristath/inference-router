use ggus::{
    GGufMetaDataValueType as Ty, GGufMetaError, GGufMetaMap, GGufMetaMapExt, GGufReader,
};
use memmap2::Mmap;
use std::collections::HashMap;
use std::fs::File;
use std::path::Path;

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

impl VramEstimate {
    /// Pure formula, independent of GGUF parsing. Unit-tested directly.
    ///
    /// ```text
    /// kv_cache = n_layers * 2 * context * n_head_kv * (n_embd / n_head) * 2   // FP16
    /// total    = (file_size + kv_cache) * 1.1                                 // 10% overhead
    /// ```
    pub fn compute(
        file_size: u64,
        context: u32,
        n_layers: u64,
        n_embd: u64,
        n_head: u64,
        n_head_kv: u64,
    ) -> Self {
        let n_embd_head = if n_head > 0 { n_embd / n_head } else { 128 };
        let kv_cache = n_layers * 2 * context as u64 * n_head_kv.max(1) * n_embd_head * 2;
        let total = (file_size + kv_cache).saturating_mul(11) / 10;
        Self { weight_vram: file_size, kv_cache_vram: kv_cache, total_vram: total }
    }

    /// Computes the estimate from a GGUF file on disk.
    ///
    /// We parse only the header + metadata KV pairs — not the tensor
    /// descriptors or data. This lets us handle oversized GGUFs (tens of GB
    /// of weights) cheaply, and sidesteps upstream bugs in tensor-bounds
    /// checking in `ggus::GGuf::new`.
    pub fn from_gguf(path: &Path, context: u32) -> Result<Self, EstimateError> {
        let info = GgufInfo::read(path)?;
        Ok(Self::compute(
            info.file_size,
            context,
            info.n_layers as u64,
            info.n_embd as u64,
            info.n_head as u64,
            info.n_head_kv as u64,
        ))
    }
}

/// All the GGUF metadata fields the orchestrator + form UI care about.
#[derive(Debug, Clone, Copy, serde::Serialize)]
pub struct GgufInfo {
    pub max_context: u32,
    pub n_layers: u32,
    pub n_embd: u32,
    pub n_head: u32,
    pub n_head_kv: u32,
    pub file_size: u64,
}

impl GgufInfo {
    pub fn read(path: &Path) -> Result<Self, EstimateError> {
        let file = File::open(path)?;
        let file_size = file.metadata()?.len();
        // Safety: we only read from the mmap for the duration of this call;
        // the file is not mutated concurrently in our workflow.
        let mmap = unsafe { Mmap::map(&file) }?;

        let kvs = read_metadata_kvs(&mmap)?;
        let map = MetaMap(kvs);

        let max_context = map.llm_context_length().map_err(meta_err)? as u32;
        let n_layers = map.llm_block_count().map_err(meta_err)? as u32;
        let n_embd = map.llm_embedding_length().map_err(meta_err)? as u32;
        // Some MoE / hybrid architectures (e.g. Step-3.5, which has 64
        // heads on every 5th layer and 96 on the rest) store a per-layer
        // u32 Array under `*.attention.head_count` instead of a single
        // scalar. We don't use n_head for anything precise — just for the
        // n_embd_head fallback in the KV cache formula — so on Array we
        // yield 0, which triggers the n_embd_head = 128 default path in
        // `VramEstimate::compute` and in the form's JS preview.
        let n_head = tolerate_array(map.llm_attention_head_count())? as u32;
        let n_head_kv = tolerate_array(map.llm_attention_head_count_kv())? as u32;

        Ok(Self { max_context, n_layers, n_embd, n_head, n_head_kv, file_size })
    }
}

/// Returns the scalar value if the field parses as one; returns 0 if the
/// field is actually an array (signalling "unknown, use default"); and
/// propagates any other error unchanged.
fn tolerate_array(res: Result<usize, GGufMetaError>) -> Result<usize, EstimateError> {
    match res {
        Ok(v) => Ok(v),
        Err(GGufMetaError::TypeMismatch(Ty::Array)) => Ok(0),
        Err(e) => Err(meta_err(e)),
    }
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
    const N_EMBD: u64 = 4096;
    const N_HEAD: u64 = 32;

    #[test]
    fn kv_cache_scales_linearly_with_context() {
        let a = VramEstimate::compute(FILE_SIZE, 4096, N_LAYERS, N_EMBD, N_HEAD, N_HEAD);
        let b = VramEstimate::compute(FILE_SIZE, 16384, N_LAYERS, N_EMBD, N_HEAD, N_HEAD);
        let ratio = b.kv_cache_vram as f64 / a.kv_cache_vram as f64;
        assert!((3.9..=4.1).contains(&ratio), "ratio={ratio}");
    }

    #[test]
    fn gqa_shrinks_kv_cache_proportionally() {
        let mha = VramEstimate::compute(FILE_SIZE, 4096, N_LAYERS, N_EMBD, N_HEAD, 32);
        let gqa = VramEstimate::compute(FILE_SIZE, 4096, N_LAYERS, N_EMBD, N_HEAD, 4);
        let ratio = mha.kv_cache_vram as f64 / gqa.kv_cache_vram as f64;
        assert!((7.9..=8.1).contains(&ratio), "ratio={ratio}");
    }

    #[test]
    fn zero_context_yields_no_kv_cache() {
        let e = VramEstimate::compute(FILE_SIZE, 0, N_LAYERS, N_EMBD, N_HEAD, N_HEAD);
        assert_eq!(e.kv_cache_vram, 0);
        assert_eq!(e.total_vram, (FILE_SIZE * 11) / 10);
    }

    #[test]
    fn overhead_is_ten_percent() {
        let e = VramEstimate::compute(FILE_SIZE, 4096, N_LAYERS, N_EMBD, N_HEAD, N_HEAD);
        let base = FILE_SIZE + e.kv_cache_vram;
        assert_eq!(e.total_vram, base * 11 / 10);
    }

    #[test]
    fn from_gguf_fails_on_missing_file() {
        let err = VramEstimate::from_gguf(Path::new("/nonexistent.gguf"), 4096).unwrap_err();
        assert!(matches!(err, EstimateError::Io(_)));
    }

    #[test]
    fn from_gguf_rejects_non_gguf_file() {
        let tmp = tempfile::NamedTempFile::new().unwrap();
        std::fs::write(tmp.path(), b"not a gguf file at all, just some bytes").unwrap();
        let err = VramEstimate::from_gguf(tmp.path(), 4096).unwrap_err();
        assert!(matches!(err, EstimateError::Gguf(_)));
    }

    #[test]
    fn tolerate_array_passes_scalars_through() {
        assert_eq!(tolerate_array(Ok(42)).unwrap(), 42);
    }

    #[test]
    fn tolerate_array_collapses_array_to_zero() {
        // Models with per-layer head counts (Step-3.5 is the motivating
        // case) store `*.attention.head_count` as a u32 Array. Treat that
        // as "unknown" by returning 0 — compute()'s n_embd_head fallback
        // kicks in, and the form's JS preview already defaults to 128.
        let err = Err(GGufMetaError::TypeMismatch(Ty::Array));
        assert_eq!(tolerate_array(err).unwrap(), 0);
    }

    #[test]
    fn tolerate_array_propagates_other_type_mismatches() {
        // A String where we expected a number is a real problem.
        let err = Err(GGufMetaError::TypeMismatch(Ty::String));
        assert!(matches!(tolerate_array(err), Err(EstimateError::Gguf(_))));
    }

    #[test]
    fn tolerate_array_propagates_missing_key() {
        assert!(matches!(
            tolerate_array(Err(GGufMetaError::NotExist)),
            Err(EstimateError::Gguf(_)),
        ));
    }
}
