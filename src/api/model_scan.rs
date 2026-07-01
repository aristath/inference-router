use axum::extract::State;
use axum::http::StatusCode;
use axum::response::IntoResponse;
use axum::Json;
use serde::{Deserialize, Serialize};
use std::collections::HashSet;
use std::path::{Path, PathBuf};

use crate::config::{ModelConfig, ModelState, WeightsFormat};
use crate::orchestrator::AppState;
use crate::vram::estimator::GgufMeta;

#[derive(Debug, Serialize)]
pub struct ModelScanResponse {
    pub folder: String,
    pub candidates: Vec<ModelScanCandidate>,
    pub missing: Vec<MissingModel>,
    pub errors: Vec<String>,
}

#[derive(Debug, Serialize)]
pub struct ModelScanCandidate {
    pub path: String,
    pub model: ModelConfig,
}

#[derive(Debug, Serialize)]
pub struct MissingModel {
    pub id: String,
    pub name: String,
    pub path: String,
}

#[derive(Debug, Deserialize)]
pub struct ModelReconcileRequest {
    #[serde(default)]
    pub add: Vec<ModelConfig>,
    #[serde(default)]
    pub remove: Vec<String>,
}

#[derive(Debug, Serialize)]
pub struct ModelReconcileResponse {
    pub added: Vec<String>,
    pub removed: Vec<String>,
    pub errors: Vec<String>,
}

pub async fn scan_models_folder(State(state): State<AppState>) -> impl IntoResponse {
    let settings = state.settings().await;
    let folder = settings.models_folder;
    let expanded = expand_tilde(&folder);
    let root = PathBuf::from(&expanded);

    if !root.is_dir() {
        return (
            StatusCode::BAD_REQUEST,
            Json(serde_json::json!({
                "error": format!("models folder '{}' is not a directory", folder)
            })),
        )
            .into_response();
    }

    let models = state.list_models().await;
    let mut known_paths = HashSet::new();
    for model in &models {
        if model.weights_format == WeightsFormat::Gguf {
            known_paths.insert(canonical_key(&model.model_path));
        }
    }

    let mut errors = Vec::new();
    let files = find_gguf_files(&root, &mut errors);
    let (model_files, mmproj_files): (Vec<_>, Vec<_>) =
        files.into_iter().partition(|path| !is_mmproj_file(path));
    let mut candidates = Vec::new();
    let mut used_ids: HashSet<String> = models.iter().map(|m| m.id.clone()).collect();

    for path in model_files {
        if known_paths.contains(&canonical_key(&path)) {
            continue;
        }
        match GgufMeta::read(&path) {
            Ok(meta) => {
                let model = candidate_model(&path, meta, &models, &mut used_ids, &mmproj_files);
                candidates.push(ModelScanCandidate {
                    path: path.to_string_lossy().into_owned(),
                    model,
                });
            }
            Err(e) => errors.push(format!("{}: {e}", path.display())),
        }
    }

    let missing = models
        .iter()
        .filter(|model| model.weights_format == WeightsFormat::Gguf)
        .filter(|model| !model.model_path.exists())
        .filter(|model| path_appears_under_root(&model.model_path, &root))
        .map(|model| MissingModel {
            id: model.id.clone(),
            name: model.name.clone(),
            path: model.model_path.to_string_lossy().into_owned(),
        })
        .collect();

    (
        StatusCode::OK,
        Json(ModelScanResponse {
            folder,
            candidates,
            missing,
            errors,
        }),
    )
        .into_response()
}

pub async fn reconcile_models_folder(
    State(state): State<AppState>,
    Json(req): Json<ModelReconcileRequest>,
) -> impl IntoResponse {
    let mut added = Vec::new();
    let mut removed = Vec::new();
    let mut errors = Vec::new();

    for model in req.add {
        let id = model.id.clone();
        match state.add_model(model).await {
            Ok(()) => added.push(id),
            Err(e) => errors.push(format!("add {id}: {e}")),
        }
    }

    for id in req.remove {
        match state.remove_model(&id).await {
            Ok(()) => removed.push(id),
            Err(e) => errors.push(format!("remove {id}: {e}")),
        }
    }

    let status = if errors.is_empty() {
        StatusCode::OK
    } else {
        StatusCode::MULTI_STATUS
    };
    (
        status,
        Json(ModelReconcileResponse {
            added,
            removed,
            errors,
        }),
    )
        .into_response()
}

fn candidate_model(
    path: &Path,
    meta: GgufMeta,
    existing: &[ModelConfig],
    used_ids: &mut HashSet<String>,
    mmproj_files: &[PathBuf],
) -> ModelConfig {
    let mut model = nearest_template(path, existing)
        .cloned()
        .unwrap_or_default();

    let suggested_id = unique_id(&meta.suggested_id, used_ids);
    model.id = suggested_id;
    model.name = meta.suggested_name.clone();
    model.weights_format = WeightsFormat::Gguf;
    model.model_path = path.to_path_buf();
    model.context = meta.max_context;
    model.state = ModelState::Idle;
    model.pid = None;
    model.estimated_vram = 0;
    model.last_used = None;
    model.gguf_meta = Some(meta);
    model.mmproj_path = matching_mmproj(path, mmproj_files);
    if model.draft_model_id.as_deref() == Some(model.id.as_str())
        || model
            .draft_model_id
            .as_deref()
            .is_some_and(|draft_id| !existing_draft_is_present(draft_id, existing))
    {
        model.draft_model_id = None;
    }
    model
}

fn existing_draft_is_present(id: &str, existing: &[ModelConfig]) -> bool {
    existing
        .iter()
        .any(|model| model.id == id && model.model_path.exists())
}

/// Find the closest existing GGUF model to use as a config template for a newly
/// discovered file, so it inherits a runnable setup (binary preset, ngl, …).
///
/// A model is a candidate when it lives in the same directory as `path`, a
/// sibling directory (shared parent), an ancestor, or a nested child — i.e. the
/// two parents agree on every path component except possibly the last. This
/// matters because quant variants are commonly laid out as sibling folders
/// (`…/9b/QWOPUS-Q8_K_XL` next to `…/9b/UD-Q4_K_XL`); requiring a strict prefix
/// relationship would miss them and seed an unrunnable `default()` config.
fn nearest_template<'a>(path: &Path, existing: &'a [ModelConfig]) -> Option<&'a ModelConfig> {
    let parent = path.parent()?;
    let parent_depth = parent.components().count();
    existing
        .iter()
        .filter(|model| model.weights_format == WeightsFormat::Gguf)
        .filter_map(|model| {
            let model_parent = model.model_path.parent()?;
            let shared = shared_prefix_len(parent, model_parent);
            // Agree on all-but-the-last component of `parent` (sibling), or
            // more (same dir / ancestor / nested child). Cousins in a different
            // subtree are excluded so we never seed from an unrelated model.
            (shared + 1 >= parent_depth).then_some((
                shared,
                model_parent.components().count(),
                model,
            ))
        })
        .max_by_key(|(shared, depth, _)| (*shared, *depth))
        .map(|(_, _, model)| model)
}

fn shared_prefix_len(a: &Path, b: &Path) -> usize {
    a.components()
        .zip(b.components())
        .take_while(|(x, y)| x == y)
        .count()
}

fn unique_id(base: &str, used_ids: &mut HashSet<String>) -> String {
    let stem = if base.trim().is_empty() {
        "model"
    } else {
        base
    };
    let mut candidate = stem.to_string();
    let mut suffix = 2;
    while used_ids.contains(&candidate) {
        candidate = format!("{stem}-{suffix}");
        suffix += 1;
    }
    used_ids.insert(candidate.clone());
    candidate
}

fn find_gguf_files(root: &Path, errors: &mut Vec<String>) -> Vec<PathBuf> {
    let mut out = Vec::new();
    let mut stack = vec![root.to_path_buf()];
    let mut visited = HashSet::new();

    while let Some(dir) = stack.pop() {
        let canonical = std::fs::canonicalize(&dir).unwrap_or_else(|_| dir.clone());
        if !visited.insert(canonical) {
            continue;
        }

        let read_dir = match std::fs::read_dir(&dir) {
            Ok(read_dir) => read_dir,
            Err(e) => {
                errors.push(format!("{}: {e}", dir.display()));
                continue;
            }
        };

        for entry in read_dir.flatten() {
            let path = entry.path();
            if path.is_dir() {
                stack.push(path);
            } else if is_gguf_file(&path) && is_runnable_gguf_entrypoint(&path) {
                out.push(path);
            }
        }
    }

    out.sort();
    out
}

fn is_gguf_file(path: &Path) -> bool {
    path.extension()
        .and_then(|ext| ext.to_str())
        .map(|ext| ext.eq_ignore_ascii_case("gguf"))
        .unwrap_or(false)
}

fn is_runnable_gguf_entrypoint(path: &Path) -> bool {
    match sharded_gguf_index(path) {
        Some((idx, _total)) => idx == 1,
        None => true,
    }
}

fn sharded_gguf_index(path: &Path) -> Option<(u32, u32)> {
    let fname = path.file_name()?.to_str()?;
    let rest = fname.strip_suffix(".gguf")?;
    let (stem_and_idx, total_str) = rest.rsplit_once("-of-")?;
    let total = total_str.parse().ok()?;
    let (_stem, idx_str) = stem_and_idx.rsplit_once('-')?;
    let idx = idx_str.parse().ok()?;
    Some((idx, total))
}

fn is_mmproj_file(path: &Path) -> bool {
    path.file_name()
        .and_then(|name| name.to_str())
        .map(|name| {
            let lower = name.to_ascii_lowercase();
            lower.contains("mmproj") || lower.contains("projector")
        })
        .unwrap_or(false)
}

fn matching_mmproj(model_path: &Path, mmproj_files: &[PathBuf]) -> Option<PathBuf> {
    let parent = model_path.parent()?;
    let mut same_dir: Vec<PathBuf> = mmproj_files
        .iter()
        .filter(|path| path.parent() == Some(parent))
        .cloned()
        .collect();
    same_dir.sort();
    if same_dir.len() == 1 {
        same_dir.pop()
    } else {
        None
    }
}

fn canonical_key(path: &Path) -> PathBuf {
    std::fs::canonicalize(path).unwrap_or_else(|_| path.to_path_buf())
}

/// Whether `path` lives inside the scan `root`. Purely lexical: it must keep
/// working when the file — and its parent directory — have been deleted, which
/// is exactly the "missing model" case. We compare against both the literal root
/// and its canonical form, because the configured folder is commonly a symlink
/// (`~/models -> /mnt/models/models`) while model paths are stored as the real
/// target — so neither representation alone catches every model.
fn path_appears_under_root(path: &Path, root: &Path) -> bool {
    let expanded_path = PathBuf::from(expand_tilde(&path.to_string_lossy()));
    let root_key = canonical_key(root);
    expanded_path.starts_with(root) || expanded_path.starts_with(&root_key)
}

fn expand_tilde(path: &str) -> String {
    shellexpand::tilde(path).to_string()
}

#[cfg(test)]
#[path = "model_scan_tests.rs"]
mod tests;
