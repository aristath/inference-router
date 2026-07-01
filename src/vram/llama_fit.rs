use crate::config::ModelConfig;
use std::path::{Path, PathBuf};
use std::process::Command;

const MIB: u64 = 1024 * 1024;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LlamaFittedArgs {
    pub context: Option<u32>,
    pub n_gpu_layers: Option<i32>,
    pub tensor_split: Option<String>,
    pub override_tensor: Option<String>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LlamaFitSizing {
    pub fitted: LlamaFittedArgs,
    pub device_vram: u64,
}

#[derive(Debug, thiserror::Error)]
pub enum LlamaFitError {
    #[error("llama-fit-params binary not found next to {0}")]
    MissingTool(PathBuf),

    #[error("failed to run {binary}: {source}")]
    Io {
        binary: PathBuf,
        source: std::io::Error,
    },

    #[error("{binary} exited with {status}: {stderr}")]
    Failed {
        binary: PathBuf,
        status: String,
        stderr: String,
    },

    #[error("could not parse llama-fit-params output: {0}")]
    Parse(String),
}

pub fn fit_binary_for_server(server: &Path) -> PathBuf {
    let mut path = server.to_path_buf();
    path.set_file_name("llama-fit-params");
    path
}

pub fn run_llama_fit_sizing(
    fit_binary: &Path,
    model: &ModelConfig,
    _draft: Option<&ModelConfig>,
    device: &str,
    fit_target: &str,
) -> Result<LlamaFitSizing, LlamaFitError> {
    if !fit_binary.is_file() {
        return Err(LlamaFitError::MissingTool(fit_binary.to_path_buf()));
    }

    let mut fit_args = base_args(model, device);
    fit_args.push("--fit-target".into());
    fit_args.push(fit_target.into());

    let fitted_stdout = run_fit_tool(fit_binary, &fit_args)?;
    let fitted = parse_fitted_args(&fitted_stdout)?;

    let mut print_args = base_args(model, device);
    apply_fitted_args(&mut print_args, &fitted);
    print_args.push("--fit-print".into());
    print_args.push("on".into());

    let memory_stdout = run_fit_tool(fit_binary, &print_args)?;
    let device_vram = parse_fit_print_device_vram(&memory_stdout)?;

    Ok(LlamaFitSizing {
        fitted,
        device_vram,
    })
}

pub fn needs_server_owned_fit(model: &ModelConfig, draft: Option<&ModelConfig>) -> bool {
    model.mmproj_path.is_some()
        || draft.is_some()
        || (draft.is_none()
            && model.draft_model_id.is_none()
            && model.mtp_tokens.filter(|n| *n > 0).is_some())
}

fn base_args(model: &ModelConfig, device: &str) -> Vec<String> {
    let mut args = vec![
        "-m".into(),
        model.model_path.to_string_lossy().into_owned(),
        "-c".into(),
        model.context.to_string(),
        "--device".into(),
        device.into(),
        "--flash-attn".into(),
        if model.flash_attn { "on" } else { "off" }.into(),
    ];

    if let Some(k) = model.cache_type_k {
        args.push("--cache-type-k".into());
        args.push(k.as_arg().into());
    }
    if let Some(v) = model.cache_type_v {
        args.push("--cache-type-v".into());
        args.push(v.as_arg().into());
    }
    if let Some(n) = model.parallel_slots {
        args.push("--parallel".into());
        args.push(n.to_string());
    }

    args
}

fn apply_fitted_args(args: &mut Vec<String>, fitted: &LlamaFittedArgs) {
    if let Some(context) = fitted.context {
        replace_or_push(args, "-c", context.to_string());
    }
    if let Some(ngl) = fitted.n_gpu_layers {
        args.push("-ngl".into());
        args.push(ngl.to_string());
    }
    if let Some(ref ts) = fitted.tensor_split {
        args.push("-ts".into());
        args.push(ts.clone());
    }
    if let Some(ref ot) = fitted.override_tensor {
        args.push("-ot".into());
        args.push(ot.clone());
    }
}

fn replace_or_push(args: &mut Vec<String>, flag: &str, value: String) {
    let mut idx = 0;
    while idx + 1 < args.len() {
        if args[idx] == flag {
            args[idx + 1] = value;
            return;
        }
        idx += 1;
    }
    args.push(flag.into());
    args.push(value);
}

fn run_fit_tool(fit_binary: &Path, args: &[String]) -> Result<String, LlamaFitError> {
    let output = Command::new(fit_binary)
        .args(args)
        .output()
        .map_err(|source| LlamaFitError::Io {
            binary: fit_binary.to_path_buf(),
            source,
        })?;
    if !output.status.success() {
        return Err(LlamaFitError::Failed {
            binary: fit_binary.to_path_buf(),
            status: output.status.to_string(),
            stderr: String::from_utf8_lossy(&output.stderr).trim().to_string(),
        });
    }
    Ok(String::from_utf8_lossy(&output.stdout).into_owned())
}

fn parse_fitted_args(stdout: &str) -> Result<LlamaFittedArgs, LlamaFitError> {
    let line = stdout
        .lines()
        .rev()
        .map(str::trim)
        .find(|line| line.starts_with("-c ") || line.contains(" -ngl "))
        .ok_or_else(|| LlamaFitError::Parse("missing fitted-args line".into()))?;
    let words = split_shell_words(line)?;
    let mut fitted = LlamaFittedArgs {
        context: None,
        n_gpu_layers: None,
        tensor_split: None,
        override_tensor: None,
    };
    let mut idx = 0;
    while idx < words.len() {
        match words[idx].as_str() {
            "-c" | "--ctx-size" | "--ctx-size=" => {
                if let Some(value) = words.get(idx + 1) {
                    fitted.context = value.parse().ok();
                }
                idx += 2;
            }
            "-ngl" | "--n-gpu-layers" => {
                if let Some(value) = words.get(idx + 1) {
                    fitted.n_gpu_layers = value.parse().ok();
                }
                idx += 2;
            }
            "-ts" | "--tensor-split" => {
                if let Some(value) = words.get(idx + 1) {
                    fitted.tensor_split = Some(value.clone());
                }
                idx += 2;
            }
            "-ot" | "--override-tensor" => {
                if let Some(value) = words.get(idx + 1) {
                    fitted.override_tensor = Some(value.clone());
                }
                idx += 2;
            }
            _ => idx += 1,
        }
    }
    Ok(fitted)
}

fn parse_fit_print_device_vram(stdout: &str) -> Result<u64, LlamaFitError> {
    let mut total_mib = 0u64;
    for line in stdout.lines() {
        let mut parts = line.split_whitespace();
        let Some(device) = parts.next() else { continue };
        if device == "Host" {
            continue;
        }
        let nums: Vec<u64> = parts.take(3).filter_map(|p| p.parse().ok()).collect();
        if nums.len() == 3 {
            total_mib = total_mib.saturating_add(nums.into_iter().sum::<u64>());
        }
    }
    if total_mib == 0 {
        return Err(LlamaFitError::Parse("missing device memory rows".into()));
    }
    Ok(total_mib.saturating_mul(MIB))
}

fn split_shell_words(line: &str) -> Result<Vec<String>, LlamaFitError> {
    let mut words = Vec::new();
    let mut cur = String::new();
    let mut quote: Option<char> = None;
    let mut chars = line.chars().peekable();

    while let Some(ch) = chars.next() {
        if ch == '\\' {
            match quote {
                Some('\'') => cur.push(ch),
                Some('"') => {
                    if matches!(chars.peek(), Some('"') | Some('\\')) {
                        cur.push(chars.next().unwrap());
                    } else {
                        cur.push(ch);
                    }
                }
                Some(_) => cur.push(ch),
                None => {
                    if matches!(chars.peek(), Some(c) if c.is_whitespace() || *c == '"' || *c == '\'' || *c == '\\')
                    {
                        cur.push(chars.next().unwrap());
                    } else {
                        cur.push(ch);
                    }
                }
            }
            continue;
        }
        match quote {
            Some(q) if ch == q => quote = None,
            Some(_) => cur.push(ch),
            None if ch == '"' || ch == '\'' => quote = Some(ch),
            None if ch.is_whitespace() => {
                if !cur.is_empty() {
                    words.push(std::mem::take(&mut cur));
                }
            }
            None => cur.push(ch),
        }
    }
    if quote.is_some() {
        return Err(LlamaFitError::Parse("unterminated quote".into()));
    }
    if !cur.is_empty() {
        words.push(cur);
    }
    Ok(words)
}

pub fn apply_sizing_to_model(model: &mut ModelConfig, sizing: &LlamaFitSizing) {
    if let Some(context) = sizing.fitted.context {
        model.context = context;
    }
    model.n_gpu_layers = sizing
        .fitted
        .n_gpu_layers
        .map(|ngl| if ngl < 0 { 999 } else { ngl as u32 });
    model.tensor_split = sizing.fitted.tensor_split.clone();
    model.override_tensor = sizing.fitted.override_tensor.clone();
    model.n_cpu_moe = None;
    model.fit_target = None;
    model.split_mode = None;
    model.estimated_vram = sizing.device_vram;
}

#[cfg(test)]
#[path = "llama_fit_tests.rs"]
mod tests;
