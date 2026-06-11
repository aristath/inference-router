//! diffusion-adapter — an OpenAI-compatible front for `llama-diffusion-cli`.
//!
//! The inference-router serves models by spawning a backend that listens on a
//! `--port` and speaks `/health` + `/v1/chat/completions`, then byte-proxies to
//! it. `llama-diffusion-cli` (the only PR #24423 binary that generates text
//! end-to-end) has no HTTP surface — it's an interactive terminal program. This
//! binary bridges the gap *entirely in the router project*, with no llama.cpp
//! changes: the router spawns THIS adapter as the model's `binary`; the adapter
//! drives a resident `llama-diffusion-cli -cnv` child over stdin/stdout and
//! synthesizes the OpenAI surface the router expects.
//!
//! Driving protocol (verified empirically against the binary):
//!   - On model load the CLI prints `> ` on stdout and waits for input.
//!   - We `/clear` (resets conversation → stateless per request), then write the
//!     prompt as ONE line (the CLI reads a line per turn via fgets, so the prompt
//!     is whitespace-collapsed to a single line).
//!   - `--diffusion-visual --diffusion-visual-interval 1` makes the CLI redraw
//!     the full canvas every denoising step as ANSI terminal frames (free — the
//!     readback rides the device-resident fast path). We split the stream on the
//!     synchronized-output end marker `ESC[?2026l`, strip ANSI, and each chunk is
//!     one full-canvas snapshot — no full VT emulator needed (COLUMNS is pinned
//!     huge to kill line-wrap).
//!   - The `total time:` line marks end-of-generation; the clean text just before
//!     it is the final answer.
//!
//! Output is a reasoning model with Harmony-style channels
//! `<|channel>thought … <channel|>answer`, split into `reasoning_content` vs
//! `content`. Streaming sends live `event: diffusion.canvas` frames (for a
//! watch-it-denoise UI) plus standard `chat.completion.chunk` text deltas.

use std::process::Stdio;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::Arc;

use axum::body::{Body, Bytes};
use axum::extract::State;
use axum::http::StatusCode;
use axum::response::{IntoResponse, Response};
use axum::routing::{get, post};
use axum::{Json, Router};
use serde_json::{json, Value};
use tokio::io::{AsyncReadExt, AsyncWriteExt, BufReader};
use tokio::process::{ChildStdin, ChildStdout, Command};
use tokio::sync::Mutex;

const DEFAULT_CLI: &str = "/home/aristath/llama.cpp-diffusion/build/bin/llama-diffusion-cli";
const FRAME_END: &str = "\x1b[?2026l"; // synchronized-output END = one full canvas frame complete
const GEN_END: &str = "total time:"; // CLI prints this once generation finishes
const PROMPT: &str = "> "; // CLI's idle input prompt

// ---------------------------------------------------------------------------
// Child process driver (serialized: one resident context = one request at a time)
// ---------------------------------------------------------------------------

struct Driver {
    stdin: ChildStdin,
    reader: BufReader<ChildStdout>,
    /// Bytes read from the child but not yet consumed by a `read_until`.
    pending: String,
}

impl Driver {
    /// Read one chunk from the child into `pending`. Returns false on EOF.
    async fn fill(&mut self) -> std::io::Result<bool> {
        let mut tmp = [0u8; 8192];
        let n = self.reader.read(&mut tmp).await?;
        if n == 0 {
            return Ok(false);
        }
        self.pending.push_str(&String::from_utf8_lossy(&tmp[..n]));
        Ok(true)
    }

    /// Consume bytes up to and including `needle`; return the consumed prefix.
    /// On EOF before `needle`, returns whatever remained.
    async fn read_until(&mut self, needle: &str) -> std::io::Result<String> {
        loop {
            if let Some(p) = self.pending.find(needle) {
                let end = p + needle.len();
                let consumed = self.pending[..end].to_string();
                self.pending.replace_range(..end, "");
                return Ok(consumed);
            }
            if !self.fill().await? {
                return Ok(std::mem::take(&mut self.pending));
            }
        }
    }

    async fn write_line(&mut self, line: &str) -> std::io::Result<()> {
        self.stdin.write_all(line.as_bytes()).await?;
        self.stdin.write_all(b"\n").await?;
        self.stdin.flush().await
    }

    /// Wait for the initial `> ` the CLI prints once the model is loaded.
    async fn wait_ready(&mut self) -> std::io::Result<()> {
        self.read_until(PROMPT).await?;
        Ok(())
    }

    /// Run one stateless generation. `on_frame(step, snapshot)` is called live as
    /// each full-canvas frame completes. Returns the final answer text (raw,
    /// channel markers intact).
    async fn generate(
        &mut self,
        prompt: &str,
        on_frame: &mut (dyn FnMut(usize, &str) + Send),
    ) -> std::io::Result<String> {
        // Reset conversation state so each OpenAI request is independent.
        self.write_line("/clear").await?;
        self.read_until(PROMPT).await?;

        // One line per turn (fgets); whitespace already collapsed by caller.
        self.write_line(prompt).await?;

        let mut gen = String::new();
        let mut frame_cursor = 0usize;
        let mut step = 0usize;
        loop {
            // Stop at the end-of-generation marker, keeping anything after it
            // (the trailing `> `) in `pending` for the invariant restore below.
            if let Some(tp) = self.pending.find(GEN_END) {
                let line_end = self.pending[tp..]
                    .find('\n')
                    .map(|x| tp + x + 1)
                    .unwrap_or(self.pending.len());
                gen.push_str(&self.pending[..line_end]);
                self.pending.replace_range(..line_end, "");
                extract_frames(&gen, &mut frame_cursor, &mut step, on_frame);
                break;
            }
            gen.push_str(&self.pending);
            self.pending.clear();
            extract_frames(&gen, &mut frame_cursor, &mut step, on_frame);
            if !self.fill().await? {
                break; // child exited mid-generation
            }
        }

        // Restore the "child is at a `> ` prompt" invariant for the next request.
        self.read_until(PROMPT).await?;
        Ok(extract_final(&gen))
    }
}

/// Emit any newly-completed full-canvas frames found past `cursor`.
fn extract_frames(
    gen: &str,
    cursor: &mut usize,
    step: &mut usize,
    on_frame: &mut (dyn FnMut(usize, &str) + Send),
) {
    while let Some(rel) = gen[*cursor..].find(FRAME_END) {
        let end = *cursor + rel;
        let snap = strip_ansi(&gen[*cursor..end]);
        let trimmed = snap.trim();
        if !trimmed.is_empty() {
            *step += 1;
            on_frame(*step, trimmed);
        }
        *cursor = end + FRAME_END.len();
    }
}

/// The final answer = clean text after the last frame, before `total time:`.
fn extract_final(gen: &str) -> String {
    let cut = gen.find(GEN_END).unwrap_or(gen.len());
    let pre = &gen[..cut];
    let tail = pre
        .rfind(FRAME_END)
        .map(|p| &pre[p + FRAME_END.len()..])
        .unwrap_or(pre);
    let cleaned = strip_ansi(tail);
    let cleaned = cleaned.trim();
    if !cleaned.is_empty() {
        return cleaned.to_string();
    }
    strip_ansi(pre).trim().to_string()
}

// ---------------------------------------------------------------------------
// ANSI + channel parsing
// ---------------------------------------------------------------------------

/// Strip ANSI/CSI/OSC escapes and `\r`, preserving UTF-8 multibyte content.
fn strip_ansi(s: &str) -> String {
    let b = s.as_bytes();
    let mut out: Vec<u8> = Vec::with_capacity(b.len());
    let mut i = 0;
    while i < b.len() {
        match b[i] {
            0x1b => {
                i += 1;
                if i < b.len() && b[i] == b'[' {
                    i += 1; // CSI: params 0x30-0x3f, intermediates 0x20-0x2f, final 0x40-0x7e
                    while i < b.len() && (0x20..=0x3f).contains(&b[i]) {
                        i += 1;
                    }
                    if i < b.len() {
                        i += 1; // final byte
                    }
                } else if i < b.len() && b[i] == b']' {
                    while i < b.len() && b[i] != 0x07 {
                        i += 1; // OSC until BEL
                    }
                    if i < b.len() {
                        i += 1;
                    }
                }
            }
            b'\r' => i += 1,
            x => {
                out.push(x);
                i += 1;
            }
        }
    }
    String::from_utf8_lossy(&out).into_owned()
}

/// Split Harmony channels: `<|channel>thought … <channel|>answer`.
/// Returns `(reasoning, content)`.
fn split_channels(text: &str) -> (Option<String>, String) {
    if let Some(pos) = text.rfind("<channel|>") {
        let answer = text[pos + "<channel|>".len()..].trim().to_string();
        let reasoning = clean_reasoning(&text[..pos]);
        let reasoning = (!reasoning.is_empty()).then_some(reasoning);
        (reasoning, answer)
    } else {
        (None, clean_reasoning(text))
    }
}

/// Strip the `<|channel>` open marker and a leading `thought`/`final` label.
fn clean_reasoning(s: &str) -> String {
    let s = s.replace("<|channel>", "");
    let s = s.trim();
    let s = s
        .strip_prefix("thought")
        .or_else(|| s.strip_prefix("final"))
        .unwrap_or(s);
    s.trim().to_string()
}

/// Split text into small streamed pieces (keeps trailing spaces on words) so a
/// completed answer is revealed token-by-token rather than as one blob.
fn chunk_text(s: &str) -> Vec<String> {
    if s.is_empty() {
        return Vec::new();
    }
    let mut out = Vec::new();
    let mut cur = String::new();
    for ch in s.chars() {
        cur.push(ch);
        if ch == ' ' {
            out.push(std::mem::take(&mut cur));
        }
    }
    if !cur.is_empty() {
        out.push(cur);
    }
    out
}

// ---------------------------------------------------------------------------
// HTTP surface
// ---------------------------------------------------------------------------

struct AppState {
    driver: Mutex<Driver>,
    /// The CLI child, so we can kill it before exiting on an unrecoverable hang.
    child: Mutex<tokio::process::Child>,
    ready: AtomicBool,
    model_name: String,
    req_counter: AtomicU64,
}

/// Generation budget. A healthy run is seconds; this only trips on a genuine
/// hang (silent/crashed child). On trip we kill the child and exit so the
/// router's dead-instance detector respawns us fresh — a stuck resident process
/// must never wedge the model permanently.
const GEN_TIMEOUT: std::time::Duration = std::time::Duration::from_secs(120);

/// Kill the CLI child and exit so the router respawns a clean instance.
async fn fail_and_exit(st: &AppState, ctx: &str) -> ! {
    eprintln!("diffusion-adapter: {ctx}; killing child and exiting for router respawn");
    {
        let mut c = st.child.lock().await;
        let _ = c.start_kill();
    }
    // Give the kill signal a moment to land before the process image vanishes.
    tokio::time::sleep(std::time::Duration::from_millis(100)).await;
    std::process::exit(1);
}

async fn health(State(st): State<Arc<AppState>>) -> Response {
    if st.ready.load(Ordering::Relaxed) {
        (StatusCode::OK, "ok").into_response()
    } else {
        (StatusCode::SERVICE_UNAVAILABLE, "loading").into_response()
    }
}

async fn list_models(State(st): State<Arc<AppState>>) -> Response {
    Json(json!({
        "object": "list",
        "data": [{"id": st.model_name, "object": "model", "created": 0, "owned_by": "local"}],
    }))
    .into_response()
}

async fn chat_completions(State(st): State<Arc<AppState>>, body: Bytes) -> Response {
    let req: Value = match serde_json::from_slice(&body) {
        Ok(v) => v,
        Err(e) => {
            return (
                StatusCode::BAD_REQUEST,
                Json(json!({"error": format!("invalid JSON body: {e}")})),
            )
                .into_response()
        }
    };

    let prompt = build_prompt(&req);
    if prompt.is_empty() {
        return (
            StatusCode::BAD_REQUEST,
            Json(json!({"error": "no usable 'messages' content in request"})),
        )
            .into_response();
    }
    let stream = req.get("stream").and_then(|s| s.as_bool()).unwrap_or(false);
    let id = format!(
        "chatcmpl-diff-{}",
        st.req_counter.fetch_add(1, Ordering::Relaxed)
    );

    if stream {
        stream_response(st, prompt, id)
    } else {
        nonstream_response(st, prompt, id).await
    }
}

async fn nonstream_response(st: Arc<AppState>, prompt: String, id: String) -> Response {
    let mut drv = st.driver.lock().await;
    let mut noop = |_: usize, _: &str| {};
    let final_text = match tokio::time::timeout(GEN_TIMEOUT, drv.generate(&prompt, &mut noop)).await {
        Ok(Ok(t)) => t,
        Ok(Err(e)) => {
            return (
                StatusCode::INTERNAL_SERVER_ERROR,
                Json(json!({"error": format!("generation failed: {e}")})),
            )
                .into_response()
        }
        Err(_) => {
            drop(drv);
            fail_and_exit(&st, "non-stream generation timed out").await
        }
    };
    drop(drv);

    let (reasoning, content) = split_channels(&final_text);
    let mut message = json!({"role": "assistant", "content": content});
    if let Some(r) = reasoning {
        message["reasoning_content"] = json!(r);
    }
    Json(json!({
        "id": id,
        "object": "chat.completion",
        "created": 0,
        "model": st.model_name,
        "choices": [{
            "index": 0,
            "message": message,
            "finish_reason": "stop",
        }],
        "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
    }))
    .into_response()
}

fn stream_response(st: Arc<AppState>, prompt: String, id: String) -> Response {
    let (tx, rx) = tokio::sync::mpsc::unbounded_channel::<Result<String, std::io::Error>>();
    let model = st.model_name.clone();

    tokio::spawn(async move {
        // Opening chunk (role).
        let _ = tx.send(Ok(sse_data(&chunk(&id, &model, json!({"role": "assistant"}), None))));

        let mut drv = st.driver.lock().await;
        // For each live denoise frame: (1) emit the full canvas as an
        // out-of-band `diffusion.canvas` event (watch-it-denoise), and (2) feed
        // it to the StreamEmitter to push append-only `content`/`reasoning`
        // deltas as the canvas stabilizes — real incremental streaming for
        // standard OpenAI clients, not a burst at the end.
        // Live denoise view: stream each canvas frame as an out-of-band
        // `diffusion.canvas` event. We deliberately do NOT derive OAI text deltas
        // from these frames. The entropy-bound sampler RENOISES unaccepted
        // positions every step, so the scraped canvas has no stable growing
        // prefix until it converges — committed-prefix streaming off it is
        // empirically INCORRECT (it streams the mutating scratchpad and the
        // accumulated text never matches the final answer). The per-position
        // commit signal that would make incremental streaming correct lives
        // inside libllama and isn't in the scraped output; under the
        // no-llama.cpp-changes constraint it's unreachable. So the standard OAI
        // text stream is delivered correctly at completion; the progressive view
        // is the `diffusion.canvas` channel (mutating, for bespoke clients).
        let tx_frames = tx.clone();
        let mut on_frame = move |step: usize, snap: &str| {
            let (reasoning, content) = split_channels(snap);
            let ev = json!({
                "object": "diffusion.canvas",
                "step": step,
                "reasoning": reasoning,
                "content": content,
            });
            let _ = tx_frames.send(Ok(sse_event("diffusion.canvas", &ev.to_string())));
        };
        let result = tokio::time::timeout(GEN_TIMEOUT, drv.generate(&prompt, &mut on_frame)).await;
        drop(on_frame);
        drop(drv);

        match result {
            Ok(Ok(final_text)) => {
                let (reasoning, content) = split_channels(&final_text);
                // Correct answer, delivered at completion, chunked for a paced
                // reveal so standard clients still render token-by-token.
                if let Some(r) = reasoning {
                    for piece in chunk_text(&r) {
                        let _ = tx.send(Ok(sse_data(&chunk(
                            &id,
                            &model,
                            json!({ "reasoning_content": piece }),
                            None,
                        ))));
                    }
                }
                for piece in chunk_text(&content) {
                    let _ = tx.send(Ok(sse_data(&chunk(&id, &model, json!({ "content": piece }), None))));
                }
                let _ = tx.send(Ok(sse_data(&chunk(&id, &model, json!({}), Some("stop")))));
                let _ = tx.send(Ok("data: [DONE]\n\n".to_string()));
            }
            Ok(Err(e)) => {
                let _ = tx.send(Ok(sse_data(
                    &json!({"error": format!("generation failed: {e}")}).to_string(),
                )));
                let _ = tx.send(Ok("data: [DONE]\n\n".to_string()));
            }
            Err(_) => {
                let _ = tx.send(Ok(sse_data(
                    &json!({"error": "generation timed out"}).to_string(),
                )));
                let _ = tx.send(Ok("data: [DONE]\n\n".to_string()));
                fail_and_exit(&st, "stream generation timed out").await;
            }
        }
    });

    let stream = futures_util::stream::unfold(rx, |mut rx| async move {
        rx.recv().await.map(|item| (item, rx))
    });

    Response::builder()
        .header("content-type", "text/event-stream")
        .header("cache-control", "no-cache")
        .body(Body::from_stream(stream))
        .unwrap()
}

fn chunk(id: &str, model: &str, delta: Value, finish: Option<&str>) -> String {
    json!({
        "id": id,
        "object": "chat.completion.chunk",
        "created": 0,
        "model": model,
        "choices": [{
            "index": 0,
            "delta": delta,
            "finish_reason": finish,
        }],
    })
    .to_string()
}

fn sse_data(payload: &str) -> String {
    format!("data: {payload}\n\n")
}

fn sse_event(event: &str, payload: &str) -> String {
    format!("event: {event}\ndata: {payload}\n\n")
}


/// Flatten OpenAI `messages` into a single whitespace-collapsed prompt line.
/// (The CLI reads one line per turn; multi-line/multi-turn fidelity is limited
/// by design — system messages are hoisted to a preamble.)
fn build_prompt(req: &Value) -> String {
    let mut sys = String::new();
    let mut convo = String::new();
    if let Some(arr) = req.get("messages").and_then(|m| m.as_array()) {
        for m in arr {
            let role = m.get("role").and_then(|r| r.as_str()).unwrap_or("user");
            let content = m.get("content").map(content_to_string).unwrap_or_default();
            let target = if role == "system" { &mut sys } else { &mut convo };
            if !target.is_empty() {
                target.push(' ');
            }
            target.push_str(&content);
        }
    }
    let joined = if sys.is_empty() {
        convo
    } else {
        format!("{sys} {convo}")
    };
    joined.split_whitespace().collect::<Vec<_>>().join(" ")
}

/// OpenAI `content` is either a string or an array of `{type, text}` parts.
fn content_to_string(v: &Value) -> String {
    if let Some(s) = v.as_str() {
        return s.to_string();
    }
    if let Some(arr) = v.as_array() {
        return arr
            .iter()
            .filter_map(|p| p.get("text").and_then(|t| t.as_str()))
            .collect::<Vec<_>>()
            .join(" ");
    }
    String::new()
}

// ---------------------------------------------------------------------------
// Arg parsing (lenient: extract what we need, ignore the router's llama-server flags)
// ---------------------------------------------------------------------------

struct Args {
    model: String,
    port: u16,
    ngl: String,
    n_predict: String,
    cli: String,
    diffusion_passthrough: Vec<String>,
}

fn parse_args() -> Args {
    let argv: Vec<String> = std::env::args().skip(1).collect();
    let mut model = String::new();
    let mut port: u16 = 0;
    let mut ngl = std::env::var("DIFFUSION_NGL").unwrap_or_else(|_| "99".into());
    let n_predict = std::env::var("DIFFUSION_NPREDICT").unwrap_or_else(|_| "512".into());
    let mut cli = std::env::var("DIFFUSION_CLI").unwrap_or_else(|_| DEFAULT_CLI.into());
    let mut passthrough = Vec::new();

    let mut i = 0;
    while i < argv.len() {
        let a = argv[i].as_str();
        let next = argv.get(i + 1).cloned();
        match a {
            "-m" | "--model" => {
                if let Some(v) = next {
                    model = v;
                    i += 2;
                    continue;
                }
            }
            "--port" => {
                if let Some(v) = next {
                    port = v.parse().unwrap_or(0);
                    i += 2;
                    continue;
                }
            }
            "-ngl" | "--gpu-layers" | "--n-gpu-layers" => {
                if let Some(v) = next {
                    ngl = v;
                    i += 2;
                    continue;
                }
            }
            "--diffusion-cli" => {
                if let Some(v) = next {
                    cli = v;
                    i += 2;
                    continue;
                }
            }
            // Forward any diffusion-specific knobs (flag + value) to the CLI.
            _ if a.starts_with("--diffusion-") => {
                passthrough.push(a.to_string());
                if let Some(v) = next {
                    if !v.starts_with('-') {
                        passthrough.push(v);
                        i += 2;
                        continue;
                    }
                }
            }
            _ => {}
        }
        i += 1;
    }

    Args {
        model,
        port,
        ngl,
        n_predict,
        cli,
        diffusion_passthrough: passthrough,
    }
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let args = parse_args();
    if args.model.is_empty() || args.port == 0 {
        anyhow::bail!("diffusion-adapter requires -m <model> and --port <port>");
    }
    let model_name = std::path::Path::new(&args.model)
        .file_stem()
        .map(|s| s.to_string_lossy().into_owned())
        .unwrap_or_else(|| "diffusion".into());

    eprintln!(
        "diffusion-adapter: cli={} model={} port={} ngl={} n_predict={}",
        args.cli, args.model, args.port, args.ngl, args.n_predict
    );

    // Spawn the resident CLI child.
    let mut cmd = Command::new(&args.cli);
    cmd.arg("-m")
        .arg(&args.model)
        .arg("-ngl")
        .arg(&args.ngl)
        .arg("-cnv")
        .arg("--diffusion-visual")
        .arg("--diffusion-visual-interval")
        .arg("1")
        .arg("-n")
        .arg(&args.n_predict);
    for p in &args.diffusion_passthrough {
        cmd.arg(p);
    }
    cmd.env("COLUMNS", "100000") // kill line-wrap so each canvas line is one physical line
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .kill_on_drop(true);

    let mut child = cmd.spawn().map_err(|e| {
        anyhow::anyhow!("failed to spawn llama-diffusion-cli '{}': {e}", args.cli)
    })?;

    let stdin = child.stdin.take().expect("child stdin piped");
    let stdout = child.stdout.take().expect("child stdout piped");
    let stderr = child.stderr.take().expect("child stderr piped");

    // Drain the child's stderr to our stderr (so its load logs are visible and
    // the pipe never fills up and blocks the child).
    tokio::spawn(async move {
        let mut lines = BufReader::new(stderr);
        let mut buf = Vec::new();
        let mut byte = [0u8; 4096];
        loop {
            match lines.read(&mut byte).await {
                Ok(0) | Err(_) => break,
                Ok(n) => {
                    buf.extend_from_slice(&byte[..n]);
                    while let Some(pos) = buf.iter().position(|&b| b == b'\n') {
                        let line = String::from_utf8_lossy(&buf[..pos]).into_owned();
                        eprintln!("[cli] {line}");
                        buf.drain(..=pos);
                    }
                }
            }
        }
    });

    let state = Arc::new(AppState {
        driver: Mutex::new(Driver {
            stdin,
            reader: BufReader::new(stdout),
            pending: String::new(),
        }),
        child: Mutex::new(child),
        ready: AtomicBool::new(false),
        model_name,
        req_counter: AtomicU64::new(0),
    });

    // Wait for model load (the first `> `) in the background, then flip ready.
    // The HTTP server starts immediately and answers /health 503 until then, so
    // the router's health poll behaves exactly as with llama-server.
    {
        let state = state.clone();
        tokio::spawn(async move {
            let mut drv = state.driver.lock().await;
            let load_timeout = std::time::Duration::from_secs(300);
            match tokio::time::timeout(load_timeout, drv.wait_ready()).await {
                Ok(Ok(())) => {
                    drop(drv);
                    state.ready.store(true, Ordering::Relaxed);
                    eprintln!("diffusion-adapter: model ready, serving /v1");
                }
                Ok(Err(e)) => {
                    drop(drv);
                    fail_and_exit(&state, &format!("child failed before ready: {e}")).await;
                }
                Err(_) => {
                    drop(drv);
                    fail_and_exit(&state, "model load timed out").await;
                }
            }
        });
    }

    let app = Router::new()
        .route("/health", get(health))
        .route("/v1/models", get(list_models))
        .route("/v1/chat/completions", post(chat_completions))
        .with_state(state);

    let addr = format!("127.0.0.1:{}", args.port);
    let listener = tokio::net::TcpListener::bind(&addr).await?;
    eprintln!("diffusion-adapter: listening on {addr}");
    axum::serve(listener, app).await?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn strip_ansi_removes_csi_and_keeps_utf8() {
        let s = "\x1b[23A\x1b[K**Blue**\x1b[?2026l café";
        assert_eq!(strip_ansi(s), "**Blue** café");
    }

    #[test]
    fn split_channels_separates_reasoning_and_answer() {
        let t = "<|channel>thought\nlet me think...\n<channel|>The answer is 4.";
        let (r, c) = split_channels(t);
        assert_eq!(r.as_deref(), Some("let me think..."));
        assert_eq!(c, "The answer is 4.");
    }

    #[test]
    fn split_channels_no_marker_is_all_content() {
        let (r, c) = split_channels("just text");
        assert_eq!(r, None);
        assert_eq!(c, "just text");
    }

    #[test]
    fn extract_final_takes_text_after_last_frame() {
        let gen = "\x1b[?2026lnoise\x1b[?2026l\x1b[22A\x1b[J final answer\ntotal time: 1ms\n";
        assert_eq!(extract_final(gen), "final answer");
    }

    #[test]
    fn extract_frames_emits_each_canvas() {
        let gen = "A frame1\x1b[?2026lB frame2\x1b[?2026l";
        let mut cursor = 0;
        let mut step = 0;
        let mut frames = Vec::new();
        let mut cb = |s: usize, t: &str| frames.push((s, t.to_string()));
        extract_frames(gen, &mut cursor, &mut step, &mut cb);
        assert_eq!(frames.len(), 2);
        assert_eq!(frames[0], (1, "A frame1".to_string()));
        assert_eq!(frames[1], (2, "B frame2".to_string()));
    }

    #[test]
    fn build_prompt_collapses_and_hoists_system() {
        let req = json!({"messages": [
            {"role": "system", "content": "Be brief."},
            {"role": "user", "content": "What is\n2+2?"},
        ]});
        assert_eq!(build_prompt(&req), "Be brief. What is 2+2?");
    }

    #[test]
    fn chunk_text_reveals_word_by_word_and_reconstructs() {
        assert_eq!(chunk_text("Red and blue."), vec!["Red ", "and ", "blue."]);
        assert_eq!(chunk_text("Red and blue.").concat(), "Red and blue.");
        assert!(chunk_text("").is_empty());
    }
}
