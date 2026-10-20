//! Real HTTP requests + owned synthetic processes; no GPU/model downloads.
use inference_router::{
    config::{BinaryPreset, JsonStore, ModelAlias, ModelConfig, ModelState, WeightsFormat},
    orchestrator::{
        media::{MediaConfig, MediaWorker},
        AppState, Orchestrator,
    },
};
use serde_json::{json, Value};
use std::{collections::BTreeMap, os::unix::fs::PermissionsExt, sync::Arc, time::Duration};
use tempfile::TempDir;
use tokio::time::timeout;

struct Fixture {
    dir: TempDir,
    state: AppState,
    url: String,
    server: tokio::task::JoinHandle<()>,
}
impl Drop for Fixture {
    fn drop(&mut self) {
        self.state.media.shutdown.cancel();
        self.server.abort();
    }
}
impl Fixture {
    async fn new(timeout_secs: u64) -> Self {
        std::env::set_var(
            "INFERENCE_ROUTER_WORKER_EXECUTABLE",
            env!("CARGO_BIN_EXE_inference-router"),
        );
        let dir = TempDir::new().unwrap();
        let llm = dir.path().join("llm.py");
        std::fs::write(&llm,format!(r#"#!/usr/bin/env python3
import argparse,http.server,json,os,time,pathlib
p=argparse.ArgumentParser();p.add_argument('--port',type=int);args,_=p.parse_known_args()
root=pathlib.Path({root:?})
with (root/'events').open('a') as f:f.write('load:'+str(os.getpid())+'\n')
class Server(http.server.BaseHTTPRequestHandler):
 def log_message(self,*args):pass
 def do_GET(self):
  while (root/'hold-health').exists():time.sleep(.01)
  self.send_response(200);self.end_headers();self.wfile.write(b'{{"status":"ok"}}')
 def do_POST(self):
  data=json.loads(self.rfile.read(int(self.headers['Content-Length'])))
  name=data.get('messages',[{{'content':'?'}}])[-1]['content']
  with (root/'events').open('a') as f:f.write('request:'+name+'\n')
  if data.get('stream'):
   self.send_response(200);self.send_header('Content-Type','text/event-stream');self.end_headers()
   self.wfile.write(b'data: {{"choices":[{{"delta":{{"content":"start"}}}}]}}\n\n');self.wfile.flush()
   while not (root/'finish-stream').exists():time.sleep(.01)
   self.wfile.write(b'data: [DONE]\n\n');self.wfile.flush()
  else:
   self.send_response(200);self.send_header('Content-Type','application/json');self.end_headers()
   self.wfile.write(json.dumps({{'model':data['model'],'choices':[{{'message':{{'role':'assistant','content':name}}}}]}}).encode())
http.server.ThreadingHTTPServer(('127.0.0.1',args.port),Server).serve_forever()
"#,root=dir.path().to_str().unwrap())).unwrap();
        std::fs::set_permissions(&llm, std::fs::Permissions::from_mode(0o755)).unwrap();
        let media = dir.path().join("media.py");
        std::fs::write(
            &media,
            format!(
                r#"import json,os,pathlib,subprocess,sys,time
root=pathlib.Path({root:?});data=json.load(sys.stdin)
(root/'media-started').write_text(str(os.getpid()))
with (root/'events').open('a') as f:f.write('media\n')
if data.get('child'):
 child=subprocess.Popen([sys.executable,'-c','import time;time.sleep(300)'])
 (root/'child-pid').write_text(str(child.pid))
if data.get('fail'):sys.exit(3)
while not (root/'finish-media').exists():time.sleep(.01)
print(json.dumps({{'image':'saved.png'}}),flush=True)
"#,
                root = dir.path().to_str().unwrap()
            ),
        )
        .unwrap();
        let models = Arc::new(JsonStore::new(dir.path().join("models.json")));
        let model = ModelConfig {
            id: "fake".into(),
            name: "Fake resident".into(),
            binary: llm,
            model_path: dir.path().join("fake.gguf"),
            weights_format: WeightsFormat::Safetensors,
            context: 128,
            ..ModelConfig::default()
        };
        models.replace(vec![model.clone()]);
        let state = Arc::new(Orchestrator::new(
            models,
            Arc::new(JsonStore::<Vec<BinaryPreset>>::new(
                dir.path().join("presets.json"),
            )),
            Arc::new(JsonStore::<Vec<ModelAlias>>::new(
                dir.path().join("aliases.json"),
            )),
            8080,
        ));
        // The synthetic initial worker ignores GPU flags. Mark it GPU-backed via
        // the real spawn path so the handoff must stop it and restore a new PID.
        let mut initial = model;
        initial.cpu_only = false;
        initial.n_gpu_layers = Some(1);
        let mut pending = state
            .process_manager
            .lock()
            .await
            .spawn_child(&initial, None)
            .unwrap();
        pending
            .wait_for_health(Duration::from_secs(5))
            .await
            .unwrap();
        let pid = pending.pid;
        drop(
            state
                .process_manager
                .lock()
                .await
                .register(pending)
                .unwrap_or_else(|_| panic!("register failed")),
        );
        {
            let mut data = state.data.lock().await;
            let m = data.models.get_mut("fake").unwrap();
            m.state = ModelState::Running;
            m.pid = Some(pid);
        }
        state
            .media
            .configure(
                MediaConfig {
                    resident_model: "fake".into(),
                    workers: BTreeMap::from([(
                        "image".into(),
                        MediaWorker {
                            command: vec![
                                "/usr/bin/python3".into(),
                                media.to_string_lossy().into(),
                            ],
                            timeout_secs,
                        },
                    )]),
                },
                dir.path().join("media-runtime"),
            )
            .unwrap();
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let url = format!("http://{}", listener.local_addr().unwrap());
        let app = inference_router::lifecycle::build_router(state.clone());
        let server = tokio::spawn(async move {
            axum::serve(listener, app).await.unwrap();
        });
        Self {
            dir,
            state,
            url,
            server,
        }
    }
    async fn fallback(&self) {
        let mut cpu = self.state.data.lock().await.models["fake"].clone();
        cpu.id = "cpu".into();
        cpu.cpu_only = true;
        cpu.weights_format = WeightsFormat::Gguf;
        cpu.state = ModelState::Idle;
        cpu.pid = None;
        self.state.add_model(cpu).await.unwrap();
        for alias in [
            ModelAlias {
                alias: "fast".into(),
                target: "cpu".into(),
                fallback: None,
            },
            ModelAlias {
                alias: "default".into(),
                target: "fake".into(),
                fallback: Some("fast".into()),
            },
        ] {
            self.state.add_alias(alias).await.unwrap();
        }
    }
    fn request_model(&self, model: &str, stream: bool) -> reqwest::RequestBuilder {
        self.inference("fallback test").json(&json!({"model":model,"stream":stream,"messages":[{"role":"user","content":"fallback test"}]}))
    }
    fn mark(&self, name: &str) {
        std::fs::write(self.dir.path().join(name), "").unwrap();
    }
    async fn wait(&self, name: &str) {
        timeout(Duration::from_secs(5), async {
            while !self.dir.path().join(name).exists() {
                tokio::time::sleep(Duration::from_millis(5)).await;
            }
        })
        .await
        .unwrap();
    }
    fn events(&self) -> Vec<String> {
        std::fs::read_to_string(self.dir.path().join("events"))
            .unwrap()
            .lines()
            .map(str::to_owned)
            .collect()
    }
    fn client(&self) -> reqwest::Client {
        reqwest::Client::builder()
            .timeout(Duration::from_secs(10))
            .build()
            .unwrap()
    }
    fn inference(&self, name: &str) -> reqwest::RequestBuilder {
        self.client()
            .post(format!("{}/v1/chat/completions", self.url))
            .json(&json!({"model":"fake","messages":[{"role":"user","content":name}]}))
    }
    fn media(&self, input: Value) -> reqwest::RequestBuilder {
        self.client()
            .post(format!("{}/api/gpu/run/image", self.url))
            .json(&input)
    }
}

#[tokio::test]
async fn drains_stream_and_queues_other_clients_until_media_cleanup_and_resident_restore() {
    let f = Fixture::new(30).await;
    let mut stream = f
        .inference("active")
        .json(
            &json!({"model":"fake","messages":[{"role":"user","content":"active"}],"stream":true}),
        )
        .send()
        .await
        .unwrap();
    assert!(stream.chunk().await.unwrap().is_some());
    let media = tokio::spawn(f.media(json!({})).send());
    timeout(Duration::from_secs(3), async {
        while f.state.gpu_gate.snapshot().queued_media == 0 {
            tokio::task::yield_now().await;
        }
    })
    .await
    .unwrap();
    let mut pending = tokio::spawn(f.inference("external-app").send());
    assert!(timeout(Duration::from_millis(60), &mut pending)
        .await
        .is_err());
    assert!(!f.dir.path().join("media-started").exists());
    f.mark("finish-stream");
    while stream.chunk().await.unwrap().is_some() {}
    f.wait("media-started").await;
    assert_eq!(f.state.gpu_gate.snapshot().state.phase, "media");
    assert!(timeout(Duration::from_millis(60), &mut pending)
        .await
        .is_err());
    assert!(!f.events().iter().any(|s| s == "request:external-app"));
    f.mark("finish-media");
    let result = media.await.unwrap().unwrap();
    assert!(
        result.status().is_success(),
        "{}",
        result.text().await.unwrap()
    );
    assert!(pending.await.unwrap().unwrap().status().is_success());
    let events = f.events();
    let media_index = events.iter().position(|s| s == "media").unwrap();
    let restore_index = events
        .iter()
        .enumerate()
        .skip(media_index + 1)
        .find(|(_, s)| s.starts_with("load:"))
        .unwrap()
        .0;
    assert!(
        events
            .iter()
            .position(|s| s == "request:external-app")
            .unwrap()
            > restore_index
    );
    assert!(f.state.gpu_gate.snapshot().state.error.is_none());
}

#[tokio::test]
async fn crashed_worker_descendants_are_cleaned_and_inference_is_restored() {
    let f = Fixture::new(10).await;
    let response = f
        .media(json!({"fail":true,"child":true}))
        .send()
        .await
        .unwrap();
    assert_eq!(response.status(), 503);
    assert!(response.text().await.unwrap().contains("exited"));
    let child: i32 = std::fs::read_to_string(f.dir.path().join("child-pid"))
        .unwrap()
        .parse()
        .unwrap();
    let status = std::fs::read_to_string(format!("/proc/{child}/stat")).unwrap_or_default();
    assert!(status.is_empty() || status.rsplit_once(") ").unwrap().1.starts_with('Z'));
    assert!(f
        .inference("after-failure")
        .send()
        .await
        .unwrap()
        .status()
        .is_success());
    assert_eq!(f.state.gpu_gate.snapshot().state.phase, "llm");
}

#[tokio::test]
async fn hung_worker_times_out_then_restores_inference() {
    let f = Fixture::new(1).await;
    let media = tokio::spawn(f.media(json!({})).send());
    f.wait("media-started").await;
    let pending = tokio::spawn(f.inference("waiting").send());
    let response = media.await.unwrap().unwrap();
    assert_eq!(response.status(), 503);
    assert!(response.text().await.unwrap().contains("runtime"));
    assert!(pending.await.unwrap().unwrap().status().is_success());
    assert_eq!(f.state.gpu_gate.snapshot().queued_inference, 0);
}

#[tokio::test]
async fn disconnecting_media_client_cleans_worker_and_resumes_waiting_inference() {
    let f = Fixture::new(20).await;
    let media = tokio::spawn(f.media(json!({"child":true})).send());
    f.wait("media-started").await;
    let pending = tokio::spawn(f.inference("after-disconnect").send());
    media.abort();
    let _ = media.await;
    let response = timeout(Duration::from_secs(5), pending)
        .await
        .expect("media cancellation must release queued inference")
        .unwrap()
        .unwrap();
    assert!(response.status().is_success());
}

#[tokio::test]
async fn cancelled_queued_inference_does_not_run_later() {
    let f = Fixture::new(20).await;
    let media = tokio::spawn(f.media(json!({})).send());
    f.wait("media-started").await;
    let pending = tokio::spawn(f.inference("cancelled-client").send());
    timeout(Duration::from_secs(3), async {
        while f.state.gpu_gate.snapshot().queued_inference != 1 {
            tokio::task::yield_now().await;
        }
    })
    .await
    .unwrap();
    pending.abort();
    let _ = pending.await;
    timeout(Duration::from_secs(3), async {
        while f.state.gpu_gate.snapshot().queued_inference != 0 {
            tokio::task::yield_now().await;
        }
    })
    .await
    .unwrap();
    f.mark("finish-media");
    assert!(media.await.unwrap().unwrap().status().is_success());
    assert!(!f.events().iter().any(|s| s == "request:cancelled-client"));
}

#[tokio::test]
async fn manual_loads_queue_and_waiting_llm_runs_before_later_media() {
    let f = Fixture::new(20).await;
    let first = tokio::spawn(f.media(json!({})).send());
    f.wait("media-started").await;
    let pending = tokio::spawn(f.inference("first-in-line").send());
    timeout(Duration::from_secs(3), async {
        while f.state.gpu_gate.snapshot().queued_inference != 1 {
            tokio::task::yield_now().await;
        }
    })
    .await
    .unwrap();
    let mut manual = tokio::spawn(
        f.client()
            .post(format!("{}/api/models/fake/load", f.url))
            .send(),
    );
    timeout(Duration::from_secs(3), async {
        while f.state.gpu_gate.snapshot().queued_inference != 2 {
            tokio::task::yield_now().await;
        }
    })
    .await
    .unwrap();
    let second = tokio::spawn(f.media(json!({})).send());
    timeout(Duration::from_secs(3), async {
        while f.state.gpu_gate.snapshot().queued_media != 1 {
            tokio::task::yield_now().await;
        }
    })
    .await
    .unwrap();
    assert!(timeout(Duration::from_millis(30), &mut manual)
        .await
        .is_err());
    f.mark("finish-media");
    assert!(first.await.unwrap().unwrap().status().is_success());
    assert!(pending.await.unwrap().unwrap().status().is_success());
    assert!(manual.await.unwrap().unwrap().status().is_success());
    assert!(second.await.unwrap().unwrap().status().is_success());
    let events = f.events();
    let media_events: Vec<_> = events
        .iter()
        .enumerate()
        .filter(|(_, s)| s.as_str() == "media")
        .map(|(i, _)| i)
        .collect();
    assert_eq!(media_events.len(), 2);
    let request = events
        .iter()
        .position(|s| s == "request:first-in-line")
        .unwrap();
    assert!(request > media_events[0] && request < media_events[1]);
}

#[tokio::test]
async fn failed_restore_is_explicit_and_recovery_can_restore_the_resident() {
    let f = Fixture::new(20).await;
    let media = tokio::spawn(f.media(json!({})).send());
    f.wait("media-started").await;
    let binary = {
        let mut data = f.state.data.lock().await;
        let m = data.models.get_mut("fake").unwrap();
        std::mem::replace(&mut m.binary, "/bin/false".into())
    };
    let pending = tokio::spawn(f.inference("cannot-run-yet").send());
    f.mark("finish-media");
    let response = media.await.unwrap().unwrap();
    assert_eq!(response.status(), 503);
    assert!(response
        .text()
        .await
        .unwrap()
        .contains("restoration failed"));
    assert_eq!(pending.await.unwrap().unwrap().status(), 503);
    assert!(f.state.gpu_gate.snapshot().state.error.is_some());
    f.state
        .data
        .lock()
        .await
        .models
        .get_mut("fake")
        .unwrap()
        .binary = binary;
    let recovery = f
        .client()
        .post(format!("{}/api/gpu/recover", f.url))
        .send()
        .await
        .unwrap();
    assert!(
        recovery.status().is_success(),
        "{}",
        recovery.text().await.unwrap()
    );
    assert!(f
        .inference("recovered")
        .send()
        .await
        .unwrap()
        .status()
        .is_success());
    assert!(f.state.gpu_gate.snapshot().state.error.is_none());
}

#[tokio::test]
async fn cancelling_media_waiting_for_a_stream_does_not_load_or_run_it() {
    let f = Fixture::new(20).await;
    let mut stream = f
        .inference("active")
        .json(
            &json!({"model":"fake","messages":[{"role":"user","content":"active"}],"stream":true}),
        )
        .send()
        .await
        .unwrap();
    assert!(stream.chunk().await.unwrap().is_some());
    let media = tokio::spawn(f.media(json!({})).send());
    timeout(Duration::from_secs(3), async {
        while f.state.gpu_gate.snapshot().queued_media != 1 {
            tokio::task::yield_now().await;
        }
    })
    .await
    .unwrap();
    media.abort();
    let _ = media.await;
    timeout(Duration::from_secs(3), async {
        while f.state.gpu_gate.snapshot().queued_media != 0 {
            tokio::task::yield_now().await;
        }
    })
    .await
    .unwrap();
    f.mark("finish-stream");
    while stream.chunk().await.unwrap().is_some() {}
    assert!(!f.dir.path().join("media-started").exists());
    assert!(f
        .inference("next")
        .send()
        .await
        .unwrap()
        .status()
        .is_success());
}

#[tokio::test]
async fn recorded_worker_is_cleaned_before_recovery_reloads_the_resident() {
    use std::process::Stdio;
    use tokio::io::AsyncWriteExt;
    let f = Fixture::new(20).await;
    f.state.stop_model("fake").await.unwrap();
    let marker = f.dir.path().join("media-runtime/active-worker.json");
    let worker = f.state.media.worker("image").unwrap();
    let mut child = tokio::process::Command::new(env!("CARGO_BIN_EXE_inference-router"))
        .arg("--gpu-worker")
        .arg(&marker)
        .arg("--")
        .args(worker.command)
        .process_group(0)
        .stdin(Stdio::piped())
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .kill_on_drop(true)
        .spawn()
        .unwrap();
    child.stdin.take().unwrap().write_all(b"{}").await.unwrap();
    f.wait("media-started").await;
    let response = f
        .client()
        .post(format!("{}/api/gpu/recover", f.url))
        .send()
        .await
        .unwrap();
    assert!(
        response.status().is_success(),
        "{}",
        response.text().await.unwrap()
    );
    assert!(!child.wait().await.unwrap().success());
    assert!(!marker.exists());
    assert!(f
        .inference("after-recovery")
        .send()
        .await
        .unwrap()
        .status()
        .is_success());
    let events = f.events();
    let media = events.iter().position(|s| s == "media").unwrap();
    assert!(events
        .iter()
        .skip(media + 1)
        .any(|s| s.starts_with("load:")));
}

#[tokio::test]
async fn recovery_refuses_to_kill_a_reused_process_identity() {
    use std::process::Stdio;
    let f = Fixture::new(20).await;
    let marker = f.dir.path().join("media-runtime/active-worker.json");
    let mut child = tokio::process::Command::new(env!("CARGO_BIN_EXE_inference-router"))
        .arg("--gpu-worker")
        .arg(&marker)
        .arg("--")
        .args(["/bin/sleep", "30"])
        .process_group(0)
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .kill_on_drop(true)
        .spawn()
        .unwrap();
    f.wait("media-runtime/active-worker.json").await;
    let original = std::fs::read_to_string(&marker).unwrap();
    let mut record: Value = serde_json::from_str(&original).unwrap();
    record["start_ticks"] = json!(record["start_ticks"].as_u64().unwrap() + 1);
    std::fs::write(&marker, serde_json::to_vec(&record).unwrap()).unwrap();
    let response = f
        .client()
        .post(format!("{}/api/gpu/recover", f.url))
        .send()
        .await
        .unwrap();
    assert_eq!(response.status(), 503);
    assert!(response.text().await.unwrap().contains("unverified"));
    assert!(child.try_wait().unwrap().is_none());
    assert_eq!(f.inference("blocked").send().await.unwrap().status(), 503);
    std::fs::write(&marker, original).unwrap();
    assert!(f
        .client()
        .post(format!("{}/api/gpu/recover", f.url))
        .send()
        .await
        .unwrap()
        .status()
        .is_success());
    let _ = child.wait().await.unwrap();
}

#[tokio::test]
async fn cancelled_caller_does_not_allow_media_to_overlap_its_detached_model_load() {
    let f = Fixture::new(20).await;
    f.state.stop_model("fake").await.unwrap();
    f.mark("hold-health");
    let loading = tokio::spawn(f.inference("cancelled-load").send());
    timeout(Duration::from_secs(5), async {
        while f.events().iter().filter(|s| s.starts_with("load:")).count() < 2 {
            tokio::task::yield_now().await;
        }
    })
    .await
    .unwrap();
    loading.abort();
    let _ = loading.await;
    let media = tokio::spawn(f.media(json!({})).send());
    timeout(Duration::from_secs(3), async {
        while f.state.gpu_gate.snapshot().queued_media != 1 {
            tokio::task::yield_now().await;
        }
    })
    .await
    .unwrap();
    tokio::time::sleep(Duration::from_millis(60)).await;
    assert!(!f.dir.path().join("media-started").exists());
    std::fs::remove_file(f.dir.path().join("hold-health")).unwrap();
    f.wait("media-started").await;
    f.mark("finish-media");
    assert!(media.await.unwrap().unwrap().status().is_success());
}

#[tokio::test]
async fn inference_queues_before_waiting_for_a_large_request_body() {
    use tokio::io::AsyncWriteExt;
    let f = Fixture::new(20).await;
    let media = tokio::spawn(f.media(json!({})).send());
    f.wait("media-started").await;
    let mut socket = tokio::net::TcpStream::connect(f.url.trim_start_matches("http://"))
        .await
        .unwrap();
    socket.write_all(b"POST /v1/chat/completions HTTP/1.1\r\nHost: localhost\r\nContent-Type: application/json\r\nContent-Length: 100000000\r\n\r\n{").await.unwrap();
    timeout(Duration::from_secs(3), async {
        while f.state.gpu_gate.snapshot().queued_inference != 1 {
            tokio::task::yield_now().await;
        }
    })
    .await
    .unwrap();
    drop(socket);
    // A partial HTTP body is not read while queued. Its EOF is diagnosed on
    // admission, before model lookup/dispatch; no incomplete request may run.
    f.mark("finish-media");
    assert!(media.await.unwrap().unwrap().status().is_success());
    timeout(Duration::from_secs(3), async {
        while f.state.gpu_gate.snapshot().queued_inference != 0 {
            tokio::task::yield_now().await;
        }
    })
    .await
    .unwrap();
    assert!(!f.events().iter().any(|event| event.starts_with("request:")));
}

#[tokio::test]
async fn fallback_answers_on_cpu_during_media_and_returns_to_primary_after_restore() {
    let f = Fixture::new(30).await;
    f.fallback().await;
    let primary: Value = f
        .request_model("default", false)
        .send()
        .await
        .unwrap()
        .json()
        .await
        .unwrap();
    assert_eq!(primary["model"], "fake");
    let media = tokio::spawn(f.media(json!({})).send());
    f.wait("media-started").await;
    let fallback: Value = timeout(
        Duration::from_secs(3),
        f.request_model("default", false).send(),
    )
    .await
    .unwrap()
    .unwrap()
    .json()
    .await
    .unwrap();
    assert_eq!(fallback["model"], "cpu");
    assert!(!media.is_finished());
    assert_eq!(f.state.gpu_gate.snapshot().queued_inference, 0);
    let direct: Value = f
        .request_model("fast", false)
        .send()
        .await
        .unwrap()
        .json()
        .await
        .unwrap();
    assert_eq!(direct["model"], "cpu");
    let no_fallback = tokio::spawn(f.request_model("fake", false).send());
    timeout(Duration::from_secs(2), async {
        while f.state.gpu_gate.snapshot().queued_inference == 0 {
            tokio::task::yield_now().await;
        }
    })
    .await
    .unwrap();
    assert!(!no_fallback.is_finished());
    f.mark("finish-media");
    assert!(media.await.unwrap().unwrap().status().is_success());
    assert_eq!(
        no_fallback
            .await
            .unwrap()
            .unwrap()
            .json::<Value>()
            .await
            .unwrap()["model"],
        "fake"
    );
    assert_eq!(
        f.request_model("default", false)
            .send()
            .await
            .unwrap()
            .json::<Value>()
            .await
            .unwrap()["model"],
        "fake"
    );
}

#[tokio::test]
async fn cpu_stream_does_not_block_media_restoration_or_switch_models_mid_response() {
    let f = Fixture::new(30).await;
    f.fallback().await;
    let media = tokio::spawn(f.media(json!({})).send());
    f.wait("media-started").await;
    let mut stream = f.request_model("default", true).send().await.unwrap();
    assert!(stream.chunk().await.unwrap().is_some());
    f.mark("finish-media");
    assert!(timeout(Duration::from_secs(4), media)
        .await
        .unwrap()
        .unwrap()
        .unwrap()
        .status()
        .is_success());
    assert_eq!(
        f.request_model("default", false)
            .send()
            .await
            .unwrap()
            .json::<Value>()
            .await
            .unwrap()["model"],
        "fake"
    );
    f.mark("finish-stream");
    let tail = stream.text().await.unwrap();
    assert!(tail.contains("[DONE]"));
}

#[tokio::test]
async fn fallback_handles_model_field_after_large_messages_without_queueing_or_reordering_body() {
    let f = Fixture::new(30).await;
    f.fallback().await;
    let media = tokio::spawn(f.media(json!({})).send());
    f.wait("media-started").await;
    let content = "x".repeat(200_000);
    let body =
        format!(r#"{{"messages":[{{"role":"user","content":"{content}"}}],"model":"default"}}"#);
    let reply = timeout(
        Duration::from_secs(3),
        f.client()
            .post(format!("{}/v1/chat/completions", f.url))
            .header("Content-Type", "application/json")
            .body(body)
            .send(),
    )
    .await
    .unwrap()
    .unwrap();
    let reply: Value = reply.json().await.unwrap();
    assert_eq!(reply["model"], "cpu");
    assert_eq!(reply["choices"][0]["message"]["content"], content);
    assert!(!media.is_finished());
    f.mark("finish-media");
    assert!(media.await.unwrap().unwrap().status().is_success());
}

#[tokio::test]
async fn fallback_remains_usable_during_restore_and_after_gpu_recovery_failure() {
    let f = Fixture::new(30).await;
    f.fallback().await;
    let media = tokio::spawn(f.media(json!({})).send());
    f.wait("media-started").await;
    assert_eq!(
        f.request_model("default", false)
            .send()
            .await
            .unwrap()
            .json::<Value>()
            .await
            .unwrap()["model"],
        "cpu"
    );
    // Warm CPU instance stays available while the GPU resident's health check stalls.
    f.mark("hold-health");
    f.mark("finish-media");
    timeout(Duration::from_secs(3), async {
        while f.state.gpu_gate.snapshot().state.phase != "restoring" {
            tokio::task::yield_now().await;
        }
    })
    .await
    .unwrap();
    assert_eq!(
        timeout(
            Duration::from_secs(2),
            f.request_model("default", false).send()
        )
        .await
        .unwrap()
        .unwrap()
        .json::<Value>()
        .await
        .unwrap()["model"],
        "cpu"
    );
    // Manual CPU loads also use the independent admission lane.
    drop(
        timeout(Duration::from_secs(2), f.state.clone().ensure_loaded("cpu"))
            .await
            .unwrap()
            .unwrap(),
    );
    std::fs::remove_file(f.dir.path().join("hold-health")).unwrap();
    assert!(media.await.unwrap().unwrap().status().is_success());
    f.state
        .gpu_gate
        .set_error(Some("GPU recovery failed".into()));
    assert_eq!(
        f.request_model("default", false)
            .send()
            .await
            .unwrap()
            .json::<Value>()
            .await
            .unwrap()["model"],
        "cpu"
    );
    assert_eq!(
        f.request_model("fake", false)
            .send()
            .await
            .unwrap()
            .status(),
        503
    );
}
