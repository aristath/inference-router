#!/usr/bin/env python3
"""
End-to-end test for the dynamic instance pool.

Flow:
  1. Pre-load a small model so estimated_vram is measured.
  2. Send request A with high max_tokens (slow, keeps the instance busy).
  3. While A is streaming, send request B to the same model.
  4. The router should detect the instance is busy, see VRAM headroom,
     and spawn a second instance in the background.
  5. Verify via ps that a second llama-server process appeared.
"""
import http.client
import json
import subprocess
import sys
import threading
import time
import urllib.request

MODEL   = "qwen35-35b-a3b-udq4kxl"
BASE    = "http://localhost:8080"
PROMPT_A = "Count from 1 to 200, one number per line, no commentary."
PROMPT_B = "List the first 100 prime numbers, one per line."
MAX_TOK  = 500


def api(path, method="GET", body=None):
    req = urllib.request.Request(
        BASE + path,
        data=json.dumps(body).encode() if body else None,
        headers={"Content-Type": "application/json"},
        method=method,
    )
    with urllib.request.urlopen(req, timeout=300) as r:
        return json.loads(r.read())


def llama_procs():
    r = subprocess.run(["pgrep", "-a", "llama-server"], capture_output=True, text=True)
    lines = [l for l in r.stdout.strip().splitlines() if l]
    return len(lines), lines


def stream_request(label, prompt, results):
    """Stream a /v1/chat/completions request, collecting chunk count."""
    body = json.dumps({
        "model": MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": MAX_TOK,
        "stream": True,
    }).encode()

    url = urllib.parse.urlparse(BASE)
    conn = http.client.HTTPConnection(url.hostname, url.port, timeout=300)
    conn.request("POST", "/v1/chat/completions",
                 body=body,
                 headers={"Content-Type": "application/json"})
    resp = conn.getresponse()
    chunks = 0
    start = time.time()
    print(f"  [{label}] HTTP {resp.status} — streaming...")
    for raw in resp:
        line = raw.decode(errors="replace").strip()
        if line.startswith("data: ") and "[DONE]" not in line:
            chunks += 1
    elapsed = time.time() - start
    print(f"  [{label}] done — {chunks} chunks in {elapsed:.1f}s")
    results[label] = chunks
    conn.close()


# ── Step 1: pre-load ─────────────────────────────────────────────────────────
import urllib.parse
print(f"Step 1: loading {MODEL} (may take a minute on first run)...")
resp = api(f"/api/models/{MODEL}/load", method="POST")
print(f"  load response: {resp}")

status = api("/api/status")
m = next((m for m in status["models"] if m["id"] == MODEL), None)
if not m:
    sys.exit(f"Model {MODEL!r} not found in status response")
print(f"  state={m['state']}  estimated_vram={m['estimated_vram']/1024**3:.1f} GiB")
if m["estimated_vram"] == 0:
    print("  WARNING: estimated_vram is 0 — scale-up won't trigger. "
          "Re-run the test; after first load the estimate is measured.")

# ── Step 2: baseline process count ───────────────────────────────────────────
n0, procs0 = llama_procs()
print(f"\nStep 2: baseline — {n0} llama-server process(es)")

# ── Step 3: concurrent requests ──────────────────────────────────────────────
print(f"\nStep 3: firing request A...")
results = {}
t_a = threading.Thread(target=stream_request, args=("A", PROMPT_A, results), daemon=True)
t_a.start()

time.sleep(1.5)          # let A get a few tokens in flight

n1, _ = llama_procs()
print(f"  processes while A is running: {n1}")

print(f"  firing request B (A still in flight)...")
t_b = threading.Thread(target=stream_request, args=("B", PROMPT_B, results), daemon=True)
t_b.start()

# Poll process count for up to 60 s waiting for a second instance to appear.
peak = n1
deadline = time.time() + 60
while time.time() < deadline:
    time.sleep(2)
    n, _ = llama_procs()
    if n > peak:
        peak = n
        print(f"  *** new process detected — now {peak} llama-server(s) running ***")
    if not t_a.is_alive() and not t_b.is_alive():
        break

t_a.join(timeout=300)
t_b.join(timeout=300)

# ── Step 4: final count ───────────────────────────────────────────────────────
n_final, _ = llama_procs()
print(f"\nStep 4: final — {n_final} llama-server process(es)")

# ── Report ────────────────────────────────────────────────────────────────────
print("\n══ Results ══")
print(f"  Baseline processes : {n0}")
print(f"  Peak processes     : {peak}")
print(f"  Final processes    : {n_final}")
print(f"  Request A chunks   : {results.get('A', 'n/a')}")
print(f"  Request B chunks   : {results.get('B', 'n/a')}")

if peak > n0:
    print(f"\n✓ PASS — second instance spawned ({n0} → {peak} processes)")
else:
    print(f"\n✗ FAIL — no second instance observed (stayed at {n0})")
    print("  Check: was estimated_vram > 0? Was request A still running when B arrived?")
