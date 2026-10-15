"""Router-owned native SYCL worker sharing B60 admission with media services."""

import ctypes
import fcntl
import os
import signal
import sys
from pathlib import Path
from urllib.error import HTTPError
from urllib.request import Request, urlopen


def main():
    args = sys.argv[1:]
    binary = os.environ.get(
        "INFERENCE_LLAMA_SERVER",
        str(Path.home() / "llama.cpp-sycl/build-fast-b60/bin/llama-server"),
    )
    parent = os.getppid()
    if (
        parent == 1
        or ctypes.CDLL(None).prctl(1, signal.SIGKILL, 0, 0, 0) != 0
        or os.getppid() != parent
    ):
        raise SystemExit("Could not establish worker parent-death protection")
    # Introspection does not reserve the GPU or evict resident models.
    introspection = any(
        arg in args for arg in ("--help", "-h", "--version", "--list-devices")
    )
    if not introspection:
        url = os.environ.get("INFERENCE_ROUTER_URL", "http://127.0.0.1:8080")
        try:
            with urlopen(
                Request(url + "/api/backends/sycl/release-idle", data=b""), timeout=30
            ):
                pass
        except HTTPError as error:
            error.close()
            raise SystemExit(
                f"B60 admission unavailable (router HTTP {error.code})"
            ) from None
        # Native exec inherits both descriptors; reservations last until it exits.
        for path in (
            Path.home() / "qwen3-tts-b60/gpu.lock",
            Path.home() / "minimax-h3-b60/runtime/locks/0000:0e:00.0.lock",
        ):
            path.parent.mkdir(parents=True, exist_ok=True)
            fd = os.open(path, os.O_CREAT | os.O_WRONLY | os.O_APPEND, 0o600)
            try:
                fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                raise SystemExit("B60 is busy; no inference was submitted") from None
            os.set_inheritable(fd, True)
    os.execv(binary, [binary, *args])


if __name__ == "__main__":
    main()
