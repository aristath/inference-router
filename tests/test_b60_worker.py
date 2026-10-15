import fcntl
import importlib.util
import os
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch
from urllib.error import HTTPError

spec = importlib.util.spec_from_file_location(
    "b60_worker", Path(__file__).parents[1] / "scripts/llama-sycl-b60-worker.py"
)
worker = importlib.util.module_from_spec(spec)
spec.loader.exec_module(worker)


class NativeExec(Exception):
    pass


class WorkerTests(unittest.TestCase):
    def setUp(self):
        self.folder = tempfile.TemporaryDirectory()
        self.addCleanup(self.folder.cleanup)
        self.home = Path(self.folder.name)
        self.opened = []
        original_open = os.open

        def tracked_open(*args):
            fd = original_open(*args)
            self.opened.append(fd)
            return fd

        def close():
            for fd in self.opened:
                os.close(fd)

        self.addCleanup(close)
        for name, value in (
            ("worker.Path.home", lambda: self.home),
            ("worker.os.getppid", lambda: 1234),
        ):
            # Patch actual module objects, not a globally importable alias.
            target = worker.Path if "Path" in name else worker.os
            attr = "home" if "Path" in name else "getppid"
            patcher = patch.object(target, attr, value)
            patcher.start()
            self.addCleanup(patcher.stop)
        self.open_patch = patch.object(worker.os, "open", tracked_open)
        self.open_patch.start()
        self.addCleanup(self.open_patch.stop)
        patcher = patch.object(
            worker.ctypes, "CDLL", return_value=SimpleNamespace(prctl=lambda *a: 0)
        )
        patcher.start()
        self.addCleanup(patcher.stop)

    def run_worker(self, args, response=None):
        with (
            patch.object(worker.sys, "argv", ["worker", *args]),
            patch.object(worker, "urlopen", response or MagicMock()) as control,
            patch.object(worker.os, "execv", side_effect=NativeExec) as native,
        ):
            with self.assertRaises(NativeExec):
                worker.main()
            return control, native

    def test_introspection_does_not_evict_or_lock(self):
        control, native = self.run_worker(["--list-devices"])
        control.assert_not_called()
        self.assertEqual(self.opened, [])
        self.assertEqual(native.call_args.args[1][-1], "--list-devices")

    def test_native_exec_inherits_both_gpu_locks(self):
        control, _ = self.run_worker(["--model", "/tmp/any-model.gguf"])
        self.assertTrue(
            control.call_args.args[0].full_url.endswith(
                "/api/backends/sycl/release-idle"
            )
        )
        self.assertEqual(len(self.opened), 2)
        for fd in self.opened:
            self.assertTrue(os.get_inheritable(fd))
            with (
                Path(f"/proc/self/fd/{fd}").resolve().open("a") as other,
                self.assertRaises(BlockingIOError),
            ):
                fcntl.flock(other, fcntl.LOCK_EX | fcntl.LOCK_NB)

    def test_active_router_request_is_never_evicted(self):
        with (
            patch.object(worker.sys, "argv", ["worker", "--model", "x"]),
            patch.object(
                worker, "urlopen", side_effect=HTTPError("x", 409, "busy", {}, None)
            ),
            patch.object(worker.os, "execv") as native,
            self.assertRaises(SystemExit),
        ):
            worker.main()
        native.assert_not_called()
        self.assertEqual(self.opened, [])

    def test_media_lock_prevents_native_start(self):
        lock = self.home / "qwen3-tts-b60/gpu.lock"
        lock.parent.mkdir()
        with lock.open("a") as busy:
            fcntl.flock(busy, fcntl.LOCK_EX | fcntl.LOCK_NB)
            with (
                patch.object(worker.sys, "argv", ["worker", "--model", "x"]),
                patch.object(worker, "urlopen", MagicMock()),
                patch.object(worker.os, "execv") as native,
                self.assertRaises(SystemExit),
            ):
                worker.main()
            native.assert_not_called()


if __name__ == "__main__":
    unittest.main()
