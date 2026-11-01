"""Exercise the exec/config boundary without a GPU or a Strata installation."""
import json
import pathlib
import subprocess
import sys
import tempfile
import unittest


class LauncherTest(unittest.TestCase):
    def test_exec_keeps_assigned_port_sampling_env_and_checks_model_context(self):
        launcher = pathlib.Path(__file__).resolve().parents[1] / "scripts/strata-server"
        with tempfile.TemporaryDirectory() as directory:
            root = pathlib.Path(directory)
            (root / "serve").mkdir()
            (root / "serve/server.py").write_text(
                "import json,sys,os\n"
                "p=sys.argv[sys.argv.index('--config')+1]\n"
                "x=json.load(open(p)); x['test_env']=os.environ['STRATA_KEEP_EMPTY_TURNS']\n"
                "print(json.dumps(x))\n"
            )
            native = root / "model.gguf"
            native.touch()
            config = root / "config.json"
            original = {"cwd": str(root), "python": sys.executable,
                        "args": ["--native", str(native), "--max-context", "262144"],
                        "host": "0.0.0.0", "lazy_load": True,
                        "sampling": {"top_p": .95},
                        "env": {"STRATA_KEEP_EMPTY_TURNS": "1"}}
            config.write_text(json.dumps(original))
            args = [sys.executable, str(launcher), "--config", str(config),
                    "--native", str(native), "--context", "262144", "--port", "9193",
                    "--model-name", "flash", "--gpu", "0", "--sampling", '{"temperature":0.7}']
            result = subprocess.run(args, text=True, capture_output=True, check=True)
            actual = json.loads(result.stdout)
            self.assertEqual((actual["host"], actual["port"], actual["model_name"]),
                             ("127.0.0.1", 9193, "flash"))
            self.assertEqual(actual["sampling"], {"top_p": .95, "temperature": .7})
            self.assertEqual(actual["test_env"], "1")
            self.assertFalse(actual["lazy_load"])
            self.assertFalse(actual["fit_max_tokens"])
            self.assertEqual(json.loads(config.read_text()), original)
            args[args.index("--context") + 1] = "131072"
            failed = subprocess.run(args, text=True, capture_output=True)
            self.assertNotEqual(failed.returncode, 0)
            self.assertIn("router context and Strata --max-context disagree", failed.stderr)


if __name__ == "__main__":
    unittest.main()
