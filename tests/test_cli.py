import json
import os
import subprocess
import sys
import tempfile
import textwrap
import unittest


class CLITests(unittest.TestCase):
    def make_fake_nvidia_smi(self, directory, fail=False):
        path = os.path.join(directory, "nvidia-smi")
        source = """
            #!{python}
            import sys
            if {fail!r}:
                print("synthetic probe failure", file=sys.stderr)
                raise SystemExit(7)
            query = " ".join(sys.argv[1:])
            if "--query-gpu=" in query:
                for index in range(4):
                    print("{{}}, GPU-{{}}, Fake GPU, 00000000:{{:02x}}:00.0, 24564".format(index, index, index))
            elif "--query-compute-apps=" not in query:
                raise SystemExit(2)
        """.format(python=sys.executable, fail=fail)
        with open(path, "w", encoding="utf-8") as handle:
            handle.write(textwrap.dedent(source).lstrip())
        os.chmod(path, 0o700)
        return path

    def run_cli(self, directory, *args):
        environment = os.environ.copy()
        environment["PATH"] = directory + os.pathsep + environment.get("PATH", "")
        source_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src"))
        environment["PYTHONPATH"] = source_root
        return subprocess.run(
            [sys.executable, "-m", "gpu_steward.cli"] + list(args),
            cwd=directory,
            env=environment,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            check=False,
        )

    def test_doctor_and_run_emit_versioned_json(self):
        with tempfile.TemporaryDirectory() as directory:
            self.make_fake_nvidia_smi(directory)
            database = os.path.join(directory, "state.sqlite3")
            doctor = self.run_cli(
                directory, "--db", database, "doctor", "--json"
            )
            self.assertEqual(0, doctor.returncode, doctor.stderr)
            doctor_payload = json.loads(doctor.stdout)
            self.assertEqual(1, doctor_payload["schema_version"])
            self.assertTrue(doctor_payload["ok"])
            self.assertEqual(4, doctor_payload["inventory"]["count"])

            child = self.run_cli(
                directory,
                "--db",
                database,
                "run",
                "--json",
                "--min",
                "1",
                "--max",
                "1",
                "--",
                sys.executable,
                "-c",
                "print('child-ok')",
            )
            self.assertEqual(0, child.returncode, child.stderr)
            self.assertEqual("child-ok", child.stdout.strip())
            result = json.loads(child.stderr)
            self.assertEqual(1, result["schema_version"])
            self.assertTrue(result["ok"])
            self.assertEqual(1, len(result["task"]["gpu_uuids"]))

    def test_probe_failure_returns_json_error_and_nonzero(self):
        with tempfile.TemporaryDirectory() as directory:
            self.make_fake_nvidia_smi(directory, fail=True)
            result = self.run_cli(
                directory,
                "--db",
                os.path.join(directory, "state.sqlite3"),
                "inventory",
                "--json",
            )
            self.assertEqual(2, result.returncode)
            payload = json.loads(result.stderr)
            self.assertFalse(payload["ok"])
            self.assertEqual(1, payload["schema_version"])
            self.assertEqual("InventoryError", payload["error"]["type"])

    def test_invalid_gpu_request_returns_json_without_traceback(self):
        with tempfile.TemporaryDirectory() as directory:
            self.make_fake_nvidia_smi(directory)
            result = self.run_cli(
                directory,
                "--db",
                os.path.join(directory, "state.sqlite3"),
                "run",
                "--json",
                "--min",
                "0",
                "--",
                sys.executable,
                "-c",
                "pass",
            )
            self.assertEqual(2, result.returncode)
            self.assertNotIn("Traceback", result.stderr)
            payload = json.loads(result.stderr)
            self.assertFalse(payload["ok"])
            self.assertEqual(1, payload["schema_version"])
            self.assertEqual("QueueError", payload["error"]["type"])


if __name__ == "__main__":
    unittest.main()
