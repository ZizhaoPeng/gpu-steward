import json
import os
import shutil
import stat
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
PLUGIN_ROOT = REPOSITORY_ROOT / "integrations" / "codex" / "gpu-steward-timeline"
MANIFEST_PATH = PLUGIN_ROOT / ".codex-plugin" / "plugin.json"
HOOKS_PATH = PLUGIN_ROOT / "hooks" / "hooks.json"
BRIDGE_PATH = PLUGIN_ROOT / "scripts" / "timeline-hook.py"
VALIDATOR = Path("/Users/pengzizhao/.codex/skills/.system/plugin-creator/scripts/validate_plugin.py")


class TimelinePluginTests(unittest.TestCase):
    def test_manifest_is_validation_ready_and_does_not_declare_unsupported_hooks(self):
        manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
        self.assertEqual("gpu-steward-timeline", manifest["name"])
        self.assertEqual("./skills/", manifest["skills"])
        self.assertNotIn("hooks", manifest)
        self.assertIn("GPU Steward", manifest["interface"]["displayName"])
        self.assertIn("本地", manifest["interface"]["shortDescription"])
        self.assertEqual(3, len(manifest["interface"]["defaultPrompt"]))

    def test_plugin_validator_passes(self):
        validator_python = sys.executable
        dependency_check = subprocess.run(
            [validator_python, "-c", "import yaml"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
        )
        if dependency_check.returncode != 0:
            validator_python = shutil.which("python") or validator_python
        result = subprocess.run(
            [validator_python, str(VALIDATOR), str(PLUGIN_ROOT)],
            cwd=str(REPOSITORY_ROOT),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            check=False,
        )
        self.assertEqual(0, result.returncode, result.stdout + result.stderr)
        self.assertIn("Plugin validation passed", result.stdout)

    def test_hooks_are_fast_commands_with_no_model_or_token_invocation(self):
        payload = json.loads(HOOKS_PATH.read_text(encoding="utf-8"))
        self.assertEqual(
            {"SessionStart", "UserPromptSubmit", "PreToolUse", "PostToolUse", "Stop"},
            set(payload["hooks"]),
        )
        for event_entries in payload["hooks"].values():
            for entry in event_entries:
                for hook in entry["hooks"]:
                    self.assertEqual("command", hook["type"])
                    command = hook["command"].lower()
                    self.assertIn("timeline-hook.py", command)
                    self.assertNotIn("token", command)
                    self.assertNotIn("codex exec", command)
                    self.assertLessEqual(hook["timeout"], 5)

    def test_hook_bridge_execs_cli_without_rewriting_stdin(self):
        payload = b'{"hook_event_name":"PreToolUse","session_id":"redacted"}\n'
        with tempfile.TemporaryDirectory() as directory:
            directory_path = Path(directory)
            capture = directory_path / "capture.bin"
            argv_capture = directory_path / "argv.json"
            fake = directory_path / "gpu-steward"
            fake.write_text(
                "#!/usr/bin/env python3\n"
                "import json, os, sys\n"
                "open(os.environ['CAPTURE'], 'wb').write(sys.stdin.buffer.read())\n"
                "json.dump(sys.argv[1:], open(os.environ['ARGV'], 'w'))\n",
                encoding="utf-8",
            )
            fake.chmod(fake.stat().st_mode | stat.S_IXUSR)
            environment = os.environ.copy()
            environment["PATH"] = str(directory_path) + os.pathsep + environment.get("PATH", "")
            environment["CAPTURE"] = str(capture)
            environment["ARGV"] = str(argv_capture)
            result = subprocess.run(
                [sys.executable, str(BRIDGE_PATH)],
                input=payload,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                env=environment,
                check=False,
            )
            self.assertEqual(0, result.returncode, result.stderr)
            self.assertEqual(b"", result.stdout)
            self.assertEqual(b"", result.stderr)
            self.assertEqual(payload, capture.read_bytes())
            self.assertEqual(["timeline", "hook"], json.loads(argv_capture.read_text()))

    def test_skill_explains_observe_plane_and_frozen_phase_boundary(self):
        contents = (PLUGIN_ROOT / "skills" / "gpu-steward-timeline" / "SKILL.md").read_text(
            encoding="utf-8"
        )
        self.assertTrue(contents.startswith("---\n"))
        self.assertIn("Observe Plane", contents)
        self.assertIn("gpu-steward timeline phase research", contents)
        self.assertIn("active-unspecified", contents)
        self.assertIn("gpu-steward timeline open", contents)
        self.assertIn("提供通过本地", contents.split("---", 2)[1])


if __name__ == "__main__":
    unittest.main()
