import io
import json
import os
import tempfile
import unittest
from contextlib import redirect_stdout
from unittest import mock

from gpu_steward.cli import main
from gpu_steward.timeline.store import TimelineStore


class TimelineCLITests(unittest.TestCase):
    def test_init_writes_private_config_and_refuses_overwrite(self):
        with tempfile.TemporaryDirectory() as directory:
            path = os.path.join(directory, "timeline.json")
            output = io.StringIO()
            with redirect_stdout(output):
                self.assertEqual(
                    0,
                    main([
                        "timeline", "init", "--config", path, "--host", "AI3",
                        "--project", "My_Paper_3rd", "--disabled-gpu", "2",
                    ]),
                )
            self.assertEqual(0o600, os.stat(path).st_mode & 0o777)
            with open(path, encoding="utf-8") as handle:
                payload = json.load(handle)
            self.assertEqual([2], payload["hosts"][0]["disabled_gpu_indices"])
            with mock.patch("sys.stderr", new=io.StringIO()):
                self.assertEqual(2, main(["timeline", "init", "--config", path]))

    def test_hook_is_silent_and_persists_only_safe_metadata(self):
        with tempfile.TemporaryDirectory() as directory:
            db = os.path.join(directory, "timeline.sqlite3")
            hook = {
                "hook_event_name": "SessionStart",
                "session_id": "s1",
                "cwd": "/secret/Demo",
                "prompt": "do not store",
            }
            output = io.StringIO()
            with mock.patch("sys.stdin", io.StringIO(json.dumps(hook))), redirect_stdout(output):
                self.assertEqual(0, main(["timeline", "hook", "--timeline-db", db]))
            self.assertEqual("", output.getvalue())
            store = TimelineStore(db)
            try:
                events = store.list_codex_events()
            finally:
                store.close()
            self.assertEqual("Demo", events[0]["project"])
            self.assertNotIn("prompt", events[0])

    def test_phase_and_json_report(self):
        with tempfile.TemporaryDirectory() as directory:
            db = os.path.join(directory, "timeline.sqlite3")
            with redirect_stdout(io.StringIO()):
                self.assertEqual(
                    0,
                    main([
                        "timeline", "phase", "research", "--timeline-db", db,
                        "--project", "Demo",
                    ]),
                )
            output = io.StringIO()
            with redirect_stdout(output):
                self.assertEqual(
                    0,
                    main(["timeline", "report", "--timeline-db", db, "--date", "2026-08-18"]),
                )
            report = json.loads(output.getvalue())
            self.assertEqual(1, report["schema_version"])
            self.assertEqual("2026-08-18", report["date"])

    def test_open_reuses_healthy_dashboard_without_launching_another_service(self):
        output = io.StringIO()
        with mock.patch(
            "gpu_steward.timeline.cli._dashboard_available", return_value=True
        ), mock.patch(
            "gpu_steward.timeline.cli._install_dashboard_agent"
        ) as install, mock.patch(
            "gpu_steward.timeline.cli.webbrowser.open", return_value=True
        ) as browser, redirect_stdout(output):
            self.assertEqual(0, main(["timeline", "open"]))
        install.assert_not_called()
        browser.assert_called_once_with("http://127.0.0.1:8765/", new=0, autoraise=True)
        payload = json.loads(output.getvalue())
        self.assertTrue(payload["ok"])
        self.assertFalse(payload["started"])
        self.assertTrue(payload["browser_opened"])

    def test_open_installs_dashboard_and_waits_for_health_before_browser(self):
        output = io.StringIO()
        with mock.patch(
            "gpu_steward.timeline.cli._dashboard_available", side_effect=[False, True]
        ), mock.patch(
            "gpu_steward.timeline.cli._install_dashboard_agent",
            return_value={"ok": True, "started": True, "plist": "/tmp/dashboard.plist"},
        ) as install, mock.patch(
            "gpu_steward.timeline.cli.webbrowser.open", return_value=True
        ) as browser, redirect_stdout(output):
            self.assertEqual(
                0,
                main(["timeline", "open", "--config", "/tmp/timeline.json"]),
            )
        install.assert_called_once_with(
            config_path="/tmp/timeline.json",
            plist_path=None,
            executable=None,
            project=None,
        )
        browser.assert_called_once_with("http://127.0.0.1:8765/", new=0, autoraise=True)
        payload = json.loads(output.getvalue())
        self.assertTrue(payload["ok"])
        self.assertTrue(payload["started"])

    def test_dashboard_install_uses_fixed_dashboard_manager(self):
        output = io.StringIO()
        with mock.patch(
            "gpu_steward.timeline.cli._install_dashboard_agent",
            return_value={"ok": True, "action": "installed", "started": True},
        ) as install, redirect_stdout(output):
            self.assertEqual(
                0,
                main([
                    "timeline", "dashboard", "install", "--config", "/tmp/timeline.json",
                    "--plist", "/tmp/dashboard.plist",
                ]),
            )
        install.assert_called_once_with(
            config_path="/tmp/timeline.json",
            plist_path="/tmp/dashboard.plist",
            executable=None,
            project=None,
        )
        self.assertTrue(json.loads(output.getvalue())["ok"])

    def test_dashboard_health_requires_our_versioned_local_service(self):
        class Response:
            status = 200

            def read(self, _size):
                return json.dumps({
                    "ok": True,
                    "service": "gpu-steward-timeline",
                    "plane": "observe",
                    "schema_version": 1,
                }).encode("utf-8")

            def close(self):
                pass

        with mock.patch("gpu_steward.timeline.cli.urllib.request.urlopen", return_value=Response()):
            from gpu_steward.timeline.cli import _dashboard_available
            self.assertTrue(_dashboard_available())
        Response.read = lambda self, _size: b'{"ok":true,"service":"unrelated"}'
        with mock.patch("gpu_steward.timeline.cli.urllib.request.urlopen", return_value=Response()):
            self.assertFalse(_dashboard_available())


if __name__ == "__main__":
    unittest.main()
