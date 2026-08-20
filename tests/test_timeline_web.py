import json
import threading
import unittest
from urllib.error import HTTPError
from urllib.request import urlopen

from gpu_steward.timeline.web import (
    REPORT_SCHEMA_VERSION,
    TimelineWebError,
    create_server,
)


def report_for(date, timezone, project):
    return {
        "schema_version": REPORT_SCHEMA_VERSION,
        "date": date,
        "timezone": timezone,
        "generated_at": 1723939200,
        "display_merge_gap_seconds": 600,
        "lanes": [
            {
                "id": "codex:session",
                "kind": "codex",
                "label": "Codex",
                "segments": [
                    {
                        "start": 1723939200,
                        "end": 1723942800,
                        "label": "implement",
                        "phase": "implement",
                        "source": "declared",
                        "confidence": 1.0,
                        "task_name": project or "gpu-steward",
                    },
                    {
                        "start": 1723942800,
                        "end": 1723944600,
                        "label": "inferred wait",
                        "phase": "waiting-tool",
                        "source": "inferred",
                        "confidence": 0.6,
                    },
                ],
            },
            {
                "id": "gpu:AI3:0",
                "kind": "gpu",
                "label": "AI3 / GPU 0",
                "segments": [
                    {
                        "start": 1723939800,
                        "end": 1723943400,
                        "label": "training",
                        "state": "training",
                        "source": "hook-rule",
                        "confidence": 0.9,
                        "observed_seconds": 3300,
                        "gap_seconds": 300,
                        "gap_count": 1,
                    },
                    {
                        "start": 1723943400,
                        "end": 1723945200,
                        "label": "unknown",
                        "state": "unknown",
                        "source": "inferred",
                        "confidence": 0.0,
                    },
                ],
            },
        ],
        "summary": {
            "codex_active_seconds": 3600,
            "codex_waiting_seconds": 1800,
            "codex_stalled_seconds": 0,
            "gpu_training_seconds": 3600,
            "gpu_idle_seconds": 0,
            "overlap_seconds": 3000,
        },
    }


class TimelineWebTests(unittest.TestCase):
    def setUp(self):
        self.calls = []

        def builder(*, date, timezone, project):
            self.calls.append((date, timezone, project))
            return report_for(date, timezone, project)

        self.server = create_server(builder, port=0)
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()
        self.base_url = "http://127.0.0.1:{}".format(self.server.server_port)

    def tearDown(self):
        self.server.shutdown()
        self.thread.join(timeout=5)
        self.server.server_close()

    def get(self, path):
        with urlopen(self.base_url + path, timeout=5) as response:
            return response.status, response.headers, response.read()

    def test_localhost_server_serves_static_dashboard_without_external_assets(self):
        status, headers, body = self.get("/")
        html = body.decode("utf-8")
        self.assertEqual(200, status)
        self.assertEqual("no-store", headers["Cache-Control"])
        self.assertIn("GPU Steward", html)
        self.assertIn("/assets/app.js", html)
        self.assertNotIn("https://", html)
        _, _, css = self.get("/assets/styles.css")
        _, _, js = self.get("/assets/app.js")
        self.assertIn(b"gpu-disabled", css)
        self.assertIn(b"gpu-unknown", css)
        self.assertIn(b"inferred", css)
        self.assertIn(b"/api/report", js)
        self.assertIn(b"showDetail", js)
        self.assertIn(b"merged-gap", css + js)
        self.assertIn("相同任务自动合成长条", html)
        self.assertNotIn(b"https://", css + js)

    def test_implicit_favicon_request_does_not_pollute_browser_console(self):
        status, _, body = self.get("/favicon.ico")
        self.assertEqual(204, status)
        self.assertEqual(b"", body)

    def test_report_api_passes_date_and_project_and_keeps_summary(self):
        status, headers, body = self.get(
            "/api/report?date=2026-08-18&project=My_Paper_3rd"
        )
        payload = json.loads(body.decode("utf-8"))
        self.assertEqual(200, status)
        self.assertEqual("application/json; charset=utf-8", headers["Content-Type"])
        self.assertEqual(
            [("2026-08-18", "Asia/Singapore", "My_Paper_3rd")], self.calls
        )
        self.assertEqual(3600, payload["summary"]["gpu_training_seconds"])
        self.assertEqual("inferred", payload["lanes"][0]["segments"][1]["source"])

    def test_health_is_local_and_versioned(self):
        status, _, body = self.get("/api/health")
        payload = json.loads(body.decode("utf-8"))
        self.assertEqual(200, status)
        self.assertEqual(1, payload["schema_version"])
        self.assertTrue(payload["ok"])
        self.assertEqual("observe", payload["plane"])

    def test_invalid_date_and_unknown_path_fail_closed(self):
        with self.assertRaises(HTTPError) as invalid:
            urlopen(self.base_url + "/api/report?date=not-a-date", timeout=5)
        self.assertEqual(400, invalid.exception.code)
        self.assertNotIn("Traceback", invalid.exception.read().decode("utf-8"))

        with self.assertRaises(HTTPError) as missing:
            urlopen(self.base_url + "/nope", timeout=5)
        self.assertEqual(404, missing.exception.code)

    def test_invalid_report_is_rejected_before_json_response(self):
        def bad_builder(*, date, timezone, project):
            payload = report_for(date, timezone, project)
            payload["lanes"][0]["segments"][0]["source"] = "guessed"
            return payload

        server = create_server(bad_builder, port=0)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            with self.assertRaises(HTTPError) as failure:
                urlopen(
                    "http://127.0.0.1:{}/api/report?date=2026-08-18".format(
                        server.server_port
                    ),
                    timeout=5,
                )
            self.assertEqual(400, failure.exception.code)
        finally:
            server.shutdown()
            thread.join(timeout=5)
            server.server_close()

    def test_invalid_merged_gap_metadata_is_rejected(self):
        payload = report_for("2026-08-18", "Asia/Singapore", None)
        payload["lanes"][1]["segments"][0]["gap_count"] = -1
        with self.assertRaises(TimelineWebError):
            from gpu_steward.timeline.web import validate_report
            validate_report(payload, requested_date="2026-08-18", timezone="Asia/Singapore")

    def test_non_local_bind_is_rejected(self):
        with self.assertRaises(ValueError):
            create_server(report_for, host="0.0.0.0", port=0)


if __name__ == "__main__":
    unittest.main()
