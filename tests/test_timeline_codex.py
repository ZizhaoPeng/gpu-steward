import json
import os
import tempfile
import unittest

from gpu_steward.timeline.codex import (
    CodexHookIngestor,
    ingest_hook,
    normalize_hook,
)
from gpu_steward.timeline.store import TimelineSchemaError, TimelineStore


class CodexHookTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.store = TimelineStore(
            os.path.join(self.tmp.name, "timeline.sqlite3"), salt=b"hook-salt"
        )

    def tearDown(self):
        self.store.close()
        self.tmp.cleanup()

    def test_lifecycle_hooks_are_allowlisted_and_idempotent(self):
        common = {
            "session_id": "session-private",
            "turn_id": "turn-private",
            "cwd": "/private/ProjectA",
        }
        start = dict(common, hook_event_name="SessionStart", timestamp=100, id="start-1")
        phase = dict(
            common,
            hook_event_name="phase",
            timestamp=110,
            id="phase-1",
            phase="research",
            source="declared",
            confidence=1,
        )
        pre = dict(
            common,
            hook_event_name="PreToolUse",
            timestamp=120,
            id="pre-1",
            tool_name="Bash",
        )
        post = dict(
            common,
            hook_event_name="PostToolUse",
            timestamp=130,
            id="post-1",
            tool_name="Bash",
        )
        stop = dict(common, hook_event_name="Stop", timestamp=140, id="stop-1")
        for payload in (start, phase, pre, post, stop):
            self.assertTrue(ingest_hook(payload, self.store))
        self.assertEqual(5, len(self.store.list_codex_events()))
        self.assertEqual(ingest_hook(start, self.store), "start-1")
        self.assertEqual(5, len(self.store.list_codex_events()))
        rows = self.store.list_codex_events()
        self.assertEqual(
            ["session-start", "phase", "pre-tool", "post-tool", "stop"],
            [row["kind"] for row in rows],
        )
        self.assertEqual("waiting-tool", rows[2]["phase"])
        self.assertTrue(rows[2]["tool_active"])
        self.assertEqual("waiting-user", rows[-1]["phase"])
        self.assertEqual("ProjectA", rows[0]["project"])
        self.assertNotIn("private", repr(rows))

    def test_prompt_response_and_command_are_never_returned(self):
        payload = {
            "hook_event_name": "UserPromptSubmit",
            "timestamp": "1970-01-01T00:02:00Z",
            "session_id": "session",
            "turn_id": "turn",
            "cwd": "/tmp/ProjectB",
            "prompt": "secret prompt must not be retained",
            "response": "secret response must not be retained",
            "command": "rm -rf /",
            "env": {"TOKEN": "do-not-store"},
        }
        safe = normalize_hook(payload)
        self.assertEqual(120, safe["occurred_at"])
        for forbidden in ("prompt", "response", "command", "env"):
            self.assertNotIn(forbidden, safe)
        ingest_hook(payload, self.store)
        self.assertNotIn("secret", repr(self.store.list_codex_events()))

    def test_unknown_hook_and_bad_phase_fail_closed(self):
        with self.assertRaises(TimelineSchemaError):
            normalize_hook({"hook_event_name": "UnknownHook"})
        with self.assertRaises(TimelineSchemaError):
            normalize_hook({"hook_event_name": "phase", "phase": "thinking"})
        with self.assertRaises(TimelineSchemaError):
            CodexHookIngestor(self.store).ingest_line("not-json")

    def test_nested_hook_metadata_and_safe_tool_category(self):
        ingestor = CodexHookIngestor(self.store)
        event_id = ingestor.ingest(
            {
                "event": "PreToolUse",
                "data": {
                    "sessionId": "s",
                    "turnId": "t",
                    "timestamp": "1970-01-01T00:03:00+00:00",
                    "cwd": "/tmp/ProjectC",
                    "tool_name": "arbitrary-command-with-secret",
                    "id": "nested-1",
                },
            }
        )
        self.assertEqual("nested-1", event_id)
        row = self.store.list_codex_events()[0]
        self.assertEqual("other", row["tool_category"])


if __name__ == "__main__":
    unittest.main()
