import os
import stat
import tempfile
import unittest

from gpu_steward.timeline.store import (
    TimelineConflictError,
    TimelineSchemaError,
    TimelineStore,
)


class TimelineStoreTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.path = os.path.join(self.tmp.name, "timeline.sqlite3")
        self.store = TimelineStore(self.path, salt=b"test-local-salt")

    def tearDown(self):
        self.store.close()
        self.tmp.cleanup()

    def test_private_file_and_schema_are_isolated(self):
        self.assertEqual(1, self.store.schema_version)
        mode = stat.S_IMODE(os.stat(self.path).st_mode)
        self.assertEqual(0o600, mode)
        tables = {
            row[0]
            for row in self.store.query(
                "SELECT name FROM sqlite_master WHERE type='table'"
            )
        }
        self.assertIn("codex_events", tables)
        self.assertIn("gpu_samples", tables)
        self.assertIn("timeline_overrides", tables)
        self.assertNotIn("tasks", tables)

    def test_codex_event_hashes_identifiers_and_is_idempotent(self):
        event_id = self.store.record_codex_event(
            "hook-1",
            100.0,
            "session-secret",
            "turn-secret",
            "/private/project-name",
            "SessionStart",
            "active-unspecified",
            "hook-rule",
            0.8,
        )
        self.assertEqual("hook-1", event_id)
        self.assertEqual(event_id, self.store.record_codex_event(
            "hook-1",
            100.0,
            "session-secret",
            "turn-secret",
            "/private/project-name",
            "session-start",
            "active-unspecified",
            "hook-rule",
            0.8,
        ))
        row = self.store.list_codex_events()[0]
        self.assertNotIn("secret", repr(row))
        self.assertEqual("project-name", row["project"])
        self.assertEqual(16, len(row["session_hash"]))
        with self.assertRaises(TimelineConflictError):
            self.store.record_codex_event(
                "hook-1",
                101.0,
                "session-secret",
                "turn-secret",
                "/private/project-name",
                "SessionStart",
            )

    def test_unknown_schema_and_sensitive_labels_fail_closed(self):
        with self.assertRaises(TimelineSchemaError):
            self.store.record_codex_event(
                "bad", 1, "s", "t", "project", "made-up-event"
            )
        with self.assertRaises(TimelineSchemaError):
            self.store.record_gpu_sample(
                1, "AI3", 0, "uuid", "idle", task_name="a\nprompt"
            )

    def test_gpu_sample_and_override_are_append_only(self):
        sample_id = self.store.record_gpu_sample(
            10,
            "AI3",
            0,
            "GPU-0-short",
            "idle",
            task_name="demo",
            attribution="inferred",
            process_basename="python",
            pid=42,
        )
        override_id = self.store.record_override(
            "gpu_sample", sample_id, "state", "training", created_at=20
        )
        self.assertEqual({"state": "training"}, self.store.current_overrides(
            "gpu_sample", sample_id, at=21
        ))
        self.assertEqual({}, self.store.current_overrides(
            "gpu_sample", sample_id, at=19
        ))
        with self.assertRaises(Exception):
            self.store.connection.execute(
                "UPDATE gpu_samples SET state='training' WHERE sample_id=?", (sample_id,)
            )
        with self.assertRaises(Exception):
            self.store.connection.execute(
                "DELETE FROM timeline_overrides WHERE override_id=?", (override_id,)
            )

    def test_direct_mutation_query_is_rejected(self):
        with self.assertRaises(TimelineSchemaError):
            self.store.query("UPDATE codex_events SET phase='idle'")

    def test_sample_like_object_is_accepted_by_collector_adapter(self):
        class Observation:
            sampled_at = 12
            host = "AI3"
            gpu_index = 1
            gpu_uuid_short = "u1"
            state = "idle"
            task_name = None
            attribution = "none"
            process_basename = None
            pid = None

        self.assertTrue(self.store.record_gpu_sample(Observation()))


if __name__ == "__main__":
    unittest.main()
