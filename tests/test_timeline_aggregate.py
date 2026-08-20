import os
import tempfile
import unittest

from gpu_steward.timeline.aggregate import TimelineAggregator, build_day
from gpu_steward.timeline.store import TimelineStore


class TimelineAggregateTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.store = TimelineStore(
            os.path.join(self.tmp.name, "timeline.sqlite3"), salt=b"aggregate-salt"
        )

    def tearDown(self):
        self.store.close()
        self.tmp.cleanup()

    def add_event(self, event_id, at, kind, phase="active-unspecified", **kwargs):
        return self.store.record_codex_event(
            event_id,
            at,
            "session-1",
            kwargs.pop("turn_id", "turn-1"),
            kwargs.pop("project", "/tmp/Demo"),
            kind,
            phase,
            kwargs.pop("source", "hook-rule"),
            kwargs.pop("confidence", 0.8),
            kwargs.pop("tool_category", ""),
            kwargs.pop("tool_active", False),
        )

    def test_half_open_segments_cross_midnight_and_summary(self):
        # Use UTC for a deterministic epoch boundary; the same clipping logic
        # is used for Asia/Singapore in production reports.
        day_start = 24 * 3600
        self.add_event("start", day_start - 60, "SessionStart")
        self.add_event("phase", day_start + 60, "phase", "research", source="declared", confidence=1)
        self.add_event("stop", day_start + 180, "Stop")
        self.store.record_gpu_sample(day_start - 30, "AI3", 0, "u0", "training", task_name="job")
        self.store.record_gpu_sample(day_start + 120, "AI3", 0, "u0", "idle")
        report = build_day(
            self.store,
            "1970-01-02",
            timezone="UTC",
            generated_at=day_start + 300,
        )
        self.assertEqual(1, report["schema_version"])
        self.assertEqual("1970-01-02", report["date"])
        self.assertEqual(180, report["summary"]["codex_active_seconds"])
        self.assertEqual(120, report["summary"]["gpu_training_seconds"])
        self.assertEqual(180, report["summary"]["gpu_idle_seconds"])
        self.assertEqual(120, report["summary"]["overlap_seconds"])
        for lane in report["lanes"]:
            last = None
            for segment in lane["segments"]:
                self.assertLess(segment["start"], segment["end"])
                self.assertGreaterEqual(segment["start"], day_start)
                self.assertLessEqual(segment["end"], day_start + 86400)
                if last is not None:
                    self.assertLessEqual(last, segment["start"])
                last = segment["end"]

    def test_parallel_sessions_and_gpus_do_not_overlap_within_lane(self):
        self.add_event("s1", 0, "SessionStart")
        self.add_event("s1-stop", 100, "Stop")
        self.store.record_codex_event(
            "s2", 50, "session-2", "turn-2", "/tmp/Demo", "SessionStart"
        )
        self.store.record_codex_event(
            "s2-stop", 150, "session-2", "turn-2", "/tmp/Demo", "Stop"
        )
        self.store.record_gpu_sample(0, "AI3", 0, "u0", "training")
        self.store.record_gpu_sample(100, "AI3", 0, "u0", "idle")
        self.store.record_gpu_sample(0, "AI3", 1, "u1", "training")
        self.store.record_gpu_sample(100, "AI3", 1, "u1", "idle")
        report = TimelineAggregator(self.store).build_day(
            "1970-01-01", timezone="UTC", generated_at=200
        )
        self.assertEqual(200, report["summary"]["codex_active_seconds"])
        self.assertEqual(200, report["summary"]["gpu_training_seconds"])
        self.assertEqual(100, report["summary"]["overlap_seconds"])
        self.assertEqual(4, len(report["lanes"]))

    def test_stall_after_ten_minutes_but_not_during_long_tool(self):
        self.add_event("start", 0, "SessionStart")
        report = build_day(self.store, "1970-01-01", timezone="UTC", generated_at=700)
        codex = next(lane for lane in report["lanes"] if lane["kind"] == "codex")
        self.assertEqual(["active-unspecified", "suspected-stall"], [
            segment["phase"] for segment in codex["segments"]
        ])
        self.assertEqual(100, report["summary"]["codex_stalled_seconds"])

        self.store.record_codex_event(
            "pre", 800, "session-2", "turn-2", "/tmp/Demo", "PreToolUse", "waiting-tool", "hook-rule", 0.8, "shell", True
        )
        report = build_day(self.store, "1970-01-01", timezone="UTC", generated_at=1500)
        codex_lanes = [lane for lane in report["lanes"] if lane["kind"] == "codex"]
        tool_lane = next(lane for lane in codex_lanes if any(
            segment.get("tool_active") for segment in lane["segments"]
        ))
        self.assertFalse(any(seg["phase"] == "suspected-stall" for seg in tool_lane["segments"]))

    def test_missing_sample_gap_becomes_unknown_and_override_applies(self):
        sample_id = self.store.record_gpu_sample(0, "AI3", 0, "u0", "idle")
        self.store.record_gpu_sample(1200, "AI3", 0, "u0", "training")
        self.store.record_override("gpu_sample", sample_id, "state", "disabled", created_at=1300)
        report = build_day(
            self.store,
            "1970-01-01",
            timezone="UTC",
            generated_at=1800,
            sample_gap_seconds=600,
        )
        lane = next(lane for lane in report["lanes"] if lane["id"] == "gpu:AI3:0")
        self.assertEqual(["disabled", "unknown", "training"], [
            segment["state"] for segment in lane["segments"]
        ])

    def test_last_successful_sample_does_not_mask_a_long_missing_gap(self):
        self.store.record_gpu_sample(0, "AI3", 0, "u0", "training")
        report = build_day(
            self.store,
            "1970-01-01",
            timezone="UTC",
            generated_at=1200,
            sample_gap_seconds=600,
        )
        lane = next(lane for lane in report["lanes"] if lane["id"] == "gpu:AI3:0")
        self.assertEqual(["training", "unknown"], [
            segment["state"] for segment in lane["segments"]
        ])
        self.assertEqual(600, report["summary"]["gpu_training_seconds"])

    def test_repeated_samples_become_one_readable_task_bar(self):
        for at in (0, 60, 120, 180):
            self.store.record_gpu_sample(
                at, "AI3", 0, "u0", "external",
                task_name="python3.10" if at < 120 else "python3",
                attribution="inferred", process_basename="python3.10" if at < 120 else "python3",
                pid=10 + at,
            )
        self.store.record_gpu_sample(240, "AI3", 0, "u0", "idle")
        report = build_day(self.store, "1970-01-01", timezone="UTC", generated_at=300)
        lane = next(lane for lane in report["lanes"] if lane["id"] == "gpu:AI3:0")
        self.assertEqual(2, len(lane["segments"]))
        occupied = lane["segments"][0]
        self.assertEqual("Python 训练/计算进程", occupied["task_name"])
        self.assertEqual(240, occupied["duration_seconds"])
        self.assertEqual(4, occupied["sample_count"])
        self.assertEqual(0, occupied["gap_count"])

    def test_short_idle_gap_is_display_merged_but_not_added_to_training_total(self):
        self.store.record_gpu_sample(0, "AI3", 0, "u0", "training", task_name="job-A", attribution="explicit")
        self.store.record_gpu_sample(300, "AI3", 0, "u0", "idle")
        self.store.record_gpu_sample(420, "AI3", 0, "u0", "training", task_name="job-A", attribution="explicit")
        self.store.record_gpu_sample(720, "AI3", 0, "u0", "idle")
        report = build_day(self.store, "1970-01-01", timezone="UTC", generated_at=900)
        lane = next(lane for lane in report["lanes"] if lane["id"] == "gpu:AI3:0")
        occupied = lane["segments"][0]
        self.assertEqual((0, 720), (occupied["start"], occupied["end"]))
        self.assertEqual(120, occupied["gap_seconds"])
        self.assertEqual(1, occupied["gap_count"])
        self.assertEqual(600, occupied["observed_seconds"])
        self.assertEqual(600, report["summary"]["gpu_training_seconds"])

    def test_long_idle_gap_remains_visible(self):
        self.store.record_gpu_sample(0, "AI3", 0, "u0", "training", task_name="job-A", attribution="explicit")
        self.store.record_gpu_sample(60, "AI3", 0, "u0", "idle")
        self.store.record_gpu_sample(900, "AI3", 0, "u0", "training", task_name="job-A", attribution="explicit")
        self.store.record_gpu_sample(960, "AI3", 0, "u0", "idle")
        report = build_day(self.store, "1970-01-01", timezone="UTC", generated_at=1000)
        lane = next(lane for lane in report["lanes"] if lane["id"] == "gpu:AI3:0")
        states = [segment["state"] for segment in lane["segments"]]
        self.assertEqual(2, states.count("training"))
        self.assertIn("idle", states)


if __name__ == "__main__":
    unittest.main()
