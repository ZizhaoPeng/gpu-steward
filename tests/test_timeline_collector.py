import unittest

from gpu_steward.timeline.collector import CollectorLoop, ExponentialBackoff
from gpu_steward.timeline.gpu import GPUSample


def sample(host="AI3", state="external", timestamp=1):
    return GPUSample(timestamp, host, 0, "GPU-a", state)


class FakeProbe:
    def __init__(self, values):
        self.values = list(values)
        self.host = "AI3"
        self.last_ok = True

    def sample(self, sampled_at=None):
        value = self.values.pop(0)
        if isinstance(value, Exception):
            self.last_ok = False
            raise value
        self.last_ok = value[0]
        return value[1]


class TimelineCollectorTests(unittest.TestCase):
    def test_backoff_is_1_2_5_10_minutes_and_resets(self):
        backoff = ExponentialBackoff()
        self.assertEqual((60.0, 120.0, 300.0, 600.0), tuple(backoff.failure_delay() for _ in range(4)))
        self.assertEqual(600.0, backoff.failure_delay())
        backoff.reset()
        self.assertEqual(60.0, backoff.failure_delay())

    def test_collects_without_model_and_records_unknown_failures(self):
        rows = []
        sleeps = []
        probe = FakeProbe(
            [
                (True, [sample(timestamp=10)]),
                (False, [GPUSample(20, "AI3", None, "", "unknown", attribution="unknown", error="probe failed")]),
                (True, [sample(timestamp=30)]),
            ]
        )
        collector = CollectorLoop(
            probe,
            rows,
            clock=lambda: 10,
            sleep=sleeps.append,
        )
        results = collector.run_iterations(3)
        self.assertEqual(3, len(results))
        self.assertEqual([60.0, 60.0], sleeps)
        self.assertEqual([True, False, True], [result.ok for result in results])
        self.assertEqual("unknown", rows[1].state)
        self.assertEqual(60.0, results[0].next_delay_seconds)
        self.assertEqual(60.0, results[1].next_delay_seconds)
        self.assertEqual(60.0, results[2].next_delay_seconds)

    def test_fully_idle_or_disabled_host_relaxes_to_low_cpu_cadence(self):
        rows = []
        probe = FakeProbe(
            [(True, [sample(state="idle"), sample(state="disabled")])]
        )
        result = CollectorLoop(probe, rows).collect_once(sampled_at=1)
        self.assertEqual(300.0, result.next_delay_seconds)

    def test_store_batch_api_is_used_once_per_probe_pass(self):
        class BatchSink:
            def __init__(self):
                self.batches = []

            def record_gpu_samples(self, rows):
                self.batches.append(tuple(rows))

        sink = BatchSink()
        probe = FakeProbe([(True, [sample(), sample(state="training")])])
        CollectorLoop(probe, sink).collect_once(sampled_at=1)
        self.assertEqual(1, len(sink.batches))
        self.assertEqual(2, len(sink.batches[0]))

    def test_empty_probe_set_is_safe_and_bounded(self):
        collector = CollectorLoop([], [], sleep=lambda _: None)
        results = collector.run_iterations(2)
        self.assertEqual(2, len(results))
        self.assertTrue(all(result.ok for result in results))
        self.assertEqual((), results[0].samples)


if __name__ == "__main__":
    unittest.main()
