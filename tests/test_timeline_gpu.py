import json
import subprocess
import unittest

from gpu_steward.timeline.config import GPUHostConfig
from gpu_steward.timeline.gpu import (
    GPUProbe,
    GPUProbeError,
    parse_nvidia_smi_csv,
)


class FakeRunner:
    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = []

    def __call__(self, argv):
        self.calls.append(list(argv))
        if not self.responses:
            return subprocess.CompletedProcess(argv, 1, "", "failure")
        return self.responses.pop(0)


class TimelineGPUTests(unittest.TestCase):
    def test_status_is_preferred_and_explicit_label_wins(self):
        status = {
            "ok": True,
            "inventory": {
                "gpus": [
                    {
                        "index": 0,
                        "uuid": "GPU-explicit-uuid",
                        "state": "managed_busy",
                        "processes": [{"pid": 41, "name": "/work/train.py"}],
                    },
                    {"index": 1, "uuid": "GPU-free-uuid", "state": "free", "processes": []},
                    {"index": 2, "uuid": "GPU-idle-uuid", "state": "free", "processes": []},
                ]
            },
            "tasks": [{"task_id": "task-1", "label": "experiment-A"}],
            "active_leases": [
                {"task_id": "task-1", "status": "active", "gpu_uuids": ["GPU-explicit-uuid"], "pid": 41}
            ],
        }
        runner = FakeRunner([])
        probe = GPUProbe(
            GPUHostConfig(name="AI3", host="ai3", disabled_gpu_indices=(2,)),
            runner=runner,
            status_provider=lambda: status,
            clock=lambda: 10,
        )
        samples = probe.sample()
        self.assertEqual(3, len(samples))
        self.assertEqual("training", samples[0].state)
        self.assertEqual("experiment-A", samples[0].task_name)
        self.assertEqual("explicit", samples[0].attribution)
        self.assertEqual("disabled", samples[2].state)
        self.assertEqual("idle", samples[2].hardware_state)
        self.assertEqual("gpu-steward-status", probe.last_source)
        self.assertEqual([], runner.calls)

    def test_nvidia_fallback_uses_safe_argv_and_marks_external_as_inferred(self):
        runner = FakeRunner(
            [
                subprocess.CompletedProcess([], 0, "0, GPU-a, RTX, x, 1\n1, GPU-b, RTX, y, 2\n", ""),
                subprocess.CompletedProcess([], 0, "GPU-b, 1234, /tmp/train.py\n", ""),
            ]
        )
        probe = GPUProbe("AI3", runner=runner, prefer_steward=False, clock=lambda: 12)
        samples = probe.sample()
        self.assertEqual(("idle", "external"), tuple(item.state for item in samples))
        self.assertEqual("train.py", samples[1].process_basename)
        self.assertEqual("train.py", samples[1].task_name)
        self.assertEqual("inferred", samples[1].attribution)
        self.assertEqual(2, len(runner.calls))
        self.assertTrue(all(call[0] == "ssh" for call in runner.calls))
        self.assertNotIn(";", " ".join(runner.calls[0]))
        self.assertIn("nvidia-smi", runner.calls[0])
        self.assertIn("--query-gpu=index,uuid,name,pci.bus_id,memory.total", runner.calls[0])

    def test_explicit_label_without_process_is_reserved(self):
        runner = FakeRunner(
            [
                subprocess.CompletedProcess([], 0, "0, GPU-a, RTX, x, 1\n", ""),
                subprocess.CompletedProcess([], 0, "", ""),
            ]
        )
        samples = GPUProbe("AI3", runner=runner, prefer_steward=False).sample(
            sampled_at=1, explicit_labels={"GPU-a": "reserved-job"}
        )
        self.assertEqual("reserved", samples[0].state)
        self.assertEqual("reserved-job", samples[0].task_name)
        self.assertEqual("explicit", samples[0].attribution)

    def test_failures_produce_host_unknown_without_command_details(self):
        def broken(argv):
            raise OSError("secret --password should not be recorded")

        probe = GPUProbe("AI3", runner=broken, prefer_steward=False, clock=lambda: 20)
        samples = probe.sample()
        self.assertEqual(1, len(samples))
        self.assertEqual("unknown", samples[0].state)
        self.assertIsNone(samples[0].gpu_index)
        self.assertNotIn("password", samples[0].as_dict()["error"])

    def test_parser_rejects_process_for_unknown_uuid(self):
        with self.assertRaises(GPUProbeError):
            parse_nvidia_smi_csv("0, GPU-a, RTX, x, 1\n", "GPU-b, 1, train.py\n")


if __name__ == "__main__":
    unittest.main()
