import subprocess
import unittest

from gpu_steward.errors import InventoryError
from gpu_steward.inventory import NvidiaSMI


class FakeRunner:
    def __init__(self, gpu_stdout, process_stdout=""):
        self.gpu_stdout = gpu_stdout
        self.process_stdout = process_stdout
        self.calls = []

    def __call__(self, argv):
        self.calls.append(list(argv))
        if "query-gpu" in argv[1]:
            return subprocess.CompletedProcess(argv, 0, self.gpu_stdout, "")
        return subprocess.CompletedProcess(argv, 0, self.process_stdout, "")


class InventoryTests(unittest.TestCase):
    def test_queries_dynamic_uuid_inventory_and_processes(self):
        runner = FakeRunner(
            "0, GPU-a, RTX, 0000:01:00.0, 24576\n"
            "1, GPU-b, RTX, 0000:02:00.0, 24576\n",
            "GPU-b, 1234, train.py\n",
        )
        snapshot = NvidiaSMI(runner=runner).query()
        self.assertEqual(("GPU-a", "GPU-b"), snapshot.gpu_uuids)
        self.assertEqual((1234,), tuple(item.pid for item in snapshot.processes))
        self.assertEqual(2, len(runner.calls))

    def test_failed_query_is_fail_closed(self):
        def runner(argv):
            return subprocess.CompletedProcess(argv, 1, "", "driver unavailable")

        with self.assertRaises(InventoryError):
            NvidiaSMI(runner=runner).query()

    def test_unknown_process_uuid_is_fail_closed(self):
        runner = FakeRunner(
            "0, GPU-a, RTX, 0000:01:00.0, 24576\n",
            "GPU-unknown, 1234, train.py\n",
        )
        with self.assertRaises(InventoryError):
            NvidiaSMI(runner=runner).query()

    def test_empty_gpu_query_is_fail_closed(self):
        with self.assertRaises(InventoryError):
            NvidiaSMI(runner=FakeRunner("", "")).query()


if __name__ == "__main__":
    unittest.main()
