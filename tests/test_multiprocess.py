import multiprocessing
import os
import tempfile
import unittest

from gpu_steward.inventory import GPUDevice, StaticInventory
from gpu_steward.runtime import Coordinator
from gpu_steward.state import StateStore


def _run_in_process(directory, barrier, results):
    store = StateStore(os.path.join(directory, "state.sqlite3"))
    inventory = StaticInventory(
        [GPUDevice(index=index, uuid="GPU-{}".format(index)) for index in range(4)]
    )
    coordinator = Coordinator(
        store=store,
        inventory=inventory,
        poll_interval=0.01,
    )
    try:
        barrier.wait(timeout=5)
        code, payload = coordinator.run_task(
            ["python3", "-c", "import time; time.sleep(0.5)"],
            cwd=directory,
        )
        results.put((code, tuple(payload["task"]["gpu_uuids"])))
    finally:
        coordinator.close()


class MultiProcessTests(unittest.TestCase):
    def test_two_processes_get_three_plus_one_without_overlap(self):
        # Separate OS processes model independent SSH/Codex sessions and prove
        # the filesystem lock + SQLite uniqueness path, not merely thread safety.
        context = multiprocessing.get_context("fork")
        with tempfile.TemporaryDirectory() as directory:
            barrier = context.Barrier(2)
            results = context.Queue()
            workers = [
                context.Process(
                    target=_run_in_process,
                    args=(directory, barrier, results),
                )
                for _ in range(2)
            ]
            for worker in workers:
                worker.start()
            for worker in workers:
                worker.join(timeout=10)
                self.assertEqual(0, worker.exitcode)

            observed = [results.get(timeout=2) for _ in workers]
            self.assertTrue(all(code == 0 for code, _ in observed))
            self.assertEqual([1, 3], sorted(len(gpus) for _, gpus in observed))
            first = set(observed[0][1])
            second = set(observed[1][1])
            self.assertFalse(first.intersection(second))
            self.assertEqual(4, len(first.union(second)))


if __name__ == "__main__":
    unittest.main()
