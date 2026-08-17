import json
import os
import signal
import sys
import tempfile
import threading
import time
import unittest

from gpu_steward.inventory import ComputeProcess, GPUDevice, StaticInventory
from gpu_steward.errors import InventoryError
from gpu_steward.runtime import Coordinator
from gpu_steward.state import SCHEMA_VERSION, StateStore


def devices(count):
    return [GPUDevice(index=i, uuid="GPU-{}".format(i)) for i in range(count)]


class RuntimeTests(unittest.TestCase):
    def make_coordinator(self, directory, count=4, processes=()):
        store = StateStore(os.path.join(directory, "state.sqlite3"))
        inventory = StaticInventory(devices(count), processes)
        return Coordinator(store=store, inventory=inventory, poll_interval=0.02)

    @staticmethod
    def wait_for(predicate, timeout=10.0):
        deadline = time.time() + timeout
        while time.time() < deadline:
            value = predicate()
            if value:
                return value
            time.sleep(0.05)
        return None

    @staticmethod
    def pid_alive(pid):
        try:
            os.kill(pid, 0)
        except OSError:
            return False
        return True

    @staticmethod
    def read_pid_file(path):
        if not os.path.exists(path):
            return None
        with open(path, "r") as handle:
            text = handle.read().strip()
        return int(text) if text else None

    def test_dynamic_inventory_count_and_solo_reserve(self):
        for count in (1, 2, 4, 8):
            with self.subTest(count=count), tempfile.TemporaryDirectory() as directory:
                coordinator = self.make_coordinator(directory, count)
                task_id = coordinator.enqueue(["/bin/true"], cwd=directory)
                with coordinator.store.locked():
                    leases = coordinator._schedule_locked()
                self.assertEqual(1, len(leases))
                self.assertEqual(max(1, count - 1), len(leases[0].gpu_ids))
                self.assertEqual(task_id, leases[0].task_id)
                coordinator.close()

    def test_external_busy_uuid_is_never_allocated(self):
        process = ComputeProcess(gpu_uuid="GPU-1", pid=999999, process_name="external")
        with tempfile.TemporaryDirectory() as directory:
            coordinator = self.make_coordinator(directory, 4, [process])
            task_id = coordinator.enqueue(["/bin/true"], cwd=directory)
            with coordinator.store.locked():
                leases = coordinator._schedule_locked()
            self.assertEqual(task_id, leases[0].task_id)
            self.assertEqual(("GPU-0", "GPU-2", "GPU-3"), leases[0].gpu_ids)
            self.assertNotIn("GPU-1", coordinator.store.active_gpu_ids())
            coordinator.close()

    def test_release_three_gpus_splits_two_plus_one(self):
        with tempfile.TemporaryDirectory() as directory:
            coordinator = self.make_coordinator(directory)
            first = coordinator.enqueue(["/bin/true"], cwd=directory)
            with coordinator.store.locked():
                coordinator._schedule_locked()
                with coordinator.store.transaction():
                    lease = coordinator.store.claim_task(first)
            second = coordinator.enqueue(["/bin/true"], cwd=directory)
            with coordinator.store.locked():
                with coordinator.store.transaction():
                    coordinator._schedule_locked()
                    coordinator.store.finish_task(first, os.getpid(), None, 0)
            older = coordinator.enqueue(["/bin/true"], cwd=directory)
            newer = coordinator.enqueue(["/bin/true"], cwd=directory)
            with coordinator.store.locked():
                leases = coordinator._schedule_locked()
            pairs = [(lease.task_id, lease.gpu_ids) for lease in leases]
            self.assertEqual(
                [(older, ("GPU-0", "GPU-1")), (newer, ("GPU-2",))],
                pairs,
            )
            coordinator.close()

    def test_release_one_gpu_starts_only_one_waiter(self):
        with tempfile.TemporaryDirectory() as directory:
            coordinator = self.make_coordinator(directory)
            first = coordinator.enqueue(["/bin/true"], cwd=directory)
            with coordinator.store.locked():
                coordinator._schedule_locked()
                with coordinator.store.transaction():
                    coordinator.store.claim_task(first)
            second = coordinator.enqueue(["/bin/true"], cwd=directory)
            with coordinator.store.locked():
                with coordinator.store.transaction():
                    coordinator._schedule_locked()
                    coordinator.store.claim_task(second)
            older = coordinator.enqueue(["/bin/true"], cwd=directory)
            newer = coordinator.enqueue(["/bin/true"], cwd=directory)
            with coordinator.store.locked():
                with coordinator.store.transaction():
                    coordinator.store.finish_task(second, os.getpid(), None, 0)
                leases = coordinator._schedule_locked()
            self.assertEqual([(older, ("GPU-3",))], [(x.task_id, x.gpu_ids) for x in leases])
            self.assertEqual("queued", coordinator.store.get_task(newer).status)
            coordinator.close()

    def test_run_passes_uuid_environment_and_releases(self):
        with tempfile.TemporaryDirectory() as directory:
            coordinator = self.make_coordinator(directory)
            output = os.path.join(directory, "env.json")
            code, payload = coordinator.run_task(
                [
                    "python3",
                    "-c",
                    (
                        "import json,os; json.dump({k:os.environ[k] for k in "
                        "('CUDA_VISIBLE_DEVICES','CUDA_DEVICE_ORDER','GPU_STEWARD_GPU_COUNT',"
                        "'GPU_STEWARD_TASK_ID','GPU_STEWARD_LEASE_ID')},open(%r,'w'))"
                    )
                    % output,
                ],
                cwd=directory,
            )
            self.assertEqual(0, code)
            self.assertTrue(payload["ok"])
            self.assertEqual([], coordinator.store.active_leases())
            with open(output, "r") as handle:
                values = json.load(handle)
            self.assertEqual("GPU-0,GPU-1,GPU-2", values["CUDA_VISIBLE_DEVICES"])
            self.assertEqual("PCI_BUS_ID", values["CUDA_DEVICE_ORDER"])
            self.assertEqual("3", values["GPU_STEWARD_GPU_COUNT"])
            self.assertEqual(payload["task_id"], values["GPU_STEWARD_TASK_ID"])
            self.assertTrue(values["GPU_STEWARD_LEASE_ID"].startswith("lease-"))
            coordinator.close()

    def test_stale_pid_start_identity_recovers_lease(self):
        with tempfile.TemporaryDirectory() as directory:
            coordinator = self.make_coordinator(directory)
            task_id = coordinator.enqueue(["/bin/true"], cwd=directory)
            with coordinator.store.locked():
                coordinator._schedule_locked()
                with coordinator.store.transaction():
                    coordinator.store.claim_task(task_id)
                    coordinator.store.update_lease_pid(task_id, 999999, 42)
                    recovered = coordinator.store.recover_stale(
                        checker=lambda pid, start: False
                    )
            self.assertEqual([task_id], recovered)
            self.assertEqual((), coordinator.store.active_gpu_ids())
            self.assertEqual("failed", coordinator.store.get_task(task_id).status)
            coordinator.close()

    def test_concurrent_run_tasks_never_duplicate_uuid(self):
        with tempfile.TemporaryDirectory() as directory:
            result = []
            result_lock = threading.Lock()
            barrier = threading.Barrier(2)

            def worker():
                coordinator = self.make_coordinator(directory)
                barrier.wait()
                code, payload = coordinator.run_task(
                    ["python3", "-c", "import time; time.sleep(.15)"], cwd=directory
                )
                with result_lock:
                    result.append((code, tuple(payload["task"]["gpu_uuids"])))
                coordinator.close()

            threads = [threading.Thread(target=worker) for _ in range(2)]
            for thread in threads:
                thread.start()
            for thread in threads:
                thread.join(timeout=10)
            self.assertEqual(2, len(result))
            self.assertTrue(all(code == 0 for code, _ in result))
            self.assertEqual([1, 3], sorted(len(gpus) for _, gpus in result))
            self.assertEqual(4, len(set(result[0][1]).union(result[1][1])))
            self.assertEqual(0, len(set(result[0][1]).intersection(result[1][1])))

    def test_inventory_failure_does_not_leave_orphan_request(self):
        class BrokenInventory:
            def query(self):
                raise InventoryError("probe failed")

        with tempfile.TemporaryDirectory() as directory:
            store = StateStore(os.path.join(directory, "state.sqlite3"))
            coordinator = Coordinator(
                store=store,
                inventory=BrokenInventory(),
                poll_interval=0.02,
            )
            with self.assertRaisesRegex(InventoryError, "probe failed"):
                coordinator.run_task(["/bin/true"], cwd=directory)
            tasks = coordinator.store.list_tasks()
            self.assertEqual(1, len(tasks))
            self.assertEqual("cancelled", tasks[0].status)
            self.assertEqual([], coordinator.store.active_leases())
            coordinator.close()

    def test_cancel_terminates_process_group_and_normalizes_exit_code(self):
        with tempfile.TemporaryDirectory() as directory:
            coordinator = self.make_coordinator(directory)
            pid_file = os.path.join(directory, "worker.pid")
            result = {}

            def runner():
                try:
                    result["value"] = coordinator.run_task(
                        [
                            sys.executable,
                            "-c",
                            (
                                "import subprocess, sys, time\n"
                                "worker = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(60)'])\n"
                                "with open(%r, 'w') as handle:\n"
                                "    handle.write(str(worker.pid))\n"
                                "time.sleep(60)\n"
                            )
                            % pid_file,
                        ],
                        cwd=directory,
                        min_gpus=1,
                        max_gpus=1,
                    )
                except Exception as exc:
                    result["error"] = exc

            thread = threading.Thread(target=runner)
            thread.start()

            task_id = self.wait_for(
                lambda: next(
                    (
                        item["task_id"]
                        for item in coordinator.status_payload()["tasks"]
                        if item["status"] in ("launching", "running", "cancelling")
                    ),
                    None,
                )
            )
            self.assertIsNotNone(task_id)

            worker_pid = self.wait_for(lambda: self.read_pid_file(pid_file))
            self.assertIsNotNone(worker_pid)

            payload = coordinator.cancel_task(task_id)
            self.assertTrue(payload["ok"])

            thread.join(timeout=10)
            self.assertFalse(thread.is_alive())
            self.assertNotIn("error", result)

            exit_code, task_payload = result["value"]
            self.assertEqual(128 + signal.SIGTERM, exit_code)
            self.assertEqual(exit_code, task_payload["exit_code"])
            self.assertEqual("cancelled", task_payload["task"]["status"])
            self.assertEqual([], coordinator.store.active_leases())
            self.assertTrue(self.wait_for(lambda: not self.pid_alive(worker_pid)))
            coordinator.close()

    def test_status_has_versioned_machine_contract(self):
        with tempfile.TemporaryDirectory() as directory:
            coordinator = self.make_coordinator(directory)
            payload = coordinator.status_payload()
            self.assertEqual(SCHEMA_VERSION, payload["schema_version"])
            self.assertTrue(payload["ok"])
            self.assertEqual(4, payload["inventory"]["count"])
            self.assertIn("free_gpu_uuids", payload)
            self.assertIn("active_leases", payload)
            coordinator.close()


if __name__ == "__main__":
    unittest.main()
