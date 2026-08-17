"""Runtime coordinator tying inventory, SQLite leases, and child supervision."""

import os
import signal
import subprocess
import sys
import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

from .errors import CommandError, InventoryError, QueueError, StateError, StewardError
from .inventory import ComputeProcess, InventorySnapshot, NvidiaSMI
from .processes import process_start_time, terminate_if_matches
from .state import SCHEMA_VERSION, LeaseRecord, StateStore


@dataclass(frozen=True)
class RuntimeView:
    snapshot: InventorySnapshot
    external_gpu_uuids: Tuple[str, ...]
    external_processes: Tuple[ComputeProcess, ...]
    free_gpu_uuids: Tuple[str, ...]


class Coordinator:
    """Single-host, whole-GPU queue coordinator.

    The coordinator holds the state lock while querying inventory and making a
    lease decision.  This keeps independent SSH sessions from allocating the
    same UUID.  It never terminates an unknown process: only a PID recorded in
    an active GPU Steward lease can be signalled by :meth:`cancel_task`.
    """

    def __init__(
        self,
        store: Optional[StateStore] = None,
        inventory=None,
        reserve_gpus: int = 1,
        strict_fifo: bool = False,
        launch_timeout: float = 300.0,
        poll_interval: float = 1.0,
    ):
        if reserve_gpus < 0:
            raise ValueError("reserve_gpus cannot be negative")
        self.store = store or StateStore()
        self.inventory = inventory or NvidiaSMI()
        self.reserve_gpus = reserve_gpus
        self.strict_fifo = strict_fifo
        self.launch_timeout = launch_timeout
        self.poll_interval = max(0.05, float(poll_interval))

    def close(self):
        self.store.close()

    @staticmethod
    def _normalize_exit_code(return_code: int) -> int:
        """Convert negative signal returns into shell-style exit codes."""

        if return_code < 0:
            return 128 + abs(return_code)
        return return_code

    def _view_locked(self, recover: bool = True) -> RuntimeView:
        snapshot = self.inventory.query()

        def collect_view() -> RuntimeView:
            if recover:
                self.store.recover_stale(launch_timeout=self.launch_timeout)
            self.store.validate_inventory_signature(snapshot.gpu_uuids)
            known = set(snapshot.gpu_uuids)
            active = set(self.store.active_gpu_ids())
            if not active.issubset(known):
                raise InventoryError(
                    "an active lease refers to a GPU UUID missing from inventory; "
                    "new allocations are paused"
                )
            # Processes on already leased GPUs are covered by that lease. Any
            # process on an unleased UUID is external and makes it ineligible.
            external = tuple(
                process
                for process in snapshot.processes
                if process.gpu_uuid not in active
            )
            external_ids = tuple(
                sorted({process.gpu_uuid for process in external})
            )
            external_set = set(external_ids)
            free = tuple(
                gpu.uuid
                for gpu in snapshot.gpus
                if gpu.uuid not in active and gpu.uuid not in external_set
            )
            return RuntimeView(
                snapshot=snapshot,
                external_gpu_uuids=external_ids,
                external_processes=external,
                free_gpu_uuids=free,
            )

        if self.store.connection.in_transaction:
            return collect_view()
        # Recovery deletes GPU rows and updates lease/task rows. Keep that
        # state transition atomic even for read-oriented commands such as
        # status and doctor.
        with self.store.transaction():
            return collect_view()

    def _schedule_locked(self) -> List[LeaseRecord]:
        view = self._view_locked()
        def allocate():
            return self.store.allocate_batch(
                free_gpu_ids=view.free_gpu_uuids,
                total_gpus=view.snapshot.gpu_count,
                external_busy_gpus=len(view.external_gpu_uuids),
                reserve_gpus=self.reserve_gpus,
                strict_fifo=self.strict_fifo,
            )
        if self.store.connection.in_transaction:
            return allocate()
        with self.store.transaction():
            return allocate()

    def inventory_payload(self) -> Dict[str, object]:
        with self.store.locked():
            view = self._view_locked()
            payload = view.snapshot.as_dict(
                view.external_gpu_uuids, self.store.active_gpu_ids()
            )
            payload.update(
                {
                    "schema_version": SCHEMA_VERSION,
                    "ok": True,
                    "external_busy_gpu_uuids": list(view.external_gpu_uuids),
                    "free_gpu_uuids": list(view.free_gpu_uuids),
                }
            )
            return payload

    def status_payload(self) -> Dict[str, object]:
        with self.store.locked():
            view = self._view_locked()
            payload = self.store.status_snapshot()
            payload.update(
                {
                    "ok": True,
                    "inventory": view.snapshot.as_dict(
                        view.external_gpu_uuids, self.store.active_gpu_ids()
                    ),
                    "external_busy_gpu_uuids": list(view.external_gpu_uuids),
                    "free_gpu_uuids": list(view.free_gpu_uuids),
                }
            )
            return payload

    def doctor_payload(self) -> Dict[str, object]:
        with self.store.locked():
            view = self._view_locked()
            db_mode = None
            lock_mode = None
            try:
                db_mode = oct(os.stat(self.store.path).st_mode & 0o777)
                lock_mode = oct(os.stat(self.store.lock_path).st_mode & 0o777)
            except OSError as exc:
                raise StateError("cannot inspect private state permissions: {}".format(exc)) from exc
            checks = {
                "python": sys.version_info >= (3, 8),
                "sqlite": True,
                "state_private": db_mode == "0o600" and lock_mode == "0o600",
                "nvidia_smi": True,
                "dynamic_gpu_count": view.snapshot.gpu_count >= 1,
            }
            return {
                "schema_version": SCHEMA_VERSION,
                "ok": all(checks.values()),
                "checks": checks,
                "database": {"path": self.store.path, "mode": db_mode, "lock_mode": lock_mode},
                "inventory": view.snapshot.as_dict(
                    view.external_gpu_uuids, self.store.active_gpu_ids()
                ),
                "external_busy_gpu_uuids": list(view.external_gpu_uuids),
            }

    def enqueue(
        self,
        command: Sequence[str],
        cwd: Optional[str] = None,
        min_gpus: int = 1,
        max_gpus: Optional[int] = None,
        priority: int = 0,
        label: str = "",
    ) -> str:
        workdir = os.path.abspath(cwd or os.getcwd())
        with self.store.locked():
            with self.store.transaction():
                return self.store.add_task(
                    command=command,
                    cwd=workdir,
                    min_gpus=min_gpus,
                    max_gpus=max_gpus,
                    priority=priority,
                    label=label,
                )

    def run_task(
        self,
        command: Sequence[str],
        cwd: Optional[str] = None,
        min_gpus: int = 1,
        max_gpus: Optional[int] = None,
        priority: int = 0,
        label: str = "",
        wait_timeout: Optional[float] = None,
    ) -> Tuple[int, Dict[str, object]]:
        """Queue, wait for, and supervise one argv command.

        The returned integer is the shell-visible exit code for the managed
        command. Signal exits are normalized to ``128 + signum`` so the CLI
        does not expose Python's negative ``Popen.returncode`` convention. A
        coordinator/inventory error raises an expected Steward exception and
        cancels this invocation's unclaimed request instead of leaving an
        orphan queue entry or guessing at GPU availability.
        """

        task_id = ""
        claimed: Optional[LeaseRecord] = None
        started_wait = time.time()
        try:
            # Keep a first arrival's enqueue, initial schedule, and claim in
            # one file-lock critical section. Without this, two simultaneous
            # arrivals can both become visible before scheduling and get 2+2
            # on a four-GPU idle host instead of the required 3+1.
            with self.store.locked():
                task_id = self.enqueue(
                    command=command,
                    cwd=cwd,
                    min_gpus=min_gpus,
                    max_gpus=max_gpus,
                    priority=priority,
                    label=label,
                )
                self._schedule_locked()
                with self.store.transaction():
                    claimed = self.store.claim_task(task_id)

            while claimed is None:
                with self.store.locked():
                    task = self.store.get_task(task_id)
                    if task.status in ("cancelled", "failed", "completed"):
                        exit_code = (
                            130
                            if task.status == "cancelled"
                            else (task.exit_code if task.exit_code is not None else 1)
                        )
                        return exit_code, self._result_payload(task, (), exit_code)
                    # _schedule_locked also recovers stale leases and validates
                    # the complete inventory before making a reservation.
                    self._schedule_locked()
                    with self.store.transaction():
                        claimed = self.store.claim_task(task_id)
                if claimed is not None:
                    break
                if wait_timeout is not None and time.time() - started_wait >= wait_timeout:
                    with self.store.locked():
                        with self.store.transaction():
                            self.store.request_cancel(task_id)
                    raise QueueError("timed out waiting for task {}".format(task_id))
                time.sleep(self.poll_interval)
        except StewardError:
            # A failed inventory/lock/claim must not leave this invocation's
            # queued or unclaimed launching task behind with no supervisor.
            if task_id:
                try:
                    with self.store.locked():
                        with self.store.transaction():
                            task = self.store.get_task(task_id)
                            lease = self.store.get_lease(task_id)
                            if task.status == "queued" or (
                                task.status == "launching"
                                and (lease is None or lease.pid is None)
                            ):
                                self.store.request_cancel(task_id)
                except StewardError:
                    # Preserve the original failure. A later gc/status call
                    # can still recover state if cleanup itself was blocked.
                    pass
            raise

        task = self.store.get_task(task_id)
        environment = os.environ.copy()
        environment.update(
            {
                "CUDA_VISIBLE_DEVICES": ",".join(claimed.gpu_ids),
                "CUDA_DEVICE_ORDER": "PCI_BUS_ID",
                "GPU_STEWARD_GPU_COUNT": str(len(claimed.gpu_ids)),
                "GPU_STEWARD_TASK_ID": task_id,
                "GPU_STEWARD_LEASE_ID": claimed.lease_id,
            }
        )
        child = None
        child_start: Optional[int] = None
        supervisor_pid = os.getpid()
        supervisor_start = process_start_time(supervisor_pid)
        child_bound = False
        try:
            child = subprocess.Popen(
                list(command),
                cwd=task.cwd,
                env=environment,
                shell=False,
                start_new_session=True,
            )
            child_start = process_start_time(child.pid)
            with self.store.locked():
                with self.store.transaction():
                    self.store.update_lease_pid(task_id, child.pid, child_start)
            child_bound = True
            return_code = self._normalize_exit_code(child.wait())
        except KeyboardInterrupt:
            if child is not None:
                self._terminate_child(child, child_start)
            return_code = 130
        except (OSError, ValueError) as exc:
            if child is not None:
                self._terminate_child(child, child_start)
            return_code = 127
            with self.store.locked():
                with self.store.transaction():
                    self.store.finish_task(
                        task_id,
                        child.pid if child_bound else supervisor_pid,
                        child_start if child_bound else supervisor_start,
                        return_code,
                        "command supervision failed: {}".format(exc),
                    )
            raise CommandError("command supervision failed: {}".format(exc)) from exc
        except StewardError:
            if child is not None:
                self._terminate_child(child, child_start)
            with self.store.locked():
                with self.store.transaction():
                    self.store.finish_task(
                        task_id,
                        child.pid if child_bound else supervisor_pid,
                        child_start if child_bound else supervisor_start,
                        127,
                        "command lease failed during launch",
                    )
            raise
        finally:
            if child is not None and child.poll() is None:
                self._terminate_child(child, child_start)
        with self.store.locked():
            with self.store.transaction():
                finished = self.store.finish_task(
                    task_id,
                    child.pid if child_bound else supervisor_pid,
                    child_start if child_bound else supervisor_start,
                    return_code,
                )
        return return_code, self._result_payload(
            finished, claimed.gpu_ids, return_code
        )

    @staticmethod
    def _result_payload(task, gpu_ids, exit_code: int) -> Dict[str, object]:
        result = task.as_dict()
        result["gpu_uuids"] = list(gpu_ids)
        return {
            "schema_version": SCHEMA_VERSION,
            "ok": exit_code == 0,
            "task_id": task.task_id,
            "exit_code": exit_code,
            "task": result,
        }

    @staticmethod
    def _terminate_child(child: subprocess.Popen, child_start: Optional[int]):
        if child.poll() is not None:
            return
        if not terminate_if_matches(child.pid, child_start):
            return
        try:
            child.wait(timeout=5)
        except (subprocess.TimeoutExpired, OSError):
            try:
                if child.poll() is None and terminate_if_matches(child.pid, child_start):
                    os.killpg(child.pid, signal.SIGKILL)
            except OSError:
                pass

    def cancel_task(self, task_id: str) -> Dict[str, object]:
        with self.store.locked():
            with self.store.transaction():
                lease = self.store.request_cancel(task_id)
                if lease is not None and lease.status == "active" and lease.pid is not None:
                    # The lease PID and start time are read from our private
                    # database immediately before signalling under the lock.
                    if not terminate_if_matches(lease.pid, lease.pid_start_time):
                        self.store.recover_stale(now=time.time())
                task = self.store.get_task(task_id)
                return {
                    "schema_version": SCHEMA_VERSION,
                    "ok": task.status in ("cancelled", "cancelling"),
                    "task": task.as_dict(self.store.get_lease(task_id)),
                }

    def gc(self) -> Dict[str, object]:
        with self.store.locked():
            with self.store.transaction():
                recovered = self.store.recover_stale(launch_timeout=self.launch_timeout)
                return {
                    "schema_version": SCHEMA_VERSION,
                    "ok": True,
                    "recovered_task_ids": recovered,
                    "recovered_count": len(recovered),
                    "status": self.store.status_snapshot(),
                }
