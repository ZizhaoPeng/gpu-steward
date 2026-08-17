"""Private SQLite state and atomic lease operations.

The database is deliberately per Unix user.  Every mutating operation is
performed while holding a companion ``flock`` lock and inside a SQLite
``BEGIN IMMEDIATE`` transaction.  The unique constraint on active GPU UUIDs
is the final guard against duplicate allocation even if two SSH sessions
arrive at the same instant.
"""

import contextlib
import json
import os
import sqlite3
import threading
import time
import uuid
from dataclasses import dataclass
from typing import Callable, Dict, Iterable, Iterator, List, Optional, Sequence, Tuple

try:
    import fcntl
except ImportError:  # pragma: no cover - GPU Steward v1 targets Linux.
    fcntl = None

from .errors import (
    InventoryChangedError,
    QueueError,
    StateError,
    TaskBusyError,
    TaskNotFoundError,
)
from .processes import pid_matches, process_start_time
from .scheduler import Allocation, Request, plan_allocations


SCHEMA_VERSION = 1
ACTIVE_TASK_STATUSES = ("launching", "running", "cancelling")
TERMINAL_TASK_STATUSES = ("completed", "failed", "cancelled")


@dataclass(frozen=True)
class TaskRecord:
    task_id: str
    status: str
    command: Tuple[str, ...]
    cwd: str
    min_gpus: int
    max_gpus: Optional[int]
    priority: int
    label: str
    created_at: float
    started_at: Optional[float]
    finished_at: Optional[float]
    exit_code: Optional[int]
    error: Optional[str]

    def as_dict(self, lease: Optional["LeaseRecord"] = None):
        result = {
            "task_id": self.task_id,
            "status": self.status,
            "command": list(self.command),
            "cwd": self.cwd,
            "min_gpus": self.min_gpus,
            "max_gpus": self.max_gpus,
            "priority": self.priority,
            "label": self.label,
            "created_at": self.created_at,
            "started_at": self.started_at,
            "finished_at": self.finished_at,
            "exit_code": self.exit_code,
            "error": self.error,
        }
        if lease is not None:
            result["lease"] = lease.as_dict()
        return result


@dataclass(frozen=True)
class LeaseRecord:
    lease_id: str
    task_id: str
    status: str
    gpu_ids: Tuple[str, ...]
    pid: Optional[int]
    pid_start_time: Optional[int]
    reserved_at: float
    claimed_at: Optional[float]
    released_at: Optional[float]

    def as_dict(self):
        return {
            "lease_id": self.lease_id,
            "task_id": self.task_id,
            "status": self.status,
            "gpu_uuids": list(self.gpu_ids),
            "pid": self.pid,
            "pid_start_time": self.pid_start_time,
            "reserved_at": self.reserved_at,
            "claimed_at": self.claimed_at,
            "released_at": self.released_at,
        }


def default_db_path() -> str:
    configured = os.environ.get("GPU_STEWARD_DB")
    if configured:
        return os.path.abspath(os.path.expanduser(configured))
    return os.path.expanduser("~/.gpu-steward/state.sqlite3")


def _lock_path(db_path: str) -> str:
    return db_path + ".lock"


class StateStore:
    """SQLite-backed queue state.

    Public methods expect the caller to hold :meth:`locked` when used as part
    of a multi-step scheduling decision.  The schema initialization is safe
    to call repeatedly and owns its own short transaction.
    """

    def __init__(self, path: Optional[str] = None):
        self.path = os.path.abspath(os.path.expanduser(path or default_db_path()))
        self.lock_path = _lock_path(self.path)
        self._thread_lock = threading.RLock()
        self._lock_depth = 0
        self._lock_handle = None
        parent = os.path.dirname(self.path)
        if parent:
            os.makedirs(parent, mode=0o700, exist_ok=True)
            try:
                os.chmod(parent, 0o700)
            except OSError:
                pass
        self._ensure_private_file(self.path)
        self._ensure_private_file(self.lock_path)
        self.connection = sqlite3.connect(
            self.path,
            timeout=30.0,
            isolation_level=None,
            check_same_thread=False,
        )
        self.connection.row_factory = sqlite3.Row
        self.connection.execute("PRAGMA foreign_keys = ON")
        self.connection.execute("PRAGMA busy_timeout = 30000")
        self.initialize()

    @staticmethod
    def _ensure_private_file(path: str):
        flags = os.O_CREAT | os.O_RDWR
        try:
            fd = os.open(path, flags, 0o600)
            os.close(fd)
        except OSError as exc:
            raise StateError("cannot create private state file {}: {}".format(path, exc)) from exc
        try:
            os.chmod(path, 0o600)
        except OSError:
            pass

    def close(self):
        self.connection.close()

    @contextlib.contextmanager
    def locked(self):
        """Serialize a scheduling decision across processes and threads."""

        with self._thread_lock:
            outermost = self._lock_depth == 0
            self._lock_depth += 1
            handle = None
            try:
                if outermost and fcntl is not None:
                    try:
                        handle = open(self.lock_path, "a+")
                        fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
                        self._lock_handle = handle
                    except OSError as exc:
                        if handle is not None:
                            handle.close()
                        raise StateError("cannot lock {}: {}".format(self.lock_path, exc)) from exc
                yield self
            finally:
                self._lock_depth -= 1
                if outermost and fcntl is not None and self._lock_handle is not None:
                    try:
                        fcntl.flock(self._lock_handle.fileno(), fcntl.LOCK_UN)
                    finally:
                        self._lock_handle.close()
                        self._lock_handle = None

    @contextlib.contextmanager
    def transaction(self):
        """Open an atomic write transaction; callers normally hold ``locked``."""

        try:
            self.connection.execute("BEGIN IMMEDIATE")
            yield self.connection
            self.connection.execute("COMMIT")
        except Exception:
            try:
                self.connection.execute("ROLLBACK")
            except sqlite3.Error:
                pass
            raise

    def initialize(self):
        # ``sqlite3.Connection.executescript`` performs its own implicit
        # commit before running a script.  Keep schema setup idempotent and
        # outside the normal BEGIN/COMMIT helper rather than pretending the
        # script participates in that transaction.
        try:
            self.connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS metadata (
                    key TEXT PRIMARY KEY,
                    value TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS tasks (
                    task_id TEXT PRIMARY KEY,
                    status TEXT NOT NULL CHECK(status IN
                        ('queued','launching','running','cancelling',
                         'completed','failed','cancelled')),
                    command_json TEXT NOT NULL,
                    cwd TEXT NOT NULL,
                    min_gpus INTEGER NOT NULL CHECK(min_gpus >= 1),
                    max_gpus INTEGER,
                    priority INTEGER NOT NULL DEFAULT 0,
                    label TEXT NOT NULL DEFAULT '',
                    created_at REAL NOT NULL,
                    started_at REAL,
                    finished_at REAL,
                    exit_code INTEGER,
                    error TEXT
                );
                CREATE INDEX IF NOT EXISTS tasks_queue_order
                    ON tasks(status, priority DESC, created_at, task_id);
                CREATE TABLE IF NOT EXISTS leases (
                    lease_id TEXT PRIMARY KEY,
                    task_id TEXT NOT NULL UNIQUE REFERENCES tasks(task_id),
                    status TEXT NOT NULL CHECK(status IN ('active','released')),
                    pid INTEGER,
                    pid_start_time INTEGER,
                    reserved_at REAL NOT NULL,
                    claimed_at REAL,
                    released_at REAL
                );
                CREATE TABLE IF NOT EXISTS lease_gpus (
                    lease_id TEXT NOT NULL REFERENCES leases(lease_id),
                    gpu_uuid TEXT NOT NULL,
                    ordinal INTEGER NOT NULL,
                    PRIMARY KEY(lease_id, gpu_uuid),
                    UNIQUE(gpu_uuid)
                );
                CREATE INDEX IF NOT EXISTS lease_gpus_gpu ON lease_gpus(gpu_uuid);
                INSERT OR IGNORE INTO metadata(key, value)
                    VALUES ('schema_version', '1');
                """
            )
        except sqlite3.Error as exc:
            raise StateError("cannot initialize state schema: {}".format(exc)) from exc

    def _metadata(self, key: str) -> Optional[str]:
        row = self.connection.execute(
            "SELECT value FROM metadata WHERE key = ?", (key,)
        ).fetchone()
        return None if row is None else row[0]

    def _set_metadata(self, key: str, value: str):
        self.connection.execute(
            "INSERT INTO metadata(key,value) VALUES(?,?) "
            "ON CONFLICT(key) DO UPDATE SET value=excluded.value",
            (key, value),
        )

    def validate_inventory_signature(self, gpu_uuids: Sequence[str]):
        current = sorted(set(gpu_uuids))
        if len(current) != len(gpu_uuids) or not current:
            raise StateError("inventory UUID set must be non-empty and unique")
        previous_raw = self._metadata("inventory_uuids")
        previous = None
        if previous_raw:
            try:
                previous = sorted(json.loads(previous_raw))
            except (TypeError, ValueError) as exc:
                raise StateError("invalid stored inventory signature") from exc
        if previous is not None and previous != current and self.active_lease_count() > 0:
            raise InventoryChangedError(
                "GPU UUID inventory changed while a lease is active; "
                "new allocations are paused"
            )
        self._set_metadata("inventory_uuids", json.dumps(current, separators=(",", ":")))

    def active_lease_count(self) -> int:
        row = self.connection.execute(
            "SELECT COUNT(*) FROM leases WHERE status = 'active'"
        ).fetchone()
        return int(row[0])

    def active_gpu_ids(self) -> Tuple[str, ...]:
        rows = self.connection.execute(
            "SELECT gpu_uuid FROM lease_gpus "
            "JOIN leases ON leases.lease_id = lease_gpus.lease_id "
            "WHERE leases.status = 'active' ORDER BY lease_gpus.ordinal"
        ).fetchall()
        return tuple(row[0] for row in rows)

    def add_task(
        self,
        command: Sequence[str],
        cwd: str,
        min_gpus: int = 1,
        max_gpus: Optional[int] = None,
        priority: int = 0,
        label: str = "",
        task_id: Optional[str] = None,
        created_at: Optional[float] = None,
    ) -> str:
        if not command or any(not isinstance(item, str) for item in command):
            raise QueueError("command must be a non-empty argv sequence")
        if min_gpus < 1:
            raise QueueError("min_gpus must be at least 1")
        if max_gpus is not None and max_gpus < min_gpus:
            raise QueueError("max_gpus must be greater than or equal to min_gpus")
        if not os.path.isdir(cwd):
            raise QueueError("working directory does not exist: {}".format(cwd))
        identifier = task_id or "task-{}".format(uuid.uuid4().hex)
        timestamp = time.time() if created_at is None else float(created_at)
        try:
            self.connection.execute(
                "INSERT INTO tasks(task_id,status,command_json,cwd,min_gpus,max_gpus,"
                "priority,label,created_at) VALUES(?,?,?,?,?,?,?,?,?)",
                (
                    identifier,
                    "queued",
                    json.dumps(list(command), ensure_ascii=False),
                    os.path.abspath(cwd),
                    min_gpus,
                    max_gpus,
                    priority,
                    label,
                    timestamp,
                ),
            )
        except sqlite3.Error as exc:
            raise StateError("cannot insert task: {}".format(exc)) from exc
        return identifier

    @staticmethod
    def _task_from_row(row: sqlite3.Row) -> TaskRecord:
        try:
            command = tuple(json.loads(row["command_json"]))
        except (TypeError, ValueError) as exc:
            raise StateError("task {} contains invalid command JSON".format(row["task_id"])) from exc
        return TaskRecord(
            task_id=row["task_id"],
            status=row["status"],
            command=command,
            cwd=row["cwd"],
            min_gpus=int(row["min_gpus"]),
            max_gpus=None if row["max_gpus"] is None else int(row["max_gpus"]),
            priority=int(row["priority"]),
            label=row["label"],
            created_at=float(row["created_at"]),
            started_at=None if row["started_at"] is None else float(row["started_at"]),
            finished_at=None if row["finished_at"] is None else float(row["finished_at"]),
            exit_code=None if row["exit_code"] is None else int(row["exit_code"]),
            error=row["error"],
        )

    def get_task(self, task_id: str) -> TaskRecord:
        row = self.connection.execute(
            "SELECT * FROM tasks WHERE task_id = ?", (task_id,)
        ).fetchone()
        if row is None:
            raise TaskNotFoundError("unknown task: {}".format(task_id))
        return self._task_from_row(row)

    def list_tasks(self) -> List[TaskRecord]:
        rows = self.connection.execute(
            "SELECT * FROM tasks ORDER BY created_at, task_id"
        ).fetchall()
        return [self._task_from_row(row) for row in rows]

    def queued_requests(self) -> List[Request]:
        rows = self.connection.execute(
            "SELECT task_id,min_gpus,max_gpus,priority,created_at FROM tasks "
            "WHERE status = 'queued' ORDER BY priority DESC, created_at, task_id"
        ).fetchall()
        return [
            Request(
                request_id=row["task_id"],
                min_gpus=int(row["min_gpus"]),
                max_gpus=None if row["max_gpus"] is None else int(row["max_gpus"]),
                priority=int(row["priority"]),
                created_at=float(row["created_at"]),
            )
            for row in rows
        ]

    @staticmethod
    def _lease_from_row(row: sqlite3.Row, gpu_ids: Sequence[str]) -> LeaseRecord:
        return LeaseRecord(
            lease_id=row["lease_id"],
            task_id=row["task_id"],
            status=row["status"],
            gpu_ids=tuple(gpu_ids),
            pid=None if row["pid"] is None else int(row["pid"]),
            pid_start_time=(
                None if row["pid_start_time"] is None else int(row["pid_start_time"])
            ),
            reserved_at=float(row["reserved_at"]),
            claimed_at=None if row["claimed_at"] is None else float(row["claimed_at"]),
            released_at=None if row["released_at"] is None else float(row["released_at"]),
        )

    def get_lease(self, task_id: str, active_only: bool = True) -> Optional[LeaseRecord]:
        clause = "AND leases.status = 'active'" if active_only else ""
        row = self.connection.execute(
            "SELECT leases.* FROM leases WHERE task_id = ? {}".format(clause),
            (task_id,),
        ).fetchone()
        if row is None:
            return None
        gpu_rows = self.connection.execute(
            "SELECT gpu_uuid FROM lease_gpus WHERE lease_id = ? ORDER BY ordinal",
            (row["lease_id"],),
        ).fetchall()
        return self._lease_from_row(row, [item[0] for item in gpu_rows])

    def active_leases(self) -> List[LeaseRecord]:
        rows = self.connection.execute(
            "SELECT * FROM leases WHERE status = 'active' ORDER BY reserved_at, lease_id"
        ).fetchall()
        result = []
        for row in rows:
            gpu_rows = self.connection.execute(
                "SELECT gpu_uuid FROM lease_gpus WHERE lease_id = ? ORDER BY ordinal",
                (row["lease_id"],),
            ).fetchall()
            result.append(self._lease_from_row(row, [item[0] for item in gpu_rows]))
        return result

    def allocate_batch(
        self,
        free_gpu_ids: Sequence[str],
        total_gpus: int,
        external_busy_gpus: int = 0,
        reserve_gpus: int = 1,
        strict_fifo: bool = False,
    ) -> List[LeaseRecord]:
        """Reserve a fair batch of queued tasks atomically."""

        # Coordinator normally opens this transaction while holding the file
        # lock.  Keeping a direct StateStore call atomic as well prevents a
        # caller that uses the low-level API from observing half a batch after
        # a UUID uniqueness error.
        if not self.connection.in_transaction:
            with self.transaction():
                return self.allocate_batch(
                    free_gpu_ids=free_gpu_ids,
                    total_gpus=total_gpus,
                    external_busy_gpus=external_busy_gpus,
                    reserve_gpus=reserve_gpus,
                    strict_fifo=strict_fifo,
                )

        active_jobs = self.active_lease_count()
        allocations = plan_allocations(
            free_gpu_ids=free_gpu_ids,
            waiting=self.queued_requests(),
            total_gpus=total_gpus,
            active_jobs=active_jobs,
            external_busy_gpus=external_busy_gpus,
            reserve_gpus=reserve_gpus,
            strict_fifo=strict_fifo,
        )
        now = time.time()
        reserved: List[LeaseRecord] = []
        for allocation in allocations:
            lease_id = "lease-{}".format(uuid.uuid4().hex)
            try:
                self.connection.execute(
                    "INSERT INTO leases(lease_id,task_id,status,reserved_at) "
                    "VALUES(?,?, 'active', ?)",
                    (lease_id, allocation.request_id, now),
                )
                for ordinal, gpu_uuid in enumerate(allocation.gpu_ids):
                    self.connection.execute(
                        "INSERT INTO lease_gpus(lease_id,gpu_uuid,ordinal) VALUES(?,?,?)",
                        (lease_id, gpu_uuid, ordinal),
                    )
                self.connection.execute(
                    "UPDATE tasks SET status = 'launching' WHERE task_id = ? AND status = 'queued'",
                    (allocation.request_id,),
                )
            except sqlite3.Error as exc:
                raise StateError("cannot reserve GPU lease: {}".format(exc)) from exc
            lease = self.get_lease(allocation.request_id)
            if lease is None:
                raise StateError("reserved lease disappeared: {}".format(allocation.request_id))
            reserved.append(lease)
        return reserved

    def claim_task(self, task_id: str, now: Optional[float] = None) -> Optional[LeaseRecord]:
        """Claim a preallocated task for this supervisor PID."""

        task = self.get_task(task_id)
        if task.status == "queued":
            return None
        if task.status in TERMINAL_TASK_STATUSES or task.status == "cancelling":
            return None
        lease = self.get_lease(task_id)
        if lease is None:
            raise StateError("task {} has no active lease".format(task_id))
        current_pid = os.getpid()
        current_start = process_start_time(current_pid)
        timestamp = time.time() if now is None else float(now)
        if lease.pid is not None:
            if lease.pid == current_pid and pid_matches(current_pid, lease.pid_start_time):
                return lease
            raise TaskBusyError("task {} is already claimed by another process".format(task_id))
        self.connection.execute(
            "UPDATE leases SET pid = ?, pid_start_time = ?, claimed_at = ? WHERE lease_id = ? AND pid IS NULL",
            (current_pid, current_start, timestamp, lease.lease_id),
        )
        self.connection.execute(
            "UPDATE tasks SET status = 'running', started_at = ? WHERE task_id = ? AND status = 'launching'",
            (timestamp, task_id),
        )
        return self.get_lease(task_id)

    def update_lease_pid(
        self, task_id: str, pid: int, pid_start: Optional[int]
    ) -> LeaseRecord:
        lease = self.get_lease(task_id)
        if lease is None:
            raise StateError("task {} has no active lease".format(task_id))
        if lease.pid != os.getpid():
            raise TaskBusyError("task {} is not owned by this supervisor".format(task_id))
        self.connection.execute(
            "UPDATE leases SET pid = ?, pid_start_time = ? WHERE lease_id = ? AND status='active'",
            (pid, pid_start, lease.lease_id),
        )
        return self.get_lease(task_id)

    def _release_lease_tx(
        self,
        task_id: str,
        status: str,
        exit_code: Optional[int] = None,
        error: Optional[str] = None,
        now: Optional[float] = None,
    ):
        lease = self.get_lease(task_id)
        timestamp = time.time() if now is None else float(now)
        if lease is not None:
            self.connection.execute(
                "DELETE FROM lease_gpus WHERE lease_id = ?", (lease.lease_id,)
            )
            self.connection.execute(
                "UPDATE leases SET status='released', released_at=? WHERE lease_id=?",
                (timestamp, lease.lease_id),
            )
        self.connection.execute(
            "UPDATE tasks SET status=?, finished_at=?, exit_code=?, error=? WHERE task_id=?",
            (status, timestamp, exit_code, error, task_id),
        )

    def finish_task(
        self,
        task_id: str,
        pid: Optional[int],
        pid_start: Optional[int],
        exit_code: int,
        error: Optional[str] = None,
    ) -> TaskRecord:
        task = self.get_task(task_id)
        lease = self.get_lease(task_id)
        if lease is not None:
            if pid is None or lease.pid != pid or lease.pid_start_time != pid_start:
                raise TaskBusyError("task {} lease identity mismatch".format(task_id))
        status = "cancelled" if task.status == "cancelling" else (
            "completed" if exit_code == 0 else "failed"
        )
        self._release_lease_tx(task_id, status, exit_code, error)
        return self.get_task(task_id)

    def request_cancel(self, task_id: str) -> Optional[LeaseRecord]:
        """Mark a managed task cancelling and return its owned lease.

        The caller must signal the returned PID while still holding the outer
        file lock.  GPU rows remain reserved until the supervisor or ``gc``
        observes process exit, so cancellation never races a live child.
        """

        task = self.get_task(task_id)
        if task.status in TERMINAL_TASK_STATUSES:
            return self.get_lease(task_id, active_only=False)
        lease = self.get_lease(task_id)
        if task.status in ("queued", "launching") and (lease is None or lease.pid is None):
            self._release_lease_tx(task_id, "cancelled", 130, "cancelled before start")
            return None
        if lease is None:
            raise StateError("task {} has no active lease".format(task_id))
        if task.status == "running":
            self.connection.execute(
                "UPDATE tasks SET status='cancelling', error=? WHERE task_id=?",
                ("cancel requested", task_id),
            )
        return self.get_lease(task_id)

    def recover_stale(
        self,
        now: Optional[float] = None,
        launch_timeout: float = 300.0,
        checker: Callable[[Optional[int], Optional[int]], bool] = pid_matches,
    ) -> List[str]:
        """Release leases whose PID identity no longer exists.

        A reserved-but-unclaimed launch is only considered stale after a
        generous timeout, allowing another invocation for the same task to
        claim it.  PID start time is checked whenever Linux ``/proc`` provides
        it, preventing accidental recovery after PID reuse.
        """

        timestamp = time.time() if now is None else float(now)
        recovered: List[str] = []
        for lease in list(self.active_leases()):
            task = self.get_task(lease.task_id)
            stale = False
            if lease.pid is None:
                stale = timestamp - lease.reserved_at >= launch_timeout
            else:
                stale = not checker(lease.pid, lease.pid_start_time)
            if not stale:
                continue
            status = "cancelled" if task.status == "cancelling" else "failed"
            message = "stale lease recovered (PID identity no longer exists)"
            self._release_lease_tx(task.task_id, status, None, message, timestamp)
            recovered.append(task.task_id)
        return recovered

    def status_snapshot(self) -> Dict[str, object]:
        leases = {lease.task_id: lease for lease in self.active_leases()}
        tasks = [task.as_dict(leases.get(task.task_id)) for task in self.list_tasks()]
        return {
            "schema_version": SCHEMA_VERSION,
            "tasks": tasks,
            "queue": [item for item in tasks if item["status"] == "queued"],
            "active_leases": [lease.as_dict() for lease in leases.values()],
            "active_gpu_uuids": list(self.active_gpu_ids()),
        }
