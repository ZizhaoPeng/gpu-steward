"""Read-only remote NVIDIA sampling for the GPU timeline.

This module is intentionally separate from :mod:`gpu_steward.inventory` and
the queue's SQLite state.  It may inspect a GPU Steward ``status --json``
payload when available, but it never allocates, claims, cancels, or releases a
lease.  If that status command is unavailable, a fixed ``nvidia-smi`` CSV
query is used as a read-only fallback.

All subprocesses are launched with an argv list and ``shell=False``.  Error
messages are deliberately generic: command arguments, remote paths, and
credentials must not become timeline data or logs.
"""

import csv
import io
import json
import ntpath
import os
import posixpath
import subprocess
import time
from dataclasses import dataclass
from typing import Any, Callable, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple, Union

from .config import GPUHostConfig


GPU_STATES: Tuple[str, ...] = (
    "training",
    "managed-other",
    "external",
    "reserved",
    "idle",
    "disabled",
    "unknown",
)
ATTRIBUTIONS: Tuple[str, ...] = ("explicit", "inferred", "none", "unknown")

GPU_QUERY_ARGS: Tuple[str, ...] = (
    "--query-gpu=index,uuid,name,pci.bus_id,memory.total",
    "--format=csv,noheader,nounits",
)
PROCESS_QUERY_ARGS: Tuple[str, ...] = (
    "--query-compute-apps=gpu_uuid,pid,process_name",
    "--format=csv,noheader,nounits",
)

_DISPLAY_LABEL_CHARS = frozenset(
    "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789 ._:/@+-"
)


class GPUProbeError(RuntimeError):
    """A probe failed before producing a trustworthy inventory."""


def _safe_text(value: Any, limit: int = 200) -> str:
    if value is None:
        return ""
    text = str(value).strip()
    text = "".join(char for char in text if ord(char) >= 32 and char not in "\x7f")
    return text[:limit]


def _safe_display_label(value: Any, limit: int = 128) -> Optional[str]:
    text = _safe_text(value, limit)
    if not text:
        return None
    # TimelineStore uses the same conservative alphabet.  Preserve the label
    # as far as possible while replacing punctuation that cannot be persisted;
    # never let a remote status payload make the local collector fail.
    text = "".join(char if char in _DISPLAY_LABEL_CHARS else "-" for char in text)
    text = text.strip(" -")[:limit]
    return text or None


def _safe_process_basename(value: Any) -> Optional[str]:
    text = _safe_text(value, 200)
    if not text:
        return None
    # ``nvidia-smi`` normally returns a basename, but stripping both path
    # styles prevents a remote absolute path from leaking into a sample.
    basename = posixpath.basename(ntpath.basename(text))
    return _safe_display_label(basename)


def short_gpu_uuid(value: Any, length: int = 12) -> str:
    """Return a stable non-full GPU UUID suitable for a timeline row."""

    uuid = _safe_text(value, 256)
    if not uuid:
        return ""
    return uuid[: max(1, int(length))]


def _safe_pid(value: Any) -> Optional[int]:
    try:
        pid = int(value)
    except (TypeError, ValueError):
        return None
    return pid if pid > 0 else None


def _safe_index(value: Any) -> Optional[int]:
    try:
        index = int(value)
    except (TypeError, ValueError):
        return None
    return index if index >= 0 else None


@dataclass(frozen=True)
class GPUProcess:
    """A process row observed on one GPU."""

    gpu_uuid: str
    pid: Optional[int]
    process_basename: Optional[str]
    gpu_index: Optional[int] = None

    @property
    def process_name(self) -> Optional[str]:
        return self.process_basename

    def as_dict(self) -> Dict[str, Any]:
        result: Dict[str, Any] = {
            "gpu_uuid_short": short_gpu_uuid(self.gpu_uuid),
            "pid": self.pid,
            "process_basename": self.process_basename,
        }
        if self.gpu_index is not None:
            result["gpu_index"] = self.gpu_index
        return result


@dataclass(frozen=True)
class GPUSample:
    """One immutable observation row matching the timeline store contract."""

    sampled_at: float
    host: str
    gpu_index: Optional[int]
    gpu_uuid_short: str
    state: str
    task_name: Optional[str] = None
    attribution: str = "none"
    process_basename: Optional[str] = None
    pid: Optional[int] = None
    # Hardware state retains the fact beneath a project-level disabled
    # override.  It is optional so a host-level unknown can remain concise.
    hardware_state: Optional[str] = None
    error: Optional[str] = None

    def __post_init__(self):
        if self.state not in GPU_STATES:
            raise ValueError("unknown GPU state")
        if self.attribution not in ATTRIBUTIONS:
            raise ValueError("unknown GPU attribution")
        if self.gpu_index is not None and self.gpu_index < 0:
            raise ValueError("GPU index must be non-negative")
        if self.pid is not None and self.pid < 1:
            raise ValueError("PID must be positive")

    @property
    def gpu_uuid(self) -> str:
        """Compatibility alias; only the shortened value is persisted."""

        return self.gpu_uuid_short

    @property
    def uuid_short(self) -> str:
        return self.gpu_uuid_short

    def as_dict(self) -> Dict[str, Any]:
        result: Dict[str, Any] = {
            "sampled_at": float(self.sampled_at),
            "host": _safe_text(self.host, 200),
            "gpu_index": self.gpu_index,
            "gpu_uuid_short": _safe_text(self.gpu_uuid_short, 32),
            "state": self.state,
            "task_name": _safe_display_label(self.task_name),
            "attribution": self.attribution,
            "process_basename": _safe_process_basename(self.process_basename),
            "pid": self.pid,
        }
        if self.hardware_state is not None:
            result["hardware_state"] = self.hardware_state
        if self.error is not None:
            result["error"] = _safe_text(self.error, 120)
        return result


# Names used by store-facing integrations.
GPUObservation = GPUSample
GPUProcessObservation = GPUProcess


def _rows(output: str) -> List[List[str]]:
    try:
        reader = csv.reader(io.StringIO(output or ""))
        return [
            [field.strip() for field in row]
            for row in reader
            if any(field.strip() for field in row)
        ]
    except (csv.Error, TypeError, ValueError) as exc:
        raise GPUProbeError("malformed nvidia-smi CSV") from exc


def parse_gpu_csv(output: str) -> List[Dict[str, Any]]:
    """Parse the fixed GPU CSV query without retaining long device strings."""

    rows = _rows(output)
    if not rows:
        raise GPUProbeError("empty GPU inventory")
    result: List[Dict[str, Any]] = []
    seen = set()
    for row in rows:
        if len(row) < 2:
            raise GPUProbeError("malformed GPU inventory row")
        index = _safe_index(row[0])
        uuid = _safe_text(row[1], 256)
        if index is None or not uuid or uuid in seen:
            raise GPUProbeError("invalid GPU inventory row")
        seen.add(uuid)
        result.append(
            {
                "index": index,
                "uuid": uuid,
                "name": _safe_text(row[2], 120) if len(row) >= 3 else "",
                "pci_bus_id": _safe_text(row[3], 120) if len(row) >= 4 else "",
            }
        )
    result.sort(key=lambda item: (item["index"], item["uuid"]))
    return result


def parse_process_csv(output: str) -> List[GPUProcess]:
    """Parse compute-process CSV, returning only basename and PID metadata."""

    stripped = _safe_text(output, 120).lower()
    if not stripped or stripped.startswith("no running processes"):
        return []
    result: List[GPUProcess] = []
    seen = set()
    for row in _rows(output):
        if len(row) < 2:
            raise GPUProbeError("malformed compute-process row")
        uuid = _safe_text(row[0], 256)
        pid = _safe_pid(row[1])
        if not uuid or pid is None:
            raise GPUProbeError("invalid compute-process row")
        key = (uuid, pid)
        if key in seen:
            raise GPUProbeError("duplicate compute-process row")
        seen.add(key)
        result.append(
            GPUProcess(
                gpu_uuid=uuid,
                pid=pid,
                process_basename=_safe_process_basename(row[2] if len(row) >= 3 else None),
            )
        )
    return result


def parse_nvidia_smi_csv(gpu_output: str, process_output: str = "") -> Tuple[List[Dict[str, Any]], List[GPUProcess]]:
    """Parse both fixed CSV responses and reject unknown process UUIDs."""

    gpus = parse_gpu_csv(gpu_output)
    processes = parse_process_csv(process_output)
    known = {item["uuid"] for item in gpus}
    if any(process.gpu_uuid not in known for process in processes):
        raise GPUProbeError("compute process refers to an unknown GPU")
    index_by_uuid = {item["uuid"]: item["index"] for item in gpus}
    return gpus, [
        GPUProcess(
            gpu_uuid=process.gpu_uuid,
            pid=process.pid,
            process_basename=process.process_basename,
            gpu_index=index_by_uuid.get(process.gpu_uuid),
        )
        for process in processes
    ]


Runner = Callable[[Sequence[str]], Any]
StatusProvider = Callable[[], Mapping[str, Any]]


def _run_subprocess(argv: Sequence[str], timeout: float) -> Any:
    try:
        return subprocess.run(
            list(argv),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            check=False,
            shell=False,
            timeout=timeout,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        raise GPUProbeError("remote GPU probe could not be executed") from exc


class GPUProbe:
    """Read-only GPU Steward status and nvidia-smi fallback probe."""

    def __init__(
        self,
        host: Union[str, GPUHostConfig],
        disabled_gpu_indices: Iterable[int] = (),
        runner: Optional[Runner] = None,
        status_provider: Optional[StatusProvider] = None,
        clock: Callable[[], float] = time.time,
        prefer_steward: bool = True,
        timeout_seconds: Optional[float] = None,
        explicit_labels: Optional[Mapping[Union[str, int], str]] = None,
    ):
        if isinstance(host, GPUHostConfig):
            self.config = host
        else:
            self.config = GPUHostConfig(
                name=str(host),
                host=str(host),
                disabled_gpu_indices=tuple(disabled_gpu_indices),
            )
        self.host = self.config.name
        self.runner = runner
        self.status_provider = status_provider
        self.clock = clock
        self.prefer_steward = bool(prefer_steward)
        self.timeout_seconds = float(timeout_seconds or self.config.timeout_seconds)
        self.explicit_labels = dict(explicit_labels or {})
        self.last_ok = True
        self.last_error: Optional[str] = None
        self.last_source: Optional[str] = None

    @property
    def disabled_indices(self) -> Tuple[int, ...]:
        return self.config.disabled_gpu_indices

    def ssh_argv(self, remote_executable: str, *remote_args: str) -> List[str]:
        """Build a fixed remote argv; never use shell interpolation."""

        remote = _safe_text(remote_executable, 120)
        if not remote or any(char.isspace() for char in remote) or remote.startswith("-"):
            raise GPUProbeError("unsafe remote executable")
        argv: List[str] = [self.config.ssh_executable]
        if self.config.port is not None:
            argv.extend(["-p", str(self.config.port)])
        argv.append(self.config.host)
        argv.append(remote)
        for arg in remote_args:
            text = _safe_text(arg, 200)
            if not text or "\x00" in text:
                raise GPUProbeError("unsafe remote argument")
            # All call sites below use fixed flags.  Refuse shell syntax even
            # if a future caller accidentally passes untrusted config data.
            if any(char in text for char in ("\n", "\r", "\x00", ";", "|", "&", "$", "`", ">", "<")):
                raise GPUProbeError("unsafe remote argument")
            argv.append(text)
        return argv

    def _invoke(self, argv: Sequence[str]) -> Any:
        try:
            if self.runner is None:
                result = _run_subprocess(argv, self.timeout_seconds)
            else:
                result = self.runner(list(argv))
        except GPUProbeError:
            raise
        except Exception as exc:
            raise GPUProbeError("remote GPU probe failed") from exc
        if getattr(result, "returncode", None) != 0:
            raise GPUProbeError("remote GPU probe returned a non-zero status")
        return result

    @staticmethod
    def _stdout(result: Any) -> str:
        output = getattr(result, "stdout", "")
        if isinstance(output, bytes):
            try:
                return output.decode("utf-8")
            except UnicodeDecodeError as exc:
                raise GPUProbeError("remote GPU probe returned invalid text") from exc
        if not isinstance(output, str):
            raise GPUProbeError("remote GPU probe returned invalid text")
        return output

    def _query_status(self) -> Mapping[str, Any]:
        try:
            if self.status_provider is not None:
                payload = self.status_provider()
            else:
                result = self._invoke(
                    self.ssh_argv(self.config.steward_executable, "status", "--json")
                )
                try:
                    payload = json.loads(self._stdout(result))
                except (TypeError, ValueError) as exc:
                    raise GPUProbeError("GPU Steward status was not valid JSON") from exc
        except GPUProbeError:
            raise
        except Exception as exc:
            raise GPUProbeError("GPU Steward status probe failed") from exc
        if not isinstance(payload, Mapping):
            raise GPUProbeError("GPU Steward status was not an object")
        inventory = payload.get("inventory", payload)
        if not isinstance(inventory, Mapping) or not isinstance(inventory.get("gpus"), list):
            raise GPUProbeError("GPU Steward status had no inventory")
        return payload

    @staticmethod
    def _task_labels(payload: Mapping[str, Any]) -> Dict[str, str]:
        by_task: Dict[str, str] = {}
        tasks = payload.get("tasks", ())
        if isinstance(tasks, list):
            for task in tasks:
                if not isinstance(task, Mapping):
                    continue
                task_id = _safe_text(task.get("task_id"), 120)
                if not task_id:
                    continue
                label = _safe_display_label(task.get("label")) or _safe_display_label(task_id) or "managed-task"
                by_task[task_id] = label
        return by_task

    @classmethod
    def _explicit_gpu_labels(cls, payload: Mapping[str, Any]) -> Dict[str, Tuple[str, str]]:
        """Map full UUID to ``(task_name, lease_state)`` from status JSON."""

        task_labels = cls._task_labels(payload)
        mapping: Dict[str, Tuple[str, str]] = {}
        leases = payload.get("active_leases", ())
        if not isinstance(leases, list):
            leases = ()
        for lease in leases:
            if not isinstance(lease, Mapping):
                continue
            if lease.get("status", "active") not in ("active", "reserved"):
                continue
            task_id = _safe_text(lease.get("task_id"), 120)
            task_name = task_labels.get(task_id, task_id or "managed-task")
            lease_state = "reserved" if lease.get("pid") is None else "training"
            gpu_uuids = lease.get("gpu_uuids", lease.get("gpu_ids", ()))
            if isinstance(gpu_uuids, list):
                for uuid in gpu_uuids:
                    uuid_text = _safe_text(uuid, 256)
                    if uuid_text:
                        mapping[uuid_text] = (task_name, lease_state)
        # Some status producers place the lease under each task instead of in
        # the top-level active_leases list.
        tasks = payload.get("tasks", ())
        if isinstance(tasks, list):
            for task in tasks:
                if not isinstance(task, Mapping):
                    continue
                lease = task.get("lease")
                if not isinstance(lease, Mapping) or lease.get("status", "active") not in (
                    "active",
                    "reserved",
                ):
                    continue
                task_id = _safe_text(task.get("task_id"), 120)
                task_name = task_labels.get(task_id, task_id or "managed-task")
                lease_state = "reserved" if lease.get("pid") is None else "training"
                gpu_uuids = lease.get("gpu_uuids", lease.get("gpu_ids", ()))
                if isinstance(gpu_uuids, list):
                    for uuid in gpu_uuids:
                        uuid_text = _safe_text(uuid, 256)
                        if uuid_text:
                            mapping[uuid_text] = (task_name, lease_state)
        return mapping

    @classmethod
    def _samples_from_status(
        cls,
        payload: Mapping[str, Any],
        host: str,
        sampled_at: float,
        disabled_indices: Iterable[int] = (),
    ) -> List[GPUSample]:
        inventory = payload.get("inventory", payload)
        gpus = inventory.get("gpus") if isinstance(inventory, Mapping) else None
        if not isinstance(gpus, list) or not gpus:
            raise GPUProbeError("GPU Steward status had an empty inventory")
        disabled = set(disabled_indices)
        explicit = cls._explicit_gpu_labels(payload)
        samples: List[GPUSample] = []
        seen_indices = set()
        seen_uuids = set()
        for row in gpus:
            if not isinstance(row, Mapping):
                raise GPUProbeError("GPU Steward status had a malformed GPU row")
            index = _safe_index(row.get("index"))
            uuid = _safe_text(row.get("uuid"), 256)
            if index is None or not uuid or index in seen_indices or uuid in seen_uuids:
                raise GPUProbeError("GPU Steward status had invalid GPU identity")
            seen_indices.add(index)
            seen_uuids.add(uuid)
            processes = row.get("processes", ())
            process_rows: List[GPUProcess] = []
            if isinstance(processes, list):
                for process in processes:
                    if not isinstance(process, Mapping):
                        continue
                    process_rows.append(
                        GPUProcess(
                            gpu_uuid=uuid,
                            pid=_safe_pid(process.get("pid")),
                            process_basename=_safe_process_basename(
                                process.get("name", process.get("process_name"))
                            ),
                            gpu_index=index,
                        )
                    )
            process_rows.sort(key=lambda item: (item.pid is None, item.pid or 0))
            process = process_rows[0] if process_rows else None
            task_entry = explicit.get(uuid)
            if task_entry is not None:
                task_name, hardware_state = task_entry
                attribution = "explicit"
            else:
                state = _safe_text(row.get("state"), 80)
                if process is not None or state in ("external_busy", "external"):
                    task_name = process.process_basename if process is not None else None
                    hardware_state = "external"
                    attribution = "inferred"
                elif state in ("managed_busy", "managed"):
                    task_name = None
                    hardware_state = "managed-other"
                    attribution = "none"
                elif state in ("free", "idle", ""):
                    task_name = None
                    hardware_state = "idle"
                    attribution = "none"
                else:
                    task_name = None
                    hardware_state = "unknown"
                    attribution = "unknown"
            visible_state = "disabled" if index in disabled else hardware_state
            samples.append(
                GPUSample(
                    sampled_at=sampled_at,
                    host=host,
                    gpu_index=index,
                    gpu_uuid_short=short_gpu_uuid(uuid),
                    state=visible_state,
                    task_name=task_name,
                    attribution=attribution,
                    process_basename=process.process_basename if process else None,
                    pid=process.pid if process else None,
                    hardware_state=hardware_state if visible_state != hardware_state else None,
                )
            )
        return sorted(samples, key=lambda item: (item.gpu_index is None, item.gpu_index or 0))

    def _samples_from_nvidia(
        self,
        gpu_output: str,
        process_output: str,
        sampled_at: float,
        explicit_labels: Optional[Mapping[Union[str, int], str]] = None,
    ) -> List[GPUSample]:
        gpus, processes = parse_nvidia_smi_csv(gpu_output, process_output)
        labels: Dict[str, str] = {}
        merged_labels: Dict[Union[str, int], str] = {}
        merged_labels.update(self.explicit_labels)
        if explicit_labels:
            merged_labels.update(explicit_labels)
        for key, value in merged_labels.items():
            label = _safe_display_label(value)
            if not label:
                continue
            labels[str(key)] = label
        by_gpu: Dict[str, List[GPUProcess]] = {}
        for process in processes:
            by_gpu.setdefault(process.gpu_uuid, []).append(process)
        disabled = set(self.disabled_indices)
        samples: List[GPUSample] = []
        for gpu in gpus:
            uuid = gpu["uuid"]
            index = gpu["index"]
            process_rows = sorted(by_gpu.get(uuid, ()), key=lambda item: (item.pid is None, item.pid or 0))
            process = process_rows[0] if process_rows else None
            label = labels.get(uuid) or labels.get(str(index))
            if label:
                hardware_state = "training" if process is not None else "reserved"
                attribution = "explicit"
                task_name = label
            elif process is not None:
                hardware_state = "external"
                attribution = "inferred"
                task_name = process.process_basename
            else:
                hardware_state = "idle"
                attribution = "none"
                task_name = None
            visible_state = "disabled" if index in disabled else hardware_state
            samples.append(
                GPUSample(
                    sampled_at=sampled_at,
                    host=self.host,
                    gpu_index=index,
                    gpu_uuid_short=short_gpu_uuid(uuid),
                    state=visible_state,
                    task_name=task_name,
                    attribution=attribution,
                    process_basename=process.process_basename if process else None,
                    pid=process.pid if process else None,
                    hardware_state=hardware_state if visible_state != hardware_state else None,
                )
            )
        return samples

    def _query_nvidia(self) -> List[Tuple[str, str]]:
        gpu_result = self._invoke(
            self.ssh_argv(self.config.nvidia_smi_executable, *GPU_QUERY_ARGS)
        )
        process_result = self._invoke(
            self.ssh_argv(self.config.nvidia_smi_executable, *PROCESS_QUERY_ARGS)
        )
        return [(self._stdout(gpu_result), self._stdout(process_result))]

    def sample(
        self,
        sampled_at: Optional[float] = None,
        explicit_labels: Optional[Mapping[Union[str, int], str]] = None,
    ) -> List[GPUSample]:
        """Return a sample list; failures produce one host-level ``unknown`` row."""

        timestamp = float(self.clock() if sampled_at is None else sampled_at)
        if not self.config.enabled:
            self.last_ok = True
            self.last_error = None
            self.last_source = "disabled"
            return []
        try:
            if self.prefer_steward:
                try:
                    payload = self._query_status()
                    samples = self._samples_from_status(
                        payload,
                        host=self.host,
                        sampled_at=timestamp,
                        disabled_indices=self.disabled_indices,
                    )
                    self.last_ok = True
                    self.last_error = None
                    self.last_source = "gpu-steward-status"
                    return samples
                except GPUProbeError:
                    # Status is an optional integration.  Fall back to the
                    # independent, fixed nvidia-smi query without surfacing
                    # status details to a user or timeline record.
                    pass
            outputs = self._query_nvidia()
            samples = self._samples_from_nvidia(
                outputs[0][0], outputs[0][1], timestamp, explicit_labels=explicit_labels
            )
            self.last_ok = True
            self.last_error = None
            self.last_source = "nvidia-smi"
            return samples
        except GPUProbeError:
            self.last_ok = False
            self.last_error = "probe failed"
            self.last_source = "unknown"
            return [
                GPUSample(
                    sampled_at=timestamp,
                    host=self.host,
                    gpu_index=None,
                    gpu_uuid_short="",
                    state="unknown",
                    attribution="unknown",
                    error="probe failed",
                )
            ]

    def probe(self, *args: Any, **kwargs: Any) -> List[GPUSample]:
        """Alias for integrations that call a read-only probe explicitly."""

        return self.sample(*args, **kwargs)

    def query(self, *args: Any, **kwargs: Any) -> List[GPUSample]:
        return self.sample(*args, **kwargs)


NvidiaSMIProbe = GPUProbe
RemoteGPUProbe = GPUProbe
NvidiaSMICollector = GPUProbe
SSHGPUSampler = GPUProbe
parse_nvidia_smi = parse_nvidia_smi_csv


def unknown_sample(host: str, sampled_at: Optional[float] = None) -> GPUSample:
    """Build a host-level unknown row for collectors and tests."""

    return GPUSample(
        sampled_at=time.time() if sampled_at is None else float(sampled_at),
        host=_safe_text(host, 200),
        gpu_index=None,
        gpu_uuid_short="",
        state="unknown",
        attribution="unknown",
        error="probe failed",
    )


__all__ = [
    "ATTRIBUTIONS",
    "GPUProcess",
    "GPUProcessObservation",
    "GPUProbe",
    "GPUProbeError",
    "GPUObservation",
    "GPUSample",
    "GPU_STATES",
    "GPU_QUERY_ARGS",
    "PROCESS_QUERY_ARGS",
    "NvidiaSMIProbe",
    "NvidiaSMICollector",
    "RemoteGPUProbe",
    "SSHGPUSampler",
    "parse_gpu_csv",
    "parse_nvidia_smi",
    "parse_nvidia_smi_csv",
    "parse_process_csv",
    "short_gpu_uuid",
    "unknown_sample",
]
