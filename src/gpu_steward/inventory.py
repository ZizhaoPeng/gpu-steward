"""NVIDIA GPU inventory and compute-process discovery.

The runtime deliberately uses ``nvidia-smi`` rather than assuming a fixed
number of cards.  A malformed or failed query is an error: allocating from a
partial inventory could duplicate a UUID or place a job on an unknown card.
"""

import csv
import io
import os
import subprocess
import time
from dataclasses import dataclass
from typing import Callable, Iterable, List, Optional, Sequence, Tuple

from .errors import InventoryError


@dataclass(frozen=True)
class GPUDevice:
    """One physical, whole-GPU allocation unit."""

    index: int
    uuid: str
    name: str = ""
    pci_bus_id: str = ""
    memory_total_mib: Optional[int] = None


@dataclass(frozen=True)
class ComputeProcess:
    """A process reported by NVIDIA as using a compute GPU."""

    gpu_uuid: str
    pid: int
    process_name: str = ""


@dataclass(frozen=True)
class InventorySnapshot:
    """A single consistent pair of GPU and compute-process queries."""

    gpus: Tuple[GPUDevice, ...]
    processes: Tuple[ComputeProcess, ...]
    queried_at: float

    @property
    def gpu_uuids(self) -> Tuple[str, ...]:
        return tuple(item.uuid for item in self.gpus)

    @property
    def gpu_count(self) -> int:
        return len(self.gpus)

    def as_dict(
        self,
        external_uuids: Optional[Iterable[str]] = None,
        managed_uuids: Optional[Iterable[str]] = None,
    ):
        external = set(external_uuids or ())
        managed = set(managed_uuids or ())
        processes_by_gpu = {}
        for process in self.processes:
            processes_by_gpu.setdefault(process.gpu_uuid, []).append(
                {"pid": process.pid, "name": process.process_name}
            )
        return {
            "count": self.gpu_count,
            "gpus": [
                {
                    "index": gpu.index,
                    "uuid": gpu.uuid,
                    "name": gpu.name,
                    "pci_bus_id": gpu.pci_bus_id,
                    "memory_total_mib": gpu.memory_total_mib,
                    "state": (
                        "external_busy"
                        if gpu.uuid in external
                        else ("managed_busy" if gpu.uuid in managed else "free")
                    ),
                    "processes": processes_by_gpu.get(gpu.uuid, []),
                }
                for gpu in self.gpus
            ],
        }


CommandRunner = Callable[[Sequence[str]], subprocess.CompletedProcess]


def _default_runner(argv: Sequence[str]) -> subprocess.CompletedProcess:
    try:
        return subprocess.run(
            list(argv),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            check=False,
        )
    except OSError as exc:
        raise InventoryError("failed to execute nvidia-smi: {}".format(exc)) from exc


class NvidiaSMI:
    """Query NVIDIA inventory using the stable CSV interface.

    ``runner`` is injectable for deterministic tests.  It receives argv and
    returns a ``subprocess.CompletedProcess``-compatible object.
    """

    GPU_QUERY = (
        "--query-gpu=index,uuid,name,pci.bus_id,memory.total"
        " --format=csv,noheader,nounits"
    )
    PROCESS_QUERY = (
        "--query-compute-apps=gpu_uuid,pid,process_name"
        " --format=csv,noheader,nounits"
    )

    def __init__(
        self,
        executable: str = "nvidia-smi",
        runner: Optional[CommandRunner] = None,
    ):
        self.executable = executable
        self.runner = runner or _default_runner

    def query(self) -> InventorySnapshot:
        gpu_argv = [
            self.executable,
            "--query-gpu=index,uuid,name,pci.bus_id,memory.total",
            "--format=csv,noheader,nounits",
        ]
        process_argv = [
            self.executable,
            "--query-compute-apps=gpu_uuid,pid,process_name",
            "--format=csv,noheader,nounits",
        ]
        gpu_result = self._run(gpu_argv)
        process_result = self._run(process_argv)
        gpus = self._parse_gpus(gpu_result.stdout or "", gpu_argv)
        processes = self._parse_processes(
            process_result.stdout or "", process_argv
        )
        known = {gpu.uuid for gpu in gpus}
        unknown = sorted({item.gpu_uuid for item in processes} - known)
        if unknown:
            raise InventoryError(
                "nvidia-smi reported compute processes on unknown GPU UUID(s): {}".format(
                    ", ".join(unknown)
                )
            )
        return InventorySnapshot(
            gpus=tuple(gpus), processes=tuple(processes), queried_at=time.time()
        )

    def _run(self, argv: Sequence[str]):
        try:
            result = self.runner(argv)
        except InventoryError:
            raise
        except OSError as exc:
            raise InventoryError("failed to execute {}: {}".format(argv[0], exc)) from exc
        except Exception as exc:
            raise InventoryError("inventory command failed: {}".format(exc)) from exc
        returncode = getattr(result, "returncode", None)
        if returncode != 0:
            stderr = (getattr(result, "stderr", "") or "").strip()
            detail = stderr or "exit code {}".format(returncode)
            raise InventoryError(
                "{} failed: {}".format(" ".join(argv), detail)
            )
        return result

    @staticmethod
    def _rows(output: str, argv: Sequence[str]):
        try:
            rows = list(csv.reader(io.StringIO(output)))
        except csv.Error as exc:
            raise InventoryError(
                "malformed CSV from {}: {}".format(" ".join(argv), exc)
            ) from exc
        return [[field.strip() for field in row] for row in rows if any(field.strip() for field in row)]

    @classmethod
    def _parse_gpus(cls, output: str, argv: Sequence[str]) -> List[GPUDevice]:
        rows = cls._rows(output, argv)
        if not rows:
            raise InventoryError("nvidia-smi returned an empty GPU inventory")
        gpus: List[GPUDevice] = []
        seen = set()
        for row in rows:
            if len(row) < 2:
                raise InventoryError(
                    "malformed GPU inventory row from {}: {!r}".format(
                        " ".join(argv), row
                    )
                )
            try:
                index = int(row[0])
            except ValueError as exc:
                raise InventoryError("invalid GPU index: {!r}".format(row[0])) from exc
            uuid = row[1]
            if not uuid or uuid in seen:
                raise InventoryError("missing or duplicate GPU UUID: {!r}".format(uuid))
            seen.add(uuid)
            memory = None
            if len(row) >= 5 and row[4] and row[4].lower() not in {"n/a", "[not supported]"}:
                try:
                    memory = int(float(row[4]))
                except ValueError as exc:
                    raise InventoryError("invalid GPU memory value: {!r}".format(row[4])) from exc
            gpus.append(
                GPUDevice(
                    index=index,
                    uuid=uuid,
                    name=row[2] if len(row) >= 3 else "",
                    pci_bus_id=row[3] if len(row) >= 4 else "",
                    memory_total_mib=memory,
                )
            )
        gpus.sort(key=lambda item: (item.index, item.uuid))
        return gpus

    @classmethod
    def _parse_processes(
        cls, output: str, argv: Sequence[str]
    ) -> List[ComputeProcess]:
        stripped = output.strip()
        if not stripped or stripped.lower().startswith("no running processes"):
            return []
        rows = cls._rows(output, argv)
        processes: List[ComputeProcess] = []
        for row in rows:
            if len(row) < 2:
                raise InventoryError(
                    "malformed compute-process row from {}: {!r}".format(
                        " ".join(argv), row
                    )
                )
            try:
                pid = int(row[1])
            except ValueError as exc:
                raise InventoryError("invalid compute-process PID: {!r}".format(row[1])) from exc
            if pid < 1 or not row[0]:
                raise InventoryError("invalid compute-process row: {!r}".format(row))
            processes.append(
                ComputeProcess(
                    gpu_uuid=row[0], pid=pid, process_name=row[2] if len(row) >= 3 else ""
                )
            )
        return processes


class StaticInventory:
    """Small inventory adapter useful for tests and local embedding."""

    def __init__(self, gpus: Iterable[GPUDevice], processes: Iterable[ComputeProcess] = ()):
        self.gpus = tuple(gpus)
        self.processes = tuple(processes)

    def query(self) -> InventorySnapshot:
        if len({gpu.uuid for gpu in self.gpus}) != len(self.gpus):
            raise InventoryError("static inventory has duplicate GPU UUIDs")
        known = {gpu.uuid for gpu in self.gpus}
        unknown = sorted({item.gpu_uuid for item in self.processes} - known)
        if not self.gpus:
            raise InventoryError("static inventory is empty")
        if unknown:
            raise InventoryError("static inventory has unknown process UUID(s)")
        return InventorySnapshot(
            gpus=tuple(sorted(self.gpus, key=lambda item: (item.index, item.uuid))),
            processes=self.processes,
            queried_at=time.time(),
        )
