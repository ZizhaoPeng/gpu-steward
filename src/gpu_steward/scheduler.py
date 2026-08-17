"""Pure scheduling policy for GPU Steward.

The scheduler assigns a snapshot of currently free GPUs to waiting jobs.  It
does not resize running jobs: every returned allocation is a launch decision.
"""

from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Tuple


@dataclass(frozen=True)
class Request:
    """A queued request ordered by priority, age, then stable identifier."""

    request_id: str
    min_gpus: int = 1
    max_gpus: Optional[int] = None
    priority: int = 0
    created_at: float = 0.0

    def capacity(self, total_gpus: int) -> int:
        maximum = total_gpus if self.max_gpus is None else self.max_gpus
        if self.min_gpus < 1:
            raise ValueError("min_gpus must be at least 1")
        if maximum < self.min_gpus:
            raise ValueError("max_gpus must be greater than or equal to min_gpus")
        return min(maximum, total_gpus)


@dataclass(frozen=True)
class Allocation:
    """A launch decision binding one request to concrete GPU UUIDs."""

    request_id: str
    gpu_ids: Tuple[str, ...]


def plan_allocations(
    *,
    free_gpu_ids: Sequence[str],
    waiting: Iterable[Request],
    total_gpus: int,
    active_jobs: int = 0,
    external_busy_gpus: int = 0,
    reserve_gpus: int = 1,
    strict_fifo: bool = False,
) -> List[Allocation]:
    """Plan one atomic allocation batch.

    On a completely idle host with one waiter, ``reserve_gpus`` remains free.
    Otherwise the batch starts as many feasible jobs as possible, grants each
    its minimum, then distributes remaining GPUs round-robin by queue order.
    """

    free = list(free_gpu_ids)
    if len(free) != len(set(free)):
        raise ValueError("free_gpu_ids must be unique")
    if total_gpus < 1:
        raise ValueError("total_gpus must be at least 1")
    if len(free) > total_gpus:
        raise ValueError("free GPU count cannot exceed total_gpus")
    if min(active_jobs, external_busy_gpus, reserve_gpus) < 0:
        raise ValueError("job, busy GPU, and reserve counts cannot be negative")

    queue = sorted(
        list(waiting),
        key=lambda item: (-item.priority, item.created_at, item.request_id),
    )
    capacities = {item.request_id: item.capacity(total_gpus) for item in queue}
    if not free or not queue:
        return []

    if active_jobs == 0 and external_busy_gpus == 0 and len(queue) == 1:
        request = queue[0]
        solo_budget = max(1, total_gpus - min(reserve_gpus, total_gpus - 1))
        grant = min(len(free), capacities[request.request_id], solo_budget)
        if grant < request.min_gpus:
            return []
        return [Allocation(request.request_id, tuple(free[:grant]))]

    remaining = len(free)
    grants: Dict[str, int] = {}
    selected: List[Request] = []
    for request in queue:
        if request.min_gpus <= remaining:
            selected.append(request)
            grants[request.request_id] = request.min_gpus
            remaining -= request.min_gpus
        elif strict_fifo:
            break

    while remaining and selected:
        progressed = False
        for request in selected:
            request_id = request.request_id
            if grants[request_id] < capacities[request_id]:
                grants[request_id] += 1
                remaining -= 1
                progressed = True
                if remaining == 0:
                    break
        if not progressed:
            break

    allocations: List[Allocation] = []
    cursor = 0
    for request in selected:
        count = grants[request.request_id]
        allocations.append(
            Allocation(request.request_id, tuple(free[cursor : cursor + count]))
        )
        cursor += count
    return allocations
