# GPU Steward architecture and scheduling

This document is the English companion to
[`research-and-design.zh-CN.md`](research-and-design.zh-CN.md). It describes the
v0.1 contract around the pure scheduler and separates policy that is already
anchored in source from the stateful integration that is still alpha.

## Boundary and invariants

GPU Steward coordinates cooperative processes on one Linux/NVIDIA host for one
Unix user. It allocates whole GPUs and does not resize a running process. A
caller asks for capacity; the coordinator chooses the concrete GPU UUIDs and
sets `CUDA_VISIBLE_DEVICES` for the child.

The stateful implementation must preserve these invariants:

1. GPU capacity comes from a runtime inventory, never from a host-specific
   constant such as `4`.
2. A GPU UUID cannot belong to two live GPU Steward leases at the same time.
3. A GPU with a compute process and no valid lease is `external_busy` and is not
   eligible for a new allocation.
4. Allocation, lease creation, and queue removal are one atomic decision.
5. Release is idempotent and must not release a lease belonging to a reused PID.
6. Probe, UUID, lock, or state failures fail closed: no new allocation is made
   from an uncertain snapshot.
7. Unknown processes are never killed by `cancel`, garbage collection, or
   normal reconciliation.

This is coordination, not a security boundary. A user can bypass the CLI and
start a process with a manually selected GPU; the process is then an external
occupant from GPU Steward's point of view.

## Component model

```text
Codex session A -- SSH --> gpu-steward CLI -- lock/transaction --> state DB
Codex session B -- SSH -->         |                  |
                                   |                  +--> inventory probe
                                   +--> pure planner --> child supervisor
```

The intended v0.1 layers are:

- **Inventory probe:** asks the NVIDIA stack for the current GPU UUIDs and
  compute processes. The current `NvidiaSMI` adapter uses
  `nvidia-smi --query-gpu=index,uuid,name,pci.bus_id,memory.total` and
  `--query-compute-apps=gpu_uuid,pid,process_name`; NVML is a natural future
  backend.
- **State store:** a per-user SQLite database plus a file lock. It records
  requests, leases, process identity, and terminal results. The database and
  lock are private to the Unix user.
- **Planner:** the pure `plan_allocations` function in
  `src/gpu_steward/scheduler.py`. It receives a snapshot and returns a batch of
  launch decisions; it does not inspect hardware or mutate state.
- **Supervisor:** starts the command with an argv vector, injects the assigned
  UUIDs and lease metadata into its environment, waits for exit, and releases
  the lease.
- **CLI:** exposes `doctor`, `inventory`, `status`, `run`, `cancel`, and `gc`.
  Each mutating operation repeats reconciliation before committing its
  decision so stale leases do not block the queue forever.

The probe, state store, and supervisor are integration surfaces. Their exact
implementation can evolve, but they must not change the planner's policy or
the safety invariants above without a versioned design decision.

## Pure scheduler API

The current anchor is deliberately small:

```python
from typing import Iterable, List, Sequence

from gpu_steward.scheduler import Allocation, Request, plan_allocations

plan_allocations(
    *,
    free_gpu_ids: Sequence[str],
    waiting: Iterable[Request],
    total_gpus: int,
    active_jobs: int = 0,
    external_busy_gpus: int = 0,
    reserve_gpus: int = 1,
    strict_fifo: bool = False,
) -> List[Allocation]
```

`Request` contains `request_id`, `min_gpus` (default `1`), optional
`max_gpus`, `priority`, and `created_at`. `Allocation` contains a
`request_id` and an ordered tuple of concrete GPU IDs. The function validates
unique free IDs, positive total capacity, and minimum/maximum consistency.

The returned list is a plan only. A stateful caller must commit it under its
lock/transaction, launch the child, and record the process identity before
another caller can rely on the lease.

## Reserve-then-fair algorithm

Requests are ordered by descending priority, then ascending creation time, then
stable request ID. Let `F` be the snapshot of free UUIDs and `N` be the queried
total GPU count.

```text
validate N, F, requests, and non-negative counters
order waiters by priority, age, request ID

if no free GPU or no waiter:
    return no allocations

if no active job, no external busy GPU, and exactly one waiter:
    budget = max(1, N - min(reserve_gpus, N - 1))
    grant min(|F|, waiter.max_gpus or N, budget)
    return one allocation if waiter.min_gpus is satisfied

remaining = |F|
select each waiter whose minimum fits in remaining
    (stop at the first infeasible request when strict_fifo is enabled)
grant each selected waiter its minimum
while remaining and a selected waiter can grow:
    walk selected waiters in queue order
    give one GPU to each request below its maximum
slice F in the resulting grant sizes
return allocations
```

The stateful layer supplies `F` after subtracting active leases and external
occupants. `external_busy_gpus` is retained as an explicit input because an
otherwise idle host must not apply the single-job reserve rule as though those
devices were available to GPU Steward.

### Capacity examples

The policy scales with `N`:

- `N=1`, one request: reserve is zero, so the request receives one GPU.
- `N=4`, empty host, one request: grant `3`, retain `1`.
- `N=4`, one three-GPU job running, one request: the reserved free GPU grants
  `1` immediately.
- three UUIDs released while two one-GPU waiters are queued: minimum grants are
  assigned first, then the remaining UUID is given by the round-robin pass,
  producing `2 + 1`.
- one UUID released while two one-GPU waiters are queued: only the oldest
  satisfiable waiter starts.
- with two or more waiters on an empty `N=8` host, minimums are granted to as
  many feasible requests as possible and the remainder is shared in queue
  order, subject to each `max_gpus`.

The planner does not move a running job when a new request arrives. Any change
to `CUDA_VISIBLE_DEVICES`, distributed world size, or framework process group
would be a new launch decision.

## Runtime sequence

### `run`

1. Parse a command as argv after `--`; do not invoke a shell string.
2. Acquire the per-user lock and reconcile process identities, expired/stale
   records, and the current inventory.
3. Insert a queued request with a fresh task ID and request metadata.
4. Plan a batch from the same inventory snapshot. Commit any selected lease
   atomically with the request transition to `starting`.
5. Launch the child with:
   - `CUDA_VISIBLE_DEVICES` containing the allocated UUIDs;
   - `CUDA_DEVICE_ORDER=PCI_BUS_ID`;
   - `GPU_STEWARD_GPU_COUNT` set to the allocation size;
   - the GPU Steward task/lease ID for status and cleanup.
6. Record the child PID and Linux process start time before releasing the lock.
7. Supervise the child in the foreground. On exit, record the exit code and
   release its UUIDs in an idempotent transaction.
8. Reconcile and plan the next release batch so waiting work can start.

If a launch fails after the lease is committed, the supervisor must record the
failure and release that lease. If the inventory changes before launch, it
must discard the plan and retry from a fresh snapshot rather than guessing
which UUID is still valid.

### `status` and `inventory`

Read-only commands still reconcile enough state to report stale leases and
external occupants, but they do not terminate unknown processes. `inventory`
describes discovered devices. `status` joins device state with queued, running,
and terminal tasks.

The current `InventorySnapshot.as_dict()` probe shape contains `count` and a
`gpus` list with `index`, `uuid`, `name`, `pci_bus_id`, `memory_total_mib`,
`state`, and observed `processes`. The stateful CLI status shape is a separate,
versioned contract. JSON consumers should check `schema_version` there. A
queued task has no allocation;
a running task has the UUIDs committed to its lease; an external busy device
has no GPU Steward task and should be treated as unavailable.

### `cancel` and `gc`

`cancel TASK_ID` is scoped to a task created by the current Unix user through
GPU Steward. It may request termination of that owned child and release its
lease after identity checks. It is not a general `kill` wrapper.

`gc` repairs records whose recorded PID and process start time are no longer
alive. It can release a stale lease record; it must never infer ownership from
PID alone and must never kill a process merely because a record is stale.

## External process reconciliation

The probe should enumerate compute processes per GPU. For every process:

1. match the process to a live GPU Steward lease by owner, PID, and process
   start time;
2. if it matches, retain the lease's GPU state;
3. otherwise mark the GPU `external_busy` and leave the process untouched.

Do not use instantaneous utilization or a memory threshold as the primary
definition of occupancy. A training process can be loading data, synchronizing,
or between kernels while still requiring exclusive access. When the process
disappears, a later reconciliation can return the UUID to `free`.

If the probe fails, the UUID list changes unexpectedly, or a device vanishes,
the safe result is an error plus no new allocation. Existing child processes
are not killed by this failure path.

## Concurrency and persistence

Two SSH sessions can call the CLI at the same time. The stateful layer must
serialize the read-modify-plan-commit sequence with a file lock and SQLite
transaction (or an equivalent atomic mechanism). A plan computed outside the
critical section is only a candidate and must be discarded if the inventory or
lease version changed before commit.

The v1 design is intentionally foreground. Disconnecting SSH may send `SIGHUP`
to the supervisor and child; it is not a guarantee of durable background
execution. Users who need continuity should run the full command inside a
remote `tmux`/`screen` session. A future user-level systemd daemon can reuse
the same state and planner contracts.

## Verification map

The scheduler tests are expected to cover:

- `N=1`, `N=2`, `N=4`, and larger dynamically supplied inventories;
- first-job reserve and second-job use of the reserve;
- `3 + 1` release fairness and one-free-GPU queue behavior;
- minimum/maximum constraints, priority/age ordering, and strict FIFO;
- external-busy GPUs never being allocated;
- unique UUIDs under concurrent enqueue/release decisions.

Host verification is read-only first: run `gpu-steward doctor`, then
`gpu-steward inventory --json` and `gpu-steward status --json`. Do not use a
live training command as a smoke test while unrelated work is running.

## Related primary documentation

- [Slurm GRES GPU scheduling](https://slurm.schedmd.com/gres.html)
- [Ray accelerator scheduling](https://docs.ray.io/en/latest/ray-core/scheduling/accelerators.html)
- [Kubernetes GPU scheduling](https://kubernetes.io/docs/tasks/manage-gpus/scheduling-gpus/)
- [Kubernetes Dynamic Resource Allocation](https://kubernetes.io/docs/concepts/scheduling-eviction/dynamic-resource-allocation/)
- [NVIDIA NVML API](https://docs.nvidia.com/deploy/nvml-api/nvml-api-reference.html)
- [`CUDA_VISIBLE_DEVICES`](https://docs.nvidia.com/cuda/cuda-programming-guide/05-appendices/environment-variables.html)
