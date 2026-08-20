# GPU Steward

GPU Steward is a small, single-host GPU queue for AI coding and training
sessions that reach a Linux/NVIDIA server over SSH. It coordinates whole-GPU
leases between cooperating processes, discovers the number of GPUs at runtime,
and keeps GPUs used by unrelated compute processes out of the allocation pool.

The project is alpha software. The scheduling policy is implemented as a pure
Python core, with a standard-library inventory probe, SQLite lease store, and
CLI around it. Validate the installed alpha on the target host before relying
on it for a long job. Do not treat this queue as a security boundary: a
process that bypasses `gpu-steward run` can still use a GPU.

## What it does

GPU Steward is deliberately narrower than Slurm or Kubernetes:

- one Linux host and one Unix user;
- NVIDIA GPUs, allocated as whole devices;
- no root, container runtime, Slurm, Kubernetes, or resident daemon required;
- queue requests describe a minimum and maximum GPU count;
- running jobs are not resized;
- external compute processes are observed and never killed.

The available GPU count is discovered on the server. Nothing in the policy
assumes that a host has four cards.

## Scheduling policy

The default policy is **reserve-then-fair**. Let `N` be the number of GPUs
reported by the server and let `R` be the configured reserve (default `1`, or
`0` when `N == 1`).

1. When the host is completely idle and exactly one request is waiting, it gets
   at most `max(1, N - R)` GPUs. Thus a four-GPU host starts the first request
   with three GPUs and leaves one for a later request.
2. A second request can use the reserved GPU immediately. A request never gets
   more than its `max_gpus` or the number of free devices.
3. If no request can be satisfied, requests stay queued; GPUs are never
   oversubscribed by the coordinator.
4. On a release event, the newly free GPUs are allocated as one batch. The
   planner first gives as many waiting requests as possible their `min_gpus`,
   then distributes remaining GPUs in queue order, round-robin, up to each
   request's `max_gpus`.
5. Priority is considered before age, and equal-priority requests are FIFO.
   `strict_fifo` can be enabled when a deployment prefers queue-head blocking
   over backfilling a smaller request.

The allocation is a launch decision. GPU IDs are selected by the coordinator
and passed to the child through `CUDA_VISIBLE_DEVICES`; callers should request
capacity, never choose card numbers themselves.

### Dynamic examples

These examples use the default reserve and assume the listed devices are
actually free after external-process inspection.

| Host capacity `N` | Waiting requests | Result |
| ---: | --- | --- |
| 1 | `A(min=1)` | `A=1` (the reserve becomes `0`) |
| 4 | `A(min=1,max=auto)` | `A=3`, one GPU remains reserved |
| 4 | `A` is running; `B(min=1)` arrives | `B=1` |
| 8 | `A` alone | `A=7`, one GPU remains reserved |
| 8 | `A` and `B` arrive on an empty host | minimums are granted, then free GPUs are shared round-robin |

For a release batch, the arithmetic is independent of the physical GPU count:

- if three devices become free and two requests are waiting, the grants are
  `2 + 1` (subject to each request's minimum and maximum);
- if one device becomes free and two requests each need one, the oldest
  satisfiable request starts and the other remains queued;
- if one request is waiting, it can receive the whole release batch, limited by
  its `max_gpus` and the reserve rule.

The scheduler returns concrete UUIDs only after the server has queried them.
UUIDs can change when hardware or the driver changes; a failed or inconsistent
inventory is a fail-closed condition, not a reason to guess a device index.

The implementation boundary, state transitions, and concurrency invariants are
described in [architecture and scheduling](docs/architecture-and-scheduling.md).

## Install on the SSH host

Install and run GPU Steward on the server where the GPUs and the cooperating
Codex sessions live. Python 3.8 or newer and a working NVIDIA driver with
`nvidia-smi` are required by the probe layer.

From a checkout on the server:

```bash
python3 -m venv .venv
. .venv/bin/activate
python -m pip install .
gpu-steward doctor
```

The package has no runtime dependency beyond the Python standard library in
the v0.1 design. A virtual environment is recommended so that unrelated
projects do not share the installation.

## Observe Plane timeline

The queue scheduler remains the Control Plane. An optional local Observe Plane
records sanitized Codex lifecycle metadata and read-only GPU state into a
separate SQLite database, then renders a daily horizontal swimlane dashboard.

- Codex work is split into frozen labels such as `research`, `review`,
  `analysis`, `implement`, `test`, `operate`, `waiting-tool`, and
  `suspected-stall`.
- GPU observations are read-only and classify each card as `training`,
  `managed-other`, `external`, `reserved`, `idle`, `disabled`, or `unknown`.
- No prompt, response, command argv, environment variable, credential, or
  training log is persisted.
- Collection is model-free: active GPUs are sampled once per minute, a fully
  idle/disabled host relaxes to once per five minutes, and each probe pass is
  committed in one SQLite transaction. Hook success and failure are silent so
  observability never adds text to the model context.
- Semantic phase commands are deliberately sparse: lifecycle hooks are the
  default signal, and an agent declares only a changed coarse phase rather
  than emitting one command per tool.
- The dashboard binds only to `127.0.0.1` and serves local static assets from
  the package.
- Repeated samples from the same GPU task are rendered as one readable bar.
  Short `idle`/`unknown` blips of at most ten minutes are folded only in the
  display; raw rows and summary totals remain unchanged, and the detail panel
  reports the folded gap count and duration.

For the current frozen product boundary, see
[timeline observability design](docs/timeline-observability-design.zh-CN.md).

### Observe Plane quick start

From this checkout on the local Codex machine:

```bash
python3 scripts/bootstrap_timeline_local.py \
  --host AI3 \
  --project My_Paper_3rd \
  --disabled-gpu 2 \
  --force
```

That helper will:

- write `~/.gpu-steward/timeline.json`;
- link `integrations/codex/gpu-steward-timeline/` into the default personal
  Codex marketplace and run `codex plugin add gpu-steward-timeline@personal`;
- install separate macOS user LaunchAgents for the collector and localhost
  dashboard. Both start at login and are independently removable.

After bootstrap, the common local commands are:

```bash
env PYTHONPATH=src python3 -m gpu_steward.cli timeline sample --config ~/.gpu-steward/timeline.json --json
env PYTHONPATH=src python3 -m gpu_steward.cli timeline report --config ~/.gpu-steward/timeline.json --date 2026-08-18
gpu-steward timeline open
```

`timeline open` verifies the versioned local health endpoint, starts or
repairs the persistent dashboard service when necessary, and opens the default
browser. The user does not need to remember or copy a localhost URL.

Use a new Codex thread after plugin installation so the updated plugin skills
and hooks are picked up cleanly.

## SSH usage

The queue is remote state. Connect to the same host and Unix account for every
cooperating session:

```bash
ssh gpu-host
gpu-steward doctor
gpu-steward inventory --json
gpu-steward status --json
gpu-steward run --min 1 --max auto -- python train.py
```

Read-only checks can also be issued as one remote command:

```bash
ssh gpu-host 'gpu-steward status --json'
```

`run` is a foreground supervisor in v1. If the SSH connection closes, the
foreground process may receive `SIGHUP` and the child may stop; this release
does not promise daemon-style continuation after disconnect. For a long job,
start a persistent terminal such as `tmux` or `screen` on the remote host and
run the complete `gpu-steward run ... -- command` inside that terminal. A
future user-level systemd integration can provide durable background jobs
without changing the scheduling core.

Do not run a GPU command directly in another session while expecting the
coordinator to account for it. If a command must run outside the queue, it is
an external occupant and its GPU remains unavailable to queued jobs until its
compute process disappears.

## CLI surface

The v0.1 command surface is intentionally small:

```text
gpu-steward [--db PATH] [--reserve COUNT] [--strict-fifo] doctor [--json]
gpu-steward [--db PATH] [--reserve COUNT] [--strict-fifo] inventory [--json]
gpu-steward [--db PATH] [--reserve COUNT] [--strict-fifo] status [--json]
gpu-steward [--db PATH] [--reserve COUNT] [--strict-fifo] run \
  [--json] [--min COUNT] [--max auto|COUNT] [--priority INT] [--label TEXT] \
  [--cwd PATH] [--wait-timeout SECONDS] -- COMMAND [ARG ...]
gpu-steward [--db PATH] cancel TASK_ID [--json]
gpu-steward [--db PATH] gc [--json]
gpu-steward timeline init --config PATH --host AI3 [--project NAME] [--disabled-gpu INDEX]
gpu-steward timeline hook [--timeline-db PATH]
gpu-steward timeline phase PHASE [--timeline-db PATH] [--project NAME]
gpu-steward timeline sample --config PATH [--json]
gpu-steward timeline collect-loop --config PATH
gpu-steward timeline report --config PATH --date YYYY-MM-DD [--format json|csv] [--output PATH]
gpu-steward timeline serve --config PATH [--port 8765]
gpu-steward timeline open [--config PATH]
gpu-steward timeline dashboard install|start|stop|status|uninstall --config PATH
gpu-steward timeline collector install|start|stop|status|uninstall --config PATH
```

- `doctor` checks the local prerequisites and reports actionable failures.
- `inventory` reports the runtime GPU inventory; `--json` is retained as the
  explicit machine-output switch.
- `status` reports external occupants, queued requests, active leases, and
  terminal task state; `--json` is the machine-readable form.
- `run` queues a request, waits when necessary, starts the command with its
  lease environment, and releases that lease when the command exits. The
  child keeps stdout/stderr; the supervisor result is JSON on stderr.
- `cancel` targets a task started by the same Unix user through GPU Steward; it
  must not be used as a general process killer.
- `gc` reclaims leases whose owner identity is no longer alive. It does not
  terminate an unknown process.

The current alpha emits JSON for these status/control commands even when the
optional `--json` flag is omitted; scripts should still pass the flag and
validate `schema_version`. `--db` selects a private SQLite path, `--reserve`
sets the solo-job reserve, and `--strict-fifo` disables smaller-request
backfilling behind an infeasible queue head.

Use `gpu-steward <command> --help` on the installed version for options added
by that release. The scheduler accepts a request with `min_gpus`, optional
`max_gpus`, `priority`, and creation time; labels and working-directory
metadata are optional request fields in the stateful layer.

## JSON and status

Machine consumers should use `--json`, check `schema_version`, and fail on an
unknown version. Human-oriented output is not an API. The versioned status
payload is designed around these concepts:

```json
{
  "schema_version": 1,
  "tasks": [
    {
      "task_id": "task-...",
      "status": "running",
      "exit_code": null,
      "lease": {
        "lease_id": "lease-...",
        "status": "active",
        "gpu_uuids": ["GPU-..."]
      }
    }
  ],
  "queue": [],
  "active_leases": [],
  "active_gpu_uuids": [],
  "ok": true,
  "inventory": {
    "count": 4,
    "gpus": [
      {"index": 0, "uuid": "GPU-...", "state": "free", "processes": []},
      {"index": 1, "uuid": "GPU-...", "state": "external_busy", "processes": [{"pid": 1234, "name": "python"}]}
    ]
  },
  "external_busy_gpu_uuids": ["GPU-..."],
  "free_gpu_uuids": ["GPU-..."]
}
```

The current runtime coordinator's `status_payload()` exposes this status shape.
Its nested `InventorySnapshot.as_dict()` contains `count` plus `gpus` with
`index`, `uuid`, `name`, `pci_bus_id`, `memory_total_mib`, `state`, and observed
process records. The CLI entry point serializes this coordinator payload after
installation.

The exact payload may gain fields, but an existing field keeps its meaning
within a schema version. `external_busy` means that a compute process was
observed without a valid GPU Steward lease; the coordinator does not infer
that it is safe to reuse or terminate that process. A queued task has no GPU
allocation yet. A task's `task_id`, allocation, and exit code are the evidence
to include in a Codex handoff.

## Codex integration

The repository includes [a Codex skill](integrations/codex/SKILL.md) that makes
the queue boundary explicit. Install it into the skills directory used by the
Codex environment, for example:

```bash
mkdir -p ~/.codex/skills/gpu-steward
cp integrations/codex/SKILL.md ~/.codex/skills/gpu-steward/SKILL.md
```

For every GPU task, the agent should first run `doctor` and `status --json`,
inspect external occupants, then wrap the actual command in `gpu-steward run`.
The agent reports the task ID, assigned GPU UUIDs, queue/running state, exit
code, and any failure evidence. It must not hand-pick GPUs or kill processes
that it does not own.

`examples/safe-run.sh` is a small wrapper for humans or Codex-generated shell
steps. It performs the read-only checks and forwards the caller's command to
the scheduler without accepting a GPU ID.

## Safety and failure boundaries

- **Whole-device coordination, not isolation.** Cooperative commands are
  assigned whole GPUs. The queue cannot prevent a user from bypassing it.
- **External occupants are protected.** A compute process without a valid lease
  marks its GPU `external_busy`; GPU Steward never signals an external process,
  runs `nvidia-smi --gpu-reset`, or guesses that low utilization means free
  memory. It may escalate from `SIGTERM` to `SIGKILL` only for the exact
  managed process group it launched and whose leader PID start-time identity
  still matches.
- **No shell interpolation.** The supervisor should execute the command as an
  argv vector, not with `shell=True`; command arguments remain the caller's
  arguments.
- **Lease identity is defensive.** Cleanup must use process identity including
  the Linux `/proc/<pid>/stat` start time so a reused PID cannot release another
  task's lease.
- **Fail closed.** GPU disappearance, UUID inconsistency, probe failure, or a
  state/lock failure pauses new allocation and reports an error.
- **User-private state.** The SQLite state and lock belong to the Unix user
  running the queue. SSH keys are never read, copied, or stored by GPU
  Steward.

## Scope and limitations

The first release does not provide multi-user ACLs, multi-host scheduling, a
web UI, MIG/time-slicing management, preemption, live world-size changes, or
durable execution after an SSH disconnect. It assumes a Linux host, NVIDIA
drivers, one Unix user, and commands that can accept `CUDA_VISIBLE_DEVICES`.
It also does not promise that every framework uses every assigned GPU; the
allocation is a resource grant, not a training configuration.

## Development verification

The pure scheduler can be tested without a GPU or an SSH server:

```bash
PYTHONPATH=src python3 -m unittest discover -s tests -v
python3 -m compileall -q src tests
git diff --check
```

The tests cover dynamic capacities, the four-GPU `3 + 1` example, release-batch
fairness, maximums, and external-busy capacity. A real host verification must
also run `gpu-steward doctor` and a read-only `inventory --json`; do not use a
live training process as a smoke test when other work is on the server.

The v0.1 release was also smoke-tested from its wheel on Linux/Python 3.8.10
against a four-GPU NVIDIA host while two GPUs had external training processes.
`doctor`, `inventory`, `status`, a CPU-only supervised command, environment
binding, and final lease release all passed without signalling those occupants.

## References

The design follows the resource-discovery and device-binding ideas in the
following primary documentation, while remaining intentionally smaller than
those systems:

- [Slurm GRES GPU scheduling](https://slurm.schedmd.com/gres.html)
- [Ray accelerator scheduling](https://docs.ray.io/en/latest/ray-core/scheduling/accelerators.html)
- [Kubernetes GPU scheduling](https://kubernetes.io/docs/tasks/manage-gpus/scheduling-gpus/)
- [Kubernetes Dynamic Resource Allocation](https://kubernetes.io/docs/concepts/scheduling-eviction/dynamic-resource-allocation/)
- [NVIDIA NVML API](https://docs.nvidia.com/deploy/nvml-api/nvml-api-reference.html)
- [`CUDA_VISIBLE_DEVICES`](https://docs.nvidia.com/cuda/cuda-programming-guide/05-appendices/environment-variables.html)

## License

GPU Steward is released under the [Apache License 2.0](LICENSE).
