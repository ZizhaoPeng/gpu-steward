---
name: gpu-steward
description: Coordinate GPU work on an SSH-connected NVIDIA host through GPU Steward.
---

# GPU Steward for Codex

Use this skill whenever a task will start, inspect, or stop a GPU workload on a
remote host. GPU Steward is a cooperative queue, not a process-isolation or
security boundary. Keep the queue boundary visible in every command and in the
final task report.

## Required startup checks

Before starting any GPU workload on the remote host:

```bash
gpu-steward doctor
gpu-steward status --json
```

If either command is unavailable, returns an error, or reports an invalid GPU
inventory, stop the GPU task and report the blocker. Do not silently fall back
to hand-picked `CUDA_VISIBLE_DEVICES` values.

Inspect the status payload. A GPU marked `external_busy` is occupied by a
compute process without a valid GPU Steward lease and must remain unavailable.
Read-only `nvidia-smi`/NVML inspection is allowed when needed to understand that
status. Never infer that a card is free from low utilization alone.

## Launch rule

Every GPU command must be wrapped by `gpu-steward run`. Ask for capacity, not a
card number:

```bash
gpu-steward run --min 1 --max auto -- python train.py --epochs 10
gpu-steward run --min 2 --max 2 -- torchrun --nproc-per-node=2 train.py
```

This applies to Python training, `torchrun`, Accelerate, JAX, custom CUDA
programs, evaluation scripts, and any command that may initialize CUDA. Do not
run the underlying command directly in a second shell. Do not set or override
`CUDA_VISIBLE_DEVICES`, `CUDA_DEVICE_ORDER`, or a GPU index manually; the
supervisor supplies the lease environment after selecting UUIDs.

The child keeps its stdout/stderr. The wrapper's machine-readable completion
payload is emitted on stderr (use `--json` explicitly in generated commands),
so capture that stream when recording the task ID, allocation, and exit code.

If the request needs a specific minimum or maximum, express that in
`--min`/`--max`. If the requirement is unknown, use the smallest safe minimum
and `--max auto`, and state the assumption in the task report. Do not change a
running job's GPU count; submit a new request if the workload must be relaunched
with a different world size.

## Queue and completion handling

`gpu-steward run` may wait in the queue. Do not bypass the wait by launching a
second unwrapped command. While diagnosing a wait, use:

```bash
gpu-steward status --json
```

Only cancel a task that this Codex session owns and has a task ID for:

```bash
gpu-steward cancel TASK_ID
```

Never run `kill`, `pkill`, `killall`, `nvidia-smi --gpu-reset`, or an equivalent
command against an unknown process. A task that is not owned by this session is
not a task to clean up. `gpu-steward gc` may repair stale records, but it must
not be used to terminate an external occupant.

After the command exits, capture a final status and report the command's exit
code. A successful wrapper return is not enough if the child wrote an error or
the inventory changed during launch.

## SSH disconnect boundary

The v1 supervisor is foreground. An SSH disconnect can send `SIGHUP` and stop
the child; GPU Steward does not promise durable background execution. For a
long-running workload, start a remote `tmux` or `screen` session first, then
run the complete `gpu-steward run ... -- command` inside it. Do not detach only
the child while leaving the queue supervisor outside the persistent terminal.

## Required report

For every GPU task, include these fields in the Codex response or handoff:

- remote host and Unix account (do not include secrets or private keys);
- GPU Steward task ID;
- requested `min`/`max` capacity;
- queue/running/terminal state;
- assigned GPU UUIDs from status, or explicitly `none` while queued;
- child exit code and relevant stderr/stdout evidence;
- any `external_busy` GPUs observed;
- whether SSH persistence (`tmux`/`screen`) was used;
- failures, cancellation, or an unverified cleanup state.

Treat `status --json` as a versioned machine interface. Check
`schema_version`, reject unknown versions, and do not claim an allocation based
on human-oriented output or an old status snapshot.

## Non-GPU commands

Read-only repository inspection, source editing, unit tests that do not load
CUDA, and ordinary SSH commands can run without GPU Steward. If a command might
initialize CUDA, wrap it by default. Never mix a queued GPU command with a
manual GPU selection in the same shell step.
