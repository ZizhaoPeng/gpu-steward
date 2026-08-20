---
name: gpu-steward-timeline
description: 提供通过本地 GPU Steward Observe Plane 查看 Codex 工作阶段和 GPU 任务占用时间线、打开仪表板及稀疏声明语义阶段的操作指导。
---

# GPU Steward Timeline

GPU Steward has a Control Plane for safe queueing and an Observe Plane for
local timeline reporting. Use this skill to make a semantic phase visible or
open the daily dashboard. The integration records only sanitized lifecycle
metadata; never copy prompts, responses, hidden reasoning, full commands,
environment variables, credentials, or training logs into a timeline event.

## Declare a semantic phase sparingly

Lifecycle hooks already provide the low-cost baseline. Do not call a phase
command for every tool or narrate successful collection. Declare only a
coarse semantic transition whose label would otherwise be lost, and only when
the phase actually changes. Use one of the frozen phase names:

```bash
gpu-steward timeline phase research
gpu-steward timeline phase review
gpu-steward timeline phase analysis
gpu-steward timeline phase implement
gpu-steward timeline phase test
gpu-steward timeline phase operate
gpu-steward timeline phase active-unspecified
gpu-steward timeline phase waiting-tool
gpu-steward timeline phase waiting-user
gpu-steward timeline phase suspected-stall
gpu-steward timeline phase idle
```

Do not invent a phase label. If the phase is not clear, use
`active-unspecified` rather than inferring intent from prompt text.

## Open the local dashboard

The Observe Plane dashboard is local-only. Prefer the one-command entry; it
starts the persistent service when needed and opens the default browser:

```bash
gpu-steward timeline open
```

Do not ask the user to remember or copy a localhost URL. The dashboard shows
one day of parallel Codex and GPU lanes, merges repeated same-task samples into
readable bars, and keeps short-gap details available on click. Closing the page
does not stop the collector or dashboard service.
