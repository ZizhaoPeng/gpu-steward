# GPU Steward 调研与设计报告

日期：2026-08-17  
状态：v1 决策冻结前报告

## 1. 目标

GPU Steward 面向通过 SSH 使用一台多卡 Linux/NVIDIA 服务器的多个 Codex 会话。它在服务器端集中维护 GPU 清单、排队和租约，保证参与调度的会话不会重复占卡，并尽量让空闲 GPU 立刻投入工作。

首版聚焦“单台服务器、同一 Unix 用户、整卡独占”。GPU 数量必须在运行时查询，不能写死为 4。

## 2. 已核验环境

- 实机验证服务器是 Linux x86_64，Python 3.8.10，支持用户级 systemd。
- `nvidia-smi` 当前发现 4 张 RTX 4090，每张约 24 GiB。
- 检查时两张 GPU 有当前用户的训练进程、另两张空闲。这验证了“外部进程感知”是必要功能。
- 4 张卡之间没有 NVLink；因此首版不应假设连续编号优于其他组合，但应保留拓扑评分接口。

环境状态随时会变化。上述数据只用于验证设计，不写入默认配置。

## 3. 同类方案结论

| 方案 | 借鉴内容 | 不直接采用的原因 |
|---|---|---|
| Slurm GRES | 自动发现、整卡租约、为任务设置 `CUDA_VISIBLE_DEVICES`、作业/资源状态 | 单机个人服务器部署和运维偏重 |
| Ray | 任务声明 GPU 数量，由运行时绑定可见设备 | 需要任务进入 Ray 运行时，不适合任意现有 shell/训练命令 |
| Kubernetes + NVIDIA 插件/DRA | 资源声明、设备隔离、可观测性 | 对 SSH 单机工作流过重；DRA 的部分能力仍在快速演进 |
| NVIDIA time-slicing/MIG | 单卡共享和硬件隔离 | 4090 不提供通用 MIG 路径；time-slicing 不提供显存/故障隔离，首版应坚持整卡独占 |

权威资料：

- [Slurm GRES GPU scheduling](https://slurm.schedmd.com/gres.html)
- [Ray accelerator scheduling](https://docs.ray.io/en/latest/ray-core/scheduling/accelerators.html)
- [Kubernetes GPU scheduling](https://kubernetes.io/docs/tasks/manage-gpus/scheduling-gpus/)
- [Kubernetes Dynamic Resource Allocation](https://kubernetes.io/docs/concepts/scheduling-eviction/dynamic-resource-allocation/)
- [NVIDIA NVML](https://docs.nvidia.com/deploy/nvml-api/nvml-api-reference.html)
- [CUDA_VISIBLE_DEVICES](https://docs.nvidia.com/cuda/cuda-programming-guide/05-appendices/environment-variables.html)

## 4. 调度语义

### 4.1 请求模型

每个任务声明：

- `min_gpus`：能启动的最小 GPU 数，默认 1；
- `max_gpus`：最多使用数，默认全部可调度 GPU；
- `priority`：默认 0；同优先级按 FIFO；
- 命令、工作目录和可选标签。

分配发生在任务启动前。已启动任务不会被动态缩放，因为普通 PyTorch/CUDA 训练通常无法在运行中安全改变 world size。

### 4.2 默认策略：reserve-then-fair

设当前可调度 GPU 总数为 `N`，默认保留量 `R = 1`（`N = 1` 时为 0）。

1. 系统完全空闲且只有一个新任务时，给它最多 `N - R` 张；因此 4 卡机器首任务得到 3 张。
2. 第二个任务可立即使用保留的 1 张。
3. 没有可用卡时继续排队，不超卖。
4. 一次释放事件后，把当时全部空闲卡作为一个批次分给等待任务：先保证尽可能多的任务各获 `min_gpus`，再按 FIFO 轮转分发余卡，且不超过各自 `max_gpus`。
5. 例：已有一个 1 卡任务，另一个 3 卡任务结束，两个任务等待，则释放的 3 卡按 2+1 分配；若只释放 1 卡，则最老的可满足任务启动，另一个继续等待。
6. 若只等待一个任务，它可拿到本次全部空闲卡，但仍受 `max_gpus` 和“完全空闲时保留 1 卡”规则约束。

该策略实现用户给出的 3+1 和批量均分，同时避免简单 FIFO 中大任务长期挡住所有 1 卡任务。可配置 `strict_fifo=true` 恢复严格队首阻塞。

### 4.3 外部占用

调度前查询 `nvidia-smi`/NVML 的 compute process。任何没有有效 GPU Steward 租约但存在计算进程的 GPU 标记为 `external_busy`，不可分配，也不会被终止。进程消失后自动重新纳入资源池。

首版采用“进程存在即占用”，而不是依赖瞬时 utilization 或显存阈值；后两者会把暂时处于数据加载阶段的训练误判为空闲。

### 4.4 公平性

- 主排序：更高 priority；次排序：等待时间。
- v1 使用静态 priority + FIFO；priority aging 留作后续可选策略，避免首版引入难以解释的动态优先级。
- 批次内余卡按轮转分配，而不是全给第一个任务。
- 调度决策、租约、任务退出状态均可查询，便于 Codex 给出证据。

## 5. 架构

首版采用“无常驻守护进程的远端协调器”：

```text
Codex session A --SSH-->
                         gpu-steward CLI -> file lock -> SQLite state
Codex session B --SSH-->                    |          -> nvidia-smi
                                            +-> child process with GPU UUIDs
```

- 所有会话访问服务器用户目录下同一个 SQLite 数据库。
- Linux `flock`/SQLite 事务保证入队、批量分配和释放原子化。
- `run` 进程监督子进程并在退出后释放租约；每次命令/轮询都会回收 PID 已不存在的陈旧租约。
- 使用 GPU UUID 设置 `CUDA_VISIBLE_DEVICES`，并设置 `CUDA_DEVICE_ORDER=PCI_BUS_ID`、`GPU_STEWARD_GPU_COUNT` 和租约 ID。
- 不要求 root、Docker、Slurm 或 Kubernetes。
- 后续如需要断开 SSH 后继续运行，可增加用户级 systemd daemon；状态与调度核心保持不变。

## 6. CLI 与机器接口

计划提供：

```bash
gpu-steward doctor
gpu-steward inventory [--json]
gpu-steward status [--json]
gpu-steward run --min 1 --max auto -- python train.py
gpu-steward cancel TASK_ID
gpu-steward gc
```

程序消费的 JSON 使用版本化 schema；错误返回非零退出码。Codex 集成通过仓库内 `integrations/codex/SKILL.md` 教会会话：先 `doctor/status`，再把 GPU 命令包在 `run` 中，绝不自行挑卡或终止未知进程。

## 7. 安全与失败边界

- 不读取或复制 SSH 私钥；SSH 仍由用户现有配置管理。
- 不自动杀死外部进程。`cancel` 默认只作用于当前用户通过 GPU Steward 启动的任务。
- 对受管任务的停止以其独立进程组为边界；可按需从 `SIGTERM` 升级到 `SIGKILL`，但不会扩展到未知进程。
- 子进程使用 argv 执行，不经 `shell=True`，降低命令注入风险。
- 数据库和锁文件权限设为当前用户私有。
- 进程身份使用 PID + Linux `/proc/<pid>/stat` start time，避免 PID 复用误回收。
- GPU 消失、UUID 变化或 `nvidia-smi` 失败时 fail closed：暂停新分配并明确报错。

## 8. 首版验收标准

1. 模拟 1、2、4、8 卡环境时均从查询结果决定容量，无硬编码 4。
2. 4 卡空闲时：首任务分配 3，第二任务分配 1，第三任务排队。
3. 3 卡释放且两个任务等待时：得到 2+1；1 卡释放且两个任务等待时：只启动一个。
4. 单个等待任务能获得本次全部可用卡（受最大值和保留规则约束）。
5. 外部占用 GPU 永不被分配；进程结束后可重新使用。
6. 并发入队/释放不会产生重复 UUID 租约。
7. 命令实际收到正确的 `CUDA_VISIBLE_DEVICES` 和计数环境变量；退出后资源释放。
8. `status --json` 通过 schema 测试，陈旧租约可恢复。
9. NVIDIA Linux 实机上完成只读 doctor/inventory 和不占 GPU 的调度 smoke；不得干扰现有训练。

## 9. 暂定选择与待反馈项

这些选择不会阻塞可逆的首版实现：

- 名称：**GPU Steward**，仓库 `gpu-steward`。
- 许可证：Apache-2.0（允许商用，同时保留专利条款）。
- 语言：Python 标准库优先，兼容服务器现有 Python 3.8。
- 首版范围：单服务器、单 Unix 用户、NVIDIA 整卡独占；多服务器和多用户 ACL 后续扩展。
- 默认策略：首任务 `N-1`（最少 1），其余按释放批次公平分配；v1 可改变 reserve 和 strict FIFO，priority aging 留作后续版本。

需要在进入稳定版前确认，但不妨碍当前原型的问题：是否需要跨 Unix 用户共享队列、是否把断开 SSH 后继续运行列为 v1 必须项、是否需要 Web UI。建议答案分别是“暂不”“v1.1”“暂不”。
