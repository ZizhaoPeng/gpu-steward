# GPU Steward Timeline：观测插件设计（冻结 MVP）

更新时间：2026-08-18

## 目标与边界

Timeline 在本机自动记录 Codex 工作阶段与远端 NVIDIA GPU 状态，并按
`Asia/Singapore` 自然日生成横向泳道。采集、分类、聚合与页面渲染均由本地脚本
执行，不使用周期性模型调用，不读取隐藏思维链，也不保存提示词、回复、完整命令、
环境变量或训练日志。

MVP 扩展现有 GPU Steward，但不改变其队列、租约或外部进程保护语义。测试项目
`/Users/pengzizhao/workspace/My_Paper_3rd` 只允许只读观测；不得启动训练、修改项目、
迁移或终止进程。该项目的 GPU 2 必须显示为 `disabled`。

## 组件

1. `timeline/store.py`：独立的私有 SQLite 数据库，保存不可变事件、GPU 样本和人工
   override。默认路径 `~/.gpu-steward/timeline.sqlite3`，文件模式 `0600`。
2. `timeline/codex.py`：接收官方 Codex Hook JSON；只保留经过白名单筛选的元数据。
3. `timeline/gpu.py`：用 argv 形式调用 `ssh HOST nvidia-smi ...`，解析 CSV，不使用
   shell 插值。失败写入 `unknown` 样本并指数退避。
4. `timeline/aggregate.py`：将事件和样本转换为半开区间 `[start, end)`，在本地日界
   切分，计算 Codex 小时、GPU 小时和重叠时间。
5. `timeline/web.py`：仅绑定 `127.0.0.1` 的标准库 HTTP 服务；页面使用本地 HTML、
   CSS 和 JavaScript，不加载 CDN 或第三方遥测。
6. `integrations/codex/gpu-steward-timeline/`：个人 Codex 插件包；默认 hooks 写入事件，
   skill 指示 Codex 在语义阶段变化时调用本地 `phase` 命令。

## 稳定词表

Codex 阶段由 `timeline/phases.py` 冻结：

`research`、`review`、`analysis`、`implement`、`test`、`operate`、
`active-unspecified`、`waiting-tool`、`waiting-user`、`suspected-stall`、`idle`。

GPU 状态由观测层冻结：

`training`、`managed-other`、`external`、`reserved`、`idle`、`disabled`、`unknown`。

阶段来源必须是 `declared`、`hook-rule` 或 `inferred`，并携带 0 到 1 的置信度。
未知值报错，不猜测成功。低瞬时利用率不能把仍有计算进程的 GPU 改成空闲。

## 事件与接口契约

所有时间均以 UTC Unix 秒存储，展示时才转换时区。原始事件使用调用方提供的稳定
`event_id` 或对安全字段计算 SHA-256 后生成的幂等 ID。

### Codex 事件

```text
record_codex_event(event_id, occurred_at, session_id, turn_id, project,
                   kind, phase, source, confidence, tool_category,
                   tool_active)
```

- `session_id`、`turn_id` 在落库前变成带本地 salt 的短哈希；不保存原值。
- `cwd` 只映射到项目 basename/配置别名；不保存任意外部路径。
- `PreToolUse` 开启 `waiting-tool` 区间，匹配的 `PostToolUse` 关闭它。
- turn 未结束且10分钟没有事件时才派生 `suspected-stall`；活跃工具期间不派生停滞。
- `Stop` 后进入 `waiting-user`；下一次 `UserPromptSubmit` 结束该状态。

### GPU 样本

```text
record_gpu_sample(sampled_at, host, gpu_index, gpu_uuid_short, state,
                  task_name, attribution, process_basename, pid)
```

- 显式 GPU Steward label 是权威 `task_name`。
- 外部进程名只能是 basename，任务名标记为推断。
- 项目配置中的禁用索引覆盖硬件空闲显示，但不改写硬件事实。
- SSH/驱动/解析失败生成主机级 `unknown`，不得沿用旧样本伪装当前状态。

### 日报

`build_day(date, timezone, project)` 返回版本化 JSON：

```json
{
  "schema_version": 1,
  "date": "2026-08-18",
  "timezone": "Asia/Singapore",
  "generated_at": 0,
  "lanes": [
    {"id": "codex:...", "kind": "codex", "label": "Codex", "segments": []},
    {"id": "gpu:AI3:0", "kind": "gpu", "label": "AI3 / GPU 0", "segments": []}
  ],
  "summary": {
    "codex_active_seconds": 0,
    "codex_waiting_seconds": 0,
    "codex_stalled_seconds": 0,
    "gpu_training_seconds": 0,
    "gpu_idle_seconds": 0,
    "overlap_seconds": 0
  }
}
```

区间必须落在当天 `[00:00, 次日00:00)` 内，排序、不重叠、持续时间非负。人工
override 不删除原记录，日报聚合时采用当前有效 override。

## CLI 与后台进程

CLI 计划为：

```text
gpu-steward timeline init
gpu-steward timeline hook                 # 从 stdin 读取官方 Hook JSON
gpu-steward timeline phase PHASE          # Codex 主动声明阶段
gpu-steward timeline sample --config ~/.gpu-steward/timeline.json
gpu-steward timeline collect-loop
gpu-steward timeline report --date YYYY-MM-DD [--format json|csv]
gpu-steward timeline serve [--port 8765]
gpu-steward timeline open
gpu-steward timeline dashboard install|start|stop|status|uninstall
gpu-steward timeline collector install|start|stop|status|uninstall
```

macOS 使用两个相互独立的用户级 LaunchAgent 保持 `collect-loop` 与本地 Dashboard；
无需 root。Dashboard 只监听 `127.0.0.1:8765`，登录后自动启动。用户或 Codex 运行
`gpu-steward timeline open` 即可校验服务、必要时修复启动并打开默认浏览器，无需记忆
或复制地址。关闭页面不停止 Dashboard 或采集器。

采集采用低资源自适应节奏：存在训练或外部进程时每60秒采一次；整机仅有
`idle`/`disabled` 时放宽到每300秒，探测失败则按1/2/5/10分钟退避。一次探测的
多张GPU记录在单个SQLite事务中提交。
Codex hook 使用进程替换而非再创建子进程，并保持 stdout/stderr 静默；采集、聚合和
页面渲染均不调用模型，因此不会产生模型 token。

## 展示

页面默认显示一天：横轴为00:00–24:00；每个 Codex session 和每张 GPU 各一条泳道。
并行区间真实叠加。同一GPU上相同占用类型和任务族的连续采样合成长条；两段相同任务
之间不超过10分钟的 `idle`/`unknown` 只在展示层折叠。原始记录与汇总不变，点击长条
可查看时间跨度、实际观察时长、短间隙数量和总时长。推断的 Python 版本名统一显示为
“Python 训练/计算进程”。`inferred` 使用斜纹；`disabled`、`unknown` 与 `idle` 颜色必须不同。
顶部显示 GPU 训练/空闲 GPU·小时、Codex 主动/等待/停滞小时及两者同时运行时长。

## 验收

1. Hook 开始、阶段、工具等待和停止事件可幂等落库。
2. AI3 只读采集能区分四张卡，GPU 2 按项目配置显示禁用。
3. 显式任务名优先，外部任务明确标记推断。
4. 日报正确处理并行、跨午夜、缺测和 override。
5. 10分钟停滞规则不会把进行中的长工具调用误报为停滞。
6. 单元测试、CLI smoke、真实 Hook smoke、AI3 只读采集和浏览器视觉检查通过。
7. 数百条同任务采样压成少量长条，短间隙元数据可见且训练汇总不膨胀。
8. Dashboard LaunchAgent 在登录态常驻；`timeline open` 无需用户输入 URL 且不产生重复服务。

原始事件的90天滚动清理与长期日报汇总属于安装后的运维策略；在安全清理机制落地前，
当前版本只追加、不自动删除，避免清理边界破坏跨日区间或审计证据。该项不影响实时
采集和日报，但会使数据库持续增长，必须作为下一维护门补齐。

## 明确不做

- 不读取或推断隐藏思维链。
- 不调用模型做周期采集或日报聚合。
- 不开放公网、不上传云端、不引入分析遥测。
- 不自动启动、重试、迁移、抢占或终止训练。
- 不把 transcript JSONL 当作稳定实时接口；历史导入仅作为后续显式 `inferred` 功能。
