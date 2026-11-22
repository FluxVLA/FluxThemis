# ZMQ 评测启动指南

用户分别在准备好的模型环境和仿真环境执行命令。入口不提交云任务、不登录服务器、
不建立 SSH 隧道、不安装环境。FluxVLA 负责预测；Themis 负责场景、episode、进度、
监控和恢复。同机与远端使用同一套评测代码，模型地址只决定 ZMQ 连接目标。

## 两端独立启动

模型端可沿用 FluxVLA 原生命令，见 [LIBERO](../docs/libero.md#local-run)
或 [RoboDojo](../docs/robodojo.md#five-episode-smoke-test)。在模型环境中，也可从
Themis 目录使用包装入口：

```bash
bash scripts/distributed_model.sh --fluxvla-root /path/to/FluxVLA \
  --config /path/to/model.py --ckpt-path /models/model.safetensors \
  --gpu-ids 0,1 --workers-per-gpu 2 --scheduling dynamic
```

它复用 FluxVLA 的模型加载和服务，不复制模型实现。需要对应 FluxVLA 版本的
`distributed_zmq_server.py` 支持 `--workers-per-gpu` 和副本到 GPU 的映射。
模型种类由模型配置决定；checkpoint、tokenizer、statistics 和预处理均在模型端准备。

RoboTwin 的 predict 和 sim 启动示例见 [RoboTwin 文档](../docs/robotwin.md)。

仿真端运行统一入口：

```bash
bash scripts/run_evaluation.sh \
  --config /path/to/evaluation.py \
  --server-endpoint tcp://MODEL_HOST:15555 \
  --gpu-ids 0,1 --workers-per-gpu 3 --scheduling dynamic \
  --output /persistent/evaluations
```

`--config` 接受 Themis Python 评测配置或启动 YAML。现有 FluxVLA 配置也可传入，
入口只保存其中的 Themis 评测部分，不加载模型。Python 配置须包含
`themis.evaluation` 和 `themis.transport`。YAML 示例及字段见
[RoboDojo 文档](../docs/robodojo.md#yaml-launch)和
[RoboTwin 文档](../docs/robotwin.md#yaml-launch)。

- `--workers-per-gpu` 只控制当前命令所在端，默认 1；两端 GPU 可以不同。
- 仿真统一入口默认使用一张可见 GPU；`--gpus` 或 `--gpu-ids` 覆盖。
  GPU ID/UUID 必须属于已有 `CUDA_VISIBLE_DEVICES` 分配。
- `--scheduling` 显式覆盖配置；配置未设置时为 dynamic。
  dynamic 允许两端 worker 数不同；fixed 要求数量相等及模型服务支持固定调度。
- `--cfg-options KEY=VALUE ...` 只覆盖 `themis.evaluation` / `themis.transport`。
  模型配置覆盖项应传给模型命令。
- `--smoke` 对 LIBERO 选择 task 0，对 RoboDojo 选择 `put_bottles_into_dustbin`，
  对 RoboTwin 选择当前设置的 `adjust_bottle`（all 使用 clean），局数等于 sim 进程数
  乘以环境配置的 `num_envs`（默认 1）。
  `--full` 选择全部任务，保留 Python 配置的逐任务配额。
- `--config` 可传多份配置，依次连接已启动的模型。若配置需要不同 checkpoint，
  用户应分别启动对应模型并分次评测，入口不会自动更换权重。
- `--dry-run` 只解析配置、生成计划和快照，不检查已安装仿真环境，不连接模型或使用 GPU。
  实际启动时会做环境预检；完整验证仍需 smoke。

## 单个模拟器的多个环境

支持原生多环境的适配器可以在一个 worker 内同时推进多个 episode。在 Python
评测配置中，通过 `themis.evaluation.environment.num_envs` 设置容量，
worker 将环境的实际容量通过 ready 消息上报给调度器。`workers-per-gpu` 仍表示
模拟器进程数；例如 2 个 worker、每个
`num_envs=4`，最多同时执行 8 个 episode。模型服务仍独立启动，模型请求按
episode 分别发送，不要求模型一次推理整个环境 batch。

每次分配最多包含 `num_envs` 个 episode，允许尾批。环境通过 ready
消息中的 `requires_single_task` 声明是否要求同任务组批；RoboDojo 和 RoboLab
声明此约束。
例如一个 RoboDojo 任务配额为 25、容量为 10 时，分成 10、10、5；空槽位不启动
episode，也不执行推理。dynamic 和 fixed 调度都按容量分配；fixed 只从该 worker
固定配对的 episode 中取 job，重试 job 也可以组批。
RoboDojo 的 `_random` 任务受 `sim.scene.clutter_env_limit`（默认 5）限制，
调度时按 worker 容量与该任务上限的较小值组批。

`run_evaluation.sh --episode-timeout-s SECONDS` 覆盖 YAML 的
`simulator.episode_timeout_s`；未配置时默认 3600 秒。该参数限制整组
assignment 的总运行时间，包括 reset、模型推理、
环境执行和录像收尾。增加 slot 数不会增加超时预算，完成一个 slot 也不会延长
剩余 slot 的期限。选择容量时应为整组的串行模型请求预留时间。

每次分配有独立的执行编号，旧进程或旧分配的迟到消息不影响新分配。同一次分配中，
已结束、被拒或中断的 episode 再上报开始、结果、拒绝或中断事件，会作为协议错误报错。
worker 重连或场景中断后重跑未完成的 episode，开始事件仍由 worker 正常发送，
coordinator 按 episode ID 去重；reset 尚未完成就中断的 episode 在重跑后首次上报开始。
结果先被 reporter 确认，才计入完成数。
每个 slot 分别累计步数、成功状态、Score 和动作片段，结束后立即上报并收尾录像；
整组结束后 worker 才接收下一组。episode ID 和 seed 按完整任务列表生成，
不随 worker 数量或完成顺序变化。策略返回 `PolicyStop` 时只结束对应 slot，
其余 slot 继续执行。策略自行声明完成的记录与环境判定的成功结果分别保存。

`duration_s` 从 reset 和 policy client 初始化完成后开始计时，到该 episode 的
录像等收尾完成为止，包含执行期间等待其他 slot 推理和处理的时间，不包含分配前
排队和 reset；多个 slot 的计时会重叠，不能将其总和视为整个评测的墙钟耗时。
比较并发配置时，应结合整段墙钟时间计算的吞吐率；纯模型推理耗时可按 episode
汇总请求 profile 中的 `server_infer_s`，前提是模型服务提供了有效的计时。

reset 或 step 阶段拒绝 case 时，coordinator 都为该 slot 分配未使用的 layout，
优先补足被拒的 episode。被拒样本不计入成功、失败、Score 或 episode 耗时，
不消耗 `--simulator-restart-limit` 的进程重启预算，也没有额外的拒绝次数上限。
phase 只用于日志。被拒 slot 的录像单独丢弃。
RoboDojo 原生调用正常返回的逐 slot 拒绝，以及取观测前检查发现的拒绝，允许健康
slot 继续运行。如果原生 reset 或动作执行因可恢复异常中断，其他未完成 slot 的
初始化或动作也可能只执行了一部分。这些 slot 标记为 interrupted，丢弃本次尝试及
录像，原 job 使用相同 seed 和 layout 优先重跑；环境在重跑前关闭并重置原生场景，
清空残留控制状态。中断尝试不计分、不计入 episode 耗时，也不消耗进程重启预算。
已经完成的 episode 结果和录像保留。
正常失败和达到步数上限仍是完成的评测结果。
现有 `--resume` 流程可以重放已确认的结果，再执行剩余 episode。

适配器接入要求：

- `BaseEnvironment` 只要求 `num_envs`、`reset_batch(requests)`、
  `step_batch(actions)` 和 `finalize_episode(slot, ...)`；slot 必须显式传入。
  只实现 scalar reset/step 的旧环境继承 `SingleEnvironmentAdapter`，由它转换为
  长度为 1 的 batch，并将原 scalar `render()` 改名为 `_render()`。
  LIBERO、RoboTwin、RoboCasa、Gymnasium 和 mock 使用该适配器。
  RoboDojo 和 RoboLab 直接实现 batch 接口，支持单个 worker 内的多个场景。
- reset 输入是长度为 `num_envs` 的列表，活动 slot 是 `EnvironmentResetRequest`，
  包含任务、episode index、环境 seed 和 case index。输出按 slot 返回
  `EnvironmentReset`，包含 observation 和 info；空 slot 的输入和输出
  都是 `None`。
- step 输入每 slot 一个动作，正常执行返回 `StepResult`；输出列表与输入等长。
  空闲或已结束 slot 的输入和输出都是 `None`。适配器
  负责冻结或隔离它们，不能影响健康 episode，也不得隐式 auto-reset。
- reset 和 step 都通过 `CaseRejection(info)` 拒绝 case；无法完成、但应保留
  原 case 重跑时，返回 `EpisodeInterruption(info)`。这两种类型与正常结果互斥，
  都不携带观测或 reward。所有结果类型定义在 `fluxthemis.structures`；
  `EnvironmentReset` 仍可从原有的 `fluxthemis.environments` 或其 `base` 模块导入。
  适配器负责确保重跑前清理本次尝试留下的场景和控制状态。
- 唯一执行入口是 `execute_assignment`。返回列表保留 slot 位置，被拒或未使用的
  slot 以及中断的尝试为 `None`，其余为 `EpisodeResult`。
- 每个 slot 单独结束或丢弃录像。通用录制器调用 `render(slot)`；原生录制器在
  `finalize_episode(slot, ...)` 中收尾，并按 `case_rejected` 或 `interrupted`
  终止原因丢弃录像。

RoboDojo 在单 env 和多 env 下都通过原生 `reset`、`get_obs_batch` 和
`take_action_batch` 执行，在一个 Isaac Sim 进程中加载独立 layout，通过 None
seed 禁用尾批空 slot。`MockVectorEnvironment` 提供 CPU 示例。

## 连接与端口

同机模型使用 `tcp://127.0.0.1:15555`；远端使用仿真进程可访问的模型地址。
已建立 SSH 隧道时填仿真主机的转发地址。`127.0.0.1` 也可能通向远端隧道，入口
不会据此启动模型。SSH 公网端口与 ZMQ 模型端口不同。

模型默认 frontend/backend 为 15555/15556；模型 torchrun 默认 29500。
仿真统一入口使用协调端口 15558、torchrun 15559，YAML 的
`simulator.coordinator_port` 可调整。直接调用 `distributed_eval.sh` 时，torchrun
默认 29501。不同任务独立提供各自的 rank/master 参数。

## 直接运行分布式仿真

已有多节点流程可以继续直接使用原有入口：

```bash
bash scripts/distributed_eval.sh --config /path/to/evaluation.py \
  --server-endpoint tcp://MODEL_HOST:15555 \
  --gpu-ids 0,1 --workers-per-gpu 3 --scheduling dynamic
```

统一入口管理一台仿真主机；多节点通过此命令配置 `WORLD_SIZE`、`RANK`、
`MASTER_ADDR`、`MASTER_PORT`。模型与仿真不得共用对方的 rank/master 设置。
`NPROC_PER_NODE` 若显式指定，应等于可见 GPU 数 × workers-per-gpu。

所有 shell 使用当前 Python，可通过 `FLUXTHEMIS_PYTHON` / `FLUXVLA_PYTHON`
指定解释器，不自动激活 Conda。RoboDojo 的 Isaac/资产预检和 LIBERO 的环境检查
按配置中的环境类型选择；阶段、事件、资源采样、结果与恢复使用公共模块。

## 脚本目录

`scripts/` 顶层保留启动、安装等命令。内部辅助模块按职责放入子目录：

- `scripts/evaluation/`：配置、进度、GPU/CPU 监控、进程管理和恢复。
- `scripts/preflight/`：仿真环境、资产和 layout 预检查，以及 RoboDojo / RoboTwin 专有检查。
- `scripts/robotwin/`：RoboTwin 原生依赖辅助工具，由安装和预检入口调用。

评测和预检入口由 LIBERO、RoboCasa、RoboDojo 与 RoboTwin 共用；使用时仍执行上面的顶层命令。

## 进度和结果

评测时终端原地刷新 `Episodes [进度条] 已完成/总数 ETA HH:MM:SS`。
完成数汇总所有 worker 已确认结束的 episode，也包含恢复的历史结果；不按成功数计数。
ETA 按本轮实际执行速度估算，尚未完成足够回合时显示 `--:--:--`。
重定向输出每分钟记录一次进度，状态切换和完成时立即记录。
资源详情和完整性能报告写入文件。

入口打印 `Results: ...`。输出由仿真端 `--output` 或 YAML `run.output_root`
决定，不要求位于 FluxVLA 仓库。每份评测包含：

- `plan.json`、`config_snapshot.py`：启动设置与冻结的评测配置。
- `simulator.log`、`events.jsonl`：仿真日志与模型已确认的事件。
- `progress.json`：最新阶段、完成局数和 `eta_s`；`progress.log`：精简的进度记录。
- `result.json`：完整性检查、成功数和模型报告位置。
- `gpu_samples.csv`、`process_samples.jsonl`、`performance_report.json/txt`：
  GPU 基线/峰值、CPU/PSS、实际执行窗口及推理等待分析。

正式 benchmark 聚合报告仍在模型端，本地事件和结果不依赖读取其文件系统。
远端 GPU 采样是可选输入；缺失时不会据此判断模型 GPU 瓶颈。
恢复使用 `--resume /path/to/stopped-run` 或 YAML `resume.from_dir`，细节见
[RoboDojo 恢复说明](../docs/robodojo.md#resume)。
