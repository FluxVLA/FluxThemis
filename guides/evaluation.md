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
  对 RoboTwin 选择当前设置的 `adjust_bottle`（all 使用 clean），局数等于 sim 进程数。
  `--full` 选择全部任务，保留 Python 配置的逐任务配额。
- `--config` 可传多份配置，依次连接已启动的模型。若配置需要不同 checkpoint，
  用户应分别启动对应模型并分次评测，入口不会自动更换权重。
- `--dry-run` 只解析配置、生成计划和快照，不检查已安装仿真环境，不连接模型或使用 GPU。
  实际启动时会做环境预检；完整验证仍需 smoke。

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
