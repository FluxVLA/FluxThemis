# LIBERO Simulator Integration

For a clean installation and the exact commands to launch both repositories,
see the [FluxVLA + FluxThemis LIBERO quickstart](quickstart.md). This document
describes the adapter contract and implementation details.

## Execution Boundaries

The LIBERO process runs only the simulator. It passes raw observations, the task prompt, and
episode information to the current ROS 1 `FluxVLAROSModelClient` (which can be replaced with
`FluxVLAROS2ModelClient` after migration), and accepts only `float32 [T, 7]` actions that have
already been restored to environment units. The following logic remains the responsibility of
the FluxVLA server:

- the 180-degree image rotation and resizing in `ProcessLiberoEvalInputs`;
- quaternion/state processing in `LiberoProprioFromInputs`;
- tokenization, model inference, and action-chunk generation;
- action denormalization with checkpoint statistics and gripper handling in
  `DenormalizeLiberoAction`.

The adapter therefore does not modify observation arrays or clip, flip, or denormalize actions.

## Installation

```bash
cd /root/projects/FluxThemis
bash scripts/install_libero.sh \
  --env-name fluxthemis \
  --fluxvla-root /root/projects/FluxVLA
```

The installer requires Python 3.10 or 3.12 in the target environment. The current primary
`fluxthemis` environment uses Python 3.10 and communicates with the unupgraded FluxVLA through
ROS 1. A separate Python 3.12 environment is used only for ROS 2 Jazzy integration testing and
migration. The installer pins the following simulator versions to match FluxVLA:

- LIBERO `058fda1ddebe92918af091cb6816759ca6d003f0`;
- robosuite `e293cc32ff3c48957a4ebcad09952432b0dc9049`;
- MuJoCo 3.2.6, BDDL 1.0.1, and Gym 0.25.2.

The source trees reside under the target Conda prefix, not in the FluxThemis repository. This
allows each simulator to retain its own dependencies and assets while keeping a single copy of
the FluxThemis code. The installer refuses to overwrite a managed checkout that is dirty or at
the wrong commit.

After the environment is activated, the LIBERO configuration defaults to
`$CONDA_PREFIX/etc/fluxthemis/libero/config.yaml`. Do not let it fall back to the shared
`~/.libero/config.yaml`; doing so can silently load assets from another environment or from a
FluxVLA checkout.

## Local Simulator Validation

This check does not require ROS or a model:

```bash
conda activate fluxthemis
cd /root/projects/FluxThemis
python scripts/libero_smoke.py \
  --suite libero_10 \
  --task-id 0 \
  --resolution 64
```

It validates dependency versions, asset/BDDL/initial-state paths, dual-camera images, proprio
state, reset, stabilization actions, and a single environment step. Production evaluation
configurations still use 256×256.

## Two-Terminal Launch

The following example reflects the current deployment: FluxVLA and FluxThemis both use Python
3.10 and communicate through a ROS 1 Noetic overlay built from the same source. ROS 2 Jazzy is
an isolated migration path; its cp312 Python packages must not be sourced into either process.
See [ROS 2 Deployment](ros2.md) for details.

The PI0.5 LIBERO-10 example is already defined in the single configuration source of truth:

```text
/root/projects/FluxVLA/configs/pi05/pi05_paligemma_libero_10_full_finetune.py
```

First, run one command on the FluxThemis side. The launcher automatically activates the
environment, starts or reuses `roscore`, waits for the service, and runs the rollout:

```bash
cd /root/projects/FluxThemis
bash scripts/run_ros1_eval.sh
```

The installation script configures a ROS 1 activation hook for the `fluxvla` environment. After
installation, open a new terminal or run `conda deactivate && conda activate fluxvla` once, then
start the model server in another terminal:

```bash
conda activate fluxvla
cd /root/projects/FluxVLA
export CUDA_VISIBLE_DEVICES=0
export HF_ENDPOINT=https://hf-mirror.com
export WANDB_MODE=disabled
bash scripts/ros_inference_server.sh \
  --config configs/pi05/pi05_paligemma_libero_10_full_finetune.py \
  --ckpt-path /abs/path/to/checkpoint \
  --ros-version 1
```

After a successful run, the launcher retains any master it started by default. This lets you run
a smoke test first and then reuse the same loaded model for the full evaluation. Add
`--shutdown-master` for a one-shot job. Failures and interrupts trigger automatic cleanup.

The two commands above are unchanged. The default configuration advertises both
`/fluxvla/predict_action` and `/fluxvla/report_evaluation`; the launcher waits for both before
rollout. FluxThemis reports the run and episode lifecycle, while FluxVLA acknowledges each event,
writes the result files, and displays progress in the model-server terminal, for example:

```text
[ros-eval] episode_start ...
[eval-progress] episodes=15/500 successes=... success_rate=...%
```

FluxVLA always owns the target configuration, checkpoint, and authoritative results. With the
default `result_output_dir='work_dirs/fluxthemis'`, one reported run uses the native layout:

```text
/root/projects/FluxVLA/work_dirs/fluxthemis/eval_runs/<checkpoint_stem>/
  EVAL-libero_10-<model_family>-YYYY_MM_DD-HH_MM_SS[-<run_name>]/
```

The directory contains the event journal and rank progress during rollout, followed by native
per-task JSON/status files and `summary.txt`, `summary.csv`, `summary.json`, and
`task_success_rates.csv` at `run_end`. The acknowledgement returned to FluxThemis is the source
of truth for this path. Run a single-task, single-episode smoke test directly with:

```bash
bash scripts/run_ros1_eval.sh --smoke
```

Set `FEISHU_SHEET_URL`, `FEISHU_APP_ID`, and `FEISHU_APP_SECRET` in the FluxVLA terminal to enable
the optional Sheets upload. Set `themis.ros_server.evaluation_reporting.feishu.timeout` to change
the 10-second request timeout. A row is uploaded only for a finished, unfiltered full suite with
every authoritative task and the configured number of trials. The smoke command above,
filtered/partial runs, and failed or interrupted runs are intentionally skipped; the FluxVLA
terminal reports the reason.

If `themis.transport.report_service_name` is omitted, the PredictAction-only compatibility path
remains available. In that local-client mode only, FluxThemis writes its former
`episodes.jsonl`, atomic `progress.json`, and `summary.json` files under
`runner.work_dir/run_name` and applies the old run-name collision rules.

Both processes must read the same configuration. CLI overrides apply only to the environment
process, so do not use them to modify transport, dataset, or model-server parameters.

## Parallel Simulator Rollouts

FluxThemis can run several persistent LIBERO environments concurrently. Each rollout worker is
created with the multiprocessing `spawn` method, builds its own environment and inference-only
ROS client, and remains pinned to one render GPU. The parent process is the only owner of metric
aggregation and the acknowledged `ReportEvaluation` lifecycle.

For example, use FluxVLA inference GPUs 0 and 1 in the model-server terminal, and reserve GPUs 2
and 3 for two simulator workers:

```bash
# FluxVLA terminal
cd /root/projects/FluxVLA
export CUDA_VISIBLE_DEVICES=0,1
export HF_ENDPOINT=https://hf-mirror.com
export WANDB_MODE=disabled
bash scripts/ros_inference_server.sh \
  --config configs/pi05/pi05_paligemma_libero_10_full_finetune.py \
  --ckpt-path checkpoints/pi05_paligemma_libero_10_full_finetune_bs64/checkpoints/step-028548-epoch-18-loss=0.0111.safetensors \
  --ros-version 1 \
  --num-workers 2

# FluxThemis terminal
cd /root/projects/FluxThemis
bash scripts/run_ros1_eval.sh \
  --workers 2 \
  --sim-gpus 2,3 \
  --run-name pi05_parallel
```

The equivalent environment-driven form is:

```bash
export FLUXTHEMIS_PARALLEL_WORKERS=2
export FLUXTHEMIS_SIM_GPU_IDS=2,3
bash scripts/run_ros1_eval.sh --run-name pi05_parallel
```

`--sim-gpus` contains physical host GPU ids. For each worker, the launcher runtime sets both
`CUDA_VISIBLE_DEVICES=<id>` and `MUJOCO_EGL_DEVICE_ID=<id>` before LIBERO or MuJoCo is loaded.
If there are fewer ids than workers, ids are assigned round-robin. Separate inference and render
GPU pools are recommended to avoid model latency and EGL rendering competing for memory and
compute.

Every scheduled episode receives the globally unique id `<report_run_id>:<ordinal>`. A worker
does not issue its first `PredictAction` request until FluxVLA acknowledges `episode_start`, and
does not take another job until FluxVLA acknowledges `episode_end`. This preserves episode
affinity for policies with action-history or KV-cache state. Use no more active simulator workers
than FluxVLA inference replicas. FluxVLA rejects a mismatched `run_start` before any rollout, so
`--workers 2` must be paired with at least `--num-workers 2` (or two explicit `--devices`).

`--num-workers N` creates logical devices `cuda:0` through `cuda:N-1` after
`CUDA_VISIBLE_DEVICES` has been applied. `--devices` also takes logical visible indices. For
example, physical `CUDA_VISIBLE_DEVICES=0,2` is addressed as `--devices cuda:0,cuda:1`. Each
process loads a complete model and checkpoint; this is replica parallelism for episode throughput,
not tensor parallelism for one action prediction. All logs, native result files, and optional
Feishu updates still come from the single FluxVLA parent process.

The corresponding runner fields are `parallel_workers`, `simulator_gpu_ids`,
`parallel_startup_timeout_s`, `parallel_event_timeout_s`, and `parallel_ack_timeout_s`. The
default `parallel_workers=1` retains the original in-process serial path. Parallel execution is
accepted only for evaluators that declare `supports_parallel_rollouts=True`. The built-in
`SuccessRateEvaluator` is supported because it aggregates complete `EpisodeResult` objects;
custom evaluators that require parent-side `on_episode_start` or `on_step` hooks remain serial
until they implement an explicit parallel merge contract.

The activation hooks for both environments automatically source ROS 1, but do not install or
upgrade any Python packages in `fluxvla`. Before a production evaluation, run:

```bash
bash scripts/check_env.sh --ros 1 --smoke-level libero \
  --config /root/projects/FluxVLA/configs/pi05/pi05_paligemma_libero_10_full_finetune.py
```

## Suites and Initial States

The adapter only discovers tasks in `setup()`; it does not create MuJoCo. On the first reset for
a task, it creates an `OffScreenRenderEnv`. Episodes for the same task reuse that environment,
and switching tasks closes the old environment. Each episode selects:

```text
fixed_init_states[episode_index % len(fixed_init_states)]
```

It then resets the environment, sets the fixed initial state, and executes 10 stabilization
actions. Stabilization actions do not count toward the runner's episode steps. LIBERO's `done`
is `_check_success()`, so the adapter maps it to `success=True, terminated=True`.
