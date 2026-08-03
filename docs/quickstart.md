# FluxVLA + FluxThemis LIBERO Quickstart

This guide covers a clean installation and the exact two-terminal commands for
evaluating the PI0.5 LIBERO-10 checkpoint. The commands use the deployment that
has been tested end to end:

| Side | Conda environment | Python | Runtime | Responsibility |
| --- | --- | ---: | --- | --- |
| FluxVLA | `fluxvla` | 3.10 | ROS 1 Noetic | Load the config/checkpoint, run inference, write results, and optionally update Feishu |
| FluxThemis | `fluxthemis` | 3.10 | ROS 1 Noetic | Run LIBERO, send observations, execute returned actions, and report episode events |

Both processes read the configuration from FluxVLA. FluxThemis does not load a
checkpoint and does not own the authoritative evaluation results.

## 1. Prerequisites

The examples assume these repository paths:

```text
/root/projects/FluxVLA
/root/projects/FluxThemis
```

The existing `fluxvla` environment must already be able to load the target
model. The FluxThemis installer adds a ROS 1 activation hook to that environment
but does not upgrade, downgrade, or reinstall its Python packages.

Check the environments before installation:

```bash
conda run -n fluxvla python --version
conda run -n fluxthemis python --version
```

Both commands should report Python 3.10 for the current ROS 1 deployment. If
`fluxthemis` does not exist yet, create it first:

```bash
conda create -n fluxthemis python=3.10 -y
```

The installer deliberately does not delete or recreate an existing Conda
environment.

## 2. Install LIBERO and ROS 1

Run the unified installer from FluxThemis:

```bash
cd /root/projects/FluxThemis
bash scripts/install.sh \
  --env-name fluxthemis \
  --ros auto \
  --fluxvla-env-name fluxvla \
  --fluxvla-root /root/projects/FluxVLA
```

For Python 3.10, `--ros auto` selects the source-built ROS 1 Noetic profile. It
also installs the pinned LIBERO/robosuite stack, MuJoCo dependencies, generated
ROS service types, and matching Conda activation hooks.

Open a new terminal after installation, or reactivate each environment before
using it. Then run the read-only acceptance check:

```bash
cd /root/projects/FluxThemis
bash scripts/check_env.sh \
  --env-name fluxthemis \
  --ros auto \
  --smoke-level libero \
  --config /root/projects/FluxVLA/configs/pi05/pi05_paligemma_libero_10_full_finetune.py
```

This check imports both ROS service types, validates the pinned packages, builds
the shared FluxVLA configuration, resets a real LIBERO environment, and takes one
simulation step. It does not load the checkpoint or start a ROS graph.

## 3. Confirm the Target Files

The commands below use:

```text
Config:
/root/projects/FluxVLA/configs/pi05/pi05_paligemma_libero_10_full_finetune.py

Checkpoint:
/root/projects/FluxVLA/checkpoints/pi05_paligemma_libero_10_full_finetune_bs64/checkpoints/step-028548-epoch-18-loss=0.0111.safetensors
```

Verify both before starting a long evaluation:

```bash
test -f /root/projects/FluxVLA/configs/pi05/pi05_paligemma_libero_10_full_finetune.py
test -f /root/projects/FluxVLA/checkpoints/pi05_paligemma_libero_10_full_finetune_bs64/checkpoints/step-028548-epoch-18-loss=0.0111.safetensors
```

## 4. Single-GPU Launch

Start the FluxThemis terminal first. It starts or reuses `roscore`, then waits
for FluxVLA while the checkpoint loads.

### Terminal A: FluxThemis

Run a one-task, one-episode smoke evaluation:

```bash
cd /root/projects/FluxThemis
bash scripts/run_ros1_eval.sh \
  --smoke \
  --run-name pi05_step028548_smoke
```

No manual Conda activation, `roscore`, or `fluxthemis-eval` command is needed on
this side. The launcher performs all of them.

### Terminal B: FluxVLA

```bash
conda activate fluxvla
cd /root/projects/FluxVLA

export CUDA_VISIBLE_DEVICES=0
export HF_ENDPOINT=https://hf-mirror.com
export WANDB_MODE=disabled

bash scripts/ros_inference_server.sh \
  --config configs/pi05/pi05_paligemma_libero_10_full_finetune.py \
  --ckpt-path \
  checkpoints/pi05_paligemma_libero_10_full_finetune_bs64/checkpoints/step-028548-epoch-18-loss=0.0111.safetensors \
  --ros-version 1 \
  --device cuda:0
```

Wait for this line:

```text
FluxVLA ROS inference ready on /fluxvla/predict_action
```

Terminal A then begins the rollout automatically. Evaluation progress, result
paths, and Feishu status are intentionally displayed in Terminal B.

After the smoke run succeeds, keep Terminal B running and start the complete
LIBERO-10 evaluation from Terminal A:

```bash
cd /root/projects/FluxThemis
bash scripts/run_ros1_eval.sh --run-name pi05_step028548_full
```

The configured full suite contains 10 tasks and 50 trials per task.

## 5. Multi-GPU Launch

Multi-GPU evaluation uses full-model inference replicas and independent
simulator processes. It is episode-parallel execution, not tensor parallelism.
The required capacity rule is:

```text
FluxThemis --workers <= FluxVLA --num-workers
```

With four GPUs, reserve GPUs 0 and 1 for inference and GPUs 2 and 3 for LIBERO
rendering. As in the single-GPU flow, start Terminal A first.

### Terminal A: two LIBERO workers

```bash
cd /root/projects/FluxThemis
bash scripts/run_ros1_eval.sh \
  --workers 2 \
  --sim-gpus 2,3 \
  --run-name pi05_step028548_full_2gpu
```

### Terminal B: two FluxVLA replicas

```bash
conda activate fluxvla
cd /root/projects/FluxVLA

export CUDA_VISIBLE_DEVICES=0,1
export HF_ENDPOINT=https://hf-mirror.com
export WANDB_MODE=disabled

bash scripts/ros_inference_server.sh \
  --config configs/pi05/pi05_paligemma_libero_10_full_finetune.py \
  --ckpt-path \
  checkpoints/pi05_paligemma_libero_10_full_finetune_bs64/checkpoints/step-028548-epoch-18-loss=0.0111.safetensors \
  --ros-version 1 \
  --num-workers 2
```

Do not pass `--device` together with a multi-replica launch. Alternatively, use
`--devices cuda:0,cuda:1` instead of `--num-workers 2`.

`CUDA_VISIBLE_DEVICES` selects physical inference GPUs, while FluxVLA's
`cuda:0`, `cuda:1`, and `--devices` values are logical indices inside that
visible set. In contrast, FluxThemis `--sim-gpus` values are physical host GPU
ids. For example:

```text
CUDA_VISIBLE_DEVICES=0,2  -> FluxVLA devices are cuda:0 and cuda:1
--sim-gpus 1,3           -> FluxThemis workers render on physical GPUs 1 and 3
```

On a two-GPU host, both sides can use physical GPUs 0 and 1:

```bash
# FluxThemis
bash scripts/run_ros1_eval.sh --workers 2 --sim-gpus 0,1 \
  --run-name pi05_step028548_full_shared_gpus

# FluxVLA
export CUDA_VISIBLE_DEVICES=0,1
bash scripts/ros_inference_server.sh \
  --config configs/pi05/pi05_paligemma_libero_10_full_finetune.py \
  --ckpt-path checkpoints/pi05_paligemma_libero_10_full_finetune_bs64/checkpoints/step-028548-epoch-18-loss=0.0111.safetensors \
  --ros-version 1 \
  --num-workers 2
```

Sharing GPUs is supported but can reduce throughput or exhaust GPU memory.
Separate inference and rendering pools are preferred when enough GPUs are
available. Each FluxVLA worker loads a complete copy of the checkpoint.

For a short test that schedules work on two simulator workers, replace the full
Terminal A command with:

```bash
cd /root/projects/FluxThemis
bash scripts/run_ros1_eval.sh \
  --workers 2 \
  --sim-gpus 2,3 \
  --run-name pi05_step028548_parallel_smoke \
  --cfg-options \
  'themis.runner.task_ids=[0,1]' \
  themis.runner.episodes_per_task=1 \
  themis.runner.max_episode_steps=20
```

This is a partial run, so it is saved locally by FluxVLA but intentionally not
uploaded to Feishu.

## 6. Logs, Results, and Feishu

FluxVLA is the authoritative reporter. With the current configuration, results
are written below:

```text
/root/projects/FluxVLA/work_dirs/fluxthemis/eval_runs/<checkpoint_stem>/
  EVAL-libero_10-pi0-YYYY_MM_DD-HH_MM_SS[-<run_name>]/
```

A completed run contains files such as:

```text
events.jsonl
rank0.txt
rank_progress/rank0.json
libero_10/task<N>_results.json
task_status/libero_10_task<N>.status
summary.json
summary.csv
summary.txt
task_success_rates.csv
```

FluxThemis prints the acknowledged FluxVLA result directory at the end, but
does not create a second authoritative copy in its own repository.

To enable Feishu Sheets reporting, export the credentials in Terminal B before
starting the FluxVLA server:

```bash
export FEISHU_SHEET_URL='<spreadsheet URL>'
export FEISHU_APP_ID='<app id>'
export FEISHU_APP_SECRET='<app secret>'
```

Only a successfully completed, unfiltered full suite with all configured tasks
and trials is uploaded. Smoke, filtered, partial, interrupted, and failed runs
remain in `work_dirs` and produce a clear `Feishu skipped` message.

## 7. Health Checks and Common Failures

From a third ROS 1 terminal, the two services should have these types:

```bash
conda activate fluxthemis
rosservice type /fluxvla/predict_action
rosservice type /fluxvla/report_evaluation
```

Expected output:

```text
fluxthemis_msgs/PredictAction
fluxthemis_msgs/ReportEvaluation
```

- If Terminal A says it is waiting for a service, watch Terminal B for model
  loading or checkpoint errors. The default service timeout is 900 seconds.
- If `ROS_VERSION` is missing or the service types cannot be imported, open a
  new terminal, reactivate the environment, or rerun `scripts/install.sh`.
- If the run is rejected at `run_start`, confirm that FluxThemis `--workers`
  does not exceed the number of FluxVLA replicas.
- robosuite warnings about a private macro file, optional robot models, Mink,
  or legacy Gym are non-fatal for the supported LIBERO Panda tasks.
- A missing LIBERO dataset-directory warning is non-fatal when the configured
  BDDL files, assets, and initial states pass `scripts/check_env.sh`.
- Keep each `--ckpt-path` filename on one shell token. Do not insert a newline
  in the middle of `step-028548-epoch-18-loss=0.0111.safetensors`.

The launcher keeps a `roscore` that it started after a successful run so the
loaded FluxVLA server can be reused. Add `--shutdown-master` to the FluxThemis
command for a one-shot run. Stop the FluxVLA server with `Ctrl-C` when all
evaluations are complete.

## 8. Optional Python 3.12 + ROS 2 Profile

ROS 2 support is retained as a separate migration profile; it is not loaded
into the current Python 3.10 environments:

```bash
conda create -n fluxthemis-ros2 python=3.12 -y

cd /root/projects/FluxThemis
bash scripts/install.sh \
  --env-name fluxthemis-ros2 \
  --ros auto

bash scripts/check_env.sh \
  --env-name fluxthemis-ros2 \
  --ros auto \
  --smoke-level none
```

An end-to-end ROS 2 evaluation additionally requires a Python 3.12-compatible
FluxVLA environment and a paired configuration using `ros_version=2` and
`FluxVLAROS2ModelClient`. Do not source ROS 1 and ROS 2 overlays in the same
process. See [Optional ROS 2 Deployment](ros2.md) for that migration path.
