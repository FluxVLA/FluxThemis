# RoboCasa Simulator Integration

RoboCasa uses the same FluxVLA/FluxThemis evaluation protocol as LIBERO, but
runs in its own Conda environment. FluxThemis owns only the simulator and
episode lifecycle. FluxVLA still owns preprocessing, normalization, model
inference, action denormalization, logs, result files, and optional Feishu
reporting.

## Environment layout

| Process | Conda environment | Contents |
| --- | --- | --- |
| FluxVLA server | `fluxvla` | Model, checkpoint, transforms, statistics, ROS services, results |
| LIBERO evaluator | `fluxthemis` | LIBERO and its pinned simulator dependencies |
| RoboCasa evaluator | `fluxthemis-robocasa` | RoboCasa, patched robosuite, MuJoCo, and the ROS client |

Do not install LIBERO and RoboCasa into the same simulator environment. The
FluxVLA server does not need the RoboCasa simulator packages for ROS evaluation.

## Installation

Create the dedicated Python 3.10 environment once, then run the unified
installer:

```bash
conda create -n fluxthemis-robocasa python=3.10 -y

cd /root/projects/FluxThemis
bash scripts/install.sh \
  --simulator robocasa \
  --env-name fluxthemis-robocasa \
  --ros auto \
  --fluxvla-env-name fluxvla \
  --fluxvla-root /root/projects/FluxVLA
```

The installer pins RoboCasa to
`4840e671596f93ca03651524b9f72ffb1aadfeff`, robosuite to
`e293cc32ff3c48957a4ebcad09952432b0dc9049`, MuJoCo to 3.2.6, and NumPy to
1.26.4. With `--robocasa-asset-mode auto`, it validates and links the complete
assets from the exact FluxVLA RoboCasa checkout. This reuses the existing
approximately 9.3 GB asset tree. Use `--robocasa-asset-root /absolute/path` to
select another complete asset tree, or `--robocasa-asset-mode download` when
no local tree is available.

The installer adds a headless EGL activation hook to `fluxthemis-robocasa` and
the ROS 1 activation hook to both `fluxthemis-robocasa` and `fluxvla`.
Reactivate the environments after installation.

## Supported FluxVLA configurations

The following configs contain a matching `themis` section and the same ordered
24-task benchmark manifest:

| Config | Trials per task | Executed action horizon | Flat action order |
| --- | ---: | ---: | --- |
| `configs/pi05/pi05_paligemma_robocasa_30_eps_full_finetune.py` | 20 | 16 | `fluxvla` |
| `configs/pi05/pi05_paligemma_robocasa_full_data_full_finetune.py` | 20 | 8 | `fluxvla` |
| `configs/gr00t/gr00t_eagle_3b_robocasa_30_eps_full_finetune.py` | 50 | 16 | `n15` |

The environment sends the raw
`video.ego_view_bg_crop_pad_res256_freq20` image, five raw state arrays, and
the exact per-episode `annotation.human.coarse_action` instruction. The
instruction is frozen for that episode, including its `unlocked_waist:`
prefix. FluxVLA performs all model-specific preprocessing and returns a
denormalized 29D action. FluxThemis only splits that vector into arms, hands,
and waist according to the config's action order.

## Single-GPU evaluation

Start FluxThemis first. It starts or reuses `roscore`, activates the dedicated
environment, and waits for both FluxVLA services.

### Terminal A: FluxThemis

```bash
cd /root/projects/FluxThemis
bash scripts/run_ros1_eval.sh \
  --env-name fluxthemis-robocasa \
  --config /root/projects/FluxVLA/configs/pi05/pi05_paligemma_robocasa_30_eps_full_finetune.py \
  --smoke \
  --run-name robocasa_pi05_smoke
```

### Terminal B: FluxVLA

```bash
conda activate fluxvla
cd /root/projects/FluxVLA

export CUDA_VISIBLE_DEVICES=0
export HF_ENDPOINT=https://hf-mirror.com
export WANDB_MODE=disabled

bash scripts/ros_inference_server.sh \
  --config configs/pi05/pi05_paligemma_robocasa_30_eps_full_finetune.py \
  --ckpt-path /absolute/path/to/robocasa-checkpoint.safetensors \
  --ros-version 1 \
  --device cuda:0
```

After the smoke run, keep the FluxVLA server loaded and launch the full suite
from Terminal A:

```bash
bash scripts/run_ros1_eval.sh \
  --env-name fluxthemis-robocasa \
  --config /root/projects/FluxVLA/configs/pi05/pi05_paligemma_robocasa_30_eps_full_finetune.py \
  --run-name robocasa_pi05_full
```

## Multi-GPU evaluation

Multi-GPU mode uses full-model replicas and independent persistent simulator
workers. The number of simulator workers must not exceed the number of
FluxVLA inference workers.

```bash
# Terminal A: two RoboCasa workers on physical GPUs 2 and 3.
cd /root/projects/FluxThemis
bash scripts/run_ros1_eval.sh \
  --env-name fluxthemis-robocasa \
  --config /root/projects/FluxVLA/configs/pi05/pi05_paligemma_robocasa_30_eps_full_finetune.py \
  --workers 2 \
  --sim-gpus 2,3 \
  --run-name robocasa_pi05_2workers

# Terminal B: two model replicas on physical GPUs 0 and 1.
conda activate fluxvla
cd /root/projects/FluxVLA
export CUDA_VISIBLE_DEVICES=0,1
export HF_ENDPOINT=https://hf-mirror.com
export WANDB_MODE=disabled
bash scripts/ros_inference_server.sh \
  --config configs/pi05/pi05_paligemma_robocasa_30_eps_full_finetune.py \
  --ckpt-path /absolute/path/to/robocasa-checkpoint.safetensors \
  --ros-version 1 \
  --num-workers 2
```

This is episode parallelism, not tensor parallelism. Each inference worker
loads a complete checkpoint. Each simulator worker owns one MuJoCo environment
at a time and creates a fresh seeded RoboCasa instance for every episode.

## Logs, results, and Feishu

Progress is displayed in the FluxVLA terminal. FluxVLA writes the authoritative
artifacts under:

```text
/root/projects/FluxVLA/work_dirs/fluxthemis/eval_runs/<checkpoint_stem>/
  EVAL-robocasa-<model_family>-YYYY_MM_DD-HH_MM_SS[-<run_name>]/
```

The layout matches native FluxVLA RoboCasa evaluation:

```text
robocasa/task<N>_results.json
task_status/robocasa_task<N>.status
summary.json
summary.csv
summary.txt
task_success_rates.csv
events.jsonl
rank0.txt
```

`summary.json` contains the native Cabinet, Drawer, Microwave, and
Generalization groups plus weighted overall statistics. A finished,
unfiltered, complete 24-task run is eligible for the existing RoboCasa Feishu
report. Smoke, filtered, failed, interrupted, or incomplete runs are saved
locally but are not uploaded.

## ROS 2 option

The installer also accepts an existing Python 3.12 environment and selects ROS
2 Jazzy with `--ros auto`. To use that profile, keep it as another dedicated
environment and change both sides of the selected FluxVLA config to ROS 2
(`FluxVLAROS2ModelClient` and `ros_version=2`). Do not source ROS 1 and ROS 2 in
the same process.
