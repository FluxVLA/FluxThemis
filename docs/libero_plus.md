# LIBERO-Plus Simulator Integration

LIBERO-Plus uses the same FluxVLA/FluxThemis evaluation protocol as standard
LIBERO, but runs in its own Conda environment. FluxThemis owns only the
simulator, task selection, episode lifecycle, and per-run metrics. FluxVLA
still owns preprocessing, model inference, action denormalization,
authoritative result files, four-suite aggregation, and optional Feishu
reporting.

The integration uses
`hqr-robotic/LIBERO-plus@d7c160eca38bf268d5965aef6d4c6dca12d5537c`
and its isolated `libero_plus` Python namespace. Standard LIBERO keeps its
existing package, configuration, and asset paths.

## Environment layout

| Process | Conda environment | Contents |
| --- | --- | --- |
| FluxVLA server | `fluxvla` | Model, checkpoint, transforms, statistics, ROS services, results |
| LIBERO evaluator | `fluxthemis` | Standard LIBERO and its pinned simulator dependencies |
| LIBERO-Plus evaluator | `fluxthemis-libero-plus` | LIBERO-Plus, patched robosuite, MuJoCo, assets, and the ROS client |

Do not install standard LIBERO and LIBERO-Plus into the same FluxThemis
simulator environment. The FluxVLA server does not import the FluxThemis
simulator stack for ROS inference. FluxVLA provides the pinned classification
data, authoritative aggregation, and Feishu reporting.

## Installation

Create the dedicated Python 3.10 environment once, then run the unified
installer:

```bash
conda create -n fluxthemis-libero-plus python=3.10 -y

cd /root/projects/FluxThemis
bash scripts/install.sh \
  --simulator libero-plus \
  --env-name fluxthemis-libero-plus \
  --ros auto \
  --fluxvla-env-name fluxvla \
  --fluxvla-root /root/projects/FluxVLA
```

The installer pins LIBERO-Plus to
`d7c160eca38bf268d5965aef6d4c6dca12d5537c`, robosuite to
`e293cc32ff3c48957a4ebcad09952432b0dc9049`, MuJoCo to 3.2.6, and NumPy to
1.26.4. The default `auto` asset mode validates and links, in order, an
explicit asset root, the exact FluxVLA LIBERO-Plus checkout, or an earlier
managed download. It downloads the official archive only when no valid local
tree is available.

Use `--libero-plus-asset-root /absolute/path` to select another complete asset
tree, or `--libero-plus-asset-mode download` to force a managed download.

The downloader pins `Sylvest/LIBERO-plus` at revision
`dd2bd61b7d9a6fef1abc52d606e983b41886a149`, checks the
6,395,849,578-byte archive against SHA-256
`96764a4bfbdaea98d4411598caeab235458318fe0f549611b93d1a323027b3cf`,
rejects unsafe ZIP paths and links, validates the extracted tree, and
publishes it only after extraction succeeds.

The installer adds a headless EGL activation hook to
`fluxthemis-libero-plus` and the ROS 1 activation hook to both
`fluxthemis-libero-plus` and `fluxvla`. Activation sets
`LIBERO_PLUS_CONFIG_PATH` and the upstream-compatible `LIBERO_CONFIG_PATH` to
`$CONDA_PREFIX/etc/fluxthemis/libero-plus`. Reactivate both environments after
installation.

Validate imports, or execute a real reset, 10 warm-up actions, and one step
without ROS or a model:

```bash
conda activate fluxthemis-libero-plus
cd /root/projects/FluxThemis
python scripts/libero_plus_smoke.py --imports-only
python scripts/libero_plus_smoke.py \
  --suite libero_10 \
  --task-id 0 \
  --resolution 64 \
  --steps 1
```

## Supported suites and FluxVLA configuration

Run each suite separately with its matching FluxVLA model configuration and
checkpoint. The official task counts and maximum episode lengths are:

| Suite | Tasks | Trials per task | Maximum steps |
| --- | ---: | ---: | ---: |
| `libero_spatial` | 2,402 | 1 | 220 |
| `libero_object` | 2,518 | 1 | 280 |
| `libero_goal` | 2,591 | 1 | 300 |
| `libero_10` | 2,519 | 1 | 520 |

LIBERO-Plus is evaluated zero-shot. FluxVLA loads the unmodified standard
LIBERO model configuration, preprocessing, normalization statistics, and
checkpoint for the corresponding suite. No LIBERO-Plus-specific training,
checkpoint, or model-side configuration is used.

FluxThemis reads the same standard config, then applies only these
environment-side runner overrides at launch:

```text
themis.runner.environment.type=LiberoPlusEnvironment
themis.runner.episodes_per_task=1
```

The matching standard config remains the source of truth for
`task_suite_name`, `task_order_index`, `max_episode_steps`, and
`execute_horizon`. The action horizon is model-specific, not a suite constant.
Do not set `episodes_per_task` above one: the benchmark contains one fixed
trial for each of its 10,030 classified tasks.

The environment sends the original LIBERO observation mapping and task prompt
to FluxVLA and accepts only denormalized 7D actions. FluxVLA retains all
model-specific image, proprioception, prompt, and action processing. MuJoCo is
created lazily on the first reset. Environment creation makes at most 20
attempts when robosuite raises `RandomizationError`; every other error is
raised immediately.

## Metadata filtering and grouped metrics

Each task includes `benchmark='libero'`,
`benchmark_variant='libero_plus'`, `task_suite_name`, zero-based `task_index`,
one-based `classification_id`, `category`, and `difficulty_level`. Add a
diagnostic filter under `themis.runner` and group success rates by metadata:

```python
task_metadata_filters=dict(
    category=['Camera Viewpoints'],
    difficulty_level=[3],
),
evaluator=dict(
    type='SuccessRateEvaluator',
    group_by_metadata=['category', 'difficulty_level'],
),
```

Values within one field use OR; different fields use AND. If `task_ids` is
also configured, the runner evaluates the intersection. Tasks with
`difficulty_level=None` remain in an unfiltered run, are excluded by a numeric
difficulty filter, and can be selected explicitly with
`difficulty_level=[None]`.

Any `task_ids` or metadata filter marks the run as `full_suite=false` so a
partial diagnostic cannot be mistaken for an official submission. Unknown
metadata fields and filters that select no tasks fail before rollout.

## Single-GPU evaluation

Start FluxThemis first. It starts or reuses `roscore`, activates the dedicated
environment, and waits for both FluxVLA services. This example evaluates the
standard PI0.5 LIBERO-10 checkpoint zero-shot on the `libero_10` robustness
suite. The runner overrides affect only FluxThemis; FluxVLA loads the standard
config and checkpoint unchanged.

### Terminal A: FluxThemis

```bash
cd /root/projects/FluxThemis
bash scripts/run_ros1_eval.sh \
  --env-name fluxthemis-libero-plus \
  --config /root/projects/FluxVLA/configs/pi05/pi05_paligemma_libero_10_full_finetune.py \
  --run-name libero_plus_pi05_smoke \
  --cfg-options \
  'themis.runner.environment.type=LiberoPlusEnvironment' \
  'themis.runner.task_ids=["0"]' \
  'themis.runner.episodes_per_task=1'
```

### Terminal B: FluxVLA

```bash
conda activate fluxvla
cd /root/projects/FluxVLA

export CUDA_VISIBLE_DEVICES=0
export HF_ENDPOINT=https://hf-mirror.com
export WANDB_MODE=disabled

bash scripts/ros_inference_server.sh \
  --config /root/projects/FluxVLA/configs/pi05/pi05_paligemma_libero_10_full_finetune.py \
  --ckpt-path /root/projects/FluxVLA/checkpoints/pi05_paligemma_libero_10_full_finetune_bs64/checkpoints/step-038064-epoch-24-loss=0.0170.safetensors \
  --ros-version 1 \
  --device cuda:0
```

After the smoke run, keep the FluxVLA server loaded and launch the unfiltered
suite from Terminal A:

```bash
bash scripts/run_ros1_eval.sh \
  --env-name fluxthemis-libero-plus \
  --config /root/projects/FluxVLA/configs/pi05/pi05_paligemma_libero_10_full_finetune.py \
  --run-name libero_plus_pi05_full \
  --cfg-options \
  'themis.runner.environment.type=LiberoPlusEnvironment' \
  'themis.runner.episodes_per_task=1'
```

Repeat with the matching config and checkpoint for `libero_spatial`,
`libero_object`, and `libero_goal`.

## Multi-GPU evaluation

Multi-GPU mode uses full-model replicas and independent persistent simulator
workers. The number of simulator workers must not exceed the number of
FluxVLA inference workers.

```bash
# Terminal A: two LIBERO-Plus workers on physical GPUs 2 and 3.
cd /root/projects/FluxThemis
bash scripts/run_ros1_eval.sh \
  --env-name fluxthemis-libero-plus \
  --config /root/projects/FluxVLA/configs/pi05/pi05_paligemma_libero_10_full_finetune.py \
  --workers 2 \
  --sim-gpus 2,3 \
  --run-name libero_plus_pi05_2workers \
  --cfg-options \
  'themis.runner.environment.type=LiberoPlusEnvironment' \
  'themis.runner.episodes_per_task=1'

# Terminal B: two model replicas on physical GPUs 0 and 1.
conda activate fluxvla
cd /root/projects/FluxVLA
export CUDA_VISIBLE_DEVICES=0,1
export HF_ENDPOINT=https://hf-mirror.com
export WANDB_MODE=disabled
bash scripts/ros_inference_server.sh \
  --config /root/projects/FluxVLA/configs/pi05/pi05_paligemma_libero_10_full_finetune.py \
  --ckpt-path /root/projects/FluxVLA/checkpoints/pi05_paligemma_libero_10_full_finetune_bs64/checkpoints/step-038064-epoch-24-loss=0.0170.safetensors \
  --ros-version 1 \
  --num-workers 2
```

This is episode parallelism, not tensor parallelism. Each inference worker
loads a complete model and checkpoint. Each simulator worker owns one MuJoCo
environment at a time, while the parent runner remains the sole owner of
metric aggregation and evaluation reporting.

## Logs, results, and Feishu

Progress is displayed in the FluxVLA terminal. FluxVLA writes each suite's
authoritative artifacts under its configured result root. The acknowledged
result path returned to FluxThemis is the source of truth; a typical run is
under:

```text
/root/projects/FluxVLA/work_dirs/fluxthemis/eval_runs/<checkpoint_stem>/
  EVAL-<suite>-<model_family>-YYYY_MM_DD-HH_MM_SS[-<run_name>]/
```

After all four unfiltered suites finish, use the FluxVLA summarizer to produce
the strict 10,030-task summary:

```bash
cd /root/projects/FluxVLA
python tools/summarize_libero_plus_results.py \
  --run-dir /path/to/libero_spatial/results \
  --run-dir /path/to/libero_object/results \
  --run-dir /path/to/libero_goal/results \
  --run-dir /path/to/libero_10/results \
  --output-dir work_dirs/libero_plus_summary \
  --title MODEL_OR_CHECKPOINT_NAME
```

The tool rejects missing tasks, duplicate results, unknown IDs, and tasks with
anything other than one trial. It writes the normal LIBERO summaries plus
`libero_plus_summary.{csv,json,txt}`, a per-suite CSV, and a missing-task CSV.
FluxThemis does not duplicate this cross-suite authority.

Set `FEISHU_SHEET_URL`, `FEISHU_APP_ID`, and `FEISHU_APP_SECRET`, or pass the
matching command-line arguments, only for a complete strict summary. Smoke,
filtered, failed, interrupted, incomplete, or duplicate-task results are not
eligible for the official Feishu report.

## ROS 2 option

The installer also accepts an existing Python 3.12 environment and selects ROS
2 Jazzy with `--ros auto`. Keep it as another dedicated environment and change
both sides of the FluxVLA configuration to ROS 2
(`FluxVLAROS2ModelClient` and `ros_version=2`). Do not source ROS 1 and ROS 2 in
the same process.
