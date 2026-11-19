# LIBERO-Plus Evaluation

LIBERO-Plus evaluates robustness using standard LIBERO model configurations,
checkpoints, preprocessing and normalization statistics. FluxThemis owns the
simulator, task selection and evaluation lifecycle; FluxVLA serves the model
over ZMQ and writes the authoritative evaluation results.

## Install

LIBERO and LIBERO-Plus share the `fluxthemis-libero` simulator environment.
If LIBERO is already installed, add Plus with:

```bash
cd /root/projects/FluxThemis
bash scripts/install_libero_plus.sh
conda activate fluxthemis-libero
```

For a fresh environment, run these commands before the Plus installer:

```bash
conda create -n fluxthemis-libero python=3.10 -y
bash scripts/install_libero.sh
```

Use `--env-name NAME` on both installers for a custom environment name.
The Plus installer requires an existing LIBERO runtime and installs only
`requirements-libero-plus.txt`, the Plus source, assets and configuration.
It leaves the installed LIBERO, robosuite and Torch packages in place.

New installations use `hqr-robotic/LIBERO-plus` at
`d7c160eca38bf268d5965aef6d4c6dca12d5537c`, the isolated `libero_plus` Python
namespace, and the FluxVLA robosuite fork at
`e293cc32ff3c48957a4ebcad09952432b0dc9049`. Existing clean source checkouts
are reused and validated through their runtime APIs, task classification and
assets. Standard LIBERO uses `yinchimaoliang/LIBERO` at
`058fda1ddebe92918af091cb6816759ca6d003f0` and the `libero` namespace.
The base installer uses `requirements-libero.txt`; the Plus requirements file
contains only the additional dependencies. Both variants share one robosuite
installation.
Model inference dependencies belong in the FluxVLA environment.

The installer stores simulator paths under
`<conda-prefix>/etc/fluxthemis/libero-plus`. The environment adapter selects
this configuration and sets EGL defaults when loading the simulator. Standard
LIBERO uses the sibling `libero` configuration directory. To override paths,
use `LIBERO_CONFIG_PATH` for standard LIBERO and `LIBERO_PLUS_CONFIG_PATH` for
LIBERO-Plus. Run each benchmark in a separate process so their simulator
registries and configuration state remain independent.

By default, installation validates and reuses existing assets or downloads
the official archive. To use a complete local asset directory:

```bash
bash scripts/install_libero_plus.sh \
  --asset-mode link --asset-root /path/to/libero-plus/assets
```

`--asset-mode download` forces the managed download. The downloader checks the
archive size and SHA-256, rejects unsafe archive entries, and validates the
extracted tree before publishing it. `--dry-run` previews installation;
`--smoke-level imports` checks imports and assets without creating a simulator.
The Plus installer's default smoke check runs both benchmarks in
separate processes. Each check resets a task, performs ten warm-up actions
and one action step, and validates both cameras and proprioception.

To repeat the Plus check in the simulator environment:

```bash
python scripts/libero_plus_smoke.py
```

Add `--imports-only` to check imports and assets without creating a simulator.
Use `python scripts/libero_smoke.py` to check standard LIBERO separately.

## Zero-shot Protocol

| Suite | Classified tasks | Rollouts per task | Maximum action steps |
| --- | ---: | ---: | ---: |
| `libero_spatial` | 2,402 | 1 | 220 |
| `libero_object` | 2,518 | 1 | 280 |
| `libero_goal` | 2,591 | 1 | 300 |
| `libero_10` | 2,519 | 1 | 520 |

Evaluate each suite separately with its matching standard LIBERO configuration
and checkpoint. Together the four suites contain 10,030 tasks. Task IDs are
zero-based; `classification_id` in task metadata is the official one-based ID.
The adapter also supplies `category`, `difficulty_level` and
`benchmark_variant='libero_plus'` metadata.

Use the checked-in simulator configuration
[configs/libero-plus/libero_10.py](../configs/libero-plus/libero_10.py)
and launch settings
[configs/libero-plus/remote.yaml](../configs/libero-plus/remote.yaml).
The Python config evaluates all 2,519 LIBERO-10 Plus tasks once, with a
520-step limit and PI0.5's ten-step execution horizon. It contains only the
simulator and observation/action interface; model weights, preprocessing and
normalization statistics remain in FluxVLA. When adapting it to another suite,
change `task_suite_name`, `unnorm_key` and `max_episode_steps` together and use
the matching model configuration and checkpoint. Set `execute_horizon` to
match the model's evaluation protocol.

The YAML holds GPU/worker counts, the model endpoint and output paths. Its
`model` section connects to an already-running service. Simulator assets use
the installed LIBERO-Plus configuration described above.

In the FluxVLA model environment, start the model with its standard LIBERO
configuration and checkpoint:

```bash
conda activate fluxvla
cd /root/projects/FluxVLA
CUDA_VISIBLE_DEVICES=0 NPROC_PER_NODE=1 \
bash scripts/distributed_zmq_server.sh \
  --config configs/pi05/pi05_paligemma_libero_10_full_finetune.py \
  --ckpt-path /path/to/step-038064-epoch-24-loss=0.0170.safetensors \
  --cfg-options 'themis.server.denormalize_per_action=True'
```

For the standard PI0.5 config's `DenormalizeLiberoAction`, public FluxVLA
revision `cd9206a` needs `denormalize_per_action=True`. Its action transform
expects one action vector; applying it to the entire predicted chunk also
truncates the time axis to seven steps. This option applies the unchanged
transform to each action and preserves the configured ten-step horizon.

This native launcher works with the public FluxVLA ZMQ server. The Themis
`distributed_model.sh` wrapper additionally requires a FluxVLA build whose
`distributed_zmq_server.py` accepts `--workers-per-gpu`; public revision
`cd9206a` does not accept that option. For one model per GPU, use the native
launcher above and set `NPROC_PER_NODE` to the number of visible model GPUs.

In the shared simulator environment, start the simulator independently:

```bash
conda activate fluxthemis-libero
cd /root/projects/FluxThemis
bash scripts/run_evaluation.sh \
  --config configs/libero-plus/remote.yaml --full
```

By default, this connects to `tcp://127.0.0.1:15555`, uses GPU 0 with one
simulator worker, and writes launcher output under `work_dirs/libero-plus`.
For a remote model, edit `model.host` in the YAML or pass
`--server-endpoint tcp://MODEL_HOST:15555`. Use `--output /persistent/evaluations`
to override the output location.
For multiple simulator GPUs, use `--gpu-ids 0,1,2,3 --workers-per-gpu 2`.
With dynamic scheduling, model and simulator worker counts are independent.
`--dry-run` validates the launch configuration without models, GPUs or simulator
dependencies. `--smoke` selects task 0 with one episode per simulator worker;
smoke results are diagnostic and do not follow the full benchmark quota.
Without `--smoke` or `--full`, this YAML runs task 0 once. `--full` uses the
Python config's one rollout per task regardless of the simulator worker count.

For a one-task diagnostic through the documented direct evaluator:

```bash
bash scripts/distributed_eval.sh \
  --config configs/libero-plus/libero_10.py \
  --server-endpoint tcp://MODEL_HOST:15555 \
  --gpu-ids 0 --workers-per-gpu 1 --scheduling dynamic \
  --run-name libero_plus_smoke \
  --cfg-options 'themis.evaluation.task_ids=["0"]'
```

This uses the same rollout and model-reporting code. The unified
`run_evaluation.sh` launcher additionally scans all declared assets to record
recovery fingerprints; its first preflight can take several minutes on shared
storage with many small files. The direct evaluator does not generate those
launcher recovery fingerprints. Add `--save-videos --video-dir /path/to/videos`
to record the diagnostic episode.

## Filtering and Metrics

For a diagnostic subset, copy `configs/libero-plus/libero_10.py` and append
the following overrides. Pass that Python file directly to
`run_evaluation.sh --config /path/to/subset.py` without `--smoke` or `--full`
so that the task IDs in the Python file determine the selection:

```python
themis['evaluation'].update(
    task_ids=['0', '1', '2'],  # Optional intersection.
    task_metadata_filters=dict(
        category=['Camera Viewpoints', 'Light Conditions'],
        difficulty_level=[1, 2, 3],
    ),
)
```

Values within a field use OR; different fields use AND. Unknown fields and
empty selections fail before any episode is dispatched. `None` matches an
explicit null difficulty, not missing metadata. Use Python configuration for
category names containing spaces; the CLI parser removes spaces in values.
Clear diagnostic filters before a full benchmark run. `--full` selects all
task IDs but retains configured metadata filters.

Metrics retain `overall` and `tasks`; enabling `group_by_metadata` additionally
produces `groups.category` and `groups.difficulty_level`. Null difficulty is
reported under `null`. Filtered runs report `full_suite=False`. Task selection
is performed once by the coordinator before jobs and seeds are assigned, so
worker completion order does not change the selected inventory or metrics.

The common [evaluation launcher](../guides/evaluation.md) provides progress,
resource monitoring and resume. Recovery records the selected task metadata
and filters and checks them before reusing completed episodes. Preflight
includes the LIBERO-Plus assets, initial states, task definitions and official
classification in recovery input identity checks.

The archived [original PR #3](https://github.com/FluxVLA/FluxThemis/pull/3)
reported 1,986 successes over 2,519 PI0.5 LIBERO-10 tasks (78.84%) using the
older ROS implementation. That is a historical result; it is not a validation
of the current ZMQ implementation. Repeat simulator and model smoke checks
before using the migrated implementation for a new full benchmark.
