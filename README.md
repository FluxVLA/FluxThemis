# FluxThemis

FluxThemis orchestrates real-robot and simulation evaluation for FluxVLA. The two
repositories connect through `fluxthemis_msgs/PredictAction` and, when remote result
reporting is enabled, `fluxthemis_msgs/ReportEvaluation`. They share the same MMEngine
configuration for the target FluxVLA model. The codebase supports two
alternative profiles: Python 3.10 + ROS 1 Noetic, or Python 3.12 + ROS 2 Jazzy. The
current installation uses the former; each installation and process selects exactly
one profile.

The responsibility boundary is a strict constraint:

| Owner | Sole responsibility |
| --- | --- |
| FluxVLA | model / inference model, configuration, checkpoint, dataset, preprocessing, statistics, action denormalization, ROS services, and authoritative result serialization under `work_dirs` |
| FluxThemis | environment, evaluation runner, evaluator, ROS client, task/episode lifecycle, and evaluation-event production |

FluxThemis does not load models or checkpoints, build model-side datasets, duplicate
preprocessing or statistics, or denormalize actions. This ensures that training,
offline evaluation, and real-robot evaluation always reuse the same model semantics
from FluxVLA. In integrated ROS runs, FluxThemis reports lifecycle events and waits
for FluxVLA to acknowledge each one; FluxVLA writes the native evaluation artifacts
beside the target model. The FluxThemis repository neither contains nor owns a
`work_dirs` directory.

This repository currently provides a generic evaluation lifecycle, ROS client, Mock
environment, generic Gymnasium adapter, dedicated LIBERO and RoboCasa simulation
adapters, and a real-robot safety base class. Each simulator keeps its conflicting
dependencies in a separate Conda environment; RoboTwin and specific real-robot
platforms should follow the same adapter boundary.

For a copy-paste installation and the exact single-GPU and multi-GPU commands for
both repositories, start with the
[FluxVLA + FluxThemis LIBERO quickstart](docs/quickstart.md).
For the dedicated RoboCasa environment and its single-GPU and multi-GPU launch
commands, see the [RoboCasa integration guide](docs/robocasa.md).

## Install FluxThemis

Choose the simulator profile first. LIBERO and RoboCasa have conflicting dependency
stacks and must be installed into separate Conda environments. Install either profile
or both, but never install both simulators into the same environment. The installer
uses an existing environment and never deletes or recreates it.

| Simulator | Recommended environment | Python | Default ROS profile |
| --- | --- | ---: | --- |
| LIBERO | `fluxthemis` | 3.10 | ROS 1 Noetic |
| RoboCasa | `fluxthemis-robocasa` | 3.10 | ROS 1 Noetic |

### Install LIBERO

Create the environment if it does not already exist, then run the unified installer
with the simulator selected explicitly:

```bash
conda create -n fluxthemis python=3.10 -y
cd /root/projects/FluxThemis
bash scripts/install.sh \
  --simulator libero \
  --env-name fluxthemis \
  --ros auto \
  --fluxvla-env-name fluxvla \
  --fluxvla-root /root/projects/FluxVLA
```

This installs the minimum environment-side dependencies, pins LIBERO and robosuite
to the revisions used by FluxVLA, configures `LIBERO_CONFIG_PATH` and headless MuJoCo
EGL, and builds the selected ROS runtime. It does not install FluxVLA model or
training dependencies.

See the [LIBERO quickstart](docs/quickstart.md) for the matching FluxVLA setup and
single-GPU or multi-GPU evaluation commands.

### Install RoboCasa

RoboCasa uses its own simulator environment while reusing the same FluxVLA model
server:

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

The installer pins the FluxVLA RoboCasa and patched robosuite commits and links the
validated FluxVLA assets by default instead of copying approximately 9.3 GB. Full
installation and launch commands are documented in
[docs/robocasa.md](docs/robocasa.md).

### ROS selection and advanced installation

`--ros auto` selects the ROS runtime from the target environment's Python minor
version:

| Python | ROS runtime | Support |
| ---: | --- | --- |
| 3.10 | Source-built ROS 1 Noetic | Current default |
| 3.12 | ROS 2 Jazzy + rosidl workspace | Optional migration profile |

Ubuntu 24.04 does not provide Noetic binaries, so the ROS 1 installer builds a pinned
minimal stack against Python 3.10. Jazzy uses the system Python 3.12 ABI. Both the
FluxVLA server and FluxThemis evaluator must use the same ROS version; ROS 1 and ROS 2
profiles cannot share one Conda environment or communicate directly.

To use ROS 2, create a separate Python 3.12 environment for the selected simulator
and run the same command with that environment name. The FluxVLA server must also be
migrated to Python 3.12 and ROS 2. `rclpy`, `cv_bridge`, and ROS 2 are supplied by the
Jazzy installation and must not be installed from pip.

Use `--simulator none` when only the ROS runtime needs to be installed or rebuilt.
`--skip-libero` remains as a compatibility alias for `--simulator none`, and
`--skip-fluxvla-hook` disables installation of the ROS 1 activation hook into the
FluxVLA environment. `--ros both` is an advanced convenience for hosts that already
have separate Python 3.10 and Python 3.12 environments; it does not combine both ROS
versions in one environment.

See the [ROS 1 deployment guide](docs/ros1.md) and
[ROS 2 deployment guide](docs/ros2.md) for build details and limitations.

## Add `themis` to the target FluxVLA configuration

Do not maintain a second configuration for FluxThemis. Add the following `themis`
section directly to the target FluxVLA configuration that will load the model. The
existing `model` / `inference_model`, the selected section's `dataset`, and
`denormalize_action` remain entirely owned by FluxVLA.

```python
themis = dict(
    # Executed only by the FluxThemis CLI to register out-of-repository environments or evaluators.
    custom_imports=dict(
        imports=['your_robot_package.fluxthemis_plugins'],
        allow_failed_imports=False,
    ),
    # The single transport configuration shared by the ROS client and FluxVLA ROS server.
    transport=dict(
        service_name='/fluxvla/predict_action',
        report_service_name='/fluxvla/report_evaluation',
        timeout_s=30.0,
        image_keys=[
            'camera_front',
            'camera_wrist',
        ],
        state_keys=[
            'robot0_eef_pos',
            'robot0_eef_quat',
            'robot0_gripper_qpos',
        ],
        unnorm_key='your_robot_stats_namespace',
        image_encoding='rgb8',
    ),
    # Built and executed only by the FluxThemis CLI.
    runner=dict(
        type='EvalRunner',
        environment=dict(
            type='YourEnvironment',
        ),
        model_client=dict(
            type='FluxVLAROSModelClient',
        ),
        evaluator=dict(
            type='SuccessRateEvaluator',
        ),
        seed=7,
        episodes_per_task=1,
        max_episode_steps=1000,
        execute_horizon=10,
        stop_on_success=True,
        parallel_workers=1,
        simulator_gpu_ids=None,
        work_dir='work_dirs/fluxthemis',
    ),
    # Consumed only by the FluxVLA ROS inference server.
    ros_server=dict(
        ros_version=1,
        dataset_section='eval',  # Only 'eval' or 'inference' is allowed
        evaluation_reporting=dict(
            result_output_dir='work_dirs/fluxthemis',
        ),
        device='cuda:0',
        workers=dict(
            startup_timeout_s=900.0,
            request_timeout_s=120.0,
            lease_timeout_s=900.0,
        ),
        mixed_precision_dtype='bf16',
        enable_mixed_precision=True,
        model_outputs_environment_actions=False,
        forward_seed=False,
        denormalize_context={},
    ),
)
```

Key conventions:

- The order of `image_keys` is the multi-view order in the ROS request; every value
  must be a `uint8 [H, W, 3]` image in the environment observation.
- `state_keys` lists, in order, the raw state fields required by the FluxVLA dataset;
  every value must be a one-dimensional, nonempty, finite floating-point array. A
  single-state pipeline is still represented as a one-element list.
- `unnorm_key` must identify the statistics namespace used by the FluxVLA checkpoint.
- Do not repeat `transport` inside `model_client`. The FluxThemis CLI injects the
  sibling `themis.transport` configuration; duplicate configuration is rejected.
- `timeout_s` is the discovery timeout for each configured service. For ROS 2,
  `request_timeout_s` may also be added as the deadline for each RPC. Action requests
  are not retried; reporting retries preserve the same event identity.
- `report_service_name` enables the acknowledged result channel. If it is omitted,
  `PredictAction` remains available by itself and the runner retains its local-client
  artifact behavior. When it is present, both endpoints require the generated
  `ReportEvaluation` type.
- The current Python 3.10 primary path pairs `ros_version=1` with
  `FluxVLAROSModelClient` and may additionally set `persistent=True`. ROS 2 requires
  switching both to `ros_version=2` and `FluxVLAROS2ModelClient`.
- `dataset_section` selects the `eval` or `inference` section in the FluxVLA
  configuration. The server obtains `dataset` and `denormalize_action` from that
  section.
- `model_outputs_environment_actions=False` means the model output must still be
  denormalized by the FluxVLA server. Set it to `True` only when the model itself
  explicitly outputs actions in environment units.
- The episode seed is always sent with the ROS request. `forward_seed=True`
  temporarily sets and restores the Python, NumPy, and Torch RNGs within a single
  request and passes the seed to `predict_action`. Enable it only if the target
  pipeline supports this determinism contract.
- `denormalize_context` is additional model-side context passed to the FluxVLA
  denormalization path and is empty by default.
- If a FluxVLA transform can process only individual `[A]` actions rather than an
  entire action chunk, explicitly set `denormalize_per_action=True` in `ros_server`.
- `ros_server.evaluation_reporting.result_output_dir` is the FluxVLA reporter's
  output root. It defaults to `work_dirs/fluxthemis`; relative and absolute values
  are accepted only when they resolve inside the target FluxVLA repository's
  `work_dirs` tree.
- `runner.work_dir`, `--work-dir`, `--run-name`, and `runner.overwrite` control the
  local artifact path used by clients without evaluation reporting. In a reported
  ROS run, `run_name` is only an optional suffix request; FluxVLA constructs and
  returns the actual authoritative directory.
- `runner.parallel_workers>1` uses spawned, persistent simulator processes. Each
  child owns one environment and an inference-only ROS client; only the parent
  aggregates metrics and reports lifecycle events. `simulator_gpu_ids` pins render
  workers by setting both `CUDA_VISIBLE_DEVICES` and `MUJOCO_EGL_DEVICE_ID` before
  the simulator is imported. The launcher exposes these as `--workers` and
  `--sim-gpus` (or `FLUXTHEMIS_PARALLEL_WORKERS` and
  `FLUXTHEMIS_SIM_GPU_IDS`).
- FluxVLA uses full-model inference replicas rather than splitting one action
  prediction across GPUs. Start replicas with `--num-workers N` or explicit
  logical devices with `--devices cuda:0,cuda:1`. The same episode remains on one
  replica until its acknowledged `episode_end`, protecting dataset/model history.
  FluxVLA rejects a run when `runner.parallel_workers` exceeds the available
  inference replicas.

### Integrated LIBERO-10 configuration

`configs/pi05/pi05_paligemma_libero_10_full_finetune.py` in FluxVLA already contains
a reusable `themis` section, selects the current Python 3.10/ROS 1 path, and enables
the acknowledged reporter at `/fluxvla/report_evaluation` by default.
The environment side sends the raw `agentview_image`,
`robot0_eye_in_hand_image`, EEF pose/quaternion, and gripper qpos. It does not rotate
images, concatenate state, or denormalize actions. These operations remain the
responsibility of the FluxVLA eval dataset and ROS server in that configuration.

`LiberoEnvironment` reproduces FluxVLA's key rollout semantics by default: 256x256,
`OSC_POSE`, simulator seed 0, pinned `.pruned_init` initial states, and 10
`[0, 0, 0, 0, 0, 0, -1]` stabilization actions. The policy step limits for each suite
are:

| suite | `max_episode_steps` |
| --- | ---: |
| `libero_spatial` | 220 |
| `libero_object` | 280 |
| `libero_goal` | 300 |
| `libero_10` | 520 |
| `libero_90` | 400 |

The limit is managed by `themis.runner.max_episode_steps`, not stored in the
simulator adapter. When switching suites, update the environment suite, `unnorm_key`,
and limit together in the target FluxVLA model configuration; do not create a second
configuration in FluxThemis. See the [LIBERO integration guide](docs/libero.md) for
complete details.

The FluxThemis CLI builds components only from `themis.runner` and executes only
`themis.custom_imports`. It does not use FluxVLA's `model`, `inference_model`,
dataset, or `themis.ros_server` to build any local model object. The FluxVLA server
reads `model` / `inference_model`, the selected section's `dataset` and
`denormalize_action`, plus `themis.transport` and `themis.ros_server`. It does not run
the FluxThemis environment, runner, or evaluator.

## Build the ROS message packages

`ros/fluxthemis_msgs` is a ROS 1 catkin package;
`ros2/fluxthemis_msgs` is a ROS 2 ament/rosidl package. Tests guarantee that the two
service definitions are byte-for-byte identical, but a process may source only the
selected version. Install and activate the Python 3.10 ROS 1 version as follows:

```bash
bash scripts/install_ros1_noetic_py310.sh \
  --env-name fluxthemis \
  --workspace /root/fluxthemis_ros1_ws
bash scripts/install_ros1_hook.sh \
  --env-name fluxvla \
  --workspace /root/fluxthemis_ros1_ws

conda activate fluxthemis
```

The full installation entry point automatically configures the same activation hook
for both `fluxthemis` and `fluxvla`. The two standalone installation commands above
are used only to repair the ROS deployment. The Python 3.10 FluxVLA environment does
not need `cv_bridge` installed or any model dependencies changed.

Build the ROS 2 Jazzy / Python 3.12 version as follows:

```bash
# Recommended: install Jazzy and build the interface workspace
bash scripts/install_ros2_jazzy.sh

# Or build it manually
source /opt/ros/jazzy/setup.bash
mkdir -p /abs/path/to/fluxthemis_ros2_ws/src
ln -s /abs/path/to/FluxThemis/ros2/fluxthemis_msgs \
  /abs/path/to/fluxthemis_ros2_ws/src/fluxthemis_msgs
cd /abs/path/to/fluxthemis_ros2_ws
colcon build --symlink-install --packages-select fluxthemis_msgs \
  --cmake-args -DPython3_EXECUTABLE=/usr/bin/python3 \
               -DPYTHON_EXECUTABLE=/usr/bin/python3
source /abs/path/to/fluxthemis_ros2_ws/install/setup.bash
ros2 interface show fluxthemis_msgs/srv/PredictAction
ros2 interface show fluxthemis_msgs/srv/ReportEvaluation
```

The commands above are written for bash; when using zsh, replace `setup.bash` with
`setup.zsh`.

Both the GPU server and environment process must source the same ROS 2 distribution
and this workspace's `install/setup.bash`. Then use the Python interpreter actually
used by the application to confirm that the generated type is visible:

```bash
python -c "import rclpy, cv_bridge; from fluxthemis_msgs.srv import PredictAction, ReportEvaluation; print(PredictAction, ReportEvaluation)"
```

## Run in two terminals

ROS 1 is the current default. The installer configures the same ROS 1 overlay for
both the `fluxthemis` and `fluxvla` environments. Open a new terminal or reactivate
the environment once after installation; no manual `source` command is needed.

If the two sides are on different hosts, the FluxVLA server and environment process
must set the same reachable `ROS_MASTER_URI` and correctly set each host's `ROS_IP` or
`ROS_HOSTNAME`.

Both processes must receive the same target FluxVLA configuration. Start the launcher
on the environment side first. It starts or reuses `roscore` and then waits for the
model service, so a slow checkpoint load does not cause a race:

```bash
cd /abs/path/to/FluxThemis
bash scripts/run_ros1_eval.sh \
  --config /abs/path/to/FluxVLA/configs/<family>/<target_config>.py
```

In the other terminal, start the FluxVLA ROS inference server on the GPU side:

```bash
cd /abs/path/to/FluxVLA
conda activate fluxvla
bash scripts/ros_inference_server.sh \
  --config /abs/path/to/FluxVLA/configs/<family>/<target_config>.py \
  --ckpt-path /abs/path/to/fluxvla_checkpoint.pt \
  --ros-version 1
```

The server builds `inference_model` when present, or otherwise `model`; loads the
checkpoint; builds the selected dataset preprocessing and denormalization path; and
then publishes `PredictAction` at `themis.transport.service_name`. When
`themis.transport.report_service_name` is configured, it also publishes the
acknowledged `ReportEvaluation` service. The launcher waits for and validates both
configured services before beginning rollout. The default statistics path matches
the existing FluxVLA runner:
`dataset_statistics.json` two directory levels above the checkpoint. It may also be
specified explicitly with `themis.ros_server.norm_stats_path`; an invalid path
terminates startup immediately.

The public ROS service remains a singleton. Internally, FluxVLA can own several
full-model worker processes, one per inference GPU. Dispatch is episode-affine, and
the parent process remains the sole owner of logs, native result artifacts, and
Feishu reporting.

The current PI0.5 LIBERO-10 configuration is the launcher's default, so a full
evaluation can run with `bash scripts/run_ros1_eval.sh`. For a single-task smoke test,
use `bash scripts/run_ros1_eval.sh --smoke`. After a successful run, the launcher
retains a master that it started, allowing the loaded FluxVLA
server to be reused. Add `--shutdown-master` for a one-shot run. The launcher still
cleans up a master it owns after a failure or Ctrl-C.

The two commands above are unchanged by result reporting. During a reported run,
FluxThemis sends ordered `run_start`, `episode_start`, `episode_end`, and `run_end`
events. Each event must be accepted by FluxVLA before the client advances. Transport
retries reuse the same request ID and sequence number, and every acknowledgement
returns the same authoritative `run_dir`. Episode progress, final metrics, artifact
paths, and any Feishu upload status therefore appear in the FluxVLA server terminal.
FluxThemis prints only the final acknowledged directory and does not create a local
`progress.json` for this remote mode.

For a two-GPU inference pool and two LIBERO render workers on separate GPUs, use:

```bash
# FluxVLA terminal: CUDA devices are logical after CUDA_VISIBLE_DEVICES.
export CUDA_VISIBLE_DEVICES=0,1
bash scripts/ros_inference_server.sh \
  --config configs/pi05/pi05_paligemma_libero_10_full_finetune.py \
  --ckpt-path /abs/path/to/checkpoint \
  --ros-version 1 \
  --num-workers 2

# FluxThemis terminal: simulator ids are physical host GPU ids.
bash scripts/run_ros1_eval.sh \
  --workers 2 \
  --sim-gpus 2,3 \
  --run-name pi05_parallel
```

Use disjoint inference and rendering GPU pools when possible. The number of active
simulator workers must not exceed the number of FluxVLA inference replicas; the
server rejects a mismatched run before rollout. Omitting both worker options retains
the original single-GPU, single-process path.

Clients without `report_service_name` keep the local behavior: FluxThemis prints
`episode_start` / `episode_done` and writes `episodes.jsonl`, atomic `progress.json`,
`summary.json`, and `failure.json` under `runner.work_dir/run_name`.

The Conda hook for `fluxthemis` automatically sources the ROS 1 overlay. The ROS 2
Jazzy migration path uses `fluxthemis-ros2`, `FluxVLAROS2ModelClient`, and
`--ros-version 2`; see the [ROS 2 deployment guide](docs/ros2.md) for the complete
commands.

For initial integration, first build the components without connecting to a service
or hardware:

```bash
fluxthemis-eval \
  --config /abs/path/to/FluxVLA/configs/<family>/<target_config>.py \
  --dry-run
```

For clients without remote reporting, `themis.runner.work_dir` is the local artifact
base; `--work-dir` overrides it and `--run-name` selects a child directory. Those
paths are still confined to the target FluxVLA repository's `work_dirs`. With remote
reporting enabled, these options do not select the authoritative directory;
`--run-name` supplies the optional suffix in FluxVLA's native run ID. Configure the
server-owned base with
`themis.ros_server.evaluation_reporting.result_output_dir` and restart the server
after changing it. `--cfg-options` affects only the current FluxThemis process and is
not passed to the separately started FluxVLA server, so use it only to adjust
`themis.runner`:

```bash
bash scripts/run_ros1_eval.sh \
  --config /abs/path/to/FluxVLA/configs/<family>/<target_config>.py \
  --run-name robot_acceptance_01 \
  --cfg-options themis.runner.episodes_per_task=3
```

The checkpoint is specified only through `--ckpt-path` on the FluxVLA server.
The FluxThemis CLI neither accepts nor loads a checkpoint.

## Action and fail-closed contract

FluxThemis's policy boundary is fixed to `float32 [T, A]` in environment units: `T` is
the action chunk length and `A` is the environment action dimension. Before
responding, the FluxVLA server must complete one of the following two paths:

```text
Normalized model action
  -> denormalize_action from the matching FluxVLA dataset
  -> ROS response: denormalized=true
  -> FluxThemis ActionChunk: environment-unit float32 [T, A]
  -> Environment.step(environment-unit [A])
```

Alternatively, when the model itself already outputs environment units, explicitly
declare `model_outputs_environment_actions=True` and return `denormalized=true`.
Neither path may denormalize the action again in the environment.

The ROS client strictly rejects a mismatched request ID, `ok=false`,
`denormalized=false`, an action element count inconsistent with `[T, A]`, an empty
action, or NaN/Inf. A failed request does not advance the episode's first-request flag
to the completed state and does not automatically resend an action command. See the
[architecture guide](docs/architecture.md).

## Evaluation artifacts

In reported ROS mode, FluxVLA creates and acknowledges a native evaluation directory:

```text
/abs/path/to/FluxVLA/work_dirs/fluxthemis/eval_runs/<checkpoint_stem>/
  EVAL-<suite>-<model_family>-YYYY_MM_DD-HH_MM_SS[-<run_name>]/
```

`result_output_dir` replaces the first path component when configured differently.
The reporter creates these files:

- during rollout: `events.jsonl`, `rank0.txt`, and `rank_progress/rank0.json`;
- at `run_end`: `<suite>/task<task_index>_results.json`,
  `task_status/<suite>_task<task_index>.status`, `failed_tasks.txt`, `summary.txt`,
  `summary.csv`, `summary.json`, and `task_success_rates.csv`.

FluxThemis neither writes nor directly chooses this directory. It carries the
requested `run_name` suffix, validates the stable `run_dir` returned by FluxVLA, and
prints that path after the final acknowledgement. There is no remote `progress.json` or
FluxThemis-side run-name collision check.

To enable Feishu Sheets reporting, set `FEISHU_SHEET_URL`, `FEISHU_APP_ID`, and
`FEISHU_APP_SECRET` in the FluxVLA server shell. The same values may be supplied as
`sheet_url`, `app_id`, and `app_secret` inside
`themis.ros_server.evaluation_reporting.feishu`; its `timeout` field defaults to 10
seconds. (`FEISHU_TIMEOUT` remains a standalone FluxVLA evaluator setting; the ROS
reporter uses this config field.) A row is uploaded only after
a finished, unfiltered full suite contains every authoritative task and exactly the
configured number of completed trials per task. Smoke, filtered, incomplete,
interrupted, and failed runs remain on disk but are not uploaded.

For a model client without `report_service_name`, FluxThemis retains its local
artifact format (`resolved_config.py`, `episodes.jsonl`, `progress.json`,
`summary.json`, and `failure.json`) under `runner.work_dir/run_name`. Only this local
mode uses `--work-dir`, run-name collision checks, and `runner.overwrite`.

## Real-robot safety

`BaseRealRobotEnvironment` defaults to `allow_actuation=False` and requires explicit
`action_low` / `action_high` bounds by default. Before enabling actuation, the operator
must clear the workspace, independently verify the emergency stop, confirm the
control mode and action units, and check velocity, force, and torque limits.

The safety base class can provide only minimal software checks: it rejects actions
that are not one-dimensional, contain NaN/Inf, have a mismatched shape, or exceed
bounds, and it attempts to disconnect after a connection failure. A
platform-specific adapter must still implement a hardware watchdog, safe stop on
communication loss, hardware limits, reliable disconnection, and an emergency-stop
chain independent of the Python/ROS processes. `--dry-run` does not connect to
hardware and cannot replace real-robot acceptance testing.

## Repository structure

```text
fluxthemis/
  cli/                       CLI entry point that reads themis.runner
  engines/runners/           Generic evaluation lifecycle
  environments/{sim,real}/   Simulation and real-robot environment boundaries
  evaluators/                Metric and result aggregation
  model_clients/             ROS client and test clients
  structures.py              Task/Step/Action/Episode data contracts
ros/fluxthemis_msgs/          PredictAction and ReportEvaluation catkin package
ros2/fluxthemis_msgs/         PredictAction and ReportEvaluation rosidl package
scripts/                     FluxThemis command-line wrapper entry points
test/                        Unit tests that require no real ROS graph or hardware
docs/                        Design and integration documentation
```

The FluxThemis repository has no `work_dirs` directory. The remote reporter confines
authoritative output to the target FluxVLA repository's `work_dirs`; local-client
fallback paths are subject to the same boundary.

Run the tests:

```bash
python -m pytest
```
