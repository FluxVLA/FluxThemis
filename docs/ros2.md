# Optional ROS 2 / Python 3.12 Deployment

## Implemented Path

FluxThemis's `FluxVLAROS2ModelClient` and FluxVLA's `FluxVLAROS2Server` use a native `rclpy`
service. The request still contains images, the prompt, named state, episode ID, reset, and seed.
The response still returns an action chunk that has already been denormalized. ROS 2 replaces
only the communication layer; it does not change ownership. FluxVLA always owns the target
configuration, checkpoint, model preprocessing, statistics, and the root directory for
`work_dirs` artifacts. FluxThemis is responsible only for the environment, evaluation lifecycle,
ROS client, and lifecycle-event production; the FluxVLA reporter serializes authoritative
results.

Selections in the single configuration source of truth must be paired:

```python
themis = dict(
    transport=dict(
        service_name='/fluxvla/predict_action',
        report_service_name='/fluxvla/report_evaluation',
        timeout_s=30.0,
        request_timeout_s=30.0,
        image_keys=[...],
        state_keys=[...],
        unnorm_key='...',
        image_encoding='rgb8',
    ),
    runner=dict(
        model_client=dict(type='FluxVLAROS2ModelClient'),
        ...,
    ),
    ros_server=dict(
        ros_version=2,
        evaluation_reporting=dict(
            result_output_dir='work_dirs/fluxthemis',
        ),
        ...,
    ),
)
```

`timeout_s` controls discovery for each configured service; `request_timeout_s` controls each
RPC. After an action deadline failure, the client removes the pending request and cancels the
future, but it does not retry automatically or advance the episode's `reset=true` state.
Reporting transport retries preserve the same request ID, sequence, event type, and payload.

## Python ABI Constraints

`rclpy`, generated message type support, and `cv_bridge` all contain CPython extensions and
cannot be used across Python minor versions:

| Deployment combination | Python | Role |
| --- | ---: | --- |
| Ubuntu 24.04 + ROS 2 Jazzy deb | 3.12 | Optional ROS 2 profile; both endpoints use 3.12 |
| Ubuntu 22.04 + ROS 2 Humble deb | 3.10 | Compatibility path for the existing deployment |
| Mixed 3.10 / 3.12 endpoints | Mixed | Outside the supported installation scope; every process must have ROS extensions matching its own Python version |

When selecting the ROS 2 profile, run the FluxVLA server and FluxThemis in two separate Python
3.12 application environments, and have both source the same Jazzy distribution and the same
interface workspace build. The Jazzy deb packages are built for Ubuntu 24.04's Python 3.12. A
matching minor version is necessary, but it does not make an arbitrary Conda interpreter
automatically compatible with apt extensions. You must still verify at each actual endpoint that
`rclpy`, `cv_bridge`, and generated type support can be imported. The DDS wire contract itself
does not depend on the Python minor version, but that does not permit a process to load
mismatched local extensions. The environment side can be installed independently with
`install.sh --env-name <python3.12-env> --ros auto`; the FluxVLA server side must also be upgraded
to Python 3.12 before this path can be selected. `pip install rclpy` is not an officially
supported fix.

Python 3.10 / Humble remains the path for the existing deployment. Consider building the entire
ROS 2 Python/type-support overlay from source for a custom interpreter only when that interpreter
is strictly required. This path has the highest maintenance cost and is not the default
installation approach.

## Interface Workspace

The ROS 2 package is located at `ros2/fluxthemis_msgs`. On Ubuntu 24.04, the idempotent installer
can install the official Jazzy apt source, `ros-jazzy-ros-base`, `ros-dev-tools`, and
`ros-jazzy-cv-bridge`, then link and build the interface:

```bash
bash scripts/install_ros2_jazzy.sh
# Preview changes only:
bash scripts/install_ros2_jazzy.sh --dry-run
# Custom workspace:
bash scripts/install_ros2_jazzy.sh --workspace /abs/path/to/fluxthemis_ros2_ws
```

The script always builds and validates with the system `/usr/bin/python3`; it does not create,
delete, or modify Conda environments. If `fluxthemis_msgs` in the workspace already points to
this repository, subsequent runs reuse it directly. If the same path points elsewhere, the
script stops without overwriting it.

For a manual build, both sides must still source the same build:

```bash
source /opt/ros/jazzy/setup.bash
mkdir -p /abs/path/to/fluxthemis_ros2_ws/src
ln -s /root/projects/FluxThemis/ros2/fluxthemis_msgs \
  /abs/path/to/fluxthemis_ros2_ws/src/fluxthemis_msgs
cd /abs/path/to/fluxthemis_ros2_ws
rosdep install --from-paths src --ignore-src -r -y --rosdistro jazzy
colcon build --symlink-install --packages-select fluxthemis_msgs \
  --cmake-args -DPython3_EXECUTABLE=/usr/bin/python3 \
               -DPYTHON_EXECUTABLE=/usr/bin/python3
source install/setup.bash
ros2 interface show fluxthemis_msgs/srv/PredictAction
ros2 interface show fluxthemis_msgs/srv/ReportEvaluation
python -c 'import rclpy, cv_bridge; from fluxthemis_msgs.srv import PredictAction, ReportEvaluation; print(PredictAction, ReportEvaluation)'
```

The commands above are written for Bash. When using Zsh, source `/opt/ros/jazzy/setup.zsh` and
the workspace's `install/setup.zsh`; do not source `setup.bash` directly from Zsh.

Do not source both the ROS 1 catkin and ROS 2 ament versions of `fluxthemis_msgs` in one process.
See [ROS 1 Deployment](ros1.md) for the environment roles of the two optional profiles and the
ROS 1 source build.

Use the following read-only command to validate the Python ABI, `rclpy`, `cv_bridge`, `launch`,
and the generated service type independently in the optional Jazzy environment:

```bash
bash scripts/check_env.sh \
  --env-name fluxthemis-ros2 \
  --ros auto \
  --smoke-level none
```

## Launch and Networking

Set the same domain on both sides, then start them separately:

```bash
export ROS_DOMAIN_ID=42

# FluxVLA environment
bash /root/projects/FluxVLA/scripts/ros_inference_server.sh \
  --config /root/projects/FluxVLA/configs/<family>/<config>.py \
  --ckpt-path /abs/path/to/checkpoint \
  --ros-version 2

# FluxThemis / simulator environment
fluxthemis-eval \
  --config /root/projects/FluxVLA/configs/<family>/<config>.py
```

The same episode-affine multi-GPU topology is available with ROS 2:

```bash
# FluxVLA / inference environment
export CUDA_VISIBLE_DEVICES=0,1
bash /root/projects/FluxVLA/scripts/ros_inference_server.sh \
  --config /root/projects/FluxVLA/configs/<family>/<config>.py \
  --ckpt-path /abs/path/to/checkpoint \
  --ros-version 2 \
  --num-workers 2

# FluxThemis / simulator environment
fluxthemis-eval \
  --config /root/projects/FluxVLA/configs/<family>/<config>.py \
  --parallel-workers 2 \
  --simulator-gpus 2,3
```

ROS 2 uses a multithreaded executor with independent prediction and reporting
callback groups. FluxVLA still requires simulator workers not to exceed inference
replicas and rejects the mismatch before rollout. This prevents queued prediction
callbacks from starving the `episode_end` callback that releases episode affinity.

Both processes must read the same target configuration from the FluxVLA repository. Only the
FluxVLA server loads the checkpoint through `--ckpt-path`; FluxThemis does not load a checkpoint
or serialize authoritative remote results. The launch commands are unchanged: when
`report_service_name` is configured, the same server also advertises `ReportEvaluation`, and the
client waits for both services.

Each lifecycle event is acknowledged with a stable authoritative directory. FluxVLA writes the
native layout below and emits `[ros-eval]` / `[eval-progress]` messages in its terminal:

```text
<result_output_dir>/eval_runs/<checkpoint_stem>/
  EVAL-<suite>-<model_family>-YYYY_MM_DD-HH_MM_SS[-<run_name>]/
```

The optional Feishu integration reads `FEISHU_SHEET_URL`, `FEISHU_APP_ID`,
and `FEISHU_APP_SECRET` from the FluxVLA server environment; the server-side
`feishu.timeout` setting defaults to 10 seconds. It uploads only a finished, unfiltered,
complete suite with the configured trial count for every authoritative task. Smoke, filtered,
partial, interrupted, and failed runs are kept locally but not uploaded.

If `report_service_name` is omitted, PredictAction still works without the report type. Only
that local-client compatibility mode writes FluxThemis's `episodes.jsonl`, `progress.json`, and
`summary.json` under `runner.work_dir/run_name` and applies local run-name collision rules.

Cross-node container or Kubernetes deployments must also verify DDS discovery, UDP, and
fragmentation of large multi-camera messages. Seeing a node in `ros2 node list` is not sufficient
evidence that the image service is stable. Before a production rollout, validate at least one
real dual-camera request/response, the RPC deadline, and fail-closed behavior after a network
disconnect.

To select ROS 1, change the server to `ros_version=1`, change the client to
`FluxVLAROSModelClient`, remove `request_timeout_s`, and use the Python 3.10 Noetic overlay. ROS 1
and Jazzy can run on the same machine, but they must use separate environments and processes.
