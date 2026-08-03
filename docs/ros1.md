# ROS 1 / Python 3.10 Deployment

For the tested PI0.5 LIBERO-10 installation and copy-paste commands for the
FluxVLA and FluxThemis terminals, see the
[FluxVLA + FluxThemis LIBERO quickstart](quickstart.md).

## Goals and Boundaries

The current `fluxthemis` Conda environment is pinned to Python 3.10 and uses a source-built ROS
1 Noetic stack for LIBERO or real-robot evaluation. The existing `/root/projects/FluxVLA`
environment does not need a Python upgrade. The installer only configures the same ROS 1
activation hook for it and does not install or upgrade FluxVLA Python packages.

Ubuntu 24.04 is outside Noetic's official support matrix, and Noetic has reached end of life.
This setup does not mix Focal deb packages into Noble. Instead, it builds the source closure
required by `ros_comm`, `sensor_msgs`, `fluxthemis_msgs/PredictAction`, and
`fluxthemis_msgs/ReportEvaluation`. The build uses Python
3.10, `ROSCONSOLE_BACKEND=print`, and `SETUPTOOLS_DEB_LAYOUT=OFF`.

The repository supports two optional profiles; this does not mean that both must currently be
installed. Only Noetic/cp310 is required now. If Jazzy/cp312 is installed later, it must use a
different environment and process:

| Environment | Python | ROS workspace |
| --- | ---: | --- |
| `fluxthemis` | 3.10 | `/root/fluxthemis_ros1_ws/install_isolated` |
| `fluxthemis-ros2` (optional) | 3.12 | `/opt/ros/jazzy` + `/root/fluxthemis_ros2_ws/install` |

The package is named `fluxthemis_msgs` in both versions, so do not source both setup files in
sequence in one shell. Open a new, clean terminal when switching versions. ROS 1 and ROS 2
graphs also do not interoperate automatically merely because their services have the same name.
Deploy a separate bridge or sidecar when communication across versions is required.

The complete installer installs the matching runtime and paired hooks into the primary
environment selected by `--env-name`. In ROS 1 mode, it also installs the same activation-only
hook in the `fluxvla` environment by default. Use `--fluxvla-env-name` to change the target, or
`--skip-fluxvla-hook` to disable this behavior. If both profiles are retained on one machine,
switching Conda environments first removes the previous overlay. If a ROS setup was sourced
manually and Conda cannot determine its origin, the next hook refuses to load a mixed setup.

## Installation

Install the complete LIBERO + ROS 1 stack from an existing Python 3.10 environment:

```bash
cd /root/projects/FluxThemis
bash scripts/install.sh --env-name fluxthemis --ros auto
```

Install or repair only ROS 1:

```bash
bash scripts/install_ros1_noetic_py310.sh \
  --env-name fluxthemis \
  --workspace /root/fluxthemis_ros1_ws
```

The script installs build dependencies, obtains the pinned Noetic release sources by commit SHA,
links the repository's catkin message package, performs an isolated build, and validates `rospy`,
`sensor_msgs/Image`, both service types, `rospack`, and `rossrv`. It also installs
`scripts/activate_ros1.sh` into `etc/conda/activate.d` in the target environment and adds the
path containing Conda's `libpython3.10.so` to `LD_LIBRARY_PATH`.

After reinstalling or switching environments, run these read-only acceptance checks. The second
command also creates a LIBERO environment and takes one step:

```bash
bash scripts/check_env.sh --ros 1 --smoke-level imports
bash scripts/check_env.sh --ros 1 --smoke-level libero
```

## Launch

Terminal one: start the environment-side launcher first. It activates `fluxthemis`, starts or
reuses `roscore`, waits for the configured FluxVLA inference and reporting services, and runs the
evaluation:

```bash
cd /root/projects/FluxThemis
bash scripts/run_ros1_eval.sh \
  --config /root/projects/FluxVLA/configs/<family>/<config>.py
```

Terminal two: keep the original FluxVLA Python 3.10 environment and start the ROS 1 server. On
first use after installation, open a new terminal or reactivate `fluxvla` once. You do not need
to source the overlay manually afterward:

```bash
conda activate fluxvla
cd /root/projects/FluxVLA
bash scripts/ros_inference_server.sh \
  --config /root/projects/FluxVLA/configs/<family>/<config>.py \
  --ckpt-path /abs/path/to/checkpoint \
  --ros-version 1
```

For parallel evaluation, keep the same two public services and add workers behind
them:

```bash
# FluxVLA terminal: two complete model replicas.
export CUDA_VISIBLE_DEVICES=0,1
bash scripts/ros_inference_server.sh \
  --config /root/projects/FluxVLA/configs/<family>/<config>.py \
  --ckpt-path /abs/path/to/checkpoint \
  --ros-version 1 \
  --num-workers 2

# FluxThemis terminal: two spawned simulator processes.
bash scripts/run_ros1_eval.sh \
  --config /root/projects/FluxVLA/configs/<family>/<config>.py \
  --workers 2 \
  --sim-gpus 2,3
```

FluxVLA assigns one full-model replica to an episode until the acknowledged
`episode_end`; it never round-robins requests within an episode. `--sim-gpus` uses
physical host ids, whereas FluxVLA device arguments are logical ids after
`CUDA_VISIBLE_DEVICES`. Prefer disjoint pools, and never request more simulator
workers than inference replicas. The server rejects that mismatch at `run_start`.

After a successful launch, the launcher retains any master it started by default, so subsequent
evaluations do not require restarting the FluxVLA server. Add `--shutdown-master` for a one-shot
run. It always cleans up its own master after a failure or interrupt.

The two terminal commands remain unchanged when reporting is enabled. Set
`themis.transport.report_service_name` (for example `/fluxvla/report_evaluation`) and configure
`themis.ros_server.evaluation_reporting.result_output_dir` in the shared FluxVLA config. The
environment side then sends acknowledged `run_start`, `episode_start`, `episode_end`, and
`run_end` events. FluxVLA writes under:

```text
<result_output_dir>/eval_runs/<checkpoint_stem>/
  EVAL-<suite>-<model_family>-YYYY_MM_DD-HH_MM_SS[-<run_name>]/
```

Progress lines such as `[eval-progress] episodes=N/T ...`, final artifact paths, and Feishu
report/skip messages appear in terminal two, where the FluxVLA server runs. The authoritative
directory is returned in every accepted response and printed by FluxThemis at the end; remote
mode does not create `progress.json` or apply a local run-name collision check.

For optional Feishu Sheets reporting, export `FEISHU_SHEET_URL`, `FEISHU_APP_ID`, and
`FEISHU_APP_SECRET` in terminal two. The server-side `feishu.timeout` setting defaults to 10
seconds. Only a finished, unfiltered, complete suite with the configured number of trials per
authoritative task is uploaded; smoke, filtered, partial, interrupted, and failed runs are
skipped.

Omitting `report_service_name` preserves the PredictAction-only compatibility path. Only that
local-client mode uses `runner.work_dir/run_name`, atomic `progress.json`, and the old
`--run-name` collision / `runner.overwrite` behavior.

The target FluxVLA configuration must pair `ros_version=1` with
`FluxVLAROSModelClient`. ROS 1 transport does not have the ROS 2 `request_timeout_s` field. Use
`persistent` to control `ServiceProxy` connection reuse.

## Why cv_bridge Is Not Installed

The Noble/Jazzy build of `cv_bridge` targets Python 3.12 and its corresponding Boost.Python ABI,
so it cannot be loaded safely into Python 3.10. The current protocol sends only contiguous
`uint8 [H, W, 3]` RGB/BGR images. The client therefore populates the `height`, `width`,
`encoding`, `step`, and `data` fields of `sensor_msgs/Image` directly, and the FluxVLA server
decodes them strictly according to the same rules. The wire schema remains unchanged, while
avoiding binary coupling to OpenCV/Boost.Python.

Only `rgb8` and `bgr8` are currently accepted. If depth, 16-bit, or compressed images are needed
in the future, add explicit endianness, stride, and dtype validation for those encodings instead
of implicitly reusing the existing three-channel path.
