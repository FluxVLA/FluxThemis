# FluxThemis Architecture

## Hard Responsibility Boundaries

FluxVLA and FluxThemis form a two-process evaluation system, but they do not
share model implementation code or runtime objects. The target FluxVLA MMEngine
configuration is the sole source of configuration.

| System | Sole ownership |
| --- | --- |
| FluxVLA | `model` / `inference_model`, configuration, checkpoint, dataset, preprocessing, statistics, action denormalization, device and mixed precision, both ROS services, and authoritative result serialization under `work_dirs` |
| FluxThemis | environment, `EvalRunner`, evaluator, ROS client, task and episode lifecycles, and evaluation-event production |

This establishes the following prohibitions:

- FluxThemis does not construct or load the FluxVLA model or checkpoint;
- FluxThemis does not construct the model-side dataset, read training statistics,
  or duplicate preprocessing;
- FluxThemis runners, evaluators, and environments do not perform action
  denormalization;
- The FluxVLA server does not connect to a robot or simulation environment and
  does not determine episode success;
- For reported ROS runs, FluxThemis does not serialize authoritative results;
  every lifecycle event must be acknowledged by the FluxVLA reporter;
- The FluxThemis repository does not contain or own `work_dirs`; authoritative
  evaluation artifacts are archived in the FluxVLA repository;
- Do not derive a second configuration for FluxThemis. Any resolved
  configuration carried in `run_start` is audit data, not a new source.

## Python and ROS Deployment Baseline

FluxThemis supports Python 3.10 and 3.12. The current compatibility baseline
uses separate Python 3.10 environments for FluxVLA and `fluxthemis`, with both
sourcing the same source-built ROS 1 Noetic overlay. Noetic has reached EOL, so
this is only a transitional path for FluxVLA before it is upgraded. The codebase
also retains optional installation and runtime support for ROS 2 Jazzy / Python
3.12, but the current installation does not require creating that environment;
select that profile after FluxVLA is upgraded. Neither the two endpoints of one
communication path nor a single process may mix Python 3.10/cp310 and Python
3.12/cp312 ROS packages.

Runtime version changes do not alter the responsibility boundaries: the target
configuration, checkpoint, and `work_dirs` root remain under the sole management
of FluxVLA. FluxThemis consumes `themis.runner`, performs rollout, and reports
results to the FluxVLA service that writes under `work_dirs`.

## Single Configuration and Its Consumers

Add a `themis` section to the target FluxVLA configuration. Its fields are
consumed as follows:

| Configuration field | Consumer | Purpose |
| --- | --- | --- |
| `model` / `inference_model` | FluxVLA server | Construct the inference model; prefer `inference_model` when present |
| `<dataset_section>.dataset` | FluxVLA server | Convert raw ROS observations into model inputs |
| `<dataset_section>.denormalize_action` | FluxVLA server | Restore environment units using the statistics associated with the checkpoint |
| `themis.transport` | Both sides | Share inference/report service names, observation schema, and ROS client parameters |
| `themis.ros_server` | FluxVLA server | Configure inference plus server-owned evaluation reporting |
| `themis.runner` | FluxThemis CLI | Construct the environment, model client, evaluator, and runner |
| `themis.runner.work_dir` | FluxThemis runner | Local-client artifact fallback; it does not select the remote reporter's run directory |
| `themis.ros_server.evaluation_reporting.result_output_dir` | FluxVLA server | Select the reporter root inside the target FluxVLA `work_dirs` tree |
| `themis.custom_imports` | FluxThemis CLI | Register out-of-repository environment/evaluator plugins |

The standard configuration shape is:

```python
themis = dict(
    custom_imports=dict(
        imports=['your_robot_package.fluxthemis_plugins'],
        allow_failed_imports=False,
    ),
    transport=dict(
        service_name='/fluxvla/predict_action',
        report_service_name='/fluxvla/report_evaluation',
        timeout_s=30.0,
        image_keys=['camera_front', 'camera_wrist'],
        state_keys=[
            'robot0_eef_pos',
            'robot0_eef_quat',
            'robot0_gripper_qpos',
        ],
        unnorm_key='your_robot_stats_namespace',
        image_encoding='rgb8',
        persistent=True,
    ),
    runner=dict(
        type='EvalRunner',
        environment=dict(type='YourEnvironment'),
        model_client=dict(type='FluxVLAROSModelClient'),
        evaluator=dict(type='SuccessRateEvaluator'),
        seed=7,
        episodes_per_task=1,
        max_episode_steps=1000,
        execute_horizon=10,
        stop_on_success=True,
        work_dir='work_dirs/fluxthemis',
    ),
    ros_server=dict(
        ros_version=1,
        dataset_section='eval',
        evaluation_reporting=dict(
            result_output_dir='work_dirs/fluxthemis',
        ),
        device='cuda:0',
        mixed_precision_dtype='bf16',
        enable_mixed_precision=True,
        model_outputs_environment_actions=False,
        forward_seed=False,
        denormalize_context={},
    ),
)
```

`dataset_section` may only be `eval` or `inference`. The selected section must
provide a `dataset` compatible with the model. When
`model_outputs_environment_actions=False`, it must also provide
`denormalize_action`. The latter may be omitted when the model explicitly emits
environment-unit actions, but a dataset is still required for input
preprocessing.

### FluxThemis CLI Loading Rules

`fluxthemis-eval` uses `Config.fromfile()` to load the target FluxVLA
configuration and merges `--cfg-options`, then performs the following restricted
procedure:

1. Require a mapping-valued `themis` section;
2. Execute only `themis.custom_imports` through the MMEngine import helper; do
   not execute the top-level FluxVLA `custom_imports`;
3. Require the configuration to reside under `configs/` in a FluxVLA repository
   that contains `fluxvla/__init__.py`, and use this to determine the FluxVLA
   root;
4. Deep-copy `themis.runner`, apply `--work-dir` and `--run-name`, and normalize
   the work directory to an absolute path within FluxVLA `work_dirs`;
5. When the model client is `FluxVLAROSModelClient` or
   `FluxVLAROS2ModelClient`, inject the sibling `themis.transport` into the
   client only at construction time;
6. Reject a duplicate `transport` definition inside
   `themis.runner.model_client`;
7. Construct FluxThemis components only from the processed `themis.runner`.

Consequently, FluxThemis does not construct objects from `model`,
`inference_model`, the selected dataset section, or `themis.ros_server`. In
reported mode it includes the resolved target configuration in `run_start` so
the FluxVLA-side event log can be audited. A local-client run retains the old
`resolved_config.py` snapshot.

`--cfg-options` modifies only the Config in the current FluxThemis process and
does not propagate to the separate FluxVLA server. It should override only
`themis.runner`. When model, dataset, transport, or server parameters must
change, modify the target configuration read by both sides and restart both
processes.

### Evaluation Artifact Path Invariant

When `themis.transport.report_service_name` is configured, the FluxVLA reporter
owns the run directory. Its root is
`themis.ros_server.evaluation_reporting.result_output_dir`, defaulting to
`<FluxVLA>/work_dirs/fluxthemis`. Relative paths are resolved inside FluxVLA and
the normalized result must remain under `<FluxVLA>/work_dirs`.

The acknowledged directory follows FluxVLA's native layout:

```text
<result_output_dir>/eval_runs/<checkpoint_stem>/
  EVAL-<suite>-<model_family>-YYYY_MM_DD-HH_MM_SS[-<run_name>]/
```

FluxThemis does not predict this path or create it. The first accepted event
returns it, every later acknowledgement must return the same value, and the CLI
prints it at completion. The reporter writes `events.jsonl`, `rank0.txt`, and
`rank_progress/rank0.json` during the run. On `run_end` it also creates the
native per-task results/status files and `failed_tasks.txt`, `summary.txt`,
`summary.csv`, `summary.json`, and `task_success_rates.csv`. Progress and artifact
messages are emitted in the FluxVLA terminal.

Without `report_service_name`, the client is in local artifact mode. Only that
mode uses `runner.work_dir/run_name`, `resolved_config.py`, `episodes.jsonl`,
atomic `progress.json`, `summary.json`, and `failure.json`; `--work-dir`, run-name
collision checks, and `runner.overwrite` apply there. They do not govern a
reported run.

The optional `themis.ros_server.evaluation_reporting.feishu` mapping accepts
`sheet_url`, `app_id`, `app_secret`, and `timeout`. Missing credential values
fall back to `FEISHU_SHEET_URL`, `FEISHU_APP_ID`, and `FEISHU_APP_SECRET` in the
FluxVLA server process. Upload is deliberately full-suite-only: `run_end` must be
`finished`, the run must have no task filter, every task in the authoritative
suite manifest must be present, and every task must have the configured number
of completed trials. Smoke, filtered, partial, interrupted, and failed runs keep
their local FluxVLA artifacts but are skipped for Feishu; the reason appears in
the FluxVLA terminal.

### FluxVLA Server Loading Rules

`FluxVLA/scripts/ros_inference_server.sh` is the public launcher. It resolves
the FluxVLA repository root and forwards all arguments to
`scripts/ros_inference_server.py`, which is responsible for:

1. Constructing the model from `inference_model`, when present, or otherwise
   from `model`;
2. Loading the checkpoint from `--ckpt-path`;
3. Reading `themis.ros_server.dataset_section`;
4. Constructing dataset preprocessing and `denormalize_action` from the selected
   section;
5. Reading `themis.transport` and `themis.ros_server`, advertising
   `PredictAction`, and optionally advertising `ReportEvaluation`;
6. Running `predict_action` on the configured device and at the configured
   precision;
7. Returning actions that are already in environment units, or refusing to
   produce a successful response;
8. When reporting is configured, validating ordered lifecycle events, writing
   native FluxVLA artifacts, logging progress, and acknowledging the stable
   authoritative run directory.

The server does not construct a FluxThemis environment, evaluator, or runner.
The reporter may copy task IDs, trial count, and run-name fields from
`themis.runner` as validation/audit metadata only.

## Components and Public Data Contracts

FluxThemis uses four MMEngine registries:

| Registry | Responsibility | Current core implementations |
| --- | --- | --- |
| `ENVIRONMENTS` | Task handling, reset, single-step environment actions, and resource management | `MockEnvironment`, `GymnasiumEnvironment`, `LiberoEnvironment`, `BaseRealRobotEnvironment` |
| `MODEL_CLIENTS` | Episode policy state, ROS requests, and ActionChunk handling | `MockModelClient`, `FluxVLAROSModelClient`, `FluxVLAROS2ModelClient` |
| `EVALUATORS` | Lifecycle hooks, metrics, and summaries | `SuccessRateEvaluator` |
| `RUNNERS` | Component composition and task × episode execution | `EvalRunner` |

The public structures are defined in `fluxthemis/structures.py`:

- `TaskSpec`: task ID, natural-language description, and environment metadata;
- `ActionChunk`: environment-unit actions with canonical shape `float32 [T, A]`;
- `StepResult`: next observation, reward, terminated, truncated, success, and
  info;
- `EpisodeResult`: episode seed, return, step count, termination reason, model
  call count, and prediction metadata.

## End-to-End Data Flow

```text
Target FluxVLA configuration
  |-- model / inference_model + checkpoint
  |-- selected section.dataset + denormalize_action
  |-- themis.transport + themis.ros_server
  |       -> FluxVLA ROS inference server
  |
  `-- themis.runner + themis.custom_imports
          -> FluxThemis EvalRunner
          -> Environment.reset()/step()
          -> FluxVLAROSModelClient or FluxVLAROS2ModelClient

Environment raw observation
  -> ROS request: images + prompt + state + episode metadata
  -> FluxVLA dataset preprocessing
  -> FluxVLA model.predict_action()
  -> FluxVLA denormalize_action (if model output is not yet in environment units)
  -> ROS response: flat actions + [T, A] metadata + denormalized=true
  -> ActionChunk(environment-unit float32 [T, A])
  -> EvalRunner splits into [A] according to execute_horizon
  -> Environment.step(environment-unit [A])

EvalRunner lifecycle result
  -> ReportEvaluation event with stable run ID and ordered sequence
  -> FluxVLA validates and persists the event
  -> acknowledgement: accepted + request ID + authoritative run_dir
  -> native FluxVLA artifacts, server-terminal progress, optional Feishu row
```

The environment only needs to produce the raw observations described by
`themis.transport.image_keys` and `state_keys`, plus the task description.
Multiple states carry their respective names and lengths, and the server
reconstructs them under their original keys. Therefore, LIBERO's
pose/quaternion/gripper state, RoboCasa's multiple joint groups, and a real
robot's single `qpos` require no protocol-level hard-coding. All model-specific
resizing, normalization, tokenization, padding, and state/action statistics
remain in the FluxVLA server.

## Runner Lifecycle

The normal state flow is:

```text
created
  -> run_setup()
       model_client.setup()
       environment.setup()
       evaluator.setup(run_dir)
       environment.tasks()
       evaluator.on_run_start(...)
  -> setup
  -> run()
       task x episode
  -> finished / failed
  -> cleanup()
       evaluator.close()
       environment.close()
       model_client.close()
  -> closed
```

With `ReportEvaluation` enabled, the lifecycle is:

1. The client creates one opaque run ID and reports `run_start`; FluxVLA creates
   the native run directory and acknowledges it;
2. The runner computes `episode_seed = seed + global_episode_index`;
3. `model_client.reset(task, episode_index, episode_seed)` only caches the
   episode information and marks the next successful request as the first
   request;
4. The runner reports `episode_start`, then `environment.reset(...)` returns
   the initial observation and info;
5. The evaluator receives `on_episode_start`;
6. The client requests an ActionChunk through `PredictAction`;
7. The runner validates that the actions are two-dimensional, nonempty, and
   finite, then executes at most `execute_horizon` actions of shape `[A]`;
8. Success, terminated, truncated, or `max_episode_steps` ends the episode;
9. The evaluator aggregates the result and FluxThemis reports `episode_end`;
   the episode is complete only after FluxVLA accepts it;
10. FluxThemis reports `run_end` with the terminal status and summary. FluxVLA
    finalizes native artifacts and may upload a qualifying full suite to Feishu.

FluxThemis performs no authoritative file writes in this path. An interrupted or
failed run makes a best-effort terminal `run_end`; the original exception remains
the evaluation failure. Without reporting, the runner instead retains its local
`episodes.jsonl` / `progress.json` / `summary.json` lifecycle.

Prediction metadata preserves the request ID, episode ID, first-request flag,
seed, denormalization confirmation, and round-trip, server inference, network,
and serialization timings.

### Parallel rollout topology

`parallel_workers>1` changes execution, not ownership. The EvalRunner parent
creates deterministic task-major jobs and assigns job ordinal `n` to simulator
worker `(n - 1) % parallel_workers`. Spawned children each own one environment
and one inference-only ROS client. The parent alone owns the report run ID,
global report sequence, result aggregation, terminal status, and FluxVLA
acknowledgements.

FluxVLA exposes the same two public services. Its frontend owns one reporter and
dispatches prediction calls to independent full-model processes. The first call
for `<report_run_id>:<ordinal>` leases one model replica; every later call for
that episode uses the same process, and only an accepted `episode_end` releases
the lease. A failed/interrupted FluxThemis run first terminates all simulator
children and then sends `run_end`, preventing inference calls after leases are
released. The simulator worker count may not exceed the inference replica count.

This is data-parallel episode throughput. It is intentionally not DDP, FSDP,
tensor parallelism, or per-request round-robin scheduling.

## ROS Service Contracts

The service types are `fluxthemis_msgs/PredictAction` and optional
`fluxthemis_msgs/ReportEvaluation`. The ROS 2 ament package is
located at `ros2/fluxthemis_msgs`; the ROS 1 catkin package is located at
`ros/fluxthemis_msgs`. Tests guarantee that the two `.srv` files are byte-for-byte
identical. Both sides must use the same ROS version and artifacts from the same
workspace build; a single process must not mix sourced environments.

### PredictAction request

| Field | ROS type | Semantics and constraints |
| --- | --- | --- |
| `header` | `std_msgs/Header` | ROS timestamp at client send time |
| `request_id` | `string` | Unique for each call; the response must echo it unchanged |
| `episode_id` | `string` | Stable ID; reported runs use globally unique `<report_run_id>:<ordinal>` |
| `reset` | `bool` | `true` until the episode's first request succeeds; `false` thereafter |
| `seed` | `int64` | Episode seed generated by the runner; `forward_seed` controls whether it is passed to the model |
| `prompt` | `string` | Nonempty task prompt, usually from `TaskSpec.description` |
| `image_names` | `string[]` | One-to-one with `images`, in `image_keys` order |
| `images` | `sensor_msgs/Image[]` | `uint8 [H, W, 3]`, encoded as specified by `image_encoding` |
| `state_names` | `string[]` | Raw observation state keys, in exactly `state_keys` order |
| `state_sizes` | `uint32[]` | One-dimensional length of each state, one-to-one with `state_names` |
| `state_values` | `float32[]` | Finite state values concatenated in `state_names` order |
| `unnorm_key` | `string` | Selects the FluxVLA statistics namespace |

### PredictAction response

| Field | ROS type | Semantics and constraints |
| --- | --- | --- |
| `header` | `std_msgs/Header` | Server response timestamp |
| `request_id` | `string` | Must exactly equal the request ID |
| `ok` | `bool` | `true` only when inference and all action processing complete successfully |
| `error` | `string` | Error description when `ok=false` |
| `actions` | `float32[]` | Row-major flattened environment-unit actions; length must equal `action_horizon * action_dim` |
| `action_horizon` | `uint32` | `T`; must be greater than zero |
| `action_dim` | `uint32` | `A`; must be greater than zero |
| `denormalized` | `bool` | Must be `true` in a successful response |
| `inference_time_s` | `float64` | Nonnegative, finite server inference duration |

The client reshapes flattened `actions` to `float32 [T, A]`. The protocol permits
neither an implicit batch dimension nor inference of the action horizon or
action dimension from the response shape.

### ReportEvaluation request and acknowledgement

This service is advertised only when
`themis.transport.report_service_name` is present. Omitting that field preserves
the PredictAction-only path and does not require importing the generated report
type.

| Request field | ROS type | Semantics and constraints |
| --- | --- | --- |
| `header` | `std_msgs/Header` | Client send timestamp |
| `request_id` | `string` | Unique event ID; reused unchanged for transport retries |
| `run_id` | `string` | Opaque ID stable for the whole evaluation |
| `sequence` | `uint64` | Strictly contiguous event order within the run, starting at 1 |
| `event_type` | `string` | `run_start`, `episode_start`, `episode_end`, or `run_end` |
| `payload_json` | `string` | JSON object containing lifecycle metadata or results |

| Response field | ROS type | Semantics and constraints |
| --- | --- | --- |
| `header` | `std_msgs/Header` | Server response timestamp |
| `request_id` | `string` | Must echo the request ID exactly |
| `accepted` | `bool` | True only after the event is valid and persisted |
| `error` | `string` | Rejection reason |
| `run_dir` | `string` | Nonempty authoritative FluxVLA directory, stable for the run |

The client retries a transport failure at most three times with the same
request ID, sequence, event type, and payload. It advances the sequence only
after an accepted response. It fails closed on a mismatched request ID, a
rejection, an empty directory, or a directory that changes during the run.

## ROS Client State and Validation

The current default path is the ROS 1 `FluxVLAROSModelClient`. Its ROS imports
are lazy: importing FluxThemis does not require ROS. `setup()` imports `rospy`,
`sensor_msgs`, and the generated `fluxthemis_msgs.srv`, then calls
`rospy.core.is_initialized()` to inspect the current process. If the environment
plugin has not already created a node, it initializes an anonymous
`fluxthemis_eval` node. It then calls
`wait_for_service(service_name, timeout_s)` before creating an optionally
persistent `ServiceProxy`. The current `timeout_s` controls only how long to wait
for service availability; it is not an application-level inference-call
deadline.

ROS 1 `themis.transport` permits only the following fields; unknown fields are
rejected at construction time:

| Field | Client behavior |
| --- | --- |
| `service_name` | ROS service name |
| `report_service_name` | Optional acknowledged evaluation-result service; enables remote artifact ownership |
| `timeout_s` | Positive and finite; used by `wait_for_service` |
| `image_keys` | Required, nonempty, and unique; determines image order |
| `state_keys` | Required, nonempty, and unique; determines the split and reconstruction order of named states |
| `unnorm_key` | Default statistics namespace; may be explicitly overridden by task context |
| `image_encoding` | `rgb8` or `bgr8`; written directly to a standard `sensor_msgs/Image` |
| `persistent` | Must be a bool; controls `ServiceProxy` connection reuse |

The migration-path `FluxVLAROS2ModelClient` lazily imports `rclpy`, `cv_bridge`,
and the rosidl-generated service in `setup()`, and creates an asynchronous client
after waiting for the service. Each call uses `call_async()` and
`spin_until_future_complete()` to implement a bounded wait. When
`request_timeout_s` expires, it removes the pending request, cancels the future,
and fails closed. The client closes only the rclpy context it initialized; it
does not perform a global shutdown when reusing an external context.

The ROS 2 configuration fields are `service_name`, `report_service_name`,
`timeout_s`, `request_timeout_s`, `image_keys`, `state_keys`, `unnorm_key`, and
`image_encoding`. Of these, `timeout_s` controls discovery and
`request_timeout_s` controls each service RPC. Action requests are not retried;
reporting uses the stable-identity retry policy described above. ROS 2 has no
counterpart to `persistent`.

Input validation includes:

- observation and context must be mappings;
- every image key must be present, and each image must be nonempty
  `uint8 [H, W, 3]`;
- every state key must be present, and each state segment must be
  one-dimensional, nonempty, floating-point, and finite;
- prompt must be a nonempty string;
- `setup()` and episode `reset()` must have completed first.

Responses are handled fail-closed: any of the following conditions raises an
error and produces no ActionChunk:

- The response request ID does not match the current request;
- `ok=false`;
- `denormalized=false`;
- `action_horizon` or `action_dim` is not a positive integer;
- actions are not a flat array, or the element count does not equal `T * A`;
- actions contain NaN/Inf or cannot be converted to float32;
- `inference_time_s` is nonnumeric, negative, or nonfinite.

Only after the complete response passes validation does the client advance the
episode's first-request state from `reset=true` to `false`. Failed calls are not
automatically retried, avoiding execution of the same action request twice.
`close()` idempotently closes the ROS proxy.

The server's dataset buffer and stateful policy history are process-level state,
so one service instance supports only one active evaluation client. Concurrent
runs must use separate server processes and service names; otherwise, their
episode histories will interrupt one another.

## FluxVLA Server Action Responsibilities

The FluxVLA server is the sole owner of preprocessing, statistics, and
denormalization:

1. Reconstruct the raw sample from `image_names`,
   `state_names/state_sizes`, and the prompt;
2. Apply preprocessing consistent with training/inference using the selected
   dataset;
3. Call the model on `device` under the mixed-precision configuration;
4. Pass `is_new_episode` to the dataset on every request and, when the dataset
   does not generate it, add `reset_history` to the model batch;
5. When `forward_seed=True`, temporarily set and restore the Python, NumPy, and
   Torch RNGs while holding the request lock, and also pass the seed to
   `predict_action`; when it is `False`, do not claim reproducibility for a
   stochastic model;
6. If `model_outputs_environment_actions=False`, call `denormalize_action` from
   the selected section using `unnorm_key` and `denormalize_context`;
7. Return `ok=true, denormalized=true` only after the actions are nonempty,
   finite, environment-unit values of shape `[T, A]`.

If the model returns `[1, T, A]`, the server may remove the batch dimension after
confirming that the batch size is 1. A single-step action may be canonicalized to
`[1, A]`. Any other rank, batch size, empty shape, or nonfinite action must
produce an error rather than requiring the client to guess.

By default, the server passes the complete model output to one
`denormalize_action` call, preserving existing FluxVLA chunk/batch semantics.
Set `themis.ros_server.denormalize_per_action=True` only when the target
transform explicitly requires individual `[A]` inputs.

### Denormalization Invariant

```text
normalized model action
  --only inside the FluxVLA server--> environment-unit action
  --ROS--> FluxThemis ActionChunk float32 [T, A]
  --> Environment.step([A])
```

`model_outputs_environment_actions=True` applies only when the model contract
explicitly guarantees environment-unit output; it is not a general-purpose
switch for bypassing statistics validation. Regardless of the path taken, every
successful response must set `denormalized=true`, and the client unconditionally
fails closed on `false`. The environment may additionally validate action bounds
in environment units, but it must not perform denormalization a second time.

## Extending Environments and Evaluators

When adding a platform:

1. Derive simulation adapters from `BaseEnvironment`; preferably derive real
   robot adapters from `BaseRealRobotEnvironment`;
2. Implement `setup()`, `tasks()`, `reset()`, `step()`, and an idempotent
   `close()`;
3. Register with `@ENVIRONMENTS.register_module()` and import out-of-repository
   plugins through `themis.custom_imports`;
4. Emit raw observation data consisting of `image_keys`, `state_keys`, and the
   task prompt;
5. Accept only one environment-unit `[A]` in `step()` and return `StepResult`;
6. Explicitly define success, terminated, truncated, action units, coordinate
   frames, and timing semantics;
7. Test deterministic reset, action shape/units, termination conditions, and
   cleanup after failure.

A new evaluator uses the same registry/import pattern. Add a new runner only
when the existing component lifecycle cannot express the requirement. Model
differences belong in the FluxVLA configuration and server and should not result
in derived FluxThemis runners.

## Real-Robot Safety and Deployment Boundaries

`BaseRealRobotEnvironment` disables actuation by default and attempts to connect
only when `allow_actuation=True` is explicitly set. By default, it also requires
one-dimensional, finite `action_low` / `action_high` values with matching shapes.
If connection partially fails, the base class makes a best-effort call to
`disconnect()`. Action transmission and emergency stop are serialized through
the same control lock.

These are only a software safety floor. Each concrete robot adapter must still
implement:

- A hardware emergency stop independent of the Python/ROS process;
- A watchdog, stopping on communication loss, and stale-command rejection;
- Velocity, acceleration, force, torque, workspace, and joint limits;
- Explicit validation of control mode, coordinate frames, units, and action
  horizon;
- Reliable disconnection on setup, exception, and shutdown paths.

ROS 1 provides no application-level authentication or encryption. The current
transitional deployment should use an isolated network, a controlled ROS master,
and host firewalls. Migrating to ROS 2 does not automatically enable DDS
Security; deployments must still restrict `ROS_DOMAIN_ID` and the discovery
scope and configure authentication and encryption for the platform. Neither
transport should expose the inference service to an untrusted network.

## Current Coverage and Gaps

The current repository can verify runner/evaluator lifecycles, the Mock path,
the generic Gymnasium adapter, LIBERO rollout, the ROS 1/ROS 2 client protocols,
the dual-version service schema, and the real-robot safety base class. RoboCasa,
RoboTwin, and concrete real-robot adapters remain missing. The current ROS 1 path
supports FluxVLA before it is upgraded. After migrating the model environment,
the long-term real-robot baseline should switch both endpoints to Python 3.12 /
ROS 2 Jazzy. Each deployed endpoint must still validate the corresponding ROS
Python ABI, generated service type, real network, and hardware path.

Before declaring a platform operational, complete at least:

- End-to-end numerical testing of the FluxVLA server with the actual model,
  target dataset preprocessing, statistics, and `denormalize_action`;
- Alignment of ROS image encoding, view ordering, state layout, prompt, and
  action units;
- Testing of episode reset/seed, stateful policy history, and first-request
  semantics;
- Drills for server failure, ROS disconnection, timeout, SIGINT/SIGTERM, and the
  hardware emergency stop;
- Target-platform success metrics and a reproducible experimental baseline.

See the [README](../README.md) for message-package build and two-process launch
instructions.
