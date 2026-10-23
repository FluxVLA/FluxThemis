# RoboDojo Integration

RoboDojo and Isaac Sim run only in FluxThemis simulator workers. Themis selects
layouts, schedules episodes and advances the simulation; FluxVLA model workers
receive observations over ZMQ and return denormalized joint actions. Assets and
Eval_Layout belong on the simulator side; weights, tokenizer and normalization
statistics belong on the model side. Model and simulator worker counts are independent.

- [Environment and model preparation](#environment-and-model-preparation)
- [Evaluation data and quotas](#evaluation-data-and-quotas)
- [Launch and network](#launch-and-network)
- [Progress, results and resume](#progress-results-and-resume)
- [Compatibility and troubleshooting](#compatibility-and-troubleshooting)
- [Full task list](#full-task-list)

## Environment and Model Preparation

Cloud model and simulator tasks each use the Python environment from their own
image. Simulation requires an RTX/RT-capable GPU; A100/A800 can run the model.
Code on shared storage can be installed into a task-private directory, while
runtime dependencies use that task's own environment.

### Install

```bash
cd /root/projects/FluxThemis
bash scripts/install_robodojo.sh
```

The installer uses RoboDojo's official environment and asset scripts, then
installs FluxThemis with its ZMQ dependencies into the `RoboDojo` environment.

### Required Configuration

- Simulator-only deployments can use `configs/robodojo/arx_x5.py` in FluxThemis. It contains
  only simulator, evaluation and observation/action interface settings; no
  local FluxVLA model checkout is required. The default benchmark matches
  FluxVLA's RoboDojo FT evaluation: 54 task variants, 2,100 episodes, seed 0,
  at most 2,000 steps per episode and 50 actions per prediction. Existing
  deployments can still pass their FluxVLA RoboDojo configuration to
  `distributed_eval.sh`; the separate simulator config is optional.
- The simulator needs the RoboDojo revision selected by
  `scripts/install_robodojo.sh`, its Isaac Sim/IsaacLab dependencies, and the
  official [Assets](https://huggingface.co/datasets/RoboDojo-Benchmark/RoboDojo/tree/main/Assets).
  Set `ROBODOJO_ROOT` (or `themis.evaluation.environment.robodojo_root`) to the simulator checkout;
  place Assets under that root and regenerate robot paths with
  `utils/update_embodiment_config_path.py` after relocating them. Training
  demonstrations are not needed for evaluation.
- For pi0.5, use the matching
  [RoboDojo FT checkpoint, tokenizer and statistics](https://huggingface.co/limxdynamics/FluxVLAEngine/tree/main/pi05_paligemma_robodojo_full_finetune_bs256)
  with `pi05_paligemma_robodojo_full_finetune.py`. This configuration uses three
  cameras and 14D ARX-X5 joint actions. Set `--ckpt-path`,
  `themis.server.norm_stats_path`, and the `ProcessPrompts` tokenizer path in
  `eval.dataset.transforms` to the downloaded files. When loading a complete
  FT checkpoint without the base checkpoint, set both
  `model.pretrained_name_or_path=None` and
  `inference_model.pretrained_name_or_path=None` via `--cfg-options`.
- Both environments require `pyzmq` and `msgpack`. The simulator connects to
  the model service's frontend using `--server-endpoint tcp://MODEL_HOST:15555`.
  `MODEL_HOST` must be reachable from the simulator process/container through
  the private network or an existing tunnel. For a same-host tunnel use
  `127.0.0.1`; the endpoint option does not establish a tunnel.
- The Themis launchers let each side select its own `--gpu-ids` and
  `--workers-per-gpu` (default 1).
  Use `--scheduling dynamic` on both sides to allow different worker counts.
  When model inference shares a memory-limited GPU with Isaac Sim, the model
  option `--cfg-options themis.server.runner.autocast_cache_enabled=False`
  saves cast-weight cache memory while retaining FP32 parameters and the
  configured autocast dtype; repeated casts can reduce throughput.

## Evaluation Data and Quotas

### Assets and Storage

Source: [RoboDojo-Benchmark/RoboDojo](https://huggingface.co/datasets/RoboDojo-Benchmark/RoboDojo).
The following totals were recorded on 2026-10-16 from Hugging Face file metadata
at revision
[`35efbc7dedfdbeeb6e95fb749bd885d73d483e41`](https://huggingface.co/datasets/RoboDojo-Benchmark/RoboDojo/tree/35efbc7dedfdbeeb6e95fb749bd885d73d483e41).
They sum logical file sizes, excluding download caches, container images, runtime
caches, evaluation videos and results. Later revisions may differ.

| Directory | Files | Exact bytes | Decimal size | Use |
| --- | ---: | ---: | ---: | --- |
| `Assets/` | 15,365 | 41,269,513,111 | 41.27 GB | Simulator scenes, objects, materials, robots and evaluation layouts |
| `data/` | 53,710 | 6,487,684,022,944 | 6.49 TB | Training demonstrations; not needed for online evaluation |
| `ckpt/` | 2,403 | 1,082,114,830,735 | 1.08 TB | Model collection; download only the model being evaluated |

`--include 'Assets/**'` selects **41.27 GB = 38.44 GiB**, where GB = 10^9 bytes
and GiB = 2^30 bytes. Allow additional space for temporary files, Xet/Isaac caches
and videos. Download only the assets with:

```bash
hf download RoboDojo-Benchmark/RoboDojo \
  --repo-type dataset \
  --revision 35efbc7dedfdbeeb6e95fb749bd885d73d483e41 \
  --include 'Assets/**' \
  --local-dir /path/to/assets-full
```

The resulting asset root is `/path/to/assets-full/Assets`. Set download proxy
and concurrency options for the deployment environment.

### Episode Quotas

The [Themis config](../configs/robodojo/arx_x5.py) and
[task groups](../fluxthemis/environments/sim/robodojo.py) match the evaluation
settings in FluxVLA's `configs/pi05/pi05_paligemma_robodojo_full_finetune.py`.
There are **42 base tasks**, including 12 generalization tasks with an additional
`_random` variant each: **54 executable task variants and 2,100 episodes**.
An episode is one complete attempt starting from a selected layout. These are
evaluation quotas, not training demonstration or asset counts.

| Category | Base tasks | Tasks including variants | Episodes per task | Total episodes |
| --- | ---: | ---: | ---: | ---: |
| Generalization | 12 | 24 | 25 | 600 |
| Memory | 6 | 6 | 50 | 300 |
| Precision | 8 | 8 | 50 | 400 |
| Long-horizon | 8 | 8 | 50 | 400 |
| Open | 8 | 8 | 50 | 400 |
| **Total** | **42** | **54** | — | **2,100** |

Defaults are `base_seed=0`, at most 2,000 action steps per episode and up to 50
executed actions before requesting another prediction. The base seed starts a
seed sequence; it does not make every episode's policy seed zero. Layout indexes
advance separately for each task. Rejected unstable layouts can require more
available layouts than the accepted episode quota.

These quotas apply to `task_name=all` or the YAML launcher's `--full` option,
regardless of worker count. Subset runs follow their resolved configuration and
runtime task manifest. See [YAML launch](#yaml-launch) for smoke settings and the
[full task list](#full-task-list) for individual IDs and quotas.

## Launch and Network

Start the model independently, then start the simulator using either the native
commands below or the [YAML launcher](#yaml-launch). The model endpoint is
independent of the simulator task's `MASTER_ADDR`; use an address reachable from
the simulator as described under [Network](#network).

### Five-Episode Smoke Test

Start FluxVLA using its existing server entry point:

```bash
conda activate fluxvla
cd /root/projects/FluxVLA
CUDA_VISIBLE_DEVICES=0 \
bash scripts/distributed_zmq_server.sh \
  --config configs/pi05/pi05_paligemma_robodojo_full_finetune.py \
  --ckpt-path /path/to/model.safetensors
```

Keep existing model config and checkpoint paths when they are already prepared.
A separate config copy is optional; asset paths must resolve in the model environment.

For Themis's optional GPU/worker and scheduling wrapper, see the
[launcher guide](../guides/evaluation.md#两端独立启动). It delegates model loading
and serving to FluxVLA's `distributed_zmq_server.py` and uses the same model
configs and checkpoints. The wrapper requires that entry point to support
`--workers-per-gpu` and replica-to-device mapping; older FluxVLA checkouts can
continue using the native command above.

Start FluxThemis:

```bash
conda activate RoboDojo
cd /root/projects/FluxThemis
CUDA_VISIBLE_DEVICES=1,2,3,4,5 \
bash scripts/distributed_eval.sh \
  --config configs/robodojo/arx_x5.py \
  --server-endpoint "tcp://${MODEL_HOST:-127.0.0.1}:15555" \
  --workers-per-gpu 1 --scheduling dynamic \
  --run-name put-bottles-5ep \
  --save-videos \
  --video-dir /persistent/eval/put-bottles-5ep \
  --cfg-options \
    themis.evaluation.environment.task_name=put_bottles_into_dustbin \
    themis.evaluation.episodes_per_task=5
```

### Full Benchmark

The checked-in RoboDojo configs contain official per-task episode overrides.
Run without task or episode overrides:

```bash
CUDA_VISIBLE_DEVICES=1,2,3,4,5 \
bash scripts/distributed_eval.sh \
  --config configs/robodojo/arx_x5.py \
  --server-endpoint "tcp://${MODEL_HOST:-127.0.0.1}:15555" \
  --workers-per-gpu 1 --scheduling dynamic \
  --run-name pi05-full
```

FluxVLA writes the official progress-score and success-rate overview at
`run_end`. FluxThemis keeps policy seeds fixed while assigning task-local
environment case indexes. For RoboDojo, a case index selects one layout. An
unstable layout is rejected before `episode_start`; the coordinator retries
that episode slot with the next case without counting the rejected attempt.
RoboDojo's PhysX monitor terminates a disposable Simulator process after a
fatal CUDA/PhysX failure. The local Supervisor starts a fresh worker, which
reconnects directly to the coordinator under the same logical identity. The
coordinator retries the exact episode without stopping other simulators. Use
`--simulator-restart-limit` to bound retries for one episode or failed startup
sequence.

For multiple simulator nodes, use the same shell launcher with the node
environment variables documented in the repository README.

### YAML Launch

The same `run_evaluation` entry handles LIBERO and RoboDojo, using a launch YAML
or a Themis Python config. It starts only simulator processes and connects to the
configured model service; it never starts or replaces the model.

Task submission, dependency installation and SSH tunnels are managed by external
deployment tools. After moving RoboDojo or Assets, regenerate robot path configs
in a task-private asset view. The launcher checks these paths and records the
environment identity for resume.

From FluxThemis in the prepared simulator environment:

```bash
python scripts/run_evaluation.py --config configs/robodojo/remote.yaml --smoke --dry-run
python scripts/run_evaluation.py --config configs/robodojo/remote.yaml --smoke
python scripts/run_evaluation.py --config configs/robodojo/remote.yaml --full
```

`--smoke` selects `put_bottles_into_dustbin` with one episode per simulator
process; `--full` selects `all` with the Python config's per-task quotas. The
options are mutually exclusive and preserve YAML GPU/worker counts, paths and
connection settings. Without either option, YAML `task`/`episodes` apply. The
supplied template defaults to one GPU, one simulator and one episode; it is also
used when `--config` is omitted.

This launcher manages one simulator host with multiple GPUs/workers. For multiple
nodes use `scripts/distributed_eval.sh`. Set GPU/worker counts for the actual
resources; defaults are not capacity recommendations.

| YAML field | Meaning |
| --- | --- |
| `simulator.python` | Simulator Python; defaults to the current interpreter |
| `simulator.robodojo_root` / `assets` | Prepared source and assets; Assets defaults to the source root's Assets directory |
| `simulator.config` | Themis evaluation config; the template uses `arx_x5.py`, with no local VLA checkout required |
| `simulator.gpu_ids` / `workers_per_gpu` | Local GPUs and simulators per GPU; independent of model replica counts |
| `simulator.task` / `episodes` | Tasks and quotas when neither `--smoke` nor `--full` is supplied |
| `simulator.cache` | Optional simulator runtime cache directory |
| `simulator.save_videos` | Save simulator videos; enabled in the RoboDojo template |
| `simulator.scheduling` | Optional dynamic/fixed override; default follows the evaluation config |
| `simulator.coordinator_port` | Simulator coordinator port; the next port is reserved for torchrun |
| `model.host` / `port` | Model frontend reachable from simulator processes |
| `model.expected_workers` | Optional model replica count check |
| `model.result_dir` | Optional model input directory: `model_identity.json` for resume; `model_metrics.jsonl` for GPU telemetry |
| `model.wait_seconds` | Time budget for model and simulator startup |
| `run.output_root` / `max_minutes` | Persistent output root and runtime limit; 0 means unlimited |
| `resume.from_dir` | Stopped previous run directory; empty starts a new evaluation |

Paths are relative to the YAML file and support `~` and `${ENV_VAR}`. Scheduling
follows the Python evaluation config or an explicit YAML/CLI override;
keep the model in the same mode. Dynamic allows different model/simulator counts;
fixed requires equal counts. See the [launcher guide](../guides/evaluation.md).

### Network

On the same host, use `127.0.0.1:15555`. On a reachable private network, use the
model host's private IP and port 15555. Across clusters, first establish an SSH
tunnel on the **simulator host**:

```bash
ssh -N -T -o ExitOnForwardFailure=yes -o ServerAliveInterval=15 \
  -p SSH_PORT -i /path/to/client_key \
  -L 127.0.0.1:15555:127.0.0.1:15555 USER@MODEL_SSH_HOST
```

Then set YAML `model.host: 127.0.0.1` and `model.port: 15555`. The public SSH port
is the login endpoint; 15555 is the ZMQ model port. The connecting host holds the
private key and authorizes its public key on the model host. An external task
script can maintain the tunnel. The launcher queries model status, shows ready
replica counts and starts simulation when the service is ready.

For local Docker, start the container separately, mount code, Assets and outputs,
and run the same Python launcher inside it. The model address must be reachable
from that container's network namespace.

`--dry-run` resolves settings and writes a plan/config snapshot without querying
resources, checking installed simulator packages, hashing assets or connecting.
Those environment checks run on actual startup.

## Progress, Results and Resume

The YAML launcher displays startup stages, then one updating line with an
episode progress bar, completed/total count and ETA. Redirected logs record
progress once per minute, plus status changes and completion. ETA uses the
current run's observed execution speed; task duration differences, restarts
and scene changes can affect it. Resource measurements and throughput remain
in the report files.

Each run creates a separate output directory with the plan and config snapshots,
`progress.log/json`, `simulator.log`, acknowledged `events.jsonl`, `result.json`,
videos, raw resource samples and `performance_report.json/txt`. The model side
saves the official benchmark summary. The launcher prints both output locations;
external controllers can use `--result-pointer FILE` to obtain the current run's
output directory.

Reports distinguish whole-GPU baseline/peak increments, simulator process-tree
CPU/PSS and the actual episode execution window. Summed RSS double-counts shared
pages; whole-GPU/host increments can include other processes. Missing samples
remain unavailable. Inference wait includes model service time;
`network_and_queue_s` includes network, serialization and queuing time.
`model.result_dir` can be shared storage or externally collected SSH telemetry.
Without it, reports do not infer model GPU bottlenecks. Telemetry needs synchronized
clocks; SSH-forwarded samples prefer reception timestamps.

### Resume

Stop the old run before setting `resume.from_dir` in YAML or passing
`--resume /path/to/old-run`. Keep the original `--smoke`/`--full` option or set the
same task and episode quotas explicitly in YAML. Since `--smoke` derives its quota
from the simulator count, changing that count requires preserving the original
quota in YAML and omitting `--smoke`. Source, Assets, Python, cache and output paths
come from the current YAML, allowing migration to a new container.

Resume uses acknowledged events and the frozen config, checking recorded runtime
dependencies, asset contents and the model fingerprint. Repository commits and
source snapshots do not gate resume. Older manifests are checked against the
runtime fields they recorded; opaque legacy source fingerprints are ignored.
Provide `model_identity.json` via `model.result_dir` from the first run and
again for the current model when
resuming. Live GPU metrics are not required for identity validation.
The frozen config expands inherited settings; the new run reuses it with current
RoboDojo paths and rejects changes to the evaluation protocol. Native FluxVLA status
currently reports readiness/queue state, not a model fingerprint or GPU samples.
Ordinary evaluation works without these files; cross-run fingerprint verification
and remote GPU statistics still use explicitly supplied external artifacts.

- Completed episodes retain task/index, both seeds and the actual layout. Their
  reports are replayed without inference.
- Pending episodes are rebuilt from the canonical manifest, avoiding layouts
  already used by completed episodes. Unrecorded rejected layouts may be checked
  again. Worker restart budgets reset for the new run.
- A resumed run can be resumed again without double counting. Accumulated time
  includes only execution intervals observable in the journal, excluding downtime
  and replay. Unrecorded time between the last event and interruption is unavailable.
- Results go into a new directory under the current `run.output_root`; old records
  remain intact. Do not start simultaneous evaluations on the same directory. The
  launcher locks directories and checks whether the source run is still active.

## Compatibility and Troubleshooting

The [PhysX log adapter](../fluxthemis/environments/sim/robodojo_log_capture.py)
writes native stdout/stderr to ordinary files, avoiding a full pipe blocking a
writer while the Python reader waits for the GIL. Existing error parsing and the
watchdog remain, and logs are forwarded to `simulator.log`. Each worker uses
separate Kit data, log and extension-cache directories to avoid concurrent writes
to shared caches.

### IsaacLab Signal Patch and Interface Checks

Older IsaacLab `AppLauncher` versions register Python callbacks for
`SIGTERM/SIGABRT/SIGSEGV` that call `SimulationApp.close()`. Reentrant or stalled
shutdown can prevent the supervisor from reclaiming the worker. See
[IsaacLab #7774](https://github.com/isaac-sim/IsaacLab/issues/7774) and
[PR #8068](https://github.com/isaac-sim/IsaacLab/pull/8068).

The `_launch_worker_app` / `_WorkerAppSignals` patch in
[robodojo.py](../fluxthemis/environments/sim/robodojo.py) temporarily proxies the
launcher's `signal` reference during construction, skips registration of the old
callback for those three signals, then restores the reference. Other registrations
pass through, preserving Kit's native crash handling and normal shutdown. Setting
all fatal signals to `SIG_DFL` after startup would override native handling.

These private interfaces vary across IsaacLab/RoboDojo versions. Even a deployment
with a fixed image must retain the following checks against the loaded implementation:

| Check | Purpose and limit |
| --- | --- |
| `_abort_signal_handle_callback` exists | Otherwise call the original launcher; absence does not prove an upstream fix |
| `module.signal is signal` | Verify the proxied module is the expected one; fail clearly on mismatch |
| `handler.__func__` matches the target callback | Filter only the old callback registration and preserve other handlers |
| Monitor attributes required by `FileLogCapture` | Check the RoboDojo PhysX parsing/watchdog structure |

Structural checks do not replace environment validation. For
`Unsupported IsaacLab signal registration` or `Unsupported RoboDojo PhysX monitor`,
inspect the actual source and dependency versions. Remove the signal patch only
after all supported environments use a verified upstream fix; assess the log
adapter separately.

The signal patch addresses exit after failure. It does not guarantee SIGTERM
behavior in every environment or eliminate native crashes and all deadlocks.
The supervisor still lacks independent startup/shutdown timeouts with forced
reclamation. If initialization stalls, use one GPU/worker to isolate AppLauncher,
library imports, cuRobo, reset or shutdown. Before stopping a failed container,
preserve `.dmp`, `.dmp.toml`, `.py.txt` and Kit logs from its temporary directories.

## Full Task List


Each generalization task and its `_random` variant have 25 episodes each,
50 per pair. All other tasks have 50 episodes. Task IDs below are case-sensitive
and can be passed directly to the launcher.

### Generalization

| Task ID | Episodes |
| --- | ---: |
| `stack_bowls` | 25 |
| `push_T` | 25 |
| `pack_objects_into_box` | 25 |
| `fold_clothes` | 25 |
| `hang_mugs` | 25 |
| `sweep_blocks` | 25 |
| `pour_liquid_into_cup` | 25 |
| `make_toast` | 25 |
| `arrange_largest_number` | 25 |
| `sort_nesting_dolls_by_size` | 25 |
| `store_laptop_and_headphones` | 25 |
| `stack_blocks` | 25 |
| `stack_bowls_random` | 25 |
| `push_T_random` | 25 |
| `pack_objects_into_box_random` | 25 |
| `fold_clothes_random` | 25 |
| `hang_mugs_random` | 25 |
| `sweep_blocks_random` | 25 |
| `pour_liquid_into_cup_random` | 25 |
| `make_toast_random` | 25 |
| `arrange_largest_number_random` | 25 |
| `sort_nesting_dolls_by_size_random` | 25 |
| `store_laptop_and_headphones_random` | 25 |
| `stack_blocks_random` | 25 |

### Memory

| Task ID | Episodes |
| --- | ---: |
| `cover_blocks` | 50 |
| `match_and_pick_from_conveyor` | 50 |
| `swap_blocks` | 50 |
| `swap_T` | 50 |
| `press_by_number` | 50 |
| `imitate_sorting_sequence` | 50 |

### Precision

| Task ID | Episodes |
| --- | ---: |
| `fasten_screws` | 50 |
| `plug_in_charger` | 50 |
| `insert_tubes` | 50 |
| `pour_balls_into_vase` | 50 |
| `play_Xylophone` | 50 |
| `deposit_coin` | 50 |
| `insert_key` | 50 |
| `build_tower` | 50 |

### Long-horizon

| Task ID | Episodes |
| --- | ---: |
| `fill_pen_holder` | 50 |
| `classify_objects` | 50 |
| `put_bottles_into_dustbin` | 50 |
| `play_tic_tac_toe` | 50 |
| `fill_egg_holder` | 50 |
| `organize_table` | 50 |
| `make_kong` | 50 |
| `play_stacking_toy` | 50 |

### Open

| Task ID | Episodes |
| --- | ---: |
| `align_blocks` | 50 |
| `general_pickup` | 50 |
| `solve_equation` | 50 |
| `stack_blocks_by_language` | 50 |
| `classify_objects_by_language` | 50 |
| `pick_from_conveyor_by_image` | 50 |
| `store_tools_in_toolbox` | 50 |
| `pour_by_language` | 50 |
