# RoboDojo Integration

FluxThemis runs RoboDojo simulation, schedules episodes and sends observations
through ZMQ to a separately started FluxVLA model service. Keep simulator assets
on the Sim host and model weights, tokenizer and statistics on the Predict host.

## Installation

Simulation requires an RTX/RT-capable GPU and RoboDojo's Isaac Sim environment.
From the FluxThemis checkout:

```bash
export ROBODOJO_ROOT=/path/to/RoboDojo
bash scripts/install_robodojo.sh
conda activate RoboDojo
```

The [installer](../scripts/install_robodojo.sh) selects the RoboDojo source,
installs its environment and assets, and adds FluxThemis. Use `--skip-assets`
when assets are already available. It includes upstream camera synchronization
and applies the [reference PhysX reset fix](https://github.com/roboterax/video-prediction-policy-2/blob/3365872910244f14df83003062f7c563f14c6d9e/patches/robodojo_spring_button_reset.patch).

For an existing or restored environment, use a source tree with those camera
interfaces and prepare it before starting simulator workers:

```bash
python scripts/preflight/robodojo.py --root "$ROBODOJO_ROOT"
python scripts/preflight/robodojo.py --root "$ROBODOJO_ROOT" --check
```

Preparation accepts an already applied fix and archives without `.git`.
Actual evaluation startup also checks the environment and required assets.

## Assets and Evaluation Quotas

Closed-loop evaluation needs `Assets/`, including `Eval_Layout`, approximately
**41 GB** at the snapshot below. Training demonstrations and unrelated model
checkpoints are unnecessary. To download assets separately:

```bash
hf download RoboDojo-Benchmark/RoboDojo \
  --repo-type dataset \
  --revision 35efbc7dedfdbeeb6e95fb749bd885d73d483e41 \
  --include 'Assets/**' \
  --local-dir "$ROBODOJO_ROOT"
```

After moving the source or assets, regenerate robot paths:

```bash
(cd "$ROBODOJO_ROOT" && python utils/update_embodiment_config_path.py)
```

The [default config](../configs/robodojo/arx_x5.py) covers **42 base tasks,
54 executable variants and 2,100 episodes** per full run. Each generalization
task and its `_random` variant gets 25 episodes; other tasks get 50.
See the config for quotas and the [environment task groups](../fluxthemis/environments/sim/robodojo.py)
for case-sensitive task IDs. Unstable layouts are replaced with spare layouts,
so available layouts may need to exceed the accepted episode quota.
This also applies when RoboDojo explicitly invalidates a sample during
execution, such as an unsuccessful scripted opponent discard in `make_kong`.
Only that slot is retried with a spare layout; healthy batch slots continue,
and the invalid sample counts as neither a success nor a failure. The journal
records the layout replacement while keeping the episode ID and seeds.
Simulator process crashes retain their bounded retry budget and retry affected
episodes individually with their recorded layouts.

The default config uses three cameras, 14D joint actions, `base_seed=0`, at most
2,000 action steps per episode and up to 50 actions per prediction. Policy seeds
and task-local layout indexes are separate. VPP2 uses the
[EE16/history config](../configs/robodojo/vpp2.py), which inherits these tasks and
quotas and overrides the action interface and replanning settings.

<a id="five-episode-smoke-test"></a>

## Launch

Start the model service using the [README commands](../README.md#zmq-evaluation-launcher)
or the [launcher guide](../guides/evaluation.md). Use matching model config,
checkpoint, tokenizer and normalization statistics. The default simulator config
matches FluxVLA's `pi05_paligemma_robodojo_full_finetune.py`; its released bundle
is [available here](https://huggingface.co/limxdynamics/FluxVLAEngine/tree/main/pi05_paligemma_robodojo_full_finetune_bs256).

### Smoke and Full Evaluation

In the prepared simulator environment, run from the FluxThemis checkout:

```bash
bash scripts/run_evaluation.sh \
  --config configs/robodojo/arx_x5.py \
  --server-endpoint tcp://MODEL_HOST:15555 \
  --gpu-ids 0 --workers-per-gpu 1 --smoke --save-videos \
  --output ./work_dirs/robodojo
```

`--smoke` runs `put_bottles_into_dustbin`, one episode per simulator worker.
Replace it with `--full` to use all tasks and their configured quotas. Start a
fresh model service between runs: FluxVLA exits after an accepted `run_end`.
Add `--dry-run` to inspect the resolved config without starting workloads;
installed dependencies, assets and connectivity are checked during a real run.

Use the model address reachable from the simulator. For a same-host service or
an existing local SSH forward, use `tcp://127.0.0.1:15555`; this is the ZMQ port,
not the SSH login port. Choose separate GPU IDs when both roles share a host.
Dynamic scheduling allows different Sim and Predict worker counts; fixed
scheduling requires equal counts. See [network and scheduling details](../guides/evaluation.md).

This launcher supports one simulator host with multiple GPUs. For multiple
simulator nodes, use `distributed_eval.sh` as described in the
[README](../README.md#multi-node-evaluation).

### VPP2

Start the model service following the
[FluxVLA guide](https://github.com/FluxVLA/FluxVLA/blob/main/docs/vpp2.md), then run:

```bash
ROBODOJO_EVAL_SEED=0 bash scripts/run_evaluation.sh \
  --config configs/robodojo/vpp2.py \
  --server-endpoint tcp://MODEL_HOST:15555 \
  --gpu-ids 0 --workers-per-gpu 1 --full --save-videos \
  --output ./work_dirs/vpp2/seed0
```

Repeat with environment seeds 1 and 2, separate output directories and a fresh
model service for each run. Without `ROBODOJO_EVAL_SEED`, the default is 1.
The config uses absolute EE16 actions, an episode anchor and eight camera frames
spaced 25 steps apart, executing 24 actions before replanning. No FluxVLA config
or Python installation is needed on the simulator host.

For the two-task baseline, omit `--full` and add
`--cfg-options themis.evaluation.environment.task_name=insert_tubes,put_bottles_into_dustbin`.
Each task runs 50 episodes. Progress, results and resume use the common flow below.

### YAML Launch

To save paths, endpoint and worker counts, edit the commented
[launch template](../configs/robodojo/remote.yaml):

```bash
bash scripts/run_evaluation.sh --config configs/robodojo/remote.yaml --smoke
```

Paths are relative to the YAML file and support `~` and `${ENV_VAR}`.
`--smoke` and `--full` override task selection and quotas; without either flag,
`simulator.task` and `simulator.episodes` apply. For other options, use
`bash scripts/run_evaluation.sh --help` and the [launcher guide](../guides/evaluation.md).

## Results and Resume

The launcher prints `Results: ...` for a new directory under `--output` or YAML
`run.output_root`. Keep it on persistent storage. Useful files are:

- `progress.json/log`: completed episode count and ETA, including failures.
- `simulator.log` and `events.jsonl`: diagnostics and acknowledged episode events.
- `result.json`: completeness, success count and the model's result location.
- `performance_report.json/txt`: resource and throughput measurements.

The model service writes the authoritative benchmark summary. Raw pooled success
counts need not equal a benchmark's official aggregation. See
[artifact details](../README.md#artifacts) for output paths and video recording.

### Resume

Stop the previous run, then add `--resume /path/to/stopped-run` to the original
launch command, or set YAML `resume.from_dir`. Keep the following requirements:

- Preserve task selection, quotas and evaluation settings. Keep the original
  `--full` or `--smoke` flag. If changing the worker count of a smoke run, set its
  original task/quota explicitly in YAML and omit `--smoke`.
- Supply `model_identity.json` through YAML `model.result_dir` on both the initial
  and resumed runs. Native FluxVLA status does not supply this fingerprint;
  provide it externally before the first run if recovery will be needed.
  Ordinary evaluation does not require it; GPU telemetry is also optional.
- Keep recorded dependencies, assets and model identity compatible. Source,
  asset, Python and output paths may change in the current YAML; the launcher
  checks the frozen evaluation settings. Repository commit IDs do not gate resume.

Completed episodes retain their outcomes, seeds and layouts and are replayed
without inference. Pending episodes avoid layouts already used by completed
ones. Results go into a new directory; the previous run remains intact.
Do not run simultaneous evaluations against the same output directory.

## Troubleshooting

| Symptom | Action |
| --- | --- |
| Missing assets or `X5A.urdf is not a file` | Check `$ROBODOJO_ROOT/Assets` and regenerate robot paths after moving files. |
| Missing camera interfaces or reset fix | Use the installer-selected source and run the preparation commands above. |
| Model connection fails | Check the address from inside the Sim container, the ZMQ port and any existing tunnel. |
| Worker startup times out | Inspect `simulator.log` with one worker first; adjust YAML `model.wait_seconds` if startup needs longer. Direct `distributed_eval.sh` uses `--startup-timeout-s`. |
| `Unsupported IsaacLab signal registration` or `Unsupported RoboDojo PhysX monitor` | Check the installed dependency versions against the supported installer setup. Preserve Kit logs and crash dumps for diagnosis. |

The supervisor terminates and replaces workers that exceed the configured startup
timeout. For fatal simulation failures, it restarts the affected worker and
retries its episode; the direct launcher exposes `--simulator-restart-limit` to
bound coordinator retries. Repeated failures require checking the simulator logs.
