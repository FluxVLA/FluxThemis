# RoboTwin Evaluation

Run the following commands from FluxThemis in the prepared model and simulator
environments, respectively. Prepare the model config and checkpoint in FluxVLA;
set `MODEL_HOST` to an address reachable from the simulator. For simulator
installation, see the [README](../README.md#installation).

## Model Service

```bash
bash scripts/distributed_model.sh --fluxvla-root /path/to/FluxVLA \
  --config /path/to/robotwin_model.py \
  --ckpt-path /path/to/model.safetensors \
  --gpu-ids 0 --workers-per-gpu 1 --scheduling dynamic
```

## Simulator

Activate the simulator environment (`fluxthemis-robotwin` by default), then run:

```bash
ROBOTWIN_ROOT=/path/to/RoboTwin \
bash scripts/run_evaluation.sh --config configs/robotwin/aloha.py \
  --server-endpoint tcp://MODEL_HOST:15555 \
  --gpu-ids 0 --workers-per-gpu 1 --scheduling dynamic \
  --full --output /path/to/results
```

The [default config](../configs/robotwin/aloha.py) selects both `clean` and
`random`: 50 tasks per setting and 100 episodes per task, totaling 10,000
episodes. To evaluate one setting, append
`--cfg-options themis.evaluation.environment.task_suite=clean` or use `random`.
`--full` evaluates all tasks within the selected setting(s).

Replace `--full` with `--smoke` for `adjust_bottle` with one episode per simulator
worker. Smoke uses the selected setting, or `clean` when `task_suite=all`.
Add `--dry-run` to resolve the configuration without starting simulation or
connecting to the model. See the [launcher guide](../guides/evaluation.md) for
shared scheduling, output and resume options.

### YAML Launch

The [launch YAML](../configs/robotwin/remote.yaml) references `aloha.py` and
keeps simulator paths, GPU/worker counts, model connection and output settings
in one file. Set `simulator.robotwin_root` and `model.host` for your environment,
then run:

```bash
bash scripts/run_evaluation.sh --config configs/robotwin/remote.yaml --smoke
```

Without `--smoke` or `--full`, the template runs one `clean/adjust_bottle`
episode. `--smoke` uses one episode per simulator worker; `--full` uses all
tasks and the quotas from `aloha.py`. Both modes override the YAML's `task`
and `episodes` fields. With `--full`, the default is 10,000 episodes; select
`clean` or `random` with `--cfg-options` as above.

Paths resolve relative to the YAML, including `~` and environment-variable
expansion. The template expects RoboTwin beside FluxThemis. An explicit
`simulator.robotwin_root` takes precedence over `ROBOTWIN_ROOT`; set it to
`null` to use the Python config, then `ROBOTWIN_ROOT`, then the sibling directory
as fallbacks. CLI launch options such as `--server-endpoint`, `--gpu-ids` and
`--output` override their YAML settings. Add `--dry-run` to inspect the plan.

## Compatibility Notes

- **OIDN and Blackwell GPUs:** SAPIEN 3.0.0b1 bundles OIDN 2.0.1, which can
  report `unsupported device type: CUDA` followed by `invalid handle` on
  Blackwell GPUs even while simulation continues. The installer replaces its
  main, core and CUDA libraries with the SHA256-verified official OIDN 2.3.3
  release and updates SAPIEN's loader together. [OIDN 2.3.3 adds Blackwell
  support](https://github.com/RenderKit/oidn/releases/tag/v2.3.3).
  The unified launcher executes a CUDA denoising filter through SAPIEN's loaded
  library on each selected GPU before connecting to the model. A failed check
  stops evaluation; details are saved in `oidn_gpu_<GPU>.log`. `--dry-run`
  remains free of GPU initialization. New recovery manifests record the OIDN
  loader and library contents and reject changes to those recorded files.
- **Camera synchronization:** The [installer](../scripts/install_robotwin.sh)
  applies the [camera patch](../scripts/patches/robotwin-camera-sync.patch),
  reading `Color` after each camera's `take_picture()` to wait for completion
  before submitting the next capture. This prevents overlapping render
  submissions without adding `get_obs()` calls or changing rendering parameters.
  The installer skips an already-applied patch and fails on source conflicts.
- **Task resource cleanup:** Consecutive episodes of the same task reuse its
  instance. Switching tasks releases the previous instance so scene and planner
  references retained by upstream `close_env()` do not accumulate across tasks.
  Reference cycles and unused GPU caches are collected before constructing the
  next task. Failed initialization discards the partial task before retrying.
- **Upstream dependency patches:** The installer applies the sapien UTF-8 and
  mplib screw-plan patches by default. The mplib patch affects expert solvability;
  keep its state identical when comparing against the native runner.
- **Source versions:** New checkouts use the installer defaults (cuRobo 0.7.8).
  Existing checkouts are reused without commit or source-hash comparisons.
  cuRobo is rebuilt from its actual source and its CUDA extension is imported
  after installation. Other revisions still need compatible APIs and patches.

CPU regression coverage for the adapter does not establish that a full GPU
evaluation passes or that scores match the native RoboTwin runner.

## Updating Existing Simulator Images

Python packages are listed in [requirements-robotwin.txt](../requirements-robotwin.txt).
OIDN is a native library bundled with SAPIEN, so a pip requirement cannot replace
its libraries and loader. The [OIDN tool](../scripts/robotwin/oidn.py) owns the
pinned version and archive checksum for both installation and validation.

Run the tool with the image's simulator Python, then rebuild and publish the
image. This also works in an existing conda environment:

```bash
python scripts/robotwin/oidn.py install
CUDA_VISIBLE_DEVICES=0 python scripts/robotwin/oidn.py check
```

For offline builds, pass `--archive /path/to/oidn-2.3.3.x86_64.linux.tar.gz`
to `install`, or set `OIDN_ARCHIVE` when running `install_robotwin.sh`.
Both paths require the same pinned SHA256. Installation itself does not need
a GPU. Restart existing simulator processes after installation so they load
the new libraries.

Pin formal jobs to the published image digest (`registry/image@sha256:...`)
and retain that reference with the run records. Updating a local environment
does not update an existing image or running job. The unified launcher checks
every selected GPU; when using `distributed_eval.sh` directly, run the CUDA
check above for each selected GPU before starting workers.
