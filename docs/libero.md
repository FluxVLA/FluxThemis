# LIBERO Integration

## Ownership

FluxThemis owns the LIBERO process, initial-state selection, rendering, action
execution, and videos. FluxVLA owns image/state preprocessing, policy inference,
and action denormalization.

## Install

```bash
conda create -n fluxthemis-libero python=3.10 -y
cd /root/projects/FluxThemis
bash scripts/install_libero.sh
```

To add LIBERO-Plus to this `fluxthemis-libero` environment, run
`bash scripts/install_libero_plus.sh` after the base installation. It installs
the Plus additions and checks both benchmarks in separate processes. See the
[LIBERO-Plus installation options](libero_plus.md#install) for its assets.

## Local Run

For independent model/evaluator launch commands, replica counts, scheduling,
progress and GPU monitoring, see the [evaluation launcher guide](../guides/evaluation.md).
The standalone commands below remain available.

Start FluxVLA:

```bash
conda activate fluxvla
cd /root/projects/FluxVLA
CUDA_VISIBLE_DEVICES=0 \
bash scripts/distributed_zmq_server.sh \
  --config configs/pi05/pi05_paligemma_libero_10_full_finetune.py \
  --ckpt-path /path/to/model.safetensors
```

Start FluxThemis:

```bash
conda activate fluxthemis-libero
cd /root/projects/FluxThemis
CUDA_VISIBLE_DEVICES=1,2 NPROC_PER_NODE=2 \
bash scripts/distributed_eval.sh \
  --config /root/projects/FluxVLA/configs/pi05/pi05_paligemma_libero_10_full_finetune.py \
  --run-name libero-10
```

LIBERO task order, initial states, wait steps, episode count, execution horizon,
and base seed come from the FluxVLA config. The episode's environment case
index selects its LIBERO initial state independently of the policy and
environment seeds. Multiple simulator workers dynamically pull canonical
episodes; completion order does not change final result order.
`NPROC_PER_NODE` is explicit because the LIBERO environment may use a CPU-only
PyTorch build while MuJoCo renders on the selected GPUs.

For distributed simulator nodes, use the same shell launcher with the node
environment variables documented in the repository README.

PI05 LIBERO scores and protocol details are in the
[LIBERO evaluation report](pi05_libero_official.md).

## Standard Benchmark

| Suite | Tasks | Episodes per task | Maximum action steps |
| --- | ---: | ---: | ---: |
| libero_spatial | 10 | 50 | 220 |
| libero_object | 10 | 50 | 280 |
| libero_goal | 10 | 50 | 300 |
| libero_10 | 10 | 50 | 520 |

The four suites cover 40 tasks and 2,000 episodes, using initial-state indexes
0–49 per task. LIBERO-90 is a separate suite. Worker counts do not change these
quotas; the configuration can select a subset for smoke testing.

The FluxVLA FT configs predict and execute 10 actions, use environment seed 0,
10 settling steps and base seed 7. The default ZMQ runner resets model RNGs
per request with `sequence_noise=False`; the native runner samples continuously
after initializing each process. Record the code revision and resolved config
when comparing results across runners.

## PI05 FT Model Preparation

FluxVLA publishes one FT checkpoint per suite in
[FluxVLAEngine](https://huggingface.co/limxdynamics/FluxVLAEngine/tree/1b24eb97cbec6da523e2ea32c3600c467b18e090).
The example below pins revision `1b24eb97cbec6da523e2ea32c3600c467b18e090`.
Each checkpoint evaluates its matching suite; the official OpenPI `pi05_libero`
checkpoint instead covers all four suites with the same weights.

Existing local weights can keep their current model config and asset paths.
For a new download, run the following from FluxThemis with FluxVLA checked out
beside it. It downloads about 29 GB of model assets and prepares four config
copies under `work_dirs/pi05_ft_zmq_configs`:

```bash
PYTHONNOUSERSITE=1 conda run --no-capture-output -n fluxvla python - <<'PY'
from pathlib import Path
from huggingface_hub import snapshot_download
from mmengine import Config

source = Path("../FluxVLA").resolve()
root = Path("../checkpoints/pi05_ft").resolve()
output = Path("work_dirs/pi05_ft_zmq_configs").resolve()
output.mkdir(parents=True, exist_ok=True)
revision = "1b24eb97cbec6da523e2ea32c3600c467b18e090"
weights = {
    "libero_spatial": "step-019872-epoch-24-loss=0.0174.safetensors",
    "libero_object": "step-025128-epoch-24-loss=0.0165.safetensors",
    "libero_goal": "step-019848-epoch-24-loss=0.0145.safetensors",
    "libero_10": "step-038064-epoch-24-loss=0.0170.safetensors",
}
for suite, filename in weights.items():
    folder = f"pi05_paligemma_{suite}_full_finetune_bs64"
    files = [f"checkpoints/{filename}", "dataset_statistics.json",
             "tokenizer/*", "config.json", "config.yaml"]
    snapshot_download("limxdynamics/FluxVLAEngine", revision=revision,
                      local_dir=str(root), max_workers=4,
                      allow_patterns=[f"{folder}/{name}" for name in files])
    model_root = root / folder
    cfg = Config.fromfile(str(source / "configs/pi05" /
                              f"pi05_paligemma_{suite}_full_finetune.py"))
    for key in ("model", "inference_model"):
        if key in cfg:
            cfg[key].pretrained_name_or_path = None
    for transform in cfg.eval.dataset.transforms:
        if transform.get("type") == "LiberoPromptFromInputs":
            transform.tokenizer.model_path = str(model_root / "tokenizer")
    cfg.themis.server.ckpt_path = str(model_root / "checkpoints" / filename)
    cfg.themis.server.norm_stats_path = str(model_root / "dataset_statistics.json")
    cfg.eval.observation_video = dict(
        image_keys=["agentview_image", "robot0_eye_in_hand_image"],
        rotate_180=True, fps=2)
    cfg.dump(str(output / f"{suite}.py"))
print(f"Prepared configs: {output}")
PY
```

For a complete FT checkpoint, `pretrained_name_or_path=None` skips loading base
weights before the serving runner loads the FT checkpoint. Tokenizer and
normalization statistics must match the selected checkpoint.
The example also enables two-camera observation videos under the model result
directory's `observation_videos/`; these record inference inputs, not every step.

## Smoke and Full Evaluation

Start the model independently using its prepared suite config. In the model
Python environment, from FluxThemis:

```bash
bash scripts/distributed_model.sh --fluxvla-root /path/to/FluxVLA \
  --config work_dirs/pi05_ft_zmq_configs/libero_spatial.py \
  --gpu-ids 0,1 --workers-per-gpu 1 --scheduling dynamic
```

Then, in the prepared simulator environment (the installer defaults to
`fluxthemis-libero`), run the same evaluation launcher used for RoboDojo:

```bash
bash scripts/run_evaluation.sh \
  --config work_dirs/pi05_ft_zmq_configs/libero_spatial.py \
  --server-endpoint tcp://MODEL_HOST:15555 \
  --gpus 2 --workers-per-gpu 1 --smoke \
  --output /persistent/evaluations
```

This runs task 0 with initial states 0 and 1. Use `--full` instead of `--smoke`
for all 500 episodes in the suite. Repeat with the matching model service and
config for `libero_object`, `libero_goal` and `libero_10`. FT suites use different
checkpoints; the simulator launcher does not switch model weights.

Existing model deployments can keep their native server command and asset paths.
The simulator accepts only the evaluation settings from a model config; it does
not load its checkpoint, tokenizer or model dependencies. On the same host use
`MODEL_HOST=127.0.0.1`; an existing SSH tunnel can also use that address.

Add `--dry-run` to resolve the launch configuration without network, GPU or
simulator work. Environment preflight runs on actual startup. Progress, resource
metrics and output locations are described in the
[launcher guide](../guides/evaluation.md#进度和结果).
