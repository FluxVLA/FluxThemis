# FluxVLA + FluxThemis Quickstart

## 1. Install

Create and install exactly one selected simulator environment.

For LIBERO:

```bash
cd /root/projects/FluxThemis
conda create -n fluxthemis-libero python=3.10 -y
bash scripts/install_libero.sh
```

For RoboCasa:

```bash
cd /root/projects/FluxThemis
conda create -n fluxthemis-robocasa python=3.10 -y
bash scripts/install_robocasa.sh
```

For RoboDojo:

```bash
cd /root/projects/FluxThemis
bash scripts/install_robodojo.sh
```

## 2. Start FluxVLA

```bash
conda activate fluxvla
cd /root/projects/FluxVLA
CUDA_VISIBLE_DEVICES=0,1 \
bash scripts/distributed_zmq_server.sh \
  --config configs/pi05/pi05_paligemma_robodojo_full_finetune.py \
  --ckpt-path /path/to/checkpoints/model.safetensors
```

The command starts one stateless model Worker per torchrun rank. Rank 0 also
starts the Coordinator.

## 3. Start FluxThemis

```bash
conda activate RoboDojo
cd /root/projects/FluxThemis
CUDA_VISIBLE_DEVICES=2,3 \
bash scripts/distributed_eval.sh \
  --config /root/projects/FluxVLA/configs/pi05/pi05_paligemma_robodojo_full_finetune.py \
  --run-name smoke \
  --save-videos \
  --video-dir /persistent/eval/smoke-videos \
  --cfg-options \
    themis.evaluation.environment.task_name=put_bottles_into_dustbin \
    themis.evaluation.episodes_per_task=2
```

The client may be launched first. It waits for server readiness instead of
failing because of startup order.

Each launcher uses the `torchrun` from its active Conda environment and starts
one worker per visible GPU. The default rendezvous ports are `29500` for
FluxVLA and `29501` for FluxThemis.

## 4. Locate Results

FluxVLA prints the authoritative run directory after `run_start` and again at
`run_end`. The default root is beside the checkpoint. Videos are written to the
evaluation-level `video_dir`.

## 5. Multi-Node

Use the same two shell launchers. See ``Multi-Node Evaluation`` in the
repository README for the required node environment variables and discovery
options.
