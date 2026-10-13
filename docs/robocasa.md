# RoboCasa Integration

## Ownership

RoboCasa and MuJoCo run only in FluxThemis simulator workers. FluxVLA model
workers receive raw observations over ZMQ and return denormalized 29D actions.
Model and simulator worker counts are independent.

## Install

Create a dedicated environment because RoboCasa and LIBERO pin different
simulator dependencies:

```bash
conda create -n fluxthemis-robocasa python=3.10 -y
cd /root/projects/FluxThemis
bash scripts/install_robocasa.sh
```

The installer pins the validated RoboCasa and robosuite revisions, installs
FluxThemis with its ZMQ runtime, configures headless EGL, and downloads the
required assets independently. Run `bash scripts/install_robocasa.sh --help`
to reuse an existing asset directory or choose a persistent download path.

## Smoke Test

Start one FluxVLA model worker:

```bash
conda activate fluxvla
cd /root/projects/FluxVLA
CUDA_VISIBLE_DEVICES=0 \
bash scripts/distributed_zmq_server.sh \
  --config configs/pi05/pi05_paligemma_robocasa_full_data_full_finetune.py \
  --ckpt-path /path/to/model.safetensors
```

Start two simulator workers and run one episode of task `0`:

```bash
conda activate fluxthemis-robocasa
cd /root/projects/FluxThemis
CUDA_VISIBLE_DEVICES=1,2 NPROC_PER_NODE=2 \
bash scripts/distributed_eval.sh \
  --config /root/projects/FluxVLA/configs/pi05/pi05_paligemma_robocasa_full_data_full_finetune.py \
  --run-name robocasa-smoke \
  --cfg-options \
    'themis.evaluation.task_ids=[0]' \
    themis.evaluation.episodes_per_task=1
```

`NPROC_PER_NODE` is explicit because the RoboCasa environment may use a
CPU-only PyTorch build while MuJoCo renders on the selected GPUs.

## Full Benchmark

Keep the FluxVLA server running and remove the smoke-test overrides:

```bash
CUDA_VISIBLE_DEVICES=1,2 NPROC_PER_NODE=2 \
bash scripts/distributed_eval.sh \
  --config /root/projects/FluxVLA/configs/pi05/pi05_paligemma_robocasa_full_data_full_finetune.py \
  --run-name robocasa-full
```

Each simulator worker dynamically pulls episodes from the shared schedule.
FluxVLA writes the authoritative journal, task results, summaries, and optional
Feishu report beside the checkpoint. A failed worker is restarted and its
episode retried.

For multiple simulator nodes, use the same shell launcher with the node
environment variables and server discovery options documented in the
repository README.
