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

## Local Run

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
