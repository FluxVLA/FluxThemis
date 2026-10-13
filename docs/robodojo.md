# RoboDojo Integration

## Ownership

RoboDojo and Isaac Sim run only in FluxThemis simulator workers. FluxVLA model
workers receive observation arrays over ZMQ and return denormalized joint
actions. Model and simulator worker counts are independent.

## Install

```bash
cd /root/projects/FluxThemis
bash scripts/install_robodojo.sh
```

The installer uses RoboDojo's official environment and asset scripts, then
installs FluxThemis with its ZMQ dependencies into the `RoboDojo` environment.

## Five-Episode Smoke Test

Start FluxVLA:

```bash
conda activate fluxvla
cd /root/projects/FluxVLA
CUDA_VISIBLE_DEVICES=0 \
bash scripts/distributed_zmq_server.sh \
  --config configs/pi05/pi05_paligemma_robodojo_full_finetune.py \
  --ckpt-path /path/to/model.safetensors
```

Start FluxThemis:

```bash
conda activate RoboDojo
cd /root/projects/FluxThemis
CUDA_VISIBLE_DEVICES=1,2,3,4,5 \
bash scripts/distributed_eval.sh \
  --config /root/projects/FluxVLA/configs/pi05/pi05_paligemma_robodojo_full_finetune.py \
  --run-name put-bottles-5ep \
  --save-videos \
  --video-dir /persistent/eval/put-bottles-5ep \
  --cfg-options \
    themis.evaluation.environment.task_name=put_bottles_into_dustbin \
    themis.evaluation.episodes_per_task=5
```

## Full Benchmark

The checked-in RoboDojo configs contain official per-task episode overrides.
Run without task or episode overrides:

```bash
CUDA_VISIBLE_DEVICES=1,2,3,4,5 \
bash scripts/distributed_eval.sh \
  --config /root/projects/FluxVLA/configs/pi05/pi05_paligemma_robodojo_full_finetune.py \
  --run-name pi05-full
```

FluxVLA writes the official progress-score and success-rate overview at
`run_end`. FluxThemis keeps policy seeds fixed while assigning task-local
environment case indexes. For RoboDojo, a case index selects one layout. An
unstable layout is rejected before `episode_start`; the coordinator retries
that episode slot with the next case without counting the rejected attempt. A
RoboDojo's PhysX monitor terminates a disposable Simulator process after a
fatal CUDA/PhysX failure. The local Supervisor starts a fresh worker, which
reconnects directly to the coordinator under the same logical identity. The
coordinator retries the exact episode without stopping other simulators. Use
`--simulator-restart-limit` to bound retries for one episode or failed startup
sequence.

For multiple simulator nodes, use the same shell launcher with the node
environment variables documented in the repository README.
