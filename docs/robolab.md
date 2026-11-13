# RoboLab remote evaluation

## 1. Prerequisites and installation

RoboLab requires an NVIDIA RTX GPU and recommends at least 48 GB VRAM.
See [RoboLab requirements](https://github.com/NVlabs/RoboLab#requirements);
memory usage also depends on the model and recording settings.
Prepare Git, Git LFS, uv, NVIDIA/Vulkan drivers and simulator assets.
The [installer](../scripts/install_robolab.sh) selects RoboLab's locked
Python 3.11 / Isaac Sim 5.0 / IsaacLab 2.2 stack.
Prepare an independent model service using the companion
[FluxVLA RoboLab guide](https://github.com/Aurorajiang17/FluxVLA/blob/a5dea4674a0fcf1f8a8a74dd4b847b04a15d8089/docs/robolab_remote_evaluation.md)
at its fixed companion commit. In an existing FluxVLA checkout, fetch and
select that version before following the guide:

```bash
git -C /path/to/FluxVLA fetch https://github.com/Aurorajiang17/FluxVLA.git a5dea4674a0fcf1f8a8a74dd4b847b04a15d8089
git -C /path/to/FluxVLA checkout --detach a5dea4674a0fcf1f8a8a74dd4b847b04a15d8089
```

Run the following installer from the FluxThemis checkout:

```bash
export ROBOLAB_ROOT=/path/to/RoboLab
bash scripts/install_robolab.sh --robolab-root "$ROBOLAB_ROOT"
export ROBOLAB_PYTHON="$ROBOLAB_ROOT/.venv/bin/python"
export FLUXTHEMIS_PYTHON="$ROBOLAB_PYTHON"
# After accepting NVIDIA's license, set this for headless evaluation:
export OMNI_KIT_ACCEPT_EULA=YES
```

See [NVIDIA's headless license requirement](https://docs.isaacsim.omniverse.nvidia.com/5.0.0/installation/install_python.html).
The installer does not accept the license or start simulation. It reconciles
Isaac kernel dependency requirements before checking the environment.
Use --env-path for a separate environment, or --python to install into an
existing compatible one. Static preflight cannot prove rendering or physics.

## 2. Configuration and execution

Start the matching model service independently, following its RoboLab guide.
Copy [remote.yaml](../configs/robolab/remote.yaml) and set the simulator
interpreter/root, model endpoint, GPU, worker count and output directory.
The default [droid.py](../configs/robolab/droid.py) runs every task for ten
serial episodes with one simulator worker and one dedicated model service.
FluxThemis connects to that service through the shared launcher.

For N GPUs, the companion FluxVLA guide provides a fixed_worker service and
the corresponding frozen YAML. Set the same GPU list on both sides,
workers_per_gpu: 1, scheduling: fixed and model.expected_workers: N.
Each GPU runs one simulator worker and one persistent model instance.
GPU IDs or full GPU UUIDs may be used. Use the guide's common launch procedure
for this mode; the commands below use the default single-worker YAML.

```bash
bash scripts/run_evaluation.sh --config configs/robolab/remote.yaml --dry-run
bash scripts/run_evaluation.sh --config configs/robolab/remote.yaml --smoke
# Start a fresh model service before full evaluation.
bash scripts/run_evaluation.sh --config configs/robolab/remote.yaml --full
```

Smoke selects BananaInBowlTask and one episode; full selects all tasks.
Without either mode flag, the default YAML selects all tasks and ten episodes.
episodes: null preserves the Python-configured episode quota.
The base config accepts the entire returned action chunk. The optional
[droid_h16.py](../configs/robolab/droid_h16.py) executes fifteen actions for
a matching sixteen-output/fifteen-execution recipe. Choose the config from
the model's exported action definition and deployment horizon.
See the [shared evaluation guide](../guides/evaluation.md) for networking,
monitoring and launcher overrides. Direct
[distributed_eval.py](../scripts/distributed_eval.py) use requires
--simulator-restart-limit 0 for RoboLab.

## 3. Evaluation protocol and limitations

Only num_envs=1 is supported. One Isaac application belongs to each worker.
Within each worker, a task reuses its native environment for the episodes
assigned to that worker, then closes it. Fixed assignment uses global task-major
episode numbers: (episode_number - 1) % N, starting at one. Quotas are global
per task, not multiplied by N; each task may have one instance per worker.
Native instance_episode counts local reuse; it can differ from episode_index.
Every episode resets native evaluation state and calls two native resets.
The creation seed is retained; episode seed metadata does not reseed layouts.

Observations contain RGB images under over_shoulder_left_camera and wrist_cam,
and states: seven joint positions followed by the gripper position.
The prompt is the native task instruction. The model returns finite numeric
[T,8] DROID joint-position actions; the shared client has no fixed chunk length.
Ordinary request metadata is used; no caller RNG state is transmitted.

The companion continuous service keeps models alive across episodes and tasks.
Readiness checks do not invoke inference. Each evaluation requires a fresh
model deployment and audit file; see the FluxVLA guide for the service contract.
Video is disabled by default. Technical retries are zero, including idle
failures; resume is unsupported. Technical errors make the run incomplete.
An ordinary unsuccessful task remains completed and counts in the results.
Do not rerun normal failures or repair invalid actions.
Single-environment request ordering requires new score measurements.
Changing GPU count or task order also changes the sampling recipe and may
change actions and scores. Keep these settings fixed when comparing models.
The shared launcher requires a readable Linux /proc process table for all
benchmarks. Permission failures remain errors after best-effort cleanup of
owned processes; restricted process visibility cannot certify full cleanup.

## 4. Results

The shared launcher prints the run directory and retains progress and logs.
FluxVLA reporting stores the authoritative evaluation journal and summary.
Native recordings remain under the environment output path in separate
worker/task directories as run_<episode>.hdf5.
Native frozen success, duration and terminal results remain authoritative.
The native HDF5-derived Score is separate from the success rate.

Missing, malformed or interrupted artifacts remain incomplete errors.
Match the model audit's completed request IDs with the episode journal's
prediction_metadata. An accepted request without completion identifies an
interrupted prediction, even if some task artifacts exist.
In fixed mode, compare each rank's continuous request count and model instance
as well. Requests can finish in a different order across ranks; one worker
finishing its quota does not end the entire run.
