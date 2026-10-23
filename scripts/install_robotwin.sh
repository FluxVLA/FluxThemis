#!/usr/bin/env bash
# Install only the simulator runtime in an existing Python 3.10 conda env.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
ENV_NAME="${FLUXTHEMIS_CONDA_ENV:-fluxthemis-robotwin}"
ROBOTWIN_ROOT="${ROBOTWIN_ROOT:-$(dirname "${PROJECT_ROOT}")/RoboTwin}"
ROBOTWIN_REPO="${ROBOTWIN_REPO:-https://github.com/RoboTwin-Platform/RoboTwin.git}"
ROBOTWIN_REF="${ROBOTWIN_REF:-c3ddfa8b97d5519efa828b075999bd0006778e5e}"
CUROBO_REPO="${CUROBO_REPO:-https://github.com/NVlabs/curobo.git}"
CUROBO_REF="${CUROBO_REF:-v0.7.8}"
TORCH_INDEX_URLS="${FLUXTHEMIS_TORCH_INDEX_URLS:-https://download.pytorch.org/whl/cu128}"
MAX_JOBS="${MAX_JOBS:-8}"
HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
DRY_RUN=0
SKIP_ASSETS=0
SKIP_CUROBO=0
UPSTREAM_PATCHES=1
CAMERA_SYNC=1

usage() {
  cat <<'EOF'
Usage: bash scripts/install_robotwin.sh [options]
  --env-name NAME         Existing Python 3.10 env (fluxthemis-robotwin)
  --robotwin-root PATH    RoboTwin checkout (<Themis parent>/RoboTwin)
  --torch-index-url URL   CUDA torch wheel index (cu128 by default)
  --skip-assets           Skip asset downloads, path updates and task imports
  --skip-curobo           Skip cuRobo compilation (evaluation requires it)
  --no-upstream-patches   Skip the optional sapien UTF-8/mplib planner patches
  --no-camera-sync       Skip the RoboTwin camera synchronization patch
  --dry-run              Check env and print commands without installing
  -h, --help             Show this help

ROBOTWIN_REPO/REF and CUROBO_REPO/REF select sources for new checkouts.
Existing checkouts are reused without comparing their commit or source hash.
TORCH_CUDA_ARCH_LIST controls cuRobo compilation; required without a GPU.
MAX_JOBS defaults to 8; HF_ENDPOINT defaults to https://hf-mirror.com.
OIDN_ARCHIVE selects a local official OIDN 2.3.3 archive (SHA256 verified).
No conda environment is created or deleted; no system packages are installed.
EOF
}

die() { echo "error: $*" >&2; exit 1; }
run() {
  printf '+' >&2
  printf ' %q' "$@" >&2
  printf '\n' >&2
  if [[ "${DRY_RUN}" == 0 ]]; then "$@"; fi
}
conda_run() {
  run "${CONDA_BIN}" run --no-capture-output -n "${ENV_NAME}" "$@"
}
python_step() {
  local source
  source="$(cat)"
  conda_run python -c "${source}" "$@"
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --env-name|--robotwin-root|--torch-index-url)
      [[ $# -ge 2 && -n "$2" ]] || die "$1 requires a value"
      case "$1" in
        --env-name) ENV_NAME="$2" ;;
        --robotwin-root) ROBOTWIN_ROOT="$2" ;;
        --torch-index-url) TORCH_INDEX_URLS="$2" ;;
      esac
      shift 2 ;;
    --skip-assets) SKIP_ASSETS=1; shift ;;
    --skip-curobo) SKIP_CUROBO=1; shift ;;
    --no-upstream-patches) UPSTREAM_PATCHES=0; shift ;;
    --no-camera-sync) CAMERA_SYNC=0; shift ;;
    --dry-run) DRY_RUN=1; shift ;;
    -h|--help) usage; exit 0 ;;
    *) die "unknown option: $1" ;;
  esac
done

CONDA_BIN="$(command -v conda || true)"
[[ -n "${CONDA_BIN}" ]] || die 'conda is not available on PATH'
PYTHON_VERSION="$("${CONDA_BIN}" run -n "${ENV_NAME}" python -c \
  'import sys; print("%s.%s" % sys.version_info[:2])' 2>/dev/null || true)"
[[ "${PYTHON_VERSION}" == 3.10 ]] || die \
  "An existing Python 3.10 env is required. Run: conda create -n ${ENV_NAME} python=3.10 -y"
command -v git >/dev/null || die 'git is required'
command -v ffmpeg >/dev/null || echo 'warning: ffmpeg missing; videos require it' >&2
"${CONDA_BIN}" run -n "${ENV_NAME}" python -c \
  'import ctypes; ctypes.CDLL("libvulkan.so.1")' || die 'libvulkan.so.1 is required'
if ! compgen -G '/usr/share/vulkan/icd.d/nvidia*.json' >/dev/null && \
   ! compgen -G '/etc/vulkan/icd.d/nvidia*.json' >/dev/null; then
  echo 'warning: NVIDIA Vulkan ICD missing; RoboTwinEnvironment supplies a fallback manifest' >&2
fi
ROBOTWIN_ROOT="$("${CONDA_BIN}" run -n "${ENV_NAME}" python -c \
  'import pathlib, sys; print(pathlib.Path(sys.argv[1]).expanduser().resolve())' \
  "${ROBOTWIN_ROOT}")"
CUROBO_ROOT="${ROBOTWIN_ROOT}/envs/curobo"
[[ "${MAX_JOBS}" =~ ^[1-9][0-9]*$ ]] || die 'MAX_JOBS must be positive'
if [[ "${SKIP_CUROBO}" == 0 && -z "${TORCH_CUDA_ARCH_LIST:-}" ]]; then
  TORCH_CUDA_ARCH_LIST="$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader \
    2>/dev/null | sort -u | paste -sd ';' - || true)"
  [[ -n "${TORCH_CUDA_ARCH_LIST}" ]] || die \
    'Set TORCH_CUDA_ARCH_LIST explicitly when GPU compute capability is unavailable'
fi

# Query metadata, never initialize CUDA merely to inspect installed versions.
if ! "${CONDA_BIN}" run -n "${ENV_NAME}" python -c '
from importlib.metadata import version
assert version("torch").split("+")[0] == "2.8.0"
assert "+cu128" in version("torch")
assert version("torchvision").split("+")[0] == "0.23.0"
' 2>/dev/null; then
  read -r -a INDEXES <<< "${TORCH_INDEX_URLS}"
  installed=0
  for index in "${INDEXES[@]}"; do
    if conda_run python -m pip install torch==2.8.0 torchvision==0.23.0 \
        --index-url "${index}"; then installed=1; break; fi
  done
  [[ "${installed}" == 1 ]] || die 'Failed to install CUDA torch wheels'
fi
conda_run python -m pip install -r "${PROJECT_ROOT}/requirements-robotwin.txt"
conda_run python -m pip install -e "${PROJECT_ROOT}"
OIDN_ARGS=()
if [[ -n "${OIDN_ARCHIVE:-}" ]]; then OIDN_ARGS+=(--archive "${OIDN_ARCHIVE}"); fi
conda_run python "${SCRIPT_DIR}/robotwin/oidn.py" install "${OIDN_ARGS[@]}"
if [[ ! -e "${ROBOTWIN_ROOT}" ]]; then
  run git clone "${ROBOTWIN_REPO}" "${ROBOTWIN_ROOT}"
  run git -C "${ROBOTWIN_ROOT}" checkout --detach "${ROBOTWIN_REF}"
fi

if [[ "${CAMERA_SYNC}" == 1 ]]; then
  CAMERA_PATCH="${SCRIPT_DIR}/patches/robotwin-camera-sync.patch"
  if [[ -d "${ROBOTWIN_ROOT}" ]] && git -C "${ROBOTWIN_ROOT}" apply \
      --reverse --check "${CAMERA_PATCH}" 2>/dev/null; then
    echo 'RoboTwin camera synchronization patch already applied'
  else
    run git -C "${ROBOTWIN_ROOT}" apply --check "${CAMERA_PATCH}"
    run git -C "${ROBOTWIN_ROOT}" apply "${CAMERA_PATCH}"
  fi
fi

if [[ "${UPSTREAM_PATCHES}" == 1 ]]; then
  python_step <<'PY'
import importlib.util
import pathlib
import re

def patch(package, relative, pattern, replacement):
    root = pathlib.Path(importlib.util.find_spec(package).origin).parent
    path = root / relative
    source = path.read_text(encoding='utf-8')
    result, count = re.subn(pattern, replacement, source)
    if count:
        path.write_text(result, encoding='utf-8')
    print(f'{package}: {count} upstream replacements ({path})')

patch('sapien', 'wrapper/urdf_loader.py',
      r'("r")(\))( as)', r'\1, encoding="utf-8") as')
patch('mplib', 'planner.py',
      r'(if np.linalg.norm\(delta_twist\) < 1e-4 )'
      r'(or collide )(or not within_joint_limit:)', r'\1\3')
PY
fi

if [[ "${SKIP_CUROBO}" == 0 ]]; then
  if [[ ! -e "${CUROBO_ROOT}" ]]; then
    run git clone --branch "${CUROBO_REF}" --depth 1 "${CUROBO_REPO}" "${CUROBO_ROOT}"
  fi
  python_step "${CUROBO_ROOT}" <<'PY'
import pathlib
import shutil
import sys

root = pathlib.Path(sys.argv[1]).resolve()
build = root / 'build'
if build.is_symlink():
    build.unlink()
elif build.exists():
    shutil.rmtree(build)
for path in (root / 'src').rglob('*.so'):
    path.unlink()
PY
  conda_run env \
    "TORCH_CUDA_ARCH_LIST=${TORCH_CUDA_ARCH_LIST}" "MAX_JOBS=${MAX_JOBS}" \
    python -m pip install --force-reinstall --no-deps --no-build-isolation \
    -e "${CUROBO_ROOT}"
  conda_run python -c 'import torch; import curobo, curobo.curobolib.geom_cu'
fi

if [[ "${SKIP_ASSETS}" == 0 ]]; then
  python_step "${ROBOTWIN_ROOT}" "${HF_ENDPOINT}" <<'PY'
import pathlib
import shutil
import subprocess
import sys
import zipfile
from huggingface_hub import snapshot_download

root = pathlib.Path(sys.argv[1])
assets = root / 'assets'
assets.mkdir(parents=True, exist_ok=True)
names = ['background_texture', 'embodiments', 'objects']
missing = [name for name in names if not (assets / name).is_dir()]
if missing:
    snapshot_download(repo_id='TianxingChen/RoboTwin2.0', repo_type='dataset',
                      endpoint=sys.argv[2], local_dir=str(assets),
                      allow_patterns=[name + '.zip' for name in missing])
    for name in missing:
        with zipfile.ZipFile(assets / (name + '.zip')) as archive:
            for member in archive.infolist():
                target = (assets / member.filename).resolve()
                if not target.is_relative_to(assets.resolve()):
                    raise ValueError(f'Unsafe archive member: {member.filename}')
            archive.extractall(assets)
        if not (assets / name).is_dir():
            raise RuntimeError(f'Asset archive did not provide {name}')
    shutil.rmtree(assets / '__MACOSX', ignore_errors=True)
subprocess.run([sys.executable, 'script/update_embodiment_config_path.py'],
               cwd=root, stdin=subprocess.DEVNULL, check=True)
PY
fi

python_step "${ROBOTWIN_ROOT}" "${SKIP_CUROBO}" "${SKIP_ASSETS}" <<'PY'
import ast
import importlib
import os
import sys
from pathlib import Path
from fluxthemis.environments import RoboTwinEnvironment
from fluxthemis.environments.sim.robotwin import ROBOTWIN_TASKS

root = sys.argv[1]
os.chdir(root)
sys.path.insert(0, root)
import sapien
import mplib
if sys.argv[2] != '1':
    import curobo
if sys.argv[3] == '1':
    # Upstream task imports read assets and load the CUDA planner. An image
    # built without assets can check source syntax without those side effects.
    for task in ROBOTWIN_TASKS:
        path = Path(root) / 'envs' / (task + '.py')
        ast.parse(path.read_text(encoding='utf-8'), filename=str(path))
    print('RoboTwin task syntax and Themis adapter verified; task imports deferred until assets are mounted')
else:
    import envs
    from envs.utils.create_actor import UnStableError
    for task in ROBOTWIN_TASKS:
        importlib.import_module('envs.' + task)
    print('RoboTwin task imports and Themis adapter verified')
PY

echo "Environment: ${ENV_NAME}; RoboTwin: ${ROBOTWIN_ROOT}"
echo "cuRobo: ${CUROBO_ROOT} (skip=${SKIP_CUROBO}); upstream patches=${UPSTREAM_PATCHES}; camera sync=${CAMERA_SYNC}"
printf 'export ROBOTWIN_ROOT=%q\n' "${ROBOTWIN_ROOT}"
echo "Next: start the model separately; see docs/robotwin.md"
printf 'In the simulator environment: conda activate %q\n' "${ENV_NAME}"
echo "bash scripts/run_evaluation.sh --config configs/robotwin/aloha.py --server-endpoint tcp://MODEL_HOST:15555 --smoke --dry-run"
