#!/usr/bin/env bash
# Use RoboLab's official uv stack, then install Themis in the simulator env.
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ROBOLAB_ROOT="${ROBOLAB_ROOT:-$(dirname "${PROJECT_ROOT}")/RoboLab}"
ROBOLAB_REPO="${ROBOLAB_REPO:-https://github.com/NVlabs/RoboLab.git}"
ROBOLAB_REF="${ROBOLAB_REF:-9cce00cadd5dc575fd2b8ea3836f5bdcde574229}"
ENV_PATH=""
PYTHON=""
SKIP_ASSETS=0
DRY_RUN=0

usage() {
  cat <<'EOF'
Usage: bash scripts/install_robolab.sh [options]

Creates RoboLab's Python 3.11 / Isaac Sim 5.0 / IsaacLab 2.2 environment
with its locked isaac50 extra, then installs RoboLab and FluxThemis together.
Requires Git, Git LFS (unless --skip-assets) and uv on PATH.

Options:
  --robolab-root PATH  Checkout (default: <Themis parent>/RoboLab).
  --env-path PATH      uv environment (default: <RoboLab>/.venv).
  --python PATH        Use an existing simulator Python instead of uv sync.
                      Its Python/Isaac versions must match the selected stack.
  --skip-assets        Skip LFS asset download and static asset verification.
  --dry-run            Print the installation plan without changing files.
  -h, --help           Show this help.

ROBOLAB_REPO and ROBOLAB_REF select the source. Existing checkouts must be
at the selected revision with no tracked edits; they are never reset or moved
to another revision. Proxy, Git and uv package-index settings are inherited.
System graphics drivers, Vulkan and ffmpeg must be prepared separately.
Installation does not start simulation, accept a simulator EULA or load a model.
EOF
}

die() { echo "error: $*" >&2; exit 1; }
run() {
  printf '+' >&2
  printf ' %q' "$@" >&2
  printf '\n' >&2
  if [[ "${DRY_RUN}" == 0 ]]; then "$@"; fi
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --robolab-root|--env-path|--python)
      [[ $# -ge 2 && -n "$2" && "$2" != --* ]] || die "$1 requires a value"
      case "$1" in
        --robolab-root) ROBOLAB_ROOT="$2" ;;
        --env-path) ENV_PATH="$2" ;;
        --python) PYTHON="$2" ;;
      esac
      shift 2 ;;
    --skip-assets) SKIP_ASSETS=1; shift ;;
    --dry-run) DRY_RUN=1; shift ;;
    -h|--help) usage; exit 0 ;;
    *) die "unknown option: $1" ;;
  esac
done
[[ -z "${PYTHON}" || -z "${ENV_PATH}" ]] || \
  die '--python and --env-path select different environment modes'
ROBOLAB_ROOT="$(realpath -m "${ROBOLAB_ROOT}")"
if [[ "${DRY_RUN}" == 0 ]]; then
  command -v git >/dev/null || die 'git is required'
  command -v uv >/dev/null || die 'uv is required; follow RoboLab installation prerequisites'
  [[ "${SKIP_ASSETS}" == 1 ]] || git lfs version >/dev/null
fi

if [[ -e "${ROBOLAB_ROOT}" ]]; then
  git -C "${ROBOLAB_ROOT}" rev-parse --is-inside-work-tree >/dev/null 2>&1 || \
    die "not a Git checkout: ${ROBOLAB_ROOT}"
  expected="$(git -C "${ROBOLAB_ROOT}" rev-parse --verify "${ROBOLAB_REF}^{commit}")"
  actual="$(git -C "${ROBOLAB_ROOT}" rev-parse HEAD)"
  [[ "${actual}" == "${expected}" ]] || die 'existing RoboLab revision differs; select a separate checkout'
  git -C "${ROBOLAB_ROOT}" diff --quiet && \
    git -C "${ROBOLAB_ROOT}" diff --cached --quiet || \
    die 'existing RoboLab checkout has tracked changes'
else
  run env GIT_LFS_SKIP_SMUDGE=1 git clone --no-checkout \
    "${ROBOLAB_REPO}" "${ROBOLAB_ROOT}"
  run env GIT_LFS_SKIP_SMUDGE=1 git -C "${ROBOLAB_ROOT}" \
    checkout --detach "${ROBOLAB_REF}"
fi
if [[ "${DRY_RUN}" == 0 ]]; then
  [[ -f "${ROBOLAB_ROOT}/pyproject.toml" && -f "${ROBOLAB_ROOT}/uv.lock" ]] || \
    die 'RoboLab pyproject.toml and uv.lock are required'
fi

if [[ -n "${PYTHON}" ]]; then
  # Preserve a venv's interpreter symlink rather than resolving its base Python.
  PYTHON="$(realpath -ms "${PYTHON}")"
  [[ "${DRY_RUN}" == 1 || -x "${PYTHON}" ]] || die 'simulator Python is not executable'
else
  ENV_PATH="$(realpath -m "${ENV_PATH:-${ROBOLAB_ROOT}/.venv}")"
  PYTHON="${ENV_PATH}/bin/python"
  if [[ -e "${ENV_PATH}" && ! -x "${PYTHON}" ]]; then
    die '--env-path already exists without a Python environment'
  fi
fi
if [[ "${DRY_RUN}" == 0 && -x "${PYTHON}" ]]; then
  version="$("${PYTHON}" -c 'import sys; print("%s.%s" % sys.version_info[:2])')"
  [[ "${version}" == 3.11 ]] || die 'the isaac50 stack requires Python 3.11'
fi
if [[ -n "${ENV_PATH}" ]]; then
  run env "UV_PROJECT_ENVIRONMENT=${ENV_PATH}" uv sync \
    --project "${ROBOLAB_ROOT}" --locked --extra isaac50 --python 3.11 --inexact
fi
# Metadata inspection avoids importing Isaac or initializing CUDA.
run "${PYTHON}" -c 'import sys; from importlib.metadata import version
from packaging.version import Version
assert sys.version_info[:2] == (3, 11), "Python 3.11 required"
assert version("isaaclab") == "2.2.0", "IsaacLab 2.2.0 required"
assert Version(version("isaacsim").split("+")[0]) == Version("5.0.0"), "Isaac Sim 5.0.0 required"'
if [[ "${SKIP_ASSETS}" == 0 ]]; then
  run git -C "${ROBOLAB_ROOT}" lfs pull --include='assets/**' --exclude=''
fi
# Isaac50 kernel wheel metadata pins websockets; the native lock omits it.
# Apply this constraint only to the selected simulator environment.
run uv pip install --python "${PYTHON}" -e "${ROBOLAB_ROOT}" -e "${PROJECT_ROOT}" \
  "websockets==12.0"
run uv pip check --python "${PYTHON}"
if [[ "${SKIP_ASSETS}" == 0 ]]; then
  run env "PYTHONPATH=${PROJECT_ROOT}/scripts:${PROJECT_ROOT}" "${PYTHON}" -c \
    'import sys; from preflight.robolab import check_environment
result = check_environment(sys.argv[1])
print("Static RoboLab task/asset checks:", len(result["task_sources"]))
print("References requiring runtime validation:", result["unchecked_assets"])' \
    "${ROBOLAB_ROOT}"
fi
if [[ "${DRY_RUN}" == 1 ]]; then
  echo 'Dry run complete; no checkout, environment, assets or packages changed.'
else
  echo 'RoboLab + FluxThemis installation complete; rendering is not tested.'
  [[ "${SKIP_ASSETS}" == 0 ]] || echo 'Assets skipped; prepare and validate them before evaluation.'
  printf 'Simulator Python: %s\nROBOLAB_ROOT: %s\n' "${PYTHON}" "${ROBOLAB_ROOT}"
fi
