#!/usr/bin/env bash
# Install the environment-side LIBERO runtime into an existing conda env.

set -euo pipefail

LIBERO_REF="058fda1ddebe92918af091cb6816759ca6d003f0"
ROBOSUITE_REF="e293cc32ff3c48957a4ebcad09952432b0dc9049"
LIBERO_REPO="https://github.com/yinchimaoliang/LIBERO.git"
ROBOSUITE_REPO="https://github.com/yinchimaoliang/robosuite.git"
SYSTEM_PACKAGES=(
  git
  libegl1
  libglvnd0
  libopengl0
  libegl-dev
  libgl1-mesa-dev
  libx11-dev
  libglew-dev
  libosmesa6-dev
)

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
ENV_NAME="${FLUXTHEMIS_CONDA_ENV:-fluxthemis-libero}"
SOURCE_ROOT=""
SYSTEM_DEPS="auto"
SMOKE_LEVEL="libero"
TORCH_VERSION="${FLUXTHEMIS_TORCH_VERSION:-2.8.0}"
TORCH_INDEX_URLS="${FLUXTHEMIS_TORCH_INDEX_URLS:-https://download.pytorch.org/whl/cpu}"
TORCH_WHEEL_URL="${FLUXTHEMIS_TORCH_WHEEL_URL-}"
WITH_DEV=0
DRY_RUN=0
CLONE_TMP=""
CHECKOUT_PATH=""

usage() {
  cat <<'EOF'
Usage: bash scripts/install_libero.sh [options]

Installs LIBERO into an existing Python 3.10 or 3.12 conda environment. The
default environment is `fluxthemis-libero`; it is never deleted or recreated.

Options:
  --env-name NAME              Target conda environment (default:
                               fluxthemis-libero).
  --source-root PATH           Managed LIBERO/robosuite checkout directory
                               (default: <conda-prefix>/src/fluxthemis).
  --system-deps auto|always|never
                               Install missing EGL system packages when
                               possible (default: auto).
  --smoke-level libero|imports|none
                               Verification depth (default: libero).
  --torch-version VERSION      CPU Torch version installed when Torch is
                               absent (default: 2.8.0).
  --with-dev                   Install FluxThemis development dependencies.
  --dry-run                    Print mutating commands without running them.
  -h, --help                   Show this help.

Environment variables:
  FLUXTHEMIS_CONDA_ENV, FLUXTHEMIS_TORCH_VERSION
  FLUXTHEMIS_TORCH_WHEEL_URL    Direct CPU wheel URL. Linux x86_64 defaults
                                to the Aliyun mirror; empty disables it.
  FLUXTHEMIS_TORCH_INDEX_URLS   Fallback CPU wheel indexes. Default: PyTorch
                                official.

Examples:
  bash scripts/install_libero.sh
  bash scripts/install_libero.sh --smoke-level imports
  bash scripts/install_libero.sh --env-name fluxthemis-libero --with-dev
EOF
}

die() {
  echo "error: $*" >&2
  exit 1
}

print_command() {
  printf '+' >&2
  printf ' %q' "$@" >&2
  printf '\n' >&2
}

run() {
  print_command "$@"
  if [[ "${DRY_RUN}" == "0" ]]; then
    "$@"
  fi
}

conda_run() {
  run "${CONDA_BIN}" run --no-capture-output -n "${ENV_NAME}" "$@"
}

cleanup() {
  if [[ -n "${CLONE_TMP}" && -d "${CLONE_TMP}" ]]; then
    case "${CLONE_TMP}" in
      "${SOURCE_ROOT}"/.clone-*) rm -rf -- "${CLONE_TMP}" ;;
    esac
  fi
}
trap cleanup EXIT

while [[ $# -gt 0 ]]; do
  case "$1" in
    --env-name)
      [[ $# -ge 2 ]] || die '--env-name requires a value'
      ENV_NAME="$2"
      shift 2
      ;;
    --source-root)
      [[ $# -ge 2 ]] || die '--source-root requires a value'
      SOURCE_ROOT="$2"
      shift 2
      ;;
    --system-deps)
      [[ $# -ge 2 ]] || die '--system-deps requires a value'
      SYSTEM_DEPS="$2"
      shift 2
      ;;
    --smoke-level)
      [[ $# -ge 2 ]] || die '--smoke-level requires a value'
      SMOKE_LEVEL="$2"
      shift 2
      ;;
    --torch-version)
      [[ $# -ge 2 ]] || die '--torch-version requires a value'
      TORCH_VERSION="$2"
      shift 2
      ;;
    --with-dev)
      WITH_DEV=1
      shift
      ;;
    --dry-run)
      DRY_RUN=1
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *) die "unknown option: $1" ;;
  esac
done

case "${SYSTEM_DEPS}" in
  auto|always|never) ;;
  *) die '--system-deps must be auto, always, or never' ;;
esac
case "${SMOKE_LEVEL}" in
  libero|imports|none) ;;
  *) die '--smoke-level must be libero, imports, or none' ;;
esac
CONDA_BIN="$(command -v conda || true)"
[[ -n "${CONDA_BIN}" ]] || die 'conda is not available on PATH'
PYTHON_VERSION="$("${CONDA_BIN}" run -n "${ENV_NAME}" python -c \
  'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")' \
  2>/dev/null || true)"
case "${PYTHON_VERSION}" in
  3.10|3.12) ;;
  *) die "conda env ${ENV_NAME@Q} must exist and use Python 3.10 or 3.12" ;;
esac
PYTHON_TAG="cp${PYTHON_VERSION/./}"
if [[ ! -v FLUXTHEMIS_TORCH_WHEEL_URL \
      && "$(uname -s)" == "Linux" && "$(uname -m)" == "x86_64" ]]; then
  TORCH_WHEEL_URL="https://mirrors.aliyun.com/pytorch-wheels/cpu/torch-${TORCH_VERSION}%2Bcpu-${PYTHON_TAG}-${PYTHON_TAG}-manylinux_2_28_x86_64.whl"
fi
ENV_PREFIX="$("${CONDA_BIN}" run -n "${ENV_NAME}" python -c \
  'import sys; print(sys.prefix)')"
[[ -n "${SOURCE_ROOT}" ]] || SOURCE_ROOT="${ENV_PREFIX}/src/fluxthemis"
SOURCE_ROOT="$(realpath -m "${SOURCE_ROOT}")"

ensure_system_dependencies() {
  local missing=()
  local package
  if ! command -v dpkg-query >/dev/null 2>&1; then
    [[ "${SYSTEM_DEPS}" == "always" ]] && \
      die 'system dependency installation currently supports apt/dpkg hosts'
    return
  fi
  for package in "${SYSTEM_PACKAGES[@]}"; do
    if ! dpkg-query -W -f='${Status}' "${package}" 2>/dev/null | \
      grep -q 'install ok installed'; then
      missing+=("${package}")
    fi
  done
  if [[ ${#missing[@]} -eq 0 && "${SYSTEM_DEPS}" != "always" ]]; then
    echo 'EGL system dependencies are already installed.'
    return
  fi
  if [[ "${SYSTEM_DEPS}" == "never" ]]; then
    echo "warning: missing system packages: ${missing[*]}" >&2
    return
  fi
  if [[ "${SYSTEM_DEPS}" == "auto" ]] && command -v git >/dev/null 2>&1 && \
      "${CONDA_BIN}" run -n "${ENV_NAME}" python -c \
      'import ctypes; [ctypes.CDLL(name) for name in ("libEGL.so.1", "libGL.so.1", "libGLESv2.so.2")]' \
      >/dev/null 2>&1; then
    echo 'EGL/OpenGL runtime is available outside dpkg; skipping optional system packages.'
    return
  fi
  if [[ "${EUID}" -eq 0 ]]; then
    run env DEBIAN_FRONTEND=noninteractive apt-get update
    run env DEBIAN_FRONTEND=noninteractive apt-get install -y \
      "${SYSTEM_PACKAGES[@]}"
  elif command -v sudo >/dev/null 2>&1 && sudo -n true 2>/dev/null; then
    run sudo -n env DEBIAN_FRONTEND=noninteractive apt-get update
    run sudo -n env DEBIAN_FRONTEND=noninteractive apt-get install -y \
      "${SYSTEM_PACKAGES[@]}"
  else
    die "missing system packages: ${missing[*]}; rerun as root or install them first"
  fi
}

bootstrap_pip() {
  if "${CONDA_BIN}" run -n "${ENV_NAME}" python -m pip --version \
    >/dev/null 2>&1; then
    return
  fi
  conda_run python -m ensurepip --upgrade
}

prepare_checkout() {
  local name="$1"
  local ref="$2"
  local remote="$3"
  local target="${SOURCE_ROOT}/${name}"
  local head
  if [[ -e "${target}" ]]; then
    [[ -d "${target}/.git" ]] || \
      die "managed source path is not a git checkout: ${target}"
    [[ -z "$(git -C "${target}" status --porcelain)" ]] || \
      die "managed source checkout is dirty: ${target}"
    head="$(git -C "${target}" rev-parse HEAD)"
    [[ "${head}" == "${ref}" ]] || \
      die "${target} is at ${head}, expected ${ref}; move it aside and rerun"
    CHECKOUT_PATH="${target}"
    return
  fi

  if [[ "${DRY_RUN}" == "1" ]]; then
    print_command mkdir -p "${SOURCE_ROOT}"
    print_command git clone --no-checkout "${remote}" "${target}"
    print_command git -C "${target}" checkout --detach "${ref}"
    CHECKOUT_PATH="${target}"
    return
  fi

  mkdir -p "${SOURCE_ROOT}"
  CLONE_TMP="$(mktemp -d "${SOURCE_ROOT}/.clone-${name}.XXXXXX")"
  run git clone --no-checkout "${remote}" "${CLONE_TMP}"
  run git -C "${CLONE_TMP}" checkout --detach "${ref}"
  [[ "$(git -C "${CLONE_TMP}" rev-parse HEAD)" == "${ref}" ]] || \
    die "failed to resolve ${name} at ${ref}"
  run mv "${CLONE_TMP}" "${target}"
  CLONE_TMP=""
  CHECKOUT_PATH="${target}"
}

ensure_system_dependencies
bootstrap_pip
conda_run python -m pip install --upgrade pip setuptools wheel

if ! "${CONDA_BIN}" run -n "${ENV_NAME}" python -c \
  'import torch; version=tuple(int(x) for x in torch.__version__.split("+")[0].split(".")[:2]); raise SystemExit(version < (2, 0))' \
  >/dev/null 2>&1; then
  # Resolve small Python dependencies from the user's normal index. Asking
  # download.pytorch.org to resolve them is both slow and needlessly fragile.
  conda_run python -m pip install filelock 'typing-extensions>=4.10' \
    'sympy>=1.13.3' networkx jinja2 fsspec
  torch_installed=0
  if [[ -n "${TORCH_WHEEL_URL}" ]]; then
    if conda_run python -m pip install --no-deps "${TORCH_WHEEL_URL}" \
      --timeout 120 --retries 2; then
      torch_installed=1
    else
      echo 'warning: direct Torch wheel failed; trying configured indexes' >&2
    fi
  fi
  if [[ "${torch_installed}" == "0" ]]; then
    for torch_index in ${TORCH_INDEX_URLS}; do
      if conda_run python -m pip install --no-deps \
        "torch==${TORCH_VERSION}" --index-url "${torch_index}" \
        --timeout 120 --retries 2; then
        torch_installed=1
        break
      fi
      echo "warning: Torch install failed via ${torch_index}; trying next index" >&2
    done
  fi
  [[ "${torch_installed}" == "1" ]] || \
    die 'failed to install CPU Torch from every configured index'
else
  echo 'Keeping the compatible Torch already installed in the target env.'
fi

PROJECT_SPEC="${PROJECT_ROOT}"
if [[ "${WITH_DEV}" == "1" ]]; then
  PROJECT_SPEC="${PROJECT_ROOT}[dev]"
fi
conda_run python -m pip install -r "${PROJECT_ROOT}/requirements-libero.txt"
conda_run python -m pip install -e "${PROJECT_SPEC}"

prepare_checkout LIBERO "${LIBERO_REF}" "${LIBERO_REPO}"
LIBERO_DIR="${CHECKOUT_PATH}"
prepare_checkout robosuite "${ROBOSUITE_REF}" "${ROBOSUITE_REPO}"
ROBOSUITE_DIR="${CHECKOUT_PATH}"

conda_run python -m pip install --no-deps -e "${ROBOSUITE_DIR}"
conda_run python -m pip install --no-deps -e "${LIBERO_DIR}"

LIBERO_CONFIG_DIR="${ENV_PREFIX}/etc/fluxthemis/libero"
run mkdir -p "${LIBERO_CONFIG_DIR}"
conda_run env "LIBERO_CONFIG_PATH=${LIBERO_CONFIG_DIR}" python -c \
  'import sys; from libero.libero import set_libero_default_path; set_libero_default_path(sys.argv[1])' \
  "${LIBERO_DIR}/libero/libero" >/dev/null

if [[ "${DRY_RUN}" == "1" ]]; then
  echo 'Dry run complete; no installation or smoke test was performed.'
  exit 0
fi

conda_run python -m pip check
[[ "$(git -C "${LIBERO_DIR}" rev-parse HEAD)" == "${LIBERO_REF}" ]]
[[ "$(git -C "${ROBOSUITE_DIR}" rev-parse HEAD)" == "${ROBOSUITE_REF}" ]]
conda_run python \
  "${PROJECT_ROOT}/scripts/distributed_eval.py" --help >/dev/null

case "${SMOKE_LEVEL}" in
  imports)
    conda_run python "${PROJECT_ROOT}/scripts/libero_smoke.py" \
      --imports-only
    ;;
  libero)
    conda_run python "${PROJECT_ROOT}/scripts/libero_smoke.py" \
      --suite libero_10 --task-id 0 --resolution 64 --steps 1
    ;;
  none) ;;
esac

cat <<EOF

LIBERO installation complete.
  conda env:    ${ENV_NAME}
  source root:  ${SOURCE_ROOT}
  config path:  ${LIBERO_CONFIG_DIR}

Activate it with:
  conda activate ${ENV_NAME}
EOF
