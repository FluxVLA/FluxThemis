#!/usr/bin/env bash
# Install the isolated environment-side LIBERO-Plus runtime.

set -euo pipefail

LIBERO_PLUS_REF="d7c160eca38bf268d5965aef6d4c6dca12d5537c"
ROBOSUITE_REF="e293cc32ff3c48957a4ebcad09952432b0dc9049"
LIBERO_PLUS_REPO="https://github.com/hqr-robotic/LIBERO-plus.git"
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
  libexpat1
  libfontconfig1-dev
  libmagickwand-dev
)

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
ENV_NAME="${FLUXTHEMIS_CONDA_ENV:-fluxthemis-libero-plus}"
FLUXVLA_ROOT="${FLUXVLA_ROOT:-/root/projects/FluxVLA}"
SOURCE_ROOT=""
SYSTEM_DEPS="auto"
SMOKE_LEVEL="libero-plus"
ASSET_MODE="auto"
ASSET_ROOT=""
ASSET_DOWNLOAD_ROOT=""
ASSET_CACHE_DIR="${FLUXTHEMIS_LIBERO_PLUS_ASSET_CACHE:-/tmp/libero-plus-assets}"
ASSET_ENDPOINT="${HF_ENDPOINT:-https://huggingface.co}"
TORCH_VERSION="${FLUXTHEMIS_TORCH_VERSION:-2.8.0}"
TORCH_INDEX_URLS="${FLUXTHEMIS_TORCH_INDEX_URLS:-https://download.pytorch.org/whl/cpu}"
TORCH_WHEEL_URL="${FLUXTHEMIS_TORCH_WHEEL_URL-}"
WITH_DEV=0
DRY_RUN=0
CLONE_TMP=""
CHECKOUT_PATH=""

usage() {
  cat <<'EOF'
Usage: bash scripts/install_libero_plus.sh [options]

Installs the pinned LIBERO-Plus fork into an existing Python 3.10 or 3.12
Conda environment. The default environment is `fluxthemis-libero-plus`; it is
never deleted or recreated and remains isolated from standard LIBERO.

Options:
  --env-name NAME              Target environment (default:
                               fluxthemis-libero-plus).
  --fluxvla-root PATH          FluxVLA checkout used as a possible existing
                               asset/source seed.
  --source-root PATH           Managed source directory (default:
                               <conda-prefix>/src/fluxthemis).
  --asset-mode auto|link|download|none
                               Asset strategy (default: auto). Auto links a
                               complete existing asset root before download.
  --asset-root PATH            Existing complete LIBERO-Plus assets directory.
  --asset-download-root PATH   Download/extraction destination (default:
                               <source-root>/libero-plus-assets).
  --asset-cache-dir PATH       Official assets.zip cache (default:
                               /tmp/libero-plus-assets).
  --asset-endpoint URL         Hugging Face endpoint (default: HF_ENDPOINT or
                               https://huggingface.co).
  --system-deps auto|always|never
                               Install missing EGL/ImageMagick packages.
  --smoke-level libero-plus|imports|none
                               Verification depth (default: libero-plus).
  --torch-version VERSION      CPU Torch version installed when absent.
  --with-dev                   Install development dependencies.
  --dry-run                    Print mutating commands without running them.
  -h, --help                   Show this help.
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
    --fluxvla-root)
      [[ $# -ge 2 ]] || die '--fluxvla-root requires a value'
      FLUXVLA_ROOT="$2"
      shift 2
      ;;
    --source-root)
      [[ $# -ge 2 ]] || die '--source-root requires a value'
      SOURCE_ROOT="$2"
      shift 2
      ;;
    --asset-mode)
      [[ $# -ge 2 ]] || die '--asset-mode requires a value'
      ASSET_MODE="$2"
      shift 2
      ;;
    --asset-root)
      [[ $# -ge 2 ]] || die '--asset-root requires a value'
      ASSET_ROOT="$2"
      shift 2
      ;;
    --asset-download-root)
      [[ $# -ge 2 ]] || die '--asset-download-root requires a value'
      ASSET_DOWNLOAD_ROOT="$2"
      shift 2
      ;;
    --asset-cache-dir)
      [[ $# -ge 2 ]] || die '--asset-cache-dir requires a value'
      ASSET_CACHE_DIR="$2"
      shift 2
      ;;
    --asset-endpoint)
      [[ $# -ge 2 ]] || die '--asset-endpoint requires a value'
      ASSET_ENDPOINT="$2"
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
case "${ASSET_MODE}" in
  auto|link|download|none) ;;
  *) die '--asset-mode must be auto, link, download, or none' ;;
esac
case "${SMOKE_LEVEL}" in
  libero-plus|imports|none) ;;
  *) die '--smoke-level must be libero-plus, imports, or none' ;;
esac
if [[ "${ASSET_MODE}" == "download" && -n "${ASSET_ROOT}" ]]; then
  die '--asset-root cannot be combined with --asset-mode download'
fi

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
FLUXVLA_ROOT="$(realpath -m "${FLUXVLA_ROOT}")"
[[ -z "${ASSET_ROOT}" ]] || ASSET_ROOT="$(realpath -m "${ASSET_ROOT}")"
[[ -n "${ASSET_DOWNLOAD_ROOT}" ]] || \
  ASSET_DOWNLOAD_ROOT="${SOURCE_ROOT}/libero-plus-assets"
ASSET_DOWNLOAD_ROOT="$(realpath -m "${ASSET_DOWNLOAD_ROOT}")"
ASSET_CACHE_DIR="$(realpath -m "${ASSET_CACHE_DIR}")"

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
    echo 'LIBERO-Plus system dependencies are already installed.'
    return
  fi
  if [[ "${SYSTEM_DEPS}" == "never" ]]; then
    echo "warning: missing system packages: ${missing[*]}" >&2
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
    die "missing system packages: ${missing[*]}; install them first"
  fi
}

bootstrap_pip() {
  if "${CONDA_BIN}" run -n "${ENV_NAME}" python -m pip --version \
    >/dev/null 2>&1; then
    return
  fi
  conda_run python -m ensurepip --upgrade
}

source_seed() {
  local local_path="$1"
  local ref="$2"
  local remote="$3"
  if [[ -d "${local_path}/.git" ]] && \
    git -C "${local_path}" cat-file -e "${ref}^{commit}" 2>/dev/null; then
    printf '%s\n' "${local_path}"
  else
    printf '%s\n' "${remote}"
  fi
}

prepare_checkout() {
  local name="$1"
  local ref="$2"
  local remote="$3"
  local seed_path="$4"
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

  local seed
  seed="$(source_seed "${seed_path}" "${ref}" "${remote}")"
  if [[ "${DRY_RUN}" == "1" ]]; then
    print_command mkdir -p "${SOURCE_ROOT}"
    print_command git clone --no-hardlinks --no-checkout "${seed}" "${target}"
    print_command git -C "${target}" checkout --detach "${ref}"
    CHECKOUT_PATH="${target}"
    return
  fi
  mkdir -p "${SOURCE_ROOT}"
  CLONE_TMP="$(mktemp -d "${SOURCE_ROOT}/.clone-${name}.XXXXXX")"
  run git clone --no-hardlinks --no-checkout "${seed}" "${CLONE_TMP}"
  run git -C "${CLONE_TMP}" checkout --detach "${ref}"
  [[ "$(git -C "${CLONE_TMP}" rev-parse HEAD)" == "${ref}" ]] || \
    die "failed to resolve ${name} at ${ref}"
  run mv "${CLONE_TMP}" "${target}"
  CLONE_TMP=""
  CHECKOUT_PATH="${target}"
}

assets_are_valid() {
  local assets="$1"
  [[ -d "${assets}" ]] || return 1
  "${CONDA_BIN}" run -n "${ENV_NAME}" python \
    "${SCRIPT_DIR}/libero_plus_assets.py" \
    --assets-dir "${assets}" --validate-only >/dev/null 2>&1
}

fluxvla_assets_are_reusable() {
  local checkout="${FLUXVLA_ROOT}/src/libero-plus"
  [[ -d "${checkout}/.git" ]] || return 1
  [[ "$(git -C "${checkout}" rev-parse HEAD 2>/dev/null || true)" == \
    "${LIBERO_PLUS_REF}" ]] || return 1
  assets_are_valid "${checkout}/libero_plus/libero/assets"
}

link_assets() {
  local source="$1"
  local destination="$2"
  local preserved="${SOURCE_ROOT}/libero-plus-base-assets"
  if [[ "${DRY_RUN}" == "0" ]]; then
    assets_are_valid "${source}" || \
      die "LIBERO-Plus asset root is incomplete: ${source}"
  fi
  if [[ -L "${destination}" ]]; then
    if [[ "$(realpath -m "${destination}")" == "${source}" ]]; then
      return
    fi
    run unlink "${destination}"
  elif [[ -e "${destination}" ]]; then
    [[ ! -e "${preserved}" ]] || \
      die "cannot preserve base assets because ${preserved} already exists"
    run mv "${destination}" "${preserved}"
  fi
  run ln -s "${source}" "${destination}"
}

ensure_system_dependencies
bootstrap_pip
if [[ "${PYTHON_VERSION}" == "3.10" ]]; then
  conda_run python -m pip install --upgrade pip 'setuptools==70.3.0' wheel
else
  conda_run python -m pip install --upgrade pip setuptools wheel
fi

if ! "${CONDA_BIN}" run -n "${ENV_NAME}" python -c \
  'import torch; version=tuple(int(x) for x in torch.__version__.split("+")[0].split(".")[:2]); raise SystemExit(version < (2, 0))' \
  >/dev/null 2>&1; then
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
      echo "warning: Torch install failed via ${torch_index}" >&2
    done
  fi
  [[ "${torch_installed}" == "1" ]] || \
    die 'failed to install CPU Torch from every configured index'
else
  echo 'Keeping the compatible Torch already installed in the target env.'
fi

PROJECT_EXTRAS='sim,real'
if [[ "${WITH_DEV}" == "1" ]]; then
  PROJECT_EXTRAS='sim,real,dev'
fi
conda_run python -m pip install -r \
  "${PROJECT_ROOT}/requirements-libero-plus.txt"
conda_run python -m pip install -e "${PROJECT_ROOT}[${PROJECT_EXTRAS}]"

prepare_checkout libero-plus "${LIBERO_PLUS_REF}" "${LIBERO_PLUS_REPO}" \
  "${FLUXVLA_ROOT}/src/libero-plus"
LIBERO_PLUS_DIR="${CHECKOUT_PATH}"
prepare_checkout robosuite "${ROBOSUITE_REF}" "${ROBOSUITE_REPO}" \
  "${FLUXVLA_ROOT}/src/robosuite"
ROBOSUITE_DIR="${CHECKOUT_PATH}"

conda_run python -m pip install --no-deps -e "${ROBOSUITE_DIR}"
conda_run python -m pip install --no-deps -e "${LIBERO_PLUS_DIR}"

PACKAGE_ASSETS="${LIBERO_PLUS_DIR}/libero_plus/libero/assets"
SELECTED_ASSETS=""
if [[ "${ASSET_MODE}" != "none" && -n "${ASSET_ROOT}" ]]; then
  if [[ "${DRY_RUN}" == "1" ]] || assets_are_valid "${ASSET_ROOT}"; then
    SELECTED_ASSETS="${ASSET_ROOT}"
  else
    die "explicit LIBERO-Plus asset root is incomplete: ${ASSET_ROOT}"
  fi
fi
if [[ "${ASSET_MODE}" == "auto" && -z "${SELECTED_ASSETS}" ]] && \
    assets_are_valid "${PACKAGE_ASSETS}"; then
  SELECTED_ASSETS="${PACKAGE_ASSETS}"
fi
if [[ "${ASSET_MODE}" == "auto" && -z "${SELECTED_ASSETS}" ]] && \
    fluxvla_assets_are_reusable; then
  SELECTED_ASSETS="${FLUXVLA_ROOT}/src/libero-plus/libero_plus/libero/assets"
  echo "Reusing FluxVLA LIBERO-Plus assets from ${SELECTED_ASSETS}."
fi
if [[ "${ASSET_MODE}" =~ ^(auto|link)$ && \
      -z "${SELECTED_ASSETS}" ]] && \
    assets_are_valid "${ASSET_DOWNLOAD_ROOT}"; then
  SELECTED_ASSETS="${ASSET_DOWNLOAD_ROOT}"
fi
if [[ "${ASSET_MODE}" == "link" && -z "${SELECTED_ASSETS}" ]]; then
  die '--asset-mode link requires --asset-root or complete existing assets'
fi
if [[ "${ASSET_MODE}" == "download" || \
      ( "${ASSET_MODE}" == "auto" && -z "${SELECTED_ASSETS}" ) ]]; then
  conda_run python "${SCRIPT_DIR}/libero_plus_assets.py" \
    --assets-dir "${ASSET_DOWNLOAD_ROOT}" \
    --cache-dir "${ASSET_CACHE_DIR}" \
    --endpoint "${ASSET_ENDPOINT}"
  SELECTED_ASSETS="${ASSET_DOWNLOAD_ROOT}"
fi
if [[ "${ASSET_MODE}" != "none" && \
      "${SELECTED_ASSETS}" != "${PACKAGE_ASSETS}" ]]; then
  link_assets "${SELECTED_ASSETS}" "${PACKAGE_ASSETS}"
fi

LIBERO_PLUS_CONFIG_DIR="${ENV_PREFIX}/etc/fluxthemis/libero-plus"
ACTIVATION_DIR="${ENV_PREFIX}/etc/conda/activate.d"
DEACTIVATION_DIR="${ENV_PREFIX}/etc/conda/deactivate.d"
run mkdir -p "${LIBERO_PLUS_CONFIG_DIR}" "${ACTIVATION_DIR}" \
  "${DEACTIVATION_DIR}"
run install -m 0644 "${PROJECT_ROOT}/scripts/activate_libero_plus.sh" \
  "${ACTIVATION_DIR}/fluxthemis-libero-plus.sh"
run install -m 0644 "${PROJECT_ROOT}/scripts/deactivate_libero_plus.sh" \
  "${DEACTIVATION_DIR}/fluxthemis-libero-plus.sh"
if [[ "${ASSET_MODE}" == "none" ]]; then
  conda_run python "${SCRIPT_DIR}/libero_plus_assets.py" \
    --write-config-only --checkout-root "${LIBERO_PLUS_DIR}" \
    --config-dir "${LIBERO_PLUS_CONFIG_DIR}"
else
  conda_run python "${SCRIPT_DIR}/libero_plus_assets.py" \
    --assets-dir "${PACKAGE_ASSETS}" --validate-only \
    --checkout-root "${LIBERO_PLUS_DIR}" \
    --config-dir "${LIBERO_PLUS_CONFIG_DIR}"
fi

if [[ "${DRY_RUN}" == "1" ]]; then
  echo 'Dry run complete; no installation or smoke test was performed.'
  exit 0
fi

SMOKE_ENV=(
  env
  "LIBERO_PLUS_CONFIG_PATH=${LIBERO_PLUS_CONFIG_DIR}"
  "LIBERO_CONFIG_PATH=${LIBERO_PLUS_CONFIG_DIR}"
  MUJOCO_GL=egl
  PYOPENGL_PLATFORM=egl
)
conda_run python -m pip check
[[ "$(git -C "${LIBERO_PLUS_DIR}" rev-parse HEAD)" == \
  "${LIBERO_PLUS_REF}" ]]
[[ "$(git -C "${ROBOSUITE_DIR}" rev-parse HEAD)" == \
  "${ROBOSUITE_REF}" ]]
conda_run "${SMOKE_ENV[@]}" fluxthemis-eval --help >/dev/null

case "${SMOKE_LEVEL}" in
  imports)
    conda_run "${SMOKE_ENV[@]}" python \
      "${PROJECT_ROOT}/scripts/libero_plus_smoke.py" --imports-only
    ;;
  libero-plus)
    conda_run "${SMOKE_ENV[@]}" python \
      "${PROJECT_ROOT}/scripts/libero_plus_smoke.py" \
      --suite libero_10 --task-id 0 --resolution 64 --steps 1
    ;;
  none) ;;
esac

cat <<EOF

LIBERO-Plus installation complete.
  conda env:    ${ENV_NAME}
  source root:  ${SOURCE_ROOT}
  asset root:   ${SELECTED_ASSETS:-not configured}
  config path:  ${LIBERO_PLUS_CONFIG_DIR}

Activate it with:
  conda activate ${ENV_NAME}
EOF
