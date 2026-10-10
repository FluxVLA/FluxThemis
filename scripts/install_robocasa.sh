#!/usr/bin/env bash
# Install the environment-side RoboCasa runtime into an existing conda env.

set -euo pipefail

ROBOCASA_REF="4840e671596f93ca03651524b9f72ffb1aadfeff"
ROBOSUITE_REF="e293cc32ff3c48957a4ebcad09952432b0dc9049"
ROBOCASA_REPO="https://github.com/robocasa/robocasa-gr1-tabletop-tasks.git"
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
REQUIRED_ASSET_PATHS=(
  fixtures
  textures
  generative_textures
  objects/objaverse
  objects/sketchfab
  objects/lightwheel
  fixtures/toasters/basic_popup_2/visuals/model_0.obj
)

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
ENV_NAME="${FLUXTHEMIS_CONDA_ENV:-fluxthemis-robocasa}"
SOURCE_ROOT=""
SYSTEM_DEPS="auto"
ASSET_MODE="auto"
ASSET_ROOT=""
ASSET_DOWNLOAD_ROOT=""
ASSET_CACHE_DIR="${FLUXTHEMIS_ROBOCASA_ASSET_CACHE:-/tmp/robocasa-assets}"
ASSET_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
TORCH_VERSION="${FLUXTHEMIS_TORCH_VERSION:-2.8.0}"
TORCH_INDEX_URLS="${FLUXTHEMIS_TORCH_INDEX_URLS:-https://download.pytorch.org/whl/cpu}"
TORCH_WHEEL_URL="${FLUXTHEMIS_TORCH_WHEEL_URL-}"
WITH_DEV=0
DRY_RUN=0
CLONE_TMP=""
CHECKOUT_PATH=""

usage() {
  cat <<'EOF'
Usage: bash scripts/install_robocasa.sh [options]

Installs RoboCasa into an existing Python 3.10 or 3.12 conda environment.
The default environment is `fluxthemis-robocasa`; it is never deleted or
recreated.

Options:
  --env-name NAME              Target environment (default:
                               fluxthemis-robocasa).
  --source-root PATH           Managed RoboCasa/robosuite checkout directory
                               (default: <conda-prefix>/src/fluxthemis).
  --asset-mode auto|link|download|none
                               Asset strategy (default: auto). Auto links an
                               explicit asset root or downloads a managed copy.
  --asset-root PATH            Existing complete RoboCasa models/assets
                               directory to link into the managed checkout.
  --asset-download-root PATH   Shared download destination (default:
                               <source-root>/robocasa-assets).
  --asset-cache-dir PATH       Hugging Face archive cache (default:
                               /tmp/robocasa-assets).
  --asset-endpoint URL         Hugging Face endpoint (default: HF_ENDPOINT or
                               https://hf-mirror.com).
  --system-deps auto|always|never
                               Install missing EGL packages when possible
                               (default: auto).
  --torch-version VERSION      CPU Torch version installed when absent
                               (default: 2.8.0).
  --with-dev                   Install FluxThemis development dependencies.
  --dry-run                    Print mutating commands without running them.
  -h, --help                   Show this help.

Examples:
  bash scripts/install_robocasa.sh
  bash scripts/install_robocasa.sh --asset-root /data/robocasa-assets
  bash scripts/install_robocasa.sh --asset-mode download
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
[[ -z "${ASSET_ROOT}" ]] || ASSET_ROOT="$(realpath -m "${ASSET_ROOT}")"
[[ -n "${ASSET_DOWNLOAD_ROOT}" ]] || \
  ASSET_DOWNLOAD_ROOT="${SOURCE_ROOT}/robocasa-assets"
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

robocasa_managed_status_is_safe() {
  local target="$1"
  local line
  while IFS= read -r line; do
    [[ -z "${line}" ]] && continue
    case "${line:3}" in
      robocasa/models/assets|robocasa/models/assets/*) ;;
      *) return 1 ;;
    esac
  done < <(git -C "${target}" status --porcelain)
  return 0
}

prepare_checkout() {
  local name="$1"
  local ref="$2"
  local remote="$3"
  local allow_asset_link="${4:-0}"
  local target="${SOURCE_ROOT}/${name}"
  local head
  if [[ -e "${target}" ]]; then
    [[ -d "${target}/.git" ]] || \
      die "managed source path is not a git checkout: ${target}"
    head="$(git -C "${target}" rev-parse HEAD)"
    [[ "${head}" == "${ref}" ]] || \
      die "${target} is at ${head}, expected ${ref}; move it aside and rerun"
    if [[ -n "$(git -C "${target}" status --porcelain)" ]]; then
      if [[ "${allow_asset_link}" != "1" ]] || \
          [[ ! -L "${target}/robocasa/models/assets" ]] || \
          ! robocasa_managed_status_is_safe "${target}"; then
        die "managed source checkout is dirty: ${target}"
      fi
    fi
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

assets_are_valid() {
  local root="$1"
  local path
  [[ -d "${root}" ]] || return 1
  for path in "${REQUIRED_ASSET_PATHS[@]}"; do
    [[ -e "${root}/${path}" ]] || return 1
  done
  return 0
}

download_assets() {
  local destination="$1"
  local base_assets="$2"
  local downloader="${PROJECT_ROOT}/scripts/download_robocasa_assets.py"
  if [[ ! -d "${destination}" ]]; then
    run mkdir -p "${destination}"
    run cp -a "${base_assets}/." "${destination}/"
  fi
  conda_run python "${downloader}" \
    --assets-dir "${destination}" \
    --cache-dir "${ASSET_CACHE_DIR}" \
    --endpoint "${ASSET_ENDPOINT}"
  if [[ "${DRY_RUN}" == "0" ]]; then
    assets_are_valid "${destination}" || \
      die "downloaded RoboCasa assets are incomplete: ${destination}"
  fi
}

link_assets() {
  local checkout="$1"
  local asset_source="$2"
  local package_assets="${checkout}/robocasa/models/assets"
  local base_assets="${SOURCE_ROOT}/robocasa-base-assets"
  case "${asset_source}" in
    "${package_assets}"|"${package_assets}"/*)
      die '--asset-root must be outside the managed RoboCasa assets path'
      ;;
  esac
  if [[ "${DRY_RUN}" == "0" ]]; then
    assets_are_valid "${asset_source}" || \
      die "RoboCasa asset root is incomplete: ${asset_source}"
  fi
  if [[ "${DRY_RUN}" == "1" && ! -e "${package_assets}" && \
        ! -L "${package_assets}" ]]; then
    print_command mv "${package_assets}" "${base_assets}"
    print_command ln -s "${asset_source}" "${package_assets}"
    return
  fi
  if [[ -L "${package_assets}" ]]; then
    if [[ "$(realpath -m "${package_assets}")" == "${asset_source}" ]]; then
      return
    fi
    run unlink "${package_assets}"
  elif [[ -e "${package_assets}" ]]; then
    [[ ! -e "${base_assets}" ]] || die \
      "cannot preserve base assets because ${base_assets} already exists"
    run mv "${package_assets}" "${base_assets}"
  fi
  run ln -s "${asset_source}" "${package_assets}"
}

ensure_system_dependencies
bootstrap_pip
if [[ "${PYTHON_VERSION}" == "3.10" ]]; then
  # Noetic's catkin setup helpers use APIs removed by newer setuptools.
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
conda_run python -m pip install -r \
  "${PROJECT_ROOT}/requirements-robocasa.txt"
conda_run python -m pip install -e "${PROJECT_SPEC}"

prepare_checkout robocasa-gr1-tabletop-tasks "${ROBOCASA_REF}" \
  "${ROBOCASA_REPO}" 1
ROBOCASA_DIR="${CHECKOUT_PATH}"
prepare_checkout robosuite "${ROBOSUITE_REF}" "${ROBOSUITE_REPO}"
ROBOSUITE_DIR="${CHECKOUT_PATH}"

PACKAGE_ASSETS="${ROBOCASA_DIR}/robocasa/models/assets"
BASE_ASSETS="${SOURCE_ROOT}/robocasa-base-assets"
SELECTED_ASSETS=""
if [[ "${ASSET_MODE}" != "none" && -n "${ASSET_ROOT}" ]]; then
  if [[ "${DRY_RUN}" == "1" ]] || assets_are_valid "${ASSET_ROOT}"; then
    SELECTED_ASSETS="${ASSET_ROOT}"
  else
    die "explicit RoboCasa asset root is incomplete: ${ASSET_ROOT}"
  fi
fi
if [[ "${ASSET_MODE}" == "link" && -z "${SELECTED_ASSETS}" ]]; then
  die '--asset-mode link requires --asset-root'
fi
if [[ "${ASSET_MODE}" == "download" || \
      ( "${ASSET_MODE}" == "auto" && -z "${SELECTED_ASSETS}" ) ]]; then
  DOWNLOAD_BASE="${PACKAGE_ASSETS}"
  [[ -d "${DOWNLOAD_BASE}" ]] || DOWNLOAD_BASE="${BASE_ASSETS}"
  [[ -d "${DOWNLOAD_BASE}" || "${DRY_RUN}" == "1" ]] || \
    die 'the managed checkout has no base RoboCasa assets'
  download_assets "${ASSET_DOWNLOAD_ROOT}" "${DOWNLOAD_BASE}"
  SELECTED_ASSETS="${ASSET_DOWNLOAD_ROOT}"
fi
if [[ "${ASSET_MODE}" != "none" ]]; then
  link_assets "${ROBOCASA_DIR}" "${SELECTED_ASSETS}"
fi

conda_run python -m pip install --no-deps -e "${ROBOSUITE_DIR}"
conda_run python -m pip install --no-deps -e "${ROBOCASA_DIR}"

if [[ "${DRY_RUN}" == "1" ]]; then
  echo 'Dry run complete; no installation was performed.'
  exit 0
fi

conda_run python -m pip check
[[ "$(git -C "${ROBOCASA_DIR}" rev-parse HEAD)" == "${ROBOCASA_REF}" ]]
[[ "$(git -C "${ROBOSUITE_DIR}" rev-parse HEAD)" == "${ROBOSUITE_REF}" ]]
conda_run python "${PROJECT_ROOT}/scripts/distributed_eval.py" --help \
  >/dev/null

cat <<EOF

RoboCasa installation complete.
  conda env:    ${ENV_NAME}
  source root:  ${SOURCE_ROOT}
  asset root:   ${SELECTED_ASSETS:-not configured}

Activate it with:
  conda activate ${ENV_NAME}
EOF
