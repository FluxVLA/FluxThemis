#!/usr/bin/env bash
# Add LIBERO-Plus to an existing LIBERO simulator environment.

set -euo pipefail

LIBERO_PLUS_REF="d7c160eca38bf268d5965aef6d4c6dca12d5537c"
LIBERO_PLUS_REPO="https://github.com/hqr-robotic/LIBERO-plus.git"
SYSTEM_PACKAGES=(libexpat1 libfontconfig1-dev libmagickwand-dev)

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
ENV_NAME="${FLUXTHEMIS_CONDA_ENV:-fluxthemis-libero}"
FLUXVLA_ROOT="${FLUXVLA_ROOT:-/root/projects/FluxVLA}"
SOURCE_ROOT=""
SYSTEM_DEPS="auto"
SMOKE_LEVEL="libero-plus"
ASSET_MODE="auto"
ASSET_ROOT=""
ASSET_DOWNLOAD_ROOT=""
ASSET_CACHE_DIR="${FLUXTHEMIS_LIBERO_PLUS_ASSET_CACHE:-/tmp/libero-plus-assets}"
ASSET_ENDPOINT="${HF_ENDPOINT:-https://huggingface.co}"
DRY_RUN=0
CLONE_TMP=""
CHECKOUT_PATH=""

usage() {
  cat <<'EOF'
Usage: bash scripts/install_libero_plus.sh [options]

Adds LIBERO-Plus to an existing Python 3.10 or 3.12 LIBERO Conda environment.
Run scripts/install_libero.sh first if the LIBERO runtime is not installed.
Only Plus dependencies, source, assets and configuration are installed.

Options:
  --env-name NAME              Existing LIBERO env (default: fluxthemis-libero).
  --source-root PATH           Managed source directory (default:
                               <conda-prefix>/src/fluxthemis).
  --fluxvla-root PATH          Optional existing Plus source/asset seed.
  --asset-mode auto|link|download|none
                               Asset strategy (default: auto).
  --asset-root PATH            Complete local LIBERO-Plus asset directory.
  --asset-download-root PATH   Extraction destination (default:
                               <source-root>/libero-plus-assets).
  --asset-cache-dir PATH       Official assets.zip cache (default:
                               /tmp/libero-plus-assets).
  --asset-endpoint URL         Hugging Face endpoint (default: HF_ENDPOINT or
                               https://huggingface.co).
  --system-deps auto|always|never
                               Install Plus system packages (default: auto).
  --smoke-level libero-plus|imports|none
                               Check both benchmarks in separate processes
                               (default: libero-plus, resets and steps each).
  --dry-run                    Preview commands; requires an existing LIBERO env.
  -h, --help                   Show this help.

Examples:
  bash scripts/install_libero_plus.sh
  bash scripts/install_libero_plus.sh --asset-mode link \
    --asset-root /path/to/libero-plus/assets
  bash scripts/install_libero_plus.sh --smoke-level imports
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
# Check package availability without importing simulator registries or creating
# the standard LIBERO configuration. The base installer owns that environment.
if ! "${CONDA_BIN}" run -n "${ENV_NAME}" python -c '
from importlib.util import find_spec
required = ("pip", "fluxthemis", "libero", "robosuite", "torch")
missing = [name for name in required if find_spec(name) is None]
if missing:
    raise SystemExit("Missing LIBERO prerequisites: " + ", ".join(missing))
'; then
  die "install the base runtime first: bash scripts/install_libero.sh --env-name ${ENV_NAME@Q}"
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
  if [[ -e "${target}" ]]; then
    [[ -d "${target}/.git" ]] || \
      die "managed source path is not a git checkout: ${target}"
    local source_status status_line
    source_status="$(git -C "${target}" status --porcelain)"
    while IFS= read -r status_line; do
      [[ -n "${status_line}" ]] || continue
      # Upstream ignores the assets directory, but not the symlink created
      # by this installer. Asset contents are validated later in the install.
      if [[ "${name}" == "libero-plus" && \
            "${status_line}" == '?? libero_plus/libero/assets' && \
            -L "${target}/libero_plus/libero/assets" ]]; then
        continue
      fi
      die "managed source checkout is dirty: ${target}"
    done <<< "${source_status}"
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
conda_run python -m pip install -r "${PROJECT_ROOT}/requirements-libero-plus.txt"

prepare_checkout libero-plus "${LIBERO_PLUS_REF}" "${LIBERO_PLUS_REPO}" \
  "${FLUXVLA_ROOT}/src/libero-plus"
LIBERO_PLUS_DIR="${CHECKOUT_PATH}"
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
run mkdir -p "${LIBERO_PLUS_CONFIG_DIR}"
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

conda_run python -m pip check
conda_run python "${PROJECT_ROOT}/scripts/distributed_eval.py" --help >/dev/null

if [[ "${SMOKE_LEVEL}" != "none" ]]; then
  for benchmark in libero libero-plus; do
    config_dir="${ENV_PREFIX}/etc/fluxthemis/${benchmark}"
    if [[ "${benchmark}" == "libero-plus" ]]; then
      smoke_script="${SCRIPT_DIR}/libero_plus_smoke.py"
    else
      smoke_script="${SCRIPT_DIR}/libero_smoke.py"
    fi
    smoke_args=()
    if [[ "${SMOKE_LEVEL}" == "imports" ]]; then
      smoke_args+=(--imports-only)
    else
      smoke_args+=(--suite libero_10 --task-id 0 --resolution 64 --steps 1)
    fi
    conda_run env "LIBERO_CONFIG_PATH=${config_dir}" \
      "LIBERO_PLUS_CONFIG_PATH=${ENV_PREFIX}/etc/fluxthemis/libero-plus" \
      MUJOCO_GL=egl PYOPENGL_PLATFORM=egl \
      python "${smoke_script}" "${smoke_args[@]}"
  done
fi

cat <<EOF

LIBERO-Plus incremental installation complete.
  conda env:       ${ENV_NAME}
  source root:     ${SOURCE_ROOT}
  asset root:      ${SELECTED_ASSETS:-not configured}
  Plus config:     ${LIBERO_PLUS_CONFIG_DIR}

Activate it with:
  conda activate ${ENV_NAME}
EOF
