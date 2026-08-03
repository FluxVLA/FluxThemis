#!/usr/bin/env bash
# Install FluxThemis plus the ROS runtime matching each Conda interpreter.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
ENV_NAME="${FLUXTHEMIS_CONDA_ENV:-fluxthemis}"
ENV_NAME_EXPLICIT=0
[[ -z "${FLUXTHEMIS_CONDA_ENV:-}" ]] || ENV_NAME_EXPLICIT=1
ROS2_ENV_NAME="${FLUXTHEMIS_ROS2_CONDA_ENV:-fluxthemis-ros2}"
FLUXVLA_ENV_NAME="${FLUXVLA_CONDA_ENV:-fluxvla}"
FLUXVLA_ROOT="${FLUXVLA_ROOT:-/root/projects/FluxVLA}"
ROS_MODE="auto"
ROS1_WORKSPACE="${FLUXTHEMIS_ROS1_WORKSPACE:-/root/fluxthemis_ros1_ws}"
ROS2_WORKSPACE="${FLUXTHEMIS_ROS2_WORKSPACE:-/root/fluxthemis_ros2_ws}"
SMOKE_LEVEL="libero"
SMOKE_LEVEL_EXPLICIT=0
SIMULATOR="libero"
ROBOCASA_ASSET_MODE="auto"
ROBOCASA_ASSET_ROOT=""
INSTALL_FLUXVLA_HOOK=1
FLUXVLA_HOOK_STATUS='not-applicable'
WITH_DEV=0
DRY_RUN=0

usage() {
  cat <<'EOF'
Usage: bash scripts/install.sh [options]

Installs FluxThemis plus one simulator into an existing Conda environment and
selects a ROS runtime from the interpreter ABI. Python 3.10 selects source-
built ROS 1 Noetic; Python 3.12 selects system ROS 2 Jazzy. Keep LIBERO and
RoboCasa in separate Conda environments.

Options:
  --env-name NAME          Primary environment (default: fluxthemis, or
                           fluxthemis-robocasa for the RoboCasa profile).
  --ros auto|1|2|both|none ROS installation mode (default: auto).
  --ros2-env-name NAME     Python 3.12 environment used by `--ros both`
                           (default: fluxthemis-ros2).
  --fluxvla-env-name NAME  Python 3.10 FluxVLA environment that receives the
                           ROS 1 activation hook (default: fluxvla).
  --fluxvla-root PATH      FluxVLA checkout (default: /root/projects/FluxVLA).
  --simulator NAME         Simulator profile: libero, robocasa, or none
                           (default: libero).
  --robocasa-asset-mode MODE
                           RoboCasa assets: auto, link, download, or none.
  --robocasa-asset-root PATH
                           Existing complete RoboCasa models/assets directory.
  --ros1-workspace PATH    ROS 1 source workspace.
  --ros2-workspace PATH    ROS 2 interface workspace.
  --smoke-level LEVEL      LIBERO verification: libero, imports, or none.
  --skip-libero            Compatibility alias for --simulator none.
  --skip-fluxvla-hook      Do not configure ROS 1 activation for FluxVLA.
  --with-dev               Install FluxThemis development dependencies.
  --dry-run                Print mutating installer commands.
  -h, --help               Show this help.

Examples:
  # Current profile: Python 3.10 -> ROS 1.
  bash scripts/install.sh --env-name fluxthemis --ros auto

  # Optional profile: an existing Python 3.12 env -> ROS 2.
  bash scripts/install.sh --env-name fluxthemis-ros2 --ros auto

  # Dedicated RoboCasa profile: Python 3.10 -> ROS 1.
  bash scripts/install.sh --simulator robocasa --ros auto

  # Optional convenience when both environments already exist.
  bash scripts/install.sh --ros both --ros2-env-name fluxthemis-ros2
EOF
}

die() {
  echo "error: $*" >&2
  exit 1
}

env_python_version() {
  local name="$1"
  "${CLEAN_CONDA_ENV[@]}" "${CONDA_BIN}" run -n "${name}" python -c \
    'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")' \
    2>/dev/null || true
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --env-name)
      [[ $# -ge 2 ]] || die '--env-name requires a value'
      ENV_NAME="$2"
      ENV_NAME_EXPLICIT=1
      shift 2
      ;;
    --ros)
      [[ $# -ge 2 ]] || die '--ros requires a value'
      ROS_MODE="$2"
      shift 2
      ;;
    --ros2-env-name)
      [[ $# -ge 2 ]] || die '--ros2-env-name requires a value'
      ROS2_ENV_NAME="$2"
      shift 2
      ;;
    --fluxvla-env-name)
      [[ $# -ge 2 ]] || die '--fluxvla-env-name requires a value'
      FLUXVLA_ENV_NAME="$2"
      shift 2
      ;;
    --fluxvla-root)
      [[ $# -ge 2 ]] || die '--fluxvla-root requires a value'
      FLUXVLA_ROOT="$2"
      shift 2
      ;;
    --simulator)
      [[ $# -ge 2 ]] || die '--simulator requires a value'
      SIMULATOR="$2"
      shift 2
      ;;
    --robocasa-asset-mode)
      [[ $# -ge 2 ]] || die '--robocasa-asset-mode requires a value'
      ROBOCASA_ASSET_MODE="$2"
      shift 2
      ;;
    --robocasa-asset-root)
      [[ $# -ge 2 ]] || die '--robocasa-asset-root requires a value'
      ROBOCASA_ASSET_ROOT="$2"
      shift 2
      ;;
    --ros1-workspace)
      [[ $# -ge 2 ]] || die '--ros1-workspace requires a value'
      ROS1_WORKSPACE="$2"
      shift 2
      ;;
    --ros2-workspace)
      [[ $# -ge 2 ]] || die '--ros2-workspace requires a value'
      ROS2_WORKSPACE="$2"
      shift 2
      ;;
    --smoke-level)
      [[ $# -ge 2 ]] || die '--smoke-level requires a value'
      SMOKE_LEVEL="$2"
      SMOKE_LEVEL_EXPLICIT=1
      shift 2
      ;;
    --skip-libero)
      SIMULATOR=none
      shift
      ;;
    --skip-fluxvla-hook)
      INSTALL_FLUXVLA_HOOK=0
      shift
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

case "${ROS_MODE}" in
  auto|1|2|both|none) ;;
  *) die '--ros must be auto, 1, 2, both, or none' ;;
esac
case "${SIMULATOR}" in
  libero|robocasa|none) ;;
  *) die '--simulator must be libero, robocasa, or none' ;;
esac
case "${ROBOCASA_ASSET_MODE}" in
  auto|link|download|none) ;;
  *) die '--robocasa-asset-mode must be auto, link, download, or none' ;;
esac
if [[ "${SIMULATOR}" == "robocasa" && \
      "${ENV_NAME_EXPLICIT}" == "0" ]]; then
  ENV_NAME="fluxthemis-robocasa"
fi
if [[ "${SIMULATOR}" == "libero" ]]; then
  case "${SMOKE_LEVEL}" in
    libero|imports|none) ;;
    *) die '--smoke-level must be libero, imports, or none' ;;
  esac
elif [[ "${SIMULATOR}" == "robocasa" ]]; then
  [[ "${SMOKE_LEVEL_EXPLICIT}" == "0" ]] || \
    die '--smoke-level is only available with --simulator libero'
else
  SMOKE_LEVEL=none
fi

CONDA_BIN="$(command -v conda || true)"
[[ -n "${CONDA_BIN}" ]] || die 'conda is not available on PATH'
CLEAN_CONDA_ENV=(
  env
  -u AMENT_PREFIX_PATH
  -u CATKIN_PREFIX_PATH
  -u CMAKE_PREFIX_PATH
  -u COLCON_PREFIX_PATH
  -u LD_LIBRARY_PATH
  -u PKG_CONFIG_PATH
  -u PYTHONHOME
  -u PYTHONPATH
  -u ROS_DISTRO
  -u ROS_PACKAGE_PATH
  -u ROS_ROOT
  -u ROS_VERSION
)
PRIMARY_PYTHON="$(env_python_version "${ENV_NAME}")"
[[ -n "${PRIMARY_PYTHON}" ]] || \
  die "Conda environment ${ENV_NAME@Q} does not exist"

if [[ "${ROS_MODE}" == "auto" ]]; then
  case "${PRIMARY_PYTHON}" in
    3.10) ROS_MODE=1 ;;
    3.12) ROS_MODE=2 ;;
    *)
      die "auto ROS selection supports Python 3.10 or 3.12; found ${PRIMARY_PYTHON}"
      ;;
  esac
  echo "Auto-selected ROS ${ROS_MODE} for Python ${PRIMARY_PYTHON}."
fi

case "${ROS_MODE}" in
  1)
    [[ "${PRIMARY_PYTHON}" == "3.10" ]] || \
      die "ROS 1 mode requires ${ENV_NAME@Q} to use Python 3.10"
    ;;
  2)
    [[ "${PRIMARY_PYTHON}" == "3.12" ]] || \
      die "Jazzy mode requires ${ENV_NAME@Q} to use Python 3.12"
    ;;
  both)
    [[ "${PRIMARY_PYTHON}" == "3.10" ]] || \
      die "dual mode requires ${ENV_NAME@Q} to use Python 3.10"
    ROS2_PYTHON="$(env_python_version "${ROS2_ENV_NAME}")"
    [[ "${ROS2_PYTHON}" == "3.12" ]] || \
      die "dual mode requires ${ROS2_ENV_NAME@Q} to use Python 3.12"
    ;;
esac

if [[ "${SIMULATOR}" == "libero" ]]; then
  libero_args=(
    --env-name "${ENV_NAME}"
    --fluxvla-root "${FLUXVLA_ROOT}"
    --smoke-level "${SMOKE_LEVEL}"
  )
  [[ "${WITH_DEV}" == "0" ]] || libero_args+=(--with-dev)
  [[ "${DRY_RUN}" == "0" ]] || libero_args+=(--dry-run)
  bash "${SCRIPT_DIR}/install_libero.sh" "${libero_args[@]}"
elif [[ "${SIMULATOR}" == "robocasa" ]]; then
  robocasa_args=(
    --env-name "${ENV_NAME}"
    --fluxvla-root "${FLUXVLA_ROOT}"
    --asset-mode "${ROBOCASA_ASSET_MODE}"
  )
  if [[ -n "${ROBOCASA_ASSET_ROOT}" ]]; then
    robocasa_args+=(--asset-root "${ROBOCASA_ASSET_ROOT}")
  fi
  [[ "${WITH_DEV}" == "0" ]] || robocasa_args+=(--with-dev)
  [[ "${DRY_RUN}" == "0" ]] || robocasa_args+=(--dry-run)
  bash "${SCRIPT_DIR}/install_robocasa.sh" "${robocasa_args[@]}"
fi

if [[ "${ROS_MODE}" == "1" || "${ROS_MODE}" == "both" ]]; then
  ros1_args=(--env-name "${ENV_NAME}" --workspace "${ROS1_WORKSPACE}")
  [[ "${DRY_RUN}" == "0" ]] || ros1_args+=(--dry-run)
  bash "${SCRIPT_DIR}/install_ros1_noetic_py310.sh" "${ros1_args[@]}"

  if [[ "${INSTALL_FLUXVLA_HOOK}" == "1" ]]; then
    fluxvla_hook_args=(
      --env-name "${FLUXVLA_ENV_NAME}"
      --workspace "${ROS1_WORKSPACE}"
    )
    [[ "${DRY_RUN}" == "0" ]] || fluxvla_hook_args+=(--dry-run)
    bash "${SCRIPT_DIR}/install_ros1_hook.sh" \
      "${fluxvla_hook_args[@]}"
    if [[ "${DRY_RUN}" == "1" ]]; then
      FLUXVLA_HOOK_STATUS='planned'
    else
      FLUXVLA_HOOK_STATUS='installed'
    fi
  else
    FLUXVLA_HOOK_STATUS='skipped'
  fi
fi

if [[ "${ROS_MODE}" == "2" || "${ROS_MODE}" == "both" ]]; then
  ros2_args=(--workspace "${ROS2_WORKSPACE}")
  [[ "${DRY_RUN}" == "0" ]] || ros2_args+=(--dry-run)
  bash "${SCRIPT_DIR}/install_ros2_jazzy.sh" "${ros2_args[@]}"

  ROS2_TARGET_ENV="${ENV_NAME}"
  [[ "${ROS_MODE}" != "both" ]] || ROS2_TARGET_ENV="${ROS2_ENV_NAME}"
  ROS2_ENV_PREFIX="$("${CLEAN_CONDA_ENV[@]}" "${CONDA_BIN}" run \
    -n "${ROS2_TARGET_ENV}" \
    python -c 'import sys; print(sys.prefix)')"
  if [[ "${DRY_RUN}" == "1" ]]; then
    printf '+ %q run -n %q python -m pip install -e %q %q\n' \
      "${CONDA_BIN}" "${ROS2_TARGET_ENV}" "${PROJECT_ROOT}[real]" \
      'lark==1.1.9' >&2
    echo "+ install ROS 2 activation hook into ${ROS2_ENV_PREFIX}" >&2
    echo "+ remove any stale ROS 1 activation hook from ${ROS2_ENV_PREFIX}" >&2
  else
    "${CLEAN_CONDA_ENV[@]}" "${CONDA_BIN}" run --no-capture-output \
      -n "${ROS2_TARGET_ENV}" python -m pip install -e \
      "${PROJECT_ROOT}[real]" 'lark==1.1.9'
    mkdir -p "${ROS2_ENV_PREFIX}/etc/conda/activate.d" \
      "${ROS2_ENV_PREFIX}/etc/conda/deactivate.d" \
      "${ROS2_ENV_PREFIX}/etc/fluxthemis"
    # Remove hooks left by an earlier ROS 1 use of the same environment.
    rm -f -- \
      "${ROS2_ENV_PREFIX}/etc/conda/activate.d/fluxthemis-ros1.sh" \
      "${ROS2_ENV_PREFIX}/etc/conda/deactivate.d/fluxthemis-ros1.sh"
    install -m 0644 "${SCRIPT_DIR}/activate_ros2.sh" \
      "${ROS2_ENV_PREFIX}/etc/conda/activate.d/fluxthemis-ros2.sh"
    install -m 0644 "${SCRIPT_DIR}/deactivate_ros.sh" \
      "${ROS2_ENV_PREFIX}/etc/conda/deactivate.d/fluxthemis-ros2.sh"
    printf '%s\n' "${ROS2_WORKSPACE}" \
      > "${ROS2_ENV_PREFIX}/etc/fluxthemis/ros2-workspace"
    CONDA_BASE="$("${CONDA_BIN}" info --base)"
    env -u CONDA_PREFIX -u CONDA_DEFAULT_ENV -u PYTHONPATH \
      -u AMENT_PREFIX_PATH -u CMAKE_PREFIX_PATH -u COLCON_PREFIX_PATH \
      -u CATKIN_PREFIX_PATH -u ROS_PACKAGE_PATH -u ROS_ROOT \
      -u ROS_MASTER_URI -u ROS_DISTRO -u ROS_VERSION \
      /usr/bin/bash --noprofile --norc -c '
set -euo pipefail
source "$1/etc/profile.d/conda.sh"
conda activate "$2"
source /opt/ros/jazzy/setup.bash
source "$3/install/setup.bash"
python -c "import cv_bridge, lark, launch, rclpy; from fluxthemis_msgs.srv import PredictAction, ReportEvaluation; from fluxthemis.model_clients.fluxvla_ros2 import FluxVLAROS2ModelClient; print(PredictAction, ReportEvaluation, FluxVLAROS2ModelClient)"
python -m pip check
' -- "${CONDA_BASE}" "${ROS2_TARGET_ENV}" "${ROS2_WORKSPACE}"
  fi
fi

VERIFY_COMMAND="bash ${SCRIPT_DIR}/check_env.sh --env-name ${ENV_NAME} --ros ${ROS_MODE}"
if [[ "${ROS_MODE}" == "both" ]]; then
  VERIFY_COMMAND+=" --ros2-env-name ${ROS2_ENV_NAME}"
fi
if [[ "${SIMULATOR}" == "robocasa" ]]; then
  VERIFY_COMMAND+=" --smoke-level none"
fi

cat <<EOF

FluxThemis installation complete.
  primary env: ${ENV_NAME} (Python ${PRIMARY_PYTHON})
  simulator:   ${SIMULATOR}
  ROS mode:    ${ROS_MODE}
  FluxVLA env: ${FLUXVLA_ENV_NAME} (ROS 1 hook: ${FLUXVLA_HOOK_STATUS})

Only the requested profile(s) were processed. ROS 1 and ROS 2 remain separate
Python ABI profiles; never source both workspaces in one process.

Verify the installed runtime(s) with:
  ${VERIFY_COMMAND}
EOF
