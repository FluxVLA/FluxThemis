#!/usr/bin/env bash
# Install the existing FluxThemis ROS 1 overlay hook into a Python 3.10
# consumer environment such as FluxVLA. This does not install or upgrade any
# Python package in that environment.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ENV_NAME="${FLUXVLA_CONDA_ENV:-fluxvla}"
WORKSPACE="${FLUXTHEMIS_ROS1_WORKSPACE:-/root/fluxthemis_ros1_ws}"
DRY_RUN=0

usage() {
  cat <<'EOF'
Usage: bash scripts/install_ros1_hook.sh [options]

Installs the FluxThemis ROS 1 activation/deactivation hooks into an existing
Python 3.10 Conda environment. No package is installed into that environment.

Options:
  --env-name NAME       Consumer Conda environment (default: fluxvla).
  --workspace PATH      ROS 1 workspace (default: /root/fluxthemis_ros1_ws).
  --dry-run             Print changes without applying them.
  -h, --help            Show this help.
EOF
}

die() {
  echo "error: $*" >&2
  exit 1
}

run() {
  if [[ "${DRY_RUN}" == "1" ]]; then
    printf '+' >&2
    printf ' %q' "$@" >&2
    printf '\n' >&2
  else
    "$@"
  fi
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --env-name)
      [[ $# -ge 2 ]] || die '--env-name requires a value'
      ENV_NAME="$2"
      shift 2
      ;;
    --workspace)
      [[ $# -ge 2 ]] || die '--workspace requires a value'
      WORKSPACE="$2"
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

WORKSPACE="$(realpath -m -- "${WORKSPACE}")"
CONDA_BIN="$(command -v conda || true)"
[[ -n "${CONDA_BIN}" ]] || die 'conda is not available on PATH'
CONDA_BASE="$("${CONDA_BIN}" info --base)"
ENV_LIST_JSON="$("${CONDA_BIN}" env list --json)"
ENV_PREFIX="$("${CONDA_BASE}/bin/python" - "${ENV_NAME}" \
  "${ENV_LIST_JSON}" <<'PY'
import json
import sys
from pathlib import Path

name = sys.argv[1]
environments = [Path(item).resolve() for item in json.loads(sys.argv[2])['envs']]
requested = Path(name).expanduser()
matches = []
for environment in environments:
    if environment.name == name:
        matches.append(environment)
    elif name == 'base' and environment == Path(sys.prefix).resolve():
        matches.append(environment)
    elif requested.is_absolute() and environment == requested.resolve():
        matches.append(environment)
if len(matches) > 1:
    raise SystemExit(f'Conda environment name {name!r} is ambiguous')
if matches:
    print(matches[0])
PY
)"
[[ -n "${ENV_PREFIX}" ]] || \
  die "Conda environment ${ENV_NAME@Q} does not exist"
PYTHON_VERSION="$("${ENV_PREFIX}/bin/python" -c \
  'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')"
[[ "${PYTHON_VERSION}" == "3.10" ]] || \
  die "ROS 1 hook requires Python 3.10; ${ENV_NAME@Q} uses ${PYTHON_VERSION}"

SETUP_PATH="${WORKSPACE}/install_isolated/setup.sh"
if [[ "${DRY_RUN}" == "0" ]]; then
  [[ -r "${SETUP_PATH}" ]] || \
    die "ROS 1 overlay is not installed at ${SETUP_PATH}"
fi

CONFIG_DIR="${ENV_PREFIX}/etc/fluxthemis"
ACTIVATE_DIR="${ENV_PREFIX}/etc/conda/activate.d"
DEACTIVATE_DIR="${ENV_PREFIX}/etc/conda/deactivate.d"
CONFIG_PATH="${CONFIG_DIR}/ros1-workspace"

run mkdir -p "${CONFIG_DIR}" "${ACTIVATE_DIR}" "${DEACTIVATE_DIR}"

if [[ "${DRY_RUN}" == "1" ]]; then
  run install -m 0644 "${SCRIPT_DIR}/activate_ros1.sh" \
    "${ACTIVATE_DIR}/fluxthemis-ros1.sh"
  run install -m 0644 "${SCRIPT_DIR}/deactivate_ros.sh" \
    "${DEACTIVATE_DIR}/fluxthemis-ros1.sh"
  run rm -f -- \
    "${ACTIVATE_DIR}/fluxthemis-ros2.sh" \
    "${DEACTIVATE_DIR}/fluxthemis-ros2.sh"
  echo "+ write ${WORKSPACE} to ${CONFIG_PATH}" >&2
  echo 'Dry run complete; no files were changed.'
  exit 0
fi

# Prove the exact workspace can be loaded by the consumer interpreter before
# replacing any existing Conda hook.
env -u AMENT_PREFIX_PATH -u CATKIN_PREFIX_PATH -u CMAKE_PREFIX_PATH \
  -u COLCON_PREFIX_PATH -u LD_LIBRARY_PATH -u PKG_CONFIG_PATH \
  -u PYTHONHOME -u PYTHONPATH -u ROS_DISTRO -u ROS_MASTER_URI \
  -u ROS_PACKAGE_PATH -u ROS_ROOT -u ROS_VERSION \
  CONDA_PREFIX="${ENV_PREFIX}" FLUXTHEMIS_ROS1_WORKSPACE="${WORKSPACE}" \
  /bin/bash --noprofile --norc -c '
set -euo pipefail
source "$1"
"${CONDA_PREFIX}/bin/python" -c "import rospy; from sensor_msgs.msg import Image; from fluxthemis_msgs.srv import PredictAction, ReportEvaluation; assert Image._type == \"sensor_msgs/Image\"; assert PredictAction._type == \"fluxthemis_msgs/PredictAction\"; assert ReportEvaluation._type == \"fluxthemis_msgs/ReportEvaluation\""
' -- "${SCRIPT_DIR}/activate_ros1.sh"

ACTIVATE_TMP="${ACTIVATE_DIR}/fluxthemis-ros1.sh.tmp.$$"
DEACTIVATE_TMP="${DEACTIVATE_DIR}/fluxthemis-ros1.sh.tmp.$$"
CONFIG_TMP="${CONFIG_PATH}.tmp.$$"
install -m 0644 "${SCRIPT_DIR}/activate_ros1.sh" "${ACTIVATE_TMP}"
install -m 0644 "${SCRIPT_DIR}/deactivate_ros.sh" "${DEACTIVATE_TMP}"
printf '%s\n' "${WORKSPACE}" > "${CONFIG_TMP}"
mv "${ACTIVATE_TMP}" "${ACTIVATE_DIR}/fluxthemis-ros1.sh"
mv "${DEACTIVATE_TMP}" "${DEACTIVATE_DIR}/fluxthemis-ros1.sh"
mv "${CONFIG_TMP}" "${CONFIG_PATH}"
rm -f -- \
  "${ACTIVATE_DIR}/fluxthemis-ros2.sh" \
  "${DEACTIVATE_DIR}/fluxthemis-ros2.sh"

env -u CONDA_PREFIX -u CONDA_DEFAULT_ENV -u CONDA_SHLVL \
  -u CONDA_PROMPT_MODIFIER -u PYTHONPATH \
  -u AMENT_PREFIX_PATH -u CMAKE_PREFIX_PATH -u COLCON_PREFIX_PATH \
  -u CATKIN_PREFIX_PATH -u ROS_PACKAGE_PATH -u ROS_ROOT \
  -u ROS_MASTER_URI -u ROS_DISTRO -u ROS_VERSION \
  -u FLUXTHEMIS_ROS1_WORKSPACE -u FLUXTHEMIS_ROS2_WORKSPACE \
  -u FLUXTHEMIS_ROS2_DISTRO -u _FLUXTHEMIS_ROS_BACKUP_ACTIVE \
  /bin/bash --noprofile --norc -c '
set -euo pipefail
source "$1/etc/profile.d/conda.sh"
conda activate "$2"
[[ "${ROS_VERSION}" == "1" ]]
python -c "import rospy; from sensor_msgs.msg import Image; from fluxthemis_msgs.srv import PredictAction, ReportEvaluation; print(rospy.__file__, Image._type, PredictAction._type, ReportEvaluation._type)"
' -- "${CONDA_BASE}" "${ENV_NAME}"

cat <<EOF
ROS 1 hook installed for ${ENV_NAME} (${ENV_PREFIX}).
Open a new terminal or reactivate the environment once:
  conda deactivate && conda activate ${ENV_NAME}
EOF
