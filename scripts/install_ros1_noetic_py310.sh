#!/usr/bin/env bash
# Build the ROS 1 runtime used by FluxThemis from pinned Noetic release sources.
# Noetic never shipped Ubuntu 24.04 / Python 3.10 binaries, so this overlay is
# intentionally source-built and does not include cv_bridge.

set -euo pipefail

ROS_DISTRO="noetic"
SOURCE_MANIFEST_SHA256="86b61b204b8ffa2e2b112f55387d5d736756dcbdceb3e486c80b40a9c25b0cae"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
MESSAGE_PACKAGE="${PROJECT_ROOT}/ros/fluxthemis_msgs"
PINNED_SOURCE_MANIFEST="${PROJECT_ROOT}/ros/noetic_py310.rosinstall"

ENV_NAME="${FLUXTHEMIS_CONDA_ENV:-fluxthemis}"
WORKSPACE="${FLUXTHEMIS_ROS1_WORKSPACE:-/root/fluxthemis_ros1_ws}"
DRY_RUN=0

APT_PACKAGES=(
  build-essential
  cmake
  git
  pkg-config
  libapr1-dev
  libaprutil1-dev
  libboost-all-dev
  libbz2-dev
  libconsole-bridge-dev
  libgpgme-dev
  liblog4cxx-dev
  liblz4-dev
  libpoco-dev
  libssl-dev
  libtinyxml2-dev
  libxml2-dev
  libyaml-cpp-dev
  libyaml-dev
  uuid-dev
)

# These versions produced the checked-in environment and the pinned source
# manifest hash below. In particular, EmPy 4 and newer are incompatible with
# Noetic's message generators, and newer setuptools removed APIs used by
# catkin's Python packaging path.
PYTHON_PACKAGES=(
  "catkin-pkg==1.1.0"
  "defusedxml==0.7.1"
  "distro==1.9.0"
  "empy==3.3.4"
  "future==1.0.0"
  "mock==5.2.0"
  "netifaces==0.11.0"
  "nose==1.3.7"
  "paramiko==5.0.0"
  "pycryptodome==3.23.0"
  "pycryptodomex==3.23.0"
  "python-gnupg==0.5.6"
  "PyYAML==6.0.3"
  "rosdep==0.26.0"
  "rosdistro==1.0.1"
  "rosinstall-generator==0.1.23"
  "rospkg==1.6.1"
  "setuptools==70.3.0"
  "vcstool==0.3.0"
  "wstool==0.1.17"
)

usage() {
  cat <<'EOF'
Usage: bash scripts/install_ros1_noetic_py310.sh [options]

Builds a minimal ROS 1 Noetic runtime against Python 3.10, then builds the
FluxThemis PredictAction service package. Existing source checkouts are never
deleted or force-reset.

Options:
  --env-name NAME      Existing Python 3.10 Conda environment
                       (default: fluxthemis).
  --workspace PATH     ROS 1 source/build/install workspace
                       (default: /root/fluxthemis_ros1_ws).
  --dry-run            Validate the host/env and print mutating commands.
  -h, --help           Show this help.

Environment variables:
  FLUXTHEMIS_CONDA_ENV
  FLUXTHEMIS_ROS1_WORKSPACE
  FLUXTHEMIS_BUILD_JOBS       Parallel build jobs (default: min(nproc, 8)).

After installation, activate a clean shell with:
  conda activate fluxthemis

The installer adds a Conda activation hook. For manual activation:
  source scripts/activate_ros1.sh
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

run_privileged() {
  if [[ "${EUID}" -eq 0 ]]; then
    run "$@"
  elif command -v sudo >/dev/null 2>&1; then
    run sudo "$@"
  else
    die 'root privileges are required for apt, and sudo is unavailable'
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

[[ -r /etc/os-release ]] || die '/etc/os-release is unavailable'
# shellcheck disable=SC1091
source /etc/os-release
[[ "${ID:-}" == "ubuntu" && "${VERSION_ID:-}" == "24.04" && \
   "${UBUNTU_CODENAME:-}" == "noble" ]] || \
  die 'this source-build recipe supports Ubuntu 24.04 (noble) only'

CONDA_BIN="${CONDA_EXE:-$(command -v conda || true)}"
[[ -n "${CONDA_BIN}" && -x "${CONDA_BIN}" ]] || \
  die 'conda is not available on PATH'

PYTHON_VERSION="$(env -u PYTHONHOME -u PYTHONPATH \
  "${CONDA_BIN}" run -n "${ENV_NAME}" python -c \
  'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")' \
  2>/dev/null || true)"
[[ "${PYTHON_VERSION}" == "3.10" ]] || \
  die "conda env ${ENV_NAME@Q} must exist and use Python 3.10; found ${PYTHON_VERSION:-nothing}"

ENV_PREFIX="$(env -u PYTHONHOME -u PYTHONPATH \
  "${CONDA_BIN}" run -n "${ENV_NAME}" python -c \
  'import sys; print(sys.prefix)')"
ENV_PYTHON="${ENV_PREFIX}/bin/python"
[[ -x "${ENV_PYTHON}" ]] || die "missing environment Python: ${ENV_PYTHON}"

WORKSPACE="$(realpath -m "${WORKSPACE}")"
SOURCE_SPACE="${WORKSPACE}/src"
INSTALL_SPACE="${WORKSPACE}/install_isolated"
SOURCE_MANIFEST="${PINNED_SOURCE_MANIFEST}"
MESSAGE_LINK="${SOURCE_SPACE}/fluxthemis_msgs"

[[ -f "${MESSAGE_PACKAGE}/package.xml" && \
   -f "${MESSAGE_PACKAGE}/CMakeLists.txt" && \
   -f "${MESSAGE_PACKAGE}/srv/PredictAction.srv" && \
   -f "${MESSAGE_PACKAGE}/srv/ReportEvaluation.srv" ]] || \
  die "invalid FluxThemis ROS 1 message package: ${MESSAGE_PACKAGE}"
[[ -f "${SOURCE_MANIFEST}" ]] || \
  die "missing pinned Noetic source manifest: ${SOURCE_MANIFEST}"

BUILD_JOBS="${FLUXTHEMIS_BUILD_JOBS:-}"
if [[ -z "${BUILD_JOBS}" ]]; then
  BUILD_JOBS="$(nproc)"
  (( BUILD_JOBS > 8 )) && BUILD_JOBS=8
fi
[[ "${BUILD_JOBS}" =~ ^[1-9][0-9]*$ ]] || \
  die 'FLUXTHEMIS_BUILD_JOBS must be a positive integer'

CLEAN_ENV=(
  env
  -u AMENT_PREFIX_PATH
  -u CATKIN_PREFIX_PATH
  -u CMAKE_PREFIX_PATH
  -u COLCON_PREFIX_PATH
  -u PKG_CONFIG_PATH
  -u PYTHONHOME
  -u PYTHONPATH
  -u ROS_DISTRO
  -u ROS_PACKAGE_PATH
  -u ROS_PYTHON_VERSION
  -u ROS_VERSION
  "CONDA_DEFAULT_ENV=${ENV_NAME}"
  "CONDA_PREFIX=${ENV_PREFIX}"
  "LD_LIBRARY_PATH=${ENV_PREFIX}/lib"
  "PATH=${ENV_PREFIX}/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin"
)

run_clean() {
  print_command "${CLEAN_ENV[@]}" "$@"
  if [[ "${DRY_RUN}" == "0" ]]; then
    "${CLEAN_ENV[@]}" "$@"
  fi
}

ensure_apt_dependencies() {
  local missing=()
  local package
  for package in "${APT_PACKAGES[@]}"; do
    if ! dpkg-query -W -f='${Status}' "${package}" 2>/dev/null | \
      grep -q '^install ok installed$'; then
      missing+=("${package}")
    fi
  done
  if [[ ${#missing[@]} -eq 0 ]]; then
    echo 'ROS 1 system build dependencies are already installed.'
    return
  fi
  run_privileged env DEBIAN_FRONTEND=noninteractive apt-get update
  run_privileged env DEBIAN_FRONTEND=noninteractive apt-get install -y \
    "${missing[@]}"
}

install_python_dependencies() {
  run_clean "${ENV_PYTHON}" -m pip install --disable-pip-version-check \
    "${PYTHON_PACKAGES[@]}"
}

validate_source_manifest() {
  local actual_sha256
  actual_sha256="$(sha256sum "${SOURCE_MANIFEST}" | awk '{print $1}')"
  [[ "${actual_sha256}" == "${SOURCE_MANIFEST_SHA256}" ]] || \
    die "pinned Noetic manifest changed: expected ${SOURCE_MANIFEST_SHA256}, got ${actual_sha256}"
  echo "Pinned Noetic source manifest: ${SOURCE_MANIFEST} (${actual_sha256})."
}

validate_source_tree() {
  if [[ "${DRY_RUN}" == "1" ]]; then
    echo "+ validate every existing ROS release checkout against the pinned manifest" >&2
    return
  fi
  print_command "${CLEAN_ENV[@]}" "${ENV_PYTHON}" - \
    "${SOURCE_MANIFEST}" "${SOURCE_SPACE}"
  "${CLEAN_ENV[@]}" "${ENV_PYTHON}" - \
    "${SOURCE_MANIFEST}" "${SOURCE_SPACE}" <<'PY'
import pathlib
import subprocess
import sys
import yaml

manifest_path = pathlib.Path(sys.argv[1])
source_space = pathlib.Path(sys.argv[2])
entries = yaml.safe_load(manifest_path.read_text(encoding="utf-8"))

def git(path, *args):
    return subprocess.run(
        ["git", "-C", str(path), *args],
        check=True,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    ).stdout.strip()

def canonical_url(value):
    return value.rstrip("/").removesuffix(".git")

for entry in entries:
    spec = entry.get("git")
    if spec is None:
        raise SystemExit(f"unsupported non-git manifest entry: {entry}")
    path = source_space / spec["local-name"]
    if not (path / ".git").is_dir():
        raise SystemExit(f"missing release checkout after import: {path}")
    status = git(path, "status", "--porcelain", "--untracked-files=all")
    dirty = []
    for line in status.splitlines():
        relative_path = pathlib.PurePosixPath(line[3:])
        generated_python = (
            line.startswith("?? ")
            and ("__pycache__" in relative_path.parts
                 or relative_path.suffix in {".pyc", ".pyo"})
        )
        if not generated_python:
            dirty.append(line)
    if dirty:
        raise SystemExit(
            f"changes found in managed ROS source {path}: {dirty}")
    actual_url = git(path, "remote", "get-url", "origin")
    if canonical_url(actual_url) != canonical_url(spec["uri"]):
        raise SystemExit(
            f"unexpected origin for {path}: {actual_url!r}, expected {spec['uri']!r}"
        )
    head = git(path, "rev-parse", "HEAD")
    expected = git(path, "rev-parse", f"{spec['version']}^{{commit}}")
    if head != expected:
        raise SystemExit(
            f"unexpected revision for {path}: {head}, expected {spec['version']} ({expected})"
        )
PY
}

import_sources() {
  run mkdir -p "${SOURCE_SPACE}"
  run_clean "${ENV_PREFIX}/bin/vcs" import "${SOURCE_SPACE}" \
    --input "${SOURCE_MANIFEST}" --skip-existing --retry 3
  validate_source_tree
}

ensure_message_link() {
  local resolved_link
  local resolved_package
  resolved_package="$(realpath -m "${MESSAGE_PACKAGE}")"
  if [[ -L "${MESSAGE_LINK}" ]]; then
    resolved_link="$(realpath -m "${MESSAGE_LINK}")"
    [[ "${resolved_link}" == "${resolved_package}" ]] || \
      die "${MESSAGE_LINK} points to ${resolved_link}; refusing to replace it"
    echo "Keeping message package symlink ${MESSAGE_LINK}."
    return
  fi
  [[ ! -e "${MESSAGE_LINK}" ]] || \
    die "${MESSAGE_LINK} exists and is not the expected symlink"
  run mkdir -p "${SOURCE_SPACE}"
  run ln -s "${MESSAGE_PACKAGE}" "${MESSAGE_LINK}"
}

build_overlay() {
  local catkin_make="${SOURCE_SPACE}/catkin/bin/catkin_make_isolated"
  if [[ "${DRY_RUN}" == "0" ]]; then
    [[ -f "${catkin_make}" ]] || \
      die "catkin_make_isolated was not imported at ${catkin_make}"
  fi
  run_clean "${ENV_PYTHON}" "${catkin_make}" \
    -C "${WORKSPACE}" \
    --install \
    --install-space "${INSTALL_SPACE}" \
    --force-cmake \
    --only-pkg-with-deps \
      roslaunch rosmaster rosout rospy rosmsg rosservice sensor_msgs \
      fluxthemis_msgs \
    -j"${BUILD_JOBS}" -l"${BUILD_JOBS}" \
    --cmake-args \
      -DCMAKE_BUILD_TYPE=Release \
      -DSETUPTOOLS_DEB_LAYOUT=OFF \
      -DROSCONSOLE_BACKEND=print \
      "-DPYTHON_EXECUTABLE=${ENV_PYTHON}" \
      "-DPython3_EXECUTABLE=${ENV_PYTHON}" \
      -DPYTHON_VERSION=3.10 \
      -DCATKIN_ENABLE_TESTING=OFF
}

install_activation_hook() {
  local config_dir="${ENV_PREFIX}/etc/fluxthemis"
  local hook_dir="${ENV_PREFIX}/etc/conda/activate.d"
  local deactivation_dir="${ENV_PREFIX}/etc/conda/deactivate.d"
  local config_path="${config_dir}/ros1-workspace"
  local config_tmp="${config_path}.tmp.$$"

  run mkdir -p "${config_dir}" "${hook_dir}" "${deactivation_dir}"
  # A Conda interpreter can source only one ROS ABI. Remove hooks left by a
  # previous ROS 2 use of the same environment before installing Noetic.
  run rm -f -- "${hook_dir}/fluxthemis-ros2.sh" \
    "${deactivation_dir}/fluxthemis-ros2.sh"
  run install -m 0644 "${SCRIPT_DIR}/activate_ros1.sh" \
    "${hook_dir}/fluxthemis-ros1.sh"
  run install -m 0644 "${SCRIPT_DIR}/deactivate_ros.sh" \
    "${deactivation_dir}/fluxthemis-ros1.sh"
  if [[ "${DRY_RUN}" == "1" ]]; then
    echo "+ write ${WORKSPACE} to ${config_path}" >&2
  else
    printf '%s\n' "${WORKSPACE}" > "${config_tmp}"
    mv "${config_tmp}" "${config_path}"
  fi
}

validate_installation() {
  local validation_script
  validation_script='set -euo pipefail
export CONDA_PREFIX="$1"
export FLUXTHEMIS_ROS1_WORKSPACE="$2"
source "$3"
"${CONDA_PREFIX}/bin/python" -c '\''import sys, rospy; from Cryptodome.Cipher import AES; from sensor_msgs.msg import Image; from fluxthemis_msgs.srv import PredictAction, ReportEvaluation; assert sys.version_info[:2] == (3, 10); print("ROS 1 Python imports: OK", PredictAction, ReportEvaluation)'\''
"${CONDA_PREFIX}/bin/python" -m pip check
rospack find rospy >/dev/null
rospack find rosservice >/dev/null
rospack find sensor_msgs >/dev/null
rospack find fluxthemis_msgs >/dev/null
command -v rosservice >/dev/null
rossrv show fluxthemis_msgs/PredictAction | grep -Fq "sensor_msgs/Image[] images"
rossrv show fluxthemis_msgs/PredictAction | grep -Fq "float32[] actions"
rossrv show fluxthemis_msgs/ReportEvaluation | grep -Fq "string payload_json"
rossrv show fluxthemis_msgs/ReportEvaluation | grep -Fq "bool accepted"
echo "rospack and rossrv: OK"'

  run_clean /bin/bash --noprofile --norc -c "${validation_script}" -- \
    "${ENV_PREFIX}" "${WORKSPACE}" "${SCRIPT_DIR}/activate_ros1.sh"
}

ensure_apt_dependencies
install_python_dependencies
validate_source_manifest
import_sources
ensure_message_link
build_overlay
install_activation_hook
validate_installation

echo
if [[ "${DRY_RUN}" == "1" ]]; then
  echo 'Dry run complete; no files or packages were changed.'
  exit 0
fi
echo "ROS 1 ${ROS_DISTRO} / Python 3.10 overlay is ready: ${INSTALL_SPACE}"
echo "Conda environment: ${ENV_NAME} (${ENV_PREFIX})"
echo 'Open a clean shell and run:'
echo "  conda activate ${ENV_NAME}"
