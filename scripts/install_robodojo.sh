#!/usr/bin/env bash
# Install RoboDojo through its official scripts, then add FluxThemis.

set -euo pipefail

ROBODOJO_REF="${ROBODOJO_REF:-2184bf8844ea9d205382c4aefa3a694311418251}"
FLUXTHEMIS_ROOT="${FLUXTHEMIS_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
ROBODOJO_ROOT="${ROBODOJO_ROOT:-$(dirname "${FLUXTHEMIS_ROOT}")/RoboDojo}"
SKIP_ASSETS=0

usage() {
  cat <<'EOF'
Usage: bash scripts/install_robodojo.sh [--skip-assets]

Clones RoboDojo into ../RoboDojo when absent, checks out ROBODOJO_REF,
calls RoboDojo's official installer, installs FluxThemis into the official
`RoboDojo` conda environment, and runs RoboDojo's official doctor checks.

Options:
  --skip-assets  Skip RoboDojo's official asset download.
  -h, --help     Show this help.

Standard HTTP(S) proxy, pip, Git, and Hugging Face environment settings are
inherited by RoboDojo's scripts. Override ROBODOJO_ROOT or ROBODOJO_REF with
environment variables when necessary.
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --skip-assets) SKIP_ASSETS=1; shift ;;
    -h|--help) usage; exit 0 ;;
    *) echo "error: unknown option: $1" >&2; exit 2 ;;
  esac
done


install_graphics_deps() {
  if [[ "$(id -u)" -eq 0 ]]; then
    apt-get install -y libglu1-mesa libvulkan1 mesa-vulkan-drivers vulkan-tools
  else
    sudo apt-get install -y libglu1-mesa libvulkan1 mesa-vulkan-drivers vulkan-tools
  fi
  vulkaninfo --summary | head -40 || true
}

if [[ ! -e "${ROBODOJO_ROOT}" ]]; then
  git clone --no-recurse-submodules \
    https://github.com/RoboDojo-Benchmark/RoboDojo.git "${ROBODOJO_ROOT}"
elif [[ ! -d "${ROBODOJO_ROOT}/.git" ]]; then
  echo "error: RoboDojo path is not a git checkout: ${ROBODOJO_ROOT}" >&2
  exit 1
fi

git -C "${ROBODOJO_ROOT}" checkout "${ROBODOJO_REF}"
[[ -f "${ROBODOJO_ROOT}/scripts/install.sh" ]] || \
  { echo "error: missing RoboDojo install.sh" >&2; exit 1; }

install_graphics_deps
bash "${ROBODOJO_ROOT}/scripts/install.sh" -i
if [[ "${SKIP_ASSETS}" == "0" ]]; then
  bash "${ROBODOJO_ROOT}/scripts/init_assets.sh"
fi

# RoboDojo's checked-in ``curobo.yml`` files can contain absolute paths baked
# from another machine (for example ``/personal/Web/RoboDojo``). Regenerate
# them from their ``*_tmp.yml`` templates for this checkout, otherwise the
# X5 planner fails with ``X5A.urdf is not a file``.
(
  cd "${ROBODOJO_ROOT}"
  python utils/update_embodiment_config_path.py
)

CONDA_BIN="${CONDA_EXE:-$(command -v conda || true)}"
if [[ -z "${CONDA_BIN}" && -x "${HOME}/miniconda3/bin/conda" ]]; then
  CONDA_BIN="${HOME}/miniconda3/bin/conda"
fi
[[ -n "${CONDA_BIN}" ]] || \
  { echo "error: conda unavailable after RoboDojo install" >&2; exit 1; }

"${CONDA_BIN}" run --no-capture-output -n RoboDojo \
  python -m pip install -e "${FLUXTHEMIS_ROOT}"
(
  cd "${ROBODOJO_ROOT}"
  "${CONDA_BIN}" run --no-capture-output -n RoboDojo \
    bash scripts/robodojo.sh doctor \
    --skip-isaac --skip-conda --skip-policy
)

echo "RoboDojo + FluxThemis installation complete."
echo "Activate with: conda activate RoboDojo"
