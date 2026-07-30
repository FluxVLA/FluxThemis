#!/usr/bin/env bash
# Configure headless MuJoCo rendering for the RoboCasa Conda environment.

if [[ "${_FLUXTHEMIS_ROBOCASA_BACKUP_ACTIVE:-0}" != "1" ]]; then
  export _FLUXTHEMIS_ROBOCASA_OLD_MUJOCO_GL="${MUJOCO_GL-}"
  export _FLUXTHEMIS_ROBOCASA_OLD_MUJOCO_GL_SET="${MUJOCO_GL+x}"
  export _FLUXTHEMIS_ROBOCASA_OLD_PYOPENGL_PLATFORM="${PYOPENGL_PLATFORM-}"
  export _FLUXTHEMIS_ROBOCASA_OLD_PYOPENGL_PLATFORM_SET="${PYOPENGL_PLATFORM+x}"
  export _FLUXTHEMIS_ROBOCASA_OLD_EGL_VENDOR="${__EGL_VENDOR_LIBRARY_FILENAMES-}"
  export _FLUXTHEMIS_ROBOCASA_OLD_EGL_VENDOR_SET="${__EGL_VENDOR_LIBRARY_FILENAMES+x}"
  export _FLUXTHEMIS_ROBOCASA_BACKUP_ACTIVE=1
fi

export MUJOCO_GL="${MUJOCO_GL:-egl}"
export PYOPENGL_PLATFORM="${PYOPENGL_PLATFORM:-${MUJOCO_GL}}"

_fluxthemis_nvidia_vendor='/usr/share/glvnd/egl_vendor.d/10_nvidia.json'
if [[ "${MUJOCO_GL}" == "egl" \
      && -z "${__EGL_VENDOR_LIBRARY_FILENAMES:-}" \
      && -f "${_fluxthemis_nvidia_vendor}" ]]; then
  export __EGL_VENDOR_LIBRARY_FILENAMES="${_fluxthemis_nvidia_vendor}"
fi
unset _fluxthemis_nvidia_vendor
