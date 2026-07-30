#!/usr/bin/env bash
# Restore rendering variables after leaving the RoboCasa Conda environment.

_fluxthemis_restore_robocasa_variable() {
  local variable_name="$1"
  local backup_name="$2"
  local old_value=''
  local was_set=''
  eval "old_value=\${_FLUXTHEMIS_ROBOCASA_OLD_${backup_name}-}"
  eval "was_set=\${_FLUXTHEMIS_ROBOCASA_OLD_${backup_name}_SET-}"
  if [[ "${was_set}" == "x" ]]; then
    export "${variable_name}=${old_value}"
  else
    unset "${variable_name}"
  fi
  unset "_FLUXTHEMIS_ROBOCASA_OLD_${backup_name}"
  unset "_FLUXTHEMIS_ROBOCASA_OLD_${backup_name}_SET"
}

if [[ "${_FLUXTHEMIS_ROBOCASA_BACKUP_ACTIVE:-0}" == "1" ]]; then
  _fluxthemis_restore_robocasa_variable MUJOCO_GL MUJOCO_GL
  _fluxthemis_restore_robocasa_variable PYOPENGL_PLATFORM PYOPENGL_PLATFORM
  _fluxthemis_restore_robocasa_variable \
    __EGL_VENDOR_LIBRARY_FILENAMES EGL_VENDOR
fi
unset _FLUXTHEMIS_ROBOCASA_BACKUP_ACTIVE
unset -f _fluxthemis_restore_robocasa_variable 2>/dev/null || true
