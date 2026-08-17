#!/usr/bin/env bash
# Restore simulator variables that existed before activating LIBERO-Plus.

_fluxthemis_restore_libero_plus_variable() {
  local variable_name="$1"
  local backup_name="$2"
  local old_value=''
  local was_set=''
  eval "old_value=\${_FLUXTHEMIS_LIBERO_PLUS_OLD_${backup_name}-}"
  eval "was_set=\${_FLUXTHEMIS_LIBERO_PLUS_OLD_${backup_name}_SET-}"
  if [[ "${was_set}" == "x" ]]; then
    export "${variable_name}=${old_value}"
  else
    unset "${variable_name}"
  fi
  unset "_FLUXTHEMIS_LIBERO_PLUS_OLD_${backup_name}"
  unset "_FLUXTHEMIS_LIBERO_PLUS_OLD_${backup_name}_SET"
}

if [[ "${_FLUXTHEMIS_LIBERO_PLUS_BACKUP_ACTIVE:-0}" == "1" ]]; then
  _fluxthemis_restore_libero_plus_variable \
    LIBERO_PLUS_CONFIG_PATH PLUS_CONFIG
  _fluxthemis_restore_libero_plus_variable LIBERO_CONFIG_PATH CONFIG_PATH
  _fluxthemis_restore_libero_plus_variable MUJOCO_GL MUJOCO_GL
  _fluxthemis_restore_libero_plus_variable \
    PYOPENGL_PLATFORM PYOPENGL_PLATFORM
  _fluxthemis_restore_libero_plus_variable \
    __EGL_VENDOR_LIBRARY_FILENAMES EGL_VENDOR
fi
unset _FLUXTHEMIS_LIBERO_PLUS_BACKUP_ACTIVE
unset -f _fluxthemis_restore_libero_plus_variable 2>/dev/null || true
