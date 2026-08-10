#!/usr/bin/env bash
set -euo pipefail
source "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)/common.sh"

execute=0
[[ "${1:-}" == --execute ]] && execute=1
[[ $# -le 1 ]] || die "usage: $0 [--execute]"

remove_tree() {
  local path resolved
  path="$1"
  [[ -e "${path}" ]] || return 0
  resolved="$(canonical_path "${path}")"
  assert_under_root "${resolved}"
  [[ "${resolved}" != "${ICECARVER_ROOT}" ]] || die "refusing to remove project root"
  if ((execute)); then rm -rf -- "${resolved}"; else log "would remove project path: ${resolved}"; fi
}

for path in "${ICECARVER_ROOT}/build" "${ICECARVER_ROOT}"/build-* "${ICECARVER_ROOT}/results" "${ICECARVER_ROOT}/artifacts"; do
  remove_tree "${path}"
done
if [[ -d "${ICECARVER_ROOT}/reports" ]]; then
  while IFS= read -r -d '' path; do
    if ((execute)); then rm -f -- "${path}"; else log "would remove report artifact: ${path}"; fi
  done < <(find "${ICECARVER_ROOT}/reports" -maxdepth 1 -type f \( -name '*.json' -o -name '*.log' -o -name '*.csv' \) -print0)
fi
while IFS= read -r -d '' path; do
  remove_tree "${path}"
done < <(find "${ICECARVER_ROOT}" -path "${ICECARVER_ROOT}/.git" -prune -o -type d \( -name __pycache__ -o -name .pytest_cache \) -print0)

for cache in "${HOME}/.nv/ComputeCache" "${XDG_CACHE_HOME:-${HOME}/.cache}/ice-carver"; do
  [[ -d "${cache}" ]] || continue
  if ((execute)); then
    find "${cache}" -mindepth 1 -maxdepth 1 -exec rm -rf -- {} +
  else
    log "would empty user cache: ${cache}"
  fi
done

log '~/.ssh was not read, modified, or deleted'
if ((execute)); then
  log 'scoped image cleanup complete; run pre_image_audit.sh next'
else
  log 'dry run only; pass --execute to apply the listed cleanup'
fi
