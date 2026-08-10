#!/usr/bin/env bash
set -euo pipefail
source "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)/common.sh"

role="${1:-author}"
[[ "${role}" == author || "${role}" == contestant ]] || die "usage: $0 [author|contestant]"
require_command rg
require_command find
failures=0

if [[ "${role}" == contestant && ${EUID} -ne 0 ]]; then
  die "contestant image audit must run as root: sudo bash scripts/pre_image_audit.sh contestant"
fi

log "audit role=${role} root=${ICECARVER_ROOT}"
credential_pattern='(-----BEGIN [A-Z ]*PRIVATE KEY-----|(?i)(compshare_(public|private)_key|access[_-]?key|secret[_-]?key|api[_-]?key|password|token)[[:space:]]*[:=][[:space:]]*["\x27]?[A-Za-z0-9_./+-]{16,})'

# Check credential-looking filenames without printing file contents.
scan_roots=("${ICECARVER_ROOT}")
if [[ "${role}" == contestant ]]; then
  scan_roots=(/home /root /tmp)
fi
mapfile -d '' suspicious_names < <(find "${scan_roots[@]}" -type f \
  \( -name '*.pem' -o -name '*.key' -o -name '.env' \
     -o -iname '*credentials*' -o -name 'id_rsa' -o -name 'id_ed25519' \) \
  -print0 2>/dev/null)
if ((${#suspicious_names[@]})); then
  printf 'suspicious credential filenames (contents hidden):\n' >&2
  printf '  %s\n' "${suspicious_names[@]}" >&2
  failures=$((failures + 1))
fi

# Search ignored files, histories, profile fragments, cloud-init state and
# temporary/user directories. rg exit >1 is an audit failure, never "no match".
content_roots=("${ICECARVER_ROOT}")
if [[ "${role}" == contestant ]]; then
  content_roots=(/home /root /tmp /etc/environment /etc/profile /etc/profile.d /etc/cloud/cloud.cfg.d)
fi
set +e
secret_output="$(rg -Il --hidden --no-ignore \
  --glob '!**/scripts/pre_image_audit.sh' "${credential_pattern}" \
  "${content_roots[@]}" 2>/dev/null)"
scan_status=$?
set -e
((scan_status <= 1)) || die "credential scan failed with status ${scan_status}"
if [[ -n "${secret_output}" ]]; then
  printf 'possible credential assignments (filenames only):\n%s\n' \
    "${secret_output}" >&2
  failures=$((failures + 1))
fi

if env | cut -d= -f1 | grep -Eqi \
  '^(COMPSHARE_(PUBLIC|PRIVATE)_KEY|.*(ACCESS|SECRET|API)[_-]?KEY|.*TOKEN)$'; then
  printf 'credential-like environment variable is present; values were not read\n' >&2
  failures=$((failures + 1))
fi

mapfile -d '' compshare_profiles < <(find /home /root -type d \
  -path '*/.config/compshare' -print0 2>/dev/null || true)
if ((${#compshare_profiles[@]})); then
  printf 'CompShare credential profile directories exist:\n' >&2
  printf '  %s\n' "${compshare_profiles[@]}" >&2
  failures=$((failures + 1))
fi

if [[ "${role}" == contestant ]]; then
  require_command sha256sum
  [[ -f "${ICECARVER_ROOT}/MANIFEST.sha256" ]] || \
    die "student package MANIFEST.sha256 is missing"
  if ! (cd -- "${ICECARVER_ROOT}" && sha256sum --quiet -c MANIFEST.sha256); then
    printf 'student package manifest verification failed\n' >&2
    failures=$((failures + 1))
  fi

  # The clean student VM must not contain another author tree or raw target
  # evidence anywhere in user/temp storage. The reviewed calibration map
  # config/calibration/target.json is intentionally allowed.
  mapfile -d '' forbidden < <(find /home /root /tmp -type f \
    \( -path '*/config/private/*' -o -name 'cuda_target.cu' \
       -o -name '*_target.json' -o -path '*/results/calibration/*' \
       -o -path '*/reports/private-*' \) -print0 2>/dev/null)
  if ((${#forbidden[@]})); then
    printf 'contestant image contains private/target raw artifacts:\n' >&2
    printf '  %s\n' "${forbidden[@]}" >&2
    failures=$((failures + 1))
  fi
  if rg -l '^ICECARVER_ENABLE_TARGET:BOOL=ON$' \
      "${ICECARVER_ROOT}"/build*/CMakeCache.txt >/dev/null 2>&1; then
    printf 'contestant image contains a TARGET=ON build cache\n' >&2
    failures=$((failures + 1))
  fi
fi

mapfile -d '' private_ssh_keys < <(find /home /root -type f \
  \( -name 'id_rsa' -o -name 'id_ed25519' -o -name '*.pem' \) \
  -path '*/.ssh/*' -print0 2>/dev/null || true)
if ((${#private_ssh_keys[@]})); then
  printf 'private SSH key files exist and must be removed before capture:\n' >&2
  printf '  %s\n' "${private_ssh_keys[@]}" >&2
  failures=$((failures + 1))
fi
log 'authorized_keys content is not printed; confirm platform key reset/injection policy in the image handoff'

for cache in /home/*/.nv/ComputeCache /home/*/.cache/ice-carver \
             /root/.nv/ComputeCache /root/.cache/ice-carver; do
  [[ -e "${cache}" ]] && log "removable cache present: ${cache}"
done
((failures == 0)) || die "image audit failed with ${failures} finding group(s)"
log "image audit passed"
