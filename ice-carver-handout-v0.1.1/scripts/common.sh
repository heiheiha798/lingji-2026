#!/usr/bin/env bash
set -euo pipefail

ICECARVER_SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
ICECARVER_ROOT="$(cd -- "${ICECARVER_SCRIPT_DIR}/.." && pwd -P)"
ICECARVER_BUILD_DIR="${ICECARVER_BUILD_DIR:-${ICECARVER_ROOT}/build}"
ICECARVER_EVAL_BIN="${ICECARVER_EVAL_BIN:-${ICECARVER_BUILD_DIR}/icecarver_eval}"
ICECARVER_PYTHON="${ICECARVER_PYTHON:-python3}"
ICECARVER_CASE_TIMEOUT_SECONDS="${ICECARVER_CASE_TIMEOUT_SECONDS:-660}"
ICECARVER_KILL_AFTER_SECONDS="${ICECARVER_KILL_AFTER_SECONDS:-30}"
ICECARVER_MANIFEST="${ICECARVER_MANIFEST:-${ICECARVER_ROOT}/evaluator/official_manifest.json}"

log() { printf '[ice-carver] %s\n' "$*"; }
die() { printf '[ice-carver] ERROR: %s\n' "$*" >&2; exit 1; }

require_command() {
  command -v "$1" >/dev/null 2>&1 || die "required command not found: $1"
}

canonical_path() { readlink -m -- "$1"; }

assert_under_root() {
  local path root
  path="$(canonical_path "$1")"
  root="$(canonical_path "${ICECARVER_ROOT}")"
  case "${path}" in
    "${root}"|"${root}/"*) ;;
    *) die "path escapes project root: ${path}" ;;
  esac
}

case_name_from_config() {
  local name
  name="$(basename -- "$1")"
  name="${name%.yaml}"
  name="${name%.yml}"
  [[ "${name}" =~ ^[A-Za-z0-9._-]+$ ]] || die "unsafe case filename: $1"
  printf '%s\n' "${name}"
}

collect_files() {
  local dir pattern array_name
  dir="$1"
  pattern="$2"
  array_name="$3"
  [[ -d "${dir}" ]] || die "directory not found: ${dir}"
  local -a found=()
  mapfile -d '' found < <(find "${dir}" -maxdepth 1 -type f -name "${pattern}" -print0 | sort -z)
  ((${#found[@]} > 0)) || die "no ${pattern} files in ${dir}"
  local -n output_ref="${array_name}"
  output_ref=("${found[@]}")
}

run_evaluator_case() {
  local config solver output tmp status
  config="$1"
  solver="$2"
  output="$3"
  [[ -f "${config}" ]] || die "config not found: ${config}"
  [[ "${solver}" == public || "${solver}" == target ]] || die "invalid solver: ${solver}"
  [[ -x "${ICECARVER_EVAL_BIN}" ]] || die "evaluator not executable: ${ICECARVER_EVAL_BIN}"
  assert_under_root "${output}"
  mkdir -p -- "$(dirname -- "${output}")"
  tmp="${output}.tmp.$$"
  rm -f -- "${tmp}"
  log "case=$(basename -- "${config}") solver=${solver}"
  set +e
  timeout --signal=TERM --kill-after="${ICECARVER_KILL_AFTER_SECONDS}s" \
    "${ICECARVER_CASE_TIMEOUT_SECONDS}s" \
    "${ICECARVER_EVAL_BIN}" --config "${config}" --solver "${solver}" \
    --output "${tmp}"
  status=$?
  set -e
  if ((status != 0)); then
    rm -f -- "${tmp}"
    if ((status == 124 || status == 137)); then
      die "case hard timeout: ${config}"
    fi
    die "evaluator failed with status ${status}: ${config}"
  fi
  [[ -s "${tmp}" ]] || die "evaluator produced no JSON: ${config}"
  mv -f -- "${tmp}" "${output}"
}

collect_calibration_side() {
  local side array_name explicit mapping raw_dir
  side="$1"
  array_name="$2"
  [[ "${side}" == baseline || "${side}" == target ]] || die "bad calibration side"
  if [[ "${side}" == baseline ]]; then
    explicit="${ICECARVER_BASELINE_FILE:-}"
  else
    explicit="${ICECARVER_TARGET_FILE:-}"
  fi
  local -a values=()
  if [[ -n "${explicit}" ]]; then
    [[ -f "${explicit}" ]] || die "calibration file not found: ${explicit}"
    values=("${explicit}")
  else
    mapping="${ICECARVER_ROOT}/config/calibration/${side}.json"
    raw_dir="${ICECARVER_ROOT}/results/calibration/${side}/cases"
    if [[ -f "${mapping}" ]]; then
      values=("${mapping}")
    elif [[ -d "${raw_dir}" ]]; then
      mapfile -d '' values < <(find "${raw_dir}" -maxdepth 1 -type f -name '*.json' -print0 | sort -z)
    fi
  fi
  ((${#values[@]} > 0)) || die "no ${side} calibration; run calibrate.sh or set ICECARVER_${side^^}_FILE"
  local -n output_ref="${array_name}"
  output_ref=("${values[@]}")
}
