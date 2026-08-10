#!/usr/bin/env bash
set -euo pipefail
source "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)/common.sh"

config_dir="${ICECARVER_PUBLIC_CONFIG_DIR:-${ICECARVER_ROOT}/config/public}"
output_dir="${ICECARVER_PUBLIC_RESULTS_DIR:-${ICECARVER_ROOT}/results/public}"
assert_under_root "${output_dir}"
if [[ "${ICECARVER_SKIP_BUILD:-0}" != 1 ]]; then
  ICECARVER_ENABLE_TARGET=OFF bash "${ICECARVER_SCRIPT_DIR}/build.sh" --public
fi
[[ -x "${ICECARVER_EVAL_BIN}" ]] || die "public evaluator is absent"

declare -a configs outputs baseline target
collect_files "${config_dir}" '*.yaml' configs
for config in "${configs[@]}"; do
  name="$(case_name_from_config "${config}")"
  output="${output_dir}/cases/${name}.json"
  run_evaluator_case "${config}" public "${output}"
  outputs+=("${output}")
done

if [[ "${ICECARVER_SKIP_SCORE:-0}" == 1 ]]; then
  log "case JSON complete; score skipped"
  exit 0
fi
collect_calibration_side baseline baseline
collect_calibration_side target target
require_command "${ICECARVER_PYTHON}"
args=(--manifest "${ICECARVER_MANIFEST}" --results "${outputs[@]}" --baseline "${baseline[@]}" --target "${target[@]}" --output "${output_dir}/summary.json")
"${ICECARVER_PYTHON}" "${ICECARVER_ROOT}/evaluator/score.py" "${args[@]}"
log "public summary: ${output_dir}/summary.json"
