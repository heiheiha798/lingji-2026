#!/usr/bin/env bash
set -euo pipefail
source "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)/common.sh"

for cmd in nvidia-smi nvcc cmake ninja timeout "${ICECARVER_PYTHON}"; do require_command "${cmd}"; done
nvcc --version | grep -Eq 'release 12\.8([, ])' || die "CUDA Toolkit 12.8 is required"
gpu_info="$(nvidia-smi --query-gpu=name,compute_cap --format=csv,noheader 2>/dev/null || nvidia-smi --query-gpu=name --format=csv,noheader)"
printf '%s\n' "${gpu_info}"
if [[ "${ICECARVER_REQUIRE_4090:-1}" == 1 ]]; then
  grep -q '4090' <<<"${gpu_info}" || die "smoke test requires RTX 4090"
fi

ICECARVER_ENABLE_TARGET=OFF bash "${ICECARVER_SCRIPT_DIR}/build.sh" --public
declare -a configs
collect_files "${ICECARVER_PUBLIC_CONFIG_DIR:-${ICECARVER_ROOT}/config/public}" '*.yaml' configs
output="${ICECARVER_ROOT}/results/smoke/$(case_name_from_config "${configs[0]}").json"
run_evaluator_case "${configs[0]}" public "${output}"
log "smoke test passed: ${output}"
