#!/usr/bin/env bash
set -euo pipefail
source "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)/common.sh"

target="${ICECARVER_ENABLE_TARGET:-OFF}"
case "${1:-}" in
  --target) target=ON ;;
  --public|'') ;;
  *) die "usage: $0 [--public|--target]" ;;
esac
case "${target^^}" in ON|1|TRUE) target=ON ;; OFF|0|FALSE) target=OFF ;; *) die "bad ICECARVER_ENABLE_TARGET" ;; esac

require_command cmake
require_command ninja
require_command nvcc
nvcc --version | grep -Eq 'release 12\.8([, ])' || die "CUDA Toolkit 12.8 is required"

assert_under_root "${ICECARVER_BUILD_DIR}"
log "configure build=${ICECARVER_BUILD_DIR} target=${target} arch=89"
cmake -S "${ICECARVER_ROOT}" -B "${ICECARVER_BUILD_DIR}" -G Ninja \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_CUDA_ARCHITECTURES=89 \
  -DICECARVER_ENABLE_TARGET="${target}" \
  -DICECARVER_BUILD_TESTS=ON
cmake --build "${ICECARVER_BUILD_DIR}" --parallel "${ICECARVER_BUILD_JOBS:-$(nproc)}"
ctest --test-dir "${ICECARVER_BUILD_DIR}" --output-on-failure
[[ -x "${ICECARVER_EVAL_BIN}" ]] || die "missing evaluator after build"
log "build complete: ${ICECARVER_EVAL_BIN}"
