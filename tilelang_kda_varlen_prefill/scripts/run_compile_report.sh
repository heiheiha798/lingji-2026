#!/usr/bin/env bash
set -euo pipefail

if (( $# < 1 )); then
    echo "usage: $0 EXPERIMENT_NAME [K1 K2 K3 K4]" >&2
    exit 2
fi

experiment_name=$1
shift
if [[ ! $experiment_name =~ ^[A-Za-z0-9][A-Za-z0-9._-]*$ ]]; then
    echo "invalid experiment name: $experiment_name" >&2
    exit 2
fi

script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
task_dir=$(cd -- "$script_dir/.." && pwd)
python_bin=/data/home/tianjianyang/miniconda3/envs/tilelang_kda_varlen_prefill/bin/python
cuda_home=/usr/local/cuda-12.8
report_dir=$task_dir/compile_reports
artifact_dir=$task_dir/compile_artifacts/$experiment_name
report_path=$report_dir/$experiment_name.json
log_path=$report_dir/$experiment_name.log

if [[ ! -x $python_bin ]]; then
    echo "dedicated Python does not exist: $python_bin" >&2
    exit 1
fi
if [[ ! -x $cuda_home/bin/nvcc || ! -x $cuda_home/bin/nvdisasm ]]; then
    echo "CUDA 12.8 toolchain is incomplete under $cuda_home" >&2
    exit 1
fi
if [[ -e $report_path || -e $log_path || -e $artifact_dir ]]; then
    echo "experiment output already exists: $experiment_name" >&2
    exit 1
fi

mkdir -p "$report_dir" "$task_dir/compile_artifacts"
export CUDA_VISIBLE_DEVICES=""
export CUDA_HOME=$cuda_home
export CUDACXX=$cuda_home/bin/nvcc
export PYTHONUNBUFFERED=1

case_args=()
if (( $# > 0 )); then
    case_args=(--cases "$@")
fi

echo "experiment=$experiment_name report=$report_path artifacts=$artifact_dir"
"$python_bin" "$task_dir/compile_report.py" \
    "${case_args[@]}" \
    --output "$report_path" \
    --artifacts-dir "$artifact_dir" 2>&1 | tee "$log_path"
