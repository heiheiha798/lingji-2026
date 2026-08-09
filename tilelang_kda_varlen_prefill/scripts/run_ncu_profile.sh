#!/usr/bin/env bash
set -euo pipefail

if (( $# != 3 )); then
    echo "usage: $0 GPU_ID {K1|K2|K3|K4} PROFILE_NAME" >&2
    exit 2
fi

gpu_id=$1
case_name=$2
profile_name=$3
if [[ ! $gpu_id =~ ^[0-9]+$ ]]; then
    echo "GPU_ID must be a non-negative integer: $gpu_id" >&2
    exit 2
fi
if [[ ! $case_name =~ ^K[1-4]$ ]]; then
    echo "case must be one of K1, K2, K3, K4: $case_name" >&2
    exit 2
fi
if [[ ! $profile_name =~ ^[A-Za-z0-9][A-Za-z0-9._-]*$ ]]; then
    echo "invalid profile name: $profile_name" >&2
    exit 2
fi

script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
task_dir=$(cd -- "$script_dir/.." && pwd)
python_bin=/data/home/tianjianyang/miniconda3/envs/tilelang_kda_varlen_prefill/bin/python
ncu_bin=/usr/local/cuda-13.0/bin/ncu
cuda_home=/usr/local/cuda-12.8
output_dir=$task_dir/ncu_reports/$profile_name
report_base=$output_dir/$case_name
report_path=$report_base.ncu-rep

if [[ ! -x $python_bin ]]; then
    echo "dedicated Python does not exist: $python_bin" >&2
    exit 1
fi
if [[ ! -x $ncu_bin ]]; then
    echo "Nsight Compute does not exist: $ncu_bin" >&2
    exit 1
fi
if [[ -e $output_dir ]]; then
    echo "profile output already exists: $output_dir" >&2
    exit 1
fi

mkdir -p "$output_dir"
export CUDA_VISIBLE_DEVICES=$gpu_id
export CUDA_HOME=$cuda_home
export CUDACXX=$cuda_home/bin/nvcc
export PYTHONUNBUFFERED=1

"$ncu_bin" \
    --profile-from-start off \
    --target-processes application-only \
    --replay-mode kernel \
    --clock-control base \
    --pipeline-boost-state stable \
    --cache-control all \
    --apply-rules yes \
    --import-sass yes \
    --import-source yes \
    --source-folders "$task_dir" \
    --section LaunchStats \
    --section Occupancy \
    --section SpeedOfLight \
    --section ComputeWorkloadAnalysis \
    --section MemoryWorkloadAnalysis \
    --section SchedulerStats \
    --section WarpStateStats \
    --section InstructionStats \
    --section SourceCounters \
    --export "$report_base" \
    "$python_bin" "$task_dir/profile_ncu.py" --case "$case_name"

"$ncu_bin" --import "$report_path" --page raw --csv \
    --print-units base --print-fp > "$output_dir/raw.csv"
"$ncu_bin" --import "$report_path" --page details \
    --print-details all --print-summary per-kernel > "$output_dir/details.txt"
"$ncu_bin" --import "$report_path" --page source --print-source sass --csv \
    --print-units base --print-fp > "$output_dir/source-sass.csv"
"$ncu_bin" --import "$report_path" --page source --print-source ptx --csv \
    --print-units base --print-fp > "$output_dir/source-ptx.csv"

echo "report=$report_path"
