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
nsys_bin=/data/home/tianjianyang/download/nsys/bin/nsys
cuda_home=/usr/local/cuda-12.8
output_dir=/data/home/tianjianyang/code/tmp/kda-baseline-results/nsys/$profile_name
report_base=$output_dir/$case_name
report_path=$report_base.nsys-rep

if [[ ! -x $python_bin ]]; then
    echo "dedicated Python does not exist: $python_bin" >&2
    exit 1
fi
if [[ ! -x $nsys_bin ]]; then
    echo "Nsight Systems does not exist: $nsys_bin" >&2
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

"$nsys_bin" profile \
    --trace=cuda,nvtx \
    --sample=none \
    --cpuctxsw=none \
    --capture-range=cudaProfilerApi \
    --capture-range-end=stop \
    --force-overwrite=false \
    --output="$report_base" \
    "$python_bin" "$task_dir/profile_fla.py" --case "$case_name" \
    2>&1 | tee "$output_dir/run.log"

"$nsys_bin" stats --report cuda_gpu_kern_sum --format csv "$report_path" \
    > "$output_dir/kernel-summary.csv"
"$nsys_bin" stats --report cuda_gpu_trace --format csv "$report_path" \
    > "$output_dir/kernel-trace.csv"
"$nsys_bin" stats --report cuda_gpu_mem_time_sum --format csv "$report_path" \
    > "$output_dir/memory-summary.csv"
"$nsys_bin" stats --report nvtx_gpu_proj_sum --format csv "$report_path" \
    > "$output_dir/nvtx-gpu-summary.csv"

echo "report=$report_path"
