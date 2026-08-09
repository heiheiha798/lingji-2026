#!/usr/bin/env bash
set -euo pipefail

if (( $# < 2 )); then
    echo "usage: $0 GPU_ID BENCHMARK_ARGUMENTS..." >&2
    exit 2
fi

gpu_id=$1
shift
if [[ ! $gpu_id =~ ^[0-9]+$ ]]; then
    echo "GPU_ID must be a non-negative integer: $gpu_id" >&2
    exit 2
fi

script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
task_dir=$(cd -- "$script_dir/.." && pwd)
python_bin=/data/home/tianjianyang/miniconda3/envs/tilelang_kda_varlen_prefill/bin/python
cuda_home=/usr/local/cuda-12.8

if [[ ! -x $python_bin ]]; then
    echo "dedicated Python does not exist: $python_bin" >&2
    exit 1
fi

export CUDA_VISIBLE_DEVICES=$gpu_id
export CUDA_HOME=$cuda_home
export CUDACXX=$cuda_home/bin/nvcc
exec "$python_bin" "$task_dir/benchmark_fla_baseline.py" "$@"
