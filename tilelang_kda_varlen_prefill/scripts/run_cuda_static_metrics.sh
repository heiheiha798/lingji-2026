#!/usr/bin/env bash
set -euo pipefail

if (( $# < 2 || $# > 3 )); then
    echo "usage: $0 GENERATED_CUDA OUTPUT_DIR [DYNAMIC_SMEM_BYTES]" >&2
    exit 2
fi

source_path=$1
output_dir=$2
script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
task_dir=$(cd -- "$script_dir/.." && pwd)
python_bin=/data/home/tianjianyang/miniconda3/envs/tilelang_kda_varlen_prefill/bin/python
cuda_home=/usr/local/cuda-12.8

if [[ ! -f $source_path ]]; then
    echo "CUDA source does not exist: $source_path" >&2
    exit 1
fi
if [[ -e $output_dir ]]; then
    echo "output directory already exists: $output_dir" >&2
    exit 1
fi
if [[ ! -x $python_bin ]]; then
    echo "dedicated Python does not exist: $python_bin" >&2
    exit 1
fi
if [[ ! -x $cuda_home/bin/nvcc || ! -x $cuda_home/bin/nvdisasm ]]; then
    echo "CUDA 12.8 toolchain is incomplete under $cuda_home" >&2
    exit 1
fi

export CUDA_VISIBLE_DEVICES=""
export CUDA_HOME=$cuda_home
export CUDACXX=$cuda_home/bin/nvcc
metric_args=()
if (( $# == 3 )); then
    metric_args=(--dynamic-smem-bytes "$3")
fi

exec "$python_bin" "$task_dir/cuda_static_metrics.py" \
    "$source_path" --output-dir "$output_dir" "${metric_args[@]}"
