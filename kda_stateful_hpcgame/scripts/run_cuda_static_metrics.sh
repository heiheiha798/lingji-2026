#!/usr/bin/env bash
set -euo pipefail

if (( $# < 2 || $# > 3 )); then
    echo "usage: $0 GENERATED_CUDA EXPERIMENT_NAME [DYNAMIC_SMEM_BYTES]" >&2
    exit 2
fi

source_path=$1
experiment_name=$2
if [[ ! $experiment_name =~ ^[A-Za-z0-9][A-Za-z0-9._-]*$ ]]; then
    echo "invalid experiment name: $experiment_name" >&2
    exit 2
fi

script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
task_dir=$(cd -- "$script_dir/.." && pwd)
python_bin=/data/home/tianjianyang/miniconda3/envs/lingji/bin/python
cuda_home=/usr/local/cuda-12.8
tilelang_package=/data/home/tianjianyang/miniconda3/envs/tilelang_kda_varlen_prefill/lib/python3.11/site-packages/tilelang
tilelang_include=$tilelang_package/src
cutlass_include=$tilelang_package/3rdparty/cutlass/include
host_compiler=/data/home/tianjianyang/miniconda3/bin/x86_64-conda-linux-gnu-c++
output_dir=$task_dir/compile_artifacts/$experiment_name

if [[ ! -f $source_path ]]; then
    echo "CUDA source does not exist: $source_path" >&2
    exit 1
fi
if [[ -e $output_dir ]]; then
    echo "experiment output already exists: $output_dir" >&2
    exit 1
fi
if [[ ! -x $python_bin ]]; then
    echo "lingji Python does not exist: $python_bin" >&2
    exit 1
fi
if [[ ! -x $cuda_home/bin/nvcc || ! -x $cuda_home/bin/nvdisasm ]]; then
    echo "CUDA 12.8 toolchain is incomplete under $cuda_home" >&2
    exit 1
fi
if [[ ! -d $tilelang_include || ! -d $cutlass_include ]]; then
    echo "TileLang v0.1.12 compile headers are incomplete under $tilelang_package" >&2
    exit 1
fi
if [[ ! -x $host_compiler ]]; then
    echo "host compiler does not exist: $host_compiler" >&2
    exit 1
fi

export CUDA_VISIBLE_DEVICES=""
export CUDA_HOME=$cuda_home
export CUDACXX=$cuda_home/bin/nvcc
metric_args=()
if (( $# == 3 )); then
    metric_args=(--dynamic-smem-bytes "$3")
fi

exec "$python_bin" "$task_dir/tools/cuda_static_metrics.py" \
    "$source_path" \
    --output-dir "$output_dir" \
    --host-compiler "$host_compiler" \
    --tilelang-include "$tilelang_include" \
    --cutlass-include "$cutlass_include" \
    "${metric_args[@]}"
