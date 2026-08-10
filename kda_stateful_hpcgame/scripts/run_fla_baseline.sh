#!/usr/bin/env bash
set -euo pipefail

if (( $# < 2 )); then
    echo "usage: $0 {6|7} RUN_NAME [PROFILE_ARGUMENTS...]" >&2
    exit 2
fi

gpu_id=$1
run_name=$2
shift 2
if [[ $gpu_id != 6 && $gpu_id != 7 ]]; then
    echo "GPU_ID must be physical GPU 6 or 7: $gpu_id" >&2
    exit 2
fi
if [[ ! $run_name =~ ^[A-Za-z0-9][A-Za-z0-9._-]*$ ]]; then
    echo "invalid run name: $run_name" >&2
    exit 2
fi

script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
task_dir=$(cd -- "$script_dir/.." && pwd)
python_bin=/data/home/tianjianyang/miniconda3/envs/lingji/bin/python
cuda_home=/usr/local/cuda-12.8
fla_dir=/data/home/tianjianyang/code/tmp/flash-linear-attention
expected_fla_commit=7843b328b0d3860a66de4eb07ba28bb020ceb1d8
output_dir=$task_dir/artifacts/baseline/$run_name

if [[ ! -x $python_bin ]]; then
    echo "lingji Python does not exist: $python_bin" >&2
    exit 1
fi
if [[ ! -x $cuda_home/bin/nvcc ]]; then
    echo "CUDA compiler does not exist: $cuda_home/bin/nvcc" >&2
    exit 1
fi
if [[ ! -d $fla_dir/.git ]]; then
    echo "FLA repository does not exist: $fla_dir" >&2
    exit 1
fi
fla_commit=$(git -C "$fla_dir" rev-parse HEAD)
if [[ $fla_commit != "$expected_fla_commit" ]]; then
    echo "FLA commit mismatch: expected $expected_fla_commit, got $fla_commit" >&2
    exit 1
fi
if [[ -e $output_dir ]]; then
    echo "run output already exists: $output_dir" >&2
    exit 1
fi

mkdir -p "$output_dir"
export CUDA_VISIBLE_DEVICES=$gpu_id
export CUDA_HOME=$cuda_home
export CUDACXX=$cuda_home/bin/nvcc
export FLA_DISABLE_BACKEND_DISPATCH=1
export FLA_GIT_COMMIT=$fla_commit
export PYTHONUNBUFFERED=1

"$python_bin" "$task_dir/tools/profile_fla_sm89.py" \
    "$@" \
    --output "$output_dir/result.json" \
    2>&1 | tee "$output_dir/run.log"
