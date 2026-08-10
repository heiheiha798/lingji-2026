#!/usr/bin/env bash
set -euo pipefail

if (( $# != 5 )); then
    echo "usage: $0 {6|7} CASE CANDIDATE PROFILE_NAME DECODE_SAMPLE_STEPS" >&2
    exit 2
fi

gpu_id=$1
case_name=$2
candidate=$3
profile_name=$4
decode_sample_steps=$5
if [[ $gpu_id != 6 && $gpu_id != 7 ]]; then
    echo "GPU_ID must be physical GPU 6 or 7: $gpu_id" >&2
    exit 2
fi
if [[ ! $profile_name =~ ^[A-Za-z0-9][A-Za-z0-9._-]*$ ]]; then
    echo "invalid profile name: $profile_name" >&2
    exit 2
fi
case $case_name in
    L24-Z|L24-N|M24-C|M48-N|C48|C24|D48-B1|D48|D24) ;;
    *)
        echo "unknown case: $case_name" >&2
        exit 2
        ;;
esac
case $candidate in
    starter32|chunk32|chunk64|recurrent|hybrid64|hybrid64_direct_epi|hybrid256) ;;
    *)
        echo "unknown candidate: $candidate" >&2
        exit 2
        ;;
esac
if [[ ! $decode_sample_steps =~ ^[1-9][0-9]*$ ]]; then
    echo "DECODE_SAMPLE_STEPS must be a positive integer: $decode_sample_steps" >&2
    exit 2
fi

script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
task_dir=$(cd -- "$script_dir/.." && pwd)
python_bin=/data/home/tianjianyang/miniconda3/envs/lingji/bin/python
cuda_home=/usr/local/cuda-12.8
ncu_bin=/usr/local/cuda-13.0/bin/ncu
fla_dir=/data/home/tianjianyang/code/tmp/flash-linear-attention
expected_fla_commit=7843b328b0d3860a66de4eb07ba28bb020ceb1d8
output_dir=$task_dir/ncu_reports/$profile_name
report_base=$output_dir/$case_name
report_path=$report_base.ncu-rep
kernel_regex='regex:^(l2norm_fwd_kernel|fused_beta_sigmoid_fwd_kernel|kda_gate_chunk_cumsum_vector_kernel|chunk_kda_fwd_kernel_intra_sub_chunk|chunk_kda_fwd_kernel_inter_solve_fused|recompute_w_u_fwd_kda_kernel|chunk_gated_delta_rule_fwd_kernel_h_blockdim64|chunk_gla_fwd_kernel_o|fused_recurrent_kda_fwd_kernel|layer_norm_gated_fwd_kernel)$'

if [[ ! -x $python_bin ]]; then
    echo "lingji Python does not exist: $python_bin" >&2
    exit 1
fi
if [[ ! -x $ncu_bin ]]; then
    echo "Nsight Compute does not exist: $ncu_bin" >&2
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
    echo "profile output already exists: $output_dir" >&2
    exit 1
fi

mkdir -p "$output_dir"
export CUDA_VISIBLE_DEVICES=$gpu_id
export CUDA_HOME=$cuda_home
export CUDACXX=$cuda_home/bin/nvcc
export FLA_DISABLE_BACKEND_DISPATCH=1
export FLA_GIT_COMMIT=$fla_commit
export PYTHONUNBUFFERED=1

"$ncu_bin" \
    --profile-from-start off \
    --target-processes application-only \
    --replay-mode kernel \
    --kernel-name-base function \
    --kernel-name "$kernel_regex" \
    --clock-control base \
    --pipeline-boost-state stable \
    --cache-control all \
    --apply-rules yes \
    --import-sass yes \
    --import-source yes \
    --source-folders "$task_dir,$fla_dir" \
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
    "$python_bin" "$task_dir/tools/profile_fla_sm89.py" \
        --mode benchmark \
        --cases "$case_name" \
        --candidates "$candidate" \
        --warmup 1 \
        --replays 1 \
        --decode-sample-steps "$decode_sample_steps" \
        --nvtx \
        --cuda-profiler-api \
        --output "$output_dir/result.json" \
    2>&1 | tee "$output_dir/run.log"

"$ncu_bin" --import "$report_path" --page raw --csv \
    --print-units base --print-fp > "$output_dir/raw.csv"
"$ncu_bin" --import "$report_path" --page details \
    --print-details all --print-summary per-kernel > "$output_dir/details.txt"
"$ncu_bin" --import "$report_path" --page source --print-source sass --csv \
    --print-units base --print-fp > "$output_dir/source-sass.csv"
"$ncu_bin" --import "$report_path" --page source --print-source ptx --csv \
    --print-units base --print-fp > "$output_dir/source-ptx.csv"

echo "report=$report_path"
