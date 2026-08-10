#!/usr/bin/env bash
set -euo pipefail

script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
task_dir=$(cd -- "$script_dir/.." && pwd)
python_bin=/data/home/tianjianyang/miniconda3/envs/lingji/bin/python

if [[ ! -x $python_bin ]]; then
    echo "lingji Python does not exist: $python_bin" >&2
    exit 1
fi

exec "$python_bin" "$task_dir/tools/compare_compile_reports.py" "$@"
