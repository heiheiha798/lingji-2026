from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import time
from pathlib import Path
from typing import Any

from benchmark import OpSpec
from cuda_static_metrics import analyze_cuda_source
from submission import (
    _compile_chunk_diagonal,
    _compile_chunk_inter,
    _compile_chunk_transform,
    _compile_transformed_chunk_output,
    _compile_transformed_state_scan,
)
from tilelang.env import env


CASES = {
    "K1": (4096, 32, 16),
    "K2": (8064, 8, 16),
    "K3": (16384, 1, 8),
    "K4": (32768, 4, 8),
}


def kernel_report(kernel: Any, stage: str, workdir: Path) -> dict[str, Any]:
    source = kernel.get_kernel_source()
    device_mod = kernel.adapter.device_mod
    if device_mod is None or len(device_mod.functions) != 1:
        raise RuntimeError(f"{stage} did not retain one lowered device function")
    attrs = next(iter(device_mod.functions.values())).attrs
    if "dyn_shared_memory_buf" not in attrs:
        raise RuntimeError(f"{stage} has no dyn_shared_memory_buf attribute")
    dynamic_smem_bytes = int(attrs["dyn_shared_memory_buf"])

    report = analyze_cuda_source(source, workdir)
    report["dynamic_smem_bytes"] = dynamic_smem_bytes
    report["source_exp_calls"] = len(re.findall(r"\bexpf?\s*\(", source))
    (workdir / "metrics.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--cases",
        nargs="+",
        choices=tuple(CASES),
        default=list(CASES),
    )
    parser.add_argument(
        "--stage",
        choices=(
            "preprocess",
            "diagonal",
            "inter",
            "transform",
            "state",
            "output",
            "tail",
            "both",
        ),
        default="both",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--artifacts-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error(f"output already exists: {args.output}")
    if args.artifacts_dir.exists():
        parser.error(f"artifacts directory already exists: {args.artifacts_dir}")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.artifacts_dir.mkdir(parents=True)
    env.disable_cache()
    if args.stage == "both":
        stages = ("diagonal", "inter", "transform", "state", "output")
    elif args.stage == "preprocess":
        stages = ("diagonal", "inter")
    elif args.stage == "tail":
        stages = ("transform", "state", "output")
    else:
        stages = (args.stage,)

    submission_path = Path(__file__).with_name("submission.py")
    result: dict[str, Any] = {
        "schema_version": 1,
        "completed": False,
        "target": "sm_89",
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "submission_sha256": hashlib.sha256(
            submission_path.read_bytes()
        ).hexdigest(),
        "artifacts_dir": str(args.artifacts_dir.resolve()),
        "cases": {},
    }
    for case_name in args.cases:
        total_tokens, num_sequences, num_heads = CASES[case_name]
        spec = OpSpec(
            total_tokens=total_tokens,
            num_sequences=num_sequences,
            num_heads=num_heads,
        )
        operator_elements = total_tokens * num_heads * 64
        token_head_elements = total_tokens * num_heads
        max_chunks = (total_tokens + 63) // 64 + num_sequences - 1
        scratch_offset = 4 * operator_elements + 10 * token_head_elements
        tail_chunk_bytes = num_heads * (
            128 * 128 * 2 + 3 * 64 * 128 * 2 + 128 * 4
        )
        segment_chunks = min(
            max_chunks,
            (spec.workspace_bytes - scratch_offset)
            // tail_chunk_bytes,
        )
        print(
            f"{case_name}: build T={total_tokens}, B={num_sequences}, "
            f"H={num_heads}, segment_chunks={segment_chunks}",
            flush=True,
        )
        started = time.perf_counter()
        kernels: dict[str, Any] = {}
        if "diagonal" in stages:
            kernels["diagonal"] = _compile_chunk_diagonal(
                total_tokens, num_sequences, num_heads
            )
        if "inter" in stages:
            kernels["inter"] = _compile_chunk_inter(
                total_tokens, num_sequences, num_heads
            )
        if "transform" in stages:
            kernels["transform"] = _compile_chunk_transform(
                total_tokens,
                num_sequences,
                num_heads,
                segment_chunks,
            )
        if "state" in stages:
            kernels["state"] = _compile_transformed_state_scan(
                total_tokens,
                num_sequences,
                num_heads,
                (
                    64
                    if num_sequences * num_heads >= 64
                    else (8 if num_sequences * num_heads <= 32 else 32)
                ),
                segment_chunks,
            )
        if "output" in stages:
            kernels["output"] = _compile_transformed_chunk_output(
                total_tokens,
                num_sequences,
                num_heads,
                segment_chunks,
            )
        build_seconds = time.perf_counter() - started

        case_artifacts = args.artifacts_dir / case_name.lower()
        case_report = {
            "total_tokens": total_tokens,
            "num_sequences": num_sequences,
            "num_heads": num_heads,
            "max_chunks": max_chunks,
            "segment_chunks": segment_chunks,
            "segments": (max_chunks + segment_chunks - 1) // segment_chunks,
            "scratch_offset_bytes": scratch_offset,
            "build_seconds": round(build_seconds, 3),
        }
        for stage in stages:
            case_report[stage] = kernel_report(
                kernels[stage], stage, case_artifacts / stage
            )
        result["cases"][case_name] = case_report
        args.output.write_text(
            json.dumps(result, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        summaries = []
        for stage in stages:
            stage_report = case_report[stage]
            summaries.append(
                f"{stage} regs={stage_report['registers']} "
                f"smem={stage_report['dynamic_smem_bytes']} "
                f"spill={stage_report['spill_store_bytes']}/"
                f"{stage_report['spill_load_bytes']}"
            )
        print(f"{case_name}: " + "; ".join(summaries), flush=True)

    result["completed"] = True
    args.output.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(f"report={args.output}", flush=True)


if __name__ == "__main__":
    main()
