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
from submission import Submission
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
        print(
            f"{case_name}: build T={total_tokens}, B={num_sequences}, "
            f"H={num_heads}",
            flush=True,
        )
        started = time.perf_counter()
        chunk_operators, persistent_recurrence, _ = Submission().build(spec)
        build_seconds = time.perf_counter() - started

        case_artifacts = args.artifacts_dir / case_name.lower()
        case_report = {
            "total_tokens": total_tokens,
            "num_sequences": num_sequences,
            "num_heads": num_heads,
            "build_seconds": round(build_seconds, 3),
            "preprocess": kernel_report(
                chunk_operators,
                "preprocess",
                case_artifacts / "preprocess",
            ),
            "persistent": kernel_report(
                persistent_recurrence,
                "persistent",
                case_artifacts / "persistent",
            ),
        }
        result["cases"][case_name] = case_report
        args.output.write_text(
            json.dumps(result, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        preprocess = case_report["preprocess"]
        persistent = case_report["persistent"]
        print(
            f"{case_name}: preprocess regs={preprocess['registers']} "
            f"smem={preprocess['dynamic_smem_bytes']} "
            f"spill={preprocess['spill_store_bytes']}/"
            f"{preprocess['spill_load_bytes']}; persistent "
            f"regs={persistent['registers']} "
            f"smem={persistent['dynamic_smem_bytes']} "
            f"spill={persistent['spill_store_bytes']}/"
            f"{persistent['spill_load_bytes']}",
            flush=True,
        )

    result["completed"] = True
    args.output.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(f"report={args.output}", flush=True)


if __name__ == "__main__":
    main()
