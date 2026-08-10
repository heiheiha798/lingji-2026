#!/usr/bin/env python3
"""Compare explicitly selected stages in SM89 TileLang compile reports."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def load_report_group(paths: list[Path], label: str) -> tuple[str, dict[str, Any]]:
    target: str | None = None
    cases: dict[str, Any] = {}
    for path in paths:
        if not path.is_file():
            raise ValueError(f"{label} report does not exist: {path}")
        payload = json.loads(path.read_text(encoding="utf-8"))
        report_target = payload.get("target")
        if not isinstance(report_target, str):
            raise ValueError(f"{path} has no target")
        if target is None:
            target = report_target
        elif target != report_target:
            raise ValueError(
                f"{label} reports mix targets {target!r} and {report_target!r}"
            )
        report_cases = payload.get("cases")
        if not isinstance(report_cases, dict):
            raise ValueError(f"{path} has no cases object")
        for case_name, case_report in report_cases.items():
            if case_name in cases:
                raise ValueError(
                    f"{label} case {case_name} appears in more than one report"
                )
            cases[case_name] = case_report
    if target is None or not cases:
        raise ValueError(f"{label} report group is empty")
    return target, cases


def format_change(baseline: int, candidate: int) -> str:
    if baseline == candidate:
        return f"{baseline} -> {candidate} (0.0%)"
    if baseline == 0:
        return f"{baseline} -> {candidate} ({candidate - baseline:+d})"
    change = (candidate - baseline) / baseline
    return f"{baseline} -> {candidate} ({change:+.1%})"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare one or more baseline and candidate compile reports."
    )
    parser.add_argument("--baseline", nargs="+", type=Path, required=True)
    parser.add_argument("--candidate", nargs="+", type=Path, required=True)
    parser.add_argument("--stages", nargs="+", required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    if len(args.stages) != len(set(args.stages)):
        parser.error("--stages contains duplicates")
    if args.output is not None and args.output.exists():
        parser.error(f"output already exists: {args.output}")
    try:
        baseline_target, baseline_cases = load_report_group(args.baseline, "baseline")
        candidate_target, candidate_cases = load_report_group(args.candidate, "candidate")
    except (ValueError, json.JSONDecodeError) as error:
        parser.error(str(error))
    if baseline_target != candidate_target:
        parser.error(f"target mismatch: {baseline_target!r} != {candidate_target!r}")
    if baseline_target != "sm_89":
        parser.error(f"this tool is restricted to sm_89, got {baseline_target!r}")
    if baseline_cases.keys() != candidate_cases.keys():
        parser.error(
            "case mismatch: baseline="
            + ",".join(sorted(baseline_cases))
            + " candidate="
            + ",".join(sorted(candidate_cases))
        )

    columns = (
        ("Regs", "registers"),
        ("Dyn smem", "dynamic_smem_bytes"),
        ("Spill store", "spill_store_bytes"),
        ("Spill load", "spill_load_bytes"),
        ("SASS inst", "instructions"),
        ("HMMA", "hmma"),
        ("MUFU", "mufu"),
        ("LDG", "global_load"),
        ("Cubin bytes", "cubin_bytes"),
    )
    lines = [
        "| Case | Stage | " + " | ".join(label for label, _ in columns) + " |",
        "| --- | --- | " + " | ".join("---:" for _ in columns) + " |",
    ]
    for case_name in sorted(baseline_cases):
        for stage_name in args.stages:
            baseline_stage = baseline_cases[case_name].get(stage_name)
            candidate_stage = candidate_cases[case_name].get(stage_name)
            if not isinstance(baseline_stage, dict):
                parser.error(f"baseline {case_name}/{stage_name} is missing or invalid")
            if not isinstance(candidate_stage, dict):
                parser.error(f"candidate {case_name}/{stage_name} is missing or invalid")
            cells = []
            for _, metric_name in columns:
                baseline_value = baseline_stage.get(metric_name)
                candidate_value = candidate_stage.get(metric_name)
                if type(baseline_value) is not int or type(candidate_value) is not int:
                    parser.error(f"{case_name}/{stage_name} has invalid {metric_name}")
                cells.append(format_change(baseline_value, candidate_value))
            lines.append(
                f"| {case_name} | {stage_name} | " + " | ".join(cells) + " |"
            )
    output = "\n".join(lines) + "\n"
    if args.output is None:
        print(output, end="")
    else:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(output, encoding="utf-8")
        print(f"comparison={args.output}")


if __name__ == "__main__":
    main()
