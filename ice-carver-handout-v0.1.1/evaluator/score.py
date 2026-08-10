#!/usr/bin/env python3
"""Validate evaluator evidence and aggregate the 70-point result score.

The official case set, weights, and run policy come only from the trusted
manifest.  Result JSON is untrusted evidence: this program validates its
structure and cross-field invariants before using any timing or correctness
claim.  No third-party Python packages are required.
"""

from __future__ import annotations

import argparse
import json
import math
import re
import statistics
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable


DEFAULT_MANIFEST = Path(__file__).with_name("official_manifest.json")
_UINT64_RE = re.compile(r"^(0|[1-9][0-9]*)$")
_TIMING_REL_TOLERANCE = 1.0e-9
_TIMING_ABS_TOLERANCE_MS = 1.0e-9


@dataclass(frozen=True)
class CaseRule:
    case_id: str
    num_isovalues: int
    correctness_weight: float
    performance_weight: float


@dataclass(frozen=True)
class OfficialManifest:
    version: str
    warmup_runs: int
    measure_runs: int
    correctness_total: float
    performance_total: float
    cases: tuple[CaseRule, ...]

    @property
    def by_id(self) -> dict[str, CaseRule]:
        return {case.case_id: case for case in self.cases}

    @property
    def case_ids(self) -> set[str]:
        return {case.case_id for case in self.cases}

    @property
    def performance_case_ids(self) -> set[str]:
        return {
            case.case_id for case in self.cases if case.performance_weight > 0.0
        }


def _reject_json_constant(token: str) -> None:
    raise ValueError(f"non-standard JSON numeric constant {token!r} is forbidden")


def _object_without_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON object key {key!r}")
        result[key] = value
    return result


def _reject_nonfinite_tree(value: Any, label: str = "JSON") -> None:
    # parse_constant rejects NaN/Infinity.  This second pass also catches a
    # standards-compliant but overflowing literal such as 1e999.
    if type(value) is float and not math.isfinite(value):
        raise ValueError(f"{label}: every number must be finite")
    if isinstance(value, list):
        for index, item in enumerate(value):
            _reject_nonfinite_tree(item, f"{label}[{index}]")
    elif isinstance(value, dict):
        for key, item in value.items():
            _reject_nonfinite_tree(item, f"{label}.{key}")


def load_strict_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        document = json.load(
            handle,
            parse_constant=_reject_json_constant,
            object_pairs_hook=_object_without_duplicate_keys,
        )
    _reject_nonfinite_tree(document, str(path))
    return document


def _number(value: Any, label: str, *, minimum: float | None = None) -> float:
    if type(value) not in (int, float):
        raise ValueError(f"{label} must be a JSON number")
    try:
        number = float(value)
    except OverflowError as exc:
        raise ValueError(f"{label} must be finite") from exc
    if not math.isfinite(number):
        raise ValueError(f"{label} must be finite")
    if minimum is not None and number < minimum:
        raise ValueError(f"{label} must be at least {minimum}")
    return number


def _positive_number(value: Any, label: str) -> float:
    number = _number(value, label)
    if number <= 0.0:
        raise ValueError(f"{label} must be positive")
    return number


def _integer(value: Any, label: str, *, minimum: int | None = None) -> int:
    if type(value) is not int:
        raise ValueError(f"{label} must be a JSON integer")
    if minimum is not None and value < minimum:
        raise ValueError(f"{label} must be at least {minimum}")
    return value


def _boolean(value: Any, label: str) -> bool:
    if type(value) is not bool:
        raise ValueError(f"{label} must be a JSON boolean")
    return value


def _mapping(value: Any, label: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"{label} must be a JSON object")
    return value


def _sequence(value: Any, label: str) -> list[Any]:
    if not isinstance(value, list):
        raise ValueError(f"{label} must be a JSON array")
    return value


def _same_number(left: float, right: float) -> bool:
    return math.isclose(left, right, rel_tol=1.0e-12, abs_tol=1.0e-12)


def load_manifest(path: Path) -> OfficialManifest:
    document = _mapping(load_strict_json(path), str(path))
    expected_keys = {
        "schema_version",
        "problem",
        "manifest_version",
        "score_totals",
        "run_policy",
        "cases",
    }
    if set(document) != expected_keys:
        missing = sorted(expected_keys - set(document))
        extra = sorted(set(document) - expected_keys)
        raise ValueError(
            f"{path}: manifest keys do not match; missing={missing}, extra={extra}"
        )
    if document["schema_version"] != "1.0" or document["problem"] != "ice-carver":
        raise ValueError(f"{path}: unsupported manifest schema or problem")
    version = document["manifest_version"]
    if not isinstance(version, str) or not version:
        raise ValueError(f"{path}: manifest_version must be a non-empty string")

    totals = _mapping(document["score_totals"], f"{path}:score_totals")
    if set(totals) != {"correctness", "absolute_performance"}:
        raise ValueError(f"{path}: score_totals has unexpected keys")
    correctness_total = _number(
        totals["correctness"], f"{path}:score_totals.correctness", minimum=0.0
    )
    performance_total = _number(
        totals["absolute_performance"],
        f"{path}:score_totals.absolute_performance",
        minimum=0.0,
    )
    if not _same_number(correctness_total, 40.0) or not _same_number(
        performance_total, 30.0
    ):
        raise ValueError(f"{path}: official result totals must be 40 and 30")

    policy = _mapping(document["run_policy"], f"{path}:run_policy")
    if set(policy) != {"warmup_runs", "measure_runs"}:
        raise ValueError(f"{path}: run_policy has unexpected keys")
    warmup_runs = _integer(
        policy["warmup_runs"], f"{path}:run_policy.warmup_runs", minimum=0
    )
    measure_runs = _integer(
        policy["measure_runs"], f"{path}:run_policy.measure_runs", minimum=1
    )
    if warmup_runs != 2 or measure_runs != 5:
        raise ValueError(f"{path}: official run policy must be 2 warmups + 5 measures")

    case_documents = _sequence(document["cases"], f"{path}:cases")
    if not case_documents:
        raise ValueError(f"{path}: manifest contains no cases")
    cases: list[CaseRule] = []
    seen: set[str] = set()
    for index, value in enumerate(case_documents):
        label = f"{path}:cases[{index}]"
        case = _mapping(value, label)
        expected_case_keys = {
            "id",
            "num_isovalues",
            "correctness_weight",
            "performance_weight",
        }
        if set(case) != expected_case_keys:
            raise ValueError(f"{label}: case rule has unexpected or missing keys")
        case_id = case["id"]
        if not isinstance(case_id, str) or not case_id:
            raise ValueError(f"{label}.id must be a non-empty string")
        if case_id in seen:
            raise ValueError(f"{path}: duplicate manifest case {case_id}")
        seen.add(case_id)
        num_isovalues = _integer(
            case["num_isovalues"], f"{label}.num_isovalues", minimum=1
        )
        if num_isovalues > 8:
            raise ValueError(f"{label}.num_isovalues cannot exceed 8")
        cases.append(
            CaseRule(
                case_id=case_id,
                num_isovalues=num_isovalues,
                correctness_weight=_number(
                    case["correctness_weight"],
                    f"{label}.correctness_weight",
                    minimum=0.0,
                ),
                performance_weight=_number(
                    case["performance_weight"],
                    f"{label}.performance_weight",
                    minimum=0.0,
                ),
            )
        )

    if not _same_number(
        sum(case.correctness_weight for case in cases), correctness_total
    ):
        raise ValueError(f"{path}: case correctness weights do not sum to 40")
    if not _same_number(
        sum(case.performance_weight for case in cases), performance_total
    ):
        raise ValueError(f"{path}: case performance weights do not sum to 30")

    return OfficialManifest(
        version=version,
        warmup_runs=warmup_runs,
        measure_runs=measure_runs,
        correctness_total=correctness_total,
        performance_total=performance_total,
        cases=tuple(cases),
    )


def _validate_counts(
    value: Any, count: int, label: str, *, allow_empty: bool
) -> list[int]:
    values = _sequence(value, label)
    allowed_lengths = {count, 0} if allow_empty else {count}
    if len(values) not in allowed_lengths:
        expectation = f"0 or {count}" if allow_empty else str(count)
        raise ValueError(f"{label} must contain exactly {expectation} entries")
    return [
        _integer(item, f"{label}[{index}]", minimum=0)
        for index, item in enumerate(values)
    ]


def _validate_run(
    run_value: Any,
    *,
    case_id: str,
    phase: str,
    index: int,
    num_isovalues: int,
) -> tuple[float, bool]:
    label = f"{case_id}:{phase}[{index}]"
    run = _mapping(run_value, label)
    if run.get("phase") != phase:
        raise ValueError(f"{label}.phase must be {phase!r}")
    if _integer(run.get("index"), f"{label}.index", minimum=0) != index:
        raise ValueError(f"{label}.index must be {index}")
    seed = run.get("seed")
    if not isinstance(seed, str) or _UINT64_RE.fullmatch(seed) is None:
        raise ValueError(f"{label}.seed must be a canonical uint64 decimal string")
    if int(seed) > 18446744073709551615:
        raise ValueError(f"{label}.seed exceeds uint64")

    solver_status = _integer(run.get("solver_status"), f"{label}.solver_status")
    solver_status_name = run.get("solver_status_name")
    if not isinstance(solver_status_name, str) or not solver_status_name:
        raise ValueError(f"{label}.solver_status_name must be a non-empty string")
    reference_status = _integer(
        run.get("reference_status"), f"{label}.reference_status"
    )
    reference_status_name = run.get("reference_status_name")
    if not isinstance(reference_status_name, str) or not reference_status_name:
        raise ValueError(f"{label}.reference_status_name must be a non-empty string")
    timed_out = _boolean(run.get("timed_out"), f"{label}.timed_out")
    cuda_ms = _number(run.get("cuda_time_ms"), f"{label}.cuda_time_ms", minimum=0.0)
    wall_ms = _number(run.get("wall_time_ms"), f"{label}.wall_time_ms", minimum=0.0)
    stated_official_ms = _number(
        run.get("official_time_ms"), f"{label}.official_time_ms", minimum=0.0
    )
    recomputed_official_ms = max(cuda_ms, wall_ms)
    if not math.isclose(
        stated_official_ms,
        recomputed_official_ms,
        rel_tol=_TIMING_REL_TOLERANCE,
        abs_tol=_TIMING_ABS_TOLERANCE_MS,
    ):
        raise ValueError(
            f"{label}.official_time_ms is inconsistent with max(cuda_time_ms, wall_time_ms)"
        )

    validation = _mapping(run.get("validation"), f"{label}.validation")
    validation_correct = _boolean(
        validation.get("correct"), f"{label}.validation.correct"
    )
    validation_flags: dict[str, bool] = {}
    for name in (
        "counts_match",
        "capacity_ok",
        "finite",
        "guards_intact",
        "values_within_tolerance",
    ):
        validation_flags[name] = _boolean(
            validation.get(name), f"{label}.validation.{name}"
        )
    validation_counters: dict[str, int] = {}
    for name in ("mismatched_values", "nonfinite_values", "guard_corruptions"):
        validation_counters[name] = _integer(
            validation.get(name), f"{label}.validation.{name}", minimum=0
        )
    _number(
        validation.get("max_abs_error"),
        f"{label}.validation.max_abs_error",
        minimum=0.0,
    )
    _number(
        validation.get("max_rel_error"),
        f"{label}.validation.max_rel_error",
        minimum=0.0,
    )
    message = validation.get("message")
    if not isinstance(message, str):
        raise ValueError(f"{label}.validation.message must be a string")
    candidate_counts = _validate_counts(
        validation.get("candidate_counts"),
        num_isovalues,
        f"{label}.validation.candidate_counts",
        allow_empty=validation_correct is False,
    )
    reference_counts = _validate_counts(
        validation.get("reference_counts"),
        num_isovalues,
        f"{label}.validation.reference_counts",
        allow_empty=validation_correct is False,
    )
    if len(candidate_counts) != len(reference_counts):
        raise ValueError(
            f"{label}.validation candidate/reference count arrays must have equal lengths"
        )

    success_evidence = (
        solver_status == 0
        and reference_status == 0
        and timed_out is False
        and validation_correct is True
    )
    if validation_correct is True:
        if not success_evidence:
            raise ValueError(
                f"{label}.validation.correct=true requires successful statuses and no timeout"
            )
        if stated_official_ms <= 0.0:
            raise ValueError(f"{label}: a successful run must have positive timing")
        for name, flag in validation_flags.items():
            if flag is not True:
                raise ValueError(
                    f"{label}.validation.correct cannot be true when {name} is false"
                )
        for name, counter in validation_counters.items():
            if counter != 0:
                raise ValueError(
                    f"{label}.validation.correct cannot be true when {name} is nonzero"
                )
        if candidate_counts != reference_counts:
            raise ValueError(
                f"{label}.validation.correct cannot be true when counts differ"
            )

    return recomputed_official_ms, success_evidence


def _validate_case(case_value: Any, rule: CaseRule, manifest: OfficialManifest, source: str) -> dict[str, Any]:
    case = _mapping(case_value, f"{source}:case")
    case_id = case.get("id")
    if case_id != rule.case_id:
        raise ValueError(f"{source}: expected case {rule.case_id}, got {case_id!r}")
    num_isovalues = _integer(
        case.get("num_isovalues"), f"{source}:{case_id}.num_isovalues", minimum=1
    )
    if num_isovalues != rule.num_isovalues:
        raise ValueError(
            f"{source}:{case_id}.num_isovalues disagrees with official manifest"
        )
    isovalues = _sequence(case.get("isovalues"), f"{source}:{case_id}.isovalues")
    if len(isovalues) != num_isovalues:
        raise ValueError(f"{source}:{case_id}.isovalues has the wrong length")
    for index, value in enumerate(isovalues):
        _number(value, f"{source}:{case_id}.isovalues[{index}]")

    if _integer(case.get("warmup_runs"), f"{source}:{case_id}.warmup_runs") != manifest.warmup_runs:
        raise ValueError(f"{source}:{case_id} must declare exactly 2 warmup runs")
    if _integer(case.get("measure_runs"), f"{source}:{case_id}.measure_runs") != manifest.measure_runs:
        raise ValueError(f"{source}:{case_id} must declare exactly 5 measured runs")
    stated_correctness_weight = _number(
        case.get("correctness_weight"), f"{source}:{case_id}.correctness_weight"
    )
    stated_performance_weight = _number(
        case.get("performance_weight"), f"{source}:{case_id}.performance_weight"
    )
    if not _same_number(stated_correctness_weight, rule.correctness_weight) or not _same_number(
        stated_performance_weight, rule.performance_weight
    ):
        raise ValueError(f"{source}:{case_id} weights disagree with official manifest")

    case_correct = _boolean(case.get("correct"), f"{source}:{case_id}.correct")
    runs = _sequence(case.get("runs"), f"{source}:{case_id}.runs")
    expected_run_count = manifest.warmup_runs + manifest.measure_runs
    if case_correct is True and len(runs) != expected_run_count:
        raise ValueError(
            f"{source}:{case_id}.runs must contain exactly 2 warmups + 5 measures when correct=true"
        )
    if case_correct is False and len(runs) > expected_run_count:
        raise ValueError(
            f"{source}:{case_id}.runs cannot exceed the official 2+5 run sequence"
        )

    measured_times: list[float] = []
    run_success: list[bool] = []
    seeds: set[str] = set()
    for run_index, run_value in enumerate(runs):
        is_warmup = run_index < manifest.warmup_runs
        phase = "warmup" if is_warmup else "measure"
        phase_index = run_index if is_warmup else run_index - manifest.warmup_runs
        official_ms, success = _validate_run(
            run_value,
            case_id=case_id,
            phase=phase,
            index=phase_index,
            num_isovalues=num_isovalues,
        )
        run = _mapping(run_value, f"{source}:{case_id}.runs[{run_index}]")
        seed = run["seed"]
        if seed in seeds:
            raise ValueError(f"{source}:{case_id} repeats run seed {seed}")
        seeds.add(seed)
        if not is_warmup:
            measured_times.append(official_ms)
        run_success.append(success)

    stated_median_value = case.get("median_time_ms")
    recomputed_median_ms: float | None = None
    if case_correct is True:
        # The true path is deliberately all-or-nothing.  This also covers both
        # warmups: the C++ evaluator cannot emit case.correct=true after any
        # failed run.
        if len(run_success) != expected_run_count or not all(run_success):
            raise ValueError(
                f"{source}:{case_id} claims correct=true without complete successful 2+5 evidence"
            )
        recomputed_median_ms = statistics.median(measured_times)
        stated_median_ms = _positive_number(
            stated_median_value, f"{source}:{case_id}.median_time_ms"
        )
        if not math.isclose(
            stated_median_ms,
            recomputed_median_ms,
            rel_tol=_TIMING_REL_TOLERANCE,
            abs_tol=_TIMING_ABS_TOLERANCE_MS,
        ):
            raise ValueError(
                f"{source}:{case_id}.median_time_ms does not equal the median of the 5 official times"
            )
    elif stated_median_value is not None:
        # A failed run can be appended before the C++ evaluator adds its time
        # to measured_times, so a partial prefix does not carry enough evidence
        # to reproduce that partial median.  Validate its type/range but never
        # use it for scoring.
        _positive_number(
            stated_median_value, f"{source}:{case_id}.median_time_ms"
        )

    return {
        "id": case_id,
        "correct": case_correct is True,
        "median_time_ms": recomputed_median_ms,
    }


def _validate_evaluator_document(
    document: Any, manifest: OfficialManifest, source: str
) -> dict[str, Any]:
    entry = _mapping(document, source)
    if entry.get("schema_version") != "1.0" or entry.get("problem") != "ice-carver":
        raise ValueError(f"{source}: unsupported evaluator result schema or problem")
    if entry.get("solver") not in ("public", "target"):
        raise ValueError(f"{source}: solver must be 'public' or 'target'")
    case = _mapping(entry.get("case"), f"{source}:case")
    case_id = case.get("id")
    if not isinstance(case_id, str) or not case_id:
        raise ValueError(f"{source}: case id is missing or invalid")
    rule = manifest.by_id.get(case_id)
    if rule is None:
        raise ValueError(f"{source}: unexpected case id {case_id!r}")
    return _validate_case(case, rule, manifest, source)


def load_submission_results(
    paths: list[Path], manifest: OfficialManifest
) -> list[dict[str, Any]]:
    results: dict[str, dict[str, Any]] = {}
    for path in paths:
        document = load_strict_json(path)
        documents = document if isinstance(document, list) else [document]
        for index, entry in enumerate(documents):
            source = str(path) if len(documents) == 1 else f"{path}[{index}]"
            normalized = _validate_evaluator_document(entry, manifest, source)
            case_id = normalized["id"]
            if case_id in results:
                raise ValueError(f"duplicate submission result for {case_id}")
            results[case_id] = normalized

    found = set(results)
    if found != manifest.case_ids:
        missing = sorted(manifest.case_ids - found)
        unexpected = sorted(found - manifest.case_ids)
        raise ValueError(
            f"submission case set does not match official manifest; missing={missing}, unexpected={unexpected}"
        )
    return [results[rule.case_id] for rule in manifest.cases]


def _calibration_entry(value: Any, case_id: str, source: str) -> float:
    label = f"{source}:{case_id} calibration time"
    if isinstance(value, dict):
        allowed = {"time_ms", "median_time_ms"} & set(value)
        if len(allowed) != 1:
            raise ValueError(
                f"{source}:{case_id} calibration object needs exactly one of time_ms or median_time_ms"
            )
        value = value[next(iter(allowed))]
    return _positive_number(value, label)


def _extract_calibration_entries(
    document: Any, source: str, manifest: OfficialManifest
) -> Iterable[tuple[str, float]]:
    if isinstance(document, list):
        for index, item in enumerate(document):
            normalized = _validate_evaluator_document(
                item, manifest, f"{source}[{index}]"
            )
            if normalized["correct"] is not True:
                raise ValueError(f"{source}[{index}]: calibration case is not correct")
            yield normalized["id"], normalized["median_time_ms"]
        return

    mapping = _mapping(document, source)
    if "case" in mapping:
        normalized = _validate_evaluator_document(mapping, manifest, source)
        if normalized["correct"] is not True:
            raise ValueError(f"{source}: calibration case is not correct")
        yield normalized["id"], normalized["median_time_ms"]
        return

    entries: Any
    if "cases" in mapping:
        entries = mapping["cases"]
    elif "times_ms" in mapping:
        entries = mapping["times_ms"]
    else:
        entries = mapping

    if isinstance(entries, list):
        for index, value in enumerate(entries):
            item = _mapping(value, f"{source}:cases[{index}]")
            case_id = item.get("id")
            if not isinstance(case_id, str) or not case_id:
                raise ValueError(f"{source}:cases[{index}].id is invalid")
            yield case_id, _calibration_entry(item, case_id, source)
        return

    entries = _mapping(entries, f"{source}:calibration map")
    for case_id, value in entries.items():
        if not isinstance(case_id, str) or not case_id:
            raise ValueError(f"{source}: calibration case id is invalid")
        yield case_id, _calibration_entry(value, case_id, source)


def load_time_map(
    paths: list[Path], label: str, manifest: OfficialManifest
) -> dict[str, float]:
    times: dict[str, float] = {}
    for path in paths:
        document = load_strict_json(path)
        for case_id, time_ms in _extract_calibration_entries(
            document, str(path), manifest
        ):
            if case_id not in manifest.case_ids:
                raise ValueError(f"{path}: unexpected {label} calibration {case_id}")
            if case_id in times:
                raise ValueError(f"duplicate {label} calibration for {case_id}")
            times[case_id] = time_ms
    missing = manifest.performance_case_ids - set(times)
    if missing:
        raise ValueError(f"missing {label} calibration cases: {sorted(missing)}")
    return times


def calculate_score(
    cases: list[dict[str, Any]],
    baseline: dict[str, float],
    target: dict[str, float],
    manifest: OfficialManifest,
) -> dict[str, Any]:
    by_id = {case["id"]: case for case in cases}
    if set(by_id) != manifest.case_ids or len(by_id) != len(cases):
        raise ValueError("calculate_score requires the exact unique manifest case set")

    correctness_score = 0.0
    performance_score = 0.0
    log_speedup_sum = 0.0
    eligible = True
    scored_cases: list[dict[str, Any]] = []

    for rule in manifest.cases:
        case = by_id[rule.case_id]
        # Identity is intentional: 1, "true", and other truthy values are not
        # correctness evidence.
        correct = case["correct"] is True
        if correct:
            correctness_score += rule.correctness_weight
        elif rule.performance_weight > 0.0:
            # The published ranking gate covers P4--P7: exactly the cases with
            # positive performance weight.  A failed correctness-only case
            # still loses its correctness points, but does not invalidate the
            # independently measured ranking metric.
            eligible = False

        case_performance_score = 0.0
        acceleration: float | None = None
        if rule.performance_weight > 0.0:
            base_ms = _positive_number(
                baseline.get(rule.case_id), f"{rule.case_id} baseline time"
            )
            target_ms = _positive_number(
                target.get(rule.case_id), f"{rule.case_id} target time"
            )
            if not target_ms < base_ms:
                raise ValueError(
                    f"{rule.case_id}: target ({target_ms} ms) must be faster than baseline ({base_ms} ms)"
                )
            if correct:
                measured_ms = _positive_number(
                    case["median_time_ms"], f"{rule.case_id} measured time"
                )
                interpolation = math.log(base_ms / measured_ms) / math.log(
                    base_ms / target_ms
                )
                case_performance_score = rule.performance_weight * min(
                    1.0, max(0.0, interpolation)
                )
                acceleration = base_ms / measured_ms
                log_speedup_sum += rule.performance_weight * math.log(acceleration)
            performance_score += case_performance_score

        scored_cases.append(
            {
                "id": rule.case_id,
                "correct": correct,
                "correctness_score": rule.correctness_weight if correct else 0.0,
                "absolute_performance_score": case_performance_score,
                "acceleration": acceleration,
            }
        )

    if correctness_score > manifest.correctness_total + 1.0e-9:
        raise ValueError("internal error: correctness score exceeds manifest total")
    if performance_score > manifest.performance_total + 1.0e-9:
        raise ValueError("internal error: performance score exceeds manifest total")
    performance_metric = (
        math.exp(log_speedup_sum / manifest.performance_total)
        if eligible and manifest.performance_total > 0.0
        else None
    )
    result_score = correctness_score + performance_score
    return {
        "schema_version": "1.0",
        "problem": "ice-carver",
        "manifest_version": manifest.version,
        "correctness_score": correctness_score,
        "absolute_performance_score": performance_score,
        "result_score": result_score,
        "performance_metric": performance_metric,
        "eligible_for_ranking": eligible and performance_metric is not None,
        "cases": scored_cases,
    }


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--manifest",
        type=Path,
        default=DEFAULT_MANIFEST,
        help=(
            "organizer-controlled official manifest "
            f"(default: {DEFAULT_MANIFEST})"
        ),
    )
    parser.add_argument(
        "--results",
        nargs="+",
        type=Path,
        required=True,
        help="one or more evaluator JSON files; together they must contain every official case exactly once",
    )
    parser.add_argument(
        "--baseline",
        nargs="+",
        type=Path,
        required=True,
        help="trusted public-baseline calibration JSON file(s)",
    )
    parser.add_argument(
        "--target",
        nargs="+",
        type=Path,
        required=True,
        help="trusted target calibration JSON file(s)",
    )
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        manifest = load_manifest(args.manifest)
        results = load_submission_results(args.results, manifest)
        baseline = load_time_map(args.baseline, "baseline", manifest)
        target = load_time_map(args.target, "target", manifest)
        score = calculate_score(results, baseline, target, manifest)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("w", encoding="utf-8", newline="\n") as handle:
            json.dump(score, handle, ensure_ascii=False, indent=2, allow_nan=False)
            handle.write("\n")
    except (OSError, ValueError, TypeError, json.JSONDecodeError) as exc:
        print(f"score.py: error: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
