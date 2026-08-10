"""Strict parser for the frozen contest manifest."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

from .contract import CaseSpec, validate_case


class ManifestError(ValueError):
    pass


@dataclass(frozen=True)
class Manifest:
    cases: tuple[CaseSpec, ...]


def load_manifest(path: Path) -> Manifest:
    try:
        root = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ManifestError("cannot read data/cases.json") from error
    root = _object(root, "root")
    if set(root) != {"cases"}:
        raise ManifestError("manifest field set changed")

    cases_raw = root.get("cases")
    if not isinstance(cases_raw, list) or not cases_raw:
        raise ManifestError("cases must be a non-empty array")
    cases = []
    seen_ids = set()
    for index, value in enumerate(cases_raw):
        item = _object(value, f"cases[{index}]")
        if set(item) != {"id", "n", "width", "weight"}:
            raise ManifestError("case field set changed")
        case_id = _string(item.get("id"), f"cases[{index}].id")
        if case_id in seen_ids:
            raise ManifestError("case ids must be unique")
        weight = _float(item.get("weight"), f"cases[{index}].weight")
        case = CaseSpec(
            case_id=case_id,
            n=_int(item.get("n"), f"cases[{index}].n"),
            width=_int(item.get("width"), f"cases[{index}].width"),
            weight=weight,
        )
        try:
            validate_case(case)
        except ValueError as error:
            raise ManifestError(str(error)) from error
        seen_ids.add(case_id)
        cases.append(case)

    weight_sum = math.fsum(item.weight for item in cases)
    if not math.isclose(weight_sum, 1.0, rel_tol=0.0, abs_tol=1e-15):
        raise ManifestError("scored case weights must sum to one")
    if not any(not item.scored for item in cases):
        raise ManifestError("at least one correctness-only edge case is required")

    return Manifest(cases=tuple(cases))


def _object(value: Any, label: str) -> Mapping[str, Any]:
    if not isinstance(value, dict):
        raise ManifestError(f"{label} must be an object")
    return value


def _string(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value:
        raise ManifestError(f"{label} must be a non-empty string")
    return value


def _int(value: Any, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ManifestError(f"{label} must be a nonnegative integer")
    return value


def _float(value: Any, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ManifestError(f"{label} must be numeric")
    result = float(value)
    if not math.isfinite(result) or result < 0.0:
        raise ManifestError(f"{label} must be finite and nonnegative")
    return result


__all__ = [
    "Manifest",
    "ManifestError",
    "load_manifest",
]
