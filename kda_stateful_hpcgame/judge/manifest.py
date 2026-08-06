"""Strict parser for the single-entry KDA Judge manifest."""

from __future__ import annotations

from dataclasses import dataclass
import json
import math
from pathlib import Path
from typing import Any, Mapping

from .contract import CaseSpec, KDAConfig, Limits, StateMode
from .counter_v1 import (
    ELEMENT_MULTIPLIER,
    FIELD_IDS,
    FIELD_NAMES,
    LUT_INDEX_SHIFT,
    LUT_SIZE,
    MIX_MULTIPLIER_1,
    MIX_MULTIPLIER_2,
    SHARD_STEPS,
    TAG_CASE_MULTIPLIER,
    TAG_FIELD_MULTIPLIER,
    TOKEN_MULTIPLIER,
    frozen_case_field_tag,
)


CASE_ORDER = (
    "L24-Z",
    "L24-N",
    "M24-C",
    "M48-N",
    "C48",
    "C24",
    "D48-B1",
    "D48",
    "D24",
)
CASE_WEIGHTS = {
    "L24-Z": 0.125,
    "L24-N": 0.125,
    "M24-C": 0.05,
    "M48-N": 0.05,
    "C48": 0.125,
    "C24": 0.125,
    "D48-B1": 0.10,
    "D48": 0.15,
    "D24": 0.15,
}
CASE_TIMING_REPLAYS = {
    "L24-Z": 32,
    "L24-N": 32,
    "M24-C": 32,
    "M48-N": 32,
    "C48": 64,
    "C24": 64,
    "D48-B1": 3,
    "D48": 3,
    "D24": 3,
}
FIELD_ORDER = FIELD_NAMES
PROFILES = ("typical", "slow_decay", "near_no_decay", "strong_decay")


class ManifestError(ValueError):
    """The public workload manifest is malformed or internally inconsistent."""


@dataclass(frozen=True)
class CounterSpec:
    canaries: tuple[Mapping[str, Any], ...]


@dataclass(frozen=True)
class FrozenCase:
    case_id: str
    case_key: int
    weight: float
    layer: int
    head_start: int
    profile: str
    field_tags: Mapping[str, int]
    timing_replays: int
    case: CaseSpec


@dataclass(frozen=True)
class SimpleManifest:
    path: Path
    counter: CounterSpec
    state_checkpoints: tuple[int, ...]
    files: Mapping[str, str]
    environment: Mapping[str, Any]
    cases: tuple[FrozenCase, ...]


def load_manifest(
    path: str | Path,
    *,
    require_canaries: bool = True,
) -> SimpleManifest:
    manifest_path = Path(path).resolve()
    try:
        value = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ManifestError("cases.json is missing or invalid JSON") from error
    root = _object(value, "manifest")
    if root.get("schema_version") != "kda-simple-judge/v1":
        raise ManifestError("unsupported cases.json schema")
    if root.get("operator") != "stateful":
        raise ManifestError("manifest must define the stateful operator")
    if root.get("status") != "QUALIFIED":
        raise ManifestError("cases.json is not a qualified release manifest")

    dimensions = _object(root.get("dimensions"), "dimensions")
    key_dim = _positive_int(dimensions.get("key_dim"), "dimensions.key_dim")
    value_dim = _positive_int(dimensions.get("value_dim"), "dimensions.value_dim")
    if (key_dim, value_dim) != (128, 128):
        raise ManifestError("formal KDA dimensions must be K=V=128")

    limits_value = _object(root.get("limits"), "limits")
    limits = Limits(
        output_relative_l2=_nonnegative_float(
            limits_value.get("output_relative_l2"), "limits.output_relative_l2"
        ),
        state_relative_l2=_nonnegative_float(
            limits_value.get("state_relative_l2"), "limits.state_relative_l2"
        ),
        normalized_max=_nonnegative_float(
            limits_value.get("normalized_max"), "limits.normalized_max"
        ),
    )
    if limits_value.get("worst_group_relative_l2") != "REPORT_ONLY":
        raise ManifestError("worst-group Relative L2 must remain report-only")

    checkpoint_value = _object(
        root.get("decode_checkpoints"), "decode_checkpoints"
    )
    if checkpoint_value.get("output_validation") != (
        "ALL_STEPS_STREAMED_IN_64_STEP_SHARDS"
    ):
        raise ManifestError("formal Decode output validation must cover every step")
    if set(checkpoint_value) != {"output_validation", "state"}:
        raise ManifestError("decode_checkpoints contains unsupported fields")
    state_checkpoints = tuple(
        _positive_int(item, f"decode_checkpoints.state[{index}]")
        for index, item in enumerate(
            _array(checkpoint_value.get("state"), "decode_checkpoints.state")
        )
    )
    if state_checkpoints != (1, 17, 257, 4096, 16384):
        raise ManifestError("formal Decode state checkpoints changed")

    counter = _parse_counter(root.get("counter_v1"), require_canaries)
    _parse_timing(root.get("timing"))
    files = _object(root.get("files"), "files")
    if set(files) != {"generator_luts", "layer_state", "golden_dir"}:
        raise ManifestError("formal data file set changed")
    environment = _parse_environment(root.get("environment"))
    cases = _parse_cases(
        root.get("cases"),
        key_dim=key_dim,
        value_dim=value_dim,
        limits=limits,
    )
    if not math.isclose(
        math.fsum(item.weight for item in cases), 1.0, rel_tol=0.0, abs_tol=1e-15
    ):
        raise ManifestError("case weights must sum to exactly 1.0")

    return SimpleManifest(
        path=manifest_path,
        counter=counter,
        state_checkpoints=state_checkpoints,
        files={str(key): _string(item, f"files.{key}") for key, item in files.items()},
        environment=environment,
        cases=cases,
    )


def _parse_counter(value: Any, require_canaries: bool) -> CounterSpec:
    counter = _object(value, "counter_v1")
    expected_keys = {
        "arithmetic",
        "token_multiplier",
        "element_multiplier",
        "mix",
        "lut_index_shift",
        "lut_entries",
        "shard_steps",
        "tag_case_multiplier",
        "tag_field_multiplier",
        "field_ids",
        "flattening",
        "decode_step_origin",
        "canaries",
    }
    if set(counter) != expected_keys:
        raise ManifestError("counter_v1 field set changed")
    if counter.get("arithmetic") != "uint32_mod_2^32":
        raise ManifestError("counter_v1 arithmetic must be uint32 modulo 2^32")
    frozen_scalars = (
        (
            _hex_u32(counter.get("token_multiplier"), "token_multiplier"),
            TOKEN_MULTIPLIER,
        ),
        (
            _hex_u32(counter.get("element_multiplier"), "element_multiplier"),
            ELEMENT_MULTIPLIER,
        ),
        (
            _positive_int(counter.get("lut_index_shift"), "lut_index_shift"),
            LUT_INDEX_SHIFT,
        ),
        (
            _positive_int(counter.get("lut_entries"), "lut_entries"),
            LUT_SIZE,
        ),
        (
            _positive_int(counter.get("shard_steps"), "shard_steps"),
            SHARD_STEPS,
        ),
        (
            _hex_u32(counter.get("tag_case_multiplier"), "tag_case_multiplier"),
            TAG_CASE_MULTIPLIER,
        ),
        (
            _hex_u32(counter.get("tag_field_multiplier"), "tag_field_multiplier"),
            TAG_FIELD_MULTIPLIER,
        ),
    )
    if any(actual != expected for actual, expected in frozen_scalars):
        raise ManifestError("counter_v1 constants differ from the frozen generator")
    mix = _array(counter.get("mix"), "counter_v1.mix")
    expected_mix_shape = (
        ("xor_right", 16),
        ("multiply", f"0x{MIX_MULTIPLIER_1:08x}"),
        ("xor_right", 15),
        ("multiply", f"0x{MIX_MULTIPLIER_2:08x}"),
        ("xor_right", 16),
    )
    normalized_mix = tuple(tuple(item) for item in mix)
    if normalized_mix != expected_mix_shape:
        raise ManifestError("counter_v1 integer permutation changed")
    field_ids_raw = _object(counter.get("field_ids"), "counter_v1.field_ids")
    if tuple(field_ids_raw) != FIELD_ORDER:
        raise ManifestError("counter_v1 field order changed")
    field_ids = {
        field: _nonnegative_int(field_ids_raw.get(field), f"field_ids.{field}")
        for field in FIELD_ORDER
    }
    if field_ids != dict(FIELD_IDS):
        raise ManifestError("counter_v1 field IDs changed")
    expected_flattening = (
        "element_index=((batch*heads+head)*channels+channel); "
        "case_index=case_key-1; case_field_tag=((case_index+1)*"
        "tag_case_multiplier) XOR ((field_id+1)*tag_field_multiplier), all uint32"
    )
    if counter.get("flattening") != expected_flattening:
        raise ManifestError("counter_v1 flattening order changed")
    if _nonnegative_int(counter.get("decode_step_origin"), "decode_step_origin") != 0:
        raise ManifestError("counter_v1 Decode step origin changed")
    canaries = tuple(
        _object(item, "counter_v1.canary")
        for item in _array(counter.get("canaries"), "counter_v1.canaries")
    )
    if require_canaries and not canaries:
        raise ManifestError("counter_v1 canaries are not frozen")
    return CounterSpec(canaries=canaries)


def _parse_timing(value: Any) -> None:
    timing = _object(value, "timing")
    if set(timing) != {
        "event_scope",
        "case_time",
        "aggregation",
    }:
        raise ManifestError("timing field set changed")
    if (
        timing.get("event_scope")
        != "one_current_stream_cuda_event_span_per_append_or_decode_call"
        or timing.get("case_time")
        != "arithmetic_mean_of_fresh_state_replays"
        or timing.get("aggregation")
        != "weighted_geometric_mean_of_case_gpu_time_ms"
    ):
        raise ManifestError("formal timing contract changed")


def _parse_environment(value: Any) -> Mapping[str, Any]:
    environment = _object(value, "environment")
    expected_keys = {
        "gpu_name_contains",
        "gpu_count",
        "max_start_temperature_c",
        "max_end_temperature_c",
        "required_graphics_clock_mhz",
        "graphics_clock_tolerance_mhz",
        "required_memory_clock_mhz",
        "memory_clock_tolerance_mhz",
        "required_pstate",
        "required_power_limit_w",
        "other_compute_processes",
    }
    if set(environment) != expected_keys:
        raise ManifestError("formal environment field set changed")
    parsed = {
        "gpu_name_contains": _string(
            environment.get("gpu_name_contains"), "environment.gpu_name_contains"
        ),
        "gpu_count": _positive_int(
            environment.get("gpu_count"), "environment.gpu_count"
        ),
        "max_start_temperature_c": _nonnegative_float(
            environment.get("max_start_temperature_c"),
            "environment.max_start_temperature_c",
        ),
        "max_end_temperature_c": _nonnegative_float(
            environment.get("max_end_temperature_c"),
            "environment.max_end_temperature_c",
        ),
        "required_graphics_clock_mhz": _positive_float(
            environment.get("required_graphics_clock_mhz"),
            "environment.required_graphics_clock_mhz",
        ),
        "graphics_clock_tolerance_mhz": _nonnegative_float(
            environment.get("graphics_clock_tolerance_mhz"),
            "environment.graphics_clock_tolerance_mhz",
        ),
        "required_memory_clock_mhz": _positive_float(
            environment.get("required_memory_clock_mhz"),
            "environment.required_memory_clock_mhz",
        ),
        "memory_clock_tolerance_mhz": _nonnegative_float(
            environment.get("memory_clock_tolerance_mhz"),
            "environment.memory_clock_tolerance_mhz",
        ),
        "required_pstate": _string(
            environment.get("required_pstate"), "environment.required_pstate"
        ),
        "required_power_limit_w": _positive_float(
            environment.get("required_power_limit_w"),
            "environment.required_power_limit_w",
        ),
        "other_compute_processes": _nonnegative_int(
            environment.get("other_compute_processes"),
            "environment.other_compute_processes",
        ),
    }
    if parsed["max_end_temperature_c"] < parsed["max_start_temperature_c"]:
        raise ManifestError("end temperature limit must not be below start limit")
    return parsed


def _parse_cases(
    value: Any,
    *,
    key_dim: int,
    value_dim: int,
    limits: Limits,
) -> tuple[FrozenCase, ...]:
    rows = _array(value, "cases")
    if tuple(row.get("id") if isinstance(row, dict) else None for row in rows) != CASE_ORDER:
        raise ManifestError("formal case roster or order changed")
    result = []
    case_keys = set()
    for index, raw in enumerate(rows):
        row = _object(raw, f"cases[{index}]")
        case_id = _string(row.get("id"), f"cases[{index}].id")
        batch = _positive_int(row.get("batch"), f"{case_id}.batch")
        heads = _positive_int(row.get("heads"), f"{case_id}.heads")
        decode_steps = _nonnegative_int(
            row.get("decode_steps"), f"{case_id}.decode_steps"
        )
        append_lengths = []
        for append_index, raw_append in enumerate(
            _array(row.get("append_calls"), f"{case_id}.append_calls")
        ):
            append = _object(raw_append, f"{case_id}.append_calls[{append_index}]")
            if append.get("kind") == "uniform":
                append_lengths.append(
                    (_positive_int(append.get("length"), "append.length"),) * batch
                )
            elif append.get("kind") == "per_sequence":
                lengths = tuple(
                    _positive_int(item, "append.lengths")
                    for item in _array(append.get("lengths"), "append.lengths")
                )
                if len(lengths) != batch:
                    raise ManifestError(f"{case_id} packed lengths must contain B values")
                append_lengths.append(lengths)
            else:
                raise ManifestError(f"{case_id} has an unsupported append kind")
        state_mode_text = _string(row.get("state_mode"), f"{case_id}.state_mode")
        try:
            state_mode = StateMode(state_mode_text)
        except ValueError as error:
            raise ManifestError(f"{case_id} has an unsupported state mode") from error
        case = CaseSpec(
            case_id=case_id,
            config=KDAConfig(heads=heads, key_dim=key_dim, value_dim=value_dim),
            batch=batch,
            state_mode=state_mode,
            limits=limits,
            append_lengths=tuple(append_lengths),
            decode_steps=decode_steps,
        )
        case_key = _positive_int(row.get("case_key"), f"{case_id}.case_key")
        if case_key in case_keys or case_key > 255:
            raise ManifestError("case_key values must be unique uint8 values")
        case_keys.add(case_key)
        profile = _string(row.get("profile"), f"{case_id}.profile")
        if profile not in PROFILES:
            raise ManifestError(f"{case_id} has an unsupported profile")
        field_tags_raw = _object(row.get("field_tags"), f"{case_id}.field_tags")
        if tuple(field_tags_raw) != FIELD_ORDER:
            raise ManifestError(f"{case_id}.field_tags order changed")
        field_tags = {
            field: _uint32(field_tags_raw.get(field), f"{case_id}.field_tags.{field}")
            for field in FIELD_ORDER
        }
        expected_tags = {
            field: frozen_case_field_tag(case_key, field)
            for field in FIELD_ORDER
        }
        if field_tags != expected_tags:
            raise ManifestError(f"{case_id}.field_tags do not match the frozen formula")
        weight = _positive_float(row.get("weight"), f"{case_id}.weight")
        if not math.isclose(
            weight, CASE_WEIGHTS[case_id], rel_tol=0.0, abs_tol=1e-15
        ):
            raise ManifestError(f"{case_id}.weight changed")
        timing_replays = _positive_int(
            row.get("timing_replays"), f"{case_id}.timing_replays"
        )
        if timing_replays != CASE_TIMING_REPLAYS[case_id]:
            raise ManifestError(f"{case_id}.timing_replays changed")
        result.append(
            FrozenCase(
                case_id=case_id,
                case_key=case_key,
                weight=weight,
                layer=_nonnegative_int(row.get("layer"), f"{case_id}.layer"),
                head_start=_nonnegative_int(
                    row.get("head_start"), f"{case_id}.head_start"
                ),
                profile=profile,
                field_tags=field_tags,
                timing_replays=timing_replays,
                case=case,
            )
        )
    return tuple(result)


def _object(value: Any, label: str) -> Mapping[str, Any]:
    if not isinstance(value, dict):
        raise ManifestError(f"{label} must be a JSON object")
    return value


def _array(value: Any, label: str) -> list[Any]:
    if not isinstance(value, list):
        raise ManifestError(f"{label} must be a JSON array")
    return value


def _string(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value:
        raise ManifestError(f"{label} must be a nonempty string")
    return value


def _identifier(value: Any, label: str) -> str:
    text = _string(value, label)
    if not text.isidentifier():
        raise ManifestError(f"{label} must be a Python identifier")
    return text


def _positive_int(value: Any, label: str) -> int:
    if type(value) is not int or value <= 0:
        raise ManifestError(f"{label} must be a positive integer")
    return value


def _nonnegative_int(value: Any, label: str) -> int:
    if type(value) is not int or value < 0:
        raise ManifestError(f"{label} must be a nonnegative integer")
    return value


def _positive_float(value: Any, label: str) -> float:
    number = _nonnegative_float(value, label)
    if number <= 0:
        raise ManifestError(f"{label} must be positive")
    return number


def _nonnegative_float(value: Any, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ManifestError(f"{label} must be numeric")
    number = float(value)
    if not math.isfinite(number) or number < 0:
        raise ManifestError(f"{label} must be finite and nonnegative")
    return number


def _hex_u32(value: Any, label: str) -> int:
    text = _string(value, f"counter_v1.{label}")
    try:
        number = int(text, 16)
    except ValueError as error:
        raise ManifestError(f"counter_v1.{label} must be hexadecimal") from error
    if not 0 <= number <= 0xFFFFFFFF:
        raise ManifestError(f"counter_v1.{label} must fit uint32")
    return number


def _uint32(value: Any, label: str) -> int:
    try:
        number = int(value, 0) if isinstance(value, str) else value
    except ValueError as error:
        raise ManifestError(f"{label} must be a uint32 integer") from error
    if type(number) is not int or not 0 <= number <= 0xFFFFFFFF:
        raise ManifestError(f"{label} must be a uint32 integer")
    return number


__all__ = [
    "CASE_ORDER",
    "FIELD_ORDER",
    "FrozenCase",
    "ManifestError",
    "SimpleManifest",
    "load_manifest",
]
