"""Trusted data loading and counter materialization for the simple Judge."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

import torch

from .contract import (
    AppendInputs,
    LayerParams,
    append_result_key,
    decode_result_key,
)
from .counter_v1 import (
    BF16_FIELDS,
    FIELD_NAMES,
    CounterCaseManifest,
    CounterV1Generator,
    GeneratorLUTs,
    SHARD_STEPS,
)
from .manifest import FrozenCase, SimpleManifest


class DataError(RuntimeError):
    """A trusted data payload or counter canary is invalid."""


@dataclass(frozen=True)
class CaseStaticData:
    layer: LayerParams
    initial_state: torch.Tensor
    append_inputs: tuple[AppendInputs, ...]


@dataclass(frozen=True)
class CaseGoldens:
    append_output: Mapping[str, torch.Tensor]
    state: Mapping[str, torch.Tensor]
    decode_directory: Path | None


def counter_case(
    frozen: FrozenCase,
    manifest: SimpleManifest,
) -> CounterCaseManifest:
    return CounterCaseManifest(
        case_id=frozen.case_id,
        case_index=frozen.case_key - 1,
        batch=frozen.case.batch,
        heads=frozen.case.config.heads,
        key_dim=frozen.case.config.key_dim,
        value_dim=frozen.case.config.value_dim,
        append_token_counts=tuple(
            frozen.case.append_tokens(index)
            for index in range(frozen.case.append_calls)
        ),
        decode_steps=frozen.case.decode_steps,
        regime=frozen.profile,
        field_tags=frozen.field_tags,
    )


def load_case_static(
    manifest: SimpleManifest,
    frozen: FrozenCase,
    generator: CounterV1Generator,
    *,
    data_dir: str | Path,
    device: torch.device | str,
) -> CaseStaticData:
    from safetensors import safe_open

    root = Path(data_dir)
    static_path = root / manifest.files["layer_state"]
    prefix = frozen.case_id
    try:
        with safe_open(str(static_path), framework="pt", device="cpu") as handle:
            a_log = handle.get_tensor(f"{prefix}.a_log")
            dt_bias = handle.get_tensor(f"{prefix}.dt_bias")
            norm = handle.get_tensor(f"{prefix}.output_norm_weight")
            state = handle.get_tensor(f"{prefix}.initial_state")
    except Exception as error:
        raise DataError(f"cannot load layer/state tensors for {prefix}") from error
    target = torch.device(device)
    layer = LayerParams(
        a_log=a_log.to(target),
        dt_bias=dt_bias.to(target),
        output_norm_weight=norm.to(target),
    )
    initial_state = state.to(target)
    generated = generator.generate_all_appends()
    inputs = []
    for lengths, views in zip(frozen.case.append_lengths, generated):
        offsets = [0]
        rows = []
        for sequence_id, length in enumerate(lengths):
            sequence_start = offsets[-1]
            offsets.append(sequence_start + length)
            for local_start in range(0, length, 64):
                rows.append(
                    (
                        sequence_id,
                        sequence_start + local_start,
                        min(64, length - local_start),
                        sequence_id,
                    )
                )
        inputs.append(
            AppendInputs(
                q_act=views.value_views["q_act"],
                k_act=views.value_views["k_act"],
                v_act=views.value_views["v_act"],
                g_raw=views.value_views["g_raw"],
                beta_raw=views.value_views["beta_raw"],
                output_gate_logits=views.value_views["output_gate_logits"],
                cu_seqlens=torch.tensor(
                    offsets, dtype=torch.int32, device=target
                ),
                descriptor=torch.tensor(rows, dtype=torch.int32, device=target),
            )
        )
    return CaseStaticData(
        layer=layer,
        initial_state=initial_state,
        append_inputs=tuple(inputs),
    )


def load_case_goldens(
    manifest: SimpleManifest,
    frozen: FrozenCase,
    *,
    data_dir: str | Path,
    device: torch.device | str,
) -> CaseGoldens:
    from safetensors.torch import load_file

    root = Path(data_dir) / manifest.files["golden_dir"]
    try:
        append_output = load_file(
            str(root / f"{frozen.case_id}.append_output.safetensors"),
            device=str(torch.device(device)),
        )
        state = load_file(
            str(root / f"{frozen.case_id}.state.safetensors"),
            device=str(torch.device(device)),
        )
    except Exception as error:
        raise DataError(f"cannot load golden tensors for {frozen.case_id}") from error
    expected_output = {
        append_result_key(index): (
            frozen.case.append_tokens(index),
            frozen.case.config.heads,
            frozen.case.config.value_dim,
        )
        for index in range(frozen.case.append_calls)
    }
    expected_state_keys = {
        *expected_output,
        *(
            decode_result_key(step)
            for step in manifest.state_checkpoints
            if step <= frozen.case.decode_steps
        ),
    }
    if set(append_output) != set(expected_output):
        raise DataError(
            f"golden append-output key set mismatch for {frozen.case_id}"
        )
    if set(state) != expected_state_keys:
        raise DataError(f"golden state key set mismatch for {frozen.case_id}")
    for key, expected_shape in expected_output.items():
        tensor = append_output[key]
        if tuple(tensor.shape) != expected_shape or tensor.dtype != torch.bfloat16:
            raise DataError(f"golden append-output shape/dtype mismatch: {key}")
    state_shape = (
        frozen.case.batch,
        frozen.case.config.heads,
        frozen.case.config.value_dim,
        frozen.case.config.key_dim,
    )
    if any(
        tuple(tensor.shape) != state_shape or tensor.dtype != torch.float32
        for tensor in state.values()
    ):
        raise DataError(f"golden state shape/dtype mismatch for {frozen.case_id}")
    return CaseGoldens(
        append_output=append_output,
        state=state,
        decode_directory=(root / frozen.case_id if frozen.case.decode_steps else None),
    )


def decode_output_shard_filename(shard_start: int, count: int) -> str:
    """Return the frozen one-based, inclusive Decode shard filename."""

    if type(shard_start) is not int or shard_start < 0:
        raise ValueError("shard_start must be nonnegative")
    if type(count) is not int or count <= 0:
        raise ValueError("Decode shard count must be positive")
    first_step = shard_start + 1
    final_step = shard_start + count
    return f"decode_output_{first_step:05d}_{final_step:05d}.safetensors"


def iter_decode_shards(
    frozen: FrozenCase,
    *,
    shard_steps: int,
) -> tuple[tuple[int, int], ...]:
    if type(shard_steps) is not int or shard_steps <= 0:
        raise ValueError("shard_steps must be positive")
    return tuple(
        (start, min(shard_steps, frozen.case.decode_steps - start))
        for start in range(0, frozen.case.decode_steps, shard_steps)
    )


def load_decode_output_shard(
    goldens: CaseGoldens,
    *,
    shard_start: int,
    count: int,
    destination: torch.Tensor,
) -> torch.Tensor:
    """Load one CPU-mapped golden shard into a reusable device buffer."""

    from safetensors import safe_open

    if goldens.decode_directory is None:
        raise DataError("a non-Decode case has no Decode golden directory")
    if destination.ndim != 4 or destination.dtype != torch.bfloat16:
        raise DataError("Decode golden destination must be a rank-4 BF16 tensor")
    if count > destination.shape[0]:
        raise DataError("Decode golden destination is smaller than the shard")
    path = goldens.decode_directory / decode_output_shard_filename(shard_start, count)
    try:
        with safe_open(str(path), framework="pt", device="cpu") as handle:
            if set(handle.keys()) != {"output"}:
                raise DataError(f"Decode golden key set mismatch: {path.name}")
            source = handle.get_tensor("output")
            expected_shape = (count, *destination.shape[1:])
            if tuple(source.shape) != expected_shape or source.dtype != torch.bfloat16:
                raise DataError(f"Decode golden shape/dtype mismatch: {path.name}")
            view = destination[:count]
            view.copy_(source)
    except DataError:
        raise
    except Exception as error:
        raise DataError(f"cannot load Decode golden shard {path.name}") from error
    return view


def expected_data_relative_paths(manifest: SimpleManifest) -> tuple[str, ...]:
    """Enumerate the exact frozen payload set covered by SHA256SUMS."""

    golden_dir = manifest.files["golden_dir"]
    paths = [
        "cases.json",
        manifest.files["generator_luts"],
        manifest.files["layer_state"],
        f"{golden_dir}/REFERENCE.json",
    ]
    for frozen in manifest.cases:
        paths.extend(
            (
                f"{golden_dir}/{frozen.case_id}.append_output.safetensors",
                f"{golden_dir}/{frozen.case_id}.state.safetensors",
            )
        )
        paths.extend(
            f"{golden_dir}/{frozen.case_id}/"
            f"{decode_output_shard_filename(start, count)}"
            for start, count in iter_decode_shards(
                frozen, shard_steps=SHARD_STEPS
            )
        )
    return tuple(paths)


def verify_data_sha256(
    manifest: SimpleManifest,
    *,
    data_dir: str | Path,
) -> Mapping[str, str]:
    root = Path(data_dir).resolve()
    checksum_path = root / "SHA256SUMS"
    try:
        lines = checksum_path.read_text(encoding="ascii").splitlines()
    except OSError as error:
        raise DataError("data/SHA256SUMS is missing") from error
    entries: dict[str, str] = {}
    for line_number, raw in enumerate(lines, start=1):
        fields = raw.strip().split()
        if not fields:
            continue
        if len(fields) != 2:
            raise DataError(f"invalid SHA256SUMS line {line_number}")
        digest, relative = fields
        relative = relative.removeprefix("*").replace("\\", "/")
        path = Path(relative)
        if (
            path.is_absolute()
            or ".." in path.parts
            or relative in entries
            or len(digest) != 64
            or any(character not in "0123456789abcdefABCDEF" for character in digest)
        ):
            raise DataError(f"unsafe SHA256SUMS line {line_number}")
        entries[relative] = digest.lower()
    expected = set(expected_data_relative_paths(manifest))
    if set(entries) != expected:
        raise DataError(
            f"SHA256SUMS payload set mismatch: missing={sorted(expected-set(entries))}, "
            f"extra={sorted(set(entries)-expected)}"
        )
    for relative, expected_digest in entries.items():
        path = (root / relative).resolve()
        try:
            path.relative_to(root)
        except ValueError as error:
            raise DataError(f"checksum path escapes data directory: {relative}") from error
        if not path.is_file() or _sha256(path) != expected_digest:
            raise DataError(f"data checksum mismatch: {relative}")
    verify_golden_reference(manifest, data_dir=root)
    return entries


def verify_golden_reference(
    manifest: SimpleManifest,
    *,
    data_dir: str | Path,
) -> None:
    """Verify the small reference metadata without scanning the tensor payload."""

    root = Path(data_dir).resolve()
    reference_path = root / manifest.files["golden_dir"] / "REFERENCE.json"
    try:
        reference = json.loads(reference_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise DataError("invalid golden REFERENCE.json") from error
    if not isinstance(reference, dict):
        raise DataError("golden REFERENCE.json must contain one object")
    required = {
        "schema_version": "kda-golden-reference/v4",
        "status": "QUALIFIED_INDEPENDENT_FP32",
        "algorithm": "direct post-update KDA recurrence",
        "uses_fla": False,
        "state_dtype": "float32",
        "raw_output_boundary_dtype": "bfloat16",
        "final_output_dtype": "bfloat16",
        "decode_output_validation": "ALL_STEPS_STREAMED_IN_64_STEP_SHARDS",
    }
    if any(reference.get(key) != value for key, value in required.items()):
        raise DataError("golden reference is not the qualified independent FP32 oracle")
    expected_hashes = {
        "counter_manifest_sha256": _sha256(manifest.path),
        "counter_luts_sha256": _sha256(
            root / manifest.files["generator_luts"]
        ),
    }
    if any(reference.get(key) != value for key, value in expected_hashes.items()):
        raise DataError("golden reference does not bind the current counter data")
    if set(reference) != {*required, *expected_hashes}:
        raise DataError("golden reference contains unsupported metadata")


def verify_counter_canaries(
    manifest: SimpleManifest,
    luts: GeneratorLUTs,
    *,
    device: torch.device | str,
) -> None:
    by_case: dict[str, list[Mapping[str, Any]]] = {}
    for row in manifest.counter.canaries:
        case_id = row.get("case_id")
        if not isinstance(case_id, str):
            raise DataError("counter canary is missing case_id")
        by_case.setdefault(case_id, []).append(row)
    frozen_by_id = {item.case_id: item for item in manifest.cases}
    decode_case_ids = {
        item.case_id for item in manifest.cases if item.case.decode_steps
    }
    if set(by_case) != decode_case_ids:
        raise DataError(
            "counter canaries must cover every Decode case exactly"
        )
    for case_id, rows in by_case.items():
        frozen = frozen_by_id[case_id]
        generator = CounterV1Generator(
            counter_case(frozen, manifest), luts, device=device
        )
        buffers = generator.allocate_decode_buffers()
        rows_by_token: dict[int, list[Mapping[str, Any]]] = {}
        for row in rows:
            token_index = row.get("token_index")
            if type(token_index) is not int:
                raise DataError("counter canary token_index must be an integer")
            rows_by_token.setdefault(token_index, []).append(row)
        for token_index, token_rows in rows_by_token.items():
            generator.fill_decode_shard(buffers, token_index, 1)
            torch.cuda.current_stream().synchronize() if torch.device(device).type == "cuda" else None
            for row in token_rows:
                field = row.get("field")
                element_index = row.get("element_index")
                if field not in FIELD_NAMES or type(element_index) is not int:
                    raise DataError("counter canary field/index is invalid")
                flat = buffers.staging.bit_views[field][0].reshape(-1)
                if not 0 <= element_index < flat.numel():
                    raise DataError("counter canary element index is outside its tensor")
                carrier = int(flat[element_index].item())
                mask = 0xFFFF if field in BF16_FIELDS else 0xFFFFFFFF
                actual = carrier & mask
                try:
                    expected = int(str(row.get("value_bits")), 16)
                except ValueError as error:
                    raise DataError("counter canary value_bits is invalid") from error
                if actual != expected:
                    raise DataError(
                        f"counter canary mismatch: {case_id} step={row.get('step')} "
                        f"field={field} element={element_index}: {actual:#x}!={expected:#x}"
                    )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


__all__ = [
    "CaseGoldens",
    "CaseStaticData",
    "DataError",
    "counter_case",
    "decode_output_shard_filename",
    "expected_data_relative_paths",
    "iter_decode_shards",
    "load_case_goldens",
    "load_case_static",
    "load_decode_output_shard",
    "verify_counter_canaries",
    "verify_data_sha256",
    "verify_golden_reference",
]
