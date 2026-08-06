"""Frozen counter_v1 tensor generation for trusted KDA inputs.

The counter maps ``(token_index, element_index, case_field_tag)`` to one of
256 stored bit patterns.  CPU and Triton paths write integer carrier views;
BF16/FP32 tensors are reinterpretations of those same bytes.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Mapping, Sequence

import torch

try:
    import triton
    import triton.language as tl
except ModuleNotFoundError:  # CPU-only publication and verifier environments.
    triton = None
    tl = None


COUNTER_VERSION = "counter_v1"
TOKEN_MULTIPLIER = 0x9E3779B1
ELEMENT_MULTIPLIER = 0x85EBCA77
MIX_MULTIPLIER_1 = 0x7FEB352D
MIX_MULTIPLIER_2 = 0x846CA68B
TAG_CASE_MULTIPLIER = 0x1F123BB5
TAG_FIELD_MULTIPLIER = 0xA5A5A5A5
UINT32_MASK = 0xFFFFFFFF
LUT_SIZE = 256
LUT_INDEX_SHIFT = 24
SHARD_STEPS = 64
LUT_FILENAME = "generator_luts.safetensors"

FIELD_NAMES = (
    "q_act",
    "k_act",
    "v_act",
    "g_raw",
    "beta_raw",
    "output_gate_logits",
)
FIELD_IDS = MappingProxyType({name: index for index, name in enumerate(FIELD_NAMES)})
BF16_FIELDS = frozenset(name for name in FIELD_NAMES if name != "beta_raw")
DECAY_REGIMES = (
    "typical",
    "slow_decay",
    "near_no_decay",
    "strong_decay",
)
G_RAW_LUT_KEYS = MappingProxyType(
    {regime: f"g_raw.{regime}" for regime in DECAY_REGIMES}
)
ALL_LUT_KEYS = frozenset(
    {
        "q_act",
        "k_act",
        "v_act",
        "beta_raw",
        "output_gate_logits",
        *G_RAW_LUT_KEYS.values(),
    }
)


def _u32(value: object, label: str) -> int:
    if isinstance(value, str):
        try:
            parsed = int(value, 0)
        except ValueError as error:
            raise ValueError(f"{label} must be a uint32 integer") from error
    elif type(value) is int:
        parsed = value
    else:
        raise ValueError(f"{label} must be a uint32 integer")
    if not 0 <= parsed <= UINT32_MASK:
        raise ValueError(f"{label} must be in [0, 2**32 - 1]")
    return parsed


def mix_u32_scalar(token_index: int, element_index: int, case_field_tag: int) -> int:
    """Scalar reference for the frozen uint32 counter and finalizer."""

    token = _u32(token_index, "token_index")
    element = _u32(element_index, "element_index")
    tag = _u32(case_field_tag, "case_field_tag")
    x = (
        ((token * TOKEN_MULTIPLIER) & UINT32_MASK)
        ^ ((element * ELEMENT_MULTIPLIER) & UINT32_MASK)
        ^ tag
    )
    x ^= x >> 16
    x = (x * MIX_MULTIPLIER_1) & UINT32_MASK
    x ^= x >> 15
    x = (x * MIX_MULTIPLIER_2) & UINT32_MASK
    x ^= x >> 16
    return x & UINT32_MASK


def frozen_case_field_tag(case_key: int, field_name: str) -> int:
    """Return the frozen uint32 tag for one case/field pair."""

    key = _u32(case_key, "case_key")
    _require_field(field_name)
    case_term = (key * TAG_CASE_MULTIPLIER) & UINT32_MASK
    field_term = (
        (FIELD_IDS[field_name] + 1) * TAG_FIELD_MULTIPLIER
    ) & UINT32_MASK
    return case_term ^ field_term


def _mul_u32_tensor(values: torch.Tensor, multiplier: int) -> torch.Tensor:
    """Multiply uint32 values in int64 carriers without signed overflow."""

    low = (values & 0xFFFF) * multiplier
    high = (((values >> 16) * multiplier) & 0xFFFF) << 16
    return (low + high) & UINT32_MASK


def reference_lut_indices(
    token_indices: Sequence[int] | torch.Tensor,
    element_count: int,
    case_field_tag: int,
) -> torch.Tensor:
    """Return CPU int64 LUT indices with shape ``[tokens, elements]``."""

    if type(element_count) is not int or not 0 < element_count <= UINT32_MASK + 1:
        raise ValueError("element_count must be in [1, 2**32]")
    tag = _u32(case_field_tag, "case_field_tag")
    tokens = torch.as_tensor(token_indices, dtype=torch.int64, device="cpu")
    if tokens.ndim != 1:
        raise ValueError("token_indices must be one-dimensional")
    if tokens.numel() and (
        int(tokens.min().item()) < 0 or int(tokens.max().item()) > UINT32_MASK
    ):
        raise ValueError("token_indices must contain only uint32 values")
    elements = torch.arange(element_count, dtype=torch.int64, device="cpu")
    x = (
        _mul_u32_tensor(tokens[:, None], TOKEN_MULTIPLIER)
        ^ _mul_u32_tensor(elements[None, :], ELEMENT_MULTIPLIER)
        ^ tag
    )
    x ^= x >> 16
    x = _mul_u32_tensor(x, MIX_MULTIPLIER_1)
    x ^= x >> 15
    x = _mul_u32_tensor(x, MIX_MULTIPLIER_2)
    x ^= x >> 16
    return (x & UINT32_MASK) >> 24


def reference_lut_bits(
    token_indices: Sequence[int] | torch.Tensor,
    element_count: int,
    case_field_tag: int,
    lut_bits: torch.Tensor,
) -> torch.Tensor:
    """Gather carrier integers without converting the represented values."""

    _validate_lut_tensor(lut_bits, "reference_lut")
    indices = reference_lut_indices(token_indices, element_count, case_field_tag)
    return lut_bits.cpu()[indices]


@dataclass(frozen=True)
class CounterCaseManifest:
    case_id: str
    case_index: int
    batch: int
    heads: int
    key_dim: int
    value_dim: int
    append_token_counts: tuple[int, ...]
    decode_steps: int
    regime: str
    field_tags: Mapping[str, int]

    def __post_init__(self) -> None:
        if not isinstance(self.case_id, str) or not self.case_id:
            raise ValueError("case_id must be nonempty")
        if type(self.case_index) is not int or self.case_index < 0:
            raise ValueError("case_index must be nonnegative")
        for label, value in (
            ("batch", self.batch),
            ("heads", self.heads),
            ("key_dim", self.key_dim),
            ("value_dim", self.value_dim),
        ):
            if type(value) is not int or value <= 0:
                raise ValueError(f"{label} must be a positive integer")
        if any(type(value) is not int or value <= 0 for value in self.append_token_counts):
            raise ValueError("append_token_counts must contain positive integers")
        if (
            type(self.decode_steps) is not int
            or not 0 <= self.decode_steps <= UINT32_MASK + 1
        ):
            raise ValueError("decode_steps must be in [0, 2**32]")
        if self.regime not in DECAY_REGIMES:
            raise ValueError("regime is outside the frozen decay table")
        if set(self.field_tags) != set(FIELD_NAMES):
            raise ValueError("field_tags must contain the six frozen fields")
        checked_tags = {
            name: _u32(self.field_tags[name], f"field_tags.{name}")
            for name in FIELD_NAMES
        }
        object.__setattr__(self, "field_tags", MappingProxyType(checked_tags))

    def lut_key(self, field_name: str) -> str:
        _require_field(field_name)
        return G_RAW_LUT_KEYS[self.regime] if field_name == "g_raw" else field_name

    def decode_element_shape(self, field_name: str) -> tuple[int, ...]:
        _require_field(field_name)
        if field_name == "beta_raw":
            return (self.batch, self.heads)
        width = self.value_dim if field_name in {"v_act", "output_gate_logits"} else self.key_dim
        return (self.batch, self.heads, width)

    def append_element_shape(self, field_name: str) -> tuple[int, ...]:
        _require_field(field_name)
        if field_name == "beta_raw":
            return (self.heads,)
        width = self.value_dim if field_name in {"v_act", "output_gate_logits"} else self.key_dim
        return (self.heads, width)


@dataclass(frozen=True)
class GeneratorLUTs:
    source_path: Path
    tensors: Mapping[str, torch.Tensor]

    def __post_init__(self) -> None:
        path = Path(self.source_path).resolve()
        if path.name != LUT_FILENAME:
            raise ValueError(f"LUT source must be named {LUT_FILENAME}")
        if set(self.tensors) != set(ALL_LUT_KEYS):
            raise ValueError("generator LUT key set is not the frozen nine-key set")
        checked: dict[str, torch.Tensor] = {}
        for key in sorted(ALL_LUT_KEYS):
            tensor = self.tensors[key]
            _validate_lut_tensor(tensor, key)
            expected_dtype = torch.int32 if key == "beta_raw" else torch.int16
            if tensor.dtype != expected_dtype:
                raise ValueError(f"{key} has the wrong carrier dtype")
            if tensor.device.type != "cpu" or not tensor.is_contiguous():
                raise ValueError(f"{key} must be a contiguous CPU LUT")
            checked[key] = tensor
        object.__setattr__(self, "source_path", path)
        object.__setattr__(self, "tensors", MappingProxyType(checked))

    def for_field(self, case: CounterCaseManifest, field_name: str) -> torch.Tensor:
        return self.tensors[case.lut_key(field_name)]


def load_generator_luts(path: str | Path) -> GeneratorLUTs:
    """Load the sole allowed LUT artifact and validate all carrier tensors."""

    source_path = Path(path).resolve()
    if source_path.name != LUT_FILENAME:
        raise ValueError(f"LUT source must be named {LUT_FILENAME}")
    try:
        from safetensors.torch import load_file
    except ModuleNotFoundError as error:
        raise RuntimeError("safetensors is required to load generator LUTs") from error
    try:
        tensors = load_file(str(source_path), device="cpu")
    except (OSError, RuntimeError) as error:
        raise ValueError(f"failed to load generator LUTs: {source_path}") from error
    return GeneratorLUTs(source_path=source_path, tensors=tensors)


@dataclass(frozen=True)
class BackingViews:
    backing: torch.Tensor
    bit_views: Mapping[str, torch.Tensor]
    value_views: Mapping[str, torch.Tensor]

    def __post_init__(self) -> None:
        if self.backing.dtype != torch.uint8 or self.backing.ndim != 1:
            raise ValueError("backing must be one flat uint8 tensor")
        if set(self.bit_views) != set(FIELD_NAMES) or set(self.value_views) != set(FIELD_NAMES):
            raise ValueError("backing views must contain the six frozen fields")
        storage_pointer = self.backing.untyped_storage().data_ptr()
        if any(
            tensor.untyped_storage().data_ptr() != storage_pointer
            for tensor in (*self.bit_views.values(), *self.value_views.values())
        ):
            raise ValueError("field view escaped its single backing storage")
        byte_intervals = []
        backing_start = self.backing.data_ptr()
        backing_end = backing_start + self.backing.numel()
        for name in FIELD_NAMES:
            bits = self.bit_views[name]
            values = self.value_views[name]
            if (
                bits.data_ptr() != values.data_ptr()
                or bits.shape != values.shape
                or bits.stride() != values.stride()
                or bits.device != self.backing.device
                or values.device != self.backing.device
            ):
                raise ValueError(f"{name} carrier/value views do not alias exactly")
            start = bits.data_ptr()
            end = start + bits.numel() * bits.element_size()
            if (
                start % bits.element_size() != 0
                or not backing_start <= start < end <= backing_end
            ):
                raise ValueError(f"{name} view is outside or misaligned in backing")
            byte_intervals.append((start, end, name))
        byte_intervals.sort()
        if any(
            left_end > right_start
            for (_, left_end, _), (right_start, _, _) in zip(
                byte_intervals, byte_intervals[1:]
            )
        ):
            raise ValueError("field views overlap in backing storage")
        object.__setattr__(self, "bit_views", MappingProxyType(dict(self.bit_views)))
        object.__setattr__(self, "value_views", MappingProxyType(dict(self.value_views)))


@dataclass
class DecodeBuffers:
    staging: BackingViews
    current: BackingViews
    shard_start: int | None = None
    shard_count: int = 0


if triton is not None:

    @triton.jit(
        do_not_specialize=("token_start", "total_count", "case_field_tag"),
        do_not_specialize_on_alignment=(
            "token_start",
            "total_count",
            "case_field_tag",
        ),
    )
    def _counter_fill_kernel(
        output_ptr,
        lut_ptr,
        token_start,
        element_count: tl.constexpr,
        total_count,
        case_field_tag,
        BLOCK_SIZE: tl.constexpr,
    ):
        offsets = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        mask = offsets < total_count
        token_index = (token_start + offsets // element_count).to(tl.uint32)
        element_index = (offsets % element_count).to(tl.uint32)
        tag = case_field_tag.to(tl.uint32)
        x = (
            token_index * 0x9E3779B1
            ^ element_index * 0x85EBCA77
            ^ tag
        )
        x ^= x >> 16
        x *= 0x7FEB352D
        x ^= x >> 15
        x *= 0x846CA68B
        x ^= x >> 16
        lut_index = x >> 24
        bits = tl.load(lut_ptr + lut_index, mask=mask, other=0)
        tl.store(output_ptr + offsets, bits, mask=mask)


class CounterV1Generator:
    """Generate append tensors and reusable 64-step Decode shards."""

    def __init__(
        self,
        case: CounterCaseManifest,
        luts: GeneratorLUTs,
        *,
        device: torch.device | str = "cpu",
    ) -> None:
        self.case = case
        self.luts = luts
        self.device = torch.device(device)
        if self.device.type not in {"cpu", "cuda"}:
            raise ValueError("counter device must be cpu or cuda")
        if self.device.type == "cuda" and triton is None:
            raise RuntimeError("Triton is required for CUDA counter generation")
        self._device_luts = {
            key: tensor.to(self.device)
            for key, tensor in self.luts.tensors.items()
        }
        self._append_cache: tuple[BackingViews, ...] | None = None

    def allocate_decode_buffers(self) -> DecodeBuffers:
        staging_shapes = {
            name: (SHARD_STEPS, *self.case.decode_element_shape(name))
            for name in FIELD_NAMES
        }
        current_shapes = {
            name: self.case.decode_element_shape(name) for name in FIELD_NAMES
        }
        staging = _allocate_backing_views(staging_shapes, self.device)
        current = _allocate_backing_views(current_shapes, self.device)
        if (
            staging.backing.untyped_storage().data_ptr()
            == current.backing.untyped_storage().data_ptr()
        ):
            raise RuntimeError("current-token buffer must be independent of staging")
        return DecodeBuffers(staging=staging, current=current)

    def fill_decode_shard(
        self,
        buffers: DecodeBuffers,
        start_step: int,
        count: int = SHARD_STEPS,
    ) -> Mapping[str, torch.Tensor]:
        if type(start_step) is not int or start_step < 0:
            raise ValueError("start_step must be nonnegative")
        if type(count) is not int or not 1 <= count <= SHARD_STEPS:
            raise ValueError(f"count must be in [1, {SHARD_STEPS}]")
        if start_step + count > self.case.decode_steps:
            raise ValueError("Decode shard exceeds the case schedule")
        if start_step + count - 1 > UINT32_MASK:
            raise ValueError("Decode shard token range exceeds uint32")
        self._validate_layout(buffers.staging, decode=True, steps=SHARD_STEPS)
        self._fill_views(
            buffers.staging,
            token_start=start_step,
            token_count=count,
            decode=True,
        )
        buffers.shard_start = start_step
        buffers.shard_count = count
        return MappingProxyType(
            {name: tensor[:count] for name, tensor in buffers.staging.value_views.items()}
        )

    def select_current_token(
        self, buffers: DecodeBuffers, step: int
    ) -> "DecodeInput":
        if buffers.shard_start is None:
            raise ValueError("Decode staging shard has not been generated")
        local_index = step - buffers.shard_start
        if not 0 <= local_index < buffers.shard_count:
            raise ValueError("Decode step is outside the resident shard")
        self._validate_layout(buffers.current, decode=True, steps=None)
        with torch.inference_mode():
            for name in FIELD_NAMES:
                buffers.current.bit_views[name].copy_(
                    buffers.staging.bit_views[name][local_index]
                )
        from .contract import DecodeInput

        return DecodeInput(
            **{name: buffers.current.value_views[name] for name in FIELD_NAMES}
        )

    def generate_append_once(
        self, token_count: int, *, token_start: int = 0
    ) -> BackingViews:
        if type(token_count) is not int or token_count <= 0:
            raise ValueError("token_count must be positive")
        _u32(token_start, "token_start")
        if token_start + token_count - 1 > UINT32_MASK:
            raise ValueError("append token range exceeds uint32")
        shapes = {
            name: (token_count, *self.case.append_element_shape(name))
            for name in FIELD_NAMES
        }
        views = _allocate_backing_views(shapes, self.device)
        self._fill_views(
            views,
            token_start=token_start,
            token_count=token_count,
            decode=False,
        )
        return views

    def generate_all_appends(self) -> tuple[BackingViews, ...]:
        if self._append_cache is not None:
            return self._append_cache
        generated = []
        token_start = 0
        for token_count in self.case.append_token_counts:
            generated.append(
                self.generate_append_once(token_count, token_start=token_start)
            )
            token_start += token_count
        self._append_cache = tuple(generated)
        return self._append_cache

    def _fill_views(
        self,
        views: BackingViews,
        *,
        token_start: int,
        token_count: int,
        decode: bool,
    ) -> None:
        with torch.inference_mode():
            for name in FIELD_NAMES:
                output = views.bit_views[name]
                element_shape = (
                    self.case.decode_element_shape(name)
                    if decode
                    else self.case.append_element_shape(name)
                )
                element_count = _shape_numel(element_shape)
                target = output[:token_count].reshape(-1)
                lut = self._device_luts[self.case.lut_key(name)]
                if self.device.type == "cpu":
                    indices = reference_lut_indices(
                        range(token_start, token_start + token_count),
                        element_count,
                        self.case.field_tags[name],
                    )
                    target.copy_(lut[indices].reshape(-1))
                else:
                    total_count = token_count * element_count
                    block_size = 256
                    assert triton is not None
                    _counter_fill_kernel[(triton.cdiv(total_count, block_size),)](
                        target,
                        lut,
                        token_start=token_start,
                        element_count=element_count,
                        total_count=total_count,
                        case_field_tag=self.case.field_tags[name],
                        BLOCK_SIZE=block_size,
                        num_warps=4,
                    )

    def _validate_layout(
        self, views: BackingViews, *, decode: bool, steps: int | None
    ) -> None:
        for name in FIELD_NAMES:
            expected = self.case.decode_element_shape(name)
            if not decode:
                expected = self.case.append_element_shape(name)
            if steps is not None:
                expected = (steps, *expected)
            if tuple(views.bit_views[name].shape) != expected:
                raise ValueError(f"{name} backing view shape mismatch")
            expected_carrier = torch.int32 if name == "beta_raw" else torch.int16
            expected_value = torch.float32 if name == "beta_raw" else torch.bfloat16
            if (
                views.bit_views[name].dtype != expected_carrier
                or views.value_views[name].dtype != expected_value
            ):
                raise ValueError(f"{name} backing view dtype mismatch")


def _require_field(field_name: str) -> None:
    if field_name not in FIELD_IDS:
        raise ValueError(f"unknown counter field: {field_name}")


def _validate_lut_tensor(tensor: object, label: str) -> None:
    if not isinstance(tensor, torch.Tensor):
        raise ValueError(f"{label} LUT must be a tensor")
    if tensor.ndim != 1 or tensor.numel() != LUT_SIZE:
        raise ValueError(f"{label} LUT must contain exactly {LUT_SIZE} entries")
    if tensor.dtype not in {torch.int16, torch.int32}:
        raise ValueError(f"{label} LUT must use int16 or int32 carrier bits")


def _shape_numel(shape: Sequence[int]) -> int:
    result = 1
    for value in shape:
        result *= value
    return result


def _align_up(value: int, alignment: int) -> int:
    return (value + alignment - 1) // alignment * alignment


def _allocate_backing_views(
    shapes: Mapping[str, tuple[int, ...]], device: torch.device
) -> BackingViews:
    if set(shapes) != set(FIELD_NAMES):
        raise ValueError("shape map must contain the six frozen fields")
    offsets: dict[str, tuple[int, int]] = {}
    cursor = 0
    for name in FIELD_NAMES:
        shape = shapes[name]
        if not shape or any(type(value) is not int or value <= 0 for value in shape):
            raise ValueError(f"{name} shape must contain positive integers")
        item_bytes = 4 if name == "beta_raw" else 2
        cursor = _align_up(cursor, 16)
        byte_count = _shape_numel(shape) * item_bytes
        offsets[name] = (cursor, byte_count)
        cursor += byte_count
    backing = torch.empty(cursor, dtype=torch.uint8, device=device)
    bit_views: dict[str, torch.Tensor] = {}
    value_views: dict[str, torch.Tensor] = {}
    for name in FIELD_NAMES:
        offset, byte_count = offsets[name]
        carrier_dtype = torch.int32 if name == "beta_raw" else torch.int16
        value_dtype = torch.float32 if name == "beta_raw" else torch.bfloat16
        bits = backing.narrow(0, offset, byte_count).view(carrier_dtype).view(shapes[name])
        bit_views[name] = bits
        value_views[name] = bits.view(value_dtype)
    return BackingViews(backing=backing, bit_views=bit_views, value_views=value_views)


__all__ = [
    "ALL_LUT_KEYS",
    "BF16_FIELDS",
    "BackingViews",
    "COUNTER_VERSION",
    "CounterCaseManifest",
    "CounterV1Generator",
    "DecodeBuffers",
    "FIELD_IDS",
    "FIELD_NAMES",
    "GeneratorLUTs",
    "LUT_FILENAME",
    "LUT_INDEX_SHIFT",
    "LUT_SIZE",
    "MIX_MULTIPLIER_1",
    "MIX_MULTIPLIER_2",
    "SHARD_STEPS",
    "TAG_CASE_MULTIPLIER",
    "TAG_FIELD_MULTIPLIER",
    "TOKEN_MULTIPLIER",
    "ELEMENT_MULTIPLIER",
    "frozen_case_field_tag",
    "load_generator_luts",
    "mix_u32_scalar",
    "reference_lut_bits",
    "reference_lut_indices",
]
