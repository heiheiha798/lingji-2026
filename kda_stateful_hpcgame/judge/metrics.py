"""Correctness and timing metrics for the single-entry Judge."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Iterable, Sequence

import torch

from .data import DataError


RELATIVE_L2_FLOOR = 1e-12
NORMALIZED_MAX_FLOOR = 1.0


class TensorCheckError(ValueError):
    """An observed tensor violates the public correctness contract."""


@dataclass(frozen=True)
class TensorMetrics:
    relative_l2: float
    worst_sequence_relative_l2: float
    worst_head_relative_l2: float
    normalized_max: float
    max_abs: float
    min_sequence_reference_norm: float
    min_head_reference_norm: float
    finite: bool


class DecodeOutputMetricsAccumulator:
    """Accumulate full Decode-trajectory metrics from bounded GPU shards."""

    def __init__(
        self,
        *,
        expected_steps: int,
        batch: int,
        heads: int,
        value_dim: int,
        device: torch.device | str,
        chunk_steps: int = 4,
    ) -> None:
        for label, value in (
            ("expected_steps", expected_steps),
            ("batch", batch),
            ("heads", heads),
            ("value_dim", value_dim),
            ("chunk_steps", chunk_steps),
        ):
            if type(value) is not int or value <= 0:
                raise ValueError(f"{label} must be a positive integer")
        self.expected_steps = expected_steps
        self.batch = batch
        self.heads = heads
        self.value_dim = value_dim
        self.device = torch.device(device)
        if self.device.type == "cuda" and self.device.index is None:
            self.device = torch.device("cuda", torch.cuda.current_device())
        self.chunk_steps = chunk_steps
        self.seen_steps = 0
        self.total_error_sq = torch.zeros((), dtype=torch.float64, device=self.device)
        self.total_reference_sq = torch.zeros_like(self.total_error_sq)
        self.sequence_error_sq = torch.zeros(
            batch, dtype=torch.float64, device=self.device
        )
        self.sequence_reference_sq = torch.zeros_like(self.sequence_error_sq)
        self.head_error_sq = torch.zeros(
            heads, dtype=torch.float64, device=self.device
        )
        self.head_reference_sq = torch.zeros_like(self.head_error_sq)
        self.maximum_error = torch.zeros_like(self.total_error_sq)
        self.maximum_reference = torch.zeros_like(self.total_error_sq)
        self.actual_all_finite = torch.ones(
            (), dtype=torch.bool, device=self.device
        )
        self.golden_all_finite = torch.ones_like(self.actual_all_finite)

    def update(
        self,
        actual: torch.Tensor,
        golden: torch.Tensor,
        *,
        shard_start: int,
    ) -> None:
        """Consume one sequential ``[steps, batch, heads, value]`` shard."""

        if type(shard_start) is not int or shard_start != self.seen_steps:
            raise TensorCheckError(
                f"Decode shard order mismatch: expected {self.seen_steps}, "
                f"received {shard_start}"
            )
        if tuple(actual.shape) != tuple(golden.shape):
            raise TensorCheckError(
                f"shape mismatch: actual={tuple(actual.shape)} "
                f"golden={tuple(golden.shape)}"
            )
        expected_tail = (self.batch, self.heads, self.value_dim)
        if actual.ndim != 4 or tuple(actual.shape[1:]) != expected_tail:
            raise TensorCheckError(
                f"Decode shard must have shape [steps,{self.batch},{self.heads},"
                f"{self.value_dim}]"
            )
        if actual.shape[0] <= 0 or self.seen_steps + actual.shape[0] > self.expected_steps:
            raise TensorCheckError("Decode shard exceeds the frozen trajectory")
        if actual.dtype != torch.bfloat16 or golden.dtype != torch.bfloat16:
            raise TensorCheckError(
                f"dtype mismatch: expected {torch.bfloat16}, actual={actual.dtype}, "
                f"golden={golden.dtype}"
            )
        if actual.device != self.device or golden.device != self.device:
            raise TensorCheckError("Decode shards are not on the accumulator device")

        for start in range(0, actual.shape[0], self.chunk_steps):
            stop = min(start + self.chunk_steps, actual.shape[0])
            actual_chunk = actual[start:stop]
            golden_chunk = golden[start:stop]
            self.actual_all_finite &= torch.isfinite(actual_chunk).all()
            self.golden_all_finite &= torch.isfinite(golden_chunk).all()
            actual_f64 = actual_chunk.to(torch.float64)
            golden_f64 = golden_chunk.to(torch.float64)
            difference = actual_f64 - golden_f64
            squared_error = difference.square()
            squared_reference = golden_f64.square()
            sequence_error = squared_error.sum(dim=(0, 2, 3))
            sequence_reference = squared_reference.sum(dim=(0, 2, 3))
            head_error = squared_error.sum(dim=(0, 1, 3))
            head_reference = squared_reference.sum(dim=(0, 1, 3))
            self.sequence_error_sq += sequence_error
            self.sequence_reference_sq += sequence_reference
            self.head_error_sq += head_error
            self.head_reference_sq += head_reference
            self.total_error_sq += sequence_error.sum()
            self.total_reference_sq += sequence_reference.sum()
            self.maximum_error = torch.maximum(
                self.maximum_error, difference.abs().max()
            )
            self.maximum_reference = torch.maximum(
                self.maximum_reference, golden_f64.abs().max()
            )
            del (
                actual_f64,
                golden_f64,
                difference,
                squared_error,
                squared_reference,
            )
        self.seen_steps += actual.shape[0]

    def finalize(self) -> TensorMetrics:
        if self.seen_steps != self.expected_steps:
            raise TensorCheckError(
                f"Decode trajectory is incomplete: {self.seen_steps}/"
                f"{self.expected_steps} steps"
            )
        if not bool(self.golden_all_finite.item()):
            raise DataError("frozen golden Decode output contains NaN or infinity")
        finite = bool(self.actual_all_finite.item())
        if not finite:
            return TensorMetrics(
                relative_l2=math.inf,
                worst_sequence_relative_l2=math.inf,
                worst_head_relative_l2=math.inf,
                normalized_max=math.inf,
                max_abs=math.inf,
                min_sequence_reference_norm=0.0,
                min_head_reference_norm=0.0,
                finite=False,
            )
        sequence_norms = torch.sqrt(self.sequence_reference_sq)
        head_norms = torch.sqrt(self.head_reference_sq)
        sequence_ratios = (
            torch.sqrt(self.sequence_error_sq)
            / sequence_norms.clamp_min(RELATIVE_L2_FLOOR)
        )
        head_ratios = (
            torch.sqrt(self.head_error_sq)
            / head_norms.clamp_min(RELATIVE_L2_FLOOR)
        )
        return TensorMetrics(
            relative_l2=float(
                torch.sqrt(self.total_error_sq)
                / torch.sqrt(self.total_reference_sq).clamp_min(RELATIVE_L2_FLOOR)
            ),
            worst_sequence_relative_l2=float(sequence_ratios.max()),
            worst_head_relative_l2=float(head_ratios.max()),
            normalized_max=float(
                self.maximum_error
                / self.maximum_reference.clamp_min(NORMALIZED_MAX_FLOOR)
            ),
            max_abs=float(self.maximum_error),
            min_sequence_reference_norm=float(sequence_norms.min()),
            min_head_reference_norm=float(head_norms.min()),
            finite=True,
        )


def compare_tensor(
    actual: torch.Tensor,
    golden: torch.Tensor,
    *,
    required_dtype: torch.dtype,
    sequence_lengths: Sequence[int] | None = None,
    chunk_tokens: int = 256,
) -> TensorMetrics:
    """Compare one tensor with FP64 accumulation and bounded temporary storage."""

    if tuple(actual.shape) != tuple(golden.shape):
        raise TensorCheckError(
            f"shape mismatch: actual={tuple(actual.shape)} golden={tuple(golden.shape)}"
        )
    if actual.dtype != required_dtype or golden.dtype != required_dtype:
        raise TensorCheckError(
            f"dtype mismatch: expected {required_dtype}, actual={actual.dtype}, "
            f"golden={golden.dtype}"
        )
    if actual.ndim < 2:
        raise TensorCheckError("checked tensors must expose sequence and head axes")
    if chunk_tokens <= 0:
        raise ValueError("chunk_tokens must be positive")
    if actual.device != golden.device:
        raise TensorCheckError("actual and golden must share one device")

    first_dim = actual.shape[0]
    lengths = tuple(sequence_lengths) if sequence_lengths is not None else (1,) * first_dim
    if not lengths or any(type(length) is not int or length <= 0 for length in lengths):
        raise TensorCheckError("sequence lengths must contain positive integers")
    if sum(lengths) != first_dim:
        raise TensorCheckError("sequence lengths do not cover the tensor")

    device = actual.device
    heads = actual.shape[1]
    total_error_sq = torch.zeros((), dtype=torch.float64, device=device)
    total_reference_sq = torch.zeros_like(total_error_sq)
    head_error_sq = torch.zeros(heads, dtype=torch.float64, device=device)
    head_reference_sq = torch.zeros_like(head_error_sq)
    sequence_ratios: list[torch.Tensor] = []
    sequence_norms: list[torch.Tensor] = []
    maximum_error = torch.zeros((), dtype=torch.float64, device=device)
    maximum_reference = torch.zeros_like(maximum_error)
    actual_all_finite = torch.ones((), dtype=torch.bool, device=device)
    golden_all_finite = torch.ones_like(actual_all_finite)

    cursor = 0
    for length in lengths:
        sequence_error_sq = torch.zeros((), dtype=torch.float64, device=device)
        sequence_reference_sq = torch.zeros_like(sequence_error_sq)
        end = cursor + length
        for start in range(cursor, end, chunk_tokens):
            stop = min(start + chunk_tokens, end)
            actual_chunk = actual[start:stop]
            golden_chunk = golden[start:stop]
            actual_all_finite &= torch.isfinite(actual_chunk).all()
            golden_all_finite &= torch.isfinite(golden_chunk).all()
            actual_f64 = actual_chunk.to(torch.float64)
            golden_f64 = golden_chunk.to(torch.float64)
            difference = actual_f64 - golden_f64
            error_sq_by_head = (
                difference.square().movedim(1, 0).flatten(1).sum(dim=1)
            )
            reference_sq_by_head = (
                golden_f64.square().movedim(1, 0).flatten(1).sum(dim=1)
            )
            head_error_sq += error_sq_by_head
            head_reference_sq += reference_sq_by_head
            chunk_error_sq = error_sq_by_head.sum()
            chunk_reference_sq = reference_sq_by_head.sum()
            sequence_error_sq += chunk_error_sq
            sequence_reference_sq += chunk_reference_sq
            total_error_sq += chunk_error_sq
            total_reference_sq += chunk_reference_sq
            maximum_error = torch.maximum(maximum_error, difference.abs().max())
            maximum_reference = torch.maximum(
                maximum_reference, golden_f64.abs().max()
            )
            del actual_f64, golden_f64, difference
        sequence_ratios.append(
            torch.sqrt(sequence_error_sq)
            / torch.sqrt(sequence_reference_sq).clamp_min(RELATIVE_L2_FLOOR)
        )
        sequence_norms.append(torch.sqrt(sequence_reference_sq))
        cursor = end

    head_norms = torch.sqrt(head_reference_sq)
    head_ratios = torch.sqrt(head_error_sq) / head_norms.clamp_min(RELATIVE_L2_FLOOR)
    if not bool(golden_all_finite.item()):
        raise DataError("frozen golden tensor contains NaN or infinity")
    finite = bool(actual_all_finite.item())
    if not finite:
        return TensorMetrics(
            relative_l2=math.inf,
            worst_sequence_relative_l2=math.inf,
            worst_head_relative_l2=math.inf,
            normalized_max=math.inf,
            max_abs=math.inf,
            min_sequence_reference_norm=0.0,
            min_head_reference_norm=0.0,
            finite=False,
        )
    return TensorMetrics(
        relative_l2=float(
            torch.sqrt(total_error_sq)
            / torch.sqrt(total_reference_sq).clamp_min(RELATIVE_L2_FLOOR)
        ),
        worst_sequence_relative_l2=float(torch.stack(sequence_ratios).max()),
        worst_head_relative_l2=float(head_ratios.max()),
        normalized_max=float(
            maximum_error / maximum_reference.clamp_min(NORMALIZED_MAX_FLOOR)
        ),
        max_abs=float(maximum_error),
        min_sequence_reference_norm=float(torch.stack(sequence_norms).min()),
        min_head_reference_norm=float(head_norms.min()),
        finite=True,
    )


def enforce_limits(
    metrics: TensorMetrics,
    *,
    relative_l2_limit: float,
    normalized_max_limit: float,
) -> float:
    """Apply hard global gates and return the diagnostic limit ratio."""

    if not metrics.finite:
        raise TensorCheckError("tensor contains NaN or infinity")
    if metrics.relative_l2 > relative_l2_limit:
        raise TensorCheckError(
            f"Relative L2 {metrics.relative_l2:.9g} exceeds {relative_l2_limit:.9g}"
        )
    if metrics.normalized_max > normalized_max_limit:
        raise TensorCheckError(
            f"NormalizedMax {metrics.normalized_max:.9g} exceeds "
            f"{normalized_max_limit:.9g}"
        )
    return max(
        metrics.relative_l2 / relative_l2_limit if relative_l2_limit else 0.0,
        metrics.normalized_max / normalized_max_limit if normalized_max_limit else 0.0,
    )


def case_gpu_time_ms(
    replay_times_ms: Iterable[float],
    *,
    expected_replays: int,
) -> float:
    """Return the arithmetic mean of the frozen fresh-state replays."""

    if type(expected_replays) is not int or expected_replays <= 0:
        raise ValueError("expected_replays must be a positive integer")
    replays = tuple(float(item) for item in replay_times_ms)
    if len(replays) != expected_replays:
        raise ValueError(f"expected {expected_replays} timing replays")
    if any(not math.isfinite(item) or item <= 0 for item in replays):
        raise ValueError("timing replays must be finite and positive")
    return math.fsum(replays) / expected_replays


def weighted_gpu_time_ms(
    case_values: Iterable[tuple[float, float]],
) -> float:
    """Return the weighted geometric leaderboard metric in milliseconds."""

    values = tuple((float(weight), float(time_ms)) for weight, time_ms in case_values)
    if not values or any(
        not math.isfinite(weight)
        or not math.isfinite(time_ms)
        or weight <= 0
        or time_ms <= 0
        for weight, time_ms in values
    ):
        raise ValueError("weights and case GPU times must be finite and positive")
    if not math.isclose(
        math.fsum(weight for weight, unused in values),
        1.0,
        rel_tol=0.0,
        abs_tol=1e-15,
    ):
        raise ValueError("case weights must sum to 1.0")
    return math.exp(math.fsum(weight * math.log(time_ms) for weight, time_ms in values))


__all__ = [
    "DecodeOutputMetricsAccumulator",
    "TensorCheckError",
    "TensorMetrics",
    "case_gpu_time_ms",
    "compare_tensor",
    "enforce_limits",
    "weighted_gpu_time_ms",
]
