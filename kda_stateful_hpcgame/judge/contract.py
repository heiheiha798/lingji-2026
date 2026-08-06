"""Public five-method ABI and immutable KDA lifecycle types."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import math
from typing import Any, Protocol

import torch


Tensor = torch.Tensor


class StateMode(str, Enum):
    ZERO = "zero"
    NONZERO = "nonzero"
    CHECKPOINT = "checkpoint"


@dataclass(frozen=True)
class Limits:
    output_relative_l2: float
    state_relative_l2: float
    normalized_max: float

    def __post_init__(self) -> None:
        values = (
            self.output_relative_l2,
            self.state_relative_l2,
            self.normalized_max,
        )
        if any(
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(value)
            or value < 0
            for value in values
        ):
            raise ValueError("correctness limits must be finite nonnegative numbers")


@dataclass(frozen=True)
class KDAConfig:
    heads: int
    key_dim: int = 128
    value_dim: int = 128

    def __post_init__(self) -> None:
        if any(
            type(value) is not int or value <= 0
            for value in (self.heads, self.key_dim, self.value_dim)
        ):
            raise ValueError("KDA dimensions must be positive integers")

    @property
    def scale(self) -> float:
        return self.key_dim**-0.5

    @property
    def lower_bound(self) -> float:
        return -5.0

    @property
    def qk_epsilon(self) -> float:
        return 1e-6

    @property
    def output_rms_epsilon(self) -> float:
        return 1e-5


@dataclass(frozen=True)
class CaseSpec:
    """One immutable trajectory; Decode output is required at every step."""

    case_id: str
    config: KDAConfig
    batch: int
    state_mode: StateMode
    limits: Limits
    append_lengths: tuple[tuple[int, ...], ...] = ()
    decode_steps: int = 0

    def __post_init__(self) -> None:
        if not isinstance(self.case_id, str) or not self.case_id:
            raise ValueError("case_id must be nonempty")
        if type(self.batch) is not int or self.batch <= 0:
            raise ValueError("batch must be a positive integer")
        for lengths in self.append_lengths:
            if len(lengths) != self.batch or any(
                type(length) is not int or length <= 0 for length in lengths
            ):
                raise ValueError("each append schedule must contain B positives")
        if type(self.decode_steps) is not int or self.decode_steps < 0:
            raise ValueError("decode_steps must be nonnegative")
        if not self.append_lengths and not self.decode_steps:
            raise ValueError("a case must append or Decode")

    @property
    def append_calls(self) -> int:
        return len(self.append_lengths)

    def append_tokens(self, index: int) -> int:
        return sum(self.append_lengths[index])


@dataclass(frozen=True)
class LayerParams:
    a_log: Tensor
    dt_bias: Tensor
    output_norm_weight: Tensor

    def tensors(self) -> tuple[Tensor, ...]:
        return self.a_log, self.dt_bias, self.output_norm_weight


@dataclass(frozen=True)
class AppendInputs:
    q_act: Tensor
    k_act: Tensor
    v_act: Tensor
    g_raw: Tensor
    beta_raw: Tensor
    output_gate_logits: Tensor
    cu_seqlens: Tensor
    descriptor: Tensor

@dataclass(frozen=True)
class DecodeInput:
    q_act: Tensor
    k_act: Tensor
    v_act: Tensor
    g_raw: Tensor
    beta_raw: Tensor
    output_gate_logits: Tensor

class Submission(Protocol):
    def prepare(
        self, config: KDAConfig, layer: LayerParams, case: CaseSpec
    ) -> Any: ...

    def load_state(self, context: Any, canonical_state: Tensor) -> Any: ...

    def append_chunk(
        self,
        context: Any,
        private_state: Any,
        args: AppendInputs,
        output: Tensor,
    ) -> None: ...

    def decode_step(
        self,
        context: Any,
        private_state: Any,
        token: DecodeInput,
        output: Tensor,
    ) -> None: ...

    def export_state(
        self,
        context: Any,
        private_state: Any,
        canonical_state_out: Tensor,
    ) -> None: ...


def validate_layer(config: KDAConfig, layer: LayerParams) -> None:
    if tuple(layer.a_log.shape) != (config.heads,):
        raise ValueError("a_log must have shape [H]")
    if tuple(layer.dt_bias.shape) != (config.heads, config.key_dim):
        raise ValueError("dt_bias must have shape [H,K]")
    if tuple(layer.output_norm_weight.shape) != (config.value_dim,):
        raise ValueError("output_norm_weight must have shape [V]")
    if any(tensor.dtype != torch.float32 for tensor in layer.tensors()):
        raise ValueError("layer tensors must use FP32")


def validate_canonical_state(case: CaseSpec, state: Tensor) -> None:
    config = case.config
    expected = (case.batch, config.heads, config.value_dim, config.key_dim)
    if tuple(state.shape) != expected:
        raise ValueError("canonical state has the wrong shape")
    if state.dtype != torch.float32 or not state.is_contiguous():
        raise ValueError("canonical state must be contiguous FP32")


def append_result_key(index: int) -> str:
    return f"append_{index + 1:08d}"


def decode_result_key(step: int) -> str:
    return f"decode_{step:08d}"


__all__ = [
    "AppendInputs",
    "CaseSpec",
    "DecodeInput",
    "KDAConfig",
    "LayerParams",
    "Limits",
    "StateMode",
    "Submission",
    "append_result_key",
    "decode_result_key",
    "validate_canonical_state",
    "validate_layer",
]
