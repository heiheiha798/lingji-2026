"""Frozen case and tensor contracts used by the Judge."""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class CaseSpec:
    case_id: str
    n: int
    width: int
    weight: float

    @property
    def scored(self) -> bool:
        return self.weight > 0.0


def validate_case(case: CaseSpec) -> None:
    if not case.case_id or not case.case_id.isascii():
        raise ValueError("case_id must be non-empty ASCII")
    if case.n <= 0:
        raise ValueError("case.n must be positive")
    if case.width not in {16, 32, 64}:
        raise ValueError("case.width must be one of 16, 32, or 64")
    if case.weight < 0.0:
        raise ValueError("case.weight must be nonnegative")


def validate_inputs(
    case: CaseSpec,
    strings: torch.Tensor,
    lengths: torch.Tensor,
    indices_out: torch.Tensor,
) -> None:
    expected_device = strings.device
    checks = (
        (
            tuple(strings.shape) == (case.n, case.width)
            and strings.dtype == torch.uint8
            and strings.is_contiguous(),
            "strings must be contiguous uint8 [N,W]",
        ),
        (
            tuple(lengths.shape) == (case.n,)
            and lengths.dtype == torch.int32
            and lengths.is_contiguous()
            and lengths.device == expected_device,
            "lengths must be contiguous int32 [N] on the input device",
        ),
        (
            tuple(indices_out.shape) == (case.n,)
            and indices_out.dtype == torch.int32
            and indices_out.is_contiguous()
            and indices_out.device == expected_device,
            "indices_out must be contiguous int32 [N] on the input device",
        ),
    )
    for passed, message in checks:
        if not passed:
            raise ValueError(message)


__all__ = [
    "CaseSpec",
    "validate_case",
    "validate_inputs",
]
