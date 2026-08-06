"""Public FLA starter for the K3-KDA contest.

Participants modify this file and may add their own extension sources. The
official Judge imports exactly ``submission.Submission``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

import torch

from judge.contract import (
    AppendInputs,
    CaseSpec,
    DecodeInput,
    KDAConfig,
    LayerParams,
    validate_canonical_state,
    validate_layer,
)


@dataclass(frozen=True)
class AppendPlan:
    cu_seqlens_cpu: torch.Tensor | None
    chunk_indices: torch.Tensor | None


@dataclass(frozen=True)
class Context:
    config: KDAConfig
    layer: LayerParams
    case: CaseSpec
    append_plans: tuple[AppendPlan, ...]


@dataclass
class PrivateState:
    canonical_state: torch.Tensor
    append_index: int = 0


def _epilogue(
    raw_output: torch.Tensor,
    output_gate_logits: torch.Tensor,
    output_norm_weight: torch.Tensor,
    epsilon: float,
) -> torch.Tensor:
    raw_fp32 = raw_output.to(torch.bfloat16).float()
    normalized = raw_fp32 * torch.rsqrt(
        raw_fp32.square().mean(dim=-1, keepdim=True) + epsilon
    )
    weighted = normalized * output_norm_weight.float()
    return (
        torch.sigmoid(output_gate_logits.float()) * weighted
    ).to(torch.bfloat16)


class Submission:
    """Unmodified public starter using one FLA call per state transition."""

    def __init__(self) -> None:
        self._l2norm_fwd: Callable[..., Any] | None = None
        self._beta_sigmoid: Callable[..., Any] | None = None
        self._chunk_kda_fwd: Callable[..., Any] | None = None
        self._prepare_chunk_indices: Callable[..., Any] | None = None
        self._fused_recurrent_kda: Callable[..., Any] | None = None

    def _load_fla_once(self) -> None:
        if self._chunk_kda_fwd is not None:
            return
        from fla.modules.l2norm import l2norm_fwd
        from fla.ops.common.gate import fused_beta_sigmoid
        from fla.ops.kda import fused_recurrent_kda
        from fla.ops.kda.chunk_fwd import chunk_kda_fwd
        from fla.ops.utils.index import prepare_chunk_indices

        self._l2norm_fwd = l2norm_fwd
        self._beta_sigmoid = fused_beta_sigmoid
        self._chunk_kda_fwd = chunk_kda_fwd
        self._prepare_chunk_indices = prepare_chunk_indices
        self._fused_recurrent_kda = fused_recurrent_kda

    def prepare(
        self, config: KDAConfig, layer: LayerParams, case: CaseSpec
    ) -> Context:
        if config != case.config:
            raise ValueError("prepare config must equal case.config")
        validate_layer(config, layer)
        self._load_fla_once()
        assert self._prepare_chunk_indices is not None
        append_plans: list[AppendPlan] = []
        for lengths in case.append_lengths:
            if case.batch == 1:
                append_plans.append(AppendPlan(None, None))
                continue
            offsets = [0]
            for length in lengths:
                offsets.append(offsets[-1] + length)
            cu_seqlens = torch.tensor(
                offsets,
                dtype=torch.int32,
                device=layer.a_log.device,
            )
            cu_seqlens_cpu = cu_seqlens.cpu()
            chunk_indices = self._prepare_chunk_indices(
                cu_seqlens,
                32,
                cu_seqlens_cpu=cu_seqlens_cpu,
            )
            append_plans.append(AppendPlan(cu_seqlens_cpu, chunk_indices))
        return Context(config, layer, case, tuple(append_plans))

    def load_state(
        self, context: Context, canonical_state: torch.Tensor
    ) -> PrivateState:
        validate_canonical_state(context.case, canonical_state)
        return PrivateState(canonical_state.clone())

    @torch.inference_mode()
    def append_chunk(
        self,
        context: Context,
        private_state: PrivateState,
        args: AppendInputs,
        output: torch.Tensor,
    ) -> None:
        assert self._l2norm_fwd is not None
        assert self._beta_sigmoid is not None
        assert self._chunk_kda_fwd is not None
        append_index = private_state.append_index
        if append_index >= context.case.append_calls:
            raise ValueError("append_chunk called beyond the case schedule")
        is_varlen = context.case.batch > 1
        plan = context.append_plans[append_index]
        q, _ = self._l2norm_fwd(args.q_act.unsqueeze(0))
        k, _ = self._l2norm_fwd(args.k_act.unsqueeze(0))
        beta = self._beta_sigmoid(args.beta_raw.unsqueeze(0))
        raw_output, final_state, *_ = self._chunk_kda_fwd(
            q=q,
            k=k,
            v=args.v_act.unsqueeze(0),
            g=args.g_raw.unsqueeze(0),
            beta=beta,
            scale=context.config.scale,
            initial_state=private_state.canonical_state,
            output_final_state=True,
            use_gate_in_kernel=True,
            safe_gate=True,
            lower_bound=context.config.lower_bound,
            state_v_first=True,
            cu_seqlens=args.cu_seqlens if is_varlen else None,
            cu_seqlens_cpu=plan.cu_seqlens_cpu,
            chunk_indices=plan.chunk_indices,
            chunk_size=32,
            A_log=context.layer.a_log,
            dt_bias=context.layer.dt_bias,
        )
        output.copy_(
            _epilogue(
                raw_output.squeeze(0),
                args.output_gate_logits,
                context.layer.output_norm_weight,
                context.config.output_rms_epsilon,
            )
        )
        private_state.canonical_state = final_state
        private_state.append_index += 1

    @torch.inference_mode()
    def decode_step(
        self,
        context: Context,
        private_state: PrivateState,
        token: DecodeInput,
        output: torch.Tensor,
    ) -> None:
        assert self._fused_recurrent_kda is not None
        batch = context.case.batch
        raw_output, final_state = self._fused_recurrent_kda(
            q=token.q_act.view(
                batch, 1, context.config.heads, context.config.key_dim
            ),
            k=token.k_act.view(
                batch, 1, context.config.heads, context.config.key_dim
            ),
            v=token.v_act.view(
                batch, 1, context.config.heads, context.config.value_dim
            ),
            g=token.g_raw.view(
                batch, 1, context.config.heads, context.config.key_dim
            ),
            beta=token.beta_raw.view(batch, 1, context.config.heads),
            A_log=context.layer.a_log,
            dt_bias=context.layer.dt_bias,
            initial_state=private_state.canonical_state,
            scale=context.config.scale,
            output_final_state=True,
            state_v_first=True,
            use_qk_l2norm_in_kernel=True,
            use_gate_in_kernel=True,
            use_beta_sigmoid_in_kernel=True,
            lower_bound=context.config.lower_bound,
        )
        private_state.canonical_state = final_state
        output.copy_(
            _epilogue(
                raw_output.squeeze(1),
                token.output_gate_logits,
                context.layer.output_norm_weight,
                context.config.output_rms_epsilon,
            )
        )

    def export_state(
        self,
        context: Context,
        private_state: PrivateState,
        canonical_state_out: torch.Tensor,
    ) -> None:
        validate_canonical_state(context.case, canonical_state_out)
        canonical_state_out.copy_(private_state.canonical_state)


__all__ = ["Submission"]
