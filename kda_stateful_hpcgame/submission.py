"""Public FLA starter for the K3-KDA contest.

Participants modify this file and may add their own extension sources. The
official Judge imports exactly ``submission.Submission``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

import torch
import triton

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
    max_length: int
    output_rstd: torch.Tensor


@dataclass
class Context:
    config: KDAConfig
    layer: LayerParams
    case: CaseSpec
    append_plans: tuple[AppendPlan, ...]
    decode_output_rstd: torch.Tensor | None
    decode_state_pool: torch.Tensor | None
    decode_state_indices: torch.Tensor | None
    decode_index_table: torch.Tensor | None
    decode_graph: torch.cuda.CUDAGraph | None = None
    decode_binding: tuple[int, ...] | None = None
    decode_load_index: int = 0
    decode_active_slot: int = -1


@dataclass
class PrivateState:
    canonical_state: torch.Tensor
    append_index: int = 0
    decode_slot: int | None = None
    decode_bound: bool = False


class Submission:
    """FLA-based KDA implementation specialized for the contest schedules."""

    def __init__(self) -> None:
        self._l2norm_fwd: Callable[..., Any] | None = None
        self._beta_sigmoid: Callable[..., Any] | None = None
        self._chunk_kda_fwd: Callable[..., Any] | None = None
        self._prepare_chunk_indices: Callable[..., Any] | None = None
        self._fused_recurrent_kda_kernel: Any = None
        self._norm_gate_kernel: Any = None

    def _load_fla_once(self) -> None:
        if self._chunk_kda_fwd is not None:
            return
        from fla.modules.l2norm import l2norm_fwd
        from fla.ops.common.gate import fused_beta_sigmoid
        from fla.modules.fused_norm_gate import layer_norm_gated_fwd_kernel
        from fla.ops.kda.chunk_fwd import chunk_kda_fwd
        from fla.ops.kda.fused_recurrent import fused_recurrent_kda_fwd_kernel
        from fla.ops.utils.index import prepare_chunk_indices

        self._l2norm_fwd = l2norm_fwd
        self._beta_sigmoid = fused_beta_sigmoid
        self._chunk_kda_fwd = chunk_kda_fwd
        self._prepare_chunk_indices = prepare_chunk_indices
        self._fused_recurrent_kda_kernel = fused_recurrent_kda_fwd_kernel
        self._norm_gate_kernel = layer_norm_gated_fwd_kernel

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
            output_rstd = torch.empty(
                sum(lengths) * config.heads,
                dtype=torch.float32,
                device=layer.a_log.device,
            )
            if case.batch == 1:
                append_plans.append(
                    AppendPlan(None, None, max(lengths), output_rstd)
                )
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
                64,
                cu_seqlens_cpu=cu_seqlens_cpu,
            )
            append_plans.append(
                AppendPlan(
                    cu_seqlens_cpu,
                    chunk_indices,
                    max(lengths),
                    output_rstd,
                )
            )
        decode_output_rstd = (
            torch.empty(
                case.batch * config.heads,
                dtype=torch.float32,
                device=layer.a_log.device,
            )
            if case.decode_steps
            else None
        )
        decode_state_pool = None
        decode_state_indices = None
        decode_index_table = None
        if case.decode_steps:
            assert self._fused_recurrent_kda_kernel is not None
            assert self._norm_gate_kernel is not None
            assert decode_output_rstd is not None
            # Public Decode cases run one validation plus three timed replays.
            decode_state_pool = torch.empty(
                4,
                case.batch,
                config.heads,
                config.value_dim,
                config.key_dim,
                dtype=torch.float32,
                device=layer.a_log.device,
            )
            decode_index_table = torch.arange(
                4 * case.batch,
                dtype=torch.int32,
                device=layer.a_log.device,
            ).view(4, case.batch)
            decode_state_indices = torch.empty(
                case.batch,
                dtype=torch.int32,
                device=layer.a_log.device,
            )
            decode_state_indices.copy_(decode_index_table[0])
            decode_state_pool[0].zero_()

            dummy_qkg = torch.zeros(
                case.batch,
                config.heads,
                config.key_dim,
                dtype=torch.bfloat16,
                device=layer.a_log.device,
            )
            dummy_vg = torch.zeros(
                case.batch,
                config.heads,
                config.value_dim,
                dtype=torch.bfloat16,
                device=layer.a_log.device,
            )
            dummy_beta = torch.zeros(
                case.batch,
                config.heads,
                dtype=torch.float32,
                device=layer.a_log.device,
            )
            dummy_output = torch.empty_like(dummy_vg)
            self._fused_recurrent_kda_kernel[
                (4 * case.batch * config.heads,)
            ](
                q=dummy_qkg,
                k=dummy_qkg,
                v=dummy_vg,
                g=dummy_qkg,
                beta=dummy_beta,
                A_log=layer.a_log,
                dt_bias=layer.dt_bias,
                o=dummy_output,
                h0=decode_state_pool,
                ht=decode_state_pool,
                cu_seqlens=None,
                ssm_state_indices=decode_state_indices,
                num_accepted_tokens=None,
                lower_bound=config.lower_bound,
                scale=config.scale,
                N=case.batch,
                T=1,
                H=config.heads,
                HV=config.heads,
                K=config.key_dim,
                V=config.value_dim,
                BK=config.key_dim,
                BV=32,
                stride_init_state_token=decode_state_pool.stride(1),
                stride_final_state_token=decode_state_pool.stride(1),
                stride_indices_seq=1,
                stride_indices_tok=1,
                INPLACE_FINAL_STATE=True,
                IS_BETA_HEADWISE=False,
                USE_QK_L2NORM_IN_KERNEL=True,
                USE_GATE_IN_KERNEL=True,
                APPLY_BETA_SIGMOID=True,
                ALLOW_NEG_EIGVAL=False,
                STATE_V_FIRST=True,
                num_warps=4,
                num_stages=2,
            )
            rows = case.batch * config.heads
            self._norm_gate_kernel[
                lambda meta: (triton.cdiv(rows, meta["BT"]),)
            ](
                x=dummy_output,
                g=dummy_vg,
                y=dummy_output,
                w=layer.output_norm_weight,
                b=None,
                residual=None,
                residual_out=None,
                mean=None,
                rstd=decode_output_rstd,
                eps=config.output_rms_epsilon,
                T=rows,
                D=config.value_dim,
                BD=config.value_dim,
                NB=triton.cdiv(rows, 2048 * 32),
                ACTIVATION="sigmoid",
                IS_RMS_NORM=True,
            )
            torch.cuda.synchronize(layer.a_log.device)

        return Context(
            config=config,
            layer=layer,
            case=case,
            append_plans=tuple(append_plans),
            decode_output_rstd=decode_output_rstd,
            decode_state_pool=decode_state_pool,
            decode_state_indices=decode_state_indices,
            decode_index_table=decode_index_table,
        )

    def load_state(
        self, context: Context, canonical_state: torch.Tensor
    ) -> PrivateState:
        validate_canonical_state(context.case, canonical_state)
        if context.decode_state_pool is not None:
            assert context.decode_state_indices is not None
            assert context.decode_index_table is not None
            slot = context.decode_load_index
            if slot >= context.decode_state_pool.shape[0]:
                raise ValueError("load_state exceeded the Decode replay schedule")
            state = context.decode_state_pool[slot]
            state.copy_(canonical_state)
            context.decode_state_indices.copy_(
                context.decode_index_table[slot]
            )
            context.decode_load_index += 1
            context.decode_active_slot = slot
            return PrivateState(state, decode_slot=slot)
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
        assert self._fused_recurrent_kda_kernel is not None
        assert self._norm_gate_kernel is not None
        append_index = private_state.append_index
        if append_index >= context.case.append_calls:
            raise ValueError("append_chunk called beyond the case schedule")
        is_varlen = context.case.batch > 1
        plan = context.append_plans[append_index]
        if plan.max_length < 64:
            state = private_state.canonical_state
            self._fused_recurrent_kda_kernel[
                (4 * context.case.batch * context.config.heads,)
            ](
                q=args.q_act,
                k=args.k_act,
                v=args.v_act,
                g=args.g_raw,
                beta=args.beta_raw,
                A_log=context.layer.a_log,
                dt_bias=context.layer.dt_bias,
                o=output,
                h0=state,
                ht=state,
                cu_seqlens=args.cu_seqlens if is_varlen else None,
                ssm_state_indices=None,
                num_accepted_tokens=None,
                lower_bound=context.config.lower_bound,
                scale=context.config.scale,
                N=context.case.batch,
                T=args.q_act.shape[0],
                H=context.config.heads,
                HV=context.config.heads,
                K=context.config.key_dim,
                V=context.config.value_dim,
                BK=context.config.key_dim,
                BV=32,
                stride_init_state_token=state.stride(0),
                stride_final_state_token=state.stride(0),
                stride_indices_seq=1,
                stride_indices_tok=1,
                INPLACE_FINAL_STATE=True,
                IS_BETA_HEADWISE=False,
                USE_QK_L2NORM_IN_KERNEL=True,
                USE_GATE_IN_KERNEL=True,
                APPLY_BETA_SIGMOID=True,
                ALLOW_NEG_EIGVAL=False,
                STATE_V_FIRST=True,
                num_warps=4,
                num_stages=2,
            )
            final_state = state
            epilogue_input = output
        else:
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
                chunk_size=64,
                A_log=context.layer.a_log,
                dt_bias=context.layer.dt_bias,
            )
            epilogue_input = raw_output

        rows = output.numel() // context.config.value_dim
        self._norm_gate_kernel[
            lambda meta: (triton.cdiv(rows, meta["BT"]),)
        ](
            x=epilogue_input,
            g=args.output_gate_logits,
            y=output,
            w=context.layer.output_norm_weight,
            b=None,
            residual=None,
            residual_out=None,
            mean=None,
            rstd=plan.output_rstd,
            eps=context.config.output_rms_epsilon,
            T=rows,
            D=context.config.value_dim,
            BD=context.config.value_dim,
            NB=triton.cdiv(rows, 2048 * 32),
            ACTIVATION="sigmoid",
            IS_RMS_NORM=True,
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
        assert self._fused_recurrent_kda_kernel is not None
        assert self._norm_gate_kernel is not None
        assert context.decode_output_rstd is not None
        assert context.decode_state_pool is not None
        assert context.decode_state_indices is not None
        assert context.decode_index_table is not None
        assert private_state.decode_slot is not None
        batch = context.case.batch
        if context.decode_active_slot != private_state.decode_slot:
            context.decode_state_indices.copy_(
                context.decode_index_table[private_state.decode_slot]
            )
            context.decode_active_slot = private_state.decode_slot

        if not private_state.decode_bound:
            binding = tuple(
                tensor.data_ptr()
                for tensor in (
                    token.q_act,
                    token.k_act,
                    token.v_act,
                    token.g_raw,
                    token.beta_raw,
                    token.output_gate_logits,
                    output,
                )
            )
            if context.decode_binding is None:
                context.decode_binding = binding
            elif binding != context.decode_binding:
                raise ValueError("Decode tensor addresses changed after graph capture")
            private_state.decode_bound = True

        if context.decode_graph is not None:
            context.decode_graph.replay()
            return

        graph = torch.cuda.CUDAGraph()
        rows = output.numel() // context.config.value_dim
        with torch.cuda.graph(graph):
            self._fused_recurrent_kda_kernel[
                (4 * batch * context.config.heads,)
            ](
                q=token.q_act,
                k=token.k_act,
                v=token.v_act,
                g=token.g_raw,
                beta=token.beta_raw,
                A_log=context.layer.a_log,
                dt_bias=context.layer.dt_bias,
                o=output,
                h0=context.decode_state_pool,
                ht=context.decode_state_pool,
                cu_seqlens=None,
                ssm_state_indices=context.decode_state_indices,
                num_accepted_tokens=None,
                lower_bound=context.config.lower_bound,
                scale=context.config.scale,
                N=batch,
                T=1,
                H=context.config.heads,
                HV=context.config.heads,
                K=context.config.key_dim,
                V=context.config.value_dim,
                BK=context.config.key_dim,
                BV=32,
                stride_init_state_token=context.decode_state_pool.stride(1),
                stride_final_state_token=context.decode_state_pool.stride(1),
                stride_indices_seq=1,
                stride_indices_tok=1,
                INPLACE_FINAL_STATE=True,
                IS_BETA_HEADWISE=False,
                USE_QK_L2NORM_IN_KERNEL=True,
                USE_GATE_IN_KERNEL=True,
                APPLY_BETA_SIGMOID=True,
                ALLOW_NEG_EIGVAL=False,
                STATE_V_FIRST=True,
                num_warps=4,
                num_stages=2,
            )
            self._norm_gate_kernel[
                lambda meta: (triton.cdiv(rows, meta["BT"]),)
            ](
                x=output,
                g=token.output_gate_logits,
                y=output,
                w=context.layer.output_norm_weight,
                b=None,
                residual=None,
                residual_out=None,
                mean=None,
                rstd=context.decode_output_rstd,
                eps=context.config.output_rms_epsilon,
                T=rows,
                D=context.config.value_dim,
                BD=context.config.value_dim,
                NB=triton.cdiv(rows, 2048 * 32),
                ACTIVATION="sigmoid",
                IS_RMS_NORM=True,
            )
        context.decode_graph = graph
        graph.replay()

    def export_state(
        self,
        context: Context,
        private_state: PrivateState,
        canonical_state_out: torch.Tensor,
    ) -> None:
        validate_canonical_state(context.case, canonical_state_out)
        canonical_state_out.copy_(private_state.canonical_state)


__all__ = ["Submission"]
