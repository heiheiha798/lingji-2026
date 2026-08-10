#!/usr/bin/env python3
"""Profile SM89-compatible FLA KDA baselines on the contest case shapes.

The public data package is intentionally not reconstructed here. Inputs are
synthetic, while tensor shapes, append schedules, state layout, dtypes, and
the timed method boundaries match the public contest contract.
"""

from __future__ import annotations

import argparse
import gc
import json
import math
import os
import statistics
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path

if os.environ.get("FLA_DISABLE_BACKEND_DISPATCH") != "1":
    raise RuntimeError("set FLA_DISABLE_BACKEND_DISPATCH=1 before importing FLA")

import torch
import fla
import triton

from fla.modules.fused_norm_gate import layer_norm_gated_fwd_kernel, rms_norm_gated
from fla.modules.l2norm import l2norm_fwd
from fla.ops.common.gate import fused_beta_sigmoid
from fla.ops.kda import fused_recurrent_kda
from fla.ops.kda.chunk_fwd import chunk_kda_fwd
from fla.ops.kda.fused_recurrent import (
    fused_recurrent_kda_fwd,
    fused_recurrent_kda_fwd_kernel,
)
from fla.ops.utils.index import prepare_chunk_indices


K = 128
V = 128
LOWER_BOUND = -5.0
OUTPUT_EPSILON = 1e-5


@dataclass(frozen=True)
class Case:
    name: str
    batch: int
    heads: int
    append_lengths: tuple[tuple[int, ...], ...]
    decode_steps: int
    weight: float


@dataclass
class AppendCall:
    lengths: tuple[int, ...]
    q: torch.Tensor
    k: torch.Tensor
    v: torch.Tensor
    g: torch.Tensor
    beta: torch.Tensor
    output_gate: torch.Tensor
    output: torch.Tensor
    output_rstd: torch.Tensor
    recurrent_raw: torch.Tensor
    cu_seqlens: torch.Tensor | None
    cu_seqlens_cpu: torch.Tensor | None
    chunk_indices: dict[int, torch.Tensor | None]


@dataclass
class DecodeCall:
    q: torch.Tensor
    k: torch.Tensor
    v: torch.Tensor
    g: torch.Tensor
    beta: torch.Tensor
    output_gate: torch.Tensor
    output: torch.Tensor
    output_rstd: torch.Tensor
    recurrent_raw: torch.Tensor


CASES = (
    Case("L24-Z", 1, 24, ((65536,),), 0, 0.125),
    Case("L24-N", 1, 24, ((65536,),), 0, 0.125),
    Case(
        "M24-C",
        8,
        24,
        ((4095, 4096, 4097, 8191, 8192, 8193, 12287, 16385),),
        0,
        0.05,
    ),
    Case(
        "M48-N",
        32,
        48,
        ((511, 512, 513, 1023, 1024, 1025, 1535, 1537) * 4,),
        0,
        0.05,
    ),
    Case("C48", 32, 48, ((64,) * 32, (256,) * 32, (64,) * 32, (256,) * 32), 0, 0.125),
    Case("C24", 128, 24, ((16,) * 128, (64,) * 128, (16,) * 128, (64,) * 128), 0, 0.125),
    Case("D48-B1", 1, 48, ((17,),), 16384, 0.10),
    Case("D48", 32, 48, ((17,) * 32,), 16384, 0.15),
    Case("D24", 128, 24, ((17,) * 128,), 16384, 0.15),
)

CANDIDATES = (
    "starter32",
    "chunk32",
    "chunk64",
    "recurrent",
    "hybrid64",
    "hybrid64_direct_epi",
    "hybrid_lt64_direct_epi",
    "hybrid_lt64_direct_launch",
    "hybrid256",
)


def allocate_append(lengths: tuple[int, ...], heads: int) -> AppendCall:
    total_tokens = sum(lengths)
    q = torch.randn(total_tokens, heads, K, device="cuda", dtype=torch.bfloat16).unsqueeze(0)
    k = torch.randn(total_tokens, heads, K, device="cuda", dtype=torch.bfloat16).unsqueeze(0)
    v = torch.randn(total_tokens, heads, V, device="cuda", dtype=torch.bfloat16).unsqueeze(0)
    g = torch.randn(total_tokens, heads, K, device="cuda", dtype=torch.bfloat16).unsqueeze(0)
    beta = torch.randn(total_tokens, heads, device="cuda", dtype=torch.float32).unsqueeze(0)
    output_gate = torch.randn(total_tokens, heads, V, device="cuda", dtype=torch.bfloat16).unsqueeze(0)
    output = torch.empty_like(v)
    output_rstd = torch.empty(total_tokens * heads, device="cuda", dtype=torch.float32)
    recurrent_raw = torch.empty_like(v)

    cu_seqlens = None
    cu_seqlens_cpu = None
    chunk_indices: dict[int, torch.Tensor | None] = {32: None, 64: None}
    if len(lengths) > 1:
        offsets = [0]
        for length in lengths:
            offsets.append(offsets[-1] + length)
        cu_seqlens = torch.tensor(offsets, device="cuda", dtype=torch.int32)
        cu_seqlens_cpu = cu_seqlens.cpu()
        for chunk_size in chunk_indices:
            chunk_indices[chunk_size] = prepare_chunk_indices(
                cu_seqlens,
                chunk_size,
                cu_seqlens_cpu=cu_seqlens_cpu,
            )

    return AppendCall(
        lengths,
        q,
        k,
        v,
        g,
        beta,
        output_gate,
        output,
        output_rstd,
        recurrent_raw,
        cu_seqlens,
        cu_seqlens_cpu,
        chunk_indices,
    )


def allocate_decode(batch: int, heads: int) -> DecodeCall:
    shape = (batch, 1, heads)
    q = torch.randn(*shape, K, device="cuda", dtype=torch.bfloat16)
    k = torch.randn(*shape, K, device="cuda", dtype=torch.bfloat16)
    v = torch.randn(*shape, V, device="cuda", dtype=torch.bfloat16)
    g = torch.randn(*shape, K, device="cuda", dtype=torch.bfloat16)
    beta = torch.randn(*shape, device="cuda", dtype=torch.float32)
    output_gate = torch.randn(*shape, V, device="cuda", dtype=torch.bfloat16)
    output = torch.empty_like(v)
    output_rstd = torch.empty(batch * heads, device="cuda", dtype=torch.float32)
    recurrent_raw = torch.empty_like(v)
    return DecodeCall(q, k, v, g, beta, output_gate, output, output_rstd, recurrent_raw)


def apply_epilogue(
    raw_output: torch.Tensor,
    output_gate: torch.Tensor,
    weight: torch.Tensor,
    output: torch.Tensor,
    output_rstd: torch.Tensor,
    fused: bool,
    direct: bool = False,
) -> None:
    if direct:
        rows = raw_output.numel() // V
        layer_norm_gated_fwd_kernel[
            lambda meta: (triton.cdiv(rows, meta["BT"]),)
        ](
            x=raw_output.view(rows, V),
            g=output_gate.view(rows, V),
            y=output.view(rows, V),
            w=weight,
            b=None,
            residual=None,
            residual_out=None,
            mean=None,
            rstd=output_rstd,
            eps=OUTPUT_EPSILON,
            T=rows,
            D=V,
            BD=V,
            NB=triton.cdiv(rows, 2048 * 32),
            ACTIVATION="sigmoid",
            IS_RMS_NORM=True,
        )
    elif fused:
        output.copy_(
            rms_norm_gated(
                raw_output,
                output_gate,
                weight,
                None,
                activation="sigmoid",
                eps=OUTPUT_EPSILON,
            )
        )
    else:
        raw_fp32 = raw_output.to(torch.bfloat16).float()
        normalized = raw_fp32 * torch.rsqrt(
            raw_fp32.square().mean(dim=-1, keepdim=True) + OUTPUT_EPSILON
        )
        output.copy_(
            (
                torch.sigmoid(output_gate.float())
                * normalized
                * weight.float()
            ).to(torch.bfloat16)
        )


def run_append(
    call: AppendCall,
    candidate: str,
    state_holder: list[torch.Tensor],
    a_log: torch.Tensor,
    dt_bias: torch.Tensor,
    norm_weight: torch.Tensor,
) -> None:
    max_length = max(call.lengths)
    use_recurrent = candidate == "recurrent"
    use_recurrent |= candidate in (
        "hybrid64",
        "hybrid64_direct_epi",
    ) and max_length <= 64
    use_recurrent |= candidate in (
        "hybrid_lt64_direct_epi",
        "hybrid_lt64_direct_launch",
    ) and max_length < 64
    use_recurrent |= candidate == "hybrid256" and max_length <= 256

    if use_recurrent:
        if candidate == "hybrid_lt64_direct_launch":
            sequences = len(call.lengths)
            heads = call.q.shape[2]
            uniform_lengths = len(set(call.lengths)) == 1
            fused_recurrent_kda_fwd_kernel[(4 * sequences * heads,)](
                q=call.q,
                k=call.k,
                v=call.v,
                g=call.g,
                beta=call.beta,
                A_log=a_log,
                dt_bias=dt_bias,
                o=call.recurrent_raw,
                h0=state_holder[0],
                ht=state_holder[0],
                cu_seqlens=(
                    None if uniform_lengths else call.cu_seqlens
                ),
                ssm_state_indices=None,
                num_accepted_tokens=None,
                lower_bound=LOWER_BOUND,
                scale=K**-0.5,
                N=sequences,
                T=(call.lengths[0] if uniform_lengths else call.q.shape[1]),
                H=heads,
                HV=heads,
                K=K,
                V=V,
                BK=K,
                BV=32,
                stride_init_state_token=state_holder[0].stride(0),
                stride_final_state_token=state_holder[0].stride(0),
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
        else:
            _, state_holder[0] = fused_recurrent_kda_fwd(
                q=call.q,
                k=call.k,
                v=call.v,
                g=call.g,
                beta=call.beta,
                A_log=a_log,
                dt_bias=dt_bias,
                initial_state=state_holder[0],
                scale=K**-0.5,
                output_final_state=True,
                inplace_final_state=True,
                state_v_first=True,
                cu_seqlens=call.cu_seqlens,
                use_qk_l2norm_in_kernel=True,
                use_gate_in_kernel=True,
                use_beta_sigmoid_in_kernel=True,
                lower_bound=LOWER_BOUND,
                out=call.recurrent_raw,
            )
        apply_epilogue(
            call.recurrent_raw,
            call.output_gate,
            norm_weight,
            call.output,
            call.output_rstd,
            fused=candidate != "starter32",
            direct=candidate in (
                "hybrid64_direct_epi",
                "hybrid_lt64_direct_epi",
                "hybrid_lt64_direct_launch",
            ),
        )
        return

    chunk_size = 32 if candidate in ("starter32", "chunk32") else 64
    q, _ = l2norm_fwd(call.q)
    k, _ = l2norm_fwd(call.k)
    beta = fused_beta_sigmoid(call.beta)
    raw_output, final_state, *_ = chunk_kda_fwd(
        q=q,
        k=k,
        v=call.v,
        g=call.g,
        beta=beta,
        scale=K**-0.5,
        initial_state=state_holder[0],
        output_final_state=True,
        use_gate_in_kernel=True,
        safe_gate=True,
        lower_bound=LOWER_BOUND,
        state_v_first=True,
        cu_seqlens=call.cu_seqlens,
        cu_seqlens_cpu=call.cu_seqlens_cpu,
        chunk_indices=call.chunk_indices[chunk_size],
        chunk_size=chunk_size,
        A_log=a_log,
        dt_bias=dt_bias,
    )
    state_holder[0] = final_state
    apply_epilogue(
        raw_output,
        call.output_gate,
        norm_weight,
        call.output,
        call.output_rstd,
        fused=candidate != "starter32",
        direct=candidate in (
            "hybrid64_direct_epi",
            "hybrid_lt64_direct_epi",
            "hybrid_lt64_direct_launch",
        ),
    )


def run_decode(
    call: DecodeCall,
    candidate: str,
    state_holder: list[torch.Tensor],
    a_log: torch.Tensor,
    dt_bias: torch.Tensor,
    norm_weight: torch.Tensor,
) -> None:
    if candidate == "starter32":
        raw_output, state_holder[0] = fused_recurrent_kda(
            q=call.q,
            k=call.k,
            v=call.v,
            g=call.g,
            beta=call.beta,
            A_log=a_log,
            dt_bias=dt_bias,
            initial_state=state_holder[0],
            scale=K**-0.5,
            output_final_state=True,
            state_v_first=True,
            use_qk_l2norm_in_kernel=True,
            use_gate_in_kernel=True,
            use_beta_sigmoid_in_kernel=True,
            lower_bound=LOWER_BOUND,
        )
        apply_epilogue(
            raw_output,
            call.output_gate,
            norm_weight,
            call.output,
            call.output_rstd,
            fused=False,
        )
        return

    if candidate == "hybrid_lt64_direct_launch":
        batch, _, heads, _ = call.q.shape
        fused_recurrent_kda_fwd_kernel[(4 * batch * heads,)](
            q=call.q,
            k=call.k,
            v=call.v,
            g=call.g,
            beta=call.beta,
            A_log=a_log,
            dt_bias=dt_bias,
            o=call.recurrent_raw,
            h0=state_holder[0],
            ht=state_holder[0],
            cu_seqlens=None,
            ssm_state_indices=None,
            num_accepted_tokens=None,
            lower_bound=LOWER_BOUND,
            scale=K**-0.5,
            N=batch,
            T=1,
            H=heads,
            HV=heads,
            K=K,
            V=V,
            BK=K,
            BV=32,
            stride_init_state_token=state_holder[0].stride(0),
            stride_final_state_token=state_holder[0].stride(0),
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
    else:
        _, state_holder[0] = fused_recurrent_kda_fwd(
            q=call.q,
            k=call.k,
            v=call.v,
            g=call.g,
            beta=call.beta,
            A_log=a_log,
            dt_bias=dt_bias,
            initial_state=state_holder[0],
            scale=K**-0.5,
            output_final_state=True,
            inplace_final_state=True,
            state_v_first=True,
            use_qk_l2norm_in_kernel=True,
            use_gate_in_kernel=True,
            use_beta_sigmoid_in_kernel=True,
            lower_bound=LOWER_BOUND,
            out=call.recurrent_raw,
        )
    apply_epilogue(
        call.recurrent_raw,
        call.output_gate,
        norm_weight,
        call.output,
        call.output_rstd,
        fused=True,
        direct=candidate in (
            "hybrid64_direct_epi",
            "hybrid_lt64_direct_epi",
            "hybrid_lt64_direct_launch",
        ),
    )


def benchmark_case(
    case: Case,
    candidate: str,
    warmup: int,
    replays: int,
    decode_sample_steps: int,
    nvtx: bool,
    cuda_profiler_api: bool,
) -> dict[str, object]:
    append_calls = [allocate_append(lengths, case.heads) for lengths in case.append_lengths]
    decode_call = allocate_decode(case.batch, case.heads) if case.decode_steps else None
    a_log = torch.zeros(case.heads, device="cuda", dtype=torch.float32)
    dt_bias = torch.zeros(case.heads, K, device="cuda", dtype=torch.float32)
    norm_weight = torch.ones(V, device="cuda", dtype=torch.float32)
    initial_state = torch.zeros(
        case.batch,
        case.heads,
        V,
        K,
        device="cuda",
        dtype=torch.float32,
    )
    measured_decode_steps = min(case.decode_steps, decode_sample_steps)

    for _ in range(warmup):
        state_holder = [initial_state.clone()]
        for call in append_calls:
            run_append(call, candidate, state_holder, a_log, dt_bias, norm_weight)
        if decode_call is not None:
            for _ in range(min(measured_decode_steps, 2)):
                run_decode(decode_call, candidate, state_holder, a_log, dt_bias, norm_weight)
        torch.cuda.synchronize()

    replay_append_ms: list[float] = []
    replay_append_calls_ms: list[list[float]] = []
    replay_decode_ms: list[float] = []
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    for replay in range(replays):
        state_holder = [initial_state.clone()]
        torch.cuda.synchronize()
        if cuda_profiler_api:
            torch.cuda.cudart().cudaProfilerStart()
        append_ms = 0.0
        append_calls_ms = []
        for index, call in enumerate(append_calls):
            if nvtx:
                torch.cuda.nvtx.range_push(f"{case.name}:{candidate}:append:{index}")
            start.record()
            run_append(call, candidate, state_holder, a_log, dt_bias, norm_weight)
            end.record()
            end.synchronize()
            call_ms = start.elapsed_time(end)
            append_ms += call_ms
            append_calls_ms.append(call_ms)
            if nvtx:
                torch.cuda.nvtx.range_pop()

        decode_ms = 0.0
        if decode_call is not None:
            for step in range(measured_decode_steps):
                if nvtx:
                    torch.cuda.nvtx.range_push(f"{case.name}:{candidate}:decode:{step}")
                start.record()
                run_decode(
                    decode_call,
                    candidate,
                    state_holder,
                    a_log,
                    dt_bias,
                    norm_weight,
                )
                end.record()
                end.synchronize()
                decode_ms += start.elapsed_time(end)
                if nvtx:
                    torch.cuda.nvtx.range_pop()
        replay_append_ms.append(append_ms)
        replay_append_calls_ms.append(append_calls_ms)
        replay_decode_ms.append(decode_ms)
        if cuda_profiler_api:
            torch.cuda.synchronize()
            torch.cuda.cudart().cudaProfilerStop()

    append_ms = statistics.mean(replay_append_ms)
    decode_sample_ms = statistics.mean(replay_decode_ms)
    decode_step_us = (
        decode_sample_ms * 1000.0 / measured_decode_steps
        if measured_decode_steps
        else 0.0
    )
    estimated_case_ms = append_ms + decode_step_us * case.decode_steps / 1000.0
    result = {
        "case": case.name,
        "candidate": candidate,
        "append_ms": append_ms,
        "decode_sample_steps": measured_decode_steps,
        "decode_sample_ms": decode_sample_ms,
        "decode_step_us": decode_step_us,
        "estimated_case_ms": estimated_case_ms,
        "replay_append_ms": replay_append_ms,
        "mean_append_calls_ms": [
            statistics.mean(replay[index] for replay in replay_append_calls_ms)
            for index in range(len(append_calls))
        ],
        "replay_append_calls_ms": replay_append_calls_ms,
        "replay_decode_ms": replay_decode_ms,
        "weight": case.weight,
    }

    del append_calls, decode_call, initial_state
    gc.collect()
    torch.cuda.empty_cache()
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mode",
        choices=("validate", "submission-validate", "benchmark"),
        default="benchmark",
    )
    parser.add_argument("--cases", default="all")
    parser.add_argument("--candidates", default="hybrid64")
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--replays", type=int, default=3)
    parser.add_argument("--decode-sample-steps", type=int, default=128)
    parser.add_argument("--validation-decode-steps", type=int, default=257)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--nvtx", action="store_true")
    parser.add_argument("--cuda-profiler-api", action="store_true")
    args = parser.parse_args()

    if torch.cuda.device_count() != 1:
        raise RuntimeError("set CUDA_VISIBLE_DEVICES to exactly one physical GPU")
    capability = torch.cuda.get_device_capability(0)
    if capability != (8, 9):
        raise RuntimeError(f"this profiler is restricted to SM89, got SM{capability[0]}{capability[1]}")
    physical_gpu = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    if physical_gpu not in ("6", "7"):
        raise RuntimeError("set CUDA_VISIBLE_DEVICES to physical GPU 6 or 7")
    if (
        args.warmup < 1
        or args.replays < 1
        or args.decode_sample_steps < 1
        or args.validation_decode_steps < 1
    ):
        raise ValueError(
            "warmup, replays, decode-sample-steps, and "
            "validation-decode-steps must be positive"
        )
    fla_git_commit = os.environ.get("FLA_GIT_COMMIT")
    if fla_git_commit is None:
        raise RuntimeError("set FLA_GIT_COMMIT explicitly")
    if args.output is not None and args.output.exists():
        raise ValueError(f"output already exists: {args.output}")

    print(
        json.dumps(
            {
                "gpu": torch.cuda.get_device_name(0),
                "physical_gpu": physical_gpu,
                "sm": f"{capability[0]}{capability[1]}",
                "torch": torch.__version__,
                "fla_source": str(Path(fla.__file__).resolve()),
            },
            sort_keys=True,
        ),
        flush=True,
    )

    if args.mode == "validate":
        torch.manual_seed(20260810)
        lengths = (17, 19)
        heads = 2
        call = allocate_append(lengths, heads)
        a_log = torch.zeros(heads, device="cuda", dtype=torch.float32)
        dt_bias = torch.zeros(heads, K, device="cuda", dtype=torch.float32)
        norm_weight = torch.randn(V, device="cuda", dtype=torch.float32)
        initial_state = torch.randn(2, heads, V, K, device="cuda", dtype=torch.float32) * 0.01

        q, _ = l2norm_fwd(call.q)
        k, _ = l2norm_fwd(call.k)
        beta = fused_beta_sigmoid(call.beta)
        chunk_raw, chunk_state, *_ = chunk_kda_fwd(
            q=q,
            k=k,
            v=call.v,
            g=call.g,
            beta=beta,
            scale=K**-0.5,
            initial_state=initial_state.clone(),
            output_final_state=True,
            use_gate_in_kernel=True,
            safe_gate=True,
            lower_bound=LOWER_BOUND,
            state_v_first=True,
            cu_seqlens=call.cu_seqlens,
            cu_seqlens_cpu=call.cu_seqlens_cpu,
            chunk_indices=call.chunk_indices[32],
            chunk_size=32,
            A_log=a_log,
            dt_bias=dt_bias,
        )
        recurrent_state = initial_state.clone()
        recurrent_raw, recurrent_state = fused_recurrent_kda_fwd(
            q=call.q,
            k=call.k,
            v=call.v,
            g=call.g,
            beta=call.beta,
            A_log=a_log,
            dt_bias=dt_bias,
            initial_state=recurrent_state,
            scale=K**-0.5,
            output_final_state=True,
            inplace_final_state=True,
            state_v_first=True,
            cu_seqlens=call.cu_seqlens,
            use_qk_l2norm_in_kernel=True,
            use_gate_in_kernel=True,
            use_beta_sigmoid_in_kernel=True,
            lower_bound=LOWER_BOUND,
            out=call.recurrent_raw,
        )
        direct_state = initial_state.clone()
        direct_raw = torch.empty_like(call.recurrent_raw)
        fused_recurrent_kda_fwd_kernel[(4 * len(lengths) * heads,)](
            q=call.q,
            k=call.k,
            v=call.v,
            g=call.g,
            beta=call.beta,
            A_log=a_log,
            dt_bias=dt_bias,
            o=direct_raw,
            h0=direct_state,
            ht=direct_state,
            cu_seqlens=call.cu_seqlens,
            ssm_state_indices=None,
            num_accepted_tokens=None,
            lower_bound=LOWER_BOUND,
            scale=K**-0.5,
            N=len(lengths),
            T=call.q.shape[1],
            H=heads,
            HV=heads,
            K=K,
            V=V,
            BK=K,
            BV=32,
            stride_init_state_token=direct_state.stride(0),
            stride_final_state_token=direct_state.stride(0),
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
        torch.cuda.synchronize()

        pt_output = torch.empty_like(chunk_raw)
        fused_output = torch.empty_like(chunk_raw)
        direct_output = torch.empty_like(chunk_raw)
        apply_epilogue(
            chunk_raw,
            call.output_gate,
            norm_weight,
            pt_output,
            call.output_rstd,
            fused=False,
        )
        apply_epilogue(
            chunk_raw,
            call.output_gate,
            norm_weight,
            fused_output,
            call.output_rstd,
            fused=True,
        )
        apply_epilogue(
            chunk_raw,
            call.output_gate,
            norm_weight,
            direct_output,
            call.output_rstd,
            fused=True,
            direct=True,
        )
        torch.cuda.synchronize()

        metrics = {}
        for name, actual, expected in (
            ("raw_output", chunk_raw, recurrent_raw),
            ("state", chunk_state, recurrent_state),
            ("direct_raw_vs_wrapper", direct_raw, recurrent_raw),
            ("direct_state_vs_wrapper", direct_state, recurrent_state),
            ("fused_epilogue", fused_output, pt_output),
            ("direct_epilogue", direct_output, pt_output),
            ("direct_vs_fla_epilogue", direct_output, fused_output),
        ):
            actual_fp32 = actual.float()
            expected_fp32 = expected.float()
            diff = actual_fp32 - expected_fp32
            metrics[name] = {
                "relative_l2": float(
                    torch.linalg.vector_norm(diff)
                    / torch.linalg.vector_norm(expected_fp32).clamp_min(1e-12)
                ),
                "normalized_max": float(
                    diff.abs().max() / expected_fp32.abs().max().clamp_min(1.0)
                ),
            }
        print(json.dumps({"validation": metrics}, indent=2, sort_keys=True))
        return

    if args.mode == "submission-validate":
        task_dir = Path(__file__).resolve().parents[1]
        sys.path.insert(0, str(task_dir))
        from judge.contract import (
            AppendInputs,
            CaseSpec,
            DecodeInput,
            KDAConfig,
            LayerParams,
            Limits,
            StateMode,
        )
        from submission import Submission as CandidateSubmission

        if args.cases == "all":
            requested_cases = ("C48", "C24", "D48-B1", "D24")
        else:
            requested_cases = tuple(args.cases.split(","))
        supported_cases = {"C48", "C24", "D48-B1", "D48", "D24"}
        if (
            not requested_cases
            or any(not item for item in requested_cases)
            or len(requested_cases) != len(set(requested_cases))
            or not set(requested_cases) <= supported_cases
        ):
            raise ValueError(
                "submission validation cases must be unique members of "
                "C48,C24,D48-B1,D48,D24"
            )

        torch.manual_seed(20260810)
        validation_results: list[dict[str, object]] = []
        passed = True
        for case_name in requested_cases:
            profile_case = next(case for case in CASES if case.name == case_name)
            config = KDAConfig(heads=profile_case.heads)
            case = CaseSpec(
                case_id=f"synthetic-{case_name}",
                config=config,
                batch=profile_case.batch,
                state_mode=StateMode.CHECKPOINT,
                limits=Limits(0.006, 0.004, 0.015),
                append_lengths=profile_case.append_lengths,
                decode_steps=min(
                    profile_case.decode_steps,
                    args.validation_decode_steps,
                ),
            )
            a_log = torch.empty(
                profile_case.heads, device="cuda", dtype=torch.float32
            ).uniform_(-0.25, 0.25)
            dt_bias = torch.randn(
                profile_case.heads, K, device="cuda", dtype=torch.float32
            ) * 0.1
            norm_weight = torch.randn(V, device="cuda", dtype=torch.float32)
            layer = LayerParams(a_log, dt_bias, norm_weight)
            initial_state = torch.randn(
                profile_case.batch,
                profile_case.heads,
                V,
                K,
                device="cuda",
                dtype=torch.float32,
            ) * 0.02
            reference_state = [initial_state.clone()]
            submission = CandidateSubmission()
            context = submission.prepare(config, layer, case)
            private_state = submission.load_state(context, initial_state)
            case_metrics: list[dict[str, object]] = []

            for append_index, lengths in enumerate(profile_case.append_lengths):
                call = allocate_append(lengths, profile_case.heads)
                if case_name in ("C24", "D24"):
                    call.g.add_(-7.0)
                elif case_name == "C48":
                    call.g.add_(3.0)
                elif case_name == "D48":
                    call.g.add_(-2.0)
                run_append(
                    call,
                    "starter32",
                    reference_state,
                    a_log,
                    dt_bias,
                    norm_weight,
                )
                cu_seqlens = call.cu_seqlens
                if cu_seqlens is None:
                    cu_seqlens = torch.tensor(
                        [0, sum(lengths)],
                        device="cuda",
                        dtype=torch.int32,
                    )
                candidate_output = torch.empty(
                    sum(lengths),
                    profile_case.heads,
                    V,
                    device="cuda",
                    dtype=torch.bfloat16,
                )
                submission.append_chunk(
                    context,
                    private_state,
                    AppendInputs(
                        q_act=call.q.squeeze(0),
                        k_act=call.k.squeeze(0),
                        v_act=call.v.squeeze(0),
                        g_raw=call.g.squeeze(0),
                        beta_raw=call.beta.squeeze(0),
                        output_gate_logits=call.output_gate.squeeze(0),
                        cu_seqlens=cu_seqlens,
                        descriptor=torch.empty(
                            0, 4, device="cuda", dtype=torch.int32
                        ),
                    ),
                    candidate_output,
                )
                torch.cuda.synchronize()
                for kind, actual, expected, limit in (
                    (
                        "output",
                        candidate_output,
                        call.output.squeeze(0),
                        case.limits.output_relative_l2,
                    ),
                    (
                        "state",
                        private_state.canonical_state,
                        reference_state[0],
                        case.limits.state_relative_l2,
                    ),
                ):
                    actual_fp32 = actual.float()
                    expected_fp32 = expected.float()
                    diff = actual_fp32 - expected_fp32
                    relative_l2 = float(
                        torch.linalg.vector_norm(diff)
                        / torch.linalg.vector_norm(expected_fp32).clamp_min(1e-12)
                    )
                    normalized_max = float(
                        diff.abs().max()
                        / expected_fp32.abs().max().clamp_min(1.0)
                    )
                    metric_passed = (
                        math.isfinite(relative_l2)
                        and math.isfinite(normalized_max)
                        and relative_l2 <= limit
                        and normalized_max <= case.limits.normalized_max
                    )
                    passed &= metric_passed
                    case_metrics.append(
                        {
                            "point": f"append-{append_index + 1}",
                            "kind": kind,
                            "relative_l2": relative_l2,
                            "normalized_max": normalized_max,
                            "passed": metric_passed,
                        }
                    )

            if case.decode_steps:
                call = allocate_decode(profile_case.batch, profile_case.heads)
                if case_name == "D24":
                    call.g.add_(-7.0)
                elif case_name == "D48":
                    call.g.add_(-2.0)
                candidate_output = torch.empty(
                    profile_case.batch,
                    profile_case.heads,
                    V,
                    device="cuda",
                    dtype=torch.bfloat16,
                )
                token = DecodeInput(
                    q_act=call.q.squeeze(1),
                    k_act=call.k.squeeze(1),
                    v_act=call.v.squeeze(1),
                    g_raw=call.g.squeeze(1),
                    beta_raw=call.beta.squeeze(1),
                    output_gate_logits=call.output_gate.squeeze(1),
                )
                checkpoints = {
                    step
                    for step in (1, 17, 257, 4096, case.decode_steps)
                    if step <= case.decode_steps
                }
                for step in range(1, case.decode_steps + 1):
                    run_decode(
                        call,
                        "starter32",
                        reference_state,
                        a_log,
                        dt_bias,
                        norm_weight,
                    )
                    submission.decode_step(
                        context,
                        private_state,
                        token,
                        candidate_output,
                    )
                    if step not in checkpoints:
                        continue
                    torch.cuda.synchronize()
                    for kind, actual, expected, limit in (
                        (
                            "output",
                            candidate_output,
                            call.output.squeeze(1),
                            case.limits.output_relative_l2,
                        ),
                        (
                            "state",
                            private_state.canonical_state,
                            reference_state[0],
                            case.limits.state_relative_l2,
                        ),
                    ):
                        actual_fp32 = actual.float()
                        expected_fp32 = expected.float()
                        diff = actual_fp32 - expected_fp32
                        relative_l2 = float(
                            torch.linalg.vector_norm(diff)
                            / torch.linalg.vector_norm(expected_fp32).clamp_min(1e-12)
                        )
                        normalized_max = float(
                            diff.abs().max()
                            / expected_fp32.abs().max().clamp_min(1.0)
                        )
                        metric_passed = (
                            math.isfinite(relative_l2)
                            and math.isfinite(normalized_max)
                            and relative_l2 <= limit
                            and normalized_max <= case.limits.normalized_max
                        )
                        passed &= metric_passed
                        case_metrics.append(
                            {
                                "point": f"decode-{step}",
                                "kind": kind,
                                "relative_l2": relative_l2,
                                "normalized_max": normalized_max,
                                "passed": metric_passed,
                            }
                        )

            case_result = {
                "case": case_name,
                "decode_steps": case.decode_steps,
                "metrics": case_metrics,
            }
            validation_results.append(case_result)
            print(json.dumps(case_result, sort_keys=True), flush=True)
            del submission, context, private_state, reference_state, initial_state
            gc.collect()
            torch.cuda.empty_cache()

        payload = {
            "submission_validation": validation_results,
            "passed": passed,
        }
        print(json.dumps({"summary": {"passed": passed}}, sort_keys=True))
        if args.output is not None:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(
                json.dumps(payload, indent=2, sort_keys=True) + "\n"
            )
        if not passed:
            raise RuntimeError("submission trajectory validation failed")
        return

    if args.cases == "all":
        selected_cases = CASES
    else:
        requested_cases = args.cases.split(",")
        if not requested_cases or any(not item for item in requested_cases):
            raise ValueError("--cases must be 'all' or a comma-separated case list")
        unknown_cases = set(requested_cases) - {case.name for case in CASES}
        if unknown_cases:
            raise ValueError(f"unknown cases: {sorted(unknown_cases)}")
        if len(requested_cases) != len(set(requested_cases)):
            raise ValueError("--cases contains duplicates")
        selected_cases = tuple(case for case in CASES if case.name in requested_cases)
    selected_candidates = tuple(args.candidates.split(","))
    if not selected_candidates or any(not item for item in selected_candidates):
        raise ValueError("--candidates must be a comma-separated candidate list")
    if len(selected_candidates) != len(set(selected_candidates)):
        raise ValueError("--candidates contains duplicates")
    unknown = set(selected_candidates) - set(CANDIDATES)
    if unknown:
        raise ValueError(f"unknown candidates: {sorted(unknown)}")
    if args.cuda_profiler_api and (
        len(selected_cases) != 1
        or len(selected_candidates) != 1
        or args.replays != 1
    ):
        raise ValueError(
            "--cuda-profiler-api requires exactly one case, one candidate, "
            "and --replays 1"
        )

    torch.manual_seed(20260810)
    started = time.time()
    results = []
    for case in selected_cases:
        for candidate in selected_candidates:
            result = benchmark_case(
                case,
                candidate,
                args.warmup,
                args.replays,
                args.decode_sample_steps,
                args.nvtx,
                args.cuda_profiler_api,
            )
            results.append(result)
            print(json.dumps(result, sort_keys=True), flush=True)

    summaries = {}
    if len(selected_cases) == len(CASES):
        for candidate in selected_candidates:
            candidate_results = [item for item in results if item["candidate"] == candidate]
            if len(candidate_results) == len(CASES):
                summaries[candidate] = math.exp(
                    sum(item["weight"] * math.log(item["estimated_case_ms"]) for item in candidate_results)
                )
    payload = {
        "metadata": {
            "gpu": torch.cuda.get_device_name(0),
            "physical_gpu": physical_gpu,
            "sm": "89",
            "torch": torch.__version__,
            "fla_git": fla_git_commit,
            "synthetic_inputs": True,
            "decode_sample_steps": args.decode_sample_steps,
            "warmup": args.warmup,
            "replays": args.replays,
            "wall_time_sec": time.time() - started,
        },
        "cases": [asdict(case) for case in selected_cases],
        "results": results,
        "estimated_weighted_gpu_time_ms": summaries,
    }
    print(json.dumps({"summary": summaries}, sort_keys=True), flush=True)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    with torch.inference_mode():
        main()
