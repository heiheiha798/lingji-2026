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
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import torch
import fla
import triton

from fla.modules.fused_norm_gate import layer_norm_gated_fwd_kernel, rms_norm_gated
from fla.modules.l2norm import l2norm_fwd
from fla.ops.common.gate import fused_beta_sigmoid
from fla.ops.kda import fused_recurrent_kda
from fla.ops.kda.chunk_fwd import chunk_kda_fwd
from fla.ops.kda.fused_recurrent import fused_recurrent_kda_fwd
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
    use_recurrent |= candidate in ("hybrid64", "hybrid64_direct_epi") and max_length <= 64
    use_recurrent |= candidate == "hybrid256" and max_length <= 256

    if use_recurrent:
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
            direct=candidate == "hybrid64_direct_epi",
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
        direct=candidate == "hybrid64_direct_epi",
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
        direct=candidate == "hybrid64_direct_epi",
    )


def benchmark_case(
    case: Case,
    candidate: str,
    warmup: int,
    replays: int,
    decode_sample_steps: int,
    nvtx: bool,
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
    parser.add_argument("--mode", choices=("validate", "benchmark"), default="benchmark")
    parser.add_argument("--cases", default="all")
    parser.add_argument("--candidates", default="hybrid64")
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--replays", type=int, default=3)
    parser.add_argument("--decode-sample-steps", type=int, default=128)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--nvtx", action="store_true")
    args = parser.parse_args()

    if torch.cuda.device_count() != 1:
        raise RuntimeError("set CUDA_VISIBLE_DEVICES to exactly one physical GPU")
    capability = torch.cuda.get_device_capability(0)
    if capability != (8, 9):
        raise RuntimeError(f"this profiler is restricted to SM89, got SM{capability[0]}{capability[1]}")
    physical_gpu = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    if physical_gpu not in ("6", "7"):
        raise RuntimeError("set CUDA_VISIBLE_DEVICES to physical GPU 6 or 7")
    if args.warmup < 1 or args.replays < 1 or args.decode_sample_steps < 1:
        raise ValueError("warmup, replays, and decode-sample-steps must be positive")

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

    selected_cases = CASES if args.cases == "all" else tuple(
        case for case in CASES if case.name in args.cases.split(",")
    )
    selected_candidates = tuple(item for item in args.candidates.split(",") if item)
    if not selected_cases:
        raise ValueError(f"no cases selected by {args.cases!r}")
    unknown = set(selected_candidates) - set(CANDIDATES)
    if unknown:
        raise ValueError(f"unknown candidates: {sorted(unknown)}")

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
            "fla_git": os.environ.get("FLA_GIT_COMMIT", "unknown"),
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
