from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from benchmark import _make_buffers, _make_l2_scrub, _make_spec, _scrub_l2
from benchmark_fla_baseline import FLASubmission
from fla.modules.l2norm import l2norm_fwd_kernel
from fla.ops.common.chunk_delta_h import chunk_gated_delta_rule_fwd_kernel_h_blockdim64
from fla.ops.gla.chunk import chunk_gla_fwd_kernel_o
from fla.ops.kda.chunk_intra import (
    chunk_kda_fwd_kernel_inter_solve_fused,
    chunk_kda_fwd_kernel_intra_sub_chunk,
)
from fla.ops.kda.gate import kda_gate_chunk_cumsum_vector_kernel
from fla.ops.kda.wy_fast import recompute_w_u_fwd_kda_kernel


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--case", choices=("K1", "K2", "K3", "K4"), required=True)
    args = parser.parse_args()

    cases = json.loads(
        Path(__file__).with_name("cases.json").read_text(encoding="utf-8")
    )["official"]
    case = next(case for case in cases if case["name"].startswith(f"{args.case}_"))
    device = torch.device("cuda")
    spec, lengths = _make_spec(case)
    submission = FLASubmission()
    state = submission.build(spec)
    inputs = _make_buffers(
        spec,
        lengths,
        case,
        int(case["seed"]),
        device,
        1,
    )[0]

    for _ in range(2):
        submission.run(state, *inputs)
    torch.cuda.synchronize(device)

    for name, autotuner in (
        ("l2norm", l2norm_fwd_kernel),
        ("gate_cumsum", kda_gate_chunk_cumsum_vector_kernel.fn),
        ("intra_sub_chunk", chunk_kda_fwd_kernel_intra_sub_chunk.fn),
        ("inter_solve", chunk_kda_fwd_kernel_inter_solve_fused.fn),
        ("recompute_w_u", recompute_w_u_fwd_kda_kernel.fn),
        ("state", chunk_gated_delta_rule_fwd_kernel_h_blockdim64.fn),
        ("output", chunk_gla_fwd_kernel_o.fn),
    ):
        for key, config in autotuner.cache.items():
            print(f"autotune {name}: key={key} config={config}", flush=True)

    l2_scrub = _make_l2_scrub(device)
    _scrub_l2(l2_scrub)
    torch.cuda.synchronize(device)

    torch.cuda.cudart().cudaProfilerStart()
    torch.cuda.nvtx.range_push(f"FLA_{args.case}")
    submission.run(state, *inputs)
    torch.cuda.synchronize(device)
    torch.cuda.nvtx.range_pop()
    torch.cuda.cudart().cudaProfilerStop()
    print(
        f"profiled FLA {args.case}: T={spec.total_tokens} "
        f"B={spec.num_sequences} H={spec.num_heads}",
        flush=True,
    )


if __name__ == "__main__":
    main()
