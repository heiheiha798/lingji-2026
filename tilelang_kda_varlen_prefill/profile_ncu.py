from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from benchmark import (
    _make_buffers,
    _make_l2_scrub,
    _make_spec,
    _scrub_l2,
)
from submission import Submission


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
    submission = Submission()
    state = submission.build(spec)
    inputs = _make_buffers(
        spec,
        lengths,
        case,
        int(case["seed"]),
        device,
        1,
    )[0]

    submission.run(state, *inputs)
    torch.cuda.synchronize(device)

    l2_scrub = _make_l2_scrub(device)
    _scrub_l2(l2_scrub)
    torch.cuda.synchronize(device)

    torch.cuda.cudart().cudaProfilerStart()
    torch.cuda.nvtx.range_push(f"TileLang_{args.case}")
    submission.run(state, *inputs)
    torch.cuda.synchronize(device)
    torch.cuda.nvtx.range_pop()
    torch.cuda.cudart().cudaProfilerStop()
    print(
        f"profiled {args.case}: T={spec.total_tokens} "
        f"B={spec.num_sequences} H={spec.num_heads}",
        flush=True,
    )


if __name__ == "__main__":
    main()
