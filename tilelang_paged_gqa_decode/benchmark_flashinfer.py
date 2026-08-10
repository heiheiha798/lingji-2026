from __future__ import annotations

import argparse
import gc
import json
import math
from pathlib import Path
from typing import Any

import flashinfer
import torch

import benchmark as contest


class FlashInferInvoker:
    def __init__(
        self,
        wrappers: list[flashinfer.BatchDecodeWithPagedKVCacheWrapper],
        buffers: list[tuple[torch.Tensor, ...]],
    ) -> None:
        self.wrappers = wrappers
        self.buffers = buffers
        self.buffer_count = len(buffers)

    def __call__(self, index: int) -> None:
        q, k_cache, v_cache, _, _, _, out = self.buffers[index]
        self.wrappers[index].run(q, (k_cache, v_cache), out=out)


def build_wrappers(
    spec: contest.OpSpec,
    buffers: list[tuple[torch.Tensor, ...]],
    *,
    use_tensor_cores: bool,
    fixed_split_pages: int | None,
) -> list[flashinfer.BatchDecodeWithPagedKVCacheWrapper]:
    wrappers: list[flashinfer.BatchDecodeWithPagedKVCacheWrapper] = []
    for _, _, _, block_table, seq_lens, _, _ in buffers:
        lengths = seq_lens.cpu()
        page_counts = torch.div(
            lengths + spec.page_size - 1,
            spec.page_size,
            rounding_mode="floor",
        )
        indptr = torch.zeros(spec.batch_size + 1, dtype=torch.int32)
        indptr[1:] = torch.cumsum(page_counts, dim=0)
        table_cpu = block_table.cpu()
        indices = torch.cat(
            [
                table_cpu[batch, : int(page_counts[batch])]
                for batch in range(spec.batch_size)
            ]
        ).contiguous()
        last_page_len = ((lengths - 1) % spec.page_size + 1).to(torch.int32)
        workspace = torch.empty(
            spec.workspace_bytes, dtype=torch.uint8, device=seq_lens.device
        )
        wrapper = flashinfer.BatchDecodeWithPagedKVCacheWrapper(
            workspace,
            kv_layout="NHD",
            use_tensor_cores=use_tensor_cores,
            backend="fa2" if use_tensor_cores else "auto",
        )
        wrapper.plan(
            indptr,
            indices,
            last_page_len,
            spec.num_q_heads,
            spec.num_kv_heads,
            spec.head_dim,
            spec.page_size,
            q_data_type=torch.bfloat16,
            kv_data_type=torch.bfloat16,
            o_data_type=torch.bfloat16,
            seq_lens=lengths,
            fixed_split_size=fixed_split_pages,
        )
        wrappers.append(wrapper)
    return wrappers


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--preset", choices=("local", "official"), default="local")
    parser.add_argument("--tensor-cores", action="store_true")
    parser.add_argument("--fixed-split-pages", type=int)
    parser.add_argument("--json-out", type=Path)
    args = parser.parse_args()
    if args.fixed_split_pages is not None and args.fixed_split_pages <= 0:
        parser.error("--fixed-split-pages must be positive")
    if args.fixed_split_pages is not None and not args.tensor_cores:
        parser.error("--fixed-split-pages requires --tensor-cores")

    device = torch.device("cuda")
    settings = contest._preset(args.preset)
    case_groups = json.loads(
        Path(__file__).with_name("cases.json").read_text(encoding="utf-8")
    )

    basic_case = case_groups["basic"][1]
    basic_spec = contest._make_spec(basic_case)
    basic_buffers = contest._make_buffers(
        basic_spec, basic_case, int(basic_case["seed"]), device, 1
    )
    basic_wrappers = build_wrappers(
        basic_spec,
        basic_buffers,
        use_tensor_cores=args.tensor_cores,
        fixed_split_pages=args.fixed_split_pages,
    )
    expected = contest.paged_gqa_reference(*basic_buffers[0][:5])
    FlashInferInvoker(basic_wrappers, basic_buffers)(0)
    torch.cuda.synchronize()
    correctness = contest.validate_output(basic_buffers[0][-1], expected)
    print(
        "basic correctness "
        f"nrmse={correctness['nrmse']:.3e} "
        f"max_abs={correctness['max_abs_error']:.3e}"
    )
    del basic_wrappers, basic_buffers, expected
    gc.collect()

    rows: list[dict[str, Any]] = []
    for case in case_groups["official"]:
        spec = contest._make_spec(case)
        variants: list[dict[str, Any]] = []
        for variant_index in range(4):
            variant_seed = int(case["seed"]) + variant_index * contest.SEED_STRIDE
            buffers = contest._make_buffers(
                spec,
                case,
                variant_seed,
                device,
                int(settings["buffers"]),
            )
            wrappers = build_wrappers(
                spec,
                buffers,
                use_tensor_cores=args.tensor_cores,
                fixed_split_pages=args.fixed_split_pages,
            )
            measurement = contest._measure(
                FlashInferInvoker(wrappers, buffers),
                device,
                warmup=int(settings["warmup"]),
                trials=int(settings["trials"]),
                min_trial_ms=float(settings["min_trial_ms"]),
                max_iterations=int(settings["max_iterations"]),
            )
            measurement["seed"] = variant_seed
            variants.append(measurement)
            del wrappers, buffers
            gc.collect()

        row = {
            "name": case["name"],
            "weight": float(case["weight"]),
            "latency_us": math.exp(
                sum(math.log(item["latency_us"]) for item in variants)
                / len(variants)
            ),
            "variants": variants,
        }
        rows.append(row)
        print(f"{row['name']:<36} {row['latency_us']:9.3f} us")

    weighted_latency_us = contest._weighted_geomean(rows, "latency_us")
    print(f"weighted geometric mean: {weighted_latency_us:.3f} us")
    result = {
        "flashinfer": flashinfer.__version__,
        "torch": torch.__version__,
        "preset": args.preset,
        "use_tensor_cores": args.tensor_cores,
        "fixed_split_pages": args.fixed_split_pages,
        "correctness": correctness,
        "rows": rows,
        "weighted_latency_us": weighted_latency_us,
    }
    if args.json_out is not None:
        args.json_out.write_text(json.dumps(result, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
