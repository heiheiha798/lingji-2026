from typing import Any

import tilelang
import tilelang.language as T
import torch


_HEAD_DIM = 128
_BLOCK_H = 16
_THREADS = 32
_LOG2E_OVER_SQRT_HEAD_DIM = 0.12749916314874343
_TARGET = {"kind": "cuda", "arch": "sm_89"}


@tilelang.jit(
    out_idx=[],
    target=_TARGET,
    pass_configs={tilelang.PassConfigKey.TL_ENABLE_FAST_MATH: True},
)
def _compile_paged_gqa_direct(
    batch_size: int,
    num_q_heads: int,
    num_kv_heads: int,
    num_pages: int,
    max_seq_len: int,
    page_size: int,
):
    group_size = num_q_heads // num_kv_heads
    max_pages = (max_seq_len + page_size - 1) // page_size

    @T.prim_func
    def kernel(
        Q: T.Tensor((batch_size, num_q_heads, _HEAD_DIM), T.bfloat16),
        KCache: T.Tensor(
            (num_pages, page_size, num_kv_heads, _HEAD_DIM), T.bfloat16
        ),
        VCache: T.Tensor(
            (num_pages, page_size, num_kv_heads, _HEAD_DIM), T.bfloat16
        ),
        BlockTable: T.Tensor((batch_size, max_pages), T.int32),
        SeqLens: T.Tensor((batch_size,), T.int32),
        Output: T.Tensor(
            (batch_size, num_q_heads, _HEAD_DIM), T.bfloat16
        ),
    ):
        with T.Kernel(
            batch_size, num_kv_heads, threads=_THREADS
        ) as (batch, kv_head):
            q_shared = T.alloc_shared((_BLOCK_H, _HEAD_DIM), T.bfloat16)
            k_shared = T.alloc_shared((page_size, _HEAD_DIM), T.bfloat16)
            v_shared = T.alloc_shared((page_size, _HEAD_DIM), T.bfloat16)
            probability_shared = T.alloc_shared(
                (_BLOCK_H, page_size), T.bfloat16
            )
            score = T.alloc_fragment((_BLOCK_H, page_size), T.float32)
            output = T.alloc_fragment((_BLOCK_H, _HEAD_DIM), T.float32)
            score_max = T.alloc_fragment((_BLOCK_H,), T.float32)
            previous_max = T.alloc_fragment((_BLOCK_H,), T.float32)
            score_scale = T.alloc_fragment((_BLOCK_H,), T.float32)
            score_sum = T.alloc_fragment((_BLOCK_H,), T.float32)
            normalizer = T.alloc_fragment((_BLOCK_H,), T.float32)

            T.clear(q_shared)
            T.copy(
                Q[
                    batch,
                    kv_head * group_size : (kv_head + 1) * group_size,
                    :,
                ],
                q_shared[0:group_size, :],
            )
            T.clear(output)
            T.clear(normalizer)
            T.fill(score_max, -T.infinity(T.float32))

            page_count = T.ceildiv(SeqLens[batch], page_size)
            for logical_page in T.Pipelined(page_count, num_stages=1):
                physical_page = BlockTable[batch, logical_page]
                T.copy(
                    KCache[physical_page, :, kv_head, :],
                    k_shared,
                )
                T.copy(
                    VCache[physical_page, :, kv_head, :],
                    v_shared,
                )
                T.clear(score)
                T.gemm(
                    q_shared,
                    k_shared,
                    score,
                    transpose_B=True,
                    policy=T.GemmWarpPolicy.FullCol,
                )
                T.copy(score_max, previous_max)
                T.fill(score_max, -T.infinity(T.float32))
                for row, token in T.Parallel(_BLOCK_H, page_size):
                    score[row, token] = T.if_then_else(
                        row < group_size
                        and logical_page * page_size + token < SeqLens[batch],
                        score[row, token],
                        -T.infinity(T.float32),
                    )
                T.reduce_max(score, score_max, dim=1, clear=False)
                for row in T.Parallel(_BLOCK_H):
                    score_max[row] = T.max(score_max[row], previous_max[row])
                    score_scale[row] = T.exp2(
                        (previous_max[row] - score_max[row])
                        * _LOG2E_OVER_SQRT_HEAD_DIM
                    )
                for row, token in T.Parallel(_BLOCK_H, page_size):
                    score[row, token] = T.exp2(
                        (score[row, token] - score_max[row])
                        * _LOG2E_OVER_SQRT_HEAD_DIM
                    )
                T.reduce_sum(score, score_sum, dim=1)
                T.copy(score, probability_shared)
                for row in T.Parallel(_BLOCK_H):
                    normalizer[row] = (
                        normalizer[row] * score_scale[row] + score_sum[row]
                    )
                for row, dim in T.Parallel(_BLOCK_H, _HEAD_DIM):
                    output[row, dim] *= score_scale[row]
                T.gemm(
                    probability_shared,
                    v_shared,
                    output,
                    policy=T.GemmWarpPolicy.FullCol,
                )

            for row, dim in T.Parallel(group_size, _HEAD_DIM):
                Output[
                    batch, kv_head * group_size + row, dim
                ] = T.cast(output[row, dim] / normalizer[row], T.bfloat16)

    return kernel


@tilelang.jit(
    out_idx=[],
    target=_TARGET,
    pass_configs={tilelang.PassConfigKey.TL_ENABLE_FAST_MATH: True},
)
def _compile_paged_gqa_split(
    batch_size: int,
    num_q_heads: int,
    num_kv_heads: int,
    num_pages: int,
    max_seq_len: int,
    page_size: int,
    num_splits: int,
):
    group_size = num_q_heads // num_kv_heads
    max_pages = (max_seq_len + page_size - 1) // page_size

    @T.prim_func
    def kernel(
        Q: T.Tensor((batch_size, num_q_heads, _HEAD_DIM), T.bfloat16),
        KCache: T.Tensor(
            (num_pages, page_size, num_kv_heads, _HEAD_DIM), T.bfloat16
        ),
        VCache: T.Tensor(
            (num_pages, page_size, num_kv_heads, _HEAD_DIM), T.bfloat16
        ),
        BlockTable: T.Tensor((batch_size, max_pages), T.int32),
        SeqLens: T.Tensor((batch_size,), T.int32),
        PartialOutput: T.Tensor(
            (batch_size, num_q_heads, num_splits, _HEAD_DIM), T.float32
        ),
        LogSumExp: T.Tensor(
            (batch_size, num_q_heads, num_splits), T.float32
        ),
    ):
        with T.Kernel(
            batch_size, num_kv_heads, num_splits, threads=_THREADS
        ) as (batch, kv_head, split):
            q_shared = T.alloc_shared((_BLOCK_H, _HEAD_DIM), T.bfloat16)
            k_shared = T.alloc_shared((page_size, _HEAD_DIM), T.bfloat16)
            v_shared = T.alloc_shared((page_size, _HEAD_DIM), T.bfloat16)
            probability_shared = T.alloc_shared(
                (_BLOCK_H, page_size), T.bfloat16
            )
            score = T.alloc_fragment((_BLOCK_H, page_size), T.float32)
            output = T.alloc_fragment((_BLOCK_H, _HEAD_DIM), T.float32)
            score_max = T.alloc_fragment((_BLOCK_H,), T.float32)
            previous_max = T.alloc_fragment((_BLOCK_H,), T.float32)
            score_scale = T.alloc_fragment((_BLOCK_H,), T.float32)
            score_sum = T.alloc_fragment((_BLOCK_H,), T.float32)
            normalizer = T.alloc_fragment((_BLOCK_H,), T.float32)

            T.clear(q_shared)
            T.copy(
                Q[
                    batch,
                    kv_head * group_size : (kv_head + 1) * group_size,
                    :,
                ],
                q_shared[0:group_size, :],
            )
            T.clear(output)
            T.clear(normalizer)
            T.fill(score_max, -T.infinity(T.float32))

            page_count = T.ceildiv(SeqLens[batch], page_size)
            pages_per_split = T.floordiv(page_count, num_splits)
            remaining_pages = T.floormod(page_count, num_splits)
            split_page_count = pages_per_split + T.if_then_else(
                split < remaining_pages, 1, 0
            )
            start_page = (
                pages_per_split * split + T.min(split, remaining_pages)
            )
            for page_offset in T.Pipelined(
                split_page_count, num_stages=1
            ):
                logical_page = start_page + page_offset
                physical_page = BlockTable[batch, logical_page]
                T.copy(
                    KCache[physical_page, :, kv_head, :],
                    k_shared,
                )
                T.copy(
                    VCache[physical_page, :, kv_head, :],
                    v_shared,
                )
                T.clear(score)
                T.gemm(
                    q_shared,
                    k_shared,
                    score,
                    transpose_B=True,
                    policy=T.GemmWarpPolicy.FullCol,
                )
                T.copy(score_max, previous_max)
                T.fill(score_max, -T.infinity(T.float32))
                for row, token in T.Parallel(_BLOCK_H, page_size):
                    score[row, token] = T.if_then_else(
                        row < group_size
                        and logical_page * page_size + token < SeqLens[batch],
                        score[row, token],
                        -T.infinity(T.float32),
                    )
                T.reduce_max(score, score_max, dim=1, clear=False)
                for row in T.Parallel(_BLOCK_H):
                    score_max[row] = T.max(score_max[row], previous_max[row])
                    score_scale[row] = T.exp2(
                        (previous_max[row] - score_max[row])
                        * _LOG2E_OVER_SQRT_HEAD_DIM
                    )
                for row, token in T.Parallel(_BLOCK_H, page_size):
                    score[row, token] = T.exp2(
                        (score[row, token] - score_max[row])
                        * _LOG2E_OVER_SQRT_HEAD_DIM
                    )
                T.reduce_sum(score, score_sum, dim=1)
                T.copy(score, probability_shared)
                for row in T.Parallel(_BLOCK_H):
                    normalizer[row] = (
                        normalizer[row] * score_scale[row] + score_sum[row]
                    )
                for row, dim in T.Parallel(_BLOCK_H, _HEAD_DIM):
                    output[row, dim] *= score_scale[row]
                T.gemm(
                    probability_shared,
                    v_shared,
                    output,
                    policy=T.GemmWarpPolicy.FullCol,
                )

            if split_page_count > 0:
                for row in T.Parallel(group_size):
                    LogSumExp[
                        batch, kv_head * group_size + row, split
                    ] = (
                        T.log2(normalizer[row])
                        + score_max[row] * _LOG2E_OVER_SQRT_HEAD_DIM
                    )
                for row, dim in T.Parallel(group_size, _HEAD_DIM):
                    PartialOutput[
                        batch, kv_head * group_size + row, split, dim
                    ] = output[row, dim] / normalizer[row]
            else:
                for row in T.Parallel(group_size):
                    LogSumExp[
                        batch, kv_head * group_size + row, split
                    ] = -T.infinity(T.float32)
                for row, dim in T.Parallel(group_size, _HEAD_DIM):
                    PartialOutput[
                        batch, kv_head * group_size + row, split, dim
                    ] = 0.0

    return kernel


@tilelang.jit(
    out_idx=[],
    target=_TARGET,
    pass_configs={tilelang.PassConfigKey.TL_ENABLE_FAST_MATH: True},
)
def _compile_split_combine(
    batch_size: int,
    num_q_heads: int,
    num_kv_heads: int,
    num_splits: int,
):
    group_size = num_q_heads // num_kv_heads

    @T.prim_func
    def kernel(
        PartialOutput: T.Tensor(
            (batch_size, num_q_heads, num_splits, _HEAD_DIM), T.float32
        ),
        LogSumExp: T.Tensor(
            (batch_size, num_q_heads, num_splits), T.float32
        ),
        Output: T.Tensor(
            (batch_size, num_q_heads, _HEAD_DIM), T.bfloat16
        ),
    ):
        with T.Kernel(
            batch_size, num_kv_heads, threads=_THREADS
        ) as (batch, kv_head):
            output = T.alloc_fragment((_BLOCK_H, _HEAD_DIM), T.float32)
            lse_max = T.alloc_fragment((_BLOCK_H,), T.float32)
            lse_sum = T.alloc_fragment((_BLOCK_H,), T.float32)
            split_weight = T.alloc_fragment((_BLOCK_H,), T.float32)

            T.clear(output)
            T.clear(lse_sum)
            T.fill(lse_max, -T.infinity(T.float32))
            for split in T.serial(num_splits):
                for row in T.Parallel(group_size):
                    lse_max[row] = T.max(
                        lse_max[row],
                        LogSumExp[
                            batch, kv_head * group_size + row, split
                        ],
                    )
            for split in T.serial(num_splits):
                for row in T.Parallel(group_size):
                    split_weight[row] = T.exp2(
                        LogSumExp[
                            batch, kv_head * group_size + row, split
                        ]
                        - lse_max[row]
                    )
                    lse_sum[row] += split_weight[row]
                for row, dim in T.Parallel(group_size, _HEAD_DIM):
                    output[row, dim] += (
                        PartialOutput[
                            batch,
                            kv_head * group_size + row,
                            split,
                            dim,
                        ]
                        * split_weight[row]
                    )
            for row, dim in T.Parallel(group_size, _HEAD_DIM):
                Output[
                    batch, kv_head * group_size + row, dim
                ] = T.cast(output[row, dim] / lse_sum[row], T.bfloat16)

    return kernel


class Submission:
    def build(self, spec: Any) -> Any:
        batch_size = int(spec.batch_size)
        num_q_heads = int(spec.num_q_heads)
        num_kv_heads = int(spec.num_kv_heads)
        max_seq_len = int(spec.max_seq_len)
        work_items = batch_size * num_kv_heads
        if max_seq_len >= 16384 and work_items <= 8:
            num_splits = 16
        elif max_seq_len >= 16384 and work_items <= 64:
            num_splits = 16
        elif max_seq_len >= 16384 and work_items <= 256:
            num_splits = 8
        else:
            num_splits = 1

        if num_splits == 1:
            return (
                num_splits,
                _compile_paged_gqa_direct(
                    batch_size,
                    num_q_heads,
                    num_kv_heads,
                    int(spec.num_pages),
                    max_seq_len,
                    int(spec.page_size),
                ),
                None,
                0,
            )

        partial_elements = (
            batch_size * num_q_heads * num_splits * _HEAD_DIM
        )
        required_bytes = 4 * (
            partial_elements + batch_size * num_q_heads * num_splits
        )
        if required_bytes > int(spec.workspace_bytes):
            raise ValueError("workspace is too small for split-KV reduction")
        return (
            num_splits,
            _compile_paged_gqa_split(
                batch_size,
                num_q_heads,
                num_kv_heads,
                int(spec.num_pages),
                max_seq_len,
                int(spec.page_size),
                num_splits,
            ),
            _compile_split_combine(
                batch_size,
                num_q_heads,
                num_kv_heads,
                num_splits,
            ),
            partial_elements,
        )

    @torch.no_grad()
    def run(
        self,
        state: Any,
        q: torch.Tensor,
        k_cache: torch.Tensor,
        v_cache: torch.Tensor,
        block_table: torch.Tensor,
        seq_lens: torch.Tensor,
        workspace: torch.Tensor,
        out: torch.Tensor,
    ) -> None:
        num_splits, attention, combine, partial_elements = state
        if num_splits == 1:
            attention(q, k_cache, v_cache, block_table, seq_lens, out)
            return

        workspace_float = workspace.view(torch.float32)
        partial_output = workspace_float[:partial_elements].view(
            q.shape[0], q.shape[1], num_splits, _HEAD_DIM
        )
        log_sum_exp = workspace_float[
            partial_elements : partial_elements
            + q.shape[0] * q.shape[1] * num_splits
        ].view(q.shape[0], q.shape[1], num_splits)
        attention(
            q,
            k_cache,
            v_cache,
            block_table,
            seq_lens,
            partial_output,
            log_sum_exp,
        )
        combine(partial_output, log_sum_exp, out)
