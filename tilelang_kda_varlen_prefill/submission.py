from typing import Any

import tilelang
import tilelang.language as T
import torch


_CHUNK_SIZE = 64
_SUBCHUNK_SIZE = 16
_HEAD_DIM = 128
_WARP_SIZE = 32
_DIAGONAL_THREADS = 128
_INTER_THREADS = 64
_THREADS = 128
_INV_SQRT_HEAD_DIM = 0.08838834764831845
_LOG2_GATE_SCALE = -7.213475204444817
_TARGET = {"kind": "cuda", "arch": "sm_89"}


@tilelang.jit(
    out_idx=[],
    target=_TARGET,
    pass_configs={tilelang.PassConfigKey.TL_ENABLE_FAST_MATH: True},
)
def _compile_chunk_diagonal(total_tokens: int, num_sequences: int, num_heads: int):
    max_chunks = (total_tokens + _CHUNK_SIZE - 1) // _CHUNK_SIZE + num_sequences - 1

    token_shape = (total_tokens, num_heads, _HEAD_DIM)
    beta_shape = (total_tokens, num_heads)
    a_log_shape = (num_heads,)
    dt_bias_shape = (num_heads, _HEAD_DIM)
    cu_shape = (num_sequences + 1,)
    diagonal_shape = (total_tokens, num_heads, _CHUNK_SIZE // 2)
    operator_shape = (total_tokens, num_heads, _CHUNK_SIZE)
    norm_shape = (total_tokens, num_heads, 2)

    @T.prim_func
    def kernel(
        Q: T.Tensor(token_shape, T.bfloat16),
        K: T.Tensor(token_shape, T.bfloat16),
        GRaw: T.Tensor(token_shape, T.bfloat16),
        BetaRaw: T.Tensor(beta_shape, T.bfloat16),
        ALog: T.Tensor(a_log_shape, T.float32),
        DtBias: T.Tensor(dt_bias_shape, T.float32),
        CuSeqLens: T.Tensor(cu_shape, T.int32),
        AInvDiagonal: T.Tensor(diagonal_shape, T.float32),
        Aqk: T.Tensor(operator_shape, T.bfloat16),
        Norm: T.Tensor(norm_shape, T.float32),
        Beta: T.Tensor(beta_shape, T.bfloat16),
    ):
        with T.Kernel(
            max_chunks,
            _CHUNK_SIZE // _SUBCHUNK_SIZE,
            num_heads,
            threads=_DIAGONAL_THREADS,
        ) as (chunk_id, subchunk, head):
            thread = T.get_thread_binding(0)
            warp = thread // _WARP_SIZE
            lane = thread % _WARP_SIZE
            sequence_id = T.alloc_local((1,), T.int32)
            chunk_in_sequence = T.alloc_local((1,), T.int32)
            chunk_prefix = T.alloc_local((1,), T.int32)

            sequence_id[0] = -1
            chunk_in_sequence[0] = 0
            chunk_prefix[0] = 0
            if num_sequences == 1:
                sequence_id[0] = 0
                chunk_in_sequence[0] = chunk_id
            else:
                for sequence in T.serial(num_sequences):
                    sequence_chunks = T.ceildiv(
                        CuSeqLens[sequence + 1] - CuSeqLens[sequence],
                        _CHUNK_SIZE,
                    )
                    if (
                        chunk_id >= chunk_prefix[0]
                        and chunk_id < chunk_prefix[0] + sequence_chunks
                    ):
                        sequence_id[0] = sequence
                        chunk_in_sequence[0] = chunk_id - chunk_prefix[0]
                    chunk_prefix[0] += sequence_chunks

            if sequence_id[0] >= 0:
                if num_sequences == 1:
                    chunk_start = chunk_in_sequence[0] * _CHUNK_SIZE
                    valid_tokens = T.min(
                        _CHUNK_SIZE, total_tokens - chunk_start
                    )
                else:
                    chunk_start = (
                        CuSeqLens[sequence_id[0]]
                        + chunk_in_sequence[0] * _CHUNK_SIZE
                    )
                    valid_tokens = T.min(
                        _CHUNK_SIZE,
                        CuSeqLens[sequence_id[0] + 1] - chunk_start,
                    )
                subchunk_offset = subchunk * _SUBCHUNK_SIZE
                valid_subchunk = T.min(
                    _SUBCHUNK_SIZE,
                    valid_tokens - subchunk_offset,
                )

                q_shared = T.alloc_shared(
                    (_SUBCHUNK_SIZE, _HEAD_DIM), T.bfloat16
                )
                k_shared = T.alloc_shared(
                    (_SUBCHUNK_SIZE, _HEAD_DIM), T.bfloat16
                )
                g_shared = T.alloc_shared(
                    (_SUBCHUNK_SIZE, _HEAD_DIM), T.float32
                )
                q_norm = T.alloc_shared((_SUBCHUNK_SIZE,), T.float32)
                k_norm = T.alloc_shared((_SUBCHUNK_SIZE,), T.float32)
                beta = T.alloc_shared((_SUBCHUNK_SIZE,), T.bfloat16)
                left_qk = T.alloc_shared(
                    (2 * _SUBCHUNK_SIZE, 32), T.float32
                )
                right_k = T.alloc_shared((_SUBCHUNK_SIZE, 32), T.float32)
                combined_fragment = T.alloc_fragment(
                    (2 * _SUBCHUNK_SIZE, _SUBCHUNK_SIZE), T.float32
                )
                combined_shared = T.alloc_shared(
                    (2 * _SUBCHUNK_SIZE, _SUBCHUNK_SIZE), T.float32
                )
                inverse_shared = T.alloc_shared(
                    (_SUBCHUNK_SIZE, _SUBCHUNK_SIZE), T.float32
                )
                coefficient_row = T.alloc_shared(
                    (_SUBCHUNK_SIZE,), T.float32
                )
                inverse_accumulator = T.alloc_local((1,), T.float32)
                inverse_coefficient = T.alloc_local((1,), T.float32)
                q_norm_partial = T.alloc_local((1,), T.float32)
                k_norm_partial = T.alloc_local((1,), T.float32)
                gate_prefix = T.alloc_local((1,), T.float32)
                a_scale = T.alloc_local((1,), T.float32)

                a_scale[0] = T.exp(ALog[head])
                for row, dim in T.Parallel(_SUBCHUNK_SIZE, _HEAD_DIM):
                    q_shared[row, dim] = T.if_then_else(
                        row < valid_subchunk,
                        Q[
                            chunk_start + subchunk_offset + row,
                            head,
                            dim,
                        ],
                        T.cast(0.0, T.bfloat16),
                    )
                    k_shared[row, dim] = T.if_then_else(
                        row < valid_subchunk,
                        K[
                            chunk_start + subchunk_offset + row,
                            head,
                            dim,
                        ],
                        T.cast(0.0, T.bfloat16),
                    )
                    g_shared[row, dim] = T.if_then_else(
                        row < valid_subchunk,
                        _LOG2_GATE_SCALE
                        * T.sigmoid(
                            a_scale[0]
                            * (
                                GRaw[
                                    chunk_start + subchunk_offset + row,
                                    head,
                                    dim,
                                ]
                                + DtBias[head, dim]
                            )
                        ),
                        0.0,
                    )
                for row in T.Parallel(_SUBCHUNK_SIZE):
                    beta[row] = T.if_then_else(
                        row < valid_subchunk,
                        T.sigmoid(
                            BetaRaw[
                                chunk_start + subchunk_offset + row,
                                head,
                            ]
                        ),
                        T.cast(0.0, T.bfloat16),
                    )
                    if row < valid_subchunk:
                        Beta[
                            chunk_start + subchunk_offset + row,
                            head,
                        ] = beta[row]
                T.sync_threads()

                for row_group in T.serial(
                    _SUBCHUNK_SIZE // (_DIAGONAL_THREADS // _WARP_SIZE)
                ):
                    row = (
                        row_group * (_DIAGONAL_THREADS // _WARP_SIZE) + warp
                    )
                    q_norm_partial[0] = 0.0
                    k_norm_partial[0] = 0.0
                    for dim_slot in T.unroll(_HEAD_DIM // _WARP_SIZE):
                        dim = lane * (_HEAD_DIM // _WARP_SIZE) + dim_slot
                        q_norm_partial[0] += (
                            T.cast(q_shared[row, dim], T.float32)
                            * T.cast(q_shared[row, dim], T.float32)
                        )
                        k_norm_partial[0] += (
                            T.cast(k_shared[row, dim], T.float32)
                            * T.cast(k_shared[row, dim], T.float32)
                        )
                    q_norm_partial[0] = T.warp_reduce_sum(q_norm_partial[0])
                    k_norm_partial[0] = T.warp_reduce_sum(k_norm_partial[0])
                    if lane == 0:
                        q_norm[row] = T.rsqrt(q_norm_partial[0] + 1.0e-6)
                        k_norm[row] = T.rsqrt(k_norm_partial[0] + 1.0e-6)
                        if row < valid_subchunk:
                            Norm[
                                chunk_start + subchunk_offset + row,
                                head,
                                0,
                            ] = q_norm[row]
                            Norm[
                                chunk_start + subchunk_offset + row,
                                head,
                                1,
                            ] = k_norm[row]
                T.sync_threads()

                for dim in T.Parallel(_HEAD_DIM):
                    gate_prefix[0] = g_shared[0, dim]
                    for row in T.serial(1, _SUBCHUNK_SIZE):
                        gate_prefix[0] += g_shared[row, dim]
                        g_shared[row, dim] = gate_prefix[0]
                T.sync_threads()

                T.clear(combined_fragment)
                for dim_block in T.serial(_HEAD_DIM // _WARP_SIZE):
                    for row, dim in T.Parallel(
                        _SUBCHUNK_SIZE, _WARP_SIZE
                    ):
                        dim_index = dim_block * _WARP_SIZE + dim
                        gate_reference = g_shared[0, dim_index]
                        left_qk[row, dim] = T.if_then_else(
                            row < valid_subchunk,
                            T.cast(q_shared[row, dim_index], T.float32)
                            * q_norm[row]
                            * T.exp2(
                                g_shared[row, dim_index] - gate_reference
                            )
                            * _INV_SQRT_HEAD_DIM,
                            0.0,
                        )
                        left_qk[_SUBCHUNK_SIZE + row, dim] = T.if_then_else(
                            row < valid_subchunk,
                            T.cast(k_shared[row, dim_index], T.float32)
                            * k_norm[row]
                            * T.cast(beta[row], T.float32)
                            * T.exp2(
                                g_shared[row, dim_index] - gate_reference
                            ),
                            0.0,
                        )
                        right_k[row, dim] = T.if_then_else(
                            row < valid_subchunk,
                            T.cast(k_shared[row, dim_index], T.float32)
                            * k_norm[row]
                            * T.exp2(
                                gate_reference - g_shared[row, dim_index]
                            ),
                            0.0,
                        )
                    T.sync_threads()
                    T.gemm(
                        left_qk,
                        right_k,
                        combined_fragment,
                        transpose_B=True,
                    )

                T.copy(combined_fragment, combined_shared)
                T.sync_threads()
                T.clear(inverse_shared)
                for row, column in T.Parallel(
                    _SUBCHUNK_SIZE, _SUBCHUNK_SIZE
                ):
                    if row < valid_subchunk and column < row:
                        inverse_shared[row, column] = combined_shared[
                            _SUBCHUNK_SIZE + row, column
                        ]
                T.sync_threads()

                for row in T.serial(_SUBCHUNK_SIZE):
                    for column in T.Parallel(_SUBCHUNK_SIZE):
                        coefficient_row[column] = T.if_then_else(
                            column < row,
                            inverse_shared[row, column],
                            0.0,
                        )
                    T.sync_threads()
                    if thread < _SUBCHUNK_SIZE:
                        inverse_accumulator[0] = T.if_then_else(
                            thread < row,
                            -coefficient_row[thread],
                            0.0,
                        )
                        for inner in T.serial(_SUBCHUNK_SIZE):
                            if inner < row and thread < inner:
                                inverse_coefficient[0] = coefficient_row[
                                    inner
                                ]
                                inverse_accumulator[0] -= (
                                    inverse_coefficient[0]
                                    * inverse_shared[inner, thread]
                                )
                        inverse_shared[row, thread] = T.if_then_else(
                            thread < row,
                            inverse_accumulator[0],
                            T.if_then_else(thread == row, 1.0, 0.0),
                        )
                    T.sync_threads()

                for row, column in T.Parallel(
                    _SUBCHUNK_SIZE, _CHUNK_SIZE
                ):
                    if row < valid_subchunk:
                        Aqk[
                            chunk_start + subchunk_offset + row,
                            head,
                            column,
                        ] = T.cast(0.0, T.bfloat16)
                for row, column in T.Parallel(
                    _SUBCHUNK_SIZE, _SUBCHUNK_SIZE
                ):
                    if row < valid_subchunk:
                        AInvDiagonal[
                            chunk_start + subchunk_offset + row,
                            head,
                            column,
                        ] = inverse_shared[row, column]
                        if column <= row:
                            Aqk[
                                chunk_start + subchunk_offset + row,
                                head,
                                subchunk_offset + column,
                            ] = combined_shared[row, column]

    return kernel


@tilelang.jit(
    out_idx=[],
    target=_TARGET,
    pass_configs={tilelang.PassConfigKey.TL_ENABLE_FAST_MATH: True},
)
def _compile_chunk_inter(total_tokens: int, num_sequences: int, num_heads: int):
    max_chunks = (total_tokens + _CHUNK_SIZE - 1) // _CHUNK_SIZE + num_sequences - 1

    token_shape = (total_tokens, num_heads, _HEAD_DIM)
    beta_shape = (total_tokens, num_heads)
    a_log_shape = (num_heads,)
    dt_bias_shape = (num_heads, _HEAD_DIM)
    cu_shape = (num_sequences + 1,)
    diagonal_shape = (total_tokens, num_heads, _CHUNK_SIZE // 2)
    operator_shape = (total_tokens, num_heads, _CHUNK_SIZE)
    norm_shape = (total_tokens, num_heads, 2)

    @T.prim_func
    def kernel(
        Q: T.Tensor(token_shape, T.bfloat16),
        K: T.Tensor(token_shape, T.bfloat16),
        GRaw: T.Tensor(token_shape, T.bfloat16),
        ALog: T.Tensor(a_log_shape, T.float32),
        DtBias: T.Tensor(dt_bias_shape, T.float32),
        CuSeqLens: T.Tensor(cu_shape, T.int32),
        AInvDiagonal: T.Tensor(diagonal_shape, T.float32),
        AInv: T.Tensor(operator_shape, T.bfloat16),
        Aqk: T.Tensor(operator_shape, T.bfloat16),
        Norm: T.Tensor(norm_shape, T.float32),
        Beta: T.Tensor(beta_shape, T.bfloat16),
    ):
        with T.Kernel(
            max_chunks, num_heads, threads=_INTER_THREADS
        ) as (chunk_id, head):
            sequence_id = T.alloc_local((1,), T.int32)
            chunk_in_sequence = T.alloc_local((1,), T.int32)
            chunk_prefix = T.alloc_local((1,), T.int32)

            sequence_id[0] = -1
            chunk_in_sequence[0] = 0
            chunk_prefix[0] = 0
            if num_sequences == 1:
                sequence_id[0] = 0
                chunk_in_sequence[0] = chunk_id
            else:
                for sequence in T.serial(num_sequences):
                    sequence_chunks = T.ceildiv(
                        CuSeqLens[sequence + 1] - CuSeqLens[sequence],
                        _CHUNK_SIZE,
                    )
                    if (
                        chunk_id >= chunk_prefix[0]
                        and chunk_id < chunk_prefix[0] + sequence_chunks
                    ):
                        sequence_id[0] = sequence
                        chunk_in_sequence[0] = chunk_id - chunk_prefix[0]
                    chunk_prefix[0] += sequence_chunks

            if sequence_id[0] >= 0:
                if num_sequences == 1:
                    chunk_start = chunk_in_sequence[0] * _CHUNK_SIZE
                    valid_tokens = T.min(
                        _CHUNK_SIZE, total_tokens - chunk_start
                    )
                else:
                    chunk_start = (
                        CuSeqLens[sequence_id[0]]
                        + chunk_in_sequence[0] * _CHUNK_SIZE
                    )
                    valid_tokens = T.min(
                        _CHUNK_SIZE,
                        CuSeqLens[sequence_id[0] + 1] - chunk_start,
                    )

                q_shared = T.alloc_shared(
                    (_CHUNK_SIZE, _HEAD_DIM), T.bfloat16
                )
                k_shared = T.alloc_shared(
                    (_CHUNK_SIZE, _HEAD_DIM), T.bfloat16
                )
                g_shared = T.alloc_shared(
                    (_CHUNK_SIZE, _HEAD_DIM), T.float32
                )
                q_norm = T.alloc_shared((_CHUNK_SIZE,), T.float32)
                k_norm = T.alloc_shared((_CHUNK_SIZE,), T.float32)
                beta = T.alloc_shared((_CHUNK_SIZE,), T.bfloat16)
                operator_shared = T.alloc_shared(
                    (_CHUNK_SIZE, _CHUNK_SIZE), T.float32
                )
                left_qk = T.alloc_shared(
                    (2 * _SUBCHUNK_SIZE, 32), T.float32
                )
                right_k = T.alloc_shared((_SUBCHUNK_SIZE, 32), T.float32)
                matrix_a = T.alloc_shared(
                    (_SUBCHUNK_SIZE, _SUBCHUNK_SIZE), T.float32
                )
                matrix_b = T.alloc_shared(
                    (_SUBCHUNK_SIZE, _SUBCHUNK_SIZE), T.float32
                )
                combined_fragment = T.alloc_fragment(
                    (2 * _SUBCHUNK_SIZE, _SUBCHUNK_SIZE), T.float32
                )
                combined_shared = T.alloc_shared(
                    (2 * _SUBCHUNK_SIZE, _SUBCHUNK_SIZE), T.float32
                )
                product_fragment = T.alloc_fragment(
                    (_SUBCHUNK_SIZE, _SUBCHUNK_SIZE), T.float32
                )
                inverse_fragment = T.alloc_fragment(
                    (_SUBCHUNK_SIZE, _SUBCHUNK_SIZE), T.float32
                )
                gate_prefix = T.alloc_local((1,), T.float32)
                a_scale = T.alloc_local((1,), T.float32)

                a_scale[0] = T.exp(ALog[head])
                for row, dim in T.Parallel(_CHUNK_SIZE, _HEAD_DIM):
                    q_shared[row, dim] = T.if_then_else(
                        row < valid_tokens,
                        Q[chunk_start + row, head, dim],
                        T.cast(0.0, T.bfloat16),
                    )
                    k_shared[row, dim] = T.if_then_else(
                        row < valid_tokens,
                        K[chunk_start + row, head, dim],
                        T.cast(0.0, T.bfloat16),
                    )
                    g_shared[row, dim] = T.if_then_else(
                        row < valid_tokens,
                        _LOG2_GATE_SCALE
                        * T.sigmoid(
                            a_scale[0]
                            * (
                                GRaw[chunk_start + row, head, dim]
                                + DtBias[head, dim]
                            )
                        ),
                        0.0,
                    )
                for row in T.Parallel(_CHUNK_SIZE):
                    q_norm[row] = T.if_then_else(
                        row < valid_tokens,
                        Norm[chunk_start + row, head, 0],
                        0.0,
                    )
                    k_norm[row] = T.if_then_else(
                        row < valid_tokens,
                        Norm[chunk_start + row, head, 1],
                        0.0,
                    )
                    beta[row] = T.if_then_else(
                        row < valid_tokens,
                        Beta[chunk_start + row, head],
                        T.cast(0.0, T.bfloat16),
                    )
                T.sync_threads()

                for dim in T.Parallel(_HEAD_DIM):
                    gate_prefix[0] = g_shared[0, dim]
                    for row in T.serial(1, _CHUNK_SIZE):
                        gate_prefix[0] += g_shared[row, dim]
                        g_shared[row, dim] = gate_prefix[0]
                T.clear(operator_shared)
                T.sync_threads()

                for row, column in T.Parallel(
                    _CHUNK_SIZE, _SUBCHUNK_SIZE
                ):
                    if row < valid_tokens:
                        operator_shared[
                            row,
                            (row // _SUBCHUNK_SIZE) * _SUBCHUNK_SIZE + column,
                        ] = AInvDiagonal[chunk_start + row, head, column]
                T.sync_threads()

                for row_block in T.serial(1, _CHUNK_SIZE // _SUBCHUNK_SIZE):
                    for column_block in T.serial(row_block):
                        T.clear(combined_fragment)
                        for dim_block in T.serial(_HEAD_DIM // _WARP_SIZE):
                            for row, dim in T.Parallel(
                                _SUBCHUNK_SIZE, _WARP_SIZE
                            ):
                                row_index = row_block * _SUBCHUNK_SIZE + row
                                column_index = (
                                    column_block * _SUBCHUNK_SIZE + row
                                )
                                dim_index = dim_block * _WARP_SIZE + dim
                                gate_reference = g_shared[
                                    row_block * _SUBCHUNK_SIZE, dim_index
                                ]
                                left_qk[row, dim] = T.if_then_else(
                                    row_index < valid_tokens,
                                    T.cast(
                                        q_shared[row_index, dim_index],
                                        T.float32,
                                    )
                                    * q_norm[row_index]
                                    * T.exp2(
                                        g_shared[row_index, dim_index]
                                        - gate_reference
                                    )
                                    * _INV_SQRT_HEAD_DIM,
                                    0.0,
                                )
                                left_qk[
                                    _SUBCHUNK_SIZE + row, dim
                                ] = T.if_then_else(
                                    row_index < valid_tokens,
                                    T.cast(
                                        k_shared[row_index, dim_index],
                                        T.float32,
                                    )
                                    * k_norm[row_index]
                                    * T.cast(beta[row_index], T.float32)
                                    * T.exp2(
                                        g_shared[row_index, dim_index]
                                        - gate_reference
                                    ),
                                    0.0,
                                )
                                right_k[row, dim] = T.if_then_else(
                                    column_index < valid_tokens,
                                    T.cast(
                                        k_shared[column_index, dim_index],
                                        T.float32,
                                    )
                                    * k_norm[column_index]
                                    * T.exp2(
                                        gate_reference
                                        - g_shared[column_index, dim_index]
                                    ),
                                    0.0,
                                )
                            T.sync_threads()
                            T.gemm(
                                left_qk,
                                right_k,
                                combined_fragment,
                                transpose_B=True,
                            )

                        T.copy(combined_fragment, combined_shared)
                        T.sync_threads()
                        for row, column in T.Parallel(
                            _SUBCHUNK_SIZE, _SUBCHUNK_SIZE
                        ):
                            row_index = row_block * _SUBCHUNK_SIZE + row
                            column_index = (
                                column_block * _SUBCHUNK_SIZE + column
                            )
                            if row_index < valid_tokens:
                                Aqk[
                                    chunk_start + row_index,
                                    head,
                                    column_index,
                                ] = combined_shared[row, column]
                                operator_shared[
                                    row_index, column_index
                                ] = combined_shared[
                                    _SUBCHUNK_SIZE + row, column
                                ]
                        T.sync_threads()

                for row, column in T.Parallel(
                    _SUBCHUNK_SIZE, _SUBCHUNK_SIZE
                ):
                    matrix_a[row, column] = operator_shared[
                        _SUBCHUNK_SIZE + row, _SUBCHUNK_SIZE + column
                    ]
                    matrix_b[row, column] = operator_shared[
                        _SUBCHUNK_SIZE + row, column
                    ]
                T.sync_threads()
                T.gemm(matrix_a, matrix_b, product_fragment, clear_accum=True)
                T.copy(product_fragment, matrix_a)
                for row, column in T.Parallel(
                    _SUBCHUNK_SIZE, _SUBCHUNK_SIZE
                ):
                    matrix_b[row, column] = operator_shared[row, column]
                T.sync_threads()
                T.gemm(matrix_a, matrix_b, inverse_fragment, clear_accum=True)
                for row, column in T.Parallel(
                    _SUBCHUNK_SIZE, _SUBCHUNK_SIZE
                ):
                    operator_shared[
                        _SUBCHUNK_SIZE + row, column
                    ] = -inverse_fragment[row, column]
                T.sync_threads()

                for row, column in T.Parallel(
                    _SUBCHUNK_SIZE, _SUBCHUNK_SIZE
                ):
                    matrix_a[row, column] = operator_shared[
                        2 * _SUBCHUNK_SIZE + row, column
                    ]
                    matrix_b[row, column] = operator_shared[row, column]
                T.sync_threads()
                T.gemm(matrix_a, matrix_b, product_fragment, clear_accum=True)
                for row, column in T.Parallel(
                    _SUBCHUNK_SIZE, _SUBCHUNK_SIZE
                ):
                    matrix_a[row, column] = operator_shared[
                        2 * _SUBCHUNK_SIZE + row,
                        _SUBCHUNK_SIZE + column,
                    ]
                    matrix_b[row, column] = operator_shared[
                        _SUBCHUNK_SIZE + row, column
                    ]
                T.sync_threads()
                T.gemm(matrix_a, matrix_b, product_fragment)
                T.copy(product_fragment, matrix_b)
                for row, column in T.Parallel(
                    _SUBCHUNK_SIZE, _SUBCHUNK_SIZE
                ):
                    matrix_a[row, column] = operator_shared[
                        2 * _SUBCHUNK_SIZE + row,
                        2 * _SUBCHUNK_SIZE + column,
                    ]
                T.sync_threads()
                T.gemm(matrix_a, matrix_b, inverse_fragment, clear_accum=True)
                for row, column in T.Parallel(
                    _SUBCHUNK_SIZE, _SUBCHUNK_SIZE
                ):
                    operator_shared[
                        2 * _SUBCHUNK_SIZE + row, column
                    ] = -inverse_fragment[row, column]
                T.sync_threads()

                for row, column in T.Parallel(
                    _SUBCHUNK_SIZE, _SUBCHUNK_SIZE
                ):
                    matrix_a[row, column] = operator_shared[
                        2 * _SUBCHUNK_SIZE + row,
                        2 * _SUBCHUNK_SIZE + column,
                    ]
                    matrix_b[row, column] = operator_shared[
                        2 * _SUBCHUNK_SIZE + row,
                        _SUBCHUNK_SIZE + column,
                    ]
                T.sync_threads()
                T.gemm(matrix_a, matrix_b, product_fragment, clear_accum=True)
                T.copy(product_fragment, matrix_a)
                for row, column in T.Parallel(
                    _SUBCHUNK_SIZE, _SUBCHUNK_SIZE
                ):
                    matrix_b[row, column] = operator_shared[
                        _SUBCHUNK_SIZE + row,
                        _SUBCHUNK_SIZE + column,
                    ]
                T.sync_threads()
                T.gemm(matrix_a, matrix_b, inverse_fragment, clear_accum=True)
                for row, column in T.Parallel(
                    _SUBCHUNK_SIZE, _SUBCHUNK_SIZE
                ):
                    operator_shared[
                        2 * _SUBCHUNK_SIZE + row,
                        _SUBCHUNK_SIZE + column,
                    ] = -inverse_fragment[row, column]
                T.sync_threads()

                for row, column in T.Parallel(
                    _SUBCHUNK_SIZE, _SUBCHUNK_SIZE
                ):
                    matrix_a[row, column] = operator_shared[
                        3 * _SUBCHUNK_SIZE + row, column
                    ]
                    matrix_b[row, column] = operator_shared[row, column]
                T.sync_threads()
                T.gemm(matrix_a, matrix_b, product_fragment, clear_accum=True)
                for row, column in T.Parallel(
                    _SUBCHUNK_SIZE, _SUBCHUNK_SIZE
                ):
                    matrix_a[row, column] = operator_shared[
                        3 * _SUBCHUNK_SIZE + row,
                        _SUBCHUNK_SIZE + column,
                    ]
                    matrix_b[row, column] = operator_shared[
                        _SUBCHUNK_SIZE + row, column
                    ]
                T.sync_threads()
                T.gemm(matrix_a, matrix_b, product_fragment)
                for row, column in T.Parallel(
                    _SUBCHUNK_SIZE, _SUBCHUNK_SIZE
                ):
                    matrix_a[row, column] = operator_shared[
                        3 * _SUBCHUNK_SIZE + row,
                        2 * _SUBCHUNK_SIZE + column,
                    ]
                    matrix_b[row, column] = operator_shared[
                        2 * _SUBCHUNK_SIZE + row, column
                    ]
                T.sync_threads()
                T.gemm(matrix_a, matrix_b, product_fragment)
                T.copy(product_fragment, matrix_b)
                for row, column in T.Parallel(
                    _SUBCHUNK_SIZE, _SUBCHUNK_SIZE
                ):
                    matrix_a[row, column] = operator_shared[
                        3 * _SUBCHUNK_SIZE + row,
                        3 * _SUBCHUNK_SIZE + column,
                    ]
                T.sync_threads()
                T.gemm(matrix_a, matrix_b, inverse_fragment, clear_accum=True)
                for row, column in T.Parallel(
                    _SUBCHUNK_SIZE, _SUBCHUNK_SIZE
                ):
                    operator_shared[
                        3 * _SUBCHUNK_SIZE + row, column
                    ] = -inverse_fragment[row, column]
                T.sync_threads()

                for row, column in T.Parallel(
                    _SUBCHUNK_SIZE, _SUBCHUNK_SIZE
                ):
                    matrix_a[row, column] = operator_shared[
                        3 * _SUBCHUNK_SIZE + row,
                        _SUBCHUNK_SIZE + column,
                    ]
                    matrix_b[row, column] = operator_shared[
                        _SUBCHUNK_SIZE + row,
                        _SUBCHUNK_SIZE + column,
                    ]
                T.sync_threads()
                T.gemm(matrix_a, matrix_b, product_fragment, clear_accum=True)
                for row, column in T.Parallel(
                    _SUBCHUNK_SIZE, _SUBCHUNK_SIZE
                ):
                    matrix_a[row, column] = operator_shared[
                        3 * _SUBCHUNK_SIZE + row,
                        2 * _SUBCHUNK_SIZE + column,
                    ]
                    matrix_b[row, column] = operator_shared[
                        2 * _SUBCHUNK_SIZE + row,
                        _SUBCHUNK_SIZE + column,
                    ]
                T.sync_threads()
                T.gemm(matrix_a, matrix_b, product_fragment)
                T.copy(product_fragment, matrix_b)
                for row, column in T.Parallel(
                    _SUBCHUNK_SIZE, _SUBCHUNK_SIZE
                ):
                    matrix_a[row, column] = operator_shared[
                        3 * _SUBCHUNK_SIZE + row,
                        3 * _SUBCHUNK_SIZE + column,
                    ]
                T.sync_threads()
                T.gemm(matrix_a, matrix_b, inverse_fragment, clear_accum=True)
                for row, column in T.Parallel(
                    _SUBCHUNK_SIZE, _SUBCHUNK_SIZE
                ):
                    operator_shared[
                        3 * _SUBCHUNK_SIZE + row,
                        _SUBCHUNK_SIZE + column,
                    ] = -inverse_fragment[row, column]
                T.sync_threads()

                for row, column in T.Parallel(
                    _SUBCHUNK_SIZE, _SUBCHUNK_SIZE
                ):
                    matrix_a[row, column] = operator_shared[
                        3 * _SUBCHUNK_SIZE + row,
                        3 * _SUBCHUNK_SIZE + column,
                    ]
                    matrix_b[row, column] = operator_shared[
                        3 * _SUBCHUNK_SIZE + row,
                        2 * _SUBCHUNK_SIZE + column,
                    ]
                T.sync_threads()
                T.gemm(matrix_a, matrix_b, product_fragment, clear_accum=True)
                T.copy(product_fragment, matrix_a)
                for row, column in T.Parallel(
                    _SUBCHUNK_SIZE, _SUBCHUNK_SIZE
                ):
                    matrix_b[row, column] = operator_shared[
                        2 * _SUBCHUNK_SIZE + row,
                        2 * _SUBCHUNK_SIZE + column,
                    ]
                T.sync_threads()
                T.gemm(matrix_a, matrix_b, inverse_fragment, clear_accum=True)
                for row, column in T.Parallel(
                    _SUBCHUNK_SIZE, _SUBCHUNK_SIZE
                ):
                    operator_shared[
                        3 * _SUBCHUNK_SIZE + row,
                        2 * _SUBCHUNK_SIZE + column,
                    ] = -inverse_fragment[row, column]
                T.sync_threads()

                for row, column in T.Parallel(
                    _CHUNK_SIZE, _CHUNK_SIZE
                ):
                    if row < valid_tokens:
                        AInv[chunk_start + row, head, column] = (
                            operator_shared[row, column] * beta[column]
                        )

    return kernel


@tilelang.jit(
    out_idx=[],
    target=_TARGET,
    pass_configs={tilelang.PassConfigKey.TL_ENABLE_FAST_MATH: True},
)
def _compile_state_scan(
    total_tokens: int,
    num_sequences: int,
    num_heads: int,
    value_tile: int,
    segment_chunks: int,
):
    token_shape = (total_tokens, num_heads, _HEAD_DIM)
    beta_shape = (total_tokens, num_heads)
    a_log_shape = (num_heads,)
    dt_bias_shape = (num_heads, _HEAD_DIM)
    state_shape = (
        num_sequences,
        num_heads,
        _HEAD_DIM,
        _HEAD_DIM,
    )
    cu_shape = (num_sequences + 1,)
    operator_shape = (total_tokens, num_heads, _CHUNK_SIZE)
    norm_shape = (total_tokens, num_heads, 2)
    scratch_shape = (
        segment_chunks,
        num_heads,
        _HEAD_DIM,
        _HEAD_DIM,
    )

    @T.prim_func
    def kernel(
        K: T.Tensor(token_shape, T.bfloat16),
        V: T.Tensor(token_shape, T.bfloat16),
        GRaw: T.Tensor(token_shape, T.bfloat16),
        ALog: T.Tensor(a_log_shape, T.float32),
        DtBias: T.Tensor(dt_bias_shape, T.float32),
        InitialState: T.Tensor(state_shape, T.bfloat16),
        CuSeqLens: T.Tensor(cu_shape, T.int32),
        AInv: T.Tensor(operator_shape, T.bfloat16),
        Norm: T.Tensor(norm_shape, T.float32),
        Beta: T.Tensor(beta_shape, T.bfloat16),
        StateScratch: T.Tensor(scratch_shape, T.bfloat16),
        VNew: T.Tensor(token_shape, T.bfloat16),
        FinalState: T.Tensor(state_shape, T.bfloat16),
        SegmentStart: T.int32,
    ):
        with T.Kernel(
            T.ceildiv(_HEAD_DIM, value_tile),
            num_heads,
            num_sequences,
            threads=_THREADS,
        ) as (value_block, head, sequence):
            thread = T.get_thread_binding(0)
            k_shared = T.alloc_shared(
                (_CHUNK_SIZE, _HEAD_DIM), T.bfloat16
            )
            x_shared = T.alloc_shared(
                (_CHUNK_SIZE, _HEAD_DIM), T.bfloat16
            )
            g_shared = T.alloc_shared(
                (_CHUNK_SIZE, _HEAD_DIM), T.float32
            )
            operator_shared = T.alloc_shared(
                (_CHUNK_SIZE, _CHUNK_SIZE), T.bfloat16
            )
            state_shared = T.alloc_shared(
                (_HEAD_DIM, value_tile), T.bfloat16
            )
            rhs_shared = T.alloc_shared(
                (_CHUNK_SIZE, value_tile), T.bfloat16
            )
            norm = T.alloc_shared((_CHUNK_SIZE,), T.float32)
            beta = T.alloc_shared((_CHUNK_SIZE,), T.bfloat16)

            state_fragment = T.alloc_fragment(
                (_HEAD_DIM, value_tile), T.float32
            )
            rhs_fragment = T.alloc_fragment(
                (_CHUNK_SIZE, value_tile), T.float32
            )
            a_scale = T.alloc_local((1,), T.float32)
            dt_bias = T.alloc_local((4,), T.float32)
            gate_prefix = T.alloc_local((1,), T.float32)
            normalized_k = T.alloc_local((4,), T.bfloat16)
            chunk_prefix = T.alloc_local((1,), T.int32)
            first_chunk = T.alloc_local((1,), T.int32)
            last_chunk = T.alloc_local((1,), T.int32)

            sequence_start = CuSeqLens[sequence]
            sequence_length = CuSeqLens[sequence + 1] - sequence_start
            sequence_chunks = T.ceildiv(sequence_length, _CHUNK_SIZE)
            chunk_prefix[0] = 0
            for prior_sequence in T.serial(num_sequences):
                if prior_sequence < sequence:
                    chunk_prefix[0] += T.ceildiv(
                        CuSeqLens[prior_sequence + 1]
                        - CuSeqLens[prior_sequence],
                        _CHUNK_SIZE,
                    )
            first_chunk[0] = T.max(
                0, SegmentStart - chunk_prefix[0]
            )
            last_chunk[0] = T.min(
                sequence_chunks,
                SegmentStart + segment_chunks - chunk_prefix[0],
            )
            a_scale[0] = T.exp(ALog[head])
            T.copy(
                DtBias[
                    head,
                    (thread % _WARP_SIZE)
                    * 4 : (thread % _WARP_SIZE) * 4
                    + 4,
                ],
                dt_bias,
            )
            if first_chunk[0] < last_chunk[0]:
                if first_chunk[0] == 0:
                    T.copy(
                        InitialState[
                            sequence,
                            head,
                            0:_HEAD_DIM,
                            value_block
                            * value_tile : (value_block + 1)
                            * value_tile,
                        ],
                        state_fragment,
                    )
                else:
                    T.copy(
                        FinalState[
                            sequence,
                            head,
                            0:_HEAD_DIM,
                            value_block
                            * value_tile : (value_block + 1)
                            * value_tile,
                        ],
                        state_fragment,
                    )
                T.async_copy(
                    GRaw[
                        sequence_start
                        + first_chunk[0]
                        * _CHUNK_SIZE : sequence_start
                        + (first_chunk[0] + 1)
                        * _CHUNK_SIZE,
                        head,
                        0:_HEAD_DIM,
                    ],
                    x_shared,
                )

                for chunk in T.serial(first_chunk[0], last_chunk[0]):
                    chunk_start = sequence_start + chunk * _CHUNK_SIZE
                    valid_tokens = T.min(
                        _CHUNK_SIZE,
                        sequence_length - chunk * _CHUNK_SIZE,
                    )

                    T.async_copy(
                        K[
                            chunk_start : chunk_start + _CHUNK_SIZE,
                            head,
                            0:_HEAD_DIM,
                        ],
                        k_shared,
                    )
                    T.ptx_wait_group(1)
                    for row, dim in T.Parallel(
                        _CHUNK_SIZE, _HEAD_DIM
                    ):
                        g_shared[row, dim] = T.if_then_else(
                            row < valid_tokens,
                            _LOG2_GATE_SCALE
                            * T.sigmoid(
                                a_scale[0]
                                * (
                                    x_shared[row, dim]
                                    + dt_bias[dim % 4]
                                )
                            ),
                            0.0,
                        )
                    for row in T.Parallel(_CHUNK_SIZE):
                        beta[row] = T.if_then_else(
                            row < valid_tokens,
                            Beta[chunk_start + row, head],
                            T.cast(0.0, T.bfloat16),
                        )
                        norm[row] = T.if_then_else(
                            row < valid_tokens,
                            Norm[chunk_start + row, head, 1],
                            0.0,
                        )
                    T.ptx_wait_group(0)
                    T.sync_threads()

                    for dim in T.Parallel(_HEAD_DIM):
                        gate_prefix[0] = g_shared[0, dim]
                        for row in T.serial(1, _CHUNK_SIZE):
                            gate_prefix[0] += g_shared[row, dim]
                            g_shared[row, dim] = gate_prefix[0]
                    T.sync_threads()

                    for row_group in T.unroll(_CHUNK_SIZE // 4):
                        for lane in T.vectorized(4):
                            row = row_group * 4 + thread // 32
                            dim = (thread % 32) * 4 + lane
                            normalized_k[lane] = (
                                k_shared[row, dim] * norm[row]
                            )
                            k_shared[row, dim] = normalized_k[lane]
                            x_shared[row, dim] = normalized_k[
                                lane
                            ] * T.exp2(g_shared[row, dim])
                    for row, column in T.Parallel(
                        _CHUNK_SIZE, _CHUNK_SIZE
                    ):
                        operator_shared[row, column] = T.if_then_else(
                            row < valid_tokens,
                            AInv[chunk_start + row, head, column],
                            T.cast(0.0, T.bfloat16),
                        )
                    T.copy(state_fragment, state_shared)
                    T.copy(
                        state_shared,
                        StateScratch[
                            chunk_prefix[0] + chunk - SegmentStart,
                            head,
                            0:_HEAD_DIM,
                            value_block
                            * value_tile : (value_block + 1)
                            * value_tile,
                        ],
                    )
                    T.sync_threads()
                    T.gemm(
                        x_shared,
                        state_shared,
                        rhs_fragment,
                        clear_accum=True,
                    )
                    for row, value in T.Parallel(
                        _CHUNK_SIZE, value_tile
                    ):
                        rhs_fragment[row, value] = T.if_then_else(
                            row < valid_tokens,
                            beta[row]
                            * (
                                V[
                                    chunk_start + row,
                                    head,
                                    value_block * value_tile + value,
                                ]
                                - rhs_fragment[row, value]
                            ),
                            0.0,
                        )
                    T.copy(rhs_fragment, rhs_shared)
                    T.gemm(
                        operator_shared,
                        rhs_shared,
                        rhs_fragment,
                        clear_accum=True,
                    )
                    T.copy(rhs_fragment, rhs_shared)

                    for row, dim in T.Parallel(
                        _CHUNK_SIZE, _HEAD_DIM
                    ):
                        x_shared[row, dim] = T.if_then_else(
                            row < valid_tokens,
                            k_shared[row, dim]
                            * T.exp2(
                                g_shared[valid_tokens - 1, dim]
                                - g_shared[row, dim]
                            ),
                            0.0,
                        )
                    for dim, value in T.Parallel(
                        _HEAD_DIM, value_tile
                    ):
                        state_fragment[dim, value] *= T.exp2(
                            g_shared[valid_tokens - 1, dim]
                        )
                    T.sync_threads()
                    T.gemm(
                        x_shared,
                        rhs_shared,
                        state_fragment,
                        transpose_A=True,
                    )
                    if chunk + 1 < last_chunk[0]:
                        T.async_copy(
                            GRaw[
                                chunk_start + _CHUNK_SIZE : chunk_start
                                + 2 * _CHUNK_SIZE,
                                head,
                                0:_HEAD_DIM,
                            ],
                            x_shared,
                        )
                    for row, value in T.Parallel(
                        _CHUNK_SIZE, value_tile
                    ):
                        if row < valid_tokens:
                            VNew[
                                chunk_start + row,
                                head,
                                value_block * value_tile + value,
                            ] = rhs_shared[row, value]
                T.copy(state_fragment, state_shared)
                T.copy(
                    state_shared,
                    FinalState[
                        sequence,
                        head,
                        0:_HEAD_DIM,
                        value_block
                        * value_tile : (value_block + 1)
                        * value_tile,
                    ],
                )

    return kernel


@tilelang.jit(
    out_idx=[],
    target=_TARGET,
    pass_configs={tilelang.PassConfigKey.TL_ENABLE_FAST_MATH: True},
)
def _compile_chunk_output(
    total_tokens: int,
    num_sequences: int,
    num_heads: int,
    segment_chunks: int,
):
    value_tile = 64
    token_shape = (total_tokens, num_heads, _HEAD_DIM)
    a_log_shape = (num_heads,)
    dt_bias_shape = (num_heads, _HEAD_DIM)
    cu_shape = (num_sequences + 1,)
    operator_shape = (total_tokens, num_heads, _CHUNK_SIZE)
    norm_shape = (total_tokens, num_heads, 2)
    scratch_shape = (
        segment_chunks,
        num_heads,
        _HEAD_DIM,
        _HEAD_DIM,
    )

    @T.prim_func
    def kernel(
        Q: T.Tensor(token_shape, T.bfloat16),
        GRaw: T.Tensor(token_shape, T.bfloat16),
        ALog: T.Tensor(a_log_shape, T.float32),
        DtBias: T.Tensor(dt_bias_shape, T.float32),
        CuSeqLens: T.Tensor(cu_shape, T.int32),
        Aqk: T.Tensor(operator_shape, T.bfloat16),
        Norm: T.Tensor(norm_shape, T.float32),
        StateScratch: T.Tensor(scratch_shape, T.bfloat16),
        VNewOut: T.Tensor(token_shape, T.bfloat16),
        SegmentStart: T.int32,
    ):
        with T.Kernel(
            segment_chunks,
            T.ceildiv(_HEAD_DIM, value_tile),
            num_heads,
            threads=_THREADS,
        ) as (segment_chunk, value_block, head):
            thread = T.get_thread_binding(0)
            chunk_id = segment_chunk + SegmentStart
            sequence_id = T.alloc_local((1,), T.int32)
            chunk_in_sequence = T.alloc_local((1,), T.int32)
            chunk_prefix = T.alloc_local((1,), T.int32)

            sequence_id[0] = -1
            chunk_in_sequence[0] = 0
            chunk_prefix[0] = 0
            for sequence in T.serial(num_sequences):
                sequence_chunks = T.ceildiv(
                    CuSeqLens[sequence + 1] - CuSeqLens[sequence],
                    _CHUNK_SIZE,
                )
                if (
                    chunk_id >= chunk_prefix[0]
                    and chunk_id < chunk_prefix[0] + sequence_chunks
                ):
                    sequence_id[0] = sequence
                    chunk_in_sequence[0] = chunk_id - chunk_prefix[0]
                chunk_prefix[0] += sequence_chunks

            if sequence_id[0] >= 0:
                chunk_start = (
                    CuSeqLens[sequence_id[0]]
                    + chunk_in_sequence[0] * _CHUNK_SIZE
                )
                valid_tokens = T.min(
                    _CHUNK_SIZE,
                    CuSeqLens[sequence_id[0] + 1] - chunk_start,
                )
                q_shared = T.alloc_shared(
                    (_CHUNK_SIZE, _HEAD_DIM), T.bfloat16
                )
                g_shared = T.alloc_shared(
                    (_CHUNK_SIZE, _HEAD_DIM), T.float32
                )
                operator_shared = T.alloc_shared(
                    (_CHUNK_SIZE, _CHUNK_SIZE), T.bfloat16
                )
                state_shared = T.alloc_shared(
                    (_HEAD_DIM, value_tile), T.bfloat16
                )
                rhs_shared = T.alloc_shared(
                    (_CHUNK_SIZE, value_tile), T.bfloat16
                )
                norm = T.alloc_shared((_CHUNK_SIZE,), T.float32)
                out_fragment = T.alloc_fragment(
                    (_CHUNK_SIZE, value_tile), T.float32
                )
                a_scale = T.alloc_local((1,), T.float32)
                dt_bias = T.alloc_local((4,), T.float32)
                gate_prefix = T.alloc_local((1,), T.float32)

                a_scale[0] = T.exp(ALog[head])
                T.copy(
                    DtBias[
                        head,
                        (thread % _WARP_SIZE)
                        * 4 : (thread % _WARP_SIZE) * 4
                        + 4,
                    ],
                    dt_bias,
                )
                T.async_copy(
                    GRaw[
                        chunk_start : chunk_start + _CHUNK_SIZE,
                        head,
                        0:_HEAD_DIM,
                    ],
                    q_shared,
                )
                T.ptx_wait_group(0)
                T.sync_threads()
                for row, dim in T.Parallel(
                    _CHUNK_SIZE, _HEAD_DIM
                ):
                    g_shared[row, dim] = T.if_then_else(
                        row < valid_tokens,
                        _LOG2_GATE_SCALE
                        * T.sigmoid(
                            a_scale[0]
                            * (
                                q_shared[row, dim]
                                + dt_bias[dim % 4]
                            )
                        ),
                        0.0,
                    )
                T.sync_threads()
                for dim in T.Parallel(_HEAD_DIM):
                    gate_prefix[0] = g_shared[0, dim]
                    for row in T.serial(1, _CHUNK_SIZE):
                        gate_prefix[0] += g_shared[row, dim]
                        g_shared[row, dim] = gate_prefix[0]
                T.async_copy(
                    Q[
                        chunk_start : chunk_start + _CHUNK_SIZE,
                        head,
                        0:_HEAD_DIM,
                    ],
                    q_shared,
                )
                for row in T.Parallel(_CHUNK_SIZE):
                    norm[row] = T.if_then_else(
                        row < valid_tokens,
                        Norm[chunk_start + row, head, 0],
                        0.0,
                    )
                for row, column in T.Parallel(
                    _CHUNK_SIZE, _CHUNK_SIZE
                ):
                    operator_shared[row, column] = T.if_then_else(
                        row < valid_tokens,
                        Aqk[chunk_start + row, head, column],
                        T.cast(0.0, T.bfloat16),
                    )
                T.copy(
                    StateScratch[
                        segment_chunk,
                        head,
                        0:_HEAD_DIM,
                        value_block
                        * value_tile : (value_block + 1)
                        * value_tile,
                    ],
                    state_shared,
                )
                for row, value in T.Parallel(
                    _CHUNK_SIZE, value_tile
                ):
                    rhs_shared[row, value] = T.if_then_else(
                        row < valid_tokens,
                        VNewOut[
                            chunk_start + row,
                            head,
                            value_block * value_tile + value,
                        ],
                        T.cast(0.0, T.bfloat16),
                    )
                T.ptx_wait_group(0)
                T.sync_threads()
                for row, dim in T.Parallel(
                    _CHUNK_SIZE, _HEAD_DIM
                ):
                    q_shared[row, dim] = T.if_then_else(
                        row < valid_tokens,
                        q_shared[row, dim]
                        * norm[row]
                        * T.exp2(g_shared[row, dim])
                        * _INV_SQRT_HEAD_DIM,
                        T.cast(0.0, T.bfloat16),
                    )
                T.sync_threads()
                T.gemm(
                    q_shared,
                    state_shared,
                    out_fragment,
                    clear_accum=True,
                )
                T.gemm(
                    operator_shared,
                    rhs_shared,
                    out_fragment,
                )
                T.copy(out_fragment, rhs_shared)
                for row, value in T.Parallel(
                    _CHUNK_SIZE, value_tile
                ):
                    if row < valid_tokens:
                        VNewOut[
                            chunk_start + row,
                            head,
                            value_block * value_tile + value,
                        ] = rhs_shared[row, value]

    return kernel


@tilelang.jit(
    out_idx=[],
    target=_TARGET,
    pass_configs={tilelang.PassConfigKey.TL_ENABLE_FAST_MATH: True},
)
def _compile_chunk_transform(
    total_tokens: int,
    num_sequences: int,
    num_heads: int,
    segment_chunks: int,
    segment_rounds: int,
):
    single_sequence_chunks = (total_tokens + _CHUNK_SIZE - 1) // _CHUNK_SIZE
    token_shape = (total_tokens, num_heads, _HEAD_DIM)
    a_log_shape = (num_heads,)
    dt_bias_shape = (num_heads, _HEAD_DIM)
    cu_shape = (num_sequences + 1,)
    norm_shape = (total_tokens, num_heads, 2)
    transform_shape = (
        segment_chunks,
        num_heads,
        _CHUNK_SIZE,
        _HEAD_DIM,
    )
    decay_shape = (segment_chunks, num_heads, _HEAD_DIM)

    @T.prim_func
    def kernel(
        Q: T.Tensor(token_shape, T.bfloat16),
        K: T.Tensor(token_shape, T.bfloat16),
        GRaw: T.Tensor(token_shape, T.bfloat16),
        ALog: T.Tensor(a_log_shape, T.float32),
        DtBias: T.Tensor(dt_bias_shape, T.float32),
        CuSeqLens: T.Tensor(cu_shape, T.int32),
        Norm: T.Tensor(norm_shape, T.float32),
        QG: T.Tensor(transform_shape, T.bfloat16),
        KForward: T.Tensor(transform_shape, T.bfloat16),
        KUpdate: T.Tensor(transform_shape, T.bfloat16),
        Decay: T.Tensor(decay_shape, T.float32),
        SegmentRound: T.int32,
    ):
        with T.Kernel(
            segment_chunks, num_heads, threads=256
        ) as (segment_chunk, head):
            thread = T.get_thread_binding(0)
            sequence_id = T.alloc_local((1,), T.int32)
            chunk_in_sequence = T.alloc_local((1,), T.int32)
            chunk_prefix = T.alloc_local((1,), T.int32)

            sequence_id[0] = -1
            chunk_in_sequence[0] = 0
            chunk_prefix[0] = 0
            if num_sequences == 1:
                first_chunk = (
                    single_sequence_chunks * SegmentRound
                    // segment_rounds
                )
                last_chunk = (
                    single_sequence_chunks * (SegmentRound + 1)
                    // segment_rounds
                )
                if segment_chunk < last_chunk - first_chunk:
                    sequence_id[0] = 0
                    chunk_in_sequence[0] = first_chunk + segment_chunk
            else:
                for sequence in T.serial(num_sequences):
                    sequence_chunks = T.ceildiv(
                        CuSeqLens[sequence + 1] - CuSeqLens[sequence],
                        _CHUNK_SIZE,
                    )
                    first_chunk = (
                        sequence_chunks * SegmentRound // segment_rounds
                    )
                    last_chunk = (
                        sequence_chunks * (SegmentRound + 1)
                        // segment_rounds
                    )
                    if (
                        segment_chunk >= chunk_prefix[0]
                        and segment_chunk
                        < chunk_prefix[0] + last_chunk - first_chunk
                    ):
                        sequence_id[0] = sequence
                        chunk_in_sequence[0] = (
                            first_chunk + segment_chunk - chunk_prefix[0]
                        )
                    chunk_prefix[0] += last_chunk - first_chunk

            if sequence_id[0] >= 0:
                if num_sequences == 1:
                    chunk_start = chunk_in_sequence[0] * _CHUNK_SIZE
                    if total_tokens % _CHUNK_SIZE == 0:
                        valid_tokens = _CHUNK_SIZE
                    else:
                        valid_tokens = T.min(
                            _CHUNK_SIZE, total_tokens - chunk_start
                        )
                else:
                    chunk_start = (
                        CuSeqLens[sequence_id[0]]
                        + chunk_in_sequence[0] * _CHUNK_SIZE
                    )
                    valid_tokens = T.min(
                        _CHUNK_SIZE,
                        CuSeqLens[sequence_id[0] + 1] - chunk_start,
                    )
                q_shared = T.alloc_shared(
                    (_CHUNK_SIZE, _HEAD_DIM), T.bfloat16
                )
                k_shared = T.alloc_shared(
                    (_CHUNK_SIZE, _HEAD_DIM), T.bfloat16
                )
                g_shared = T.alloc_shared(
                    (_CHUNK_SIZE, _HEAD_DIM), T.float32
                )
                q_norm = T.alloc_shared((_CHUNK_SIZE,), T.float32)
                k_norm = T.alloc_shared((_CHUNK_SIZE,), T.float32)
                a_scale = T.alloc_local((1,), T.float32)
                dt_bias = T.alloc_local((4,), T.float32)
                gate_prefix = T.alloc_local((1,), T.float32)
                normalized_k = T.alloc_local((1,), T.float32)

                a_scale[0] = T.exp(ALog[head])
                T.copy(
                    DtBias[
                        head,
                        (thread % _WARP_SIZE)
                        * 4 : (thread % _WARP_SIZE) * 4
                        + 4,
                    ],
                    dt_bias,
                )
                T.async_copy(
                    GRaw[
                        chunk_start : chunk_start + _CHUNK_SIZE,
                        head,
                        0:_HEAD_DIM,
                    ],
                    q_shared,
                )
                T.ptx_wait_group(0)
                T.sync_threads()
                for row, dim in T.Parallel(
                    _CHUNK_SIZE, _HEAD_DIM
                ):
                    g_shared[row, dim] = T.if_then_else(
                        row < valid_tokens,
                        _LOG2_GATE_SCALE
                        * T.sigmoid(
                            a_scale[0]
                            * (
                                q_shared[row, dim]
                                + dt_bias[dim % 4]
                            )
                        ),
                        0.0,
                    )
                T.sync_threads()
                for dim in T.Parallel(_HEAD_DIM):
                    gate_prefix[0] = g_shared[0, dim]
                    for row in T.serial(1, _CHUNK_SIZE):
                        gate_prefix[0] += g_shared[row, dim]
                        g_shared[row, dim] = gate_prefix[0]
                    Decay[segment_chunk, head, dim] = T.exp2(
                        g_shared[valid_tokens - 1, dim]
                    )
                T.async_copy(
                    Q[
                        chunk_start : chunk_start + _CHUNK_SIZE,
                        head,
                        0:_HEAD_DIM,
                    ],
                    q_shared,
                )
                T.async_copy(
                    K[
                        chunk_start : chunk_start + _CHUNK_SIZE,
                        head,
                        0:_HEAD_DIM,
                    ],
                    k_shared,
                )
                for row in T.Parallel(_CHUNK_SIZE):
                    q_norm[row] = T.if_then_else(
                        row < valid_tokens,
                        Norm[chunk_start + row, head, 0],
                        0.0,
                    )
                    k_norm[row] = T.if_then_else(
                        row < valid_tokens,
                        Norm[chunk_start + row, head, 1],
                        0.0,
                    )
                T.ptx_wait_group(0)
                T.sync_threads()
                for row, dim in T.Parallel(
                    _CHUNK_SIZE, _HEAD_DIM
                ):
                    if row < valid_tokens:
                        normalized_k[0] = k_shared[row, dim] * k_norm[row]
                        QG[segment_chunk, head, row, dim] = (
                            q_shared[row, dim]
                            * q_norm[row]
                            * T.exp2(g_shared[row, dim])
                            * _INV_SQRT_HEAD_DIM
                        )
                        KForward[segment_chunk, head, row, dim] = (
                            normalized_k[0] * T.exp2(g_shared[row, dim])
                        )
                        KUpdate[segment_chunk, head, row, dim] = (
                            normalized_k[0]
                            * T.exp2(
                                g_shared[valid_tokens - 1, dim]
                                - g_shared[row, dim]
                            )
                        )
                    else:
                        QG[segment_chunk, head, row, dim] = T.cast(
                            0.0, T.bfloat16
                        )
                        KForward[segment_chunk, head, row, dim] = T.cast(
                            0.0, T.bfloat16
                        )
                        KUpdate[segment_chunk, head, row, dim] = T.cast(
                            0.0, T.bfloat16
                        )

    return kernel


@tilelang.jit(
    out_idx=[],
    target=_TARGET,
    pass_configs={tilelang.PassConfigKey.TL_ENABLE_FAST_MATH: True},
)
def _compile_transformed_state_scan(
    total_tokens: int,
    num_sequences: int,
    num_heads: int,
    value_tile: int,
    segment_chunks: int,
    segment_rounds: int,
):
    single_sequence_chunks = (total_tokens + _CHUNK_SIZE - 1) // _CHUNK_SIZE
    token_shape = (total_tokens, num_heads, _HEAD_DIM)
    beta_shape = (total_tokens, num_heads)
    state_shape = (
        num_sequences,
        num_heads,
        _HEAD_DIM,
        _HEAD_DIM,
    )
    cu_shape = (num_sequences + 1,)
    operator_shape = (total_tokens, num_heads, _CHUNK_SIZE)
    scratch_shape = (
        segment_chunks,
        num_heads,
        _HEAD_DIM,
        _HEAD_DIM,
    )
    transform_shape = (
        segment_chunks,
        num_heads,
        _CHUNK_SIZE,
        _HEAD_DIM,
    )
    decay_shape = (segment_chunks, num_heads, _HEAD_DIM)

    @T.prim_func
    def kernel(
        V: T.Tensor(token_shape, T.bfloat16),
        InitialState: T.Tensor(state_shape, T.bfloat16),
        CuSeqLens: T.Tensor(cu_shape, T.int32),
        AInv: T.Tensor(operator_shape, T.bfloat16),
        Beta: T.Tensor(beta_shape, T.bfloat16),
        KForward: T.Tensor(transform_shape, T.bfloat16),
        KUpdate: T.Tensor(transform_shape, T.bfloat16),
        Decay: T.Tensor(decay_shape, T.float32),
        StateScratch: T.Tensor(scratch_shape, T.bfloat16),
        VNew: T.Tensor(token_shape, T.bfloat16),
        FinalState: T.Tensor(state_shape, T.bfloat16),
        SegmentRound: T.int32,
    ):
        with T.Kernel(
            T.ceildiv(_HEAD_DIM, value_tile),
            num_heads,
            num_sequences,
            threads=_THREADS,
        ) as (value_block, head, sequence):
            x_shared = T.alloc_shared(
                (_CHUNK_SIZE, _HEAD_DIM), T.bfloat16
            )
            operator_shared = T.alloc_shared(
                (_CHUNK_SIZE, _CHUNK_SIZE), T.bfloat16
            )
            state_shared = T.alloc_shared(
                (_HEAD_DIM, value_tile), T.bfloat16
            )
            rhs_shared = T.alloc_shared(
                (_CHUNK_SIZE, value_tile), T.bfloat16
            )
            decay_shared = T.alloc_shared((_HEAD_DIM,), T.float32)

            state_fragment = T.alloc_fragment(
                (_HEAD_DIM, value_tile), T.float32
            )
            rhs_fragment = T.alloc_fragment(
                (_CHUNK_SIZE, value_tile), T.float32
            )
            chunk_prefix = T.alloc_local((1,), T.int32)
            first_chunk = T.alloc_local((1,), T.int32)
            last_chunk = T.alloc_local((1,), T.int32)

            if num_sequences == 1:
                sequence_start = 0
                sequence_length = total_tokens
                sequence_chunks = single_sequence_chunks
            else:
                sequence_start = CuSeqLens[sequence]
                sequence_length = CuSeqLens[sequence + 1] - sequence_start
                sequence_chunks = T.ceildiv(sequence_length, _CHUNK_SIZE)
            chunk_prefix[0] = 0
            if num_sequences == 1:
                first_chunk[0] = (
                    single_sequence_chunks * SegmentRound
                    // segment_rounds
                )
                last_chunk[0] = (
                    single_sequence_chunks * (SegmentRound + 1)
                    // segment_rounds
                )
            else:
                for prior_sequence in T.serial(num_sequences):
                    if prior_sequence < sequence:
                        prior_chunks = T.ceildiv(
                            CuSeqLens[prior_sequence + 1]
                            - CuSeqLens[prior_sequence],
                            _CHUNK_SIZE,
                        )
                        chunk_prefix[0] += (
                            prior_chunks * (SegmentRound + 1)
                            // segment_rounds
                            - prior_chunks * SegmentRound
                            // segment_rounds
                        )
                first_chunk[0] = (
                    sequence_chunks * SegmentRound // segment_rounds
                )
                last_chunk[0] = (
                    sequence_chunks * (SegmentRound + 1)
                    // segment_rounds
                )

            if first_chunk[0] < last_chunk[0]:
                if first_chunk[0] == 0:
                    T.copy(
                        InitialState[
                            sequence,
                            head,
                            0:_HEAD_DIM,
                            value_block
                            * value_tile : (value_block + 1)
                            * value_tile,
                        ],
                        state_fragment,
                    )
                else:
                    T.copy(
                        FinalState[
                            sequence,
                            head,
                            0:_HEAD_DIM,
                            value_block
                            * value_tile : (value_block + 1)
                            * value_tile,
                        ],
                        state_fragment,
                    )

                for chunk in T.serial(first_chunk[0], last_chunk[0]):
                    chunk_start = sequence_start + chunk * _CHUNK_SIZE
                    if (
                        num_sequences == 1
                        and total_tokens % _CHUNK_SIZE == 0
                    ):
                        valid_tokens = _CHUNK_SIZE
                    else:
                        valid_tokens = T.min(
                            _CHUNK_SIZE,
                            sequence_length - chunk * _CHUNK_SIZE,
                        )
                    scratch_chunk = (
                        chunk_prefix[0] + chunk - first_chunk[0]
                    )

                    T.async_copy(
                        KForward[
                            scratch_chunk,
                            head,
                            0:_CHUNK_SIZE,
                            0:_HEAD_DIM,
                        ],
                        x_shared,
                    )
                    for row, column in T.Parallel(
                        _CHUNK_SIZE, _CHUNK_SIZE
                    ):
                        operator_shared[row, column] = T.if_then_else(
                            row < valid_tokens,
                            AInv[chunk_start + row, head, column],
                            T.cast(0.0, T.bfloat16),
                        )
                    T.copy(
                        Decay[scratch_chunk, head, 0:_HEAD_DIM],
                        decay_shared,
                    )
                    T.copy(state_fragment, state_shared)
                    T.copy(
                        state_shared,
                        StateScratch[
                            scratch_chunk,
                            head,
                            0:_HEAD_DIM,
                            value_block
                            * value_tile : (value_block + 1)
                            * value_tile,
                        ],
                    )
                    T.ptx_wait_group(0)
                    T.sync_threads()
                    T.gemm(
                        x_shared,
                        state_shared,
                        rhs_fragment,
                        clear_accum=True,
                    )
                    for row, value in T.Parallel(
                        _CHUNK_SIZE, value_tile
                    ):
                        rhs_fragment[row, value] = T.if_then_else(
                            row < valid_tokens,
                            V[
                                chunk_start + row,
                                head,
                                value_block * value_tile + value,
                            ]
                            - rhs_fragment[row, value],
                            0.0,
                        )
                    T.copy(rhs_fragment, rhs_shared)
                    T.async_copy(
                        KUpdate[
                            scratch_chunk,
                            head,
                            0:_CHUNK_SIZE,
                            0:_HEAD_DIM,
                        ],
                        x_shared,
                    )
                    T.gemm(
                        operator_shared,
                        rhs_shared,
                        rhs_fragment,
                        clear_accum=True,
                    )
                    T.copy(rhs_fragment, rhs_shared)
                    for dim, value in T.Parallel(
                        _HEAD_DIM, value_tile
                    ):
                        state_fragment[dim, value] *= decay_shared[dim]
                    T.ptx_wait_group(0)
                    T.sync_threads()
                    T.gemm(
                        x_shared,
                        rhs_shared,
                        state_fragment,
                        transpose_A=True,
                    )
                    for row, value in T.Parallel(
                        _CHUNK_SIZE, value_tile
                    ):
                        if row < valid_tokens:
                            VNew[
                                chunk_start + row,
                                head,
                                value_block * value_tile + value,
                            ] = rhs_shared[row, value]

                T.copy(state_fragment, state_shared)
                T.copy(
                    state_shared,
                    FinalState[
                        sequence,
                        head,
                        0:_HEAD_DIM,
                        value_block
                        * value_tile : (value_block + 1)
                        * value_tile,
                    ],
                )

    return kernel


@tilelang.jit(
    out_idx=[],
    target=_TARGET,
    pass_configs={tilelang.PassConfigKey.TL_ENABLE_FAST_MATH: True},
)
def _compile_transformed_chunk_output(
    total_tokens: int,
    num_sequences: int,
    num_heads: int,
    segment_chunks: int,
    segment_rounds: int,
):
    single_sequence_chunks = (total_tokens + _CHUNK_SIZE - 1) // _CHUNK_SIZE
    value_tile = 64
    token_shape = (total_tokens, num_heads, _HEAD_DIM)
    cu_shape = (num_sequences + 1,)
    operator_shape = (total_tokens, num_heads, _CHUNK_SIZE)
    scratch_shape = (
        segment_chunks,
        num_heads,
        _HEAD_DIM,
        _HEAD_DIM,
    )
    transform_shape = (
        segment_chunks,
        num_heads,
        _CHUNK_SIZE,
        _HEAD_DIM,
    )

    @T.prim_func
    def kernel(
        CuSeqLens: T.Tensor(cu_shape, T.int32),
        Aqk: T.Tensor(operator_shape, T.bfloat16),
        QG: T.Tensor(transform_shape, T.bfloat16),
        StateScratch: T.Tensor(scratch_shape, T.bfloat16),
        VNewOut: T.Tensor(token_shape, T.bfloat16),
        SegmentRound: T.int32,
    ):
        with T.Kernel(
            segment_chunks,
            T.ceildiv(_HEAD_DIM, value_tile),
            num_heads,
            threads=_THREADS,
        ) as (segment_chunk, value_block, head):
            sequence_id = T.alloc_local((1,), T.int32)
            chunk_in_sequence = T.alloc_local((1,), T.int32)
            chunk_prefix = T.alloc_local((1,), T.int32)

            sequence_id[0] = -1
            chunk_in_sequence[0] = 0
            chunk_prefix[0] = 0
            if num_sequences == 1:
                first_chunk = (
                    single_sequence_chunks * SegmentRound
                    // segment_rounds
                )
                last_chunk = (
                    single_sequence_chunks * (SegmentRound + 1)
                    // segment_rounds
                )
                if segment_chunk < last_chunk - first_chunk:
                    sequence_id[0] = 0
                    chunk_in_sequence[0] = first_chunk + segment_chunk
            else:
                for sequence in T.serial(num_sequences):
                    sequence_chunks = T.ceildiv(
                        CuSeqLens[sequence + 1] - CuSeqLens[sequence],
                        _CHUNK_SIZE,
                    )
                    first_chunk = (
                        sequence_chunks * SegmentRound // segment_rounds
                    )
                    last_chunk = (
                        sequence_chunks * (SegmentRound + 1)
                        // segment_rounds
                    )
                    if (
                        segment_chunk >= chunk_prefix[0]
                        and segment_chunk
                        < chunk_prefix[0] + last_chunk - first_chunk
                    ):
                        sequence_id[0] = sequence
                        chunk_in_sequence[0] = (
                            first_chunk + segment_chunk - chunk_prefix[0]
                        )
                    chunk_prefix[0] += last_chunk - first_chunk

            if sequence_id[0] >= 0:
                if num_sequences == 1:
                    chunk_start = chunk_in_sequence[0] * _CHUNK_SIZE
                    if total_tokens % _CHUNK_SIZE == 0:
                        valid_tokens = _CHUNK_SIZE
                    else:
                        valid_tokens = T.min(
                            _CHUNK_SIZE, total_tokens - chunk_start
                        )
                else:
                    chunk_start = (
                        CuSeqLens[sequence_id[0]]
                        + chunk_in_sequence[0] * _CHUNK_SIZE
                    )
                    valid_tokens = T.min(
                        _CHUNK_SIZE,
                        CuSeqLens[sequence_id[0] + 1] - chunk_start,
                    )
                q_shared = T.alloc_shared(
                    (_CHUNK_SIZE, _HEAD_DIM), T.bfloat16
                )
                operator_shared = T.alloc_shared(
                    (_CHUNK_SIZE, _CHUNK_SIZE), T.bfloat16
                )
                state_shared = T.alloc_shared(
                    (_HEAD_DIM, value_tile), T.bfloat16
                )
                rhs_shared = T.alloc_shared(
                    (_CHUNK_SIZE, value_tile), T.bfloat16
                )
                out_fragment = T.alloc_fragment(
                    (_CHUNK_SIZE, value_tile), T.float32
                )

                T.async_copy(
                    QG[
                        segment_chunk,
                        head,
                        0:_CHUNK_SIZE,
                        0:_HEAD_DIM,
                    ],
                    q_shared,
                )
                for row, column in T.Parallel(
                    _CHUNK_SIZE, _CHUNK_SIZE
                ):
                    operator_shared[row, column] = T.if_then_else(
                        row < valid_tokens,
                        Aqk[chunk_start + row, head, column],
                        T.cast(0.0, T.bfloat16),
                    )
                T.copy(
                    StateScratch[
                        segment_chunk,
                        head,
                        0:_HEAD_DIM,
                        value_block
                        * value_tile : (value_block + 1)
                        * value_tile,
                    ],
                    state_shared,
                )
                for row, value in T.Parallel(
                    _CHUNK_SIZE, value_tile
                ):
                    rhs_shared[row, value] = T.if_then_else(
                        row < valid_tokens,
                        VNewOut[
                            chunk_start + row,
                            head,
                            value_block * value_tile + value,
                        ],
                        T.cast(0.0, T.bfloat16),
                    )
                T.ptx_wait_group(0)
                T.sync_threads()
                T.gemm(
                    q_shared,
                    state_shared,
                    out_fragment,
                    clear_accum=True,
                )
                T.gemm(
                    operator_shared,
                    rhs_shared,
                    out_fragment,
                )
                T.copy(out_fragment, rhs_shared)
                for row, value in T.Parallel(
                    _CHUNK_SIZE, value_tile
                ):
                    if row < valid_tokens:
                        VNewOut[
                            chunk_start + row,
                            head,
                            value_block * value_tile + value,
                        ] = rhs_shared[row, value]

    return kernel


class Submission:
    def build(self, spec: Any) -> Any:
        total_tokens = int(spec.total_tokens)
        num_sequences = int(spec.num_sequences)
        num_heads = int(spec.num_heads)
        operator_elements = total_tokens * num_heads * _CHUNK_SIZE
        token_head_elements = total_tokens * num_heads
        max_chunks = (
            (total_tokens + _CHUNK_SIZE - 1) // _CHUNK_SIZE
            + num_sequences
            - 1
        )
        scratch_offset = 4 * operator_elements + 10 * token_head_elements
        tail_chunk_bytes = num_heads * (
            _HEAD_DIM * _HEAD_DIM * 2
            + 3 * _CHUNK_SIZE * _HEAD_DIM * 2
            + _HEAD_DIM * 4
        )
        segment_capacity = min(
            max_chunks,
            (int(spec.workspace_bytes) - scratch_offset)
            // tail_chunk_bytes,
        )
        if segment_capacity <= 0:
            raise ValueError("workspace cannot hold one transformed chunk")
        if segment_capacity < num_sequences:
            raise ValueError("workspace cannot hold one chunk per sequence")
        segment_rounds = (
            max_chunks + segment_capacity - 1
        ) // segment_capacity
        while (
            (max_chunks + segment_rounds - 1) // segment_rounds
            + num_sequences
            - 1
            > segment_capacity
        ):
            segment_rounds += 1
        segment_chunks = min(
            max_chunks,
            (max_chunks + segment_rounds - 1) // segment_rounds
            + num_sequences
            - 1,
        )

        chunk_diagonal = _compile_chunk_diagonal(
            total_tokens, num_sequences, num_heads
        )
        chunk_inter = _compile_chunk_inter(
            total_tokens, num_sequences, num_heads
        )
        chunk_transform = _compile_chunk_transform(
            total_tokens,
            num_sequences,
            num_heads,
            segment_chunks,
            segment_rounds,
        )
        state_scan = _compile_transformed_state_scan(
            total_tokens,
            num_sequences,
            num_heads,
            (
                64
                if num_sequences * num_heads >= 64 and num_sequences > 8
                else (8 if num_sequences * num_heads <= 32 else 32)
            ),
            segment_chunks,
            segment_rounds,
        )
        chunk_output = _compile_transformed_chunk_output(
            total_tokens,
            num_sequences,
            num_heads,
            segment_chunks,
            segment_rounds,
        )
        return (
            chunk_diagonal,
            chunk_inter,
            chunk_transform,
            state_scan,
            chunk_output,
            operator_elements,
            scratch_offset,
            segment_chunks,
            segment_rounds,
            max_chunks,
        )

    @torch.no_grad()
    def run(
        self,
        state: Any,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        g_raw: torch.Tensor,
        beta_raw: torch.Tensor,
        a_log: torch.Tensor,
        dt_bias: torch.Tensor,
        initial_state: torch.Tensor,
        cu_seqlens: torch.Tensor,
        workspace: torch.Tensor,
        out: torch.Tensor,
        final_state: torch.Tensor,
    ) -> None:
        (
            chunk_diagonal,
            chunk_inter,
            chunk_transform,
            state_scan,
            chunk_output,
            operator_elements,
            scratch_offset,
            segment_chunks,
            segment_rounds,
            max_chunks,
        ) = state
        workspace_bf16 = workspace.view(torch.bfloat16)
        ainv_diagonal = workspace[: 2 * operator_elements].view(
            torch.float32
        ).view(q.shape[0], q.shape[1], _CHUNK_SIZE // 2)
        ainv = workspace_bf16[:operator_elements].view(
            q.shape[0], q.shape[1], _CHUNK_SIZE
        )
        aqk = workspace_bf16[
            operator_elements : 2 * operator_elements
        ].view(q.shape[0], q.shape[1], _CHUNK_SIZE)
        token_head_elements = q.shape[0] * q.shape[1]
        norm_offset = 4 * operator_elements
        norm = workspace[
            norm_offset : norm_offset + 8 * token_head_elements
        ].view(torch.float32).view(q.shape[0], q.shape[1], 2)
        beta = workspace[
            norm_offset
            + 8 * token_head_elements : norm_offset
            + 10 * token_head_elements
        ].view(torch.bfloat16).view(q.shape[0], q.shape[1])
        state_scratch_bytes = (
            2 * segment_chunks * q.shape[1] * _HEAD_DIM * _HEAD_DIM
        )
        transform_scratch_bytes = (
            2 * segment_chunks * q.shape[1] * _CHUNK_SIZE * _HEAD_DIM
        )
        decay_scratch_bytes = (
            4 * segment_chunks * q.shape[1] * _HEAD_DIM
        )
        scratch_cursor = scratch_offset
        state_scratch = workspace[
            scratch_cursor : scratch_cursor + state_scratch_bytes
        ].view(torch.bfloat16).view(
            segment_chunks,
            q.shape[1],
            _HEAD_DIM,
            _HEAD_DIM,
        )
        scratch_cursor += state_scratch_bytes
        qg_scratch = workspace[
            scratch_cursor : scratch_cursor + transform_scratch_bytes
        ].view(torch.bfloat16).view(
            segment_chunks,
            q.shape[1],
            _CHUNK_SIZE,
            _HEAD_DIM,
        )
        scratch_cursor += transform_scratch_bytes
        k_forward_scratch = workspace[
            scratch_cursor : scratch_cursor + transform_scratch_bytes
        ].view(torch.bfloat16).view(
            segment_chunks,
            q.shape[1],
            _CHUNK_SIZE,
            _HEAD_DIM,
        )
        scratch_cursor += transform_scratch_bytes
        k_update_scratch = workspace[
            scratch_cursor : scratch_cursor + transform_scratch_bytes
        ].view(torch.bfloat16).view(
            segment_chunks,
            q.shape[1],
            _CHUNK_SIZE,
            _HEAD_DIM,
        )
        scratch_cursor += transform_scratch_bytes
        decay_scratch = workspace[
            scratch_cursor : scratch_cursor + decay_scratch_bytes
        ].view(torch.float32).view(
            segment_chunks,
            q.shape[1],
            _HEAD_DIM,
        )

        chunk_diagonal(
            q,
            k,
            g_raw,
            beta_raw,
            a_log,
            dt_bias,
            cu_seqlens,
            ainv_diagonal,
            aqk,
            norm,
            beta,
        )
        chunk_inter(
            q,
            k,
            g_raw,
            a_log,
            dt_bias,
            cu_seqlens,
            ainv_diagonal,
            ainv,
            aqk,
            norm,
            beta,
        )
        for segment_round in range(segment_rounds):
            chunk_transform(
                q,
                k,
                g_raw,
                a_log,
                dt_bias,
                cu_seqlens,
                norm,
                qg_scratch,
                k_forward_scratch,
                k_update_scratch,
                decay_scratch,
                segment_round,
            )
            state_scan(
                v,
                initial_state,
                cu_seqlens,
                ainv,
                beta,
                k_forward_scratch,
                k_update_scratch,
                decay_scratch,
                state_scratch,
                out,
                final_state,
                segment_round,
            )
            chunk_output(
                cu_seqlens,
                aqk,
                qg_scratch,
                state_scratch,
                out,
                segment_round,
            )
