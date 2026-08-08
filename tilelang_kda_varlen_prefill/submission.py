from typing import Any

import tilelang
import tilelang.language as T
import torch


_CHUNK_SIZE = 64
_HEAD_DIM = 128
_VALUE_TILE = 8
_OPERATOR_THREADS = 32
_THREADS = 128
_INV_SQRT_HEAD_DIM = 0.08838834764831845
_LOG2_GATE_SCALE = -7.213475204444817
_TARGET = {"kind": "cuda", "arch": "sm_89"}


@tilelang.jit(
    out_idx=[],
    target=_TARGET,
    pass_configs={tilelang.PassConfigKey.TL_ENABLE_FAST_MATH: True},
)
def _compile_chunk_operators(total_tokens: int, num_sequences: int, num_heads: int):
    max_chunks = (total_tokens + _CHUNK_SIZE - 1) // _CHUNK_SIZE + num_sequences - 1

    token_shape = (total_tokens, num_heads, _HEAD_DIM)
    beta_shape = (total_tokens, num_heads)
    a_log_shape = (num_heads,)
    dt_bias_shape = (num_heads, _HEAD_DIM)
    cu_shape = (num_sequences + 1,)
    operator_shape = (total_tokens, num_heads, _CHUNK_SIZE)

    @T.prim_func
    def kernel(
        Q: T.Tensor(token_shape, T.bfloat16),
        K: T.Tensor(token_shape, T.bfloat16),
        GRaw: T.Tensor(token_shape, T.bfloat16),
        BetaRaw: T.Tensor(beta_shape, T.bfloat16),
        ALog: T.Tensor(a_log_shape, T.float32),
        DtBias: T.Tensor(dt_bias_shape, T.float32),
        CuSeqLens: T.Tensor(cu_shape, T.int32),
        AInv: T.Tensor(operator_shape, T.bfloat16),
        Aqk: T.Tensor(operator_shape, T.bfloat16),
    ):
        with T.Kernel(
            max_chunks, num_heads, threads=_OPERATOR_THREADS
        ) as (chunk_id, head):
            sequence_id = T.alloc_local((1,), T.int32)
            chunk_in_sequence = T.alloc_local((1,), T.int32)
            chunk_prefix = T.alloc_local((1,), T.int32)

            sequence_id[0] = -1
            chunk_in_sequence[0] = 0
            chunk_prefix[0] = 0
            for sequence in T.serial(num_sequences):
                sequence_chunks = T.ceildiv(
                    CuSeqLens[sequence + 1] - CuSeqLens[sequence], _CHUNK_SIZE
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
                k_shared = T.alloc_shared(
                    (_CHUNK_SIZE, _HEAD_DIM), T.bfloat16
                )
                g_shared = T.alloc_shared(
                    (_CHUNK_SIZE, _HEAD_DIM), T.float32
                )
                aqk_shared = T.alloc_shared(
                    (_CHUNK_SIZE, _CHUNK_SIZE), T.bfloat16
                )
                ainv_shared = T.alloc_shared(
                    (_CHUNK_SIZE, _CHUNK_SIZE), T.float32
                )
                q_norm = T.alloc_shared((_CHUNK_SIZE,), T.float32)
                k_norm = T.alloc_shared((_CHUNK_SIZE,), T.float32)
                beta = T.alloc_shared((_CHUNK_SIZE,), T.bfloat16)

                left_q = T.alloc_shared((16, 32), T.bfloat16)
                left_k = T.alloc_shared((16, 32), T.bfloat16)
                right_k = T.alloc_shared((16, 32), T.bfloat16)
                aqk_fragment = T.alloc_fragment((16, 16), T.float32)
                akk_fragment = T.alloc_fragment((16, 16), T.float32)
                a_scale = T.alloc_local((1,), T.float32)
                coefficient_row = T.alloc_shared(
                    (_CHUNK_SIZE,), T.float32
                )

                a_scale[0] = T.exp(ALog[head])
                for row, dim in T.Parallel(_CHUNK_SIZE, _HEAD_DIM):
                    q_shared[row, dim] = T.if_then_else(
                        row < valid_tokens,
                        Q[chunk_start + row, head, dim],
                        0.0,
                    )
                    k_shared[row, dim] = T.if_then_else(
                        row < valid_tokens,
                        K[chunk_start + row, head, dim],
                        0.0,
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
                    beta[row] = T.if_then_else(
                        row < valid_tokens,
                        T.sigmoid(BetaRaw[chunk_start + row, head]),
                        0.0,
                    )
                T.sync_threads()

                for dim in T.Parallel(_HEAD_DIM):
                    for row in T.serial(1, _CHUNK_SIZE):
                        if row < valid_tokens:
                            g_shared[row, dim] += g_shared[row - 1, dim]
                T.sync_threads()

                for row in T.Parallel(_CHUNK_SIZE):
                    q_norm[row] = 1.0e-6
                    k_norm[row] = 1.0e-6
                    for dim in T.serial(_HEAD_DIM):
                        q_norm[row] += (
                            T.cast(q_shared[row, dim], T.float32)
                            * T.cast(q_shared[row, dim], T.float32)
                        )
                        k_norm[row] += (
                            T.cast(k_shared[row, dim], T.float32)
                            * T.cast(k_shared[row, dim], T.float32)
                        )
                    q_norm[row] = T.rsqrt(q_norm[row])
                    k_norm[row] = T.rsqrt(k_norm[row])
                T.sync_threads()

                for row, dim in T.Parallel(_CHUNK_SIZE, _HEAD_DIM):
                    q_shared[row, dim] *= q_norm[row]
                    k_shared[row, dim] *= k_norm[row]
                T.clear(aqk_shared)
                T.clear(ainv_shared)

                for row_block in T.serial(4):
                    for column_block in T.serial(row_block + 1):
                        T.clear(aqk_fragment)
                        T.clear(akk_fragment)
                        for dim_block in T.serial(4):
                            for row, dim in T.Parallel(16, 32):
                                row_index = row_block * 16 + row
                                column_index = column_block * 16 + row
                                dim_index = dim_block * 32 + dim
                                gate_reference = g_shared[
                                    row_block * 16, dim_index
                                ]
                                left_q[row, dim] = T.if_then_else(
                                    row_index < valid_tokens,
                                    q_shared[row_index, dim_index]
                                    * T.exp2(
                                        g_shared[row_index, dim_index]
                                        - gate_reference
                                    )
                                    * _INV_SQRT_HEAD_DIM,
                                    0.0,
                                )
                                left_k[row, dim] = T.if_then_else(
                                    row_index < valid_tokens,
                                    k_shared[row_index, dim_index]
                                    * beta[row_index]
                                    * T.exp2(
                                        g_shared[row_index, dim_index]
                                        - gate_reference
                                    ),
                                    0.0,
                                )
                                right_k[row, dim] = T.if_then_else(
                                    column_index < valid_tokens,
                                    k_shared[column_index, dim_index]
                                    * T.exp2(
                                        gate_reference
                                        - g_shared[column_index, dim_index]
                                    ),
                                    0.0,
                                )
                            T.sync_threads()
                            T.gemm(
                                left_q,
                                right_k,
                                aqk_fragment,
                                transpose_B=True,
                            )
                            T.gemm(
                                left_k,
                                right_k,
                                akk_fragment,
                                transpose_B=True,
                            )

                        for row, column in T.Parallel(16, 16):
                            row_index = row_block * 16 + row
                            column_index = column_block * 16 + column
                            if (
                                row_index < valid_tokens
                                and column_index < valid_tokens
                                and column_index <= row_index
                            ):
                                aqk_shared[row_index, column_index] = (
                                    aqk_fragment[row, column]
                                )
                            if (
                                row_index < valid_tokens
                                and column_index < row_index
                            ):
                                ainv_shared[row_index, column_index] = (
                                    akk_fragment[row, column]
                                )
                        T.sync_threads()

                for row in T.serial(_CHUNK_SIZE):
                    if row < valid_tokens:
                        for column in T.Parallel(_CHUNK_SIZE):
                            coefficient_row[column] = T.if_then_else(
                                column < row,
                                ainv_shared[row, column],
                                0.0,
                            )
                        T.sync_threads()
                        for column in T.Parallel(_CHUNK_SIZE):
                            if column < row:
                                ainv_shared[row, column] = -coefficient_row[
                                    column
                                ]
                                for inner in T.serial(_CHUNK_SIZE):
                                    if inner > column and inner < row:
                                        ainv_shared[row, column] -= (
                                            coefficient_row[inner]
                                            * ainv_shared[inner, column]
                                        )
                            if column == row:
                                ainv_shared[row, column] = 1.0
                            if column > row:
                                ainv_shared[row, column] = 0.0
                        T.sync_threads()

                for row, column in T.Parallel(
                    _CHUNK_SIZE, _CHUNK_SIZE
                ):
                    if row < valid_tokens:
                        AInv[chunk_start + row, head, column] = ainv_shared[
                            row, column
                        ]
                        Aqk[chunk_start + row, head, column] = aqk_shared[
                            row, column
                        ]

    return kernel


@tilelang.jit(
    out_idx=[],
    target=_TARGET,
    pass_configs={tilelang.PassConfigKey.TL_ENABLE_FAST_MATH: True},
)
def _compile_persistent_recurrence(
    total_tokens: int, num_sequences: int, num_heads: int
):
    max_chunks_per_sequence = (
        total_tokens + _CHUNK_SIZE - 1
    ) // _CHUNK_SIZE

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

    @T.prim_func
    def kernel(
        Q: T.Tensor(token_shape, T.bfloat16),
        K: T.Tensor(token_shape, T.bfloat16),
        V: T.Tensor(token_shape, T.bfloat16),
        GRaw: T.Tensor(token_shape, T.bfloat16),
        BetaRaw: T.Tensor(beta_shape, T.bfloat16),
        ALog: T.Tensor(a_log_shape, T.float32),
        DtBias: T.Tensor(dt_bias_shape, T.float32),
        InitialState: T.Tensor(state_shape, T.bfloat16),
        CuSeqLens: T.Tensor(cu_shape, T.int32),
        AInv: T.Tensor(operator_shape, T.bfloat16),
        Aqk: T.Tensor(operator_shape, T.bfloat16),
        Out: T.Tensor(token_shape, T.bfloat16),
        FinalState: T.Tensor(state_shape, T.bfloat16),
    ):
        with T.Kernel(
            T.ceildiv(_HEAD_DIM, _VALUE_TILE),
            num_heads,
            num_sequences,
            threads=_THREADS,
        ) as (value_block, head, sequence):
            k_shared = T.alloc_shared(
                (_CHUNK_SIZE, _HEAD_DIM), T.bfloat16
            )
            x_shared = T.alloc_shared(
                (_CHUNK_SIZE, _HEAD_DIM), T.bfloat16
            )
            g_shared = T.alloc_shared(
                (_CHUNK_SIZE, _HEAD_DIM), T.float32
            )
            ainv_shared = T.alloc_shared(
                (_CHUNK_SIZE, _CHUNK_SIZE), T.bfloat16
            )
            aqk_shared = T.alloc_shared(
                (_CHUNK_SIZE, _CHUNK_SIZE), T.bfloat16
            )
            state_shared = T.alloc_shared(
                (_HEAD_DIM, _VALUE_TILE), T.bfloat16
            )
            rhs_shared = T.alloc_shared(
                (_CHUNK_SIZE, _VALUE_TILE), T.bfloat16
            )
            value_new_shared = T.alloc_shared(
                (_CHUNK_SIZE, _VALUE_TILE), T.bfloat16
            )
            out_shared = T.alloc_shared(
                (_CHUNK_SIZE, _VALUE_TILE), T.bfloat16
            )
            norm = T.alloc_shared((_CHUNK_SIZE,), T.float32)
            beta = T.alloc_shared((_CHUNK_SIZE,), T.bfloat16)

            state_fragment = T.alloc_fragment(
                (_HEAD_DIM, _VALUE_TILE), T.float32
            )
            rhs_fragment = T.alloc_fragment(
                (_CHUNK_SIZE, _VALUE_TILE), T.float32
            )
            value_new_fragment = T.alloc_fragment(
                (_CHUNK_SIZE, _VALUE_TILE), T.float32
            )
            out_fragment = T.alloc_fragment(
                (_CHUNK_SIZE, _VALUE_TILE), T.float32
            )
            a_scale = T.alloc_local((1,), T.float32)

            T.copy(
                InitialState[
                    sequence,
                    head,
                    0:_HEAD_DIM,
                    value_block
                    * _VALUE_TILE : (value_block + 1)
                    * _VALUE_TILE,
                ],
                state_shared,
            )
            T.copy(state_shared, state_fragment)

            sequence_start = CuSeqLens[sequence]
            sequence_length = CuSeqLens[sequence + 1] - sequence_start
            a_scale[0] = T.exp(ALog[head])

            for chunk in T.serial(max_chunks_per_sequence):
                if chunk * _CHUNK_SIZE < sequence_length:
                    chunk_start = sequence_start + chunk * _CHUNK_SIZE
                    valid_tokens = T.min(
                        _CHUNK_SIZE,
                        sequence_length - chunk * _CHUNK_SIZE,
                    )

                    for row, dim in T.Parallel(
                        _CHUNK_SIZE, _HEAD_DIM
                    ):
                        k_shared[row, dim] = T.if_then_else(
                            row < valid_tokens,
                            K[chunk_start + row, head, dim],
                            0.0,
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
                        beta[row] = T.if_then_else(
                            row < valid_tokens,
                            T.sigmoid(BetaRaw[chunk_start + row, head]),
                            0.0,
                        )
                    T.sync_threads()

                    for dim in T.Parallel(_HEAD_DIM):
                        for row in T.serial(1, _CHUNK_SIZE):
                            if row < valid_tokens:
                                g_shared[row, dim] += g_shared[row - 1, dim]
                    T.sync_threads()

                    for row in T.Parallel(_CHUNK_SIZE):
                        norm[row] = 1.0e-6
                        for dim in T.serial(_HEAD_DIM):
                            norm[row] += (
                                T.cast(k_shared[row, dim], T.float32)
                                * T.cast(k_shared[row, dim], T.float32)
                            )
                        norm[row] = T.rsqrt(norm[row])
                    T.sync_threads()

                    for row, dim in T.Parallel(
                        _CHUNK_SIZE, _HEAD_DIM
                    ):
                        k_shared[row, dim] *= norm[row]
                    for row, column in T.Parallel(
                        _CHUNK_SIZE, _CHUNK_SIZE
                    ):
                        ainv_shared[row, column] = T.if_then_else(
                            row < valid_tokens,
                            AInv[chunk_start + row, head, column],
                            0.0,
                        )
                        aqk_shared[row, column] = T.if_then_else(
                            row < valid_tokens,
                            Aqk[chunk_start + row, head, column],
                            0.0,
                        )
                    T.copy(state_fragment, state_shared)
                    T.sync_threads()

                    for row, dim in T.Parallel(
                        _CHUNK_SIZE, _HEAD_DIM
                    ):
                        x_shared[row, dim] = T.if_then_else(
                            row < valid_tokens,
                            k_shared[row, dim]
                            * T.exp2(g_shared[row, dim]),
                            0.0,
                        )
                    T.sync_threads()
                    T.gemm(
                        x_shared,
                        state_shared,
                        rhs_fragment,
                        clear_accum=True,
                    )
                    for row, value in T.Parallel(
                        _CHUNK_SIZE, _VALUE_TILE
                    ):
                        rhs_fragment[row, value] = T.if_then_else(
                            row < valid_tokens,
                            beta[row]
                            * (
                                V[
                                    chunk_start + row,
                                    head,
                                    value_block * _VALUE_TILE + value,
                                ]
                                - rhs_fragment[row, value]
                            ),
                            0.0,
                        )
                    T.copy(rhs_fragment, rhs_shared)
                    T.gemm(
                        ainv_shared,
                        rhs_shared,
                        value_new_fragment,
                        clear_accum=True,
                    )
                    T.copy(value_new_fragment, value_new_shared)

                    for row, dim in T.Parallel(
                        _CHUNK_SIZE, _HEAD_DIM
                    ):
                        x_shared[row, dim] = T.if_then_else(
                            row < valid_tokens,
                            Q[chunk_start + row, head, dim],
                            0.0,
                        )
                    T.sync_threads()
                    for row in T.Parallel(_CHUNK_SIZE):
                        norm[row] = 1.0e-6
                        for dim in T.serial(_HEAD_DIM):
                            norm[row] += (
                                T.cast(x_shared[row, dim], T.float32)
                                * T.cast(x_shared[row, dim], T.float32)
                            )
                        norm[row] = T.rsqrt(norm[row])
                    T.sync_threads()
                    for row, dim in T.Parallel(
                        _CHUNK_SIZE, _HEAD_DIM
                    ):
                        x_shared[row, dim] = T.if_then_else(
                            row < valid_tokens,
                            x_shared[row, dim]
                            * norm[row]
                            * T.exp2(g_shared[row, dim])
                            * _INV_SQRT_HEAD_DIM,
                            0.0,
                        )
                    T.sync_threads()
                    T.gemm(
                        x_shared,
                        state_shared,
                        out_fragment,
                        clear_accum=True,
                    )
                    T.gemm(
                        aqk_shared,
                        value_new_shared,
                        out_fragment,
                    )
                    T.copy(out_fragment, out_shared)
                    for row, value in T.Parallel(
                        _CHUNK_SIZE, _VALUE_TILE
                    ):
                        if row < valid_tokens:
                            Out[
                                chunk_start + row,
                                head,
                                value_block * _VALUE_TILE + value,
                            ] = out_shared[row, value]

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
                        _HEAD_DIM, _VALUE_TILE
                    ):
                        state_fragment[dim, value] *= T.exp2(
                            g_shared[valid_tokens - 1, dim]
                        )
                    T.sync_threads()
                    T.gemm(
                        x_shared,
                        value_new_shared,
                        state_fragment,
                        transpose_A=True,
                    )

            T.copy(state_fragment, state_shared)
            T.copy(
                state_shared,
                FinalState[
                    sequence,
                    head,
                    0:_HEAD_DIM,
                    value_block
                    * _VALUE_TILE : (value_block + 1)
                    * _VALUE_TILE,
                ],
            )

    return kernel


class Submission:
    def build(self, spec: Any) -> Any:
        total_tokens = int(spec.total_tokens)
        num_sequences = int(spec.num_sequences)
        num_heads = int(spec.num_heads)
        operator_elements = total_tokens * num_heads * _CHUNK_SIZE

        chunk_operators = _compile_chunk_operators(
            total_tokens, num_sequences, num_heads
        )
        persistent_recurrence = _compile_persistent_recurrence(
            total_tokens, num_sequences, num_heads
        )
        return chunk_operators, persistent_recurrence, operator_elements

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
        chunk_operators, persistent_recurrence, operator_elements = state
        workspace_bf16 = workspace.view(torch.bfloat16)
        ainv = workspace_bf16[:operator_elements].view(
            q.shape[0], q.shape[1], _CHUNK_SIZE
        )
        aqk = workspace_bf16[
            operator_elements : 2 * operator_elements
        ].view(q.shape[0], q.shape[1], _CHUNK_SIZE)

        chunk_operators(
            q,
            k,
            g_raw,
            beta_raw,
            a_log,
            dt_bias,
            cu_seqlens,
            ainv,
            aqk,
        )
        persistent_recurrence(
            q,
            k,
            v,
            g_raw,
            beta_raw,
            a_log,
            dt_bias,
            initial_state,
            cu_seqlens,
            ainv,
            aqk,
            out,
            final_state,
        )
