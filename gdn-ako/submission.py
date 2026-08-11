"""选手提交模板：实现 gdn_chunk_scan(q, k, v, g, beta, scale) -> o。

语义须等价于 fla 的 chunk_gated_delta_rule(..., use_qk_l2norm_in_kernel=True)。
完整题面 / 数据范围 / 合法性边界见 README（尤其 2.2 接口 与 2.6 口径）。

输入（单卡，无并行）：
  q, k : bf16 [B,T,H,K]   query/key，kernel 内做 L2 norm（H=8, K=128）
  v    : bf16 [B,T,Hv,V]  value（Hv=8, V=128）
  g    : fp32 [B,T,H]     log-decay 门控，exp(g)∈(0,1)
  beta : fp32 [B,T,H]     delta-rule 更新强度 ∈(0,1)
  scale: float            = K**-0.5；chunk 固定 64；状态累加 fp32
输出：
  o    : bf16 [B,T,Hv,V]

autograd：评测会对 o 做 backward，梯度须回传到 q/k/v/beta。只优化前向、把 backward
交回 fla（如下默认实现）也能拿前向分；实现自定义反向走 bonus track（反向 ≈ 前向 3.7×）。

红线（见 README 2.6）：可对固定形状特化 / 预编译 / autotune / CUDA graph / 持久 workspace，
但**不得跨独立 trial 缓存依赖输入值的输出**——评测逐轮刷新输入并复验，缓存/快照会被判正确性失败。
"""

import tilelang
import tilelang.language as T
import torch

from fla.modules.l2norm import l2norm_bwd, l2norm_fwd
from fla.ops.common.chunk_delta_h import chunk_gated_delta_rule_bwd_dhu
from fla.ops.common.chunk_o import chunk_bwd_dqkwg, chunk_bwd_dv_local, chunk_fwd_o
from fla.ops.gated_delta_rule.chunk_fwd import chunk_gated_delta_rule_fwd_intra
from fla.ops.gated_delta_rule.wy_fast import prepare_wy_repr_bwd
from fla.ops.utils import chunk_local_cumsum
from fla.ops.utils.constant import RCP_LN2


@tilelang.jit(
    out_idx=[-2, -1],
    pass_configs={tilelang.PassConfigKey.TL_ENABLE_FAST_MATH: True},
)
def _chunk_state(B, S):
    H, K, V, BT, BV = 8, 128, 128, 64, 16

    @T.prim_func
    def kernel(
        k: T.Tensor((B, S, H, K), T.bfloat16),
        w: T.Tensor((B, S, H, K), T.bfloat16),
        u: T.Tensor((B, S, H, V), T.bfloat16),
        g: T.Tensor((B, S, H), T.float32),
        h: T.Tensor((B, (S + BT - 1) // BT, H, K, V), T.bfloat16),
        v_new: T.Tensor((B, S, H, V), T.bfloat16),
    ):
        with T.Kernel(V // BV, B * H, threads=128) as (i_v, i_bh):
            i_b, i_h = i_bh // H, i_bh % H
            state_shared = T.alloc_shared((K, BV), T.bfloat16)
            state = T.alloc_fragment((K, BV), T.float32)
            u_shared = T.alloc_shared((BT, BV), T.bfloat16)
            u_fragment = T.alloc_fragment((BT, BV), T.float32)
            w_shared = T.alloc_shared((BT, K), T.bfloat16)
            v_new_fragment = T.alloc_fragment((BT, BV), T.float32)
            v_new_shared = T.alloc_shared((BT, BV), T.bfloat16)
            k_shared = T.alloc_shared((BT, K), T.bfloat16)
            g_last = T.alloc_var(T.float32)
            g_shared = T.alloc_shared((BT, BV), T.float32)
            g_fragment = T.alloc_fragment((BT, BV), T.float32)

            T.annotate_layout({
                u_shared: tilelang.layout.make_swizzled_layout(u_shared),
                g_shared: tilelang.layout.make_swizzled_layout(g_shared),
            })
            T.use_swizzle(10)
            T.clear(state)
            T.copy(state, state_shared)

            for i_t in T.Pipelined((S + BT - 1) // BT, num_stages=1):
                for j_v, i_k in T.Parallel(BV, K):
                    h[i_b, i_t, i_h, i_v * BV + j_v, i_k] = state_shared[i_k, j_v]
                T.copy(w[i_b, i_t * BT : (i_t + 1) * BT, i_h, 0:K], w_shared)
                T.gemm(w_shared, state_shared, v_new_fragment, clear_accum=True)

                T.copy(u[i_b, i_t * BT : (i_t + 1) * BT, i_h, i_v * BV : (i_v + 1) * BV], u_shared)
                T.copy(u_shared, u_fragment)
                for i_s, j_v in T.Parallel(BT, BV):
                    v_new_fragment[i_s, j_v] = u_fragment[i_s, j_v] - v_new_fragment[i_s, j_v]
                T.copy(v_new_fragment, v_new_shared)
                T.copy(v_new_shared, v_new[i_b, i_t * BT : (i_t + 1) * BT, i_h, i_v * BV : (i_v + 1) * BV])

                g_last = g[i_b, T.min((i_t + 1) * BT, S) - 1, i_h]
                for i_s, j_v in T.Parallel(BT, BV):
                    if i_t * BT + i_s < S:
                        g_shared[i_s, j_v] = g[i_b, i_t * BT + i_s, i_h]
                    else:
                        g_shared[i_s, j_v] = g_last
                T.copy(g_shared, g_fragment)
                for i_s, j_v in T.Parallel(BT, BV):
                    if i_t * BT + i_s < S:
                        v_new_fragment[i_s, j_v] = (
                            v_new_fragment[i_s, j_v] * T.exp2(g_last - g_fragment[i_s, j_v])
                            if g_last - g_fragment[i_s, j_v] <= 0
                            else 0
                        )
                    else:
                        v_new_fragment[i_s, j_v] = 0
                g_last = T.exp2(g_last)
                for i_k, j_v in T.Parallel(K, BV):
                    state[i_k, j_v] *= g_last

                T.copy(v_new_fragment, v_new_shared)
                T.copy(k[i_b, i_t * BT : (i_t + 1) * BT, i_h, 0:K], k_shared)
                T.gemm(k_shared, v_new_shared, state, transpose_A=True)
                T.copy(state, state_shared)

    return kernel


def _gdn_fwd(q, k, v, g, beta, scale):
    g = chunk_local_cumsum(g, chunk_size=64, scale=RCP_LN2)
    w, u, A = chunk_gated_delta_rule_fwd_intra(k=k, v=v, g=g, beta=beta)
    h, v_new = _chunk_state(q.shape[0], q.shape[1])(k, w, u, g)
    o = chunk_fwd_o(q=q, k=k, v=v_new, h=h, g=g, scale=scale, state_v_first=True)
    return g, o, A, w, h, v_new


class _GDNFunction(torch.autograd.Function):

    @staticmethod
    def forward(ctx, q, k, v, g, beta, scale):
        q, q_rstd = l2norm_fwd(q)
        k, k_rstd = l2norm_fwd(k)
        g, o, A, w, h, v_new = _gdn_fwd(q, k, v, g, beta, scale)
        ctx.save_for_backward(
            q, q_rstd, k, k_rstd, v, g, beta, A, w, h, v_new
        )
        ctx.scale = scale
        return o.to(q.dtype)

    @staticmethod
    def backward(ctx, do):
        q, q_rstd, k, k_rstd, v, g, beta, A, w, h, v_new = ctx.saved_tensors
        dv = chunk_bwd_dv_local(q=q, k=k, g=g, do=do, scale=ctx.scale)
        dh, _, dv = chunk_gated_delta_rule_bwd_dhu(
            q=q, k=k, w=w, g=g, h0=None, dht=None, do=do, dv=dv,
            scale=ctx.scale, state_v_first=True,
        )
        dq, dk, dw, dg = chunk_bwd_dqkwg(
            q=q, k=k, v=v_new, w=w, g=g, h=h, dv=dv, do=do, dh=dh,
            scale=ctx.scale, state_v_first=True,
        )
        dk2, dv, db, dg2 = prepare_wy_repr_bwd(
            k=k, v=v, beta=beta, g=g, A=A, dw=dw, du=dv,
        )
        dk.add_(dk2)
        dg.add_(dg2)
        dg = chunk_local_cumsum(dg, chunk_size=64, reverse=True)
        dq = l2norm_bwd(q, q_rstd, dq)
        dk = l2norm_bwd(k, k_rstd, dk)
        return dq.to(q), dk.to(k), dv.to(v), dg.to(g), db.to(beta), None


def gdn_chunk_scan(q, k, v, g, beta, scale):
    if q.requires_grad or k.requires_grad or v.requires_grad or beta.requires_grad:
        return _GDNFunction.apply(q, k, v, g, beta, scale)

    q_norm, _ = l2norm_fwd(q)
    k_norm, _ = l2norm_fwd(k)
    _, o, _, _, _, _ = _gdn_fwd(q_norm, k_norm, v, g, beta, scale)
    return o.to(q.dtype)
