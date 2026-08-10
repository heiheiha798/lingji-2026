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


def gdn_chunk_scan(q, k, v, g, beta, scale):
    # 默认实现 = 直接调 fla（即基线）。请替换成你的 kernel。
    from fla.ops.gated_delta_rule import chunk_gated_delta_rule
    o, _ = chunk_gated_delta_rule(
        q=q, k=k, v=v, g=g, beta=beta,
        scale=scale, initial_state=None, output_final_state=False,
        use_qk_l2norm_in_kernel=True,
        state_v_first=True,
    )
    return o
