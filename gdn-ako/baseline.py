"""性能基线：fla 官方 chunk_gated_delta_rule（bf16）。选手要在正确性等价前提下超过它。"""

from config import HEAD_K

SCALE = HEAD_K ** -0.5


def baseline_chunk(q, k, v, g, beta, scale=SCALE):
    """fla 官方 chunked GDN 前向（可微）。返回 output [B,T,Hv,V]。"""
    from fla.ops.gated_delta_rule import chunk_gated_delta_rule
    o, _ = chunk_gated_delta_rule(
        q=q, k=k, v=v, g=g, beta=beta,
        scale=scale, initial_state=None, output_final_state=False,
        use_qk_l2norm_in_kernel=True,
    )
    return o
