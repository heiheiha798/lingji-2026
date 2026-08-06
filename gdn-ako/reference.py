"""参考实现（golden）与数据生成。两条 golden：
  (A) reference_output：fla fused_recurrent(fp32)，评测默认用——recurrent 是 chunk 的数学参考。
  (B) naive_recurrent_reference：自包含 fp32 递归，不依赖 fla，用于校准 (A) + 干净环境防篡改抽查。
真机第一次跑 `python reference.py --calibrate` 核对 (A)≈(B)（安装/校准详见 INSTALL.md）。
"""

import math
import torch
import torch.nn.functional as F

from config import (NUM_HEADS, NUM_V_HEADS, HEAD_K, HEAD_V, IO_DTYPE)

SCALE = HEAD_K ** -0.5


def make_inputs(B, T, device="cuda", dtype=IO_DTYPE, seed=0, requires_grad=False):
    """生成一组合理的 GDN 输入。q/k 会在 kernel 内做 L2 norm，这里给原始随机值。"""
    g = torch.Generator(device=device).manual_seed(seed)
    q = torch.randn(B, T, NUM_HEADS,   HEAD_K, device=device, dtype=dtype, generator=g)
    k = torch.randn(B, T, NUM_HEADS,   HEAD_K, device=device, dtype=dtype, generator=g)
    v = torch.randn(B, T, NUM_V_HEADS, HEAD_V, device=device, dtype=dtype, generator=g)
    # g_gate: log-decay ∈ (-inf, 0]，exp(g) ∈ (0,1)，多数接近 1（长记忆）
    g_gate = F.logsigmoid(torch.randn(B, T, NUM_HEADS, device=device, dtype=torch.float32, generator=g))
    # beta: 更新强度 ∈ (0,1)
    beta = torch.sigmoid(torch.randn(B, T, NUM_HEADS, device=device, dtype=torch.float32, generator=g))
    tensors = {"q": q, "k": k, "v": v, "g": g_gate, "beta": beta}
    if requires_grad:
        for name in ("q", "k", "v"):
            tensors[name].requires_grad_(True)
        tensors["beta"].requires_grad_(True)
    return tensors


def reference_output(inp):
    """(A) 权威 golden：fla fused_recurrent 在 fp32 下。返回 fp32 output [B,T,Hv,V]。"""
    from fla.ops.gated_delta_rule import fused_recurrent_gated_delta_rule
    o, _ = fused_recurrent_gated_delta_rule(
        q=inp["q"].float(), k=inp["k"].float(), v=inp["v"].float(),
        g=inp["g"].float(), beta=inp["beta"].float(),
        scale=SCALE, initial_state=None, output_final_state=False,
        use_qk_l2norm_in_kernel=True,
    )
    return o.float()


@torch.no_grad()
def naive_recurrent_reference(inp):
    """(B) 自包含 fp32 递归，无 fla 依赖。O(T) 循环，仅用于小形状校准/兜底。

    每 (b, head)，状态 S ∈ R^{K×V}：
        qt, kt = l2norm(q_t), l2norm(k_t)
        a = exp(g_t)                       # per-head 标量衰减
        S = a * S                          # 门控衰减
        pred = kt @ S                      # [V]  从衰减后的状态预测
        S = S + beta_t * outer(kt, v_t - pred)     # delta-rule 秩一更新
        o_t = scale * (qt @ S)             # [V]
    """
    q, k, v = inp["q"].float(), inp["k"].float(), inp["v"].float()
    g, beta = inp["g"].float(), inp["beta"].float()
    B, T, H, K = q.shape
    V = v.shape[-1]
    q = F.normalize(q, dim=-1)
    k = F.normalize(k, dim=-1)
    S = torch.zeros(B, H, K, V, device=q.device, dtype=torch.float32)
    out = torch.empty(B, T, H, V, device=q.device, dtype=torch.float32)
    for t in range(T):
        a = g[:, t].exp().unsqueeze(-1).unsqueeze(-1)          # [B,H,1,1]
        kt = k[:, t].unsqueeze(-1)                             # [B,H,K,1]
        vt = v[:, t].unsqueeze(-2)                             # [B,H,1,V]
        bt = beta[:, t].unsqueeze(-1).unsqueeze(-1)            # [B,H,1,1]
        S = a * S
        pred = (kt * S).sum(dim=-2, keepdim=True)              # [B,H,1,V]
        S = S + bt * kt * (vt - pred)                         # [B,H,K,V]
        qt = q[:, t].unsqueeze(-1)                             # [B,H,K,1]
        out[:, t] = SCALE * (qt * S).sum(dim=-2)              # [B,H,V]
    return out


def _rel_l2(a, b):
    return (a - b).norm() / (b.norm() + 1e-12)


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--calibrate", action="store_true")
    args = ap.parse_args()
    if args.calibrate:
        # 小形状上校准 (A) fla-recurrent vs (B) naive-recurrent（均 fp32）
        for (B, T) in [(1, 256), (2, 512)]:
            inp = make_inputs(B, T, seed=1)
            a = reference_output(inp)
            b = naive_recurrent_reference(inp)
            err = _rel_l2(a, b).item()
            print(f"[calib] B={B} T={T}  rel_l2(fla_recurrent, naive) = {err:.3e}"
                  f"  {'OK' if err < 1e-3 else 'MISMATCH -> 调整 naive 递归顺序'}")
