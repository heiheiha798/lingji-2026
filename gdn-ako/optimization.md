# 优化说明

## TileLang 状态扫描

FLA 0.5.1 的端到端 trace 显示，`chunk_gated_delta_rule_fwd_h` 在训练前向与
backward 重算中各执行一次，是主要热点。提交以 TileLang 0.1.12 官方 GDN 示例中的
状态扫描 kernel 为基线，针对本题固定的 `H=8, K=V=128, chunk=64, bf16/fp32`
形状做静态特化：每个 CTA 负责一个 head 的 32 列状态，使用 128 threads 和单 stage
pipeline，在寄存器中沿所有 chunk 保持 fp32 递归状态。

官方示例接收自然对数 gate，而 FLA 0.5.1 在进入状态扫描前已用 `RCP_LN2` 将
chunk cumsum 转成 `log2`。提交因此直接对 gate 使用 `exp2`，避免重复乘
`log2(e)`；组件对照中 `h` 与 `v_new` 均和 FLA 逐元素一致。kernel 还直接支持非
64 整除的尾块，不存在调用 FLA 的条件 fallback。

状态以 value-major `[V,K]` 布局写回，后续 FLA 前向和反向 kernels 均使用
`state_v_first=True`。自定义 autograd 在训练前向和 backward 状态重算中复用同一个
TileLang kernel，其余 WY、输出、反向与 L2Norm 组件继续使用 FLA 0.5.1 的原生实现。

## 验证结果

在 RTX 4090 物理 GPU 6、NUMA 1 上，以 FLA 0.5.1 / Torch 2.6.0 / Triton 3.2.0
运行公开评测器的刷新输入、pool 轮换和 `20 + 5x50` 计时口径：随机正确性 5/5，
固定形状前向与反向 8/8 全部通过，前向最大相对 L2 误差约 `4.08e-3`，相对 FLA
的反向梯度误差为 0。加权性能指数从旧实现的 1.046 提升到 **1.122**。

| `(B,T)` | 前向加速 | 前向+反向加速 |
|---|---:|---:|
| `(1,2048)` | 1.290x | 1.136x |
| `(2,2048)` | 1.160x | 1.126x |
| `(1,8192)` | 1.234x | 1.126x |
| `(2,8192)` | 1.089x | 1.137x |
| `(1,16384)` | 1.131x | 1.130x |
| `(2,16384)` | 1.076x | 1.135x |
| `(1,32768)` | 1.150x | 1.053x |
| `(4,8192)` | 0.998x | 1.134x |

已排除的候选包括：强制 FLA 的 TileLang 反向 backend（核心 kernel `533 us`，慢于
Triton 的 `448 us`）、双 stage pipeline（Ada 动态共享内存超限）、256 threads，
以及 `BV=64`；后三者均比最终的 128-thread、`BV=32` 配置慢。

## 额外依赖

`tilelang==0.1.12`，安装方式见 `requirements.txt`。
