# 优化说明

## TileLang 状态扫描

FLA 0.5.1 的端到端 trace 显示，`chunk_gated_delta_rule_fwd_h` 在训练前向与
backward 重算中各执行一次，是主要热点。提交以 TileLang 0.1.12 官方 GDN 示例中的
状态扫描 kernel 为基线，针对本题固定的 `H=8, K=V=128, chunk=64, bf16/fp32`
形状做静态特化：每个 CTA 负责一个 head 的 16 列状态，使用 128 threads 和单 stage
pipeline，在寄存器中沿所有 chunk 保持 fp32 递归状态。相对 32 列基线，16 列分块将
`B=1` 的持久 CTA 数从 32 增至 64，缓解长序列扫描的低并行度。

官方示例接收自然对数 gate，而 FLA 0.5.1 在进入状态扫描前已用 `RCP_LN2` 将
chunk cumsum 转成 `log2`。提交因此直接对 gate 使用 `exp2`，避免重复乘
`log2(e)`；组件对照中 `h` 与 `v_new` 均和 FLA 逐元素一致。kernel 还直接支持非
64 整除的尾块，不存在调用 FLA 的条件 fallback。

状态以 value-major `[V,K]` 布局写回，后续 FLA 前向和反向 kernels 均使用
`state_v_first=True`。其余 WY、输出、反向与 L2Norm 组件继续使用 FLA 的原生实现。

## Backward 激活复用

训练前向已经产生 backward 所需的 `w`、`h` 和 `v_new`。旧实现只保存输入与
`A`，因此 backward 先调用 `recompute_w_u_fwd` 重算完整 `w/u`，再第二次执行
TileLang 状态扫描以重建 `h/v_new`。当前实现把前向的三个 bf16 张量直接加入
autograd context，backward 消费原值并删除两段重算；数学、浮点顺序、反向 kernel
和 tiling 均未改变，也没有增加条件 fallback。

最大固定 shape 的额外存活激活约为 256 MiB：`w` 64 MiB、`v_new` 64 MiB、
`h` 128 MiB。评测每次 forward 后立即 backward，完整 8-shape 运行未出现 OOM。

## 验证结果

在 RTX 4090 物理 GPU 6、NUMA 1 上，以 FLA 0.5.1 / Torch 2.6.0 / Triton 3.2.0
运行公开评测器的刷新输入、pool 轮换和 `20 + 5x50` 计时口径：随机正确性 5/5，
固定形状前向与反向 8/8 全部通过，前向最大相对 L2 误差约 `4.08e-3`，相对 FLA
的反向梯度误差为 0。加权性能指数从旧实现的 1.046、32 列实现的 1.122 提升到
**1.138**。

| `(B,T)` | 前向加速 | 前向+反向加速 |
|---|---:|---:|
| `(1,2048)` | 1.19x | 1.11x |
| `(2,2048)` | 1.21x | 1.12x |
| `(1,8192)` | 1.19x | 1.12x |
| `(2,8192)` | 1.09x | 1.12x |
| `(1,16384)` | 1.29x | 1.13x |
| `(2,16384)` | 1.09x | 1.11x |
| `(1,32768)` | 1.26x | 1.12x |
| `(4,8192)` | 0.99x | 1.12x |

## 当前开源基线复核

2026-08-11 又以 FLA 0.5.2 源码提交
`7843b328b0d3860a66de4eb07ba28bb020ceb1d8` 作为同机分母，使用同一 RTX 4090、
Torch 2.6.0、Triton 3.2.0、TileLang 0.1.12 和评测器完整
`20 warmup + 5x50` 口径重跑。为固定 Ada 上的原生 Triton 路径，运行时设置了
`FLA_DISABLE_BACKEND_DISPATCH=1`。当前 FLA 的 FlashQLA backend 只支持
SM90、SM100/SM103 和 SM120，不能在指定的 SM89 GPU 上运行；FLA 的 common
TileLang backend 在 Ada 默认关闭，且下文已有强制启用后 backward 更慢的实测。

加入 backward 激活复用后，8/8 前向与 8/8 反向全部通过，最大前向 relative L2
为 `4.08e-3`，反向梯度与 FLA 路径逐元素一致。加权性能指数由 **1.138** 提升到
**1.182**：

| `(B,T)` | 前向加速 | 前向+反向加速 |
|---|---:|---:|
| `(1,2048)` | 1.09x | 1.18x |
| `(2,2048)` | 1.20x | 1.18x |
| `(1,8192)` | 1.21x | 1.18x |
| `(2,8192)` | 1.11x | 1.18x |
| `(1,16384)` | 1.25x | 1.19x |
| `(2,16384)` | 1.09x | 1.19x |
| `(1,32768)` | 1.27x | 1.25x |
| `(4,8192)` | 0.99x | 1.19x |

`(4,8192)` 的纯前向路径略慢于 FLA，但按该 shape 条件切回 FLA 会引入 fallback；
最终提交保持单一 TileLang 状态扫描实现，不加入该分支。

已排除的候选包括：强制 FLA 的 TileLang 反向 backend（核心 kernel `533 us`，慢于
Triton 的 `448 us`）、双 stage pipeline（Ada 动态共享内存超限）、256 threads，
以及 `BV=64`。进一步全量测试中，`BV=8` 的性能指数为 1.129，按 shape 混用
`BV=16/32` 为 1.127，均低于最终的 128-thread、`BV=16` 配置。

## 额外依赖

`tilelang==0.1.12`，安装方式见 `requirements.txt`。
