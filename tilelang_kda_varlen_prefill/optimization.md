# 优化说明

实现使用纯 TileLang `sm_89` kernel，将 varlen prefill 按 64-token chunk
分解为 chunk 内因果计算、chunk 状态变换、有序状态递推和输出重建。可并行的
工作按 chunk、head 和 value tile 展开；序列内不可避免的 causal dependency
仅保留在紧凑的 chunk-level state recurrence 中。

主要优化如下：

- 使用分段 scratch layout，在评测器提供的 128 MiB workspace 内保存 chunk
  operator 和中间状态。
- diagonal/inter-chunk 路径使用 BF16 Tensor Core 矩阵运算，gate prefix、norm
  和关键状态计算保留 FP32。
- transform 和 transformed-output kernel 使用 256 threads，提高加载、逐元素
  变换与输出阶段的并行度。
- 对 `B > 1` 的 varlen shape，Q 和 K 顺序复用同一个 shared-memory staging
  buffer，将 transform shared memory 从 66,048 B 降到 49,664 B，使 RTX 4090
  上能够同时驻留两个 256-thread CTA。
- 对 `B = 1` 的长序列 shape 保留 Q/K 并行 staging，避免在无法获得双 CTA
  收益时引入额外的加载等待。

最终实现没有 CUDA/Triton fallback，没有额外依赖，也没有修改评测接口或数学
定义。

## 开源基线对比

当前可在 Ada GPU 上运行的强开源基线选用 FLA 0.5.2 的 Triton
`fla.ops.kda.chunk_kda`，源码固定在
`flash-linear-attention@7843b32`。`benchmark_fla_baseline.py` 显式设置
`FLA_DISABLE_BACKEND_DISPATCH=1`，因此不会隐式切换到 TileLang 或其他可选
backend；输出和 final state 到评测器 buffer 的 copy 也计入时间。FLA 基线在独立
Conda 环境 `fla_kda_varlen_prefill` 中运行，使用 PyTorch 2.7.1、CUDA 12.8 和
Triton 3.3.1。TileLang 实现在 PyTorch 2.6.0、CUDA 12.4、TileLang 0.1.12
环境中运行。两者都绑定到同一张 RTX 4090，并使用官方 cold-L2、多 buffer、
4 seeds 测量口径。

| 官方用例 | 本实现 (us) | FLA 0.5.2 Triton (us) | 加速比 |
| --- | ---: | ---: | ---: |
| K1 balanced short | 694.755 | 1094.096 | 1.575x |
| K2 variable lengths | 1303.642 | 1610.635 | 1.235x |
| K3 single 16k | 1407.925 | 1798.775 | 1.278x |
| K4 one long, three short | 2953.546 | 3500.078 | 1.185x |
| 加权几何平均 | 1497.628 | 1934.318 | 1.292x |

FLA 基线的 basic correctness 全部通过：`out` 最大 NRMSE 为 0.004655，
`final_state` 最大 NRMSE 为 0.003662。当前更新的
[FlashKDA](https://github.com/MoonshotAI/FlashKDA) 是值得参考的更新实现，但其
官方要求 SM90+ 和 CUDA 12.9+；指定的 RTX 4090 是 SM89，因此本轮没有通过
兼容分支或 fallback 伪造 FlashKDA 实测结果。

## K4 profiling 与停止条件

K4 的一次 cold-L2 Nsight Systems trace 共 20 次 kernel launch，约 3.114 ms：

- `chunk_inter` 约 1.013 ms，是最大的单个 kernel；
- 6 轮 state scan 合计约 1.031 ms；
- 6 轮 transform 和 output 分别合计约 0.366 ms 和 0.250 ms；
- `chunk_diagonal` 约 0.440 ms。

对 `chunk_inter` 单独采集 NCU 后，其 dynamic shared memory 为 90.75 KiB，
寄存器为 255/thread，无 spill；shared-memory 限制每个 SM 只能驻留一个
64-thread CTA，achieved occupancy 为 4.17%。若要驻留两个 CTA，shared memory
必须降到约 50 KiB 以下，而 Q/K/G 三个贯穿 block-pair 计算的 staging buffer
本身已经占用 64 KiB，需要拆 kernel 或重排完整算法，不能靠生命周期别名安全
实现。

本轮还验证了将 `chunk_inter` 从 64 threads 提到 128 threads 的直接候选。
TileLang 的 16x16 MMA 布局无法把该 kernel 映射到 4 warps，编译期即因
`m_warp * n_warp != num_warps` 拒绝。结合此前多轮 chunk/state/transform/output
结构优化，以及当前实现已在四个 shape 全部超过可运行的 FLA 0.5.2 基线，继续
优化需要改写 inter-chunk 算法，当前版本在此停止。

## 收尾静态审查

收尾审查发现 K1 的固定数据只有 64 个实际 chunk，而通用上界为 95，后者会令
workspace 分成两轮。没有保留仅按 `(T=4096, B=32, H=16)` 假设每条序列长度
均为 128 的候选：题目声明域允许 `T/B/H` 范围内任意严格递增的运行时 GPU
`cu_seqlens`，所以相同静态 shape 不能唯一确定长度分布。为固定性能点牺牲该
输入合同不是合法优化。

另实现并撤销了一个保持任意 varlen 正确性的候选：当 `B>=16` 时，只让 state
scan CTA 的 thread 0 计算该 sequence/round 的 chunk prefix，再通过 shared memory
广播给其余线程。构造的 `B=32, H=16, lengths=1..32` 用例中，`out` 和
`final_state` 均完全通过，NRMSE 分别为 0.004224 和 0.004152。但同一 GPU 6、
NUMA 1、official cold-L2/4-buffer/4-seed 口径下，K1 从 694.860 us 回退到
699.517 us（0.67%）；thread-0 串行段和额外 CTA barrier 大于删除重复短扫描的
收益。候选已完全撤销，最终源码保持不变。
