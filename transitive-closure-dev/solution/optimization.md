# 优化说明

## 当前实现

实现采用 64 位 bitset 的 blocked Warshall，并保留严格的 pivot 顺序。每轮依次
执行三个阶段：

1. `ClosePivotBlock` 在一个 CTA 内求 pivot 子块的完整闭包。
2. `BuildPivotRows` 将块内闭包展开成已经合并好的 pivot 行。
3. `ApplyPivotBlock` 将整块 pivot 一次应用到所有图行。

`words<=512` 时使用 256-pivot / 4-mask 路径，以减少全矩阵 Apply 次数；更大
矩阵使用独立编译的 128-pivot / 2-mask Close 和 Apply，避免更大的 pivot
working set 和额外寄存器拖慢 P5。两条路径共享二维 `BuildPivotRows`。

Apply 使用以下等价变换减少实际工作：

- 依据块内 `pivot_masks` 贪心压缩生成元。一条闭包 pivot 行已经包含其块内
  可达点的全部出边，因此可从 remaining mask 一次删除这些冗余 pivot。
- 单生成元使用独立主路径，不再进入通用多生成元循环。
- 单位 pivot block 直接令 selected mask 等于行的 pivot mask，跳过每个 warp
  重复执行的生成元压缩。
- `words<=256` 时为空 pivot mask 的行整 CTA 早退；大矩阵不编译该分支。
- 只有结果变化时才写回 reachability，避免稠密闭包后期的无效 store。

`BuildPivotRows` 将 word 维拆为二维 grid。P3 的 512 words 由每条 pivot 行的
4 个 CTA 并行处理；256-pivot 时一轮共有 1024 个 CTA，不再受“一张卡只有
128 个 row CTA”的低 occupancy 限制。

对 `words<=512` 的 pivot row buffer 设置 persisting-L2 access-policy window。
传输范围达到 128 MiB 时，输入和输出 host range 使用 `cudaHostRegister`；小于
该阈值时 registration 的固定成本不稳定，因此保持 pageable copy。注册失败
直接返回错误，没有隐藏的 fallback 路径。

## 开源基线

优先核对了公开实现中语义相同的 Datalog/transitive-closure workload：

- MNMGDatalog，ICS 2025，commit `fb8d497e180b16a05ceada4c09a3e80991da18c9`
  - https://github.com/harp-lab/MNMGDatalog
  - https://thomas.gilray.org/pdf/multi-node-gpu-datalog.pdf
- Multi-GPU Datalog，USENIX ATC 2023
  - https://github.com/harp-lab/usenixATC23
  - https://www.usenix.org/conference/atc23/presentation/shovon

MNMGDatalog 使用其官方 CUDA 12.8 / OpenMPI 构建选项，并使用相同边集；成功
完成的 case 还校验了最终闭包 popcount。MNMG 的内部计时与本实现包含 context、
传输和 kernel 的 `closure_run` cold-process 计时对比如下；这是对本实现更保守的
比较。

| case | 本实现中位数 (ms) | MNMGDatalog (ms) | 结果 |
|---|---:|---:|---:|
| P1 | 286.906 | 828.2 | 2.89x |
| P2 | 306.832 | 39874.8 | 129.96x |
| P3 | 323.967 | 运行 4.45 s 后报 `cudaErrorInvalidDevice` | 本实现 PASS |
| P4 | 254.745 | 862.3 | 3.38x |
| P5 | 979.121 | 57465.0 | 58.69x |

ATC 2023 实现在已跑的 P1/P2/P4 上分别为 1769.055、75014.671、1619.962 ms，
本实现对应快 6.17x、244.48x、6.36x。

GraphBLAST 和 Bit-GraphBLAS 仓库中的 `TC` 是 triangle counting，不是
transitive closure；Hornet 的对应实现仍为 TODO；ECL-APSP 是带权 `int` 的
稠密 APSP，P5 存储规模也不适用，因此没有把这些不同语义 workload 混入表格。

## 性能结果

环境为 CUDA 12.8、sm_89、物理 GPU 7。每个样本是独立 judge 进程，因而包含
cold CUDA context、host/device transfer、kernel 和资源释放。最终 5 次样本
取中位数：

| case | 初始合法 baseline (ms) | 最终 (ms) | 加速 |
|---|---:|---:|---:|
| P1 layered-dag | 369.442 | 286.906 | 1.29x |
| P2 block-scc | 645.194 | 306.832 | 2.10x |
| P3 random-sparse | 804.665 | 323.967 | 2.48x |
| P4 grid-dag | 336.064 | 254.745 | 1.32x |
| P5 large-mixed | 2059.116 | 979.121 | 2.10x |

最终 P0 至 P5 的中位数为 251.781、286.906、306.832、323.967、254.745、
979.121 ms。固定 P0-P5 均与 trusted reference 逐字节一致，随机五类测试也
全部通过；最近一次随机 seed 为 `8091548694775575726`。

## Profiling 证据

P3 的 NSYS 汇总显示 256-pivot 后每类 kernel 各 128 次：

| kernel | 累计时间 (ms) | 占 kernel 时间 |
|---|---:|---:|
| ApplyPivotBlock | 21.692 | 69.5% |
| BuildPivotRows | 5.901 | 18.9% |
| ClosePivotBlock | 3.631 | 11.6% |

此前 64-pivot checkpoint 的 P3 kernel 总时间约 98.69 ms；当前为 31.22 ms。
其中 128-pivot 到 256-pivot 这一轮把 Apply 从 40.70 ms 降至 21.69 ms，Build
保持约 5.8-5.9 ms。

P3 中段 Apply 的 NCU 指标为：

- 192.42 us，DRAM throughput 85.48%，840.21 GB/s。
- L1/TEX hit 60.59%，L2 hit 25.01%。
- achieved occupancy 91.12%，每 scheduler 11.20 active warp，但仅 0.19
  eligible warp；平均 72.8 cycles 等待 L1TEX scoreboard dependency。

因此当前 Apply 已主要受实际 global-memory latency/bandwidth 限制。中段
Build 的 grid 为 1024 blocks，L2 hit 98.97%，achieved occupancy 66.00%。二维
grid 之前同一 kernel 只有 128 blocks、8.37% occupancy；二维版本在 128-pivot
阶段已将对应 NCU duration 从 150.50 us 降至 41.06 us。

PTX/SASS 与 `cuobjdump` 检查结果：256-pivot Apply 使用 32 registers，128-pivot
Apply 使用 24；Build 使用 20；宽/窄 Close 分别使用 26/24 registers 和
32/16 B shared memory。所有 kernel 均无 local spill，主路径保持合并的 64-bit
global load/store 和 warp shuffle。

pageable P3 的 134.218 MB H2D/D2H 分别约 27.65/27.53 ms；注册后降至
11.17/10.40 ms。P2/P3/P5 的交错 cold-process 对照均为正向，而 32 MiB 的
P1/P4 保持未注册路径。

## 已否决方向

以下候选均经过 correctness、cold timing 或 NCU 检查后撤销：

- 双 word/vectorized Apply：寄存器升至 32，P3 回退约 6.8%。
- 64-thread Apply、4 warp 各处理一行：eligible warp 和 long-scoreboard 恶化。
- shared-memory selected mask、两次迭代强制 MLP、streaming cache load/store。
- 多种 shared/tiled Build 版本：只改善部分稠密 case，整体负向。
- 全行饱和计数：P3 并非逐行完全全 1，计数和 34-register 开销大于跳过收益。
- 对 P5 无条件使用 256-pivot：5 对测试全部回退，已改为独立 128-pivot 路径。
- 对 32 MiB 小矩阵做 host registration：pin/unpin 固定成本造成不稳定回退。

## 额外依赖

无。
