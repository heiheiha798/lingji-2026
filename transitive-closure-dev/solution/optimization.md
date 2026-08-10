# 优化说明

## 当前实现

实现首先做轻量候选筛选，再由 GPU 对结构性质做完整验证。所有分派都基于已经
验证的图性质或设备能力；CUDA API、kernel launch 或 registration 失败时直接返回
错误码 2，不会静默切换到另一条兜底路径。

### 通用 blocked Warshall

任意图使用 64 位 bitset 的 blocked Warshall，并严格保留 pivot 顺序。每轮包含：

1. `ClosePivotBlock` 在一个 CTA 内求 pivot 子块的完整闭包。
2. `BuildPivotRows` 将块内闭包展开成已合并的 pivot 行。
3. `ApplyPivotBlock` 将整块 pivot 应用到所有图行。

`words<=512` 使用 256-pivot / 4-mask 路径，减少全矩阵 Apply 次数；更大矩阵
使用独立编译的 128-pivot / 2-mask 路径，避免 P5 上更大的 working set 和寄存器
开销。两条路径共享二维 `BuildPivotRows`。

Apply 还使用以下等价变换减少实际工作：

- 依据 `pivot_masks` 贪心压缩生成元；一条已闭包的 pivot 行能够一次删除它包含
  的冗余 pivot。
- 单生成元和单位 pivot block 使用独立主路径。
- `words<=256` 时为空 pivot mask 的整 CTA 早退。
- 只有结果发生变化时才写回，避免稠密闭包后期的无效 store。

`BuildPivotRows` 将 word 维拆成二维 grid。P3 的 512 words 由每条 pivot 行的
4 个 CTA 并行处理，256-pivot 时每轮共有 1024 个 CTA，避免只有 128 个 row CTA
造成的低 occupancy。`words<=512` 的 pivot row buffer 使用 persisting-L2
access-policy window。

### 严格有序 DAG

CPU 只扫描相邻后继和前 1024 行，用来排除明显不适合的候选。GPU
`AnalyzeDagCandidate` 随后扫描每一行，精确验证所有边都指向更大的顶点编号，
同时产生 degree 和最小目标描述符、写入反身对角位并清理 tail bits。

通过验证后，host 根据最小目标构造反向 dependency batches。同一 batch 内的行
互不依赖，可以并行闭包。batch 数量和 cooperative residency 允许时，所有 batch
由单个 `CloseUpperTriangularDagCooperative` persistent kernel 完成；否则依据精确
的 batch 规模和设备 cooperative-launch 能力选择 multi-launch 或单 CTA 算法。
这属于能力分派，不是 launch 失败后的 fallback。

### 64-row block-upper 图

GPU 分析还独立验证一条较弱但精确的性质：任意边都不会从当前 64-row block
指向更早的 block。只有该性质成立且每行 forward degree 不超过 8 时，才进入
block-upper 专用闭包。

`CloseOrderedDiagonalBlocks` 先闭包所有 64x64 对角块，并为闭包行完全相同的
顶点生成 representative。随后 cooperative persistent grid 按 block 逆序执行：

- 非完整 block 只计算 representative 行，后继读取也映射到 representative。
- 完整 SCC block 的 64 行相同，64 个 row CTA 分摊直接后继闭包，最后只归并
  一条代表行。
- 每个 block 只保留两次必要的 grid barrier；上一 block 完成后才处理依赖它的
  更早 block。
- 非 representative 行延迟到所有 block 闭包完成后再 materialize。

对至少 128 MiB、顶点数为 64 的倍数、且完整 SCC block 不少于四分之一的输出，
实现进一步跳过这些完整 block 的 GPU 重复行 materialization。D2H 只下载每个
完整 block 的代表行和所有非完整 block，host 再以固定、有限的 8 个标准 C++
worker 展开相同行。该路径由对角闭包产生的精确 block 标记驱动，不依赖 CPU
型号、NUMA 拓扑或本机 core 数。其他 block-upper 图仍在 GPU 完整 materialize
并执行一次连续 D2H。

传输范围达到 128 MiB 时，输入和输出 host range 使用 `cudaHostRegister`；较小
范围保持 pageable copy。registration 失败直接返回错误，不存在隐藏 fallback。

## 开源基线

优先核对了公开实现中语义相同的 Datalog/transitive-closure workload：

- MNMGDatalog，ICS 2025，commit `fb8d497e180b16a05ceada4c09a3e80991da18c9`
  - https://github.com/harp-lab/MNMGDatalog
  - https://thomas.gilray.org/pdf/multi-node-gpu-datalog.pdf
- Multi-GPU Datalog，USENIX ATC 2023
  - https://github.com/harp-lab/usenixATC23
  - https://www.usenix.org/conference/atc23/presentation/shovon

MNMGDatalog 使用其官方 CUDA 12.8 / OpenMPI 构建选项和相同边集；成功完成的
case 还校验了最终闭包 popcount。MNMG 的内部计时与本实现包含 context、传输、
kernel 和资源释放的 cold-process 计时对比如下，因此比较口径对本实现更保守。

| case | 本实现中位数 (ms) | MNMGDatalog (ms) | 结果 |
|---|---:|---:|---:|
| P1 | 242.353 | 828.2 | 3.42x |
| P2 | 279.656 | 39874.8 | 142.59x |
| P3 | 310.641 | 运行 4.45 s 后报 `cudaErrorInvalidDevice` | 本实现 PASS |
| P4 | 253.664 | 862.3 | 3.40x |
| P5 | 423.219 | 57465.0 | 135.78x |

ATC 2023 实现在已跑的 P1/P2/P4 上分别为 1769.055、75014.671、1619.962 ms，
本实现对应快 7.30x、268.24x、6.39x。

GraphBLAST 和 Bit-GraphBLAS 仓库中的 `TC` 是 triangle counting，不是
transitive closure；Hornet 的对应实现仍为 TODO；ECL-APSP 是带权 `int` 稠密
APSP，P5 存储规模也不适用，因此没有把这些不同语义 workload 混入表格。

## 性能结果

环境为 CUDA 12.8、sm_89、物理 GPU 7。每个样本是独立 judge 进程，包含 cold
CUDA context、host/device transfer、kernel 和资源释放。最终正式报告为 3 次
样本中位数：

| case | 初始合法 baseline (ms) | 最终 (ms) | 加速 |
|---|---:|---:|---:|
| P1 layered-dag | 369.442 | 242.353 | 1.52x |
| P2 block-scc | 645.194 | 279.656 | 2.31x |
| P3 random-sparse | 804.665 | 310.641 | 2.59x |
| P4 grid-dag | 336.064 | 253.664 | 1.33x |
| P5 large-mixed | 2059.116 | 423.219 | 4.86x |

P0 至 P5 的最终中位数为 247.235、242.353、279.656、310.641、253.664、
423.219 ms。P0-P5 均与 trusted reference 逐字节一致，固定 seed
`8091548694775575726` 的随机五类测试也全部通过。

## Profiling 证据

通用 blocked 路径的 P3 中，256-pivot 后每类 kernel 各 128 次：Apply、Build、
Close 累计分别为 21.692、5.901、3.631 ms。此前 64-pivot checkpoint 的 kernel
总时间约 98.69 ms，当前约 31.22 ms。中段 Apply 的 NCU 指标为 85.48% DRAM
throughput、91.12% achieved occupancy；平均 72.8 cycles 等待 L1TEX scoreboard，
说明该阶段已经主要受实际 global-memory latency/bandwidth 限制。

P3 中段 Build 的二维 grid 有 1024 blocks，L2 hit 98.97%，achieved occupancy
66.00%。二维化前只有 128 blocks、8.37% occupancy；对应 NCU duration 从
150.50 us 降到 41.06 us。256/128-pivot Apply 分别使用 32/24 registers，Build
使用 20，所有 kernel 均无 local spill。

结构专用路径的 NSYS 变化如下：

- P1 的 DAG 分析和 cooperative closure GPU 总时间约 0.25 ms。
- P5 原 blocked 路径 kernel 总时间约 565.67 ms，其中 Apply 为 551.48 ms。
- block-upper persistent 路径将 P5 cooperative kernel 降到 10.190 ms；压缩完整
  block 输出后进一步降到约 6.22 ms。
- P5 D2H 从 41.407 ms 降到 21.871 ms；8-worker host 展开约 14.4 ms，GPU 与
  PCIe 的确定性节省仍大于展开成本。
- P2 cooperative kernel 从 3.625 ms 降到 2.172 ms，D2H 从 10.345 ms 降到
  0.191 ms，因为所有 64-row block 都只需下载一条代表行。

完整 SCC 并行化前的 P5 persistent NCU duration 为 27.37 ms，barrier stall
80.7%；64 个 row CTA 分摊后为 12.46 ms，barrier stall 降到 71.26%，L2 hit
81.82%。之后减少 barrier 和压缩 materialization 得到上述 6.22 ms NSYS 结果。

pageable P3 的 134.218 MB H2D/D2H 分别约 27.65/27.53 ms；host registration
后降到 11.17/10.40 ms。32 MiB 的 P1/P4 保持 pageable 路径。

## 已否决方向

以下候选均经过 correctness、cold timing、NSYS 或 NCU 检查后撤销：

- 双 word/vectorized Apply、64-thread Apply、shared selected mask、强制 MLP、
  streaming cache load/store，以及多种 shared/tiled Build。
- 全行饱和计数；统计和 34-register 开销大于跳过收益。
- P5 无条件使用 256/512-pivot；working set 和寄存器开销造成稳定回退。
- SCC representative 查询塞入通用 Apply；额外间接读取抵消跳过收益。
- 合并为单次 device allocation，以及只为 DAG 缩减 allocation/init/L2 setup；
  cold-process 结果中性或负向。
- 将 DAG 邻接表打包到 GPU scratch；Analyze 从约 22.98 us 增至 31.39 us，
  cooperative closure 也从约 222.08 us 增至 242.82 us。
- 把 CPU 相邻后继扫描与 pinned H2D 重叠；P2 A/B 中位数回退约 9.39 ms，原因是
  CPU 稀疏读取与 DMA 竞争 host memory bandwidth。
- 并行执行两个 `cudaHostRegister`；driver 内部争用使 P5 registration wall time
  从约 51.6 ms 增至约 66 ms。
- `cudaHostRegisterReadOnly`、只 pin 输入、以及对 32 MiB 小矩阵做 registration。
- 压缩输出后的单线程逐行或倍增 materialization；host copy 成本大于 GPU/PCIe
  节省，最终改为固定的有限并行。
- 根据本机 socket、NUMA 或 core 数把 host worker 调到 16/24；cold A/B 负向且
  不具备跨评测 CPU 的可移植性，未保留。

## 额外依赖

无。host 并行仅使用 C++17 标准库 `std::thread`。
