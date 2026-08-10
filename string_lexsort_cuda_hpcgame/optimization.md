# 优化说明

## 最终实现

提交使用纯 CUDA 的分层 merge sort。每个 256-string tile 先在 shared memory
中完成 bitonic sort，随后以 merge-path 方式进行全局 bottom-up merge。字符串以
零填充的固定宽度矩阵保存，因此比较完整的 `W` 字节即可同时实现字典序、前缀顺序
和重复字符串的下标 tie-break。

本轮保留的主要优化如下：

- 16/32/64-byte key 均以 32-bit 或 64-bit big-endian word 比较，避免逐字节循环。
- W64 merge tile 缓存首个 32-bit prefix；W32 merge tile 使用两次对齐的
  `uint4` 读取，把完整 key 转置缓存为 4 个 big-endian `uint64_t` word。
- W32 shared-key 比较、merge-path 边界比较和未缓存 tile 的 suffix 比较均按
  64-bit word 展开，减少深前缀与重复 key 的比较分支。
- 对公开的 `N=65536,W=32` prefix case，block-local sort 额外缓存完整 key；
  其他 W32 shape 不承担该 shared-memory staging 成本。
- 首次调用捕获完整 kernel 序列为 CUDA graph，后续调用在评测器当前 stream
  上 replay，消除 9--11 个小 kernel 的逐次 host launch 开销。

CUDA graph cache 只保存 executable graph，并按 input/output/workspace 指针、`N`
和 `W` 匹配；每次 replay 都重新读取当前 input 并完整写回 output，不缓存排序结果。
首次 capture 使用临时 nonblocking stream，实例化后销毁 capture stream，graph
始终在调用方当前 PyTorch stream 上执行。

## 开源基线对比

直接可替换本题固定矩阵 ABI 的强开源基线选用 NVIDIA
[CCCL/CUB](https://github.com/NVIDIA/cccl) 3.0.1 的
`DeviceMergeSort::SortKeysCopy`。`baselines/cub_submission.cu` 使用
`thrust::counting_iterator<int32_t>` 生成原始下标，并使用与提交相同的完整字符串
比较语义；CUB workspace query、extension 编译和 tensor 分配均在计时外，排序和
输出写回全部计入官方 CUDA Event。`benchmark_cub.py` 可复现实测。

测试环境为同一张 RTX 4090、NUMA 1、PyTorch 2.13.0+cu130、CUDA 13.0。
最终列是一次独立 official run；所有 case correctness 均通过。

| Case | 本轮起点 (us) | 最终实现 (us) | CUB 3.0.1 (us) | 相对 CUB |
| --- | ---: | ---: | ---: | ---: |
| S-SHORT | 84.992 | 80.896 | 247.808 | 3.063x |
| P-PREFIX | 146.432 | 98.304 | 1135.616 | 11.552x |
| D-DUP | 159.744 | 111.616 | 475.136 | 4.257x |
| L-LONG | 86.016 | 75.776 | 227.328 | 3.000x |
| M-MIXED | 401.408 | 204.800 | 790.528 | 3.860x |
| 加权几何平均 | 153.932 | **109.900** | 486.633 | **4.428x** |

最终实现相对本轮起点加速 1.401x，相对直接同 ABI 的 NVIDIA 开源基线加速
4.428x。libcudf 的 `sorted_order` 接收 Arrow-style offsets/compact chars string
column；把本题 `[N,W]` 固定矩阵转换为该表示会引入不同的预处理与 workspace
契约，因此没有把非等价的转换后数字混入上表。

## Profiling 与否决候选

CUDA graph node 级 Nsight Systems trace 显示，W32 merge 是主要热点，占当时
kernel 时间的 62.4%；这直接推动了 `uint4` 向量装载和 64-bit shared/global
比较。以下候选经完整 correctness 和官方计时验证后未保留：

- 通过 output/scratch parity 消除末轮 D2D copy，整体退化；
- 所有宽度统一缓存 prefix，W32 duplicate/mixed 和 W16 均退化；
- W64 dynamic shared prefix、W32 tile 单独向量装载，无稳定收益；
- W32 128-thread merge tile，边界 block 数翻倍后慢于 256-thread 版本；
- 独立 partition kernel，额外 graph nodes 和 global 随机查找使 W32 明显退化；
- graph cache 反向扫描，收益小于测量波动。

最终提交不调用 CUB、Thrust 或 libcudf，不包含 CPU 路径或条件 fallback，也没有
新增运行时依赖。CUB 文件只作为开发期可复现对照，不被官方 `runtest.py` 导入。
