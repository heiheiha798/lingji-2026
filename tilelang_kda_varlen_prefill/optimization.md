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
