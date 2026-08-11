# 优化说明

实现使用一个 KV head 对应一个 16-row MMA tile，同时计算该 KV head
对应的 4 或 8 个 Query head。Query 在 CTA 内复用，K/V 按 `block_table`
映射后的物理页直接读入 shared memory；每页分别通过 BF16 Tensor Core
完成 `QK^T` 和 `PV`，Softmax 最大值、归一化因子与输出累加使用 FP32。

短序列和大 batch 使用单 kernel 在线 Softmax。长序列按固定页块切分连续
KV 页，每个 split 生成 BF16 partial output 和 FP32 log-sum-exp，再由单
warp TileLang kernel 以 FP32 权重及累加做数值稳定的全局合并。相较原先
的 FP32 partial，workspace 中间向量的写入和读取流量减半。kernel 根据运行时
`seq_lens` 计算有效 split 数，combine 也只读取这些有效 partial，避免
短请求读取 workspace 中的旧数据。G1、G2 使用 16 个最大 split、每块
128 页；G3 使用 4 个最大 split、每块 256 页；G4 不切分。direct 和
split attention CTA 均使用 64 线程，combine 保持 32 线程。

RTX 4090 物理 GPU 6 上，开发阶段 cold-L2 local preset 的公开用例加权
延迟从未切分版本的 761.640 us 降至 281.123 us。最终 official preset
测得 G1/G2/G3/G4 分别为 131.673、275.871、410.200、339.481 us，
加权几何平均为 279.366 us；相较 FP32 partial 的 280.434 us 改善
0.38%。重新验证受影响的 G1/G2/G3 大 shape correctness，`nrmse` 不超过
3.250e-3，最大绝对误差不超过 4.883e-4；G4 的 direct 路径未改变。固定
7 组 direct correctness 观测到的最大绝对误差为 7.812e-3。

开发对照脚本 `benchmark_flashinfer.py` 使用 FlashInfer 0.6.16.post3，
把 plan 放在计时区外，并复用本题的 cold-L2 计时器。FlashInfer FA2 的
G1/G2/G3/G4 为 133.234、277.938、441.355、333.005 us，加权
285.781 us；CUDA-core 后端为 135.441、278.760、412.643、349.861 us，
加权 284.034 us。当前实现的加权延迟快于任一完整 FlashInfer 后端；若
逐 shape 选取两个后端的最小值，复合下界为 280.072 us，当前实现仍快
约 0.253%。

NCU 显示 G4 的当前 kernel 与 FlashInfer 都约为 92% DRAM 吞吐、16.67%
理论 occupancy。剩余差距主要来自当前静态整页 `T.copy` 读取每条请求的
无效尾页元素。TileLang 中尝试手写 predicated tail load 和动态 extent
`T.copy` 后，G4 local 分别退化到 341.313 us 和 344.069 us，因此保留
静态整页 copy。128-thread CTA、双 stage pipeline 和 K/V shared buffer
复用同样没有正向信号。

提交实现额外依赖：无。FlashInfer 仅用于开发期对照，不被 `submission.py`
导入。
