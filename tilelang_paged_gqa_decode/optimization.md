# 优化说明

实现使用一个 KV head 对应一个 16-row MMA tile，同时计算该 KV head
对应的 4 或 8 个 Query head。Query 在 CTA 内复用，K/V 按 `block_table`
映射后的物理页直接读入 shared memory；每页分别通过 BF16 Tensor Core
完成 `QK^T` 和 `PV`，Softmax 最大值、归一化因子与输出累加使用 FP32。

短序列和大 batch 使用单 kernel 在线 Softmax。长序列按静态 workload
切分连续 KV 页，每个 split 独立生成 FP32 partial output 和 log-sum-exp，
再由单 warp TileLang kernel 做数值稳定的全局合并。公开用例 G1、G2、G3
分别使用 16、16、8 个 split；G4 不切分。该结构把长请求的页循环分散到
足够多的 CTA，同时避免大 batch 短请求承担 workspace 写入和第二次 launch。

RTX 4090 物理 GPU 6 上，开发阶段 cold-L2 local preset 的公开用例加权
延迟从未切分版本的 761.640 us 降至 287.615 us。最终 official preset
测得 G1/G2/G3/G4 分别为 130.554、286.018、428.925、342.230 us，
加权几何平均为 286.190 us。固定 7 组 correctness、固定 seed 的 5 组
随机 correctness，以及 4 组公开大 shape correctness 均通过；观测到的
最大绝对误差为 7.812e-3。

额外依赖：无。
