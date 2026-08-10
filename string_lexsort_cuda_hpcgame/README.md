# CUDA 字符串字典序排序

| 项目 | 说明 |
|---|---|
| 赛题类型 | GPU 算子优化 |
| 目标设备 | NVIDIA GeForce RTX 4090（Ada / SM89 / 24 GiB） |
| 运行命令 | `python runtest.py` |
| 可修改文件 | `submission.cu` |
| 最高分 | 100 分 |
| 满分线 | `WeightedGpuTimeUs <= 220.0` |
| 时间限制 | 单次完整运行不超过 10 分钟 |

## 1. 任务

输入 \(N\) 个变长 ASCII 字符串。请在 GPU 上按字典序排序，并输出排序后的原始下标。

发布包提供可编译的 ABI 模板，但不提供排序实现。参赛者需要完成 `submission.cu`，然后运行：

```bash
python runtest.py
```

Judge 会将 `submission.cu` 与固定的 `judge/binding.cpp` 编译为 PyTorch CUDA extension。

## 2. 排序规则

字符串按无符号字节值比较，有效字节范围为 `0x21..0x7e`。对字符串 \(a\) 和 \(b\)：

1. 从首字节开始依次比较；
2. 第一个不同字节较小的字符串排在前面；
3. 若一个字符串是另一个字符串的前缀，较短者排在前面；
4. 若内容和长度均相同，原始下标较小者排在前面。

因此每个输入只有一个正确输出。例如：

```text
input  = ["ab", "a", "ab", "!", ""]
output = [4, 3, 1, 0, 2]
```

## 3. 数据与接口

每个测试点提供：

```text
strings [N,W] UINT8
lengths [N]   INT32
```

`strings[i, :lengths[i]]` 是第 \(i\) 个字符串，其余位置为零填充。数据满足：

$$
W\in\{16,32,64\},\qquad 0\leq \mathrm{lengths}[i]\leq W.
$$

Judge 在 GPU 上分配以下连续张量：

```text
strings     UINT8 [N,W]
lengths     INT32 [N]
indices_out INT32 [N]
workspace   UINT8 [workspace_bytes(N,W)]
```

`strings` 和 `lengths` 只读，`indices_out` 必须完整写入。

`submission.cu` 需要实现：

```cpp
int64_t workspace_bytes(int64_t n, int64_t width);

void lexsort_cuda(
    torch::Tensor strings,
    torch::Tensor lengths,
    torch::Tensor indices_out,
    torch::Tensor workspace);
```

`workspace_bytes` 返回所需 workspace 字节数，不计入 GPU 时间。`lexsort_cuda` 必须在当前 PyTorch CUDA stream 上完成本次排序。

## 4. 测试点

| Case | \(N\) | \(W\) | 输入特点 | 考察重点 | 权重 |
|---|---:|---:|---|---|---:|
| `E-EDGE` | 513 | 32 | 空串、前缀、重复串、边界字符 | 排序语义 | 仅检查 |
| `S-SHORT` | 65,536 | 16 | 长度 \(0\) 到 \(16\) 的短字符串 | 短 key 吞吐 | 20% |
| `P-PREFIX` | 65,536 | 32 | 大量字符串共享较长前缀 | 深比较 | 20% |
| `D-DUP` | 131,072 | 32 | 从固定字符串池重复采样 | 相等 key 与下标 tie-break | 15% |
| `L-LONG` | 65,536 | 64 | 长度主要分布在 \(32\) 到 \(64\) | 长 key 访存 | 20% |
| `M-MIXED` | 262,144 | 32 | 随机串、共享前缀、重复串混合 | 大规模综合负载 | 25% |

`E-EDGE` 只检查正确性，避免小规模 kernel 启动开销影响排名。五个计分点均至少包含 \(65{,}536\) 个字符串，并分别覆盖短 key、长前缀、高重复、长 key 和混合数据，避免单一数据分布决定成绩。

固定输入和正确下标保存在 `data/*.npz`。`data/SHA256SUMS` 用于校验数据完整性。

## 5. 正确性

每个测试点先执行一次不计时验证：

```text
将 indices_out 填为 -1
调用 lexsort_cuda
同步当前 CUDA stream
逐元素比较正确下标
检查 strings 和 lengths 未被修改
```

验证轮同时用于 CUDA 编译后的 warmup。任一输出错误、漏写或输入修改都会得到 `WRONG_ANSWER`。

计时调用结束后，Judge 会在计时区间外再次检查输出和输入。

## 6. 计时与评分

每个计分测试点只计时一次，每次计时只调用一次完整排序：

```python
indices_out.fill_(-1)  # 计时外
start.record(current_stream)
lexsort_cuda(strings, lengths, indices_out, workspace)
end.record(current_stream)
end.synchronize()
```

CUDA Event 覆盖 `lexsort_cuda` 的完整执行过程。索引初始化、key 处理、排序、数据搬移和结果写回，只要是实现本次排序所必需的工作，都必须在该函数内完成。

数据读取、Host-to-Device copy、extension 编译、workspace 分配和正确性比较不计入 GPU 时间。

设测试点 \(i\) 的 GPU 时间为 \(t_i\)，权重为 \(w_i\)，先计算加权 GPU 时间：

$$
\operatorname{WeightedGpuTimeUs}
=\exp\!\left(\sum_i w_i\ln t_i\right).
$$

满分线为：

$$
T_{\mathrm{full}}=220.0\ \mu\mathrm{s}.
$$

正确性全部通过后，分数为：

$$
\operatorname{Score}
=\min\!\left(
100,
100\times\frac{T_{\mathrm{full}}}
{\operatorname{WeightedGpuTimeUs}}
\right).
$$

未通过正确性检查不计分。达到满分线即可获得 100 分，更快的实现仍记 100 分。该标准面向正常的 CUDA 并行排序实现，不要求极限调参；普通的逐字节比较 merge sort 不能达到满分。

## 7. 实现规则

只能修改 `submission.cu`。每次 `lexsort_cuda` 调用都必须根据当前输入重新完成排序并完整写入输出。

可以自行实现 merge sort、radix sort、分桶、scan、histogram、key packing、CUDA kernel、warp primitive、shared memory 和 PTX。

不得：

- 使用 CUB、Thrust、cuCollections 或其他现成排序实现；
- 调用 PyTorch、NumPy、CuPy、JAX 或 C++ 标准库排序 API；
- 把输入复制到 CPU 后排序；
- 缓存前一次结果、硬编码答案、修改输入或 Judge；
- 干扰 CUDA Event 或通过外部进程完成计算。

提交代码将进行源码审核。

## 8. 环境与提交

```text
GPU:     NVIDIA GeForce RTX 4090, 24 GiB, SM89
OS:      Linux x86_64
Python:  3.10.12
CUDA:    12.8
PyTorch: 2.11.0+cu128
Driver:  570.153.02
```

组委会在固定平台镜像内执行：

```bash
python runtest.py
```

平台会在 10 分钟时终止仍未结束的进程。提交内容为修改后的 `submission.cu` 和实现说明。

## 9. 输出

```text
CASE          N   W      GPU_TIME(us)
E-EDGE        513  32         CHECK
S-SHORT     65536  16       ...
...
Status: PASS
WeightedGpuTimeUs: ...
Score: ...
WallTimeSec: ...
RESULT_JSON={"status":"PASS",...}
```

| 状态 | 含义 |
|---|---|
| `PASS` | 所有测试点通过 |
| `WRONG_ANSWER` | 输出错误或输入被修改 |
| `INVALID_TIME` | CUDA Event 时间非法 |
| `OOM` | GPU 显存不足 |
| `COMPILE_ERROR` | `submission.cu` 编译或加载失败 |
| `RUNTIME_ERROR` | 提交代码运行失败 |
| `TIME_LIMIT` | 完整运行超过 10 分钟 |
| `JUDGE_ERROR` | 数据、GPU 或 Judge 异常 |

退出码：

```text
0 = PASS
1 = 提交错误、错误答案、非法时间、OOM 或超时
2 = 数据、GPU 或 Judge 错误
```

## 10. 实现提示

公开的 `submission.cu` 只有函数签名和未实现提示，不包含可通过评测的排序代码。

本题不需要复杂算法。以下性质可以直接利用：

- 有效字符均大于零，字符串尾部又以零填充，因此直接比较完整的 \(W\) 字节即可得到正确的前缀顺序；
- 可以按 32 位或 64 位读取字符串，找到第一个不同字节，避免逐字节循环；
- 常规的 parallel merge sort、merge-path 或自行实现的 radix sort 均可达到较好成绩；
- 下标作为最终 tie-break，可保证重复字符串的唯一顺序。
