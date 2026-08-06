# KDA Stateful Operator 优化

| 项目 | 说明 |
|---|---|
| 赛题类型 | GPU 算子优化 |
| 目标设备 | NVIDIA GeForce RTX 4090（Ada / SM89 / 24 GiB） |
| 运行命令 | `python runtest.py` |
| 排名指标 | `WeightedGpuTimeMs`，越小越好 |
| 可修改内容 | `submission.py` 和参赛者新增的实现源码 |

## 1. 任务说明

Kimi Delta Attention（KDA）使用一个随 token 更新的矩阵状态。Prefill 需要高效处理 packed sequence，Decode 则需要在连续调用之间保留并更新状态。

本题提供投影后的 KDA Core 输入。参赛者需要实现 Append 和 Decode 计算，在保持数学结果和接口行为正确的前提下，尽量降低 GPU 执行时间。

本题不包括线性投影、ShortConv、MLA、采样、动态 batching、分布式通信或完整模型推理。

发布包内已经提供可运行的 FLA 起始实现。直接运行：

```bash
python runtest.py
```

## 2. 数学定义

固定维度为 $K=V=128$。对每个 batch、head 和 token：

$$
q,k,g_{\mathrm{raw}}\in\mathbb{R}^{K},\qquad
v,z_{\mathrm{gate}}\in\mathbb{R}^{V},\qquad
\beta_{\mathrm{raw}}\in\mathbb{R},\qquad
R\in\mathbb{R}^{V\times K}.
$$

每层参数为：

| 参数 | 形状 | dtype |
|---|---:|---|
| `A_log` | $[H]$ | FP32 |
| `dt_bias` | $[H,K]$ | FP32 |
| `output_norm_weight` | $[V]$ | FP32 |

下文用 $A_{\log}$、$b_{\mathrm{dt}}$、$w_{\mathrm{norm}}$ 和 $z_{\mathrm{gate}}$ 分别表示接口中的 `A_log`、`dt_bias`、`output_norm_weight` 和 `output_gate_logits`。

首先对 query 和 key 做 L2 归一化，并计算衰减与更新系数：

$$
\begin{aligned}
\bar q &= \frac{q}{\sqrt{\sum_{j=1}^{K}q_j^2+10^{-6}}}, &
\bar k &= \frac{k}{\sqrt{\sum_{j=1}^{K}k_j^2+10^{-6}}},\\
\log\alpha &= -5\,\sigma\!\left(\exp(A_{\log})\left(g_{\mathrm{raw}}+b_{\mathrm{dt}}\right)\right), &
\alpha &= \exp(\log\alpha),\\
\beta &= \sigma(\beta_{\mathrm{raw}}).
\end{aligned}
$$

随后更新矩阵状态。$\alpha$ 按列广播：

$$
\begin{aligned}
R_{:,j} &\leftarrow \alpha_jR_{:,j},\qquad j=1,\ldots,K,\\
e &= v-R\bar k,\\
R &\leftarrow R+\beta\,e\bar k^{\mathsf T}.
\end{aligned}
$$

最后使用更新后的状态计算输出：

$$
\begin{aligned}
o_{\mathrm{raw}}^{\mathrm{FP32}} &= R\frac{\bar q}{\sqrt{128}},\\
o_{\mathrm{raw}} &= \operatorname{BF16}\!\left(o_{\mathrm{raw}}^{\mathrm{FP32}}\right),\\
x &= \operatorname{FP32}(o_{\mathrm{raw}}),\\
o_{\mathrm{rms}} &=
x\left(\frac{1}{V}\sum_{j=1}^{V}x_j^2+10^{-5}\right)^{-1/2}
\odot w_{\mathrm{norm}},\\
o &= \operatorname{BF16}\!\left(
\sigma\!\left(\operatorname{FP32}(z_{\mathrm{gate}})\right)
\odot o_{\mathrm{rms}}
\right).
\end{aligned}
$$

输出使用当前 token 更新后的状态，也就是 post-update output。除公式中显式的 $\operatorname{BF16}(\cdot)$ 外，参考计算中的运算和累积均使用 FP32。融合实现可以省略中间张量的写回，但最终输出和状态必须满足第 5 节的误差要求。

## 3. 接口说明

`submission.py` 必须提供一个可无参数构造的 `Submission` 类：

```python
class Submission:
    def prepare(self, config, layer, case): ...
    def load_state(self, context, canonical_state): ...
    def append_chunk(self, context, private_state, args, output): ...
    def decode_step(self, context, private_state, token, output): ...
    def export_state(self, context, private_state, canonical_state_out): ...
```

五个方法都由参赛者实现，也都允许修改。通常只需要重点优化 `append_chunk` 和 `decode_step`，因为只有这两个方法计入 GPU 时间。

另外三个方法用于初始化、状态格式转换和正确性检查：

- `prepare` 每个测试点调用一次，可以编译扩展、分配 workspace，以及预计算只依赖 layer 的常量。`layer` 只读，返回的 context 会被该测试点的所有轮次复用。
- `load_state` 每轮调用一次，可以把 Judge 提供的只读 FP32 初始状态转换成自定义布局、精度或量化格式。每次必须返回相互独立的内部状态对象。
- `export_state` 在 Judge 检查状态时调用，可以把当前内部状态转换回 Judge 要求的 FP32 格式；同一轮中可能调用多次。

这三个方法不计时，但不得计算、缓存或补做任何 token 对应的 output 或 state update。每次 `append_chunk` 和 `decode_step` 都必须在本次调用中完成规定的全部计算。

### 3.1 状态

Judge 使用的标准状态 `canonical_state` 是形状为 $[B,H,V,K]$ 的 contiguous FP32 张量。

`load_state` 返回参赛者自定义的内部状态对象。它可以采用不同的 dtype、布局、量化方式或补偿结构。不同测试轮次的内部状态必须相互独立。

`export_state` 只读取当前状态并写入 `canonical_state_out`，不得推进、清空、替换或释放内部状态。调用结束后，Judge 可以继续使用同一个内部状态对象。

### 3.2 Append 输入

一次 `append_chunk` 接收一个 packed batch：

| 张量 | 形状 | dtype |
|---|---:|---|
| `q_act` | $[T_{\mathrm{total}},H,K]$ | BF16 |
| `k_act` | $[T_{\mathrm{total}},H,K]$ | BF16 |
| `v_act` | $[T_{\mathrm{total}},H,V]$ | BF16 |
| `g_raw` | $[T_{\mathrm{total}},H,K]$ | BF16 |
| `beta_raw` | $[T_{\mathrm{total}},H]$ | FP32 |
| `output_gate_logits` | $[T_{\mathrm{total}},H,V]$ | BF16 |
| `cu_seqlens` | $[B+1]$ | INT32 |
| `descriptor` | $[N_{\mathrm{tiles}},4]$ | INT32 |
| `output` | $[T_{\mathrm{total}},H,V]$ | BF16 |

`cu_seqlens` 给出各序列的边界。`descriptor` 每行依次保存 `sequence_id`、`token_start`、`valid_tokens` 和 `state_id`。本题满足 $\mathtt{state\_id}=\mathtt{sequence\_id}$。实现可以使用或忽略 `descriptor`，但必须按各序列的 token 顺序更新状态，并完整写入 `output`。

### 3.3 Decode 输入

每次 `decode_step` 只提供当前一个 token：

| 张量 | 形状 | dtype |
|---|---:|---|
| `q_act`、`k_act`、`g_raw` | $[B,H,K]$ | BF16 |
| `v_act` | $[B,H,V]$ | BF16 |
| `beta_raw` | $[B,H]$ | FP32 |
| `output_gate_logits` | $[B,H,V]$ | BF16 |
| `output` | $[B,H,V]$ | BF16 |

实现必须更新内部状态并完整写入当前 token 的 `output`。Judge 不会向提交代码提供后续 token。

每轮依次调用 `load_state`、零次或多次 `append_chunk`、零次或多次 `decode_step`，最后按检查需要调用 `export_state`。

同一轮始终使用同一个内部状态对象，Append 和 Decode 之间不会强制转换回标准 FP32 状态。

## 4. 测试点

head 数为 $H\in\{24,48\}$。六个测试点只执行 Append，另外三个测试点在 Append 后继续执行 16384 步 Decode。

### 4.1 Append 测试

| Case | B | H | Append 安排 | Profile | 初始状态 | 权重 |
|---|---:|---:|---|---|---|---:|
| `L24-Z` | 1 | 24 | 每个序列 65536 token，一次调用 | `typical` | `zero` | 12.5% |
| `L24-N` | 1 | 24 | 每个序列 65536 token，一次调用 | `strong_decay` | `nonzero` | 12.5% |
| `M24-C` | 8 | 24 | 一次 packed 调用，8 个不同长度 | `typical` | `checkpoint` | 5% |
| `M48-N` | 32 | 48 | 一次 packed 调用，8 个长度组成一组，共 4 组 | `slow_decay` | `nonzero` | 5% |
| `C48` | 32 | 48 | 连续 4 次调用，每个序列依次追加 64 / 256 / 64 / 256 token | `strong_decay` | `checkpoint` | 12.5% |
| `C24` | 128 | 24 | 连续 4 次调用，每个序列依次追加 16 / 64 / 16 / 64 token | `near_no_decay` | `checkpoint` | 12.5% |

- `M24-C` 的 8 个序列长度为：4095、4096、4097、8191、8192、8193、12287、16385。
- `M48-N` 的一组长度为：511、512、513、1023、1024、1025、1535、1537；该组重复 4 次得到 32 个序列。

### 4.2 Decode 测试

| Case | B | H | Append | Decode | Profile | 初始状态 | 权重 |
|---|---:|---:|---|---|---|---|---:|
| `D48-B1` | 1 | 48 | 每个序列 17 token | 16384 步 | `typical` | `checkpoint` | 10% |
| `D48` | 32 | 48 | 每个序列 17 token | 16384 步 | `slow_decay` | `checkpoint` | 15% |
| `D24` | 128 | 24 | 每个序列 17 token | 16384 步 | `near_no_decay` | `checkpoint` | 15% |

`zero` 表示全零初始状态，`nonzero` 表示固定的非零初始状态，`checkpoint` 表示预先生成的检查点状态。Profile 决定 $g_{\mathrm{raw}}$ 的衰减范围。同一行中的连续 Append 调用共享同一个内部状态对象。完整参数以 `data/cases.json` 为准。

## 5. 正确性标准

Judge 将提交结果与预先生成的独立 FP32 参考结果比较。每个测试点先运行一次不计时的验证轮，并检查：

- 所有 Append output；
- 每次 Append 结束后的 FP32 state；
- `D48-B1`、`D48` 和 `D24` 全部 16384 步的 Decode output；
- Decode 第 $1,17,257,4096,16384$ 步的 FP32 state；
- shape、dtype、完整写入以及所有数值是否 finite。

最后一个计时轮结束后，Judge 还会检查最终 output 和最终 state。该检查发生在计时区间之后。

令提交结果为 $y$，参考结果为 $y^*$。误差定义为：

$$
\operatorname{RelativeL2}(y,y^*)=
\frac{\lVert y-y^*\rVert_2}
{\max\!\left(\lVert y^*\rVert_2,10^{-12}\right)},
$$

$$
\operatorname{NormalizedMax}(y,y^*)=
\frac{\max_j\lvert y_j-y_j^*\rvert}
{\max\!\left(\max_j\lvert y_j^*\rvert,1\right)}.
$$

| 检查对象 | Relative L2 上限 | NormalizedMax 上限 |
|---|---:|---:|
| output | `0.006` | `0.015` |
| state | `0.004` | `0.015` |

worst sequence/head Relative L2 和相应的参考 norm 仅供诊断，不影响成绩。

输出中的 `ACC_RATIO` 定义为

$$
\mathrm{ACC\_RATIO}=\max_{c\in\mathcal C}\frac{\mathrm{error}_c}{\mathrm{limit}_c},
$$

其中 $\mathcal C$ 是全部硬性误差检查。有效结果必须满足 $\mathrm{ACC\_RATIO}\leq 1$。验证缓冲区会预先填入 NaN，未完整写入也会判错。

## 6. 计时和排名

每个测试点先执行 1 次完整验证轮，同时完成 JIT、autotune 和 warmup；随后执行固定次数的计时轮：

| Case | 计时轮数 $R_i$ |
|---|---:|
| `L24-Z` | 32 |
| `L24-N` | 32 |
| `M24-C` | 32 |
| `M48-N` | 32 |
| `C48` | 64 |
| `C24` | 64 |
| `D48-B1` | 3 |
| `D48` | 3 |
| `D24` | 3 |

每个计时轮都从相同初始状态开始，并完整执行该测试点规定的全部 Append 和 Decode 调用。

函数 `append_chunk` 和 `decode_step` 计时。每次调用分别使用当前 PyTorch stream 上的 CUDA Event 测量，所有调用时间相加得到该轮的 GPU 时间。

以下工作不计时：`prepare` 和 JIT/autotune、`load_state`、输入准备、`export_state`、参考结果读取和正确性比较。

选手方法内部的格式转换、copy、kernel 间隔、output 后处理以及维持内部状态所需的工作都计时。所有必要 CUDA 工作必须提交到当前 stream，或者在方法返回前让当前 stream 等待其他 stream 完成。

记测试点 $i$ 的第 $r$ 个计时轮中，所有计时方法的 CUDA Event 时间之和为 $T_{i,r}$。该测试点的 GPU 时间定义为：

$$
t_i=\frac{1}{R_i}\sum_{r=1}^{R_i}T_{i,r}.
$$

结果 JSON 将 $t_i$ 记录为 `CaseGpuTimeMs`。Judge 不计算或输出 P50、P95、sample list 或 CV。

设表中权重为 $w_i$。排名指标定义为：

$$
\operatorname{WeightedGpuTimeMs}
=\exp\!\left(\sum_i w_i\ln t_i\right).
$$

成功输出示例：

```text
CASE     REPLAYS  CASE_GPU_MS  ACC_RATIO
L24-Z        32          ...          ...
...
Validation: PASS
WeightedGpuTimeMs: ...
WallTimeSec: ...
RESULT_JSON={"status":"PASS","weighted_gpu_time_ms":..., ...}
```

常见结果状态：

| 状态 | 含义 |
|---|---|
| `PASS` | 所有测试点通过 |
| `WRONG_ANSWER` | 输出或状态不满足正确性要求 |
| `OOM` | 提交代码显存不足 |
| `RUNTIME_ERROR` | 提交代码导入、构造或运行失败 |
| `JUDGE_ERROR` | 数据、环境或 Judge 自身异常 |

| 退出码 | 含义 |
|---:|---|
| `0` | `PASS` |
| `1` | 提交错误、错误答案或 OOM |
| `2` | 数据、环境或 Judge 错误 |

## 7. 平台镜像与提交

参赛者提交一个可直接运行的平台镜像。镜像内必须包含完整的赛题目录：

```text
README.md
runtest.py
submission.py
judge/
data/
选手新增的实现源码
构建脚本和运行依赖
SOLUTION.md
```

`runtest.py`、`judge/` 和 `data/` 必须与组委会发布的内容一致。参赛者可以修改 `submission.py`，也可以新增实现源码、编译扩展、构建文件和运行依赖。

镜像必须包含评测所需的全部用户态软件和完整数据包，包括 Python、PyTorch、Triton、FLA 以及选手使用的自定义扩展或其他合法依赖。评测过程中不得从网络下载文件。

`SOLUTION.md` 应说明构建和复现方法、依赖版本以及第三方许可证。提交必须包含实现所需的全部源码，并能在所提交的镜像中复现。

起始 `submission.py` 使用 FLA `chunk_kda_fwd(chunk_size=32)` 处理 Append，使用 `fused_recurrent_kda` 处理逐 token Decode，并使用 FP32 内部状态。它只用于提供可运行起点。

## 8. 禁止行为

- 根据 case ID、step、检查位置或已知输入硬编码、查表返回答案；
- 根据验证轮、计时轮或调用次数切换成不同的数学路径；
- 在任何 Decode step 跳过 output 计算或不完整写入 output；
- 获取后续 Decode token、合并多个 step 或改变调用顺序；
- 把 token 对应的计算移到 `prepare`、`load_state` 或 `export_state`；
- 保存完整 token 历史，并在 `export_state` 时重放以代替逐步状态更新；
- 修改只读输入、layer、初始状态、descriptor、Judge 或参考数据；
- 干扰 CUDA Event、系统计时、评测进程或其他提交；
- 通过文件、网络、外部进程或其他未提供的通道保存跨调用数学状态；
- 隐瞒第三方代码、预编译二进制、实际 dispatch 或许可证；
- 利用未定义行为、资源耗尽或 Judge 缺陷获取成绩。

违规提交将通过源码审核处理。

## 9. 参考资料

- Kimi Linear / KDA：[Kimi Linear: An Expressive, Efficient Attention Architecture](https://arxiv.org/abs/2510.26692)
- Kimi Linear 实现：[MoonshotAI/Kimi-Linear](https://github.com/MoonshotAI/Kimi-Linear)
- FLA KDA 起始实现：[fla-org/flash-linear-attention](https://github.com/fla-org/flash-linear-attention/tree/cf0242f9b7106cdba3b7334c97a7da88c177c2774/fla/ops/kda)，commit `cf0242f9b7106cdba3b7334c97a7da88c177c2774`，MIT License
