# TileLang 变长 KDA Prefill 算子优化

使用 TileLang 实现支持变长序列的 Kimi Delta Attention Prefill 算子，并优化其运行延迟。

## 背景介绍

[TileLang](https://github.com/tile-ai/tilelang) 是用于开发高性能 GPU 和 CPU 算子的领域专用语言。它采用接近 Python 的语法，并在 TVM 之上构建编译基础设施。开发者可以控制数据分块、内存使用和并行计算。官方仓库提供矩阵乘法、FlashAttention 和线性 Attention 等高性能算子实现可供参考。

Kimi Delta Attention 使用固定大小的循环状态保存历史信息。逐 Token 计算会产生较多串行计算和状态张量读写。优化时需要按块处理 Token，并利用各序列和 Attention Head 之间的并行性。

## 测试环境与 TileLang 入门

比赛平台为每名选手提供单张 NVIDIA GeForce RTX 4090 24GB 算力，用于开发、测试和评测。正式环境的完整版本见 `ENVIRONMENT.md`，选手在个人运行环境中安装所列软件。

请先按 [PyTorch 安装说明](https://pytorch.org/get-started/locally/) 配置支持 CUDA 的 PyTorch，再参考 [TileLang 官方安装指南](https://tilelang.com/get_started/Installation.html) 执行以下命令：

```bash
python -m pip install "git+https://github.com/tile-ai/tilelang.git@v0.1.12"
python -c "import torch, tilelang; print(torch.__version__, tilelang.__version__); print(torch.cuda.get_device_name())"
```

开始编写算子前，请完成以下内容：

1. 阅读 [TileLang 语言基础](https://tilelang.com/programming_guides/language_basics.html)，运行页面中的向量加法和矩阵乘法示例。该页面说明了 Kernel 定义、内存空间和 JIT 调用方式。
2. 阅读 [TileLang 指令说明](https://tilelang.com/programming_guides/instructions.html) 和 [软件流水线说明](https://tilelang.com/programming_guides/software_pipeline.html)，了解数据搬运、矩阵乘法、归约和流水执行。
3. 阅读 `v0.1.12` 中的 [KDA 分块状态更新](https://github.com/tile-ai/tilelang/blob/v0.1.12/examples/kda/chunk_delta_h_fwd.py) 和 [线性 Attention 前向实现](https://github.com/tile-ai/tilelang/blob/v0.1.12/examples/linear_attention/example_linear_attn_fwd.py)。这些示例展示分块计算和循环状态处理。实现时请按本题接口调整张量布局。
4. 编译或数值出现问题时，查阅 [调试工具](https://tilelang.com/tutorials/debug_tools_for_tilelang.html) 和 [自动调优](https://tilelang.com/tutorials/auto_tuning.html)。

RTX 4090 对应 `sm_89`。TileLang 的 [编译目标说明](https://tilelang.com/get_started/targets.html) 给出了目标配置方法。本题可使用：

```python
target = {"kind": "cuda", "arch": "sm_89"}
```

选手应以本题的 `reference.py`、接口和数据范围为准。

## 赛题描述

一个 Batch 含有 `B` 条非空序列。所有 Token 按序列顺序存放在同一个张量中，`cu_seqlens` 给出每条序列的起止位置。每条序列从对应的 `initial_state` 开始计算。

对每条序列、每个 Head 和每个 Token 依次执行：

```text
q_hat   = q / sqrt(sum(q^2) + 1e-6)
k_hat   = k / sqrt(sum(k^2) + 1e-6)

x       = exp(a_log) * (g_raw + dt_bias)
log_a   = -5.0 * sigmoid(x)
a       = exp(log_a)
beta    = sigmoid(beta_raw)

S_decay = Diag(a) @ S_prev
res     = v - S_decay^T @ k_hat
S       = S_decay + beta * k_hat @ res^T
out     = (S^T @ q_hat) / sqrt(128)
```

状态 `S` 的形状为 `[128, 128]`。第一维对应 Key，第二维对应 Value。每条序列、每个 Head 分别维护状态。算子需要写出全部 Token 的 `out`，并写出每条序列的 `final_state`。

`reference.py` 使用 FP32 完成归一化、Gate、状态更新和输出计算，最后转为 BF16。选手可以使用数学等价的分块、扫描或 Kernel 融合方法。评测范围为单卡前向 Prefill，输入从 Attention 所需张量开始，输出为 `out` 和 `final_state`。

## 输入输出

| 名称                           | 类型  | 形状                 | 说明                 |
| ------------------------------ | ----- | -------------------- | -------------------- |
| `q`, `k`, `v`, `g_raw` | BF16  | `[T, H, 128]`      | Token 输入           |
| `beta_raw`                   | BF16  | `[T, H]`           | 更新系数             |
| `a_log`                      | FP32  | `[H]`              | Gate 参数            |
| `dt_bias`                    | FP32  | `[H, 128]`         | Gate 偏置            |
| `initial_state`              | BF16  | `[B, H, 128, 128]` | 初始状态             |
| `cu_seqlens`                 | INT32 | `[B + 1]`          | 序列边界             |
| `workspace`                  | UINT8 | `[128 MiB]`        | 评测器提供的临时空间 |
| `out`                        | BF16  | `[T, H, 128]`      | Token 输出           |
| `final_state`                | BF16  | `[B, H, 128, 128]` | 最终状态             |

所有张量均连续存储，并位于同一张 GPU。`cu_seqlens[0] = 0`，`cu_seqlens[B] = T`，相邻元素严格递增。

默认输入分布如下。正确性测试还会覆盖 Gate 饱和、接近零的 Q/K 和非零初始状态。所有输入元素均为有限值。

```text
q, k, v       ~ Normal(0, 0.5)
g_raw         ~ Normal(0, 0.5)
beta_raw      ~ Normal(0, 1.0)
a_log         ~ Uniform(-0.1, 0.1)
dt_bias       ~ Normal(0, 0.1)
initial_state ~ Normal(0, 0.02)
```

## 提交接口

选手需要实现 `submission.py` 中的 `Submission`：

```python
class Submission:
    def build(self, spec):
        ...

    def run(
        self,
        state,
        q, k, v, g_raw, beta_raw,
        a_log, dt_bias,
        initial_state, cu_seqlens,
        workspace, out, final_state,
    ) -> None:
        ...
```

`build(spec)` 接收静态配置。选手可以在此编译 Kernel 并返回 `run()` 所需的对象。评测器在性能计时前完成 `build()`。`spec` 包含：

| 字段                | 含义                     |
| ------------------- | ------------------------ |
| `total_tokens`    | `T`                    |
| `num_sequences`   | `B`                    |
| `num_heads`       | `H`                    |
| `head_dim`        | 固定为`128`            |
| `chunk_size`      | 固定为`64`             |
| `workspace_bytes` | 固定为`128 MiB`        |
| `dtype`           | 固定为`torch.bfloat16` |

评测器在当前 CUDA Stream 上调用 `run()`。`run()` 将结果写入评测器提供的输出张量。`workspace`、`out` 和 `final_state` 为可写张量，其余输入按只读张量处理。`cu_seqlens` 是运行时 GPU Tensor。

题目目录包含以下文件：

| 文件              | 用途               |
| ----------------- | ------------------ |
| `README.md`     | 题目说明           |
| `submission.py` | 选手修改的提交文件 |
| `reference.py`  | FP32 参考实现      |
| `benchmark.py`  | 正确性和性能评测   |
| `cases.json`    | 公开测试点         |
| `ENVIRONMENT.md` | 正式环境名称和版本 |
| `optimization.md` | 优化思路与额外依赖说明 |

`submission.py` 和其中引用的自建源码由选手修改。`benchmark.py`、`reference.py` 和 `cases.json` 构成统一评测链路；正式复测时，主办方会用组织者版本覆盖这些文件和测试配置。修改本地评测文件不会改变正式成绩。

## 数据范围

| 项目           | 范围或配置         |
| -------------- | ------------------ |
| `T`          | `1` 至 `32768` |
| `B`          | `1` 至 `32`    |
| `H`          | `8` 或 `16`    |
| Head Dimension | `128`            |
| Chunk Size     | `64`             |
| Workspace      | `128 MiB`        |

公开性能用例为：

| 用例 | `T` | `B` | `H` | 序列长度                                     | 权重 |
| ---- | ----: | ----: | ----: | -------------------------------------------- | ---: |
| K1   |  4096 |    32 |    16 | `128 × 32`                                |  20% |
| K2   |  8064 |     8 |    16 | `64, 128, 256, 512, 768, 1024, 2048, 3264` |  25% |
| K3   | 16384 |     1 |     8 | `16384`                                    |  25% |
| K4   | 32768 |     4 |     8 | `24576, 4096, 2048, 2048`                  |  30% |

赛后复测沿用上述输入规模，并更换张量数据和序列排列。`cases.json` 提供公开的固定正确性点和固定性能点；评测器还会生成不计性能的随机正确性用例。

## 正确性

评测器将提交结果与 FP32 参考计算转成的 BF16 结果比较。元素匹配条件为：

```text
abs(actual - reference) <= atol + rtol * abs(reference)
```

`NRMSE` 为均方根误差除以参考结果的均方根。`out` 的局部 NRMSE 按每条序列、每个 Head 和 64 Token 窗口计算。`final_state` 的局部 NRMSE 按每条序列和每个 Head 计算。

| 输出            | `rtol` | `atol` | 最低匹配比例 | 最大绝对误差 | 全局 NRMSE | 最大局部 NRMSE |
| --------------- | -------: | -------: | -----------: | -----------: | ---------: | -------------: |
| `out`         | `2e-2` | `2e-2` |   `0.9999` |     `0.25` |   `1e-2` |       `2e-2` |
| `final_state` | `3e-2` | `2e-2` |   `0.9999` |     `0.30` | `1.5e-2` |     `2.5e-2` |

评测器还会检查形状、类型、NaN、Inf 和输入内容。每次执行前，评测器更换 `workspace` 的填充值，并把输出填充为 NaN。每个正确性输入重复执行 3 次。全部正确性检查通过后，评测器记录性能成绩。

## 性能与评分

评测器使用 CUDA Event 统计 `run()` 发起的全部 GPU 工作。每个用例预热 10 次，随后执行至少 5 组测量。每组累计 GPU 时间达到 200 ms 后结束。每组测量前会清理 L2 缓存，并轮换 4 套输入缓冲区。评测器取各组单次调用延迟的中位数。同一用例使用 4 组随机数据复测，4 个中位延迟取几何平均。

评测器在计时前完成 `build()`、输入生成和输出分配。计时范围覆盖 `run()` 内的临时数据初始化、多次 Kernel 调用及调用间隔。

最终指标为 4 个用例延迟的加权几何平均：

```text
score_latency = exp(sum_i(weight_i * log(latency_i)))
```

单位为微秒。指标越低，排名越高。

## 实现要求

- 核心计算由 TileLang Kernel 完成。
- 选手可以使用多个 Kernel、Tensor Core、共享内存、软件流水线和离线自动调优。
- 所有 Kernel 均在当前 CUDA Stream 上启动，`run()` 使用评测器传入的张量完成计算。
- `run()` 采用异步 Kernel 启动方式，评测器负责同步和计时。
- 运行时中间数据使用评测器提供的 `workspace`。
- 计时区域内的计算由 TileLang Kernel 执行。PyTorch、Triton、CUTLASS、cuBLAS 等工具可用于计时区域外的开发验证。
- CUDA、PTX 和 SASS 代码由 TileLang `v0.1.12` 官方编译器生成。
- 运行时张量保留在 GPU，所有输入按只读方式使用。
- 提交程序根据每次传入的张量计算结果。

## 测试与提交

`submission.py` 初始提供可运行的 PyTorch 版本，用于检查数据和接口。正式提交采用 TileLang Kernel 实现核心计算。

```bash
# 基础正确性检查
python benchmark.py --mode test --case basic

# 完整公开正确性检查
python benchmark.py --mode test --case correctness

# 新 kernel 首次遇到多个静态 shape 时并行预编译；GPU 检查仍然串行
python benchmark.py --mode test --case correctness --compile-workers 4

# 随机正确性测试：不计时，不进入性能结果
python benchmark.py --mode test --random-correctness --random-seed 20260809 --random-count 5 --compile-workers 4

# 开发阶段固定性能测试，同时检查正确性并生成机器报告
python benchmark.py --mode bench --case performance --json-out benchmark.json

# 公开性能用例的本地测试
python benchmark.py --mode bench --case official --preset official --device cuda --json-out benchmark.json
```

`--compile-workers N` 只并行预热不同 `(T, B, H)` 的 TileLang 磁盘 cache；正确性检查和性能测量仍按 case 串行执行。相同源码和静态 shape 的后续进程会直接加载 cache，不再重新 lowering。

公开性能用例的本地命令用于测量性能。逐 Token 的 PyTorch 参考计算耗时较长，正式评测会先运行独立的正确性检查，再测量 4 个公开用例的性能。

请在完成 TileLang 实现后再运行这组性能用例。

本题提交修改后的 `submission.py`（以及它引用的自建源码）和简短的 `optimization.md`。说明文档自由描述优化思路；如使用题目环境未列出的依赖，请写明名称、版本和安装方法。`benchmark.json` 必须由 `benchmark.py` 生成，不需要也不允许手工填写。正式复测通过 `from submission import Submission` 加载算子。

## 参考资料

- [Kimi Linear 技术报告](https://arxiv.org/abs/2510.26692)
