# TileLang 变长 Paged GQA Decode 算子优化

使用 TileLang 实现支持变长请求和分页 KV Cache 的 GQA Decode Attention 算子，并优化其运行延迟。

## 背景介绍

[TileLang](https://github.com/tile-ai/tilelang) 是用于开发高性能 GPU 和 CPU 算子的领域专用语言。它采用接近 Python 的语法，并在 TVM 之上构建编译基础设施。开发者可以控制数据分块、内存使用和并行计算。官方仓库提供矩阵乘法、FlashAttention 和线性 Attention 等高性能算子实现可供参考。

大模型推理常用分页 KV Cache 管理各个请求的历史 Key 和 Value。Decode 阶段中，每个请求对应一个 Query Token，并读取该请求已有的全部 KV 数据。请求长度差异、分页寻址和 Softmax 归约会共同影响算子性能。

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
3. 阅读 `v0.1.12` 中的 [GQA Decode 实现](https://github.com/tile-ai/tilelang/blob/v0.1.12/examples/flash_decoding/example_gqa_decode.py) 和 [分页 MLA Decode 实现](https://github.com/tile-ai/tilelang/blob/v0.1.12/examples/deepseek_mla/example_mla_decode_paged.py)。前者展示分段计算和结果合并，后者展示 `block_table` 的分页寻址。实现时请按本题接口调整张量布局。
4. 编译或数值出现问题时，查阅 [调试工具](https://tilelang.com/tutorials/debug_tools_for_tilelang.html) 和 [自动调优](https://tilelang.com/tutorials/auto_tuning.html)。

RTX 4090 对应 `sm_89`。TileLang 的 [编译目标说明](https://tilelang.com/get_started/targets.html) 给出了目标配置方法。本题可使用：

```python
target = {"kind": "cuda", "arch": "sm_89"}
```

选手应以本题的 `reference.py`、接口和数据范围为准。

## 赛题描述

对请求 `b` 和 Query Head `h` 计算：

```text
group_size = Hq / Hkv
kv_head    = floor(h / group_size)
score_j    = dot(q[b, h], K[b, kv_head, j]) / sqrt(128)
p_j        = softmax(score)_j
out[b, h]  = sum_j(p_j * V[b, kv_head, j])
```

其中 `0 <= j < seq_lens[b]`。逻辑位置 `j` 通过 `block_table` 映射到分页 KV Cache：

```text
logical_page  = floor(j / PageSize)
offset        = j % PageSize
physical_page = block_table[b, logical_page]

K[b, kv_head, j] = k_cache[physical_page, offset, kv_head]
V[b, kv_head, j] = v_cache[physical_page, offset, kv_head]
```

有效物理页编号位于 `[0, Npage)`。`block_table` 在有效页之后填充 `-1`。Kernel 根据 `seq_lens` 确定访问范围。多个请求可以引用同一物理页。

`reference.py` 使用 FP32 计算 Score、Softmax 和 Value Reduction，最后转为 BF16。选手可以使用数学等价的在线 Softmax、KV 分段计算或多 Kernel 合并方法。评测范围为单卡单 Token Decode。输入从 Query 和分页 KV Cache 开始，输出为 Attention 结果。

## 输入输出

| 名称                     | 类型  | 形状                                | 说明                 |
| ------------------------ | ----- | ----------------------------------- | -------------------- |
| `q`                    | BF16  | `[B, Hq, 128]`                    | 单 Token Query       |
| `k_cache`, `v_cache` | BF16  | `[Npage, PageSize, Hkv, 128]`     | 分页 KV Cache        |
| `block_table`          | INT32 | `[B, ceil(MaxSeqLen / PageSize)]` | 逻辑页到物理页的映射 |
| `seq_lens`             | INT32 | `[B]`                             | 各请求的有效 KV 长度 |
| `workspace`            | UINT8 | `[128 MiB]`                       | 评测器提供的临时空间 |
| `out`                  | BF16  | `[B, Hq, 128]`                    | Attention 输出       |

所有张量均连续存储，并位于同一张 GPU。每个请求满足 `1 <= seq_lens[b] <= MaxSeqLen`。

默认情况下，`q`、`k_cache` 和 `v_cache` 服从 `Normal(0, 0.5)`。正确性测试还会覆盖较大的 Attention Score、接近均匀的 Softmax、页边界和共享物理前缀。所有输入元素均为有限值。

## 提交接口

选手需要实现 `submission.py` 中的 `Submission`：

```python
class Submission:
    def build(self, spec):
        ...

    def run(
        self,
        state,
        q, k_cache, v_cache,
        block_table, seq_lens,
        workspace, out,
    ) -> None:
        ...
```

`build(spec)` 接收静态配置。选手可以在此编译 Kernel 并返回 `run()` 所需的对象。评测器在性能计时前完成 `build()`。`spec` 包含：

| 字段                | 含义                     |
| ------------------- | ------------------------ |
| `batch_size`      | `B`                    |
| `num_q_heads`     | `Hq`                   |
| `num_kv_heads`    | `Hkv`                  |
| `max_seq_len`     | `MaxSeqLen`            |
| `num_pages`       | `Npage`                |
| `head_dim`        | 固定为`128`            |
| `page_size`       | `16` 或 `32`         |
| `workspace_bytes` | 固定为`128 MiB`        |
| `dtype`           | 固定为`torch.bfloat16` |

评测器在当前 CUDA Stream 上调用 `run()`。`run()` 将结果写入评测器提供的 `out`。`workspace` 和 `out` 为可写张量，其余输入按只读张量处理。`block_table` 与 `seq_lens` 是运行时 GPU Tensor，每次调用都可能变化。

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

| 项目                   | 范围或配置                |
| ---------------------- | ------------------------- |
| `B`                  | `1、8、16、32、64、128` |
| `Hq`                 | `32` 或 `64`          |
| `Hq / Hkv`           | `4` 或 `8`            |
| Head Dimension         | `128`                   |
| `MaxSeqLen`          | `1` 至 `32768`        |
| Page Size              | `16` 或 `32`          |
| 单次有效 KV Token 总数 | 最大 `524288`           |
| Workspace              | `128 MiB`               |

公开性能用例为：

| 用例 | `B` | `Hq / Hkv` | Page Size | KV 长度分布                   | 权重 |
| ---- | ----: | -----------: | --------: | ----------------------------- | ---: |
| G1   |     1 |       32 / 8 |        16 | 24K 至 32K                    |  20% |
| G2   |     8 |       32 / 8 |        16 | 32K 至 256 的阶梯式长度       |  30% |
| G3   |    32 |       32 / 8 |        16 | 4 条长、8 条中等、20 条短请求 |  30% |
| G4   |   128 |       64 / 8 |        32 | 128 至 1K 的短请求            |  20% |

赛后复测沿用输入张量形状和长度分布范围，并更换实际长度、物理页映射和张量数据。`cases.json` 提供公开的固定正确性点和固定性能点；评测器还会生成不计性能的随机正确性用例。

## 正确性

评测器将提交结果与 FP32 参考计算转成的 BF16 结果比较。元素匹配条件为：

```text
abs(actual - reference) <= atol + rtol * abs(reference)
```

`NRMSE` 为均方根误差除以参考结果的均方根。余弦相似度按每个请求和每个 Query Head 单独计算。

| 指标                             |            要求 |
| -------------------------------- | --------------: |
| `rtol`                         |        `1e-2` |
| `atol`                         |        `1e-2` |
| 元素匹配比例                     |         `1.0` |
| 最大绝对误差                     |        `5e-2` |
| 全局 NRMSE                       |        `1e-2` |
| 每个请求、每个 Head 的余弦相似度 | 最低 `0.999`  |

评测器还会检查形状、类型、NaN、Inf 和输入内容。每次执行前，评测器更换 `workspace` 的填充值，并把输出填充为 NaN。每个正确性输入重复执行 3 次。全部正确性检查通过后，评测器记录性能成绩。

## 性能与评分

评测器使用 CUDA Event 统计 `run()` 发起的全部 GPU 工作。每个用例预热 10 次，随后执行至少 5 组测量。每组累计 GPU 时间达到 200 ms 后结束。每组测量前会清理 L2 缓存，并轮换 4 套输入缓冲区。评测器取各组单次调用延迟的中位数。同一用例使用 4 组随机数据复测，4 个中位延迟取几何平均。

评测器在计时前完成 `build()`、输入生成和输出分配。计时范围覆盖 `run()` 内的任务表生成、临时数据初始化、分段结果合并及多次 Kernel 调用间隔。

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
- 选手可以在 GPU 上生成任务表，并使用评测器提供的 `workspace`。
- 计时区域内的计算由 TileLang Kernel 执行。PyTorch、Triton、CUTLASS、cuBLAS、FlashAttention 和 FlashInfer 等工具可用于计时区域外的开发验证。
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

# 随机正确性测试：不计时，不进入性能结果
python benchmark.py --mode test --random-correctness --random-count 5

# 开发阶段固定性能测试，同时检查正确性并生成机器报告
python benchmark.py --mode bench --case performance --json-out benchmark.json

# 公开性能用例的本地测试
python benchmark.py --mode bench --case official --preset official --device cuda --json-out benchmark.json
```

公开性能用例的本地命令用于测量性能。逐请求的 PyTorch 参考计算耗时较长，正式评测会先运行独立的正确性检查，再测量 4 个公开用例的性能。

请在完成 TileLang 实现后再运行这组性能用例。

本题提交修改后的 `submission.py`（以及它引用的自建源码）和简短的 `optimization.md`。说明文档自由描述优化思路；如使用题目环境未列出的依赖，请写明名称、版本和安装方法。`benchmark.json` 必须由 `benchmark.py` 生成，不需要也不允许手工填写。正式复测通过 `from submission import Submission` 加载算子。

## 参考资料

- [PagedAttention 论文](https://arxiv.org/abs/2309.06180)
