# 赛题：GatedDeltaNet 线性注意力 kernel 优化（单卡 4090）

> **一句话**：为 Qwen3-Next 类混合架构大模型里的 **GatedDeltaNet（GDN）线性注意力层**，
> 在单张 RTX 4090 上写一个比 `flash-linear-attention` 官方 chunked kernel **更快、且数值等价、可微**
> 的实现，使其能直接替换进大模型的训练/推理。

---

## 1 背景介绍

近一年主流大模型开始转向**混合线性注意力架构**：Qwen3-Next、MiniMax-01、以及本题背景所依据的
`Qwen3.6-35B-A3B`（256 专家 MoE，40 层）都属于这一类。它们的注意力不再是清一色的
softmax full-attention，而是**大部分层用线性注意力、少数层保留 full-attention**——以
`Qwen3.6-35B-A3B` 为例，40 层里 **30 层是 GatedDeltaNet（GDN）线性注意力**，只有 10 层是标准
full-attention。

GDN 层的核心计算，是 `flash-linear-attention`（下称 **fla**）库里的
[`chunk_gated_delta_rule`](https://github.com/fla-org/flash-linear-attention)——一个用 Triton
写的 **chunked 并行扫描** kernel。它把序列切成固定大小 64 的 chunk，chunk 内用矩阵乘并行、chunk
间顺序传递一个 `K×V` 的递归状态。

**为什么值得优化：** 在 RL / 预训练里这个 kernel 占比很高——30/40 层，训练一步的前向要跑 3 次
（策略、参考、旧策略）外加一次反向。与 full-attention 不同：full-attention 走 FlashAttention /
FlashInfer，已被反复调优到接近硬件峰值；而 GDN 这条线的 `fla` 官方 kernel 是**通用实现**，没有针对
本题的具体形状（每卡 8 head、head_dim 128、chunk 64）做过专门调优，留有优化空间——适合交给自动化
kernel 优化 Agent 迭代。

**困难：**
1. **并行 × 顺序的平衡**：chunk 内是可并行的矩阵运算（算力友好），chunk 间是顺序的状态传递（延迟敏感），
   两者的 tiling / 访存 / 流水线安排相互牵制。
2. **数值敏感**：递归状态用 fp32 累加，delta-rule 的秩一更新会把误差沿序列累积——**快而不对没有意义**，
   必须通过严格的数值等价门。
3. **可微**：要真正进训练，前向和反向都得对；而 GDN 的**反向计算量约是前向的 3.7×**，是训练侧收益的主体。

**想达到的效果：** 一个在 4090 上比 `fla` 官方 chunk kernel 更快、数值等价、支持 autograd 的
`chunk_gated_delta_rule` 替代实现——拿来就能替换训练里的 GDN 层。

### 1.1 推荐工作方式：用 AI 优化 kernel（AKO）

> 💡 **本赛道鼓励——但不强制——用 AI 来做 kernel 优化。** 你既可以 **AI 辅助**（把 profiling、瓶颈
> 分析、kernel 改写交给编码 Agent 迭代），也可以让 **AI 完全自主优化**：读代码 → 改 kernel →
> 跑 `eval.py` → 按反馈继续迭代。

特别推荐使用本赛道的自动化 kernel 优化工具 **AKO（Automated Kernel Optimization）**，它驱动编码
Agent 在真机上自主迭代 kernel：

- 项目主页：<https://tongminglaic.github.io/AKO/>
- 开源工具：<https://github.com/TongmingLAIC/AKO4ALL>

评测只看最终 `submission.py` 的正确性与速度，**不限制**你用什么方式得到它——手写、AI 辅助、AI 自主皆可；
可在 `optimization.md` 中简要写明优化过程与所用工具。

---

## 2 赛题描述

### 2.1 大致思路

实现一个函数（Triton / CUDA / CUTLASS 任意技术栈）：

```python
def gdn_chunk_scan(q, k, v, g, beta, scale) -> o
```

其语义必须**等价于** `fla.ops.gated_delta_rule.chunk_gated_delta_rule(..., use_qk_l2norm_in_kernel=True)`。

- **前向（必做）**：直接对应推理 / prefill 场景，也是训练前向的一部分。
- **反向（bonus）**：让 `o.backward()` 的梯度能正确回传到 `q/k/v/beta`。训练里反向 ≈ 前向的 3.7×，
  是收益大头。只优化前向、反向交回 `fla` 兜底，也能拿到前向分。

计算定义（每个 head，状态 `S ∈ R^{K×V}`，沿时间步 t 递归）：

```
qt, kt = L2normalize(q_t), L2normalize(k_t)     # kernel 内做 L2 norm
a  = exp(g_t)                                    # 门控衰减，标量 ∈ (0,1)
S  = a · S                                       # 门控衰减
S  = S + beta_t · kt ⊗ (v_t − ktᵀ·S)            # delta-rule 秩一更新
o_t = scale · qtᵀ · S                            # 输出
```

### 2.2 数据范围与运行场景

- **硬件**：比赛平台为每名选手提供单张 RTX 4090 24GB 算力，不使用多卡并行。
- **软件环境**：正式名称和版本见 `INSTALL.md`；文档不依赖主办方机器上的本地路径或环境变量。
- **固定结构**（不可改）：`H = Hv = 8` 个 head，`head_dim K = V = 128`，`chunk = 64`。
- **dtype**：`q/k/v/o` 为 **bfloat16**；门控 `g`、`beta` 与递归状态为 **float32**。
- **输入分布（声明域，合法性以此为准）**：`q, k, v ~ N(0,1)`；`g = logsigmoid(N(0,1))`（log-decay，`exp(g)∈(0,1)`）；
  `beta = sigmoid(N(0,1)) ∈ (0,1)`（生成实现见 `reference.py::make_inputs`）。正确性须在**从该分布重新抽样**的
  输入上成立（组织方复测会换 seed 重抽）；针对该分布 + 下表 shape/dtype 的特化是合法的（见 2.6）。
- **输入接口**：

  | 张量 | dtype | 形状 | 含义 |
  |---|---|---|---|
  | `q` | bf16 | `[B, T, H, K]` | query（kernel 内 L2 norm） |
  | `k` | bf16 | `[B, T, H, K]` | key（kernel 内 L2 norm） |
  | `v` | bf16 | `[B, T, Hv, V]` | value |
  | `g` | fp32 | `[B, T, H]` | log-decay 门控，`exp(g)∈(0,1)` |
  | `beta` | fp32 | `[B, T, H]` | delta-rule 更新强度 `∈(0,1)` |
  | `scale` | float | 标量 | `= K**-0.5` |

  **输出** `o`：bf16 `[B, T, Hv, V]`。

- **评测形状** `(B, T)` 与权重（按生产相关度加权，长序列权重更高；与 `config.py::SHAPES` 完全一致）：

  | (B, T) | 权重 |
  |---|---|
  | (1, 2048) · (2, 2048) | 0.5 · 0.5 |
  | (1, 8192) · (2, 8192) · (4, 8192) | 1.0 · 1.0 · 1.0 |
  | (1, 16384) · (2, 16384) | 2.0 · 2.0 |
  | (1, 32768) | 1.5 |

  （`T=16384` 最贴近生产训练每卡的工作量。）

### 2.3 正确性标准与评测方法

- **golden**：`fla` 的 `fused_recurrent_gated_delta_rule` 跑在 **fp32**——recurrent 形式是 chunk
  形式的数学参考。评测另附一份**不依赖 fla 的自包含 fp32 递归**（`reference.py::naive_recurrent_reference`），
  供组织方在干净环境 `python reference.py --calibrate` 核对 golden 未被篡改、并在小形状上人工抽查。
- **容差自校准**：**逐形状**测 `fla` **自己的** chunk(bf16) 对 recurrent(fp32) 的参考差距 `ref_gap`，
  该形状通过阈值 = `max(2e-2, 2×ref_gap)`——即「一个正确的 bf16 实现本就该有的误差」的 2 倍（逐形状校准
  避免用单形状的差距跨形状误杀）。
- **判据**：前向 output 的相对 L2 误差 < 阈值 **且** 余弦相似度 > 0.999；反向对 `fla` chunk(**bf16**)
  的梯度（与提交同为 bf16 口径），相对 L2 误差 < 反向阈值。
- **不得改变数学**：如采用任何近似（不同 chunk 策略、低精度中间态等），须在 `optimization.md` 说明，且仍须过上述门。

### 2.4 性能标准与评测方法

- **基线**：`fla` 官方 `chunk_gated_delta_rule`（bf16）。
- **测量**：CUDA Event 计时，warmup 20；计时 **50 次/轮 × 5 轮，取各轮均值的中位数**（抗集群噪声/降频漂移）；
  每轮之间在**计时区外**刷新输入值、循环内轮换多个输入（见 2.6）；分别测 **前向** 与 **前向+反向** 时延。
- **加速比** = 基线时延 / 提交时延（>1 即更快）；按形状权重聚合。
- 一条命令即出结果：`python eval.py --submission submission.py`（同时打印正确性与性能）。

### 2.5 其他需求

- 提交物：`submission.py`（以及它引用的自建源码）和简短的 `optimization.md`。说明文档自由描述优化思路；如使用题目环境未列出的依赖，请写明名称、版本和安装方法。
- 允许 Triton / CUDA / CUTLASS / 手写 PTX 等任意实现；只要在单张 4090 上更快且数值等价。

`eval.py`、`reference.py`、`baseline.py` 和 `config.py` 构成统一评测链路。正式复测时，主办方会用组织者版本覆盖这些文件和测试配置；修改本地评测文件不会改变正式成绩。`benchmark.json` 由 `eval.py` 生成，不需要也不允许手工填写。

### 2.6 评测口径与合法性边界（重要）

**分数估计的是什么**：本题分数估计的是「从声明的输入分布上（近似）**独立抽取**的输入的中位延迟」。
为此计时**逐轮刷新输入值**（每轮换一批新随机值，轮内再轮换多个不同 data_ptr 的输入），而不是
「在同一份固定输入上一直重复计时」——刷新在计时区外，不计入时延。

**完全合法（不算作弊，评测不会误杀）**——对**声明域**的任何特化：
shape / stride / dtype 特化代码、预编译 PTX/cubin、autotune、CUDA graph、持久 workspace / plan 复用、
静态权重预处理——只要它对声明域里的每个输入都算出正确结果。**你可以放心针对本题固定的形状做专门优化。**

**唯一红线**：持久状态可以依赖**声明域**（表 2.2 的 shape/dtype），**不可以依赖某一次计时的实际输入值**——
即不得把「依赖输入值的输出或中间结果」跨独立 trial 缓存 / 复用 / 以快照返回。

**对所有选手自动执行的硬检测（按上面这条红线设计，合法 kernel 零误杀）**：
1. 每个形状用**多个随机 seed** 验前向正确性；golden 先在纯净输入上算好、选手拿输入的拷贝，故 in-place
   改输入也骗不过。
2. 计时**逐轮刷新输入值**、并在计时后用**同一缓冲区、新随机值**再验一次前向（缓存/快照类作弊会对
   「值已变」的输入给出旧答案而被抓）。
3. 赛后在组织方**同一台复测机、同一 session（固定时钟/散热）**上重测：先 `--make-baseline` 生成一份
   **统一 baseline** 做分母（缺形状即拒，不静默现测），再对每份提交 `--submission ... --baseline ...` 评分，
   并换 seed（`--seed`）/ 换形状（改 `config.py::SHAPES`）⇒ **主变量 = 选手 kernel**，保证跨选手排名公平
   （选手自测时 baseline 是各自当场测的，仅供自参考、不用于排名）。
   > baseline 与提交在不同时刻测，时钟/散热/后台负载仍是残留噪声源；组织方复测建议锁频、同 session 连测、
   > 打乱提交顺序，必要时周期性重测 baseline 做校核。机器指纹只校验 GPU 型号 + 库版本，同型号同版本视为同机。

**留给获奖名单确定时的人工 / profiler 复核（不对所有选手自动跑，以免误杀）**：执行来源核查（是否借道外部库
冒充自研 kernel）、异步计时深度审计、**进程级隔离复跑**（自动分在同进程内执行选手代码，不防篡改 golden 一类
投毒；入围方案必须隔离复跑坐实）——仅对入围方案单独深入 review。

---

## 3 评分标准

设每个形状 `i` 的权重为 `w_i`，正确性通过后取加速比，否则该项计 0：

```
shape_score_i = 0.4 · fwd_speedup_i  +  0.6 · fwdbwd_speedup_i
总分 S        = Σ_i w_i · shape_score_i  /  Σ_i w_i
```

- **正确性门（硬）**：某形状**前向**未过 → 该形状 `fwd_speedup=0` **且** `fwdbwd_speedup=0`；前向过但**反向**未过
  → 仅 `fwdbwd_speedup=0`（即 fwd+bwd 项要求前向、反向都过）。只做前向、反向交回 fla 兜底的选手，反向仍算过，
  可凭 0.4 权重的前向分 + 前向提速带来的 fwd+bwd 部分收益参赛。
- **性能分**：总分 `S`。`S>1.0` 表示整体快于 `fla` 官方 kernel；`S` 越高越好。反向权重（0.6）高于前向（0.4），
  因为训练里反向是主体。
- **稳健性**：换 seed / 换形状重测后总分不得显著下降（防 hack 测试点）。

> 名次先通过正确性门筛选，再按正式复测得到的性能指标排序。`optimization.md` 用于技术交流，不由评测程序计分。

---

## 4 参考文献

1. Yang et al., *Gated Delta Networks: Improving Mamba2 with Delta Rule*, ICLR 2025.
   https://arxiv.org/abs/2412.06464
2. Yang et al., *Parallelizing Linear Transformers with the Delta Rule over Sequence Length*
   （DeltaNet 的 chunked 并行化）. https://arxiv.org/abs/2406.06484
3. **flash-linear-attention** 仓库（`fla.ops.gated_delta_rule`）:
   https://github.com/fla-org/flash-linear-attention
4. Qwen3-Next（混合线性注意力 + full-attention 架构说明）.
5. Triton 官方文档：https://triton-lang.org
6. **AKO（Automated Kernel Optimization）** — 本赛道推荐的自动化 kernel 优化工具（见 1.1）：
   主页 https://tongminglaic.github.io/AKO/ ；开源 https://github.com/TongmingLAIC/AKO4ALL

---

## 附：随包代码

见同目录 `config.py` / `reference.py` / `baseline.py` / `submission.py` / `eval.py`
与 `requirements.txt` / `INSTALL.md` / `optimization.md`。选手直接修改 `submission.py`，
`python eval.py --submission submission.py --json-out benchmark.json` 可得到正确性、性能和机器生成的结果文件。**环境版本敏感，动手前先读 `INSTALL.md`。**
