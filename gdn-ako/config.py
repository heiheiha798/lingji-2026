"""
赛题公共配置：形状、dtype、容差、评分权重。
选手与评测程序共用本文件，保证口径一致。

形状取自生产 Qwen3.6-35B-A3B 的单个 GatedDeltaNet(GDN)层、单卡（per-rank）的工作规模：
H = Hv = 8 个 head，head_dim = 128，chunk = 64。
（全模型 linear_num_key_head = 16，40 层里 30 层是 GDN；本题取单层 per-rank 的代表形状。）

一次前向 = 一个 GDN 层在一段序列上的 chunked 线性注意力扫描。
"""

import torch

# ---- 固定结构参数（不可改，改了就不是这个 kernel）----
NUM_HEADS = 8          # per-rank key/query head 数 (H)
NUM_V_HEADS = 8        # per-rank value head 数 (Hv)；本模型 H == Hv
HEAD_K = 128           # key/query head_dim
HEAD_V = 128           # value head_dim
CHUNK_SIZE = 64        # fla gated_delta_rule 的固定 chunk 大小

# ---- dtype 口径 ----
IO_DTYPE = torch.bfloat16     # q/k/v/o 的 dtype（生产口径）
STATE_DTYPE = torch.float32   # 递归状态/gate 的累加 dtype（mamba_ssm_dtype=float32）

# ---- 评测形状 (batch, seqlen)，按生产相关度加权 ----
# 生产训练每 rank（CP4）看到 ~16K token；这里覆盖 2K~32K。
# 权重偏向长序列——长序列是 kernel 成本与生产训练的主战场。
SHAPES = [
    # (B,   T,     weight)
    (1,   2048,   0.5),
    (2,   2048,   0.5),
    (1,   8192,   1.0),
    (2,   8192,   1.0),
    (1,  16384,   2.0),   # 最贴近 CP4 per-rank 生产点
    (2,  16384,   2.0),
    (1,  32768,   1.5),
    (4,   8192,   1.0),
]

# ---- 正确性容差 ----
# GDN chunked-scan 对数值敏感；容差以「fla 自己的 chunk(bf16) vs recurrent(fp32) 的
# 参考差距」自校准：pass 阈值 = max(下限, REL_TOL_MULT × 参考差距)。
FWD_REL_L2_FLOOR = 2e-2       # 前向 output 相对 L2 误差下限阈值
BWD_REL_L2_FLOOR = 5e-2       # 反向 梯度 相对 L2 误差下限阈值（梯度更噪）
REL_TOL_MULT = 2.0            # 相对 fla 参考差距放宽的倍数
COSINE_MIN = 0.999           # 前向 output 余弦相似度下限

# ---- 性能测量 ----
WARMUP_ITERS = 20
TIMED_ITERS = 50
TIMED_ROUNDS = 5             # 多轮，每轮均值，取中位——抗集群噪声/降频漂移
TIMING_POOL = 4              # 计时轮换的独立输入份数（不同 data_ptr）；与「逐轮刷新输入值」
                            # 合起来堵缓存/伪 graph 空子（机制见 README 2.6，合法 graph 不受影响）
# 反向在训练里约占 fwd 的 3.7×，故 fwd+bwd 权重更高（贴训练收益）
SCORE_W_FWD = 0.4            # 前向加速比在总分里的权重
SCORE_W_FWDBWD = 0.6        # fwd+bwd 加速比在总分里的权重（bonus track，未做则为 0）

# ---- 随机数据种子（下发用；赛后换 seed 重测防 hack）----
DEFAULT_SEED = 20260723

# ---- 防作弊（零误杀口径；机制与依据见 README 2.6）----
NUM_CORRECT_SEEDS = 3         # 每形状用多个 seed 验前向正确性（挡单点预存 / overfit）
FRESHNESS_RECHECK = True      # 计时后 in-place 刷新输入值再验一次（挡内容缓存 / 快照）
