# 安装与运行

## 环境
- 比赛平台为每名选手提供单张 NVIDIA RTX 4090 24GB 算力。
- Ubuntu 22.04（x86-64），NVIDIA Driver 570.153.02，CUDA Toolkit 12.8。
- **Python 3.11**（不是 3.10——见下方「环境坑」第 1 条）。
- 实测组合：Python 3.11 + `torch==2.6.0`(cu124) + `triton==3.2.0` +
  `tilelang==0.1.12` + `flash-linear-attention==0.5.1`。

## 安装
```bash
conda create -n gdn --override-channels -c conda-forge python=3.11 pip -y
conda activate gdn
python -m pip install -r requirements.txt
# flash-linear-attention 若装不上，可从源码：
# pip install "flash-linear-attention==0.5.1"  # 或 git+https://github.com/fla-org/flash-linear-attention
```

显式使用 `conda-forge` 可避免新版 Miniconda 在非交互环境中因 Anaconda 默认渠道 TOS 未确认而中止。
完整环境约占 6GB，pip 下载缓存还会占约 6GB；建议预留至少 15GB 可用磁盘。

## ⚠️ 环境坑（动手前必看，都是实测踩过的）
1. **fla 0.5.1 必须 Python 3.11+**：py3.10 下 fla 的 `@triton.jit` kernel 会在 import 时炸
   （inspect 提取装饰器源码的方式不同）。用 py3.11。
2. **torch/triton 别用最新**：torch 2.13 / triton 3.7 过不了 fla 的 JIT。用 **torch 2.6 + triton 3.2** 稳
   （triton 3.2 启动会警告「建议升级到 3.3」，可忽略，能跑）。
3. **triton 3.2 静态检查严格**（自写 kernel 会踩）：① `@triton.jit` 内不能直接用模块级全局变量，
   要么声明成 `tl.constexpr`、要么传参、要么写字面量；② 同名变量在 if/else 不同分支必须类型一致
   （不同 shape 用不同变量名）。
4. **fla 的 chunk kernel 链不能干净地被 CUDA graph capture**（内部 autotune + 动态中间分配）——想上
   CUDA graph 得自己写全部 kernel、或用固定 buffer 且避开 fla 的 autotune 路径。
5. 当前优化提交会 import `tilelang`，必须安装 requirements 中固定的 0.1.12；它在首次遇到新的
   `(B,T)` 时 JIT 编译静态特化 kernel，编译发生在评测 warmup/正确性阶段，不进入 CUDA Event 计时。

## 自检（真机第一次务必跑）
```bash
# 1) 校准 golden：确认自包含 fp32 递归与 fla 的 recurrent 一致
python reference.py --calibrate       # 期望每行 rel_l2 < 1e-3 且 OK

# 2) 用随包默认实现（=fla 基线）跑通固定评测流程，性能指数应接近 1.0
python eval.py --submission submission.py --json-out benchmark.json
```

## 提交你的实现
1. 在 `submission.py` 里把 `gdn_chunk_scan` 换成你的 kernel（Triton/CUDA/CUTLASS 均可）。
2. 运行随机正确性测试；该命令不进行性能计时：
   ```bash
   python eval.py --submission submission.py --random-correctness --random-count 5
   ```
3. 运行固定性能点并由程序生成结果文件：
   ```bash
   python eval.py --submission submission.py --json-out benchmark.json
   ```
4. 换 seed 自查稳健性（赛后组织方会换 seed + 换形状在干净环境重测）：
   ```bash
   python eval.py --submission submission.py --seed 12345
   ```

> 自测口径：baseline 与你的 kernel **当场同机同时**测，得「相对 fla 的环境无关加速比」供迭代。
> 这个数**跨选手不保证可比**（各人机器不同），最终排名以组织方复测为准（下）。

## 组织方复测（公平排名，非选手用）

所有选手在**同一台复测机、同一 session**上评，共用一份 baseline 做分母 ⇒ 主变量 = 选手 kernel。

```bash
# ① 生成统一 baseline（此进程不 import 任何选手代码 ⇒ baseline 天然免投毒）
python eval.py --make-baseline baseline.json

# 先用原始 fla 模板做控制组；总分应接近 1.0，否则先处理锁频/温度/后台负载问题
python eval.py --submission submission.py --baseline baseline.json

# ② 逐份提交，共用同一分母（可换 seed 防 hack；换形状改 config.py::SHAPES 后重跑 ①②）
python eval.py --submission contestant_A/submission.py --baseline baseline.json --json-out contestant_A.json
python eval.py --submission contestant_B/submission.py --baseline baseline.json --seed 12345 --json-out contestant_B.json
```

- `baseline.json` 带**机器指纹**（GPU 型号 / torch / triton / fla / CUDA 版本）；换型号或换版本跑会被**拒绝**。
  注：指纹不含 GPU UUID/driver/clocks——同型号同版本的另一台会被视为同机。
- baseline 时延与随机值无关，换 seed 复测时 `baseline.json` 照常复用；但**缺某个形状会被拒绝**（不静默现测）。
- 减少残留噪声：**锁频**（`nvidia-smi -lgc`）、同 session 连测所有提交、打乱提交顺序，长批量中周期性重测 baseline 校核。
  每次生成 baseline 后先跑上述 fla 控制组；若总分明显偏离 1.0，应废弃该 baseline 并重新稳定 GPU 状态后测量。
- **入围（获奖候选）方案必须进程级隔离复跑**：自动分在同进程内执行选手代码，不防「篡改 golden」一类投毒。

## 文件说明
| 文件 | 作用 |
|---|---|
| `config.py` | 形状 / dtype / 容差 / 权重（选手与评测共用） |
| `reference.py` | golden（fla fused_recurrent fp32）+ 数据生成 + 自包含递归校准 |
| `baseline.py` | 性能基线（fla chunk bf16） |
| `submission.py` | **选手起始实现**，实现 `gdn_chunk_scan` |
| `eval.py` | **评测程序**，出正确性 + 性能 |
| `optimization.md` | 选手简要说明优化思路和额外依赖 |

正式复测会用组织者版本覆盖 `eval.py`、`reference.py`、`baseline.py` 和 `config.py`。选手可修改 `submission.py` 并增加它引用的源码文件；不要依赖对评测链路文件的修改。
