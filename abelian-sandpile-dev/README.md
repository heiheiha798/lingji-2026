# Abelian Sandpile Handout

本目录是可直接下发的赛题工程。题面见 `problem.md`，选手可以自由修改整个 `solution/`。

## 目录结构

```text
handout/
├─ problem.md
├─ README.md
├─ run_judge.py
├─ demo.py
├─ interface/sandpile_api.h
├─ judge/                         固定评测链路
└─ solution/                      选手工程
   ├─ Makefile
   ├─ src/
   ├─ include/
   └─ optimization.md
```

公开基线是一个能够通过正确性测试的多文件 CUDA 工程。可以重写 Makefile、增加或删除源码，也可以使用完全不同的稳定化算法。

## 软件与版本

开发和正式复核环境使用：

- Ubuntu 22.04 LTS，Linux x86-64；
- NVIDIA GeForce RTX 4090 24 GiB；
- CUDA Toolkit 12.8，目标架构 `sm_89`；
- Python 3.10；
- GNU Make 4.3；
- GCC/G++ 11，支持 C++17。

性能分析可选使用 Nsight Systems 2024.6 和 Nsight Compute 2025.1。公开基线不需要 Python 第三方包或 CMake。

可以用以下命令确认软件版本：

```bash
nvidia-smi
nvcc --version
python3 --version
make --version
g++ --version
```

## 构建契约

`run_judge.py` 会调用 `solution/Makefile`，并提供构建目录、接口目录和 CUDA 编译器信息。Makefile 必须生成：

```text
build/submission/libsandpile_submission.so
```

动态库必须导出 `sandpile_run`。它接收 CPU 输入和输出，一次调用完成显存
分配、数据传输、GPU 计算、结果下载和资源释放。评测器不限制选手工程
内部有多少文件或如何组织源码。

## 运行评测

```bash
# 最小固定点
python3 run_judge.py --case 0 --repeats 1

# 五类随机功能测试，不输出性能
python3 run_judge.py --random-only

# 使用指定 seed 复现随机测试
python3 run_judge.py --random-only --random-seed 184725193

# 单个固定性能点，默认运行 3 次取中位数
python3 run_judge.py --case 2

# 随机测试和前三个固定点
python3 run_judge.py --quick

# 完整评测
python3 run_judge.py
```

脚本默认重新构建动态库。确认源码和构建产物均未变化时，可以添加 `--no-build`。

## JSON 自测记录

```bash
python3 run_judge.py --json-output solution/benchmark.json
```

`benchmark.json` 完全由脚本生成并覆盖写入，不需要也不应手工编辑。它包含：

- 运行时间、命令和软硬件环境；
- 随机功能测试的正确性、规模和复现 seed；
- 固定性能点的中位数、原始样本、reference 时间和加速比；
- 整体 `PASS/FAIL` 状态。

随机功能测试只记录正确性，不记录性能。

## 可修改范围

可以自由修改 `solution/` 下的全部内容，包括 Makefile，并增加任意源码、头文件或其他构建文件。

以下内容属于统一评测链路，正式复核时会使用组织者版本覆盖：

- `interface/sandpile_api.h`；
- `run_judge.py`；
- `judge/` 下的测试生成器、reference、runner 和固定参数。

## 优化说明

提交中的 `solution/optimization.md` 只需自由说明：

- 采用的算法和主要优化思路；
- 使用了哪些额外依赖；如有，写明名称、版本和安装方法。

如果没有额外依赖，写“无”即可。不要求固定格式、逐项性能报告或指定篇幅。

## Makefile 快捷命令

```bash
make random      # 随机功能测试
make quick       # 随机测试和前三个固定点
make evaluate    # 完整评测
make report      # 完整评测并生成 solution/benchmark.json
make demo        # 生成小规模沙堆图像
make clean       # 删除 build/
```

## 性能分析

先运行一次评测完成构建，再直接分析 runner：

```bash
nsys profile -o sandpile_case2 \
  ./build/judge 2 ./build/submission/libsandpile_submission.so candidate -

ncu --set basic \
  ./build/judge 2 ./build/submission/libsandpile_submission.so candidate -
```

上述命令只分析选手完整任务；`run_judge.py` 会在独立进程运行 reference 并
比较临时输出文件。Profiler 开销不计入正式成绩。

建议关注真正活动格点比例、全局访存、原子冲突、队列管理、kernel 启动和 CPU/GPU 同步。

## 提交内容

个人参赛选手提交完整 `solution/`，至少包含：

- 能生成规定动态库的 Makefile；
- 全部源码、头文件和需要随提交携带的依赖；
- `optimization.md`；
- 由脚本生成的 `benchmark.json`。

身份信息由比赛平台记录，不写入 `benchmark.json` 或 `optimization.md`。

## 常见问题

### 找不到 `nvcc`

确认已经安装 CUDA Toolkit 12.8，而不只是 NVIDIA 驱动，并确认 `nvcc --version` 可以正常运行。

### 动态库缺少接口

`sandpile_run` 必须使用 `extern "C"`，并与 `interface/sandpile_api.h`
完全一致。

### 固定点运行时显存不足

显存由选手实现自行管理。检查是否释放了前一阶段不再使用的缓冲区，并按
RTX 4090 的 24 GiB 显存规模设计算法。

### 固定点通过、随机测试失败

使用输出的 suite seed 复现，重点检查边界、非规则尺寸、重复入队、终止检测和整数溢出。
