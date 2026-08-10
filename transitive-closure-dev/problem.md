# 有向图传递闭包

## 背景

在依赖分析、调用图、访问控制和数据血缘中，人们经常需要回答同一个问题：从节点 `u` 出发，能否经过若干条有向边到达节点 `v`？当查询数量很大时，可以预先计算图的传递闭包，将后续可达性查询转化为一次 bit 查询。

传递闭包的定义简单，但完整输出包含 `V^2` 个布尔值。算法既要处理逐步扩张的可达关系，又要控制大规模矩阵的访存和同步。不同图可能是 DAG、包含大量强连通分量，或者迅速变成稠密闭包，因此不存在对所有输入都占优的单一稀疏策略。

## 任务

给定一个包含 `V` 个顶点的有向图，顶点编号为 `0..V-1`。请计算反身传递闭包 `R`：

```text
R[u, v] = 1  当且仅当存在一条从 u 到 v 的有向路径
```

长度为 0 的路径也计入，因此所有 `R[u, u]` 必须为 1。

## 数据表示

输入邻接矩阵和输出闭包均按行进行 64 位 bit packing：

```text
words_per_row = ceil(V / 64)
bit(u, v) = matrix[u * words_per_row + v / 64] 的第 (v mod 64) 位
```

输入中 `bit(u, v)=1` 表示存在边 `u -> v`。每行最后一个 `uint64_t` 中超出顶点范围的高位恒为 0，输出也必须保持为 0。

## 工程与接口

选手可以任意修改 `solution/` 工程及其 Makefile。评测脚本要求工程生成：

```text
build/submission/libclosure_submission.so
```

动态库必须实现 `interface/closure_api.h` 中的接口：

```cpp
extern "C" int closure_run(
    const std::uint64_t *adjacency,
    std::uint64_t *reachability,
    int vertices,
    int words_per_row);
```

约定如下：

- `adjacency` 和 `reachability` 都位于 CPU 内存；
- `adjacency` 只读，`reachability` 必须被完整覆写；
- 选手代码自行完成 CUDA Context 初始化、显存分配、输入上传、GPU
  计算、结果下载、同步和资源释放；
- 函数返回时 CPU 输出必须已经可用，不得遗留后台任务继续修改输出；
- 返回 0 表示成功，非 0 表示本次任务失败；
- 可以任意增加源码文件和内部接口，评测器只检查最终动态库 ABI。

## 数据范围

- `1 <= V <= 65536`；
- 最大 bit-packed 矩阵为 512 MiB；
- 输入图可以包含自环、环、多个强连通分量或孤立点；
- 正确性采用 `uint64_t` 逐元素精确比较。

## 计时

计时覆盖一次完整的 `closure_run` 调用，包括 CUDA Context 初始化、显存
分配和释放、H2D/D2H、kernel、CPU/GPU 协调、同步和输出生成。图生成、
动态库构建和加载、reference 以及结果比较不计时。

固定性能点在 3 个独立评测进程中各运行一次，不进行预热，也不复用前一
次调用的 CUDA Context、显存或内部状态。每次均验证正确性，成绩使用
3 次完整任务时间的中位数；reference 也用相同冷启动口径独立测量。

## 测试方式

### 随机功能测试

一次随机功能测试包含 layered DAG、block SCC、随机稀疏图、grid DAG 和混合图五种中小规模数据。它使用 CPU 与 GPU reference 交叉验证，只输出 `PASS/FAIL` 和复现 seed，不输出任何性能数据。

```bash
python3 run_judge.py --random-only --random-seed 987654321
```

### 固定性能测试

固定输入由公开生成器、参数和种子在线生成，标准输出由可信 GPU reference 在线计算。

| 编号 | 名称 | 顶点数 | 结构 | 主要考察 | 评分 | 权重 | 目标时间 |
|---:|---|---:|---|---|---|---:|---:|
| 0 | tiny-correctness | 257 | 小型混合图 | 基本正确性 | 正确即得 | 10 | - |
| 1 | layered-dag | 16384 | 128 层 DAG | 规则层次传播 | 固定目标 | 20 | 250 ms |
| 2 | block-scc | 32768 | 64 点 SCC 块组成的 DAG | 局部稠密闭包 | 开放竞技 | 20 | 不设 |
| 3 | random-sparse | 32768 | 平均出度 8 | 稀疏输入、稠密结果 | 开放竞技 | 20 | 不设 |
| 4 | grid-dag | 16384 | 128 x 128 网格 DAG | 结构化可达区域 | 开放竞技 | 15 | 不设 |
| 5 | large-mixed | 65536 | SCC 与前向边混合 | 大矩阵与访存局部性 | 开放竞技 | 15 | 不设 |

## 评分

结果错误的测试点成绩为 0。固定目标组与开放竞技组的名义权重为 `30:70`。

`tiny-correctness` 正确即获得 10 权重。`layered-dag` 的目标时间为
`g = 250 ms`，该值在正式 RTX 4090 环境中按上述冷启动完整任务口径校准。
选手中位时间为 `t`：

```text
score = 20 * min(1, g / t)
```

其余四点不设满分时间。设权重为 `w`，同机 reference 时间为 `r`：

```text
performance_index = w * (r / t)^(1/4)
```

开放竞技指数不封顶。正式分数由比赛平台计算，`run_judge.py` 只输出正确性和性能数据。

## 提交内容与公平性

个人参赛选手提交完整 `solution/`，包括 Makefile、全部源码、`optimization.md` 和完整评测生成的 `benchmark.json`：

```bash
python3 run_judge.py --json-output solution/benchmark.json
```

`benchmark.json` 由脚本完整生成并覆盖写入，不需要也不得手工修改。`optimization.md` 自由说明优化思路即可；若使用额外依赖，再写明依赖名称、版本和安装方法。

`interface/closure_api.h`、`run_judge.py` 和 `judge/` 属于统一评测链路。正式复核会使用组织者版本覆盖这些内容。选手可以自由修改 `solution/`、增加任意源码或使用可复现依赖。

动态库加载不计时，因此不得在全局/静态构造函数、加载回调或其他
`closure_run` 之外的路径中初始化 CUDA、分配 GPU 资源、编译实现、处理
输入或预先计算输出。完成当前任务所需的全部工作必须发生在本次
`closure_run` 调用内。

## 运行

```bash
python3 run_judge.py --case 0 --repeats 1
python3 run_judge.py --quick
python3 run_judge.py
```

软件版本、Profiler 和常见问题见 `README.md`。

## 参考资料

1. S. Warshall, A Theorem on Boolean Matrices, Journal of the ACM 9(1), 1962.
2. T. H. Cormen et al., Introduction to Algorithms, 关于传递闭包和 Floyd-Warshall 算法。
3. NVIDIA CUDA C++ Programming Guide，关于共享内存、bit 操作和 kernel 调度。
