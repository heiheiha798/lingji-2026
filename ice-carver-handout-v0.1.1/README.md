# Ice Carver：单卡 GPU 多等值面提取优化

|项目|说明|
|---|---|
|赛题类型|CUDA / HPC 性能优化|
|目标设备|单张 NVIDIA GeForce RTX 4090（`sm_89`）|
|固定环境|Ubuntu 22.04、CUDA Toolkit 12.8、CMake 3.22+、Ninja、C++17|
|运行命令|`python3 runtest.py`|
|提交代码|仅 `handout/src/solution.cu`|
|结果文件|`results/public/summary.json`|
|计分构成|正确性 40 分 + Baseline 相对性能 30 分 + 最终排名 30 分|

## 1. 任务说明

给定一个三维 `float32` 标量场与 1 至 8 个等值面阈值，请在单张 RTX 4090 上用 CUDA 实现经典 Marching Cubes，输出每个阈值对应的三角形数组。目标是在保证结果正确、顺序确定、无越界访问的前提下，尽量缩短 GPU 求解时间。

你只需实现公开函数 `icecarver_solve`。正式评测会把提交的 `handout/src/solution.cu` 放入干净的官方源码树，使用隐藏 seed 重新构建和运行；本地生成的 JSON 不是正式成绩凭证。

## 2. 数学定义

体素尺寸记为 $n_x\times n_y\times n_z$，其中 $x$ 维是连续存储的最快变化维。体素编号为：

$$
\operatorname{voxel\_id}(x,y,z)=(z n_y+y)n_x+x.
$$

单元尺寸为 $(n_x-1)\times(n_y-1)\times(n_z-1)$。令 $c_x=n_x-1$、$c_y=n_y-1$，则单元编号为：

$$
\operatorname{cell\_id}(x,y,z)=(z c_y+y)c_x+x.
$$

对单元 $c$、等值面阈值 $\tau$，角点值为 $v_{c,j}$。角点只能按严格小于进行分类，等于阈值时属于外部：

$$
\operatorname{case}(c,\tau)=\sum_{j=0}^{7}\mathbf{1}[v_{c,j}<\tau]2^j.
$$

相交边两端为 $(p_0,v_0)$ 与 $(p_1,v_1)$ 时，顶点插值固定为：

$$
p=p_0+\frac{\tau-v_0}{v_1-v_0}(p_1-p_0).
$$

必须使用 `include/icecarver/mc_tables.cuh` 中冻结的经典 256-case 查表，不得替换拓扑规则或自行加入 epsilon。

## 3. 接口说明

```cpp
extern "C" int icecarver_solve(
    const icecarver::Input* input,
    icecarver::Output* output,
    void* workspace,
    std::size_t workspace_bytes,
    cudaStream_t stream);
```

- `input->volume`、`input->isovalues`、`output->triangle_counts`、各个 `output->triangles[iso]` 与 `workspace` 都位于 device memory。
- `Input`、`Output` 描述对象以及 `output->capacities[]` 位于 host memory。
- `output->triangle_counts` 是含 `icecarver::kMaxIsovalues`（8）个元素的 device 数组；当前 case 只校验前 `num_isovalues` 个计数。
- 每个 iso 拥有独立的三角形数组与容量，禁止把多个 iso 合并为一个共享输出数组。
- 必须使用传入的 CUDA stream；函数返回后不得残留继续写输出的后台工作。
- `emit_triangles == 0` 时只需给出正确计数；非零时必须输出完整三角形数组。

完整字段、状态码和 workspace 描述符以 `include/icecarver/api.h` 为准。

## 4. 正确性与确定性

每个 iso 的输出顺序必须为：

1. 按 `cell_id` 升序；
2. 同一单元内按 `kTriangleTable[case]` 中每连续三个边号的顺序；
3. 三角形的三个顶点按查表边号顺序。

评测器会检查所有输出计数、容量、全量浮点值、NaN/Inf、CUDA 错误和输出缓冲区前后的 guard。浮点值在冻结的绝对/相对容差下比较。任一正式变体失败，则该测试点不获得正确性分和该点性能分。

## 5. 测试点

|ID|典型尺寸|阈值数|公开场类型|每 iso 容量|正确性分|性能分|
|---|---:|---:|---|---:|---:|---:|
|P0|`64³`|1|sphere|250,000|4|0|
|P1|`96³`|1|metaball|1,000,000|4|0|
|P2|`160³`|1|gyroid|4,000,000|5|0|
|P3|`256×192×160`|1|mixed|4,000,000|5|0|
|P4|`256³`|2|multiscale|6,000,000|5|5|
|P5|`320×256×224`|4|dense|8,000,000|5|7|
|P6|`384³`|4|mixed|10,000,000|6|8|
|P7|`512×384×320`|8|dense|16,000,000|6|10|
|合计|||||40|30|

每个测试点运行 2 次预热和 5 次正式测量。5 次正式测量使用 5 个不同的派生 seed，而不是反复计时同一份输入。公开配置用于调试；正式环境会在不改变算法语义、权重与计时边界的前提下使用隐藏变体。

## 6. 计时规则

生成输入、分配内存、验证结果和写 JSON 均不计时。单次求解同时记录 CUDA Event 与完成 device synchronization 的宿主单调墙钟时间：

$$
t=\max(t_{\mathrm{event}},t_{\mathrm{wall}}).
$$

测试点时间 $T_i$ 是 5 次正式测量的中位数。每个 evaluator 进程外还有宿主硬超时；超时、CUDA 错误、非零状态或错误输出都会使该测试点无效。

## 7. 评分规则

### 7.1 正确性 40 分

P0 至 P7 的正确性权重见测试矩阵。一个测试点的所有正式变体全部通过才取得该点的全部正确性分。

### 7.2 Baseline 相对性能 30 分

只有 P4 至 P7 计性能。设同一正式环境中的公开 baseline、私有 target 与选手时间分别为 $T_{\mathrm{base},i}$、$T_{\mathrm{target},i}$、$T_i$，性能权重为 $p_i$：

$$
A_i=p_i\,\operatorname{clamp}\left(
\frac{\log T_{\mathrm{base},i}-\log T_i}
{\log T_{\mathrm{base},i}-\log T_{\mathrm{target},i}},0,1\right).
$$

发布时冻结的 RTX 4090 参考 baseline 中位数如下；平台复测记录是最终校准依据：

|测试点|P4|P5|P6|P7|
|---|---:|---:|---:|---:|
|Public baseline / ms|1.429464|2.880538|5.497278|16.607520|

公开 baseline 通过全部测试点时得到正确性 40 分；由于它与自身校准时间相同，Baseline 相对性能为 0 分，即本地结果分为 40/70。剩余 30 分由最终排名产生，不在单份本地 JSON 中计算。

### 7.3 最终排名 30 分

只有 P4 至 P7 全部正确的提交参与排名；P0 至 P3 失败仍会失去对应正确性分。排名指标为相对 baseline 加速比的性能权重加权几何平均：

$$
G=\exp\left(
\frac{\sum_i p_i\log(T_{\mathrm{base},i}/T_i)}{\sum_i p_i}
\right).
$$

按 $G$ 降序排名。有效提交数为 $N$、名次为 $r$（第一名为 1）时，$N=1$ 得 30 分；$N>1$ 时：

$$
R=30\frac{N-r}{N-1}.
$$

最终总分为 $S=C+A+R$。

## 8. 运行与提交

```bash
# 唯一公开评测入口，与参考赛题目录格式一致
python3 runtest.py

# 快速检查环境与最小测试点
bash scripts/smoke_test.sh
```

公开入口会以 `ICECARVER_ENABLE_TARGET=OFF` 构建，并把各测试点证据写入 `results/public/cases/`，汇总写入 `results/public/summary.json`。

提交前：

1. 只修改 `handout/src/solution.cu`；
2. 可在 `SOLUTION.md` 说明方法、资源使用和本地结果；
3. 删除自行产生的无关大文件，不要提交 build/results 缓存；
4. 按平台要求上传源码 tar 包或基于组委会共享镜像提交。

## 9. 禁止行为

- 使用 VTK、CGAL 或其他现成等值面提取实现；
- 修改评测器、生成器、公开 ABI、配置、计时器或查表；
- 针对公开 seed、尺寸、文件名或答案硬编码；
- 通过网络、文件、其他进程或残留镜像文件取得隐藏答案；
- 越界访问、未定义行为、破坏 guard，或让工作跨越计时/验证边界；
- 在学生包或学生镜像中保留 `config/private/`、`reference/cuda_target.cu`、隐藏 seed 或命题人原始校准结果。

更细的算法边界、题面与测量协议分别见 `docs/algorithm.md`、`docs/problem.md`、`docs/testcases.md`。
