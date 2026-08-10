# 正式题面

## 1. 任务

给定一个三维 `float32` 标量场和 1 至 8 个等值面阈值，在单张 NVIDIA GeForce RTX 4090 上使用 CUDA 实现经典 Marching Cubes，输出每个阈值对应的三角形数组。目标是在保持结果正确、确定且不越界的前提下缩短 GPU 求解时间。

选手只修改 `handout/src/solution.cu`，实现公开 ABI：

```cpp
extern "C" int icecarver_solve(
    const icecarver::Input* input,
    icecarver::Output* output,
    void* workspace,
    std::size_t workspace_bytes,
    cudaStream_t stream);
```

具体类型、状态码和 Workspace 描述符以 `include/icecarver/api.h` 为准。输入、阈值、输出与 workspace 指针均指向 device memory；描述对象和 `capacities[]` 位于 host memory。

## 2. 数据布局

体素尺寸为 `nx × ny × nz`，其中 `x` 为最快变化维：

```text
volume_index(x,y,z) = (z * ny + y) * nx + x
```

单元尺寸为 `(nx-1) × (ny-1) × (nz-1)`，单元编号为：

```text
cell_id(x,y,z) = (z * (ny-1) + y) * (nx-1) + x
```

合法范围是 `0 <= x < nx-1`、`0 <= y < ny-1`、`0 <= z < nz-1`。顶点坐标输出为体素索引坐标，不是生成器用于计算场值的归一化坐标。

八角点编号固定为：

|角点|偏移 `(dx,dy,dz)`|角点|偏移 `(dx,dy,dz)`|
|---:|---|---:|---|
|0|`(0,0,0)`|4|`(0,0,1)`|
|1|`(1,0,0)`|5|`(1,0,1)`|
|2|`(1,1,0)`|6|`(1,1,1)`|
|3|`(0,1,0)`|7|`(0,1,1)`|

十二条边端点依次为：`(0,1)`、`(1,2)`、`(2,3)`、`(3,0)`、`(4,5)`、`(5,6)`、`(6,7)`、`(7,4)`、`(0,4)`、`(1,5)`、`(2,6)`、`(3,7)`。

## 3. 分类与插值

角点分类只能使用：

```cpp
inside = value < isovalue;
```

等于阈值时属于外部，不得自行加入 epsilon。case 编号的第 `c` 位对应角点 `c`。题目使用 `mc_tables.cuh` 给出的经典 256-case Lorensen–Cline 查表，严格按表处理，不做渐近判别或其他拓扑歧义消解。

对查表指定边的两个端点，插值固定为：

$$
p=p_0+\frac{\tau-v_0}{v_1-v_0}(p_1-p_0).
$$

测试数据不会要求对非相交等值边插值；结果不得包含 NaN 或 Inf。

## 4. 输出契约

每个等值面拥有独立的 `Triangle* triangles[iso]` 数组和独立容量 `capacities[iso]`，禁止把多个等值面合并为一个共享扁平数组。`triangle_counts` 是长度至少为 `kMaxIsovalues` 的 device 数组。

每个 iso 数组的顺序必须完全确定：

1. 按 `cell_id` 升序；
2. 同一单元内按 `kTriangleTable[case]` 每连续三个边号的顺序；
3. 每个三角形的三个顶点按表中边号顺序。

成功时返回 `kSuccess` 并写出所有有效计数。输入不合法、尺寸溢出、workspace 不足、输出容量不足或 CUDA 失败时返回对应 `Status`；不得越过任何容量写入。公开 baseline 会在生成前同步核对真实总数，并以 `kInsufficientOutput` 拒绝不足容量。正式配置均预留足够容量，该检查不改变题目数据规模。评测器会检查输出前后 guard、CUDA 错误、计数、有限性和浮点容差。

`emit_triangles == 0` 时只需正确计算每个 iso 的三角形数；非零时必须生成完整数组。实现必须使用传入的 stream，不得隐式依赖默认 stream，不得创建后台工作在函数返回后继续写输出。

## 5. 允许与禁止

允许 CUDA Runtime、CUDA 12.8 自带的 CUB/CCCL 基础并行原语、多个 kernel、CUDA Graph 与合理的临时 workspace。禁止：

- VTK、CGAL 或其他现成等值面提取实现；
- 图形 API 代替计算；
- 修改评测器、配置、计时或输出缓冲；
- 针对公开 seed、尺寸或答案硬编码；
- 从网络、文件或进程间通道取得隐藏答案；
- 越界访问、未定义行为或让验证工作在计时结束后继续改变输出。

## 6. 环境与提交

正式环境为 Ubuntu 22.04、RTX 4090、CUDA Toolkit 12.8、`sm_89`、CMake 3.22+、Ninja、C++17。正式构建固定 `-DICECARVER_ENABLE_TARGET=OFF`，并使用组委会签发的 `evaluator/official_manifest.json` 核对配置。选手提交 `solution.cu` 及组委会要求的镜像/哈希；本地 JSON 仅供反馈，最终成绩以组委会在隐藏变体上复跑为准。
