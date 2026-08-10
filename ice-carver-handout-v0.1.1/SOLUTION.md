# Ice Carver 解题说明

> 本文件用于说明实现思路，正式计分代码仅为 `handout/src/solution.cu`。

## 1. 基本信息

- 队伍：
- 成员：
- 提交版本：
- 测试镜像/日期：

## 2. 算法与 CUDA 实现

分类 kernel 一次读取八个角点并处理全部等值面，生成 iso-major 的紧凑计数。
每个 iso 独立执行稳定的 exclusive scan，因此输出仍严格遵循 cell id 和冻结表
顺序。2/4/8-isovalue workload 会先保留全部 row offsets，再由对应的编译期模板
实例化一个 fused emit kernel：每个 warp 负责一条 y-z 行，把同一 cell 的计数
压入一个 64-bit 寄存器，只要任一表面活跃就读取一次八个角点，随后按 iso 顺序
生成三角形。这既复用体数据，又将多次行遍历和 kernel launch 合成一次。
64-thread CTA 包含两个独立行 warp，并用 launch bounds 将 kernel 控制在
64 registers/thread；其他数量的等值面保持线性 cell 映射。

## 3. 内存与同步

workspace 保存全部 iso 的 uint8 计数、uint32 scan 区和 CUB 临时区。2/4/8-way
路径在 scan 区同时保存每个 iso 的 row counts 与 row offsets；其他 workload
复用一组 cell offsets。各 iso 的 scan 和 total 发布在传入 stream 上顺序执行，
fused emit 读取 device total 做容量保护。所有 iso 完成后只进行一次 totals D2H
和 stream 同步。只读 marching-cubes 查表通过线程安全的一次性初始化装入
constant memory，后续 solve 不再重复传输 4.25 KiB 静态数据。

## 4. 正确性处理

尺寸与 workspace 算术均检查溢出。若任一 total 超过 host capacity，对应 emit
kernel 不写输出；最终同步后返回 `kInsufficientOutput`，不会越界。角点分类、
严格小于比较和插值均直接使用冻结的官方实现。

## 5. 本地结果

|测试点|是否正确|中位时间 / ms|相对 public baseline 加速|
|---|---:|---:|---:|
|P4|是|0.71858|1.989x|
|P5|是|1.37008|2.102x|
|P6|是|1.40801|3.904x|
|P7|是|5.72561|2.901x|

官方本地 scorer 为 70.0/70.0，其中 correctness 40.0/40.0，absolute
performance 30.0/30.0，加权 performance metric 为 2.735x。

P7 fused emit 的 NCU 结果为 64 registers/thread、0 local-memory spill，理论/实测
occupancy 为 66.67%/64.51%。相同 NCU section replay 下，64-register kernel 为
5.43 ms，未限制的 72-register 版本为 5.59 ms。

## 6. 复现命令

```bash
python3 runtest.py
```

其他编译参数或复现注意事项：

本地使用 CUDA 12.8、GCC/G++ 11.2 和 `sm_89`，无额外依赖。
