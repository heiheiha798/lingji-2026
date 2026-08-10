# Ice Carver 解题说明

> 本文件用于说明实现思路，正式计分代码仅为 `handout/src/solution.cu`。

## 1. 基本信息

- 队伍：
- 成员：
- 提交版本：
- 测试镜像/日期：

## 2. 算法与 CUDA 实现

分类 kernel 一次读取八个角点并处理全部等值面，生成 iso-major 的紧凑计数。
每个 iso 仍独立执行稳定的 exclusive scan，emit kernel 按 cell id 和冻结表顺序
写出三角形，因此输出顺序与参考实现一致。

## 3. 内存与同步

workspace 保存全部 iso 的 uint8 计数、一个 iso 的 uint32 offsets 和 CUB scan
临时区。各 iso 的 scan、total 发布和 emit 在传入 stream 上顺序执行；emit 直接
读取 device total 做容量保护。所有 iso 完成后只进行一次 totals D2H 和 stream
同步，取代每个 iso 一次同步。

## 4. 正确性处理

尺寸与 workspace 算术均检查溢出。若任一 total 超过 host capacity，对应 emit
kernel 不写输出；最终同步后返回 `kInsufficientOutput`，不会越界。角点分类、
严格小于比较和插值均直接使用冻结的官方实现。

## 5. 本地结果

|测试点|是否正确|中位时间 / ms|相对 public baseline 加速|
|---|---:|---:|---:|
|P4|是|1.10462|1.005x|
|P5|是|2.10775|1.059x|
|P6|是|2.79407|1.003x|
|P7|是|10.8842|0.997x|

## 6. 复现命令

```bash
python3 runtest.py
```

其他编译参数或复现注意事项：

本地使用 CUDA 12.8、GCC/G++ 11.2 和 `sm_89`，无额外依赖。
