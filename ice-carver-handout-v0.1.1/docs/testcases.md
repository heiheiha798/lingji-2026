# 测试与测量协议

## 测试矩阵

|ID|典型尺寸|阈值数|公开场类型|每 iso 容量|正确性权重|性能权重|
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

YAML 中的实际尺寸、阈值、容量和权重是机器可读权威值。命题组可在赛后更换 seed、尺寸和场参数，但不得改变同一版本已经冻结的算法语义、权重或计时边界。

## 变体与 seed

每个 case 固定 `warmup_runs: 2` 和 `measure_runs: 5`。生成器通过冻结的 `DeriveVariantSeed(case_seed, variant_index)` 派生互相独立的变体；5 次正式测量必须使用 5 个不同 seed，而不是对同一输入重复计时。公开配置公开基础 seed；私有配置、隐藏基础 seed 和参数只存在于命题人镜像。

生成器直接在 GPU 上产生体数据，三个坐标轴分别归一化到 `[-1,1]` 计算场值。生成阶段、H2D/D2H、分配、验证和 JSON 写入不计时。输入、输出和 workspace 在计时前分配。

## 单次测量

1. 根据本次变体 seed 生成输入并同步；
2. 重置输出、计数、guard 和 workspace；
3. 在传入 stream 记录 start event 和宿主起点；
4. 调用 solver，记录 stop event；
5. 等待 stop event/device 工作完成，记录宿主终点；
6. 使用 `max(event_ms, wall_ms)`；
7. 在计时外验证结果。

case 时间为 5 个正式变体的中位数。任何 CUDA 错误、非零状态、错误输出或宿主硬超时都会使该 case 无效。每个 evaluator 进程外层必须使用 `timeout --kill-after`；不能只依赖 CUDA Event 或进程内超时。

评分脚本必须通过 `--manifest evaluator/official_manifest.json` 核对完整 case 集、权重、运行证据和校准版本。C++ 评测器只读取由组委会控制的配置文件；选手本地公开配置与 JSON 不构成正式成绩凭证。

## 配置格式

配置只支持扁平 `key: value` YAML，核心字段包括：`id`、`field`、`shape`、`seed`、`isovalues`、`per_iso_capacity`、`warmup_runs`、`measure_runs`、容差、超时和两类权重。每个 iso 都按 `per_iso_capacity` 获得独立数组。

性能校准必须 fresh configure/build 并显式传 `-DCMAKE_CUDA_ARCHITECTURES=89`，不能复用曾以旧默认架构生成的 build 目录或结果。发布前建议在两个全新同规格 RTX 4090 虚机上复测；最低验收要求是在命题机完成三轮私有配置校准，并在独立学生镜像机完成 clean build 与公开 smoke。两类验收都应记录 GPU/驱动/CUDA、镜像版本、源码版本、最小/中位/最大时间和峰值显存。
