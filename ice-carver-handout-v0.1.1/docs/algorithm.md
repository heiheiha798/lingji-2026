# 算法与优化边界

## 基础正确方案

对每个 `(iso_id, cell_id)`：读取八角点，按 `value < isovalue` 生成 case，查 `kTriangleCount`。把计数写入紧凑数组，做 exclusive scan 得到该单元在对应 iso 独立数组中的偏移，再次遍历活动单元并按查表顺序插值写出三角形。

这一方案清晰且足以通过正确性测试。`uint8_t` 足以保存单单元三角形数；全局偏移和总数必须能表示最坏容量并检查溢出。

## Workspace 契约

不得在正式计时区间调用 `cudaMalloc`/`cudaFree`。评测器预分配 workspace；实现应检查 `workspace_bytes`，不足时返回 `kInsufficientWorkspace`，不得部分越界写入。成功启动后，workspace 起始处应写入 `WorkspaceDescriptor`，其中 magic、版本、所需字节数与各区域偏移可供评测器审计。偏移必须满足对齐并落在 workspace 内。

## 可优化方向

- 一次加载角点后处理多个阈值；
- 压缩计数或活动单元，按稀疏度选择路径；
- block scan、warp ballot 与分层前缀和；
- 共享内存 tile/只读缓存降低相邻单元重复读取；
- 减少完整 offsets 的存储；
- 在不改变确定顺序的前提下融合 kernel；
- 分 slab 处理以限制峰值 workspace。

优化不能改变独立 iso 数组、cell 顺序、查表顺序或浮点正确性。经典表的拓扑歧义不是本题优化目标；不得替换为不同三角剖分。

## 常见错误

- 把 `value <= isovalue` 当作 inside；
- cell 展平时误用 `nx`/`ny` 而非 `nx-1`/`ny-1`；
- 多 iso 共用一个全局输出数组；
- atomic append 导致输出顺序不确定；
- 只检查总容量，不检查每个 iso 的容量；
- 在非默认 stream 上启动后用错误 stream 记录 event；
- 用 32 位乘法计算大尺寸索引后才转换为 64 位；
- `icecarver_solve` 返回时仍有未纳入计时/验证的写操作。

## 性能分析建议

先用 Nsight Systems 确认 kernel/scan/同步边界，再用 Nsight Compute 检查内存吞吐、分支效率、occupancy 和寄存器压力。公开得分以评测器结果为准，分析工具不得出现在正式计时路径中。P0–P3 只计正确性；稳定性能区分主要来自 P4–P7 的多阈值与不同活动率场景。
