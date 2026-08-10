# 选手提交说明

你只允许修改 `handout/src/solution.cu` 中的 `icecarver_solve` 实现。请勿修改头文件、评测器、生成器、配置、查表、CMake 或脚本；正式评测会把你的 `solution.cu` 放入干净的官方仓库重新构建。

环境固定为 Ubuntu 22.04、RTX 4090、CUDA Toolkit 12.8、`sm_89`。构建与公开测试：

```bash
bash scripts/build.sh
python3 runtest.py
```

每个 case 有两次预热和五个不同 seed 的正式变体。你必须为每个 iso 写入独立三角形数组，使用 `value < isovalue`，并严格保持 cell 与经典查表顺序。不要使用 atomic append 产生不确定输出。

返回错误码或容量不足时不得越界写入。官方会检查 guard、NaN/Inf、CUDA 错误和浮点容差。公开结果仅用于反馈；最终成绩以隐藏变体和统一镜像复跑为准。

学生包及正式学生镜像中不应出现 `reference/cuda_target.cu`、`config/private/` 或隐藏 seed。若发现此类文件，请停止使用并通知组委会。
