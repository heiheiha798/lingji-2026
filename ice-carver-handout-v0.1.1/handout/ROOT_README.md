# 冰川切片：学生包

这是 Ice Carver v0.1.1 的完整学生环境。你只需要修改 `handout/src/solution.cu`，正式评测会把该文件放回干净仓库，以 `TARGET=OFF` 重新构建并在隐藏 seed 上复跑。

环境固定为 Ubuntu 22.04、单张 RTX 4090、CUDA Toolkit 12.8、CMake 3.22+、Ninja、C++17/CUDA C++17 和 `sm_89`。

```bash
bash scripts/smoke_test.sh
bash scripts/run_public.sh
```

- [正式题面](docs/problem.md)
- [算法与优化边界](docs/algorithm.md)
- [评分规则](docs/scoring.md)
- [测试与测量协议](docs/testcases.md)
- [提交说明](handout/README.md)

公开 JSON 只用于本地反馈，不能作为成绩凭证。最终成绩以组委会在统一镜像和隐藏变体上的复跑为准。学生包中不应出现 `config/private/`、`reference/cuda_target.cu`、隐藏 seed 或命题人的原始校准结果；若发现，请停止使用并通知组委会。
