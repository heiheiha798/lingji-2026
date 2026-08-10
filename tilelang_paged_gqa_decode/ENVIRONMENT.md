# 正式环境

比赛平台为每名选手提供单张 NVIDIA GeForce RTX 4090 24GB 算力。选手在个人账号环境中安装依赖，题目不要求配置主办方机器上的本地路径或环境变量。

| 项目 | 正式版本 |
| --- | --- |
| 操作系统 | Ubuntu 22.04 x86-64 |
| NVIDIA Driver | 570.153.02 |
| CUDA Toolkit | 12.8 |
| Python | 3.11 |
| PyTorch | 2.6.0 |
| TileLang | v0.1.12 |
| Triton | 本题不作为参赛接口单独使用 |
| flash-linear-attention | 本题不需要 |

安装：

```bash
python -m pip install -r requirements.txt
```

4090 真机验收时会按此版本表检查安装、正确性和性能命令；如上游依赖解析存在冲突，主办方将在正式发布前同步修订版本表。
