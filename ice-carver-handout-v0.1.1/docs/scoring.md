# 评分规则

总分 100 分，由正确性 40 分、Baseline 相对性能 30 分和最终排名 30 分组成。

## 正确性：40 分

每个配置有独立正确性权重。该 case 的 5 个正式 seed 变体必须全部在硬超时前返回，并通过：每 iso 计数、独立数组容量、确定顺序、全量浮点比较、NaN/Inf、guard 与 CUDA 错误检查。任一正式变体失败，该 case 正确性为 0；否则取得全部权重。

## 相对性能：30 分

P0–P3 不计性能；冻结后的权重为：P4=5、P5=7、P6=8、P7=10，总和 30。实测中 P2/P3 小于约 0.2 ms 且 public/target 差距过小，不适合稳定计分；P4–P7 的 target 加速具有稳定区分度。

每个正式变体的时间定义为：

$$
t=\max(t_{event},t_{wall}),
$$

其中 `t_event` 是同一 CUDA stream 上覆盖完整求解的 CUDA Event 时间，`t_wall` 是从记录开始到 stop event 完成 device synchronization 的单调宿主墙钟时间。case 时间 `T_i` 为 5 个不同 seed 正式变体时间的中位数；两次预热不计分。

设同一环境中的 public baseline、private target 和选手时间分别为 `T_base,i`、`T_target,i`、`T_i`：

$$
A_i=p_i\,\operatorname{clamp}\left(
\frac{\log T_{base,i}-\log T_i}
{\log T_{base,i}-\log T_{target,i}},0,1\right).
$$

只有 case 正确时才取得该点性能分。若 `T_target,i >= T_base,i`、分母过小或校准无效，组委会必须先重新校准，不能用异常基线评分。

## 最终排名：30 分

只有 P4–P7 全部正确的提交参与排名。综合指标使用相对 baseline 加速比的性能权重加权几何平均：

$$
G=\exp\left(\frac{\sum_i p_i\log(T_{base,i}/T_i)}{\sum_i p_i}\right).
$$

按 `G` 降序排名。有效提交数为 `N`、名次为 `r`（第一名为 1）时：`N=1` 得 30 分；`N>1` 时 `R=30(N-r)/(N-1)`。并列按组委会冻结的高精度 `G` 和复跑记录处理。

最终分为 `S=C+A+R`。选手本地 `result.json` 不能作为最终凭证；组委会在统一镜像、隐藏 seed 和相同校准版本上复跑后计算排名。
