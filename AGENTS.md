使用 GPU 前请查看 GPU 是否空闲，只使用空卡，如果没有空卡，或者指定的卡被其他进程占用请暂停，或者直接停止向我报告。

优化问题优先寻找开源实现，我们的优化目标是至少要在题目给定的 case 下全面强于开源sota。对于 cuda kernel 优化问题一个简单的建议是结合 Nsight Compute、Nsight System，以及反编译拿到PTX/SASS来分析kernel bottleneck。如果开源实现没有特别适合的 baseline 那么先使用 triton 等强编译器的 dsl 来做一个强 baseline，继续使用 cuda 进行深入优化。

git clone 优先 ssh ，如果必要请你新开 tmux 挂~/mihomo的网络代理进行网络操作。