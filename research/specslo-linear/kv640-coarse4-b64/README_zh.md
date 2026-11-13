# SpecSLO 串行 γ=4：真实 b64 的 650 Goodput 候选

这份目录保存 2026-11-13 实测的完整代码增量、运行脚本和逐轮原始结果。代码基底是本仓库 `SpecSLO-Chain` 分支的 `b8f3ad9e326d2407902cae6f1fd918e62f43c6ca`；`source.patch` 是对该提交的完整工作树增量，包含当时未提交的新文件。它已在该提交的干净工作树上通过 `git apply --check` 和 Python `compileall`。请勿把该补丁直接应用于 `SpecSLO-Tree` 当前源码，两条分支的实现已经分叉。

运行配置见 [run.sh](run.sh) 与 [body_candidate.sh](body_candidate.sh)：Qwen3-0.6B TP1 draft、Qwen3-32B TP3 target、固定线性 γ=4、MC2 ON、全局活跃上限 64、每个 home 的 target 上限 32、KV cache 640 页、四档 mixed prefill 图、loose 入场 tight 门槛默认 8、KV 接纳预留阈值 624 页。门槛 10 的实验在调用脚本时设置 `SPECSLO_LOOSE_ADMISSION_TIGHT_GATE=10`，其余配置相同。脚本保留实验机器的绝对路径；在其他机器执行前需按实际模型、扩展和环境路径调整。

同一 463 请求清单、40/50/150 ms 的引擎内部 paper TPOT、完整端到端测量时长作为 Goodput 分母：

| 轮次 | Raw token/s | Goodput token/s | tight 达标 | normal | loose | 实际活跃峰值 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| [r1](evidence/r1/result.json) | 750.34 | 649.86 | 209/271 | 96/96 | 96/96 | 64 |
| [r2](evidence/r2/result.json) | 743.45 | 671.19 | 226/271 | 96/96 | 96/96 | 64 |
| [r3](evidence/r3/result.json) | 749.01 | 650.33 | 210/271 | 96/96 | 96/96 | 64 |
| [门槛 10，r1](evidence/gate10-r1/result.json) | 755.66 | 659.37 | 213/271 | 95/96 | 96/96 | 64 |
| [门槛 10，r2](evidence/gate10-r2/result.json) | 751.89 | 641.46 | 203/271 | 96/96 | 96/96 | 64 |

门槛 8 的 Goodput 三轮中位数 **650.33 token/s**。每轮的 [机制门禁](evidence/r3/mechanism-gate.json)、实际活跃峰值审计以及原始结果均在 `evidence/r1` 至 `r3`；三轮图资格固定点成立，正式阶段无图容量或形状回退。首轮低于 650，因此该配置虽达到中位数目标，余量仍小，不能称为每轮稳定超过 650。门槛 10 的两轮为 **659.37、641.46** token/s，同样跨过阈值，不能称为稳定达标。两轮 tight 的 TPOT 中位数分别约 36.07、36.77 ms，40 ms 附近的请求使达标数从 213 变为 203。与固定 γ=4 b64 对照的 Goodput 514.96 token/s 相比，门槛 8 的中位数提高约 26.3%；两者用相同 463 请求清单、SLO 阈值和计时定义。

还原代码时，从 `SpecSLO-Chain` 的上述提交创建工作树，在工作树根目录执行 `git apply --check /path/to/source.patch`，随后执行 `git apply /path/to/source.patch`。本目录保存的是完整增量与数据；实验运行还依赖本地模型权重、MC2 profile、已编译扩展及 Ascend 环境。
