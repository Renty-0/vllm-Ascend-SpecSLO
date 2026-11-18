# SpecSLO 线性 γ=4、b64 的 Goodput 650 候选

本分支以 `SpecSLO-Chain` 的 `b8f3ad9e326d2407902cae6f1fd918e62f43c6ca` 为基底，**直接包含** 2026-11-13 实测候选的源码增量；不需要再应用 `source.patch`。实验使用 Qwen3-0.6B TP1 draft、Qwen3-32B TP3 target、MC2 ON、固定线性 γ=4、全局活跃上限 64、每个 target home 上限 32。

关键调度改动在 `vllm_ascend/spec_decode/pearl/native_engine.py`：有已到达的 tight 请求等待，或活跃 tight 数超过门槛时，暂缓新 loose 请求首次入场；已入场的 loose 不会被暂停。另按提示长度、输出上限和 γ 估算 KV 页预留，超出阈值时暂缓新请求，以防预热和正式运行耗尽 KV。最优实测门槛为 6。

运行参数见 `run.sh`：KV cache 640 页、接纳保护阈值 624 页，mixed target prefill 图桶为 `128,256,384,512`，固定 γ=4、target SLO-first 排序关闭。图桶从原来的八档减为四档，释放显存容纳更多 KV。`run.sh` 和 `body_candidate.sh` 从原实验脚本复制，仅将仓库路径和结果路径改为相对于本分支工作树的路径；本地模型、扩展、MC2 profile、Python 环境和 Ascend 环境脚本仍须按机器实际路径配置。

在原实验机器上，同一 463 请求清单、40/50/150 ms 引擎内部 paper TPOT、完整 E2E 测量时长作为 Goodput 分母，门槛 6 的三轮结果为：

| 轮次 | Raw token/s | Goodput token/s | tight | normal | loose | 实际活跃峰值 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| r1 | 736.10 | 664.56 | 226/271 | 96/96 | 96/96 | 64 |
| r2 | 735.00 | 657.21 | 222/271 | 96/96 | 96/96 | 64 |
| r3 | 742.99 | 662.75 | 221/271 | 96/96 | 96/96 | 64 |

三轮 Goodput 中位数 662.75 token/s，均超过 650。三轮图资格固定点、正式阶段零图容量或形状回退、机制门禁通过。原始结果、输入哈希、运行契约和并发审计保存在 [`SpecSLO-Tree` 分支的实验目录](https://github.com/Renty-0/vllm-Ascend-SpecSLO/tree/SpecSLO-Tree/research/specslo-linear/kv640-coarse4-b64/evidence)。这些数据只支持该机器与清单下的结论。此前固定 γ=4 b64 对照单轮 Goodput 为 514.96 token/s，但对照与本候选还存在设备编号及其他配置差异，不能把差值归因于单个开关。

在原实验环境下执行门槛 6 的命令：

```bash
SPECSLO_LOOSE_ADMISSION_TIGHT_GATE=6 bash research/specslo-linear-650/run.sh gate6-r1 0.4
```

`SPECSLO_ENVIRONMENT_SCRIPT`、`SPECSLO_DEVICE_MAP_SCRIPT` 和 `SPECSLO_PRODUCTION_GATE_SCRIPT` 可覆盖原机器的辅助脚本路径。该脚本运行前还需要本地模型权重、已编译扩展、MC2 profile、正确的设备编号和 Python 环境；脚本中的这些路径默认保留原实验机配置。源码通过 `compileall`，相关 `test_mixed_width_layout.py` 单测 4/4 通过；单测在当前容器中需设置 `TORCH_DEVICE_BACKEND_AUTOLOAD=0`。
