# SpecSLO 线性 b64：动态 γ=0/4 的 Goodput 700 候选

当前分支的逐步复现说明见 [REPRODUCE_zh.md](REPRODUCE_zh.md)。该文档包含非 Git 资产的 SHA、设备映射、运行命令及结果审计。

2026-11-20 在物理 NPU `2,3,5,7` 上，保持全局活跃上限 64、单次 target 上限 32，给 normal 活跃请求设上限 8，并以类别偏置 0.1 调整 rolling-eager 门槛。每轮候选预算 `B=128` 时，额外草稿按紧迫度动态选择 γ=0 或 4。三轮引擎内部 paper TPOT Goodput 为 **700.73 / 709.11 / 701.79 token/s**，中位数 **701.79**；实际活跃峰值分别为 **64 / 63 / 64**。完整合同、逐类达标数、机制计数和结果 SHA 见复现文档。

本分支以 `SpecSLO-Chain` 的 `b8f3ad9e326d2407902cae6f1fd918e62f43c6ca` 为基底，**直接包含** 2026-11-13 实测候选的源码增量；不需要再应用 `source.patch`。原候选使用 Qwen3-0.6B TP1 draft、Qwen3-32B TP3 target、MC2 ON、固定线性 γ=4、全局活跃上限 64、每个 target home 上限 32；当前 B=128 候选保留这些设备和模型配置，在额外草稿上动态取 γ=0/4。

关键调度改动在 `vllm_ascend/spec_decode/pearl/native_engine.py`：有已到达的 tight 请求等待，或活跃 tight 数超过门槛时，暂缓新 loose 请求首次入场；已入场的 loose 不会被暂停。另按提示长度、输出上限和 γ 估算 KV 页预留，超出阈值时暂缓新请求，以防预热和正式运行耗尽 KV。最优实测门槛为 6。

运行参数见 `run.sh`：KV cache 640 页、接纳保护阈值 624 页，mixed target prefill 图桶为 `128,256,384,512`，获准请求使用完整 γ=4 图，target SLO-first 排序关闭。图桶从原来的八档减为四档，释放显存容纳更多 KV。`run.sh` 和 `body_candidate.sh` 固定运行合同、透传本次的类别偏置与 B，并将结果写到本分支工作树；本地模型、扩展、MC2 profile、Python 环境和 Ascend 环境脚本仍须按机器实际路径配置。

在原实验机器上，同一 463 请求清单、40/50/150 ms 引擎内部 paper TPOT、完整 E2E 测量时长作为 Goodput 分母，门槛 6 的三轮结果为：

| 轮次 | Raw token/s | Goodput token/s | tight | normal | loose | 实际活跃峰值 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| r1 | 736.10 | 664.56 | 226/271 | 96/96 | 96/96 | 64 |
| r2 | 735.00 | 657.21 | 222/271 | 96/96 | 96/96 | 64 |
| r3 | 742.99 | 662.75 | 221/271 | 96/96 | 96/96 | 64 |

三轮 Goodput 中位数 662.75 token/s，均超过 650。三轮图资格固定点、正式阶段零图容量或形状回退、机制门禁通过。原始结果、输入哈希、运行契约和并发审计保存在 [`SpecSLO-Tree` 分支的实验目录](https://github.com/Renty-0/vllm-Ascend-SpecSLO/tree/SpecSLO-Tree/research/specslo-linear/kv640-coarse4-b64/evidence)。这些数据只支持该机器与清单下的结论。此前固定 γ=4 b64 对照单轮 Goodput 为 514.96 token/s，但对照与本候选还存在设备编号及其他配置差异，不能把差值归因于单个开关。

2026-11-19 当前容器物理卡 `2,3,5,7` 的五次固定 γ=4 测量得到 Goodput **681.53、675.72、678.68、690.35、694.34 token/s**，Raw **740.72、741.37、739.36、741.61、742.45 token/s**，实际活跃峰值均为 **64**。后四次从启动到门禁完整退出，日志保留；第一次的脚本在写完结果后异常退出，日志不完整。第三次运行标签虽含 `dynamic`，但开关被脚本覆盖；第四次虽然标签含 `40-24`，实际未启用 SLO home 分区；第五次虽然标签含 `workers4`，实际仍为 2 线程。这三次均只能当作默认配置的重复测量。配置、原始结果 SHA 和审计方式见复现文档。

在原实验环境下执行门槛 6 的命令：

```bash
SPECSLO_VISIBLE_DEVICES=2,3,5,7 SPECSLO_LOOSE_ADMISSION_TIGHT_GATE=6 \
  bash research/specslo-linear-650/run.sh gate6-r1 0.4
```

示例卡号是当前容器可见的物理卡；实际运行应先按 `npu-smi info` 修改。`SPECSLO_ENVIRONMENT_SCRIPT`、`SPECSLO_DEVICE_MAP_SCRIPT` 和 `SPECSLO_PRODUCTION_GATE_SCRIPT` 可覆盖随分支保存的辅助脚本。运行前仍需要本地模型权重、已编译扩展、MC2 profile、正确的设备编号和 Python/CANN 环境；默认资产路径保留原实验机配置。源码通过 `compileall`，相关 `test_mixed_width_layout.py` 单测 4/4 通过；单测在当前容器中需设置 `TORCH_DEVICE_BACKEND_AUTOLOAD=0`。
