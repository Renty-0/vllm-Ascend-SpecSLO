# Llama 70B / 1B 的 b32、b16、b8 对照

原计划按 b32 → b16 → b8 的顺序测试：每档先将官方 vLLM Ascend v0.21.0 的 Llama 3.1 70B TP4、MC2 OFF、自回归 baseline 独立测三轮，Goodput 取三轮算术平均并冻结 1.5× 门槛，然后测试 Llama 3.2 1B TP1 draft + 70B TP3 target 的 SpecSLO。SpecSLO 保持双 batch、rolling eager。**2026-11-26 按用户要求停止了 b32 第二轮及后续排队测试；b32/b16/b8 均未冻结基线，也没有可比较的 SpecSLO 结果。**

三档都使用物理 NPU 1、2、3、4，同一份 463 请求清单、在线到达时刻、每请求 256 输出 token、40/50/150 ms 三类阈值，以及引擎内部 paper TPOT 计时。baseline 的 `max_num_seqs`、`--batch-sizes` 和 warmup 请求数设置为对应档位；`max_num_batched_tokens=640`、图捕获尺寸集合、`gpu_memory_utilization=0.8` 和模型/软件版本与已冻结的 b64 baseline 相同。每轮都须核对输入 token 的 SHA256 与 b64 baseline 一致。

复现入口：

```bash
cd /root/SpecSLO/specslo-linear-650-publish/research/specslo-linear-650/reproduction/llama70b
SPECSLO_VISIBLE_DEVICES=1,2,3,4 bash run_baseline_v021.sh <唯一标签> <32|16|8>
python freeze_baseline.py --batch <32|16|8> --tags <三轮标签>
```

冻结文件为 `baseline-v021-llama70b-b{batch}-three-run-mean.json`。正式比较使用该文件记录的均值，不按后续 SpecSLO 结果挑选 baseline 轮次。b64 冻结均值为 122.58 token/s，只适用于 b64。

本分支的 `gate.py` 目前只审计已冻结的 **b64**。本机实验目录另有未发布的跨 batch `gate_llama.py`；只有取得各档三轮 baseline 并冻结门槛后，才可把它用于该档 SpecSLO 结果。旧 b64 r21 结果已用该扩展门禁重审，仍为 `pass`。

三档 baseline 冻结后，才可设计并测试对应的 SpecSLO 参数。本机实验目录预备过双 batch 初始配置脚本 `run_spec_batch.sh`，但它没有被验证，本次发布分支不把它列为复现入口，也不预先宣称达到 1.5×。

三轮分别启动独立进程并使用独立编译缓存。b32 第 2 轮首次尝试 `...-b32-r2-20261126/` 复用第 1 轮缓存时，在正式测量前因 NPU 图加载设备不匹配失败，没有 Goodput；改用新的运行标签及独立缓存重测。计时只覆盖正式请求，编译和预热不计入 Goodput。运行目录中的 `launch-environment.json` 记录实际缓存目录。

| batch 上限 | baseline 三轮 Goodput | baseline 均值 | SpecSLO 1.5× 门槛 | SpecSLO 最好有效结果 | 结论 |
| ---: | --- | ---: | ---: | ---: | --- |
| 32 | r1: 83.62；r2 主动中止；r3 未启动 | 未冻结 | 未设定 | 未测 | 暂停 |
| 16 | 未测 | 未冻结 | 未设定 | 未测 | 暂停 |
| 8 | 未测 | 未冻结 | 未设定 | 未测 | 暂停 |

Llama b64 的完整方法和已验证结果见 [REPRODUCE_zh.md](REPRODUCE_zh.md)。

b32 首轮 `baseline-results/runs/official-v021-llama70b-b32-b32-r1-20261126/result.json` 的 Raw 为 403.27、Goodput 为 83.62 token/s，达标 96/463（tight 0、normal 0、loose 96）；输入 token SHA256 为 `2327fc4be1069c109fda2450767b5e0e5230a64ae7c6ba74fe38ea22cc754631`，与 b64 冻结 baseline 相同。该值不是固定门槛。

b32 第二轮首次尝试 `...-b32-r2-20261126/` 因共享编译缓存导致图加载设备不匹配，正式测量前失败。使用独立缓存重启的 `...-b32-r2-fresh-20261126/` 随后按用户要求被主动中止，没有有效 `result.json`；排队的 r3、b16、b8 未启动。这些运行均不计入平均值，也不能据此判断 b32 的 1.5× 是否达标。
