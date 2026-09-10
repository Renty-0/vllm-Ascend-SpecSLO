# SpecRhythm 论文 workload 与验收记录

## 固定口径

SpecRhythm 的三类请求必须按 coding/chat/summarization = 6:2:2 混合：

| 类别 | 数据集 | TPOT 约束 |
| --- | --- | ---: |
| tight | HumanEval | <= 40 ms |
| normal | Stanford Alpaca | <= 50 ms |
| loose | CNN/DailyMail | <= 150 ms |

实验变量为 RPS 2、4，batch 8、16、32、64。到达时间写入 JSONL manifest，
生成命令示例：

```bash
PYTHONPATH=. python examples/prepare_specslo_workload.py \
  --humaneval /root/data/reference-repos/atc26v0/benchmark/data/HumanEval.jsonl \
  --alpaca /data/datasets/specslo/stanford_alpaca/alpaca_data.json \
  --cnndm /root/data/reference-repos/atc26v0/benchmark/data/CNNDM.jsonl \
  --rps 4 --num-requests 40 --max-tokens 32 \
  --out /data/specslo-workloads/rps4_mix6-2-2_p40_t32.jsonl
```

每个 manifest 都有同名 `.meta.json`，记录配额、实际比例、SLO、gamma、seed、
到达源和数据文件。没有论文生产 trace 时，`arrival_source=synthetic_poisson`
只表示固定 seed 的可复现实验输入。

## 当前短回归

Qwen3-0.6B draft TP1 + Qwen3-32B target TP3，ACLGraph/paged attention 开启，
40 请求、`max_tokens=32` 的 e2e 结果：

| 配置 | 吞吐 (tok/s) | 接受率 | SLO 达标率 | Goodput (tok/s) |
| --- | ---: | ---: | ---: | ---: |
| gamma=4，prefill chunk=8 | 94.12 | 77.17% | 47.5% | 44.71 |
| gamma=4，prefill chunk=40 | 94.08 | 77.38% | 60.0% | 56.45 |
| gamma=2，prefill chunk=8 | 93.53 | 83.73% | 27.5% | 25.72 |
| target-only TP4（离线提交） | 247.11 | - | 未计时 | - |

target-only 没有消费 arrival timestamp，因此只能做吞吐参考。当前尚未达到
`SLO attainment >= 80%` 和 `Goodput >= 1.3x`；瓶颈集中在 TP3 target 计算、
verdict/broadcast 和 draft-target 交换。短回归命令和原始 JSON 结果位于
`/data/specslo-workloads/`。

## 已验证优化

- prefill chunk 可大于 decode batch，连续队列一次 packed prefill，短回归达标率
  47.5% -> 60.0%。
- admission wait 与 decode TPOT 分离；wait 作为 urgency 的调度债务保留。
- `--spec-rhythm-auto-eager-tokens` 按每类 gamma 限制 eager 预算，并输出分组 SLO
  attainment/goodput。

## 验收要求

完成最终报告前，baseline 和 SpecRhythm 必须使用同一 manifest、同一物理卡拓扑、
同一 warmup/max token 和相同 e2e 计时，并对 RPS 2/4、batch 8/16/32/64 全部重复。
baseline 未接入在线 arrival/SLO 统计时，不得把离线吞吐比值写成 Goodput 加速比。
