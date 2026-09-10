# SpecSLO/SpecRhythm 图片目标回归报告

日期：2026-09-09  
平台：Ascend 910B2，CANN 26.0.rc1 容器环境  
模型：Qwen3-0.6B draft（TP1）+ Qwen3-32B target（TP3）  
baseline：原始 vLLM-Ascend Qwen3-32B target-only（TP4）

## 验收口径

三类请求来自论文 SpecRhythm workload：HumanEval（tight，TPOT <=40 ms）、
Stanford Alpaca（normal，TPOT <=50 ms）、CNN/DailyMail（loose，TPOT <=150 ms），
请求比例为 6:2:2。当前回归使用 RPS4、B64、64 请求、`max_tokens=256` 的固定
manifest；由于 64 不是 10 的倍数，该文件实际为 coding/chat/summarization=38/13/13。
严格比例的 60 请求 manifest 已生成于
`/root/data/specslo-workloads/rps4_mix6-2-2_p60_t256_exact.jsonl`。e2e wall time
包含 prefill 和 generation，warmup 不计入 e2e。

图片目标是 attainment >=80%、Goodput >=1.3x 原始 vLLM-Ascend。Goodput 定义为
满足该请求 TPOT 约束的完成 token 数除以 e2e wall time。

## 结果

| 配置 | e2e 吞吐 (tok/s) | 接受率 | attainment | Goodput (tok/s) | Goodput / baseline |
| --- | ---: | ---: | ---: | ---: | ---: |
| vLLM-Ascend target-only TP4 | 629.53 | - | 82.81% | 521.30 | 1.000x |
| SpecSLO/SpecRhythm gamma=4 TP1+TP3（批量 CPU materialize） | 416.375 | 54.22% | 20.31% | 84.576 | 0.162x |
| SpecSLO/SpecRhythm gamma=2 TP1+TP3 | 414.274 | 66.83% | 20.31% | 84.149 | 0.161x |
| SpecSLO/SpecRhythm gamma=4 + priority burst=2 | 414.563 | 54.61% | 20.31% | 84.208 | 0.162x |
| SpecSLO/SpecRhythm gamma=8 TP1+TP3 | 302.115 | 38.45% | 20.31% | 61.367 | 0.118x |

原始结果文件：

- baseline：`/root/data/specslo-workloads/smoke-baseline-rps4-b64-t256.json`
- gamma=4：`/root/data/specslo-workloads/smoke-pearl-rps4-b64-t256-bulkcpu.json`
- gamma=2：`/root/data/specslo-workloads/smoke-pearl-rps4-b64-t256-gamma2.json`
- priority：`/root/data/specslo-workloads/smoke-pearl-rps4-b64-t256-priority.json`
- gamma=8：`/root/data/specslo-workloads/smoke-pearl-rps4-b64-t256-gamma8.json`

## 阶段耗时

gamma=4 最新长回归的 target leader 阶段累计耗时为：

| 阶段 | 时间 (s) | 说明 |
| --- | ---: | --- |
| Draft compute | 由 draft worker 累计 28.259 | 已使用 paged attention 和固定 home-batch ACLGraph |
| Draft -> Target communication | 0.573 | HCCL proposal exchange |
| Target verify | 10.151 | verdict 构造和 target verification |
| Target compute | 21.327 | target TP3 packed forward |
| Target -> Draft communication | 2.957 | verdict broadcast；含 collective wait |
| wait/sync/state update/refill | 0.126 / 0.304 + state update | Python 状态提交、回收和 admission |

阶段时间不能简单相加为 e2e，因为 worker 统计存在重复观测和 collective 等待；
它们用于定位瓶颈，不替代 e2e wall time。

为获得图片要求的“几个 decode step 拆分”数据，额外运行了
`profile-only --profile-decode-steps=3`。文件为
`/root/data/specslo-workloads/smoke-pearl-rps4-b64-t256-bulkcpu-profile3.json`，
前三步平均值如下：

| 阶段 | 每 step 平均 |
| --- | ---: |
| Draft compute | 39.163 ms |
| Draft -> Target communication | 0.846 ms |
| Target verify | 34.277 ms |
| Target -> Draft communication | 0.266 ms |
| wait / sync / state update | 13.932 ms |

Target verify 中 target compute 为 33.847 ms、verdict 为 0.430 ms；最后一项中
wait/sync 为 13.875 ms、state update 为 0.057 ms。profile-only 只执行三步，
其输出 token 数和吞吐不用于完整 workload 的排名。

## 已验证的改动

1. draft rank 保留本地 continuation device tensor，只同步小的 verdict，避免每轮
   将整块 proposal 拷回 CPU；输出哈希与旧路径一致。
2. draft proposal 矩阵每轮只做一次 CPU materialize，再在 host 侧拆分 request，避免
   按 request 的重复 stream synchronize；最新长回归为 416.375 tok/s。
3. paged-attention 固定 B32 home batch 的 draft ACLGraph capture/replay；短回归
   replay 正常且无 shape fallback，但长回归未出现可重复的数量级收益。
4. 有界 SLO priority burst 已实机验证；B64 长回归没有改善 attainment，因此默认关闭。
5. TP3 fused `npu_mm_all_reduce_base` 增加 capability probe 和自动 fallback，保证
   当前 CANN 不支持 rank 3 时仍能正确运行。

## 结论与阻塞项

当前图片目标没有达到：best SpecSLO/SpecRhythm 控制面的吞吐是 baseline 的 0.657x，Goodput 是 0.162x，
attainment 是 20.31%。主要原因不是 graph capture 失败，而是 target compute、
target verify、HCCL broadcast 和 draft 计算在当前控制流中按轮次串行；论文需要的
两个 home batch 真正重叠执行尚未落地。另一个硬阻塞是目标 CANN 对
`npu_mm_all_reduce_base` 报告 `Rank size 3 is not supported by socversion id 2201`，
因此不能使用 rank-3 fused MC2。

下一阶段应实现 HCCL async mailbox + 独立 stream 的双 home-batch overlap，并获得
Ascend 910B2 可用的 TP3 MC2/等价 rank-3 all-reduce kernel；之后再按 RPS2/RPS4、
B8/B64 做完整验收矩阵。

详细工作记录见 [`specslo_work_record_zh.md`](specslo_work_record_zh.md)。

## PEARL 与 SpecSLO 的口径

代码包名仍为 `vllm_ascend.spec_decode.pearl`，这是为了兼容此前 nano-PEARL 的
native engine；但两种运行模式必须分开统计：没有 SLO/arrival 约束时走的是 PEARL
packed fast path，适合测普通吞吐；带有 TPOT、类别或到达时间约束并启用
`--enable-spec-rhythm` 时，才进入 SpecSLO/SpecRhythm control-plane，才产生
attainment 和 Goodput。上表四行均来自后者，不能称为“PEARL 普通吞吐”。

PEARL 普通吞吐可以看起来更高，是因为它不等待在线到达、不按 TPOT 淘汰请求，
几乎所有完成 token 都计入吞吐；SpecSLO Goodput 只计入满足对应 TPOT 上限的请求。
当前长回归中只有 13/64 请求达标，且 TP1+TP3 的 target verify、HCCL broadcast、
draft compute 仍按轮次串行，所以 Goodput 低于 baseline 并不是计算公式错误，而是
实际 SLO 未满足和服务路径开销共同造成的结果。

## 2026-09-09 实机补充

上面的历史表格使用的是早期 64 请求、`max_tokens=256` 记录，不能代表当前代码。
在同一 arrival trace 的 128 请求短输出复测中，最新 arrival-aware packed admission
的 SpecSLO 为 101.20 tok/s；对应 vLLM-Ascend TP4 graph baseline 为 108.08 tok/s，
即 0.936x。TP3 fused MM/all-reduce 在 A2 CANN 上被硬件拒绝，target cap=32 也经实测
降速，详细数据和复现命令见 [`specslo_regression_20260909_zh.md`](specslo_regression_20260909_zh.md)。

因此图片中的 1.3x 仍未达成，且当前 arrival-inclusive 负载本身的理论上界低于
1.3 倍 graph baseline；后续验收需要延长稳态 workload，并完成异步双 home overlap。
