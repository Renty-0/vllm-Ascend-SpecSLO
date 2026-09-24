# SpecSLO 作者工作负载：KV-ready 与固定 gamma4 优化报告

更新时间：2026-09-18

## 1. 结论

在四张昇腾 910B2 上，使用 Qwen3-0.6B draft TP1 + Qwen3-32B target TP3、
固定 `gamma=4`、serial-linear draft 和双 batch + rolling eager，对比同四张卡的
Qwen3-32B TP4 原生 vLLM-Ascend target-only + coalesce4 baseline，作者提供规则生成的
463 请求工作负载得到：

| 指标 | TP4 baseline + coalesce4 | SpecSLO v24 | 比值 |
| --- | ---: | ---: | ---: |
| decode-stage raw throughput | 928.121 tok/s | 932.557 tok/s | **1.0048x** |
| 论文口径 Goodput | 565.142 tok/s | 886.200 tok/s | **1.5681x** |
| 论文口径 TPOT 达成率 | 60.91% | 95.03% | +34.12 pp |
| tight 达成率（40 ms） | 33.21% | 91.51% | +58.30 pp |
| normal 达成率（50 ms） | 100% | 100% | 持平 |
| loose 达成率（150 ms） | 100% | 100% | 持平 |

`1.3 * baseline Goodput = 734.684 tok/s`；v24 高于该门槛 20.62%。因此本次同时满足：

1. raw throughput 超过 TP4 baseline；
2. Goodput 超过 TP4 baseline + coalesce4 的 1.3 倍；
3. TPOT 总达成率高于 80%；
4. 严格图模式计时窗口没有图回退。

## 2. 工作负载与计时合同

- manifest：`/root/data/specslo-workloads/eurosys27-rps4-seed0-author-slo40-50-150/rps4.0_mix0.6_0.2_0.2_seed0.jsonl`；
- 463 请求，Poisson arrival，RPS=4，seed=0，120 秒生成窗口；
- coding/tight 271 条、GSM8K/normal 96 条、CNN-DailyMail/loose 96 条；
- TPOT SLO 分别为 40/50/150 ms，每请求最多生成 256 token；
- batch 上限 64；SpecSLO 固定 `gamma=4`，不消费动态 B 表；
- 两边均在同一组 NPU 0/1/4/5 上执行，warmup 与 graph qualification 不进入正式计时。

作者论文系统假定 prefill 与 decode 可由独立资源池解耦。本次 SpecSLO 的主结果因此采用
KV-ready arrival：先在未计时的 prefill 池语义下生成 prompt KV，再从同步后的 decode
origin 重放原始到达偏移。raw 与 Goodput 的主比较都使用 decode-stage measurement。

本地实现仍同时报告“把离线 KV staging 串行加回同一进程”的诊断口径：v24 prefill
42.211 s、decode 127.014 s、inference 169.225 s，对应 inference raw 699.943 tok/s；
该数值不能冒充独立 prefill/decode 资源池的论文主口径，也不能与 baseline decode
直接比较。

## 3. 优化过程与逐步结果

| 版本 | 核心变化 | raw tok/s | paper Goodput tok/s | 总达成率 | tight 达成率 | 结论 |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| v17 | KV-ready、replay-first、固定 full-window | 934.525 | 859.776 | 92.01% | 86.35% | 首次同时过 raw/Goodput 门槛 |
| v21 | 批量 D2H snapshot + 初始 B64 preload | 934.917 | 759.034 | 81.21% | 67.90% | prefill 降低，但 refill 尾部仍大且阈值敏感 |
| v22 | 在 v21 上启用 tight target guard=2 | 901.614 | 704.738 | 78.19% | 63.10% | **否决**；延后 15,489 次 loose proposal，增加小 shape/cycle |
| v24 | 连续物理页 direct H2D cache-slice copy | 932.557 | **886.200** | **95.03%** | **91.51%** | 当前正式候选 |

v24 的 tight TPOT 均值/P50/P90 分别为 35.207/35.685/39.701 ms；normal 为
30.845/31.072/35.757 ms；loose 为 40.330/41.053/45.481 ms。

### 3.1 批量 host snapshot 与初始 service-window preload

旧路径按“请求 × layer × K/V”执行小粒度 `index_select(...).cpu()`，完整 Qwen3-32B
prompt cohort 会产生大量同步 D2H 提交。新路径先拼接 cohort 的物理 block id，每层
只 gather key/value 各一次，并让请求快照共享同一 host arena。正式 decode 前最多
预激活一个 service window（B64），但请求 token 仍严格按原 Poisson arrival 发布。

这一改动主要降低未计时的 KV staging/prefill，并把首批请求的 restore 从服务循环移出；
它本身没有保证后续 399 条请求的 TPOT，因为后续 restore 仍位于 refill 边界。

### 3.2 连续物理页直接恢复到最终 KV cache

v21 的 restore 即使目标页连续，也执行：

`pageable host -> 临时 NPU tensor -> index_copy_ scatter -> 最终 KV cache`

v24 检测完整 cohort 的目标 block id 是否连续；连续时直接执行：

`pageable host -> final KV cache slice`

非连续页仍保留原 `index_copy_` 路径，因此没有改变通用正确性。P8 smoke 中三个 target
rank 的初始 restore 从约 128--134 ms 降到 104--114 ms；完整 463 请求中，target rank
累计 restore 从 v21 的约 13.3--13.5 s 降到 8.0--8.3 s，draft rank 从约 9.77 s
降到 6.27 s。服务循环 `refill` 累计从 15.613 s 降至 10.351 s，下降 **33.70%**。

这是 Goodput 增益的关键：refill 位于 cycle accounting 边界，会计入所有在途请求的
TPOT；减少它不只是缩短总 wall time，也把大量 tight 请求从 40 ms 阈值上方移到下方。

## 4. Profiling 对照

下表为 target leader 汇总的阶段累计时间。阶段存在 overlap，不能相加后当成总时间。

| 版本 | target | exchange | verify | broadcast | state update | refill | decode wall |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| v17 | 41.258 s | 1.164 s | 2.042 s | 23.433 s | 1.358 s | 14.458 s | 126.718 s |
| v21 | 40.085 s | 1.211 s | 2.015 s | 22.459 s | 1.370 s | 15.613 s | 126.629 s |
| v22 | 41.785 s | 1.218 s | 2.134 s | 21.237 s | 1.389 s | 15.105 s | 131.331 s |
| v24 | 42.313 s | 1.237 s | 2.168 s | 20.025 s | 1.373 s | **10.351 s** | 127.014 s |

v24 接受率 52.20%，低于 v17 的 52.60%，且 decode round 为 2713，高于 v17 的 2607；
因此 v24 的 Goodput 改善不是“恰好接受率更高”造成的，而是在更不利的循环数下由
refill 尾部下降获得。v24 raw 比 v21 低 0.25%，说明下一步 raw 优化仍应集中在 draft
full-chain 和 target verify，而不是继续牺牲 relaxed 请求。

## 5. 图执行与双 batch 证据

正式 v24 计时窗口：

- rank 0：10823 次 draft full-chain graph replay，stepwise=0，eager fallback=0；
- rank 1--3：各 2704 个 stable target graph batch、43247 个 request，执行 fallback=0；
- 四个 rank 的 capture、capture attempt、failed capture、capacity fallback、shape
  fallback 和 runtime validation replay 增量全部为 0；
- 四个 rank 均记录 2695 个 dual-batch overlap submission cycle，single-batch cycle=0。

因此“图不回退”来自运行期 counter 硬证据，而不是只根据启动参数推断；双 batch 也确实
进入了生产循环，不是离线 scheduler 模拟。

## 6. 被否决的 SLO 调度方向

`tight target guard=2` 会在 tight 请求 `a_need > 0` 时连续两个 cycle 延后同 home 的
loose proposal，再放行一个 cycle。v22 中它产生 1289 个 guard cycle、延后 15489 次
loose proposal，使 loose TPOT P50 上升到 120.99 ms，并把 decode round 增至 2774。
虽然 proposal 始终整条保留、没有截断 gamma 链，也没有图回退，但 raw、Goodput 和
tight 达成率都下降，因此不进入生产默认值。

这一结果说明固定 gamma4/B64 下 target 并非主要拥塞点；强制 target 侧让路会把压力
转移到 draft 与 home 轮转，不能替代减少真实 refill/通信/计算开销。

## 7. 代码、测试与原始证据

主要实现：

- `vllm_ascend/spec_decode/pearl/native_engine.py`：批量 KV snapshot/restore、初始
  service-window preload、连续物理页 D2H/H2D 直拷与 lazy admission；
- `tests/ut/spec_decode/test_pearl_native.py`：跨新旧物理页恢复、两请求共享 host arena、
  连续页禁止退回 `index_copy_` 的数值测试；
- `vllm_ascend/spec_decode/pearl/spec_rhythm.py`：整 proposal 的 target 选择/延后，
  用于验证并否决 tight guard 方向。

验证结果：KV-ready snapshot/restore 测试 3 passed、arrival rebase 测试 1 passed、相关
SLO scheduler 测试 4 passed；`py_compile`、Ruff 和 `git diff --check` 均通过。

原始证据：

- baseline：`/root/data/nano-pearl-benchmark-results/20260917-specslo-further-optimization/baseline-tp4-coalesce4-author463/result.json`；
- v17：`/root/data/nano-pearl-benchmark-results/20260917-specslo-further-optimization/kv-ready-linear-replay-first-author463-b64-v17/result.json`；
- v21：`/root/data/nano-pearl-benchmark-results/20260917-specslo-further-optimization/kv-ready-batched-kv-preload-author463-b64-v21/result.json`；
- v22 否决实验：`/root/data/nano-pearl-benchmark-results/20260917-specslo-further-optimization/kv-ready-batched-kv-preload-tightguard2-author463-b64-v22/result.json`；
- v24 当前候选：`/root/data/nano-pearl-benchmark-results/20260917-specslo-further-optimization/kv-ready-direct-contiguous-preload-author463-b64-v24/result.json`，SHA256
  `909bef4288703591b69842ebe1f4f33c50afbbcae8986fc22a3058be3cc2090e`。

本报告只证明固定 gamma4、RPS4、B64、作者 463 请求工作负载上的结果；不能外推为动态 B、
其他 RPS/batch、树形 draft 或单资源池冷启动 E2E 的完成结论。
