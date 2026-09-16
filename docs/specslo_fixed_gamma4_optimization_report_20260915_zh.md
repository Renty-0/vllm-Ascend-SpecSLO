# SpecSLO 固定 gamma=4 优化与 1.3913×（约 1.4×）Goodput 验收报告

- 更新时间：2026-09-15
- 当前链式投机解码分支：`SpecSLO-Chain`
- 测试后整理的代码提交：[`032527b3f7a7258376d204f7a4ad79f4d5c1eff8`](https://github.com/Renty-0/vllm-Ascend-SpecSLO/commit/032527b3f7a7258376d204f7a4ad79f4d5c1eff8)
- 远端仓库：[`Renty-0/vllm-Ascend-SpecSLO`](https://github.com/Renty-0/vllm-Ascend-SpecSLO/tree/SpecSLO-Chain)

## 1. 汇报结论

本轮完成了用户指定的固定 `gamma=4` SpecSLO 验收点：

- Qwen3-0.6B draft TP1 + Qwen3-32B target TP3；
- 原生 vLLM-Ascend Qwen3-32B target-only TP4 baseline；
- 昇腾 910B2，四张相同物理 NPU；
- HumanEval、Alpaca、CNN/DailyMail 按 6:2:2 混合；
- tight/normal/loose TPOT 约束分别为 40/50/150 ms；
- RPS=4、batch 上限 64、60 个请求、每请求输出 256 token；
- 固定 `gamma=4`、串行 linear draft，ACLGraph 正式计时窗口零回退。

三次正式候选的论文口径 TPOT 达成率为 **90.00%--93.33%**。以三次
baseline 中 Goodput 最高的一次作为最严格分母，最差候选仍达到：

```text
520.534792 / 374.132495 = 1.391311×
```

候选/基线中位 Goodput 比为 **1.489712×**。因此，本轮通过的是“约 1.4×
Goodput”门禁；不是 1.4× 原始 token 吞吐。候选原始吞吐中位数为
578.372 token/s，基线为 622.519 token/s，仅为 **0.9291×**。收益来自更多请求跨过
TPOT 阈值，而不是总 token 计算量变成原来的 1.4 倍。

正式同合同的早期负优化点为 **0.1672× baseline Goodput**，最终保守结果为
**1.3913×**。绝对 Goodput 从 62.538 提高到 520.535 token/s，即整体提高
**8.323×（+732.35%）**。这是完整工程演进结果，不是可以分摊给某一个 kernel 的
单项收益，也不能把下文各阶段百分比相加。

## 2. 名称、范围与口径

### 2.1 SpecSLO 与 nano-PEARL

本报告中的系统称为 **SpecSLO**，论文中的调度机制称为 **SpecRhythm**。
nano-PEARL 提供了双模型、异构 TP、KV cache 和投机验证基础；SpecSLO 在其上增加
双 batch、rolling eager、SLO-aware 调度和预算整形。由于代码沿用了早期迁移目录，
部分实现仍位于 `vllm_ascend/spec_decode/pearl/`，这不代表本报告在用 PEARL 的历史
吞吐冒充 SpecSLO 结果。

### 2.2 本次固定 gamma 验收不等于动态 B 验收

本次正式性能点遵循用户要求，令每个获选请求固定连续 draft 4 个 token，使用
`serial_linear` 和 `tree_width=tree_depth=1`；`spec_rhythm_roofline`、
`spec_rhythm_verification_budget`、`spec_rhythm_draft_token_budget` 均为空。
因此它**没有消费离线 `B_roof` 表，也没有用论文 §4.4 的逐请求可变候选数产生这组
1.3913× 数据**。

通用动态实现仍在
[`SpecRhythmBudgetShaper`](../vllm_ascend/spec_decode/pearl/spec_rhythm.py) 中：它能查询
全局 verification roof、先满足紧急请求，再用剩余预算最大化 Goodput，并把标量预算
转换为请求级 tree plan。该路径需要作为另一份动态 B 实验单独验收，不能倒推成本次
固定 gamma 结果。

### 2.3 Goodput 统计边界

请求 TPOT 使用论文的 `decode_elapsed / output_tokens` 语义；项目 Goodput 为满足 TPOT
请求的输出 token 数除以**项目实测 E2E 时间**。E2E 从 arrival origin 开始，包含
enqueue、worker IPC、prefill 和 decode；输入已 tokenize，warmup 不计入正式时间。

论文 §6.1 使用生产 trace 重放，并将 prefill 与 decode 分离。本轮使用固定 seed 的
synthetic Poisson 到达和 E2E 分母，所以这是面向当前验收合同的 Ascend 复现结果，
不是对论文实验表格的逐数值复刻。论文相对 best-performing baseline 公开报告的最高
Goodput 提升是 1.27×；
“至少 1.3×”是本项目验收目标，不应写成论文原始结论。

## 3. 公平实验合同

| 项目 | SpecSLO 候选 | baseline |
| --- | --- | --- |
| 模型 | Qwen3-0.6B draft + Qwen3-32B target | Qwen3-32B target-only |
| 拓扑 | draft TP1 + target TP3 | target TP4 |
| 物理 NPU | 1、2、3、4 | 1、2、3、4 |
| 数据配额 | HumanEval/Alpaca/CNN-DM = 36/12/12 | 同一 manifest、同一 token IDs |
| TPOT SLO | 40/50/150 ms | 相同 |
| 到达 | synthetic Poisson，固定 seed，RPS=4 | 相同 arrival timestamps |
| 工作量 | 60 prompts，batch cap=64，输出 60×256 token | 相同 |
| 图模式 | draft PA ACLGraph + target packed-FIA ACLGraph | 原生 vLLM-Ascend graph |
| 投机方式 | fixed gamma4，serial-linear full window | 不适用 |
| warmup | graph fixed point 后 seal；不计入 E2E | 不计入 E2E |

正式 manifest：
`/root/data/specslo-workloads/rps4_mix6-2-2_p60_t256_exact_fixedg4.jsonl`，SHA256：
`b4a412ade4ca744e2524c69f96d44e4fbf9aef17ece364ae18dec42595382b36`。

## 4. 论文机制与当前实现的对应关系

以下章节号以 `/root/data/atc26-paper1664.pdf` 为准；参考代码审计固定在
[`rzwang22/atc26v0@6b1cebf`](https://github.com/rzwang22/atc26v0/tree/6b1cebfd3f02105cb08d6b0f222f1b9b416f7363)。ACLGraph、
replay-first、Triton-Ascend verdict 和 HCCL transport 是为了让这些机制在昇腾执行而
增加的后端设计，并非论文声称已有的 Ascend 实现。

| 论文位置 | 论文机制 | 当前实现 | 本次固定 gamma 路径是否启用 |
| --- | --- | --- | --- |
| §4.2、Fig. 5(a) | Dual-Batch Execution Pipeline | 两个持久 logical home 轮换；一个 home verify 时，另一 home draft；周期末统一切换 | **启用** |
| §4.3、Fig. 5(b) | Rolling Eager Continuation | 用 `a_need`、urgency 和接受收益选择提前猜测；full accept 才 promotion，reject 则 invalidation | **启用** |
| §4.4、Fig. 5(c) | Individual Speculative Budget Shaping | 通用 shaper 实现全局 B 内“紧急请求优先 + 剩余 Goodput 最大化”，保留祖先依赖 | **实现但本次未消费** |
| §5.1、Fig. 6 | Scheduler、state tracker、budget shaper、step plan、execution/state management | 控制面和 native 执行面分层，proposal 带 request/proposal/epoch/home 元数据 | **除动态 B 外启用** |
| §5.2 | 双 logical slot、rolling eager 的 exact dependency、guarded commit | ready/staged-eager 生命周期，验证前校验，完整 batch 后原子 commit/rollback | **启用** |
| §5.3 | 离线测量 `B_roof`、merged draft、flattened tree、运行时上限 | strict roofline schema、tree/linear 执行和上限检查已实现 | **本次固定 gamma 未用 `B_roof`** |

主要代码入口：

- SLO state、`a_need`、动态预算、双 batch 和 guarded lifecycle：
  [`spec_rhythm.py`](../vllm_ascend/spec_decode/pearl/spec_rhythm.py)；
- 固定 full-window 执行、双模型并行提交、online admission 和原子状态提交：
  [`native_engine.py`](../vllm_ascend/spec_decode/pearl/native_engine.py)；
- ACLGraph capture/qualification/replay-first/seal：
  [`native_graph.py`](../vllm_ascend/spec_decode/pearl/native_graph.py)；
- 服务和 benchmark 配置边界：
  [`api.py`](../vllm_ascend/spec_decode/pearl/api.py) 与
  [`benchmark_nano_pearl_speculative.py`](../examples/benchmark_nano_pearl_speculative.py)。

## 5. 从负优化到约 1.4×：正式 Goodput 演进

下面各行都使用相同 RPS4/B64/P60/T256 exact manifest，但旧阶段的源码、graph
配置或候选结构并非全部相同。“相邻增量”只描述观测到的阶段变化；只有明确标为
“同源码、受控 A/B”的行才属于参数范围清楚的策略组合对照。

统一比较分母为最强 TP4 baseline Goodput `374.132495 token/s`。

| 阶段 | raw tok/s | TPOT 达成 | Goodput tok/s | 相对 baseline | 相邻变化与证据等级 |
| --- | ---: | ---: | ---: | ---: | --- |
| v25 tree/B120 初始 | 312.689 | 20.00% | 62.538 | 0.1672× | 正式负优化起点；单次 |
| v31 改 2×3 tree | 311.955 | 20.00% | 62.391 | 0.1668× | 无收益；单次 |
| v33 publish + packed cache | 319.017 | 20.00% | 63.803 | 0.1705× | vs v31 Goodput +2.26%；组合、单次 |
| v36 no-split/no-draft-map/mask-KV | 347.035 | 20.00% | 69.407 | 0.1855× | vs v33 +8.78%；组合、单次 |
| v40 eager reserve/KV graph | 316.877 | 13.33% | 42.250 | 0.1129× | vs v36 -39.13%；多参数，否决 |
| v49 serial/no-B/full-chain | 344.018 | 20.00% | 68.804 | 0.1839× | vs v36 -0.87%；架构切换，不是单项收益 |
| v53 dense12 | 337.041 | 20.00% | 67.408 | 0.1802× | vs v49 -2.03%；否决 |
| v55 same-bucket | 366.609 | 20.00% | 73.322 | 0.1960× | vs v53 +8.77%；方向性单次 |
| v59 paid headroom | 351.107 | 20.00% | 70.221 | 0.1877× | vs v55 -4.23%；否决 |
| full-window/draft-RF/KV512/coalesce2 组合 | 523.799 | 28.33% | 148.410 | 0.3967× | vs 相邻 v59 raw +49.19%、GP +111.35%；vs 此前最优 v55 raw +42.88%、GP +102.41%；组合包 |
| coalesce4、无 target replay-first | 518.829 | 30.00% | 155.649 | 0.4160× | target-RF 近 A/B 左侧 |
| coalesce2、target-RF + 图生命周期 | 576.989 | 73.33% | 423.125 | 1.1310× | 新图执行组合；非单项 A/B |
| coalesce4、target-RF + 图生命周期 | 581.291 | 91.67% | 532.850 | 1.4242× | **同源码、受控单次策略 A/B：vs 上行 GP +25.93%** |
| 最终测试工作树三次 | 573.285--587.930 | 90.00--93.33% | 520.535--548.735 | **保守 1.3913×** | 三次正式重复确认 |

这条曲线说明大 B120 并不会自动挽救性能，因此本轮优先修复树形控制、动态 shape、
图生命周期和到达批处理，而不是反复重测 B。先固定 gamma 并修通完整 graph pipeline
之后，才出现稳定的 SLO/Goodput 收益；由于演进不是单变量 B A/B，这一现象不能用于
排除 B 本身的影响。

Goodput 是阈值型指标。raw 只改变少量时，许多 tight 请求可能一起从略高于 40 ms
移动到略低于 40 ms，于是 Goodput 会跃升。所以上表的 +242.34% 或 +25.93% Goodput
不能解释成同等幅度的 kernel 加速。

## 6. 每种有效优化用了什么方法、提高了多少

### 6.1 同源码、受控但单次的策略 A/B：online prefill coalesce

方法：等待同一到达窗口内累计至少 4 个请求再做一次 packed prefill，最长等待
1100 ms；减少零散小 prefill 对 decode graph 的干扰。对应代码位于
[`api.py`](../vllm_ascend/spec_decode/pearl/api.py) 的配置校验和
[`native_engine.py`](../vllm_ascend/spec_decode/pearl/native_engine.py) 的 online admission/
prefill coalescing。

同一 tracked source diff、同一模型、卡、manifest 和 target replay-first 下，仅将
`2 requests / 600 ms` 改为 `4 requests / 1100 ms`：

| 指标 | coalesce=2 | coalesce=4 | 变化 |
| --- | ---: | ---: | ---: |
| raw throughput | 576.989 | 581.291 | **+0.746%** |
| TPOT 达成 | 44/60 = 73.33% | 55/60 = 91.67% | **+18.33 个百分点** |
| Goodput | 423.125 | 532.850 | **+25.932%** |

这是本轮唯一同源码、受控的正式策略组合 A/B，但同时调整了 `min_requests` 和
`max_wait_ms`，且每侧只有一次运行。因此 +25.932% 是这组 coalesce 策略的观测增量，
不是单个参数或多次统计意义上的独立因果值。它属于 Ascend 服务工程优化，不是论文
三项算法机制之一；其目标是保护论文 §4.2 的 decode overlap 窗口。

### 6.2 组合前后对照：target replay-first 与配套 graph 生命周期

方法：预先完成输入 staging/event 建立，正式周期先 replay graph，再在受控事件边界
更新下一轮 task；同时增加 graph qualification、无效 entry prune、fixed-point seal 和
正式区间 fail-closed 门禁。这样避免每个 target verify 周期在 graph 前后重复做同步、
校验或重新捕获。

代码位于 [`envs.py`](../vllm_ascend/envs.py) 的 replay-first 开关、
[`native_graph.py`](../vllm_ascend/spec_decode/pearl/native_graph.py) 的 replay/capture/seal
实现，以及 benchmark 的 qualification/fallback gate。

在相同 coalesce4 可见合同下：

- raw `518.829 → 581.291 token/s`，观察增量 **+12.04%**；
- TPOT 达成率 `30.00% → 91.67%`，提高 **61.67 个百分点**；
- Goodput `155.649 → 532.850 token/s`，观察增量 **+242.34%**。

这不是严格单开关 A/B：两边 tracked diff 从 `5f392…` 变为 `5a37f…`，并伴随图生命周期
和结果 schema 修复。因此只能称为“包含 target replay-first 的图执行组合收益”，不能把
+242.34% 全归到一个环境变量，也不能证明 replay-first 在组合中占主导。

### 6.3 full-window、draft replay-first、KV/graph 容量组合

方法包括：

1. 将固定 gamma4 改为一次产生四个连续 token、一次 target 验证完整窗口；
2. 为 serial draft full chain 预编译 ACLGraph，并使用 draft replay-first；
3. 将 KV blocks 调到 512、graph entry 容量调到 32；
4. 修复 endpoint/timing、到达和 graph qualification；
5. 以固定形状承载逻辑有效行，减少动态 bucket 抖动。

该组合使正式点从 v55 的 raw/Goodput `366.609/73.322` 提高到
`523.799/148.410 token/s`，观察增量分别为 **+42.88% / +102.41%**。由于这些改动一起
进入结果，不能为五个方法分别捏造百分比。

固定 gamma4 选择 serial linear 而不是浅树，是本轮的工程决策，不是论文 §4.4 的
动态 tree budget：只有四个候选时，2×2 浅树牺牲可连续预测长度，并引入 tree plan、
mask、KV compaction 和动态图形状成本。架构切换当下并未立即变快；它的价值是让后续
full-chain graph 和通信协议成为稳定固定形状。

### 6.4 双 batch overlap 与 rolling eager

双 batch 来自论文 §4.2：controller 保持 home A/B，当前周期将 target(A) 和 draft(B)
提交到互斥设备组，下周期交换角色。rolling eager 来自 §4.3：
`a_need = ceil(max(0, projected_progress - delivered_tokens))`，再结合 urgency、近期接受率
和 draft confidence 选择提前猜测；full accept 时 promotion，任一 rejection 时丢弃
continuation。

最终 profile 的 rank 累计计数显示该机制确实执行，而不是只生成 metadata：每 rank
normal proposals 5790、rolling-eager proposals 175、promoted 65、invalidated 110；
1364 个 eligible 中 admitted 175、deferred 1189。由于没有同源码“关闭/开启 rolling
eager”的正式独立 A/B，本报告不为它单列虚构的百分比。

### 6.5 fused verdict、compact transport 与原子 commit

这些是为昇腾实现论文 §5.1/§5.2 执行面的自研扩展：

- [`fixed_greedy_verdict.py`](../vllm_ascend/ops/triton/spec_decode/fixed_greedy_verdict.py)
  用一个 Triton-Ascend kernel 为每行计算 accepted prefix 和 correction token，减少
  Python/D2H 小算子链；
- [`fixed_greedy_transport.py`](../vllm_ascend/spec_decode/pearl/fixed_greedy_transport.py)
  将 B64 的 int64 envelope 从 961 个元素压到 257 个，减少 **704 个（73.26%）**；
- [`native_engine.py`](../vllm_ascend/spec_decode/pearl/native_engine.py) 在设备 verdict、
  KV 写入、proposal epoch 校验全部成功后才提交状态；失败时统一 rollback，避免 partial
  commit。

这三项与 full-window 执行一起进入源码，缺少逐项正式 A/B；可报告的直接机制数据是：
verdict 0.464 ms、D→T compact submit/wait 0.429/0.135 ms、完整 envelope 1.138 ms、
T→D correction 1.009 ms、state update 0.256 ms。

### 6.6 独立短测中的程序优化增量

为了定位 Python/shape 管理开销，另有一条 B8/P8/T16、tree2×2/gamma4、三轮 E2E
短测。它与正式 RPS4/B64 Goodput 合同不同，只能说明方法方向，不能与第 5 节百分比相加。

| 阶段 | E2E 中位 tok/s | 相邻增量 | 主要方法 |
| --- | ---: | ---: | --- |
| graph-validation-fix v25 | 166.922 | — | 修复 first/tail shape 重复 eager；相对 TP4 baseline 1.1783× |
| plan-pack-publish v31 | 174.595 | **+4.597%** | 一次构建并复用 plan、pack、publish 元数据 |
| output-boundary-health v34 | 182.214 | **+4.364%** | 将非有限值/健康检查移到输出边界，减少热循环同步 |
| cached-plans v35 | 184.405 | **+1.203%** | geometry/plan LRU 与 identity-tree fast path |

从 v25 到 v35 合计 `166.922 → 184.405 token/s`，提高 **10.474%**；最终为短测 TP4
baseline `141.663 token/s` 的 **1.3017×**。旧记录中的 167.785 是 inference throughput，
这里统一使用 E2E 中位 166.922，避免混口径。

### 6.7 源码审核索引

下面链接固定到已推送的代码提交 `032527b3`，不会因为后续文档提交导致行号漂移。

| 实现 | 可直接审核的源码范围 |
| --- | --- |
| fixed full-window 状态提交 API | [`native_engine.py:L521-L651`](https://github.com/Renty-0/vllm-Ascend-SpecSLO/blob/032527b3f7a7258376d204f7a4ad79f4d5c1eff8/vllm_ascend/spec_decode/pearl/native_engine.py#L521-L651) |
| full-window 运行时主路径 | [`native_engine.py:L6382-L6784`](https://github.com/Renty-0/vllm-Ascend-SpecSLO/blob/032527b3f7a7258376d204f7a4ad79f4d5c1eff8/vllm_ascend/spec_decode/pearl/native_engine.py#L6382-L6784)、[`L10011-L10189`](https://github.com/Renty-0/vllm-Ascend-SpecSLO/blob/032527b3f7a7258376d204f7a4ad79f4d5c1eff8/vllm_ascend/spec_decode/pearl/native_engine.py#L10011-L10189)、[`L10492-L10647`](https://github.com/Renty-0/vllm-Ascend-SpecSLO/blob/032527b3f7a7258376d204f7a4ad79f4d5c1eff8/vllm_ascend/spec_decode/pearl/native_engine.py#L10492-L10647) |
| 双 home 调度与 guarded lifecycle | [`spec_rhythm.py:L561-L863`](https://github.com/Renty-0/vllm-Ascend-SpecSLO/blob/032527b3f7a7258376d204f7a4ad79f4d5c1eff8/vllm_ascend/spec_decode/pearl/spec_rhythm.py#L561-L863) |
| 动态 B/逐请求预算（本次结果未消费） | [`spec_rhythm.py:L175-L493`](https://github.com/Renty-0/vllm-Ascend-SpecSLO/blob/032527b3f7a7258376d204f7a4ad79f4d5c1eff8/vllm_ascend/spec_decode/pearl/spec_rhythm.py#L175-L493)、[`schedule:L1028-L1118`](https://github.com/Renty-0/vllm-Ascend-SpecSLO/blob/032527b3f7a7258376d204f7a4ad79f4d5c1eff8/vllm_ascend/spec_decode/pearl/spec_rhythm.py#L1028-L1118) |
| 双模型并发 cycle | [`native_engine.py:L6142-L6562`](https://github.com/Renty-0/vllm-Ascend-SpecSLO/blob/032527b3f7a7258376d204f7a4ad79f4d5c1eff8/vllm_ascend/spec_decode/pearl/native_engine.py#L6142-L6562) |
| online prefill coalesce | [`native_engine.py:L4867-L5154`](https://github.com/Renty-0/vllm-Ascend-SpecSLO/blob/032527b3f7a7258376d204f7a4ad79f4d5c1eff8/vllm_ascend/spec_decode/pearl/native_engine.py#L4867-L5154) |
| graph qualify/prune/seal/fail-closed | [`native_graph.py:L589-L757`](https://github.com/Renty-0/vllm-Ascend-SpecSLO/blob/032527b3f7a7258376d204f7a4ad79f4d5c1eff8/vllm_ascend/spec_decode/pearl/native_graph.py#L589-L757) |
| draft replay-first | [`native_graph.py:L1070-L1147`](https://github.com/Renty-0/vllm-Ascend-SpecSLO/blob/032527b3f7a7258376d204f7a4ad79f4d5c1eff8/vllm_ascend/spec_decode/pearl/native_graph.py#L1070-L1147) |
| target replay-first | [`native_graph.py:L2194-L2275`](https://github.com/Renty-0/vllm-Ascend-SpecSLO/blob/032527b3f7a7258376d204f7a4ad79f4d5c1eff8/vllm_ascend/spec_decode/pearl/native_graph.py#L2194-L2275) |
| fixed greedy fused verdict | [`fixed_greedy_verdict.py:L22-L111`](https://github.com/Renty-0/vllm-Ascend-SpecSLO/blob/032527b3f7a7258376d204f7a4ad79f4d5c1eff8/vllm_ascend/ops/triton/spec_decode/fixed_greedy_verdict.py#L22-L111) |
| compact transport | [`fixed_greedy_transport.py:L35-L356`](https://github.com/Renty-0/vllm-Ascend-SpecSLO/blob/032527b3f7a7258376d204f7a4ad79f4d5c1eff8/vllm_ascend/spec_decode/pearl/fixed_greedy_transport.py#L35-L356) |
| graph qualification 与正式门禁 | [`benchmark_nano_pearl_speculative.py:L683-L800`](https://github.com/Renty-0/vllm-Ascend-SpecSLO/blob/032527b3f7a7258376d204f7a4ad79f4d5c1eff8/examples/benchmark_nano_pearl_speculative.py#L683-L800)、[`L1384-L1794`](https://github.com/Renty-0/vllm-Ascend-SpecSLO/blob/032527b3f7a7258376d204f7a4ad79f4d5c1eff8/examples/benchmark_nano_pearl_speculative.py#L1384-L1794) |
| replay 数值诊断 | [`check_specslo_target_replay_correctness.py:L208-L432`](https://github.com/Renty-0/vllm-Ascend-SpecSLO/blob/032527b3f7a7258376d204f7a4ad79f4d5c1eff8/examples/check_specslo_target_replay_correctness.py#L208-L432) |

## 7. 最终性能结果

### 7.1 baseline 三次

| run | E2E s | raw tok/s | 论文口径 TPOT 达成 | Goodput tok/s |
| --- | ---: | ---: | ---: | ---: |
| schema-v3-targetenv | 24.6330 | 623.554 | 36/60 = 60.00% | 374.132 |
| final-repeat1 | 24.7381 | 620.904 | 34/60 = 56.67% | 351.845 |
| final-repeat2 | 24.6740 | 622.519 | 34/60 = 56.67% | 352.761 |

### 7.2 正式候选三次

| run | E2E s | raw tok/s | 论文口径 TPOT 达成 | Goodput tok/s | gate/最强 baseline |
| --- | ---: | ---: | ---: | ---: | ---: |
| current-head-repeat1 | 26.7929 | 573.285 | 55/60 = 91.67% | 525.511 | 1.4046× |
| current-head-repeat2 | 26.1256 | 587.930 | 56/60 = 93.33% | 548.735 | 1.4667× |
| current-head-profile500-repeat3 | 26.5573 | 578.372 | 54/60 = 90.00% | 520.535 | 1.3913× |

按请求类型，候选三次 tight 达成 30/31/32（总数 36），normal 和 loose 均为 12/12；
基线 tight 只有 11/11/12。最终收益主要来自 40 ms tight 请求，而不是宽松请求重复计数。

三次候选实际运行时的 provenance 为 Git base revision `1421e3c7` 加未提交 tracked diff
`ea394ee1…`；随后才将测试工作树中的实现、测试和文档整理为 `032527b3` 并推送。
提交中包含本报告审核的实现，但测试后还有注释/文档整理，尚未在 `032527b3` 提交状态下
重新进行四卡正式复跑，所以本报告不把结果表述为“commit-exact 性能认证”。两次
`baseline-final` 与候选使用同一 `ea394ee1…` tracked diff；Goodput 更高的
`schema-v3-targetenv` baseline 来自较早的 `cd63e48…` diff，刻意把它作为更难超过的
保守分母。若只比较同 diff，最差候选 / 较好 `baseline-final` 为 1.4756×。

## 8. Profiling 证据链

### 8.1 Host 双 batch 时间线

将四个 rank 的本地时间戳按 step 合并为 500 个 cycle，其中 479 个同时具有 draft 和
target 工作。按同 step 的 `worker_host_timeline` 明确排除 16 个在线 prefill cycle 后，
得到 463 个纯 decode 稳态 dual cycle；没有按 wall time 事后删除慢样本。

| 阶段 | 均值 ms | 解释 |
| --- | ---: | --- |
| cycle wall | 42.466 | rank 最大墙钟窗口 |
| scheduler | 1.432 | plan、budget、ticket 和控制面 |
| Draft full-chain host submit | 36.006 | TP1 四步 full-chain graph 异步提交窗口 |
| Target forward host submit | 14.223 | TP3 packed-FIA target graph 异步提交窗口 |
| Draft/Target host-submit overlap | 14.217 | 覆盖 target host-submit 短窗口的 **99.96%** |
| Target verdict | 0.464 | fused fixed-greedy verdict |
| Draft→Target full envelope | 1.138 | compact submit 至 exchange 完成，含 host bookkeeping |
| └ compact submit / wait | 0.429 / 0.135 | HCCL 提交/等待窗口 |
| Target→Draft correction | 1.009 | correction/commit 临界窗口 |
| wait/sync coordination | 23.662 | 短 target 等长 draft；与 draft 窗口重叠，不能重复相加 |
| state update | 0.256 | 原子请求状态更新 |
| post-correction tail | 0.923 | correction 后尾部工作 |

这两个数是 CPU 从调用到异步入队返回的时间；当前路径没有在 forward 返回前调用
`torch.npu.synchronize()`，因此 **14.223 ms 不是 target 的 NPU 完成延迟**，也不包含后续
device verdict。若只按 host 提交窗口串行，两路约为
`36.006 + 14.223 = 50.229 ms`；实际 host 交集
14.217 ms，扣除交集约 36.012 ms。这证明 worker/host 提交窗口发生了双 batch overlap，
但不能单凭 host 时间戳宣称物理 AI Core 完全重叠。

16 个带 online prefill 的 dual cycle 中，prefill 均值为 197.413 ms，全 dual cycle
wall 均值升至 48.298 ms；这一 profiling 现象与 coalesce 的受控 A/B 结果一致，并能
帮助解释其阈值收益，但不能单独作为因果证明。

### 8.2 CANN 设备 kernel overlap

独立静态 B8/P8/T32 诊断使用相同 TP1+TP3、fixed gamma4、serial-linear、
replay-first 和 sealed graph。profiler active 的 3 个 cycle 均为 draft4 + target4；四个
rank 各 27 次 graph replay，capture、runtime validation 和 fallback 均为 0。

| target rank / NPU | strict AI Core overlap | AI Core + MIX_AIC | 全设备计算 overlap |
| --- | ---: | ---: | ---: |
| rank 1 / NPU 2 | 10.538 ms（draft busy 54.58%） | 16.571 ms | 24.766 ms |
| rank 2 / NPU 3 | 10.632 ms（55.09%） | 16.572 ms | 24.746 ms |
| rank 3 / NPU 4 | 10.653 ms（55.18%） | 16.630 ms | 24.715 ms |

三个 target rank 都存在真实 `MatMul` 与 draft `MatMul` 的同时执行片段。三个 pair
不能相加，因为同一 draft 区间可以同时覆盖三个 target rank。profiler 会扰动 wall
time，因此该 trace 只证明 sampled physical overlap，不代替第 7 节无 profiler 性能值。

### 8.3 正式图模式零回退

| candidate run | 每个 rank replay | capture attempt/capture | runtime validation | failed/capacity/shape fallback |
| --- | ---: | ---: | ---: | ---: |
| repeat1 | 544 | 0/0 | 0 | 0/0/0 |
| repeat2 | 541 | 0/0 | 0 | 0/0/0 |
| profile repeat3 | 528 | 0/0 | 0 | 0/0/0 |

正式计时前使用完整 workload qualification 至 fixed point，prune 未验证 entry 后 seal。
rank 0 全部进入 `spec_rhythm_linear_draft_full_chain` replay；rank 1--3 全部进入 target
generic packed-FIA replay。seal 后遇到未命中 shape 会直接失败，不会静默回 eager。

### 8.4 数值正确性与软件回归

- 静态 B8/P8/T32 greedy 回归共执行 6 个 case：1 个 packed-FIA eager oracle、
  1 个 update-first control、3 个 replay-first 以及 1 个 validate-every-replay；oracle
  之外五个对照的逐 token、输出 SHA256、轮数和接受统计一致；
- replay-first target ranks 均为 25 次 replay、0 fallback；强制校验为 25 次 validation、
  0 failure；
- 完整 `tests/ut/spec_decode`：**1179 passed、13 skipped、16 warnings**；
- 当前修改 Python 的 Ruff、`git diff --check`、正式 runner `bash -n` 和新示例
  `py_compile` 均通过。

这证明当前 greedy fixed-gamma4 验收形状正确；不能外推为 stochastic sampling、其他
模型或所有 context/batch shape 的完整数值证明。

## 9. 已否决或暂不进入正式配置的方向

| 方向 | 结果 | 决策 |
| --- | --- | --- |
| 增大 B10/B12/B16 | 短测均低于 B8 最优点 | 固定 gamma4 验收不再用大 B 掩盖程序问题 |
| tree2×3 / B12 | 仅 164.856 tok/s（短测） | 四候选下连续 linear horizon 更合适 |
| dead-mask/skip mask copy | E2E 中位 165.720，较 v25 -0.72% | 否决 |
| eager reserve/KV graph v40 | 正式 Goodput -39.13% | 否决该组合 |
| dense12 | 正式 Goodput -2.03% | 否决 |
| paid headroom | 正式 Goodput -4.23% | 否决 |
| 非均匀 TP3 attention head padding | 无稳定收益 | 不进入生产默认 |
| TP3 MC2 custom kernel | 当前 CANN 上没有形状同时满足数值与性能门禁 | 正式路径回退生产 matmul + all-reduce |

当前 CANN/HCCL 还存在一个必须保留的安全边界：若在 target TP all-reduce 尚处于设备
队列时提前 post 跨模型 D→T receive，会形成跨 communicator stream wait cycle 并死锁。
当前顺序是：draft/target model phases overlap → 同步各自本地 model stream → Gloo
coordination → compact HCCL。不能把 D→T 通信误报为已与 TP3 all-reduce 重叠。

## 10. 尚未由本报告证明的范围

- RPS=2 以及 batch=8/16/32 的完整 Goodput 矩阵；
- 论文 §4.4/§5.3 动态 `B_roof` 和逐请求可变 gamma 的正式同合同收益；
- stochastic sampling、长上下文、Llama/Qwen2.5 等其他模型组合；
- TTFT SLO。coalesce=4/1100 ms 可能增加 TTFT，加入 TTFT 门禁后必须重新优化；
- 冷启动性能。候选需要 3--4 次未计时 workload qualification，本文只比较双方都
  排除 warmup 的稳态 E2E；
- 与 AdaServe 等更强 speculative baseline 的比较。本文 baseline 是用户指定的原生
  vLLM-Ascend TP4 target-only；
- 原始 token 吞吐超过 baseline。当前达到的是 Goodput 目标。

## 11. 证据索引

### 11.1 正式结果和 profiling

根目录：
`/root/data/nano-pearl-benchmark-results/20260914-specslo-fixed-gamma4-full-window/`

以下原始 JSON、日志和约 141 MB CANN trace 当前只保存在本机结果目录，未随 GitHub
仓库上传；GitHub 中保存的是源码和本文抽取后的结果/哈希/证据索引。

- 正式综合报告：`README.md`；
- host/CANN 摘要：`PROFILING.md`；
- baseline：`runs/baseline-schema-v3-targetenv-20260914T2305Z/`、
  `runs/baseline-final-repeat1-20260914T2315Z/`、
  `runs/baseline-final-repeat2-20260914T2305Z/`；
- candidate：`runs/formal-current-head-repeat1-20260914T2310Z/`、
  `runs/formal-current-head-repeat2-20260914T2315Z/`、
  `runs/formal-current-head-profile500-repeat3-20260914T2320Z/`；
- 受控单次 coalesce 策略 A/B：
  `runs/formal-target-replayfirst-coalesce2-host500-20260914T2245Z/`
  与 `runs/formal-target-replayfirst-coalesce4-host500-20260914T2255Z/`；
- 数值回归：
  `runs/static-replay-correctness-current-head-v3-20260914T2240Z/result.json`；
- CANN overlap：`validation/device-overlap-fixedg4-retry-5OxoEK/`；
- 全量单测：`validation/pytest-spec-decode.log`。

### 11.2 历史消融

- 正式 Goodput 负优化阶段：
  `/root/data/nano-pearl-benchmark-results/20260914-specslo-fixed-gamma-goodput/`；
- B8 三轮 tree/program 消融：
  `/root/data/nano-pearl-benchmark-results/20260910-specslo-performance-tuning/`；
- 现有 profiling 工作记录：
  [`specslo_profiling_and_optimization_20260911_zh.md`](specslo_profiling_and_optimization_20260911_zh.md)；
- 项目总工作记录：[`specslo_work_record_zh.md`](specslo_work_record_zh.md)。

## 12. 可直接用于汇报的一段话

> 我们在昇腾 910B2 上实现了 SpecSLO 的双 batch、rolling eager 和 guarded proposal
> lifecycle，并为固定 gamma4 路径设计了 serial full-window ACLGraph、replay-first
> task update、sealed graph 零回退门禁、融合 greedy verdict、紧凑 HCCL envelope 和
> online prefill coalesce。正式 Qwen3-0.6B TP1 + Qwen3-32B TP3 对原生 Qwen3-32B TP4
> 的 RPS4/B64/P60/T256 测试中，TPOT 达成率由 baseline 的 56.67%--60.00% 提升到
> 90.00%--93.33%，最保守 Goodput 比为 1.3913×，中位比为 1.4897×。Host profiling
> 显示 target host-submit 14.223 ms 窗口的 99.96% 与 draft host-submit 重叠，
> CANN trace 在三个 target rank
> 上分别测得 10.538/10.632/10.653 ms strict AI Core overlap；正式计时窗口每 rank
> 528--544 次 graph replay 且 capture、validation、fallback 全为 0。需要强调，这是一组
> fixed-gamma Goodput 结果，raw throughput 中位仅为 baseline 的 0.9291×，也不等同于
> 论文动态 B 的完整性能矩阵。

## 13. 2026-09-16：四步 draft 设备计时纠偏与后续方向

### 13.1 结论：PARD 不进入主路径

此前把 `worker_draft_seconds / steps` 与 `worker_target_seconds / steps` 当成模型计算时间，
会把 host 入队、task update 和同步等待混进 compute。使用跨 cycle 携带的 NPU Event
重新计时后，P128/T256、fixed gamma4、TP1 draft + TP3 target 的 475 个可比较 cycle 中，
**475/475 个 cycle 都是完整四步 draft device 窗口低于 target 可消费 device 窗口**。
正确的可比较周期均值为：

| 筛选范围 | cycle 数 | 四步 causal AR draft | target forward + device verdict |
| --- | ---: | ---: | ---: |
| 所有双边 Event 均有效的 cycle | 475 | 33.508 ms | 44.506 ms |
| 再排除 mixed target-prefill | 454 | 33.493 ms | 37.256 ms |

报告初版写的 31.833/43.092 ms 是对 500 行 dashboard 直接求均值：draft 因为
Event 滞后一个 cycle，前 25 行为 0，使 31.833 ms 被人为压低；target 的
43.092 ms 则还混入 mixed target-prefill cycle。该口径与“475 个可比较 cycle”不一致，
现已更正。

另一个必须区分的边界是：8.1 的 14.223 ms 是 P60 稳态纯 decode 的 target
**host 异步提交窗口**，而本节 Event 在 target forward 前记录起点、在
`_verify_target_tokens_batch()` 后记录终点，测的是 **target forward + device verdict**
整个可消费窗口。因此 14.223 ms 与 37--45 ms 不是同一个指标，不代表 target
性能从 14 ms 退化到 43 ms。现有 Event 还不能单独给出不含 verdict 的 target-forward-only
device 时间；若要回答这个更严格的问题，必须在 forward 返回后、verdict 之前再加一个
NPU Event 边界。

这里的 draft device 数值是 `root -> d1 -> d2 -> d3 -> d4` 整条四步链，不是单个 draft
token 的时间。四个依赖 forward 捕获在一次 ACLGraph 中，稳态只 replay 一次。因此，
“用 PARD 避免四步 draft 比 target 慢”没有成立的前提。PARD 的 B2 eager 试验虽然降低
draft 时间，但接受率由 serial 的 93.75% 降到 65%，E2E 仅改善约 4.1%；该路径保持
实验性、默认关闭，不作为 SpecSLO-Chain 的优化方向。

### 13.2 persistent FIA staging 受控 A/B：降低 draft，但没有提高 E2E

为了确认链外 metadata 是否仍拖慢 draft，实现了默认关闭的 persistent FIA staging：
复用四步 input IDs、position、slot mapping、FULL mask 和 common-KV request table，避免
每 cycle 重建 Python metadata wrapper。相同 P128/T256 合同的单次受控 A/B 如下：

| 指标 | OFF（当前源码） | ON | 变化 |
| --- | ---: | ---: | ---: |
| 四步 draft device mean（有效 Event cycle） | 33.508 ms | 29.557 ms | -11.79% |
| target 可消费 device mean（有效 Event cycle） | 44.506 ms | 45.502 ms | +2.24% |
| cycle wall mean | 48.520 ms | 51.341 ms | +5.81% |
| coordination critical mean | 26.456 ms | 30.752 ms | +16.24% |
| accounting tail mean | 1.177 ms | 3.229 ms | +174.23% |
| raw throughput | 720.155 tok/s | 713.889 tok/s | -0.87% |
| 论文口径 Goodput | 630.136 tok/s | 552.149 tok/s | -12.38% |
| TPOT 达成率 | 87.50% | 77.34% | -10.16 pp |

两边 graph qualification 都到 fixed point，draft/target/mixed-target fallback 与 runtime
validation failure 都为 0。ON 路径记录 3022 hits、71 misses。结果说明 staging 确实
缩短 draft device 时间，但 target 已是关键路径；额外 staging copy/刷新还扩大了
coordination 和尾部窗口，最终 raw 反而下降。该实现继续保持默认关闭，不进入正式配置。

证据：

- OFF：`/root/data/nano-pearl-benchmark-results/20260916-specslo-serial-chain-current-profile/runs/p128-c4-stable-barrier-persistent-staging-off-current-host500/result.json`；
- ON：`/root/data/nano-pearl-benchmark-results/20260916-specslo-serial-chain-current-profile/runs/p128-c4-stable-barrier-persistent-staging-on-host500/result.json`。

### 13.3 新发现的数值回归风险

同一 prompt hash、相同 seed 和相同 OFF 配置的两次在线运行，最终 output token hash
也不一致，128 个请求中有 112 个在较后位置出现分歧；OFF 与 ON 则有 107 个请求不同。
因此不能只凭 OFF/ON hash 不同把问题归因给 staging，现有在线双 batch/graph 路径本身
还存在跨运行输出漂移。原生 target-only 在相同 manifest、不同 coalesce/batch shape 下
也会得到不同输出 hash，说明 BF16 kernel/batch shape 与在线 admission 时序至少是一个
混杂因素，不能直接判成 speculative 语义错误。当前正式路径仍通过已有静态
changed-input oracle；后续所有 task-update 优化必须增加固定 active-set、固定 shape 的
逐 token oracle，再单独检查在线跨重复一致性。在该问题解释清楚前，不把任何新候选设为
默认。

### 13.4 下一优化优先级

当前 target graph 每次 replay 对 Qwen3-32B 的 64 层 FIA task 逐层执行 task update；
profiling 显示三个 TP3 rank 每 cycle 的 FIA task submit 约 11.7--12.4 ms，另有约
0.8 ms 的 64 次 event record。CANN 8.5/9.0 的 task-group 接口目前只允许单算子调用，
因此不能安全地把 2/4 层塞进同一个 update handle，也不能减少逐层 begin/end。下一项
实验只让连续 2 层或 4 层共享一个 ExternalEvent：每层 handle/begin/end 保持不变，
组首 wait/reset、组末 record 一次，同时保留 replay-first 和 changed-input oracle。
该路径理论上先回收 event 开销；不通过移除同步边界换取表面性能。

### 13.5 target FIA event-group 实测：微观开销下降，但 E2E 无收益

已按上述边界实现了默认关闭的 target FIA event-group，合法组大小为
1/2/4。它不改变 64 层 attention 的 handle 更新数，只让相邻层共享一个
`ExternalEvent`。group=2 先通过了固定 B8/P8/T32 的 NPU oracle：

- packed-FIA eager oracle、update-first control 和三次 replay-first 的 token IDs、
  输出 hash、轮数、accepted/verified 统计全部一致；
- validate-every-replay case 在每个 target rank 执行 25 次 replay/25 次校验，
  零 failure/fallback。

开启相同 PA task-update 计时插桩的 P128/T256 诊断 A/B 显示，group=1 三个
target rank 每 replay 的 event record 平均约为 0.82--0.87 ms，group=2 约为
0.45--0.50 ms，符合事件数减半的预期。但关闭 host timeline 与 task-update profiling
后的同源正式 A/B 为：

| 指标 | group=1 | group=2 | 变化 |
| --- | ---: | ---: | ---: |
| raw E2E throughput | 715.393 tok/s | 714.810 tok/s | -0.08% |
| inference throughput | 717.305 tok/s | 716.689 tok/s | -0.09% |
| E2E | 45.804 s | 45.842 s | +0.08% |
| mean accepted tokens | 3.876 | 3.866 | -0.010 |
| 论文口径 Goodput | 586.846 tok/s | 558.445 tok/s | -4.84% |
| TPOT 达成率 | 82.03% | 78.13% | -3.91 pp |

两边 graph qualification 均到 fixed point，三个 target rank 的 failed capture、capacity/
shape/eager fallback 和 runtime-validation failure 全为 0。raw 差异处于噪声内且没有
正收益，Goodput 反而因在线 admission 轨迹与 40 ms 阈值放大而下降。因此不跑
group=4，生产默认保持 group=1；该实验证明单纯减少 `event.record` 不是当前
主瓶颈。后续优先级转向 target 每层 handle submit 和 verify 计算本身，而不是
PARD 或继续压缩已低于 target 的 serial gamma4 draft。

证据：

- 固定形状 oracle：
  `/root/data/nano-pearl-benchmark-results/20260916-specslo-target-fia-event-group-correctness-g2.json`；
- 诊断与正式 A/B：
  `/root/data/nano-pearl-benchmark-results/20260916-specslo-target-event-group-ab/runs/`。
