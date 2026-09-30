# SpecSLO 链式路径：插件更新后的续优化记录（2026-09-28）

本文承接 `specslo_optimization_checkpoint_20260928_zh.md`。这里只记录本轮新增实现和实测；**不能把单次预检当作正式三轮 A/B**，也不能把默认关闭的 target replay-ahead 算作生产加速。

## 固定验收口径

- Qwen3-0.6B draft TP1 + Qwen3-32B target TP3，对照 Qwen3-32B target-only TP4；物理 NPU 均为 2、3、4、5。
- 固定串行 `gamma=4`，在线 Poisson RPS=4、463 请求、batch 上限 64、每请求 256 输出 token；tight/normal/loose 为 271/96/96，请求 TPOT SLO 为 40/50/150 ms。
- 在线 prefill 随请求陆续到达，双方 `coalesce1`；warm-up/图资格化不计入正式 E2E，正式区间必须无 graph capture/fallback。
- 候选每轮须同时满足 Raw > 同轮 TP4，以及论文口径 Goodput ≥1.3× 同轮 TP4；达成后才做三轮交错成对 A/B。

固定 manifest：`/root/data/specslo-workloads/eurosys27-rps4-seed0-author-slo40-50-150/rps4.0_mix0.6_0.2_0.2_seed0.jsonl`。

## 本轮代码改动

1. `examples/specslo_benchmark_report.py` 增加 replay-ahead 的四 rank 证据门：每个 rank 的 `attempts = eligible + rejects`、`eligible = submitted = consumed`、零签名不匹配，并要求各 rank 计数一致。计数必须来自每次 measured generate 的 `worker_aclgraph_metrics`，不能错读 ACLGraph-only delta。`run_online_coalesce1_candidate.sh`、`run_online_coalesce1_formal_ab.sh` 传递开关并执行门禁；候选脚本另有只校验既有结果的 `SPECSLO_VALIDATE_ONLY=1`。候选脚本还增加默认不改变模型布局的 `SPECSLO_TARGET_NZ_MODE=default|5`，用于本轮定向权重布局实验。
2. `vllm_ascend/spec_decode/pearl/native_engine.py` 增加一个**默认关闭实验路径**的在线前沿证明。只有无 staged prefill/悬挂请求，且 batch 已满、无 prefetch-ahead/优先级抢占、当前验证请求忽略 EOS 且即使接受 `gamma+1` 个 token 也不会达到输出上限时，才允许“仍有 pending arrival”不阻止下一轮 target replay。其余情况仍 fail-closed；不能把 pending arrival 条件整体删除。
3. 曾实验性修改 `vllm_ascend/spec_decode/pearl/native_graph.py`，让专属 FIA 更新线程在设备未变化时复用已绑定的 NPU context，避免每次 replay 重复 `torch.npu.set_device`。下述系统实验未证明 E2E 收益，**已经回滚源码和对应测试**；本条仅是已否决的尝试记录。

当前保留的新增/扩展测试在 `tests/ut/spec_decode/test_specslo_benchmark_report.py` 和 `test_pearl_native.py`。线程绑定实验的源码和对应测试均已回滚；最终源码的三组相关回归为 **653 passed、14 warnings**。Ruff、`bash -n`、`git diff --check` 均通过。

## 单次在线预检

以下候选与同一历史 TP4 baseline 作口径一致的探索性比较，**不是本轮成对 A/B，也不能据此称稳定倍率**。Raw/Goodput 均为输出 token/s；baseline 的 Raw 字段是 `output_throughput_tokens_per_second`，候选是 `e2e_throughput_tokens_per_second`，两者均使用各自 measured E2E 时间及 118,528 个输出 token。

| 配置 | Raw | 对 TP4 | Goodput | 对 TP4 | 结论 |
| --- | ---: | ---: | ---: | ---: | --- |
| TP4 baseline | 929.684 | 1.0000× | 377.496 | 1.0000× | 历史单次对照 |
| FIA2 + DraftPA2、replay 关闭 | 924.114 | 0.9940× | 522.933 | 1.3853× | 仍是当前 Raw 较优候选 |
| FIA4 + DraftPA2、replay 关闭 | 923.372 | 0.9932× | 498.581 | 1.3208× | FIA4 不采用 |
| FIA2 + DraftPA2、原保守 replay | 920.958 | 0.9906× | 475.398 | 1.2593× | 负优化 |
| FIA2 + DraftPA2、满 batch 安全证明 replay | 919.849 | 0.9894× | 484.759 | 1.2841× | 覆盖仍太低，负优化 |
| FIA2 + DraftPA2、线程设备绑定、replay 关闭 | 923.255 | 0.9931× | 522.447 | 1.3840× | 单次未证明系统收益 |
| FIA2 + DraftPA2、target NZ5、replay 关闭 | 918.877 | 0.9884× | 462.415 | 1.2250× | gate 通过，但系统负优化；不采用 |

上述候选 gate 均通过：正式测量无 graph fallback，MC2 生产路由及 FIA/DraftPA 并行证据符合各自配置。输出散列会随批次形状和调度轨迹分叉；gate 的四 rank 签名一致**不等于**跨 run bit-exact 数值证明。

结果文件：

- baseline：`/root/data/nano-pearl-benchmark-results/20260927-specslo-drain-ready-age-precheck/runs/baseline-target-only-tp4-cards2-3-4-5-drain-age-paired-r1/result.json`
- FIA2 + DraftPA2：`/root/data/nano-pearl-benchmark-results/20260928-specslo-v135-m1-128-draftpa-system-precheck/runs/candidate-on-v135-m1-128-fia2-draftpa2-r1/result.json`
- FIA4：`/root/data/nano-pearl-benchmark-results/20260928-specslo-resume-targetfia4/runs/candidate-on-fia4-draftpa2-r1/`
- 原保守 replay：`/root/data/nano-pearl-benchmark-results/20260928-specslo-replay-ahead-resume/runs/candidate-on-replay-ahead-precheck-r1/`
- 满 batch 证明 replay：`/root/data/nano-pearl-benchmark-results/20260928-specslo-replay-ahead-resume/runs/candidate-on-replay-ahead-full-batch-proof-r1/`
- FIA 线程设备绑定：`/root/data/nano-pearl-benchmark-results/20260928-specslo-target-fia-thread-binding/runs/candidate-on-fia-thread-bind-r1/`
- target NZ5：`/root/data/nano-pearl-benchmark-results/20260928-specslo-target-nz5-precheck/runs/candidate-on-target-nz5-r1/`

## replay-ahead 为何没有成为在线主阶段收益

保守版四 rank 各尝试 1,879 次，只提交/消费 6 次；拒绝中 `mutable_frontier=1,714`、`pending_eager=144`。加入满 batch 安全证明后各尝试 1,892 次，提交/消费 11 次；`mutable_frontier=1,685`、`pending_eager=175`。两轮均零签名不匹配、零 graph fallback，但前沿证明只多释放 5 次。每 cycle 的跨 rank preflight 成本仍在，无法被这点覆盖收益摊销。

因此不要继续仅放宽布尔 gate。真正的下一步是给 scheduler 引入下一轮 target plan reservation/frontier epoch：在在线到达仍可推进 prefill 的前提下，明确冻结下轮 verify 的请求/票据/KV 身份，并在提交前后进行一致性证明。这是独立的调度协议改造，还需数值回归和同时在线到达的 SLO 验证。

## 四卡 profiling：实际重叠与关键路径

为了区分“host 提交窗口”与“设备真正算完”，另跑了一次同配置 FIA2 + DraftPA2 的诊断测量：记录 700 个 host decode step，并在约第 400 step 同时抓四卡 CANN trace。profiling 会扰动性能；本次 Raw 922.091、Goodput 503.864 **仅作时间分解，不参与吞吐 A/B**。原始文件在 `/root/data/nano-pearl-benchmark-results/20260928-specslo-critical-window-profile/runs/candidate-on-fia2-pa2-host700-npu400/`。

在排除在线 prefill 和 profiler 启停窗口后，414 个同时含 draft、target 的纯 decode cycle：

| 口径 | 平均耗时 | 解释 |
| --- | ---: | --- |
| cycle wall | 46.076 ms | 四 rank 的实际端到端临界窗口 |
| scheduler critical | 2.107 ms | 请求/票据与批次计划 |
| draft host submit | 25.429 ms | host 侧四步 graph 提交窗口，不等同设备执行耗时 |
| target host submit | 13.482 ms | host 侧 target graph 提交窗口，不等同设备执行耗时 |
| host 窗口重叠 | 13.475 ms | 两个提交窗口的交集，不代表 NPU 计算重叠 |
| draft NPU event | 31.435 ms | 设备事件测得的 draft 完成区间 |
| target NPU event | 41.442 ms | 设备事件测得的 target 完成区间 |
| draft→target critical | 0.792 ms | 输送临界段，不等于纯 HCCL 时间 |
| target→draft critical | 8.900 ms | 含等待/D2H，不能全部归因于 HCCL |

这 414 个 cycle 中，**每一个的 target NPU event 都长于 draft NPU event**。因此以前“draft 约 36 ms、target 约 14 ms”的 host 表格不能解释为设备端 draft 比 target 慢；当时 14 ms 是 target 的 host 提交短窗口。

跨卡 `ProfilerStep#402` 是一个无 prefill 的样本：draft 27 个请求、target 20 个请求、80 个验证候选。经四卡时钟校准后，使用官方 CANN `trace_view` 的 Ascend Hardware 区间求交，并保守排除 HCCL-like 内核，draft 卡与三个 target rank 的**实际设备计算重叠**分别为 **16.324/16.109/16.291 ms**，占 draft 计算忙时的 **77.55%/76.53%/77.39%**；仅 MatMul 对 MatMul 的真重叠亦有 **4.236/4.223/4.261 ms**。`device-overlap-gate.json` 为 `passed`。这是单个纯 decode step 的四卡实证，不可外推为所有 cycle 都具有相同重叠比例。

该 step 中 target 三个 rank 的 `MatMulV2` 约 **23.2–23.4 ms**，`AllreduceAddRmsnorm` 约 **8.2–9.2 ms**，FIA 约 **3.7 ms**；独立 HCCL 未重叠时间仅约 **0.04–0.07 ms**。每层四个 MatMul 的设备执行时间从重复顺序推断约为 40、35、180、100 μs；其中约 180 μs 的热点很可能是 FFN gate-up，**算子归属是推断，不是 trace 的直接标签**。这是**设备算子耗时汇总**，不与上述 NPU event 或 cycle wall 直接相加。因此下一轮应优先检查 target 矩阵/融合归一化的实际 model-runner 路由及数值影响，而不是只优化单独 HCCL 调用。

据此在相同线上设置测试了现有 NZ mode5，即只把 target FFN `gate_up_proj` 权重改为 FRACTAL_NZ、保留其余投影与 MC2 down 路由。它完成 463×256 输出且 gate 通过，0 graph fallback，但 Raw **918.877**、Goodput **462.415**，均低于此前 ND 候选的 924.114/522.933。target 阶段总时间 72.083 s 对 72.859 s，看似略短，但 decode round 同时从 1,978 降到 1,906；折算每轮约 37.82 ms 对 36.83 ms，反而变慢约 2.7%。两次输出 hash 和接受轨迹不同，不可把 Goodput 差异全归因于布局。**本轮明确否决把 NZ5 设为默认，也不据此申请正式三轮 A/B。**

独立探索性微基准使用 TP3 gate-up 的本地形状 `weight=[17152,5120]`、BF16、单卡 910B2，均先 warm-up 20 次，再用 NPU event 计 200 次 `F.linear`，结果如下。此测试没有 model-runner 的图、争用和流水线，仅用于检查布局方向；ND/NZ 的输出最大绝对差均为 0。

| 输入行数 M | ND ms/op | NZ ms/op | NZ 相对 ND |
| ---: | ---: | ---: | ---: |
| 20 | 0.05806 | 0.06918 | 慢 19.2% |
| 40 | 0.06257 | 0.07325 | 慢 17.1% |
| 80 | 0.07904 | 0.09555 | 慢 20.9% |
| 128 | 0.09304 | 0.11982 | 慢 28.8% |

四个常见 M 点都指向同一方向，与线上每轮 target 变慢一致；不能因历史上别的模型/shape 的 NZ 收益推断当前 Qwen3-32B TP3 gate-up 也会受益。

同法补测 QKV `[3840,5120]`、attention output `[5120,3072]`、FFN down `[5120,8576]` 在 M=20/40/80/128 的 ND/NZ。QKV 最大差仅约 0.00075 ms/op，后两者也基本持平；未发现值得再启一轮完整在线 NZ 布局实验的可靠单算子收益。尤其 attention output/down 还涉及已资格化的 MC2 路由，不能仅凭持平的单算子结果更换权重格式。

## 同日续测：当前最佳、target 事件粒度与尾部 fallback

在上述记录之后，固定同一 workload、TP1+TP3、gamma4、coalesce1、物理卡
3/4/5/7 和 target 4/5/7，完成了以下增量预检。它们仍是单次候选，不能替代
三轮交错成对 A/B。

当前单次最佳来自 proposal matrix 零拷贝 view、复用 Gloo envelope 内已有的 host
continuation，以及固定线性 acceptance-only 下省略未被调度器消费的 draft confidence：

- Raw **925.362 tok/s**，E2E **128.088 s**；
- 论文口径 Goodput **539.628 tok/s**，达成率 **58.315%**；
- 2,017 个 decode round，acceptance rate **52.478%**，平均每轮提交
  **4.618 token/request**；
- 正式测量 graph shape/capacity fallback 均为 0；三个 target rank 的 MC2
  success 均为 26,624，fallback/exception 均为 0。

证据目录：
`/root/data/nano-pearl-benchmark-results/20260928-specslo-zero-copy-confidence-elision-precheck/runs/candidate-on-zero-copy-confidence-elision-target457-r1/`。

针对 target FIA host task update 又做了事件粒度实验。单算子微基准中，event
group 2/4 的提交窗口确实短于 event1；完整系统却没有转化为 Raw 收益：

| target 事件策略 | Raw tok/s | Goodput tok/s | E2E s | 结论 |
| --- | ---: | ---: | ---: | --- |
| event1、无 prefix（当前保留） | **925.362** | **539.628** | **128.088** | 当前最佳 |
| event4、无 prefix | 922.814 | 508.245 | 128.442 | 否决 |
| event4、prefix1 | 924.109 | 526.922 | 128.262 | 否决 |
| event2、prefix1 | 922.965 | 516.302 | 128.421 | 否决 |

event4 把三个 target rank 累计 host target 阶段降到约 69.9--71.1 s，但 graph
首层必须等一个事件组的 task 全部更新才可启动；节省的 host 记事件时间被转移到
后续 correction/broadcast 等待。prefix1 只部分恢复首层启动，仍不及 event1。
因此生产默认保持 event1/prefix0，不把孤立 microbenchmark 收益当作系统收益。

另外验证了一个默认关闭的最终排空实验：当全部请求已到达且 active<=4 时，draft
跳过不会再使用的 KV forward，并把 target token 从 WORLD HCCL 改为
target-leader/draft-leader Gloo。路径正确触发 46 次，零图回退，但 Raw 仅
**923.202 tok/s**；draft 端 Gloo 阻塞 1.338 s，target leader 的 D2H 等待累计
623 ms，说明原 HCCL 和被隐藏的 draft forward 均非关键路径。该实验源码已回滚，
只保留结果证据：
`/root/data/nano-pearl-benchmark-results/20260928-specslo-gloo-terminal-fallback-precheck/`。

历史 `fallback0` 结果在同类单次运行间为 921.029--923.408 tok/s，fallback3
为 923.160 tok/s，也未超过当前 fallback4 最佳值。后续不再仅凭阈值扫描宣称尾部
收益；先以完全相同物理卡重跑 TP4 baseline，排除 2/3/4/5 与 3/4/5/7 的拓扑差异。

## 9 月 29 日续测：post-correction target replay 已否决并回滚

为验证“先完成本轮 correction，再冻结下一轮真实 scheduler 选择”能否提高 target
replay-ahead 覆盖率，曾加入一个默认关闭的实验路径。它在 controller 副本上执行真实
SLO scheduler，把 Rolling Eager 已确认晋升的 ticket 一并纳入下一轮 target 集合，
并对 request 顺序、proposal/continuation epoch、KV/graph entry、correction epoch 做
跨 rank 两阶段 preflight。前三次运行分别暴露并 fail-closed 于 correction 身份已消费、
replay 与 mixed-prefill 生命周期冲突、priority 顺序与 active-row 顺序不一致；修复后第
四次运行完整通过正确性和图门禁。

第四次运行的事实证据为：四个 rank 均尝试 1,511 次，821 次
`eligible=submitted=consumed=reserved`，688 次因资源契约拒绝、2 次协议拒绝；签名不匹配
为 0，正式测量 graph capture/fallback 为 0，MC2 fallback/exception 为 0。说明该协议
能安全覆盖约 54.3% 的 cycle，而不是旧 replay-ahead 的低覆盖问题。

但完整系统 Raw 仅 **683.793 tok/s**，E2E **173.339 s**，相对同卡 TP4 baseline
**930.005 tok/s** 下降约 **26.47%**。821 次 replay 的 target graph 设备区间累计
36,829.778 ms，约 **44.86 ms/次**；更关键的是三个 target rank 的
`worker_target_replay_ahead_seconds` 为 12.6--13.2 s，即
`_target_full_window_outputs_batch` 的 graph task-update/host submit 平均额外串行暴露约
**16 ms/次**。原本 current-target early submit 会把这段 host 更新与 TP1 draft 重叠；
移动到 correction 后再提交，相当于把它放到串行尾部，提前启动设备计算不足以抵消新增
的 host 临界路径。因此该方向不是“图回退导致负优化”，而是提交位置错误造成的结构性
负优化。

该实验源码、环境开关、launcher 契约和专用单测均已回滚；保留本节和运行结果作为否决
证据，不进入生产配置。证据目录：
`/root/data/nano-pearl-benchmark-results/20260928-specslo-v143-post-correction-replay-system-target457/runs/candidate-on-v143-post-correction-replay-target457-r4/`。
后续若再研究 future-target replay，前提必须是消除或异步隐藏 graph task update，而不
只是继续放宽 scheduler reservation。

## 9 月 29 日续测：FIA 连续分区与小 M MC2 系统门

target FIA 两路 host task update 原先按 event group 轮转分配，导致 layer 0 在
caller、layer 1 在后台线程。后台线程每次 replay 都要恢复 NPU context，图执行很早就
可能在第二层等待。现改为**连续且平衡的 layer/event-group 分区**：caller 更新图前半段，
后台线程更新后半段；每个 handle、begin/end 和 event 均保留，不删除同步或数值门。
对应 CPU 契约回归 11/11 通过。

同一卡组 3/4/5/7、相同大 M v143 MC2 profile 的两次完整在线预检为：

| 运行 | Raw tok/s | 对同卡 TP4 | paper Goodput tok/s | 对同卡 TP4 | E2E s |
| --- | ---: | ---: | ---: | ---: | ---: |
| TP4 baseline | 930.005 | 1.0000× | 385.661 | 1.0000× | 127.449 |
| 连续分区 r1 | 926.533 | 0.9963× | 542.312 | 1.4062× | 127.926 |
| 连续分区 r2 | 924.517 | 0.9941× | 531.148 | 1.3772× | 128.205 |

两轮均完成固定 463×256 输出，正式测量 graph capacity/shape fallback 为 0，三个
target rank 的 MC2 fallback/exception 为 0。连续分区把单次 Raw 推到更接近 baseline，
但第二次未复现第一轮全部增益，因此仍不能宣称 Raw 已超过 TP4，更不能进入三轮正式门。

为检查 M<40 的 MC2 覆盖缺口，新增
`examples/derive_specslo_mc2_row_captures.py`。它只从真实 M40 capture 的 activation、
residual 前缀派生 M4--36，保留原 weight/gamma，并在 manifest 中记录每个输入/输出文件
SHA256；工具拒绝覆盖已有目录。M4/8/.../36 在 layer 0/31/63、31 次重复的严格资格化中
全部通过数值和 p95>1.02 门。大/小 M 合并生产 profile 的 SHA256 为
`68a07364a676c75a99f4cb7be7afdaaec9126fd0c4b78109f58ce9dea150f985`。

但完整线上启用 M4--128 后，三个 target rank 的 fused success 从各 25,472 增至
36,224，仍为零 fallback/exception，Raw 却为 **924.205 tok/s**，E2E
**128.249 s**，paper Goodput **534.961 tok/s**；相对同卡 TP4 分别为
**0.9938×/1.3871×**。target rank 累计 target 窗口也由约 73.15--73.48 s 增至
73.60--74.39 s。说明孤立 captured-operator 的小 M 收益没有保持完整 graph 内的执行
重叠/排队关系；该 profile 只保留为资格化证据，**系统候选继续使用 M>=40 profile**，
不把算子微基准冒充系统优化。

证据：

- 连续分区 r1：`/root/data/nano-pearl-benchmark-results/20260929-specslo-fia-contiguous-partition-precheck/`；
- 连续分区 r2：`/root/data/nano-pearl-benchmark-results/20260929-specslo-fia-contiguous-partition-repeat/`；
- 小 M 严格资格化：`/root/data/nano-pearl-benchmark-results/20260929-specslo-mc2-v143-small-m-qualification-s31/`；
- M4--128 profile：`/root/data/nano-pearl-benchmark-results/20260929-specslo-mc2-v143-full-m-profile/`；
- M4--128 在线门：`/root/data/nano-pearl-benchmark-results/20260929-specslo-fia-contiguous-full-mc2-precheck/`。

## 当前判断与后续次序

当前**未达到** Raw > TP4；单次 Goodput 最佳值虽超过 1.3×，不能掩盖 Raw 缺口。FIA4、两种 replay-ahead、线程设备绑定及 target NZ5 都未在完整系统上形成可确认正收益，不改为生产默认。

后续先针对上述矩阵/融合归一化热点选能落到完整 model-runner 的改动；任何候选先过数值/图门禁与单次 Raw/Goodput 双门槛，最后做三轮成对 A/B。期间不运行占卡程序，实验结束释放 NPU。

## 9 月 29 日续测：在线预取与 mixed early-submit 系统门

为判断 TP1+TP3 比同卡 TP4 慢约 0.4%--0.6% 是否主要来自尾部在线 admission，先测试了
`prefetch-ahead`：active B64 已满时，最多为 4 个已经到达、尚未发布的请求预留私有 KV
页并执行 prefill；只有真实空槽出现后才发布 first token。该机制实际预取 8 批、16 个请求，
发生 28 次延迟发布和 8 次部分发布，证明路径真实命中；但 Raw 为 **924.412 tok/s**、
paper Goodput 为 **525.098 tok/s**、达成率 **56.803%**。相对无预取连续分区 r1 的
926.533/542.312，Raw 下降约 0.23%，Goodput 下降约 3.17%。提前占用 mixed target
预算伤害 tight TPOT，故默认关闭。

随后把原本只覆盖 decode-only graph 的 current-target early submit 扩展到固定
mixed verify+prefill graph。它不增加 target forward，只把当前已选定的 verification 和
已到达 staged prompt 的共同固定图移到 next-draft 调度之前提交；请求仍等真实空槽才发布。
完整在线门中三个 target rank 各有 **596/596** 个 mixed cycle 提前提交，固定
463×256 输出完成，ACLGraph shape/capacity fallback、MC2 fallback/exception 均为 0。
但 Raw 仅 **924.462 tok/s**，paper Goodput **535.109 tok/s**，分别为同卡 TP4 的
**0.9940×/1.3875×**，没有形成 Raw 收益。因此实现保留在独立、默认关闭的
`VLLM_ASCEND_SPECRHYTHM_EARLY_CURRENT_MIXED_TARGET_SUBMIT` 开关后，不与已有
decode-only early-submit 绑定。

四 rank 的 staged-prefill 本地计时不能直接相加：draft rank 累计约 43.45 s，三个
target rank 各约 2.49--2.56 s。rank 间异步推进使 draft prompt prefill 已经与 target
mixed graph 大量重叠；系统可见尾部接近 target rank 的等待，而不是 43 s 全部串行暴露。
历史 side-stream draft-prefill A/B 也只从 920.251 变为 919.878 tok/s，故不重复启用。

证据：

- prefetch-ahead：`/root/data/nano-pearl-benchmark-results/20260929-specslo-prefetch-ahead-precheck/`；
- mixed early-submit：`/root/data/nano-pearl-benchmark-results/20260929-specslo-early-mixed-target-precheck/`。

下一轮给每个请求写出 arrival wait、first-token 后 decode 起点和完成时刻，以区分最后约
0.6% Raw 缺口来自 admission/TTFT 还是 target decode；在证据出来前不再凭累计 rank-local
阶段时间推断关键路径。

## 9 月 29 日续测：逐请求尾部诊断与 target-only drain LRTF

连续分区第三次运行在结果行中新增了 `arrival_wait_ms`、首 token 后 decode 起点、完成时刻
和 `request_service_e2e_ms`。本轮 Raw 为 **924.147 tok/s**、E2E **128.257 s**、paper
Goodput 为 **530.935 tok/s**、达成率 **57.451%**；相对同卡 TP4 baseline 分别为
**0.9937×** 和 **1.3767×**。正式测量 graph shape/capacity fallback、MC2
fallback/exception 均为 0，四 rank 双 batch overlap submission 均为 1,954 次，门禁通过。

逐请求数据否定了“最后一个 tight 请求因 admission 等待拖慢 Raw”的假设。最后一个请求
`000462` 的首 token 等待只有 **179.76 ms**，首 token 后 decode 为 **6,474.40 ms**，
从到达到完成合计 **6,654.16 ms**，其 paper TPOT 为 **25.29 ms**。真正的 makespan
straggler 是晚到的两个 loose 请求：`000458` 和 `000461` 的归一化完成点分别约为
**127.608 s** 和 **127.549 s**；同卡 TP4 的最后完成点约为 **126.978 s**。因此当前 Raw
缺口主要是 SLO 优先级留下的 final-drain 剩余服务不均衡，而不是在线 prefill 或 tight
admission。

证据目录：
`/root/data/nano-pearl-benchmark-results/20260929-specslo-fia-contiguous-arrival-timing-repeat/`。

现有统一 drain-LRTF 会同时跨 home 改写 target 选择和 normal/repair draft 选择；历史系统
预检 Raw 约 922.8--922.9 tok/s，未形成收益。为隔离该副作用，本轮新增一个默认关闭的
`VLLM_ASCEND_SPECRHYTHM_DRAIN_TARGET_LRTF`：仅在 pending/staged/suspended 均为空的
immutable drain 边界，按平滑后的最长剩余服务排序受固定 32-row 上限约束的 ready target
ticket；normal/repair draft 仍保持原有 opposite-home 轮转。launcher 明确禁止它与统一
drain-LRTF 同时启用，并要求四 rank 都报告 target 排序命中、且 normal/cross-home 计数为
零。相关 683 项单测已通过；完整在线预检结果尚待写入。

完整预检表明上述 final-drain target-only LRTF **没有实际调度覆盖**：四 rank 均进入
171 个 drain cycle，但受限 target 排序命中为 0；此时 ready target 从未超过固定 32-row
上限，所以改变排序不可能改变 plan。门禁据此判定失败。该轮 Raw **923.909 tok/s**、
paper Goodput **540.776 tok/s**，不能归因于该开关，也不采用。结果目录：
`/root/data/nano-pearl-benchmark-results/20260929-specslo-drain-target-lrtf-precheck/`。

更早的在线阶段才有真实 target 竞争：第三轮诊断中 tight ready/selected/deferred 为
25,074/25,074/0，normal 为 7,852/7,848/4，loose 为 10,752/10,176/576。下一候选因此
实现 `VLLM_ASCEND_SPECRHYTHM_TARGET_RESIDUAL_LRTF`：受限 target plan 中，正
`a_need` 请求仍按论文第 4.3 节优先；只有 `a_need=0` 的第二阶段 residual 候选按平滑
最长剩余服务排序。它不改 normal draft home、固定 gamma/B、图或 MC2 shape。门禁要求
四 rank 均有 residual candidates、selected 和 deferred 的真实计数，避免再次把未改变
plan 的开关当作优化。相关 727 项回归（含定向集成测试）通过。

首版全局 residual-LRTF 的完整在线门通过，四 rank 均在 149 个受限 cycle 中观察到
2,433 个 residual 候选、1,877 次选中和 556 次延后，证明并非空开关；graph
shape/capacity fallback 与 MC2 fallback/exception 仍全为 0。但 Raw 仅
**923.367 tok/s**，paper Goodput **520.516 tok/s**，达成率 **56.371%**。与无该策略的
第三轮诊断相比，tight target 从零延后变为 62 次，tight 达成请求由 76 降为 69；最后两个
loose straggler 的完成点也从约 127.608/127.549 s 变为 127.714/127.655 s。因此“所有
`a_need=0` 请求跨类别做 LRTF”被否决，不作为生产默认。证据目录：
`/root/data/nano-pearl-benchmark-results/20260929-specslo-target-residual-lrtf-precheck/`。

下一版收窄为**类别保持 residual-LRTF**：正 `a_need` 仍全局优先；零缺口请求先维持
tight > normal > loose 的 SLO 类边界，只在同类内部按剩余服务排序。这样仅针对已确认的
loose 尾部失衡，不允许长 loose 请求消耗 tight 的领先余量。定向 controller 与线性集成
回归均通过，完整在线门单独留证，不沿用首版结果。

类别保持版的完整在线门得到 Raw **925.983 tok/s**、paper Goodput
**529.990 tok/s（1.374× TP4）**、达成率 **57.235%**；四 rank graph/MC2 fallback 仍为
0。tight ready/selected/deferred 恢复为 25,087/25,087/0，两个 loose 尾部完成点由无策略
第三轮的约 127.608/127.549 s 前移到 127.366/127.336 s，证明尾部均衡方向有效；但 Raw
仍只有 TP4 的 **0.9957×**，不能验收。证据目录：
`/root/data/nano-pearl-benchmark-results/20260929-specslo-target-residual-class-lrtf-precheck/`。

下一候选进一步采用**tight-only residual guard**：正 `a_need` 仍全局优先；零缺口 tight
继续受保护，normal 与 loose 共同按最长剩余服务排序。类别保持版尚有 93/96 个 normal
请求达标，按固定 256-token 口径，即便少量 normal 达成换取 loose makespan 前移，仍有
Goodput 1.3×余量。同时将 controller 中重复的 urgency/`a_need` 计算改为每 plan 缓存，并
把 remaining-service 全局速率改为首次实际排序时才计算。64-row 受限 plan 的 CPU 微基准
由约 **0.850 ms/次降至 0.686 ms/次**；该改动不改变 legacy 排序语义。

另外，线上约 93% target cycle 需要将跨 compact envelope 的 proposal row 重新组成
`[B,4]`。单卡微基准显示 `torch.stack(out=固定缓冲)` 为约 **0.0342 ms/次**，原每轮分配
为 **0.0444 ms/次**。现为非连续行复用一个 cycle-local 固定输出缓冲；当前 cycle 的
correction 完成后才允许下一轮覆盖，replay-ahead 的并发 next-cycle proposal 不共享该
缓冲。相关调度、transport、native engine 与报告门禁共 **858 项回归通过**。该收益量级
很小，不能单独解释系统目标，必须由完整在线门验证组合效果。

tight-only residual guard 的完整在线门随后得到 Raw **923.210 tok/s**、E2E
**128.387 s**、paper Goodput **504.475 tok/s（1.3088× TP4）**、达成率
**54.644%**。四 rank 均有 207 个 residual 排序 cycle、3,159 个候选、2,363 次选中和
796 次延后，graph/MC2 fallback 仍为 0；但 normal target 被延后 337 次，tight 达成请求
从类别保持版的 76 降到 63，最后两个 loose 完成点也退到约 127.247/127.191 s。该实验
既未改善 Raw，也几乎耗尽 Goodput 门槛余量，故否决 normal/loose 跨类竞争，源码恢复为
类别保持 residual-LRTF；保留 scheduler 计算缓存和 proposal 固定输出缓冲这两个不改变
排序语义的控制面优化。证据目录：
`/root/data/nano-pearl-benchmark-results/20260929-specslo-target-residual-tight-guard-precheck/`。

## 9 月 29 日续测：关键路径重新标定与 accounting 快路径

在类别保持 residual-LRTF 源码上运行 500-cycle host timeline（关闭 CANN profiler，避免
device profiler 改写吞吐）后，369 个纯 decode cycle 的均值如下：cycle wall
**43.329 ms**、TP1 draft device event **29.643 ms**、TP3 target device event
**39.889 ms**。因此 draft 已低于 target；早期报告中“draft 约 36 ms、target 约 14 ms”
把异步 graph host submit、延迟 event 读取和并行窗口当成了可相加阶段，不能用于判断设备
关键路径。当前纯 decode 已接近 target-device 临界：draft/target host window 的重叠约
**12.637 ms**，scheduler 的表观 **15.002 ms** 中嵌入了约 **12.734 ms** 的 early
current-target host submit；扣除后独占 scheduler/control 约 **2.269 ms**。correction
附近约 **8.711 ms** 的 host wait 主要是在等待已提前提交的 target device 尾部，并不是
能从 39.889 ms 再减去的独立串行开销。

固定形状 accounting envelope 原先虽复用了目标 tensor，仍每轮创建
`torch.as_tensor(values)` 并在 `tolist()` 后逐项再次调用 `float`。B64 主 envelope 的
CPU 微基准分别由约 **31.312 us 降至 8.355 us**、**11.263 us 降至 4.152 us**：实现改为
持久 host tensor 的 zero-copy NumPy view 全量覆写，并直接返回 float64 tensor 的
`tolist()`。157 项定向回归通过。但完整在线门只有 Raw **923.954 tok/s**、paper
Goodput **518.851 tok/s（1.345× TP4）**、E2E **128.283 s**；graph fallback 和 MC2
fallback/exception 均为 0。该改动语义正确且减少分配，但收益量级不足以越过系统噪声，
不把它单独计作系统性能跃迁。

证据：

- host-only timeline：`/root/data/nano-pearl-benchmark-results/20260929-specslo-class-residual-host500/`；
- accounting 完整在线门：`/root/data/nano-pearl-benchmark-results/20260929-specslo-accounting-zero-copy-precheck/`。

逐请求完成点再次显示最后两个 makespan straggler 是 late loose 请求，而不是 tight
admission。下一候选不再对整个 immutable drain 启用统一 LRTF，而增加正整数 active-row
门：在线主体及 drain 前段保持论文 SLO 调度，只在请求全部到达且 active row 不超过门值
时允许跨-home remaining-service 均衡。`0` 保留旧的全-drain 实验语义；正式候选先测
`active<=16`，并由每-rank executed/skipped counter 证明开关真实命中。

## 9 月 29 日续测：尾部均衡、结果诊断传输和 FIA3 否决

`active<=16` 的 late-drain LRTF 完整在线预检得到 Raw **922.941 tok/s**、paper
Goodput **534.229 tok/s**、达成率 **57.883%**。四 rank 均实际执行 52 个 late-drain
cycle、跳过 112 个 cycle，但排序命中只发生在 normal/repair 路径，target/unified 命中为
0；它没有缩短 makespan，也没有越过 Raw 门槛，故否决。证据目录：
`/root/data/nano-pearl-benchmark-results/20260929-specslo-late-drain-lrtf16-precheck/`。

benchmark 结果回传原先为每个请求复制同一份大型运行诊断字典。现改为 463 个结果行共享同一
个只读快照，利用 pickle memoization 将合成回传载荷由约 **1.48 MB** 降到约 **249 KB**。
完整在线门 Raw **921.889 tok/s**、paper Goodput **509.727 tok/s**；E2E 与 inference-only
计时差由此前约 0.26 s 降至 **0.228 s**，只减少约 30 ms，证明回传优化正确但不是模型
执行瓶颈。证据目录：
`/root/data/nano-pearl-benchmark-results/20260929-specslo-shared-result-diagnostics-noresidual-precheck/`。

为验证 target 图的 host task-update 是否可由 3 个 worker 加速，target FIA 支持扩展为
1/2/3/4 worker（draft 仍只允许 1/2/4），并加入 launcher、报告门禁和单测。FIA3 正式
测量执行 1,934 次并行 replay、123,776 个 task，graph capture/replay 失败、shape/capacity
fallback 均为 0；但 Raw 只有 **920.755 tok/s**，paper Goodput **526.998 tok/s**、达成率
**57.235%**。3 线程管理开销高于 FIA2，故生产默认仍为 FIA2。证据目录：
`/root/data/nano-pearl-benchmark-results/20260929-specslo-target-fia3-precheck/`。

## 9 月 29 日续测：标准 gamma4 bonus-token 实验及结构性否决

为区分“4 个 draft token”与“标准 speculative decoding 最多提交 4+1 token”，新增默认关闭的
`SPECSLO_LINEAR_BONUS_TOKEN` 实验开关，并把 eligible、committed、eager/output-limit/KV
抑制原因纳入四 rank 正式测量门禁。bonus 使 target query width 从 4 变为 5；因此 mixed
graph 最大 token 容量必须由 `32*4+512+4+1=645` 提升为
`32*5+512+4+1=677`。首轮准确暴露每 rank 195 次 capacity fallback，修正容量公式后，
第二轮正式测量 graph capture/replay、shape/capacity fallback 和 MC2 fallback/exception
全部为 0。

容量修正后的结果为 Raw **904.804 tok/s**、paper Goodput **383.027 tok/s**、达成率
**42.333%**、总 target round **1,698**；相对无 bonus 的约 2,016 round 确实减少，但仍是
明显负优化。每 rank 有 16,959 个 bonus eligible、4,322 个 bonus commit，同时有
23,816 次因 rolling-eager staged/pending continuation 被抑制；DraftPA2 还因不再是严格的
gamma4 full-chain shape 发生 1,652 次 `not_gamma4_full_chain` fallback。更重要的是，bonus
verdict 决定下一 token，当前实现不能在 verdict 前安全提交 next-draft，所以 current-target
early-submit 路径明确不适用于 bonus。结果等价于用 **+25% target query rows** 换取较少 round，
却丢失 full-chain PA 与关键 draft/target overlap。

因此 bonus-token 代码只作为显式实验能力保留，默认关闭；生产固定 gamma4 继续采用一次
draft 四个连续 proposal、一次 target 验证四行、最多提交四 token 的 full-chain 协议。
若未来重做 bonus，必须先设计能携带 bonus frontier 的 rolling-eager continuation，不能仅靠
扩大 graph 容量。证据：

- 首轮容量门失败：`/root/data/nano-pearl-benchmark-results/20260929-specslo-linear-bonus-token-precheck/`；
- 容量 677、零图回退正式结果：`/root/data/nano-pearl-benchmark-results/20260929-specslo-linear-bonus-token-capacity677-precheck/`。

当前生产候选仍为 FIA2、linear bonus 关闭、M>=40 的 v143 MC2 profile。最新两条实验同时
说明：继续增加 host worker 或机械增加单轮 token 数都不能补齐 Raw；下一步应以约
**39.889 ms** 的 TP3 target device 窗口和约 **2.269 ms** 的独占 scheduler/control 为
优化对象，并在真实控制路径 profiling 后再决定是否启动完整在线回归。

## 9 月 29 日续测：混合大 M MC2、行 padding 与大 M NZ 双布局

根据真实 B32 mixed-target graph 的 replay 直方图，补采了 M=133/165/197/229/293/389/645
在 attention/down、layer 0/31/63、三个 target rank 上的输入。原 v143 内核重新做 31 样本
保守跨层门后，只有 down M165/M197 同时通过数值和 P95 门。它们与原 M40--128 profile
合并后完成了一轮 463 请求线上预检：Raw **922.906 tok/s（0.9924x TP4）**，paper
Goodput **548.162 tok/s（1.4214x TP4）**，正式测量 graph shape/capacity fallback、MC2
fallback/exception 均为 0。新加入的 M165/M197 每 rank 只命中 34/2038 个 replay，覆盖率
不足以补齐 Raw。证据目录：
`/root/data/nano-pearl-benchmark-results/20260929-specslo-mc2-v143-mixed-b32-system-precheck/`。

为判断大 M 是否适合改写 rank-3 MC2 kernel，又在真实 layer-31 输入上测试了完整
MatMul+AllReduce+AddRMSNorm 融合版本。M293 的 fused P50 为 **0.25353 ms**，split 为
**0.21985 ms**，只有 **0.867x**；M645 fused 也慢约 24%，且 added 输出的 scaled error
超过严格门。该 C++ 实验已完整回滚，源码哈希恢复为 v143；不能进入系统门。

另对真实 mixed M 做了 FFN TP3 图微基准。把 token rows padding 到 4 的倍数在所有测点都
慢约 5%--10%，且改变部分 BF16 输出，故明确否决。相反，大 M 下同时使用 gate-up/down NZ
比 ND 的完整 FFN 最慢 rank 快约 8%--20%，但小 M 的历史证据显示 gate-up NZ 慢 17%--29%。
因此不能把全模型永久切到 NZ，只能使用按真实行数选择的双布局。

先测试仅 down-proj 的双布局。全 64 层 mode11 在图资格化尾段达到约 59.3/61.0 GiB，最终
连 2 MiB 临时张量都无法分配，明确 OOM；偶数 32 层 mode12 可运行，阈值设为 161，使
M<=128 保留原 MC2、M165/M197 保留已资格化 ND-MC2，其余大 M 才用 NZ。完整线上结果为：

| 配置 | Raw tok/s | 对 TP4 | paper Goodput tok/s | 对 TP4 | 图/MC2 fallback |
| --- | ---: | ---: | ---: | ---: | ---: |
| 同卡 TP4 baseline | 930.005 | 1.0000x | 385.661 | 1.0000x | - |
| 无 NZ、混合大 M profile | 922.906 | 0.9924x | 548.162 | 1.4214x | 0/0 |
| mode12，偶数层 down 双布局 | 924.592 | 0.9942x | 531.191 | 1.3774x | 0/0 |

mode12 相对本轮无 NZ 控制的 Raw 仅提升 **0.18%**，仍低于 TP4 约 0.58%；Goodput 的单次
变化受 40 ms 边界请求和生成轨迹影响，不能归因给 NZ。它证明双布局可行，但收益不足。
证据目录：
`/root/data/nano-pearl-benchmark-results/20260929-specslo-dual-down-nz161-even-system-precheck/`。

下一候选为 mode14：只在每八层中的一层保存 gate-up 与 down 的大 M NZ 副本，共 8/64
层，额外 HBM 约 2.1 GiB；小 M 与其余层保持 ND。该路径必须在当前源码下重新做 MC2
设备资格化，不能手工替换 profile source hash。实现和完整线上结果将在本节后续补录。

## 9 月 29 日续测：v149 自适应 MC2 与系统收益重新标定

v149 对 TP3 `AllReduce+AddRMSNorm` 做了两项设备侧改动：

1. tiler 不再对所有 M 固定发射 24 个 AIV。M<=128 或 M>=229 时限制为 16 个 AIV，
   M=129--228 仍使用 24 个，减少每个 AIV 两次远端 mailbox poll 的 HCCS 控制面竞争；
2. TP3 专用 epilogue 不再逐行执行 vector-to-scalar、scalar-to-vector 同步。FP32 reciprocal
   用 `Brcb` 扩展到一个 8-element block，再以 zero-stride `Mul` 重用该 block 完成整行缩放；
   reciprocal 和乘法的数值边界不变，并继续通过跨层严格数值门。

当前生产 profile 为：
`/root/data/nano-pearl-benchmark-results/20260929-specslo-mc2-v149-adaptive-production-target457/qwen3-32b-tp3-v149-adaptive-production.json`。
在 FIA2、DraftPA2、early current-target submit、NZ14、完整 mixed 图族下，单轮完整在线门为
Raw **925.872 tok/s**、paper Goodput **531.926 tok/s（1.379x TP4）**、达成率
**57.451%**；图 shape/capacity fallback 和 MC2 fallback/exception 都为 0。它仍未越过
同卡 TP4 Raw **930.005 tok/s**，所以该结果是新的算子/系统候选，不是最终验收结果。

为了避免把 host submit 区间误判为 scheduler 计算，timeline 新增了
`scheduler_policy_start_seconds`、`scheduler_policy_budget_critical_ms` 和
`scheduler_pre_policy_critical_ms`。重新标定后，真正的 policy/budget 临界均值约
**0.207 ms/cycle**；旧的 `scheduler_budget_critical_ms` 约 12.5 ms，是因为其中包含了
提前提交的 target graph host 窗口。由此否决“继续微调 Python 排序即可补齐 Raw”的判断。

证据：

- v149 系统门：`/root/data/nano-pearl-benchmark-results/20260929-specslo-v149-adaptive-system-precheck/`；
- 当前源码 host/CANN profile：`/root/data/nano-pearl-benchmark-results/20260929-specslo-v149-fia2-host700-npu431-profile/`。

## 9 月 29 日续测：双 batch 的真实设备 overlap 证明

四 rank 的 CANN trace 已按同一个 host step 对齐，并排除前 16 个在线 prefill cycle。
在 433/434 两个纯 decode step 中，三个 target rank 的 target 窗口被 draft 设备计算覆盖
**77.51%/77.30%/77.08%**；target busy 约 61.5 ms（两个 step），同时可见三个 target NPU
上的 `MatMulV2` 与 draft NPU 的 full-chain graph kernel。也就是说当前双 batch 并非仅有
host 上“同时提交”的计数，而是已经形成真实跨 NPU device overlap。

关闭 MC2 的同源码对照也通过设备 overlap 门：纯 decode step 434 中三个 target rank 的
覆盖率为 **76.39%/76.52%/76.05%**。两边 graph fallback 都为 0，因此 MC2 系统差异不能
用图回退或“draft/target 实际串行”解释。门禁文件分别为：

- MC2 on：`/root/data/nano-pearl-benchmark-results/20260929-specslo-v149-fia2-host700-npu431-profile/device-overlap-gate.json`；
- MC2 off：`/root/data/nano-pearl-benchmark-results/20260929-specslo-v149-fia2-host700-npu431-mc2off-profile/device-overlap-gate.json`。

## 9 月 29 日续测：为什么 MC2 算子收益没有等比例转成 E2E

同源码、同在线配置的单轮结果为：

| 配置 | Raw tok/s | paper Goodput tok/s | 达成率 |
| --- | ---: | ---: | ---: |
| v149 MC2 on | 924.870 | 521.363 | 56.371% |
| MC2 off | 924.208 | 501.029 | 54.212% |

Raw 单轮差只有 **+0.0716%**。CANN trace 中，MC2-on 的纯 decode M80/M92 target model
span 约 39.63/41.52 ms；MC2-off M84 约 45.39 ms。但这个差值不能直接外推 E2E：profiler
对 off 路径的 128 个独立 HCCL 做了逐项 instrumentation，而生产 graph replay 不承担同样
的 trace 扰动。更可靠的 700-cycle host event 按相同 M 分桶后，M64--96 的 target event
收益只有约 **1.8%--3.2%（0.75--1.39 ms/step）**。

此外，本轮 MC2 对真实 replay-row 的覆盖率只有约 **64.3%**。小 M 和 mixed-prefill 大 M
仍走 split 路径，而此前真实 M293/M645 已证明 fused 比 split 慢，不能为了覆盖率强行启用。
因此 v149 保留为资格化候选，但“微基准更快”等价于“系统必然同比变快”的结论被否决；
系统验收仍以完整在线 Raw/Goodput 的重复 A/B 为准。

## 9 月 29 日续测：target 输入 embedding 复制实验

新增默认关闭的 TP3 target 输入 embedding 复制路径：模型加载后用一次
`all_gather_into_tensor` 形成本 rank 的完整 BF16 vocabulary table，target forward 直接
本地 gather；原 sharded embedding 参数继续保留给设备定位，LM head ownership 不变。
三个 target rank 各增加约 1.56 GB HBM。完整 8-bucket mixed 图族在资格化尾段差 90 MiB
发生 OOM；allocator `max_split_size_mb:256` 无法解决。把 stable target request bucket 改为
2 虽可装入图，但 rows=31 padding 到 capacity=32 时 greedy token 与精确 unpadded eager
不一致，正确性门主动拒绝，不能用该结果。

随后只把 mixed-prefill token bucket 从 8 个压到 4 个，保留 decode 的精确 request bucket=1，
完成了严格的 replica on/off 单变量 A/B：

| mixed4 配置 | Raw tok/s | E2E s | paper Goodput tok/s | 达成率 | 图/MC2 fallback |
| --- | ---: | ---: | ---: | ---: | ---: |
| embedding replica on | 925.166 | 128.115 | 515.536 | 55.724% | 0/0 |
| embedding replica off | 924.361 | 128.227 | 505.104 | 54.644% | 0/0 |

复制路径的 Raw 观测收益为 **+0.806 tok/s（+0.087%）**，E2E 缩短约 111.65 ms。方向为正，
但远小于 CANN trace 中单独 embedding all-reduce 的表观耗时，说明该通信大部分已被现有
draft/target overlap 隐藏。Goodput/达成率同时受 40 ms tight 边界和在线调度轨迹分叉，
不能作为该单算子的独立收益。该功能继续默认关闭，不凭一次小幅 A/B 晋升生产默认。

证据目录：
`/root/data/nano-pearl-benchmark-results/20260929-specslo-v149-replicated-target-embedding-precheck/`。

## 9 月 29 日续测：immutable-drain 免费 padding repair 与否决实验

逐请求完成点确认当前 v149 的最后 straggler 是晚到的 loose 请求 455/458/461。为不扩大
gamma、B 或 ACLGraph bucket，新增默认关闭的
`VLLM_ASCEND_SPECRHYTHM_DRAIN_PADDING_REPAIR`：只在 pending admission、staged prefill 和
suspended request 全部为空的 immutable drain 中，把**已经选中 serial-draft 图的 padding
行**用于 opposite-home 非 ready 请求的 proposal repair。ready/staged 行不会被覆盖，候选按
剩余输出 token 排序，且绝不打开新的图 bucket。

同卡 TP4 baseline 为 Raw **930.005 tok/s**、paper Goodput **385.661 tok/s**。v149 加入
padding repair 的完整在线结果为 Raw **926.956 tok/s（0.9967x）**、paper Goodput
**560.578 tok/s（1.4535x）**、达成率 **60.475%**；四 rank 各实际命中 64 个 cycle、观察
189 行候选、准入 93 行 repair，图 shape/capacity fallback 和 MC2 fallback/exception 均为
0。相对无 repair 的 v149，Raw 增加 **1.084 tok/s**，Goodput 增加 **28.652 tok/s**；但 Raw
仍低于 TP4 约 **0.328%**，因此它是当前最优候选而非最终双门槛验收结果。launcher 已加入
`SPECSLO_DRAIN_PADDING_REPAIR` 的布尔校验、环境导出、运行契约和每-rank 实际准入门禁。

以下增量均已做完整 463 请求在线测试并否决，源码恢复到上述最长剩余服务 padding 版本：

| 实验 | Raw tok/s | paper Goodput tok/s | 达成率 | 否决原因 |
| --- | ---: | ---: | ---: | --- |
| padding repair + 可选 eager 行交换 | 925.536 | 497.750 | 53.780% | 103 cycle 交换 310 行，破坏 rolling-eager 连续性，平均接受 token 下降 |
| padding repair + 类别保持 target residual-LRTF | 926.235 | 532.135 | 57.451% | 两个非重叠策略不能叠加，tight 达成数下降 |
| 线性 W 跨 draft bucket repair | 924.627 | 523.223 | 56.587% | 线性估计过度准入 847 行、150 次跨 bucket |
| bucket 实测 W 跨 draft bucket repair | 925.402 | 519.664 | 56.156% | 已压到 131 行、30 次跨 bucket，图阶梯成本仍高于收益 |
| 最新请求优先的免费 padding repair | 925.959 | 515.977 | 55.724% | 不增加工作但降低接受率/tight 达成率，劣于剩余服务排序 |

对应证据目录依次为：

- `/root/data/nano-pearl-benchmark-results/20260929-specslo-v149-drain-padding-repair-precheck/`；
- `/root/data/nano-pearl-benchmark-results/20260929-specslo-v149-drain-balance-precheck/`；
- `/root/data/nano-pearl-benchmark-results/20260929-specslo-v149-class-lrtf-padding-precheck/`；
- `/root/data/nano-pearl-benchmark-results/20260929-specslo-v149-drain-window-padding-precheck/`；
- `/root/data/nano-pearl-benchmark-results/20260929-specslo-v149-drain-window-bucket-timing-precheck/`；
- `/root/data/nano-pearl-benchmark-results/20260929-specslo-v149-padding-latest-precheck/`。

这组实验把下一步范围进一步收窄：不能再增加 draft 行或交换 rolling-eager 行；需要在保持
padding repair 的条件下缩短约 **0.4 s** 的 target 服务关键路径。候选 inference 为
**127.092 s / 932.614 tok/s**，TP4 `online_timing.service_step_seconds` 为 **126.665 s**；二者
服务段仍相差约 **0.427 s**，不是单纯的 benchmark detokenize/IPC 统计差异。后续应继续优化
真实命中的 TP3 target graph/collective，并在越过 Raw 后做三轮交错稳定 A/B。
