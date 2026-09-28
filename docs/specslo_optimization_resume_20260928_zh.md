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

## 当前判断与后续次序

当前**未达到** Raw > TP4；单次 Goodput 最佳值虽超过 1.3×，不能掩盖 Raw 缺口。FIA4、两种 replay-ahead、线程设备绑定及 target NZ5 都未在完整系统上形成可确认正收益，不改为生产默认。

后续先针对上述矩阵/融合归一化热点选能落到完整 model-runner 的改动；任何候选先过数值/图门禁与单次 Raw/Goodput 双门槛，最后做三轮成对 A/B。期间不运行占卡程序，实验结束释放 NPU。
