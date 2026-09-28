# SpecSLO 链式投机解码优化暂停检查点（2026-09-28）

本文档是插件更新前的可恢复检查点。自本文档生成起，暂停新的 NPU
实验、性能改动和占卡程序；恢复工作时应先核对代码、插件、CANN 和模型环境，
再继续实机验证。

## 1. 当前结论

当前工作分支为 `SpecSLO-Chain`，目标口径为：

- Qwen3-0.6B draft TP1 + Qwen3-32B target TP3；
- 对照为 Qwen3-32B target-only TP4；
- 固定 `gamma=4`、串行链式 draft、在线 Poisson RPS=4；
- 463 个请求，batch 上限 64，每请求最多生成 256 token；
- tight/normal/loose 数量为 271/96/96，TPOT SLO 为 40/50/150 ms；
- prefill 按在线到达插入，双方均使用 `coalesce1`，不预先保存全部 KV；
- warm-up 不计入正式计时，但正式区间必须无新 graph capture、无 graph fallback；
- 每轮同时满足 Raw 吞吐高于 TP4 baseline，论文口径 Goodput 至少为
  baseline 的 1.3 倍，最后再做三轮成对 A/B。

当前最新的单次预检状态是：

| 项目 | TP4 baseline | 当前 Raw 最佳候选 | 当前 GP 最佳候选 |
| --- | ---: | ---: | ---: |
| 组合 | target-only | FIA2 + DraftPA2 | FIA2 caller-primary + DraftPA2 |
| E2E 时间 | 127.492844 s | 128.261286 s | 128.335318 s |
| Raw 吞吐 | 929.683552 tok/s | 924.113610 tok/s（0.994009x） | 923.580520 tok/s（0.993435x） |
| 论文 Goodput | 377.495697 tok/s | 522.932540 tok/s（1.385268x） | 526.620426 tok/s（1.395037x） |
| SLO 达成率 | 40.60%（188/463） | 56.59%（262/463） | 57.02%（264/463） |

所以当前不能宣称完成最终目标：Goodput 的单次预检已经超过 1.3x，
但 Raw 仍比 baseline 低约 0.60%，并且尚未完成三轮成对稳定复测。

旧工作记录中 1.5681x Goodput、Raw 1.0048x 的结果使用的是不同的历史组合，
其中 baseline 为 coalesce4。它是有效历史结果，但不能替代当前双方 coalesce1、
在线 prefill 公平口径的验收。

## 2. 固定工作负载和证据

正式 workload：

`/root/data/specslo-workloads/eurosys27-rps4-seed0-author-slo40-50-150/rps4.0_mix0.6_0.2_0.2_seed0.jsonl`

manifest SHA256 以结果文件中的 `request_manifest_sha256` 为准，当前固定值以
`dc58` 开头、`662` 结尾。恢复时必须同时校验完整 SHA256，不能只比较文件名。

关键结果：

- baseline：
  `/root/data/nano-pearl-benchmark-results/20260927-specslo-drain-ready-age-precheck/runs/baseline-target-only-tp4-cards2-3-4-5-drain-age-paired-r1/result.json`
- 当前 Raw 最佳候选（FIA2 + DraftPA2）：
  `/root/data/nano-pearl-benchmark-results/20260928-specslo-v135-m1-128-draftpa-system-precheck/runs/candidate-on-v135-m1-128-fia2-draftpa2-r1/result.json`
- caller-primary 对照：
  `/root/data/nano-pearl-benchmark-results/20260928-specslo-v135-m1-128-draftpa-system-precheck/runs/candidate-on-v135-m1-128-fia2caller-draftpa2-r1/result.json`
- FIA1/FIA2/LRTF 对照：
  `/root/data/nano-pearl-benchmark-results/20260928-specslo-v135-m1-128-system-precheck/runs/`
- 当前源码的 MC2-on/off 系统预检：
  `/root/data/nano-pearl-benchmark-results/20260927-specslo-online-coalesce1-current-s31-precheck/runs/`

所有倍率必须从对应的 `result.json` 重算，不允许从日志中人工抄写后混用。

## 3. 已实现并有实机收益的优化

### 3.1 SpecSLO 链式执行基础路径

已具备 fixed-gamma4 full-window 链式 draft、双 batch 调度、rolling eager、
在线 Poisson 到达、coalesce1、compact proposal/correction transport、混合
prefill/decode target graph、论文口径 Goodput 和 graph 无回退证据链。

主要代码位置：

- `vllm_ascend/spec_decode/pearl/native_engine.py`
- `vllm_ascend/spec_decode/pearl/spec_rhythm.py`
- `vllm_ascend/spec_decode/pearl/native_graph.py`
- `examples/benchmark_nano_pearl_speculative.py`
- `examples/specslo_benchmark_report.py`

### 3.2 target FIA 图任务更新并行化

环境变量 `VLLM_ASCEND_PEARL_TARGET_FIA_TASK_UPDATE_WORKERS` 支持 1/2/4，
当前收益点为 2。它并行刷新稳定 target ACLGraph 内 FIA task 的 host 元数据，
并保留严格 shape、事件和 fallback 检查。

FIA1 到 FIA2 的一次系统预检：

- target 本地关键路径约 40.507 ms/cycle 降至 38.252 ms/cycle，约降低 5.57%；
- E2E 128.857951 s 降至 128.388763 s；
- Raw 919.834583 增至 923.196058 tok/s，约提升 0.366%。

这说明 host 图任务更新是实测有效的小幅优化，但单独不足以让系统 Raw 超过
TP4 baseline。

### 3.3 draft PagedAttention 图任务更新并行化

环境变量 `VLLM_ASCEND_PEARL_DRAFT_PA_TASK_UPDATE_WORKERS` 支持 1/2/4，
只在 pure PagedAttention、固定 gamma4、full-chain 且事件条件满足时启用，
其余情况 fail-closed 回到原路径。当前收益点为 2。

在 FIA2 基础上启用 DraftPA2 的一次系统预检：

- draft 本地耗时约 30.870 ms/cycle 降至 25.270 ms/cycle，约降低 18.14%；
- E2E 再降低约 0.1275 s；
- Raw 再提升约 0.0994%；
- 1928 次 replay、215936 个 task（112 task/replay），无该路径 fallback/reject。

系统收益远小于 draft 局部收益，是因为 target 仍是约 38.47 ms/cycle 的关键路径，
大部分 draft 降时被双 batch overlap 隐藏。

### 3.4 rank-3 MC2 小 M 生产资格链

已完成 Qwen3-32B TP3 的真实层输入 capture、projection-aware 资格判断、
M=1..128 静态路由、bit-exact/changed-input/单算子图/P95 门槛、adapter/vendor/
profile 身份绑定，以及 measured replay coverage 报告。

生产 profile：

`/root/data/nano-pearl-benchmark-results/20260928-specslo-mc2-v135-m1-128-union-s31/qwen3-32b-tp3-v135-m1-128-projection-aware-production-profile.json`

profile SHA256：
`db5821a50b2ea9594381a53965bf70e8682448e0556786d9f2a6beb0af01184f`

隔离 vendor：

`/root/data/specslo-mc2-v135-small-m/package-extracted/packages/vendors/specslo_mc2_v135_small_m_transformer`

MC2 源码 SHA256：
`a7db0857b5273a02c8773d84a3cea3df047ee7354431d6865902838da7d135ff`

当前 MC2 算子在合格小 M shape 上有局部收益；最新系统 run 的每个 target rank
均观测 2005 次，合格 any-projection 1409 次、full-chain 1390 次，覆盖率约
70.2743%，三 rank 一致。但 MC2-on/off 的完整系统 Raw 几乎相同，因此只能称为
“算子和路由资格完成”，不能称为已产生稳定系统收益。

## 4. 已实现但尚未实机验证的优化

### 4.1 target replay-ahead

新增默认关闭的 `VLLM_ASCEND_SPECRHYTHM_TARGET_REPLAY_AHEAD=0`。当前代码只允许
固定 gamma4、不可变 drain 场景，包含：

- 全角色逻辑 preflight 和 target 资源 pre/post consensus；
- proposal、prefix、KV、correction epoch/plan/home/request digest 校验；
- graph entry lease、完成事件和 fatal fence；
- 禁止提交阶段临时分配 KV；
- fallback4 最坏情况安全证明；
- consume 前再次核对所有 rank 状态，防止单 rank 分叉；
- graph validate replay、形状变化或状态不一致时 fail-closed。

该实现通过了 CPU/静态回归审查，但尚未在 NPU 上完成数值、图回退和吞吐验证，
也尚未完整接入 candidate/formal launcher。当前保守 gate 要求 admission/staged/
suspended 均为空且没有 eager，因此预计主要覆盖 drain 尾部，不能直接作为在线主区间
收益来源。

恢复后禁止直接放宽这些 gate。正确方向是先加入 scheduler next-plan reservation/
frontier epoch，冻结下一轮精确 target 集合，再让在线到达进入等待队列。

### 4.2 正式 runner 的 replay-ahead 证据

`examples/specslo_benchmark_report.py` 已开始记录 replay-ahead 环境和拒绝原因，
但 candidate/formal shell runner 还未完成开关传递和 `submitted == consumed`、
zero mismatch/fallback 的验收门。此项是明确未完成工作，恢复后首先补齐。

## 5. 已验证无效或暂不采用的方向

| 方向 | 结论 | 原因 |
| --- | --- | --- |
| v135 large-M MC2 | 否决 | 数值正确但所有候选均未通过完整速度门 |
| v136 large-M core48 MC2 | 否决 | 8/8 数值正确但 0/8 同时满足 P95 速度门；attention 和 down projection 不能形成共同生产覆盖 |
| v136 chunk1024 MC2 | 否决 | 只有 down M261 单点通过，attention 无合格 shape；固定 fullmesh 流量/同步仍是瓶颈 |
| target caller-primary | 暂不采用 | 局部只省约 0.057 ms/cycle，单次系统 Raw 反而从 924.114 降至 923.581 tok/s，轨迹也发生变化 |
| LRTF | 暂不采用 | 单次 Raw 922.904 tok/s，没有超过 FIA2/DraftPA2 |
| target fallback0 | 否决 | 历史 E2E 约 128.691 s，劣于保留 fallback4 的约 128.397 s |
| event group4 | 否决 | 历史系统结果更慢 |
| TP3 head padding/light-rank | 否决 | 增加 padding/不均衡后未形成系统收益 |
| 继续调 core/chunk 微参数 | 暂停 | 当前瓶颈是 fullmesh 通信和 barrier，不是单纯 AIV 核数或 chunk 循环 |
| 单独异步 Gloo correction | 未实现 | D2H 未就绪时预计只省约 0.1--0.2 ms/cycle，且有跨 PG 线程和 buffer 生命周期风险；只适合与 replay-ahead 联合评估 |

large-M v136 证据目录：

- `/root/data/nano-pearl-benchmark-results/20260928-specslo-mc2-v135-large-m-qualification/`
- `/root/data/nano-pearl-benchmark-results/20260928-specslo-mc2-v136-large-m-core48-qualification/`
- `/root/data/nano-pearl-benchmark-results/20260928-specslo-mc2-v136-large-m-chunk1024-qualification/`

后续 MC2 若继续，应转向 reduce-scatter/all-gather 或真实通信/计算重叠，
而不是重复 core/chunk 扫描。

## 6. 当前瓶颈解释

当前候选的局部算子改进没有等比例转化为 E2E，主要有三层原因：

1. draft 和 target 已重叠，缩短非关键的 draft 窗口只会减少隐藏区间，不能直接
   缩短由 target 决定的 cycle wall；
2. 小 M MC2 只覆盖部分 projection/row shape，完整 model-runner 仍包含未命中的
   MatMul、attention、collective、host 调度和 graph replay；
3. Poisson 在线 workload 的调度轨迹、接受率和请求完成顺序会改变 Goodput，
   单次 GP 上升不能替代 Raw 和三轮成对稳定性。

因此当前优先级不是再做孤立算子微调，而是把 target replay-ahead 安全地覆盖到在线
主区间，让下一轮 target graph 与本轮 draft/commit 控制面真正重叠。

## 7. 已完成的非 NPU 验证

在暂停前已完成以下回归：

- 本次 CPU 相关 18 个测试文件：1144 passed，14 warnings；
- `tests/ut/spec_decode/test_pearl_native.py`：465 passed；
- 加上 `test_spec_rhythm_linear_service.py` 的组合回归：547 passed；
- 另一组 graph/MC2/transport 重点回归：259 passed；
- target replay-ahead 定向回归：21 passed；
- Ruff 和 `git diff --check` 已通过当时检查。

这些结果证明控制面和 fail-closed 逻辑具备继续上 NPU 的条件，不代表 NPU 数值、
ACLGraph 无回退或系统吞吐已经通过。

## 8. 恢复工作顺序

插件更新后按以下顺序恢复，不能直接启动正式长测：

1. checkout 私有仓库 `SpecSLO-Chain`，核对本文档对应提交、工作树和 submodule；
2. 核对 Python、PyTorch、vLLM、vLLM-Ascend、CANN、固件、驱动和自定义 vendor
   版本，重新验证 profile/vendor/adapter/source SHA；
3. 执行非 NPU 单测、Ruff、`git diff --check`，确认插件更新没有改变接口；
4. 完成 replay-ahead 的 candidate/formal launcher、环境透传和报告 gate；
5. 先跑小规模 NPU correctness：token/output/KV parity、所有 rank consensus、
   graph capture/fallback/mismatch 均为零；
6. 跑一次完整 candidate 预检，检查 attempts/eligible/submitted/consumed/reject 计数；
7. 若覆盖仅在 drain 尾部，设计并实现 next-plan reservation/frontier epoch，
   不得直接删除 pending-admission 安全条件；
8. 只有单次预检同时满足 Raw > TP4 baseline、Goodput >= 1.3x 后，才执行三轮成对
   正式 A/B，顺序为 baseline-candidate、candidate-baseline、baseline-candidate；
9. 三轮每一轮都通过后，才能更新生产默认值和对外结论。

## 9. 暂停边界

本检查点刻意保留以下边界：

- 不把 target replay-ahead 标记为已验证或默认启用；
- 不把单次 Goodput 1.385x 标记为正式三轮结果；
- 不把 MC2 局部算子收益写成系统收益；
- 不合并到 `main`；
- 不运行占卡程序；
- 不在插件更新期间继续 NPU 测试或修改性能路径。

恢复时先从本文件和 `docs/specslo_work_record_zh.md` 读取上下文，再查看具体结果
JSON 和 git diff；聊天记录只作为补充，不能取代源码、结果和本检查点。
