# SpecSLO 程序级 Profiling 与性能优化记录（2026-09-11）

本文记录进入算子调优前的程序级 profiling 结论、已经验证有效的优化、已否决方向和
下一阶段问题。本文只描述 **SpecSLO/SpecRhythm**；PEARL 的历史吞吐不能替代本项目
的功能或性能证据。

## 1. 当前结论

在固定短回归 workload 上，当前稳定版本已满足以下两个功能门槛：

- 树状 target verification 全程使用 ACLGraph，失败捕获、容量回退和 shape 回退均为 0；
- TP1 draft 与 TP3 target 存在真实设备计算重叠，而不是只在 Python 时间线上看起来并行。

当前尚未达到最终性能目标。Qwen3-0.6B TP1 + Qwen3-32B TP3 的三次生产路径中位吞吐为
**167.785 token/s**，相对生产 vLLM-Ascend Qwen3-32B TP4 target-only 的
**141.663 token/s** 为 **1.1844x**。首要目标“超过 baseline”已在该短 workload 上
达到，但 **1.3x 尚未达到**，也不能由该短回归外推到 RPS/Goodput 验收矩阵。

## 2. 固定测量条件与证据边界

本阶段用于快速迭代的固定条件如下：

| 项目 | 配置 |
| --- | --- |
| 数据 | GSM8K 固定前 8 条 prompt |
| 输出 | 每请求 16 token，共 128 token |
| SpecSLO | `B=8`、tree width/depth=`2/2`、`gamma=4` |
| 拓扑 | draft TP1 + target TP3，NPU 1/2/3/4 |
| 模型 | Qwen3-0.6B + Qwen3-32B |
| 执行 | Graph、CPU tree verdict、continuous batching、preemptive scheduling |
| baseline | 生产 vLLM-Ascend Qwen3-32B TP4 target-only |

生产吞吐、同步细分和设备 trace 是三种不同实验，不能混为一个数字：

1. **生产 E2E**：不额外插入同步或 profiling collective，用于报告吞吐；
2. **低扰动 host timeline**：只记录各 rank 的 `time.perf_counter()`，不增加 NPU
   synchronize 或 collective，用于定位生命周期、队列和首次 shape 异常；
3. **同步细分 profile**：在指定 decode step 的阶段边界同步，用于拆解已完成工作，
   会降低吞吐，不能作为生产性能；
4. **CANN device trace**：用于核实设备 kernel 和真实 overlap，也有明显采集扰动。

固定 TP4 baseline 证据：
`/root/data/nano-pearl-benchmark-results/20260910-specslo-performance-tuning/baseline-vllm-tp4-b8-p8-t16-warm.json`。

## 3. 已定位并修复的程序 bug

### 3.1 首次尾部 Graph shape 被重复执行完整 32B eager forward

修复前的低扰动 timeline 显示：step 1--9 约 43 ms，step 10 约 77 ms，随后 B2/B1
尾部 shape 在 step 11--14 突然升至约 187--199 ms。异常集中在 target Graph replay，
scheduler、verdict 和 state update 均正常，因此不是普通 Python 抖动。

根因在 `NativeACLGraphRunner._capture_target` 的验证状态：capture 阶段已经执行真实 replay
并与 eager reference 比较，但新建 `NativeTargetACLGraphEntry` 的
`runtime_validated` 仍为 false。某个 shape 第一次在后续请求中 replay 时，运行时又执行
一次完整 Qwen3-32B eager reference。warmup 已覆盖 B8 shape，所以稳定段不触发；B2/B1
只在测量尾部出现，因而形成“首轮/尾部随机慢”的假象。

修复后：

- 默认以严格的 capture-time replay/eager 比较作为生产资格；
- 后续 changed-input 二次验证仅在显式诊断环境变量
  `VLLM_ASCEND_PEARL_VALIDATE_GRAPH_REPLAYS=1` 下启用；
- 新增 `aclgraph_runtime_validation_replays` 计数器，生产测量必须为 0；
- 保留 failed/capacity/shape fallback 的硬失败门禁。

修复后三次吞吐为 165.409、167.785、167.988 token/s，中位 167.785 token/s；最慢与
最快差异约 1.56%，首次运行不再坍塌到 88 token/s。三次输出 SHA256 均为
`3204949b4f0aecb1b4a5398994b38d008c21bb30163c03073c245bb46d74e3b1`，与 native
TP3 target-only 完全一致。

证据：

- 修复前：`tree-host-profile-first-run-v24.json`；
- 修复后：`tree-b8-B8-graph-validation-fix-3run-v25.json`。

以上文件位于
`/root/data/nano-pearl-benchmark-results/20260910-specslo-performance-tuning/`。

## 4. 已验证有效并保留的优化

### 4.1 Graph 与 FULL-mask FIA

- 树 verification 强制保持 FULL-mask FIA，移除 causal/FULL backend 摆动；
- 对 KV 长度做稳定 bucket，长度不变的 replay 跳过逐层 graph-task update，只刷新捕获
  event；
- capture validation 与 steady replay 使用相同的 update-stream 依赖；
- Graph fallback 由普通统计字段提升为 benchmark 硬门禁。

### 4.2 树 KV 与候选路径

- exploratory draft KV 延迟到全局候选选择之后再物化；
- draft publish 和 target accepted-path compaction 改为按轮次、按 layer 合批；
- identity mapping 跳过无意义 KV 搬运；
- 缓存模型生命周期内已验证的 layer KV 视图与容量，避免每轮重新遍历 64 层。

### 4.3 控制面与数值保护

- target leader 的树 verdict 改为一次紧凑 D2H 后由 CPU 遍历小树，再一次 H2D 发布；
- draft/target 窗口时间搭载现有 proposal/verdict envelope，移除生产路径额外 timing
  collective；
- 数值健康 bit 合入已有 commit consensus，移除 preflight 中额外 `flag.item()` 设备栅栏；
- 每层 Q/K/V 三次有限值检查合并为一次 fused QKV 检查，同时保留 attention 输出、最终
  hidden 和 logits 的 fail-before-commit 保护。

### 4.4 Profiling 工具

- 新增 `--profile-host-decode-steps`，记录不加同步、不加 collective 的 rank-local
  decode timeline；
- worker metrics 支持 list 型 timeline，公共聚合只对数值指标求和；
- 新增 Graph runtime validation、task update/skip 计数；
- host profile 已进一步拆分 scheduler 与 commit/state 子阶段。

## 5. 当前耗时画像

### 5.1 生产路径低扰动 host timeline

最新一次用于归因的样本吞吐为 166.254 token/s（1.1736x baseline），输出 hash 一致且
零 Graph fallback。该单次结果用于时间归因，不替代三次中位生产结果。

在同时存在 target 和 draft 工作的周期内，中位耗时如下：

| 阶段 | 中位耗时/step |
| --- | ---: |
| 完整 cycle | 43.209 ms |
| Scheduler 总计 | 2.216 ms |
| ├─ roof/B 查询 | 0.017 ms |
| ├─ dual-batch plan | 0.062 ms |
| ├─ budget shaping | 0.123 ms |
| └─ tree plan/ticket 构造 | **2.014 ms** |
| Draft compute host window | 22.861 ms |
| Target compute host window | 33.888 ms |
| Draft/Target host window overlap | 22.802 ms |
| Draft -> Target collective 临界跨度 | 0.479 ms |
| Target verdict | 0.456 ms |
| Target -> Draft collective 临界跨度 | 1.799 ms |
| State update 总计 | 3.072 ms |
| ├─ preflight | 0.634 ms |
| ├─ commit consensus | **2.182 ms** |
| ├─ KV compaction | 0.032 ms |
| └─ request commit | 0.279 ms |

这组数据证明目前最大的两个非模型程序热点是 tree plan/ticket 构造和全局 commit
consensus，不是 roofline 查表或预算算法本身。证据：
`tree-b8-B8-host-subphases-v30.json`。

### 5.2 同步细分 profile

同步细分会扰动流水线，本次吞吐只有 159.18 token/s。8 个被测 step 的平均耗时为：

| 阶段 | 平均耗时/step |
| --- | ---: |
| Draft compute | 24.482 ms |
| Draft -> Target communication 临界跨度 | 10.349 ms |
| Target verify | 31.439 ms |
| ├─ target model | 30.940 ms |
| └─ verdict | 0.499 ms |
| Target -> Draft communication | 1.678 ms |
| wait/sync/state update | 10.657 ms |

其中 10.349 ms 不能直接解释为 HCCL 网络 kernel：同步 profile 中 draft 会在 collective
入口等待更慢的 target，该值包含排队/等待。未加同步的实际 host enqueue 临界跨度约
0.45--0.50 ms。证据：`tree-profile-b8-B8-graph-validation-fix-v26.json`。

### 5.3 CANN 设备 trace 与 overlap

四 rank 的第二个完整 capture 已用官方离线 `torch_npu` 分析。三个 target rank 与 draft
rank 的 AICore+MixAIC 计算重叠分别为 8.918/8.502/8.537 ms，占 draft device busy 的
54.26%/51.73%/51.94%。若统计全部 device compute，overlap 为
18.195/18.251/18.172 ms，占 draft device busy 约 57.6%--57.9%。

因此双 batch 已有真实设备 overlap，但仍只覆盖 draft device busy 的约一半；host
timeline 中 target 每步比 draft 长约 11 ms，尾部尚未被普通 draft 覆盖。

三步聚合的主要 kernel：

- draft rank：MatMul 12.272 ms、FIA 6.267 ms、IsFinite 3.256 ms、ReduceAll
  1.104 ms、RMSNorm 3.492 ms；
- target rank：MatMul 约 61.77--62.01 ms、FIA 约 4.66--4.67 ms、IsFinite
  3.32--3.55 ms、ReduceAll 1.12--1.15 ms；
- target TP3 还包含 387 次 `hcom_allReduce_`（约 9.8 ms）及展开 HCCL task（约
  6.7 ms）。

当前 target 最大算子瓶颈是矩阵计算，TP3 all-reduce 次之；但在先解决上述程序热点前，
不能只凭 operator aggregate 直接设计 kernel。设备证据目录：
`parsed-graph-overlap-b8-validation-fix-v27/`。

## 6. 已否决或暂不采用的方向

### 6.1 简单增大 B

手动 probe（不是论文 §5.2/§5.3 最终校准）：

| B | 三次中位吞吐 | 相对 TP4 baseline | 结论 |
| ---: | ---: | ---: | --- |
| 8 | 167.785 | 1.1844x | 当前固定调优点 |
| 10 | 150.256 | 1.0607x | 否决 |
| 12 | 146.277 | 1.0326x | 否决 |
| 16 | 157.996 | 1.1153x | 否决 |

B10/B12/B16 会触发非前缀候选选择、draft KV move 和 target KV compaction，同时接受率
下降。当前差距不是靠放宽 B 就能弥补。正式 B 只在执行架构发生大变化后按论文流程重测，
不因每次普通源码修改反复校准。

### 6.2 FULL FIA Graph 中跳过二维 mask copy

FULL FIA 实际消费 4D `tree_attention_mask`，因此尝试不捕获/复制 2D logical mask。
数值与零回退门禁通过，但三次吞吐为 162.491/166.598/167.673 token/s，中位
166.598（1.1760x）；decode 中位约 611.5 ms，也没有优于保留路径的约 607.5 ms。
该修改已经撤销，不计作有效优化。证据：`tree-b8-B8-dead-mask-3run-v29.json`。

### 6.3 TP3 fused MC2

当前 910B2/CANN 对 rank-3 MC2 的真实模型 shape 不提供数值且性能合格的 fused kernel；
已有 probe 返回 rank size 3 不支持或比 matmul + HCCL 更慢。生产仍使用分离路径，除非
新 kernel 先通过独立数值和性能资格，不能强制打开历史实验开关。

## 7. 接下来要解决的问题（按优先级）

1. **tree plan/ticket 构造 2.014 ms/step**：继续拆解 `_spec_rhythm_tree_plan`、
   eager plan、NPU tensor/attention-mask 创建和重复 rank-local 工作；优先消除重复构造与
   隐式同步，再决定是否缓存静态拓扑。
2. **commit consensus 2.182 ms/step**：区分 health-flag 聚合 kernel、world
   all-reduce 和 D2H scalar fence。必须保留 fail-before-commit 语义，候选方案是共享/分层
   sticky flag 或将 consensus 搭载已有 correction 边界，而不是删除数值保护。
3. **target 未覆盖尾部约 11 ms/step**：在 SLO workload 上验证 rolling eager
   continuation 是否真的填充该窗口；不能用普通无 SLO 短测中 eager=0 的结果宣称论文
   continuation 已产生性能收益。
4. **target MatMul 与 TP3 HCCL all-reduce**：在程序热点收敛后做 shape 级 roofline，
   设计/筛选 rank-3 kernel、通信与计算融合或 model-runner compile 融合。
5. **最终验收**：架构有显著变化后重新执行正式 B profiling，再跑 RPS=2/4、
   batch=8/16/32/64、SLO 6:2:2 的完整 TPOT attainment/Goodput 矩阵。最终门槛仍是
   attainment 稳定不低于 80%，Goodput 至少为 TP4 baseline 的 1.3x。

## 8. 当前回归状态

- 本阶段相关 spec-decode 回归：288 passed；
- 新增 host 子阶段统计后的聚焦回归：171 passed；
- 实际改动文件 Ruff：全部通过；
- 生产短测输出与 native TP3 target-only hash 一致；
- Graph failed/capacity/shape fallback 均为 0；
- `aclgraph_runtime_validation_replays` 为 0。

本文是继续优化前的检查点。下一轮不得把同步 profile 的吞吐当生产吞吐，不得把 host
collective span 当纯网络时间，也不得在未重跑完整 SLO workload 前宣称达到 1.3x。
