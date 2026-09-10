# SpecRhythm 核心功能及第五章实现审计（2026-09-10）

> 本文开头保留首次审计的源码快照与反例，不是后续修复代码的现状列表。
> 同日各次追加继续保留当时验收边界；最新修复、真实 NPU 结果与未通过项统一见
> [核心闭环回归记录](specslo_core_regression_20260910_zh.md)。最新冻结版的 8/8 回归
> 已通过，但不能由此推出跨注意力后端全词表严格 allclose 或 Goodput 1.3× 目标也已通过。

下述“不能声称完成”是首次审计快照的结论，已被同日后续修复取代。当前文本生成范围
内的 5.1/5.2 核心闭环已经完成并通过真实 Qwen3 TP1+TP3 Graph 生命周期验收；最终
schema v3 `B_roof` 也已在功能冻结后按 §5.3 实测。尚未完成的是 RPS/Goodput 1.3×
性能目标、跨 dense/PA 与 FIA 的严格 raw-logit 一致性（保留为跨后端诊断失败），以及
被实机资格测试否决的 TP3 fused MC2；这些不能倒推为 5.1/5.2 功能未实现。

首次审计结论：当时不能声称论文核心功能，包括 5.1、5.2，已经全部完成。
本次依据论文第 6–9 页和当前工作区执行代码重新检查；本记录优先于此前
“只剩性能验收”“三条核心机制已经闭合”的描述。本次进行了只读代码审计、CPU
回归和文档更正，没有修改推理实现或运行新的 NPU 性能实验。

审计基准为 HEAD `f22c3d7d` 加当前未提交修改。论文 SHA256：
`3a3017cf736bb14b80ea81cb7ba991c3fb5068e44bd7aac66215bc9e52e69d50`。
审计时 `native_engine.py` SHA256：
`dc27a21ad06903c3e2e25794cadae148478f35fe6c12f9215d9600d6b3463b4b`。

## 论文要求与现状

| 论文要求 | 已有实现 | 未完成的关键部分 |
| --- | --- | --- |
| 5.1：调度、并发执行、状态管理三阶段 | runtime tracker、两步预算、执行计划、提交状态机均有实现 | 树路径的跨 rank 调度状态不一致；当前不是一个已验证的完整服务循环 |
| 5.1 / 4.2：同一步验证 A、生成 B，并在同一 draft 窗口容纳 eager 请求 | 线性路径具有跨 worker 的同一步并发结构 | 树路径在 target forward 前同步等待当前 draft，实际串行化 |
| 5.2：双槽动态维护，新请求进入负载较低的槽，完成后补入 | 通用 scheduler 与线性 native 有相关准入逻辑 | 树循环把全部请求直接设为 active，没有容量限制、arrival gate 和后续准入 |
| 5.2：精确依赖验证、完整接受后 promotion、拒绝后丢弃 | 树路径已记录并比较依赖 token 与 frontier，控制器有 epoch 检查 | 树路径在 epoch 校验前修改 token 状态；记录的父 proposal id 未参与 promotion 判定 |
| 4.4 / 5.3：全局验证预算约束实际候选集合 | 显式 verification_budget 可与 gamma 解耦，存在两步标量分配器 | 只约束本轮新增 proposal，缺少当前 ready 集合的跨轮预算核算；树候选选择尚非论文描述的跨请求置信度细化 |
| 树状图执行 | 树 target 已接入 ACLGraph 和 eager fallback | 物理 forward 仍计算固定树的 padding；graph key 随 context 改变，短 replay 不能证明长解码稳定复用；draft 树扩展仍逐请求、逐层执行 |

## 5.1：同一步执行与状态一致性

论文允许每步计算结束后统一提交，并没有要求多个 service step 必须同时在飞。
因此此前把“没有跨轮深流水线”直接等同于“没有实现论文 overlap”的解释应撤回。

线性路径中，`native_engine.py:2512` 的 draft 和 `:2545` 的 target 都在
proposal exchange 前启动；两个函数分别在非本模型 rank 早退（`:4247`、`:4546`），
所以不能从同一个源文件的调用顺序判断它们在不同 NPU 上串行。实际重叠比例仍需
共同时间轴的 NPU trace，CPU 回调的 overlap 测试不能替代它。

树路径存在明确的同步依赖：

```text
draft_tree_forward (:1722)
  -> 同步候选 broadcast (:1743；实现 :1473)
  -> target_tree_forward (:1839)
```

target rank 在进入自己的 forward 前等待本轮 draft 消息。因此计数器
`spec_rhythm_dual_batch_overlap_protocol=1`（`:1621`）不是实测重叠证据。

另一个问题是树 payload 的 confidence：`:1802` 在 draft rank 保存真实值，在
target rank 使用 `1.0`。候选通信只包含 token 和 frontier（`:1450`），没有传递
confidence；`:1947` 又按各 rank 本地值更新 EMA。这会使后续预算或 eager 集合
产生分歧，进而可能形成不匹配的 HCCL 消息。这里只确认代码上的分歧风险，未把它
描述成已经在本次 NPU 测试中观察到的死锁。

## 5.2：动态准入与 guarded commit

线性 `admit_ready`（`:2223`）有容量与 arrival 判断，并把请求分配到当前请求数
较少的 home（`:2256`）。树循环则在 `:1595` 使用
`active = list(range(len(local_states)))`，未使用传入的 `initial_batch_size`
和 `continuous_batching` 来执行这些维护操作。因此树模式不能被当作完整的在线
双槽服务实现；`online_prefill` 组合还必须补齐进入树循环前的 KV 可用性。

树 eager 已有真实 token 依赖检查：`:1786` 保存父 proposal、依赖路径、frontier；
`:1916` 至 promotion 前比较实际接受路径与 frontier。这是已完成的工作。
但 `:1902` 先调用 `apply_tree_verification` 修改请求状态，到 `:1941` 调用
控制器时才检查 prefix epoch；异常路径并非 mutation 前的完整校验。
`eager_parent_proposal_id` 目前只被记录，没有参与此处依赖判定。

此外，native API 在整批 `generate_batch` 返回后才发送结果（`api.py:659`、`:671`），
当前请求内部 committed token 统计不能直接等同于逐步向客户端交付 token。

## 固定 B 与树图执行的边界

论文约束为每个验证周期 `sum(n_i) <= B_roof(t)`；B 是上限，不要求每步无条件填满。
固定 B 的实验可以把 B 与 gamma 上限解耦，论文的 B_roof 则按模型、活跃批量和
上下文范围查离线表。两者都需要约束实际送入验证的候选，而不只是新生成的预算。

`spec_rhythm.py:213` 已优先读取显式固定 B，但 `:353` 为填满 B 又分配剩余槽位，
这段没有重复执行 acceptance 门控。树循环也先选 ready payload 再为新 normal/eager
分配 B（`native_engine.py:1643` 附近），送入 target 前没有对跨轮保留的 ready
集合重新做全局候选上限检查。

纯 CPU 控制器反例已复现：4 请求，固定 `B=8`、`max_gamma=8`、
`min_gamma=1`，请求 0/1 的 SLO 为 40ms，2/3 为 150ms；默认双槽交替、
priority 开启，projected wait=100ms，每轮时间递增 50ms，使用
`random.Random(0)` 产生全接受/部分拒绝序列（全接受概率 0.7）。末三步为：

| step（从 0 开始） | 本步待 verify 候选 | 本步新增 normal | 本步新增 eager |
| --- | --- | --- | --- |
| 8 | 请求 1:3、请求 3:5（共 8） | 请求 0:3、请求 2:1 | 请求 1:4 |
| 9 | 请求 0:3、请求 2:1（共 4） | 请求 3:8 | 无 |
| 10 | 请求 1:4、请求 3:8（共 12） | 尚未分配即发现超限 | — |

每步新增预算都未超过 8，但 step 10 实际选入 verify 的候选为 12。这是控制器
级复现，没有声称本次在 NPU 执行了超预算 forward，也不将反例推广为所有
`merge_ready_homes` 配置必然出现相同问题。

树主循环使用固定 spine-first 前缀作为候选；`select_tree_candidates` 目前只有
定义，没有被 native 主循环调用。target forward（`:3139`）仍按
`width * depth + 1` 个位置计算每棵树，`:3199` 后才截取有效预算。
padding 可以用于图执行，但其实际计算成本必须纳入 profiling，不能据逻辑有效
节点数就声称 target 计算量已经被固定 B 完整限制。

`native_graph.py:445` 把 context lengths 写入 tree graph key；context 改变可能
新增 capture，达到容量后 eager fallback。`used_aclgraph`（`native_engine.py:3219`）
只反映开关开启，不能独立证明此次调用发生了 graph replay。

## 本次验证与后续验收条件

执行以下现有测试得到 `144 passed, 14 warnings`：

```bash
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  /root/miniconda3/envs/vllm-hust-dev/bin/python -m pytest -q \
  tests/ut/spec_decode/test_spec_rhythm_native.py \
  tests/ut/spec_decode/test_pearl_tree.py \
  tests/ut/spec_decode/test_pearl_native.py
```

这证明现有组件测试通过，不覆盖上述跨 rank 计划一致性、树模式在线准入和完整
NPU 时间轴。此前文档记录的短 smoke 也不能单独证明这些功能已经齐备。

后续应按以下顺序验收：统一计划与跨 rank 状态；约束实际 ready verify 候选集合；
树路径同一步 draft/target 并发；动态双槽准入；mutation 前完整依赖校验；最后
完成树候选细化与长序列 graph/eager 数值对照。在这些条件成立后，才能用相同
decode-stage TPOT/Goodput 口径评价 80% 达成率和 1.3x 性能目标。

## 后续修复进展与待硬件验证（2026-09-10，同日追加）

上文保留的是首次审计快照，包括原始 B=8、待验证候选累积到 12 的反例，不能
删掉后再把组件存在描述成一直正确。随后已对实现、CPU 集成测试和公开调用接口
作出以下修改。本节只记录已检查的代码与 CPU 证据，不预填正在进行的 NPU
回归结果，也不据此宣称论文全部功能或 1.3x Goodput 目标已达成。

| 首次审计问题 | 当前软件修复 | 验收边界 |
| --- | --- | --- |
| 本轮 draft 通信阻塞 target forward | 树循环先调用角色本地 draft 和 target forward，再发布本轮候选；非本模型 rank 立即返回 | CPU 测试验证调用顺序；真实 NPU 重叠及比例须看共同时间轴 |
| rank 使用不同 confidence | envelope 统一发布候选 token、原始选中节点、frontier 和 confidence；各 rank 使用相同接收值更新状态 | 四个模拟 rank 的真实编码/解码一致；不能替代多卡长期通信验证 |
| B 仅约束新增 proposal | `build_plan` 按 ready 集合的实际候选数打包/延后；target forward 前再次检查实际候选总数 | 固定 B 不再放行累积超限；动态 B 缩小的处理见下文复核项 |
| 候选只取固定主干前缀 | draft 返回探索树逐节点 confidence；主循环调用 `select_global_tree_candidates` 做全局两阶段、祖先闭合选择 | 初步标量预算用于挑选探索请求，最终送入 target 的树使用真实逐节点选择结果 |
| 物理计算仍执行全部 padding | `pack_selected_tree_plan` 生成仅含选中节点的 target 输入 | 物理查询数为 `候选总数 + 验证请求数`，根节点成本必须纳入离线测量 |
| 树 draft 逐请求串行调用 | normal/eager 在同一深度合并调用；eager 使用不覆盖未提交父树的 scratch KV，promotion 后才移动 | 批处理、遮罩及非连续页 KV 测试已加入；多轮模型数值待 NPU 对照 |
| 无动态容量、准入及补入 | 树主循环维护 pending/active，遵守容量和在线 arrival，按较轻槽准入，完成后补入；请求未提交期间保持原槽 | 覆盖请求数大于容量及小于等于容量两种在线到达情况 |
| mutation 先于完整校验 | target forward 前校验，提交前对整个验证批次预检 epoch、身份及祖先路径，再修改任何请求；eager 还比较父 proposal ID、完整路径和 frontier | 已故障注入首行/末行 stale；末行错误不能先提交首行 |
| 缺少实时交付 | `PEARLEngine.generate(on_token_commit=...)` 支持逐 guarded commit 交付，worker 的 commit 消息先于 final result | 真实 CPU Pipe 覆盖多事件/最终结果收集；客户端慢回调会造成背压，并非 HTTP 服务接口 |
| 回调失败可能破坏后续步骤 | 客户端回调失败先排空 worker 消息；native 回调异常延迟到推理结束报告，`finally` 清理回调引用 | 验证异常后请求完成、缓存释放、连接无残留消息，以及下一批不复用旧回调 |
| 首轮 TPOT 时间被漏算 | 准入时计入实际已交付 prefill token，并从该处建立 decode 时间基点，不在首轮多 token 验证后重新置零 | 确定性 CPU 例：20ms warmup + 50ms 首轮验证、1+3 tokens，TPOT 为 70/3ms |
| W 仅有独立组件未接入 | native 树循环根据共享角色计时更新 `DraftWindowEstimator`，为完整探索树计费，再用剩余窗口限制 eager 请求数 | 测试 W=2ms 禁止额外 eager、W=16ms 允许；未校准时只执行必要 normal 工作 |
| 图开关被当作 replay 成功 | 返回值改用 graph runner 的实际执行状态；图 key/可变 metadata 及 eager fallback 独立测试 | `used_aclgraph`、capture/replay/fallback 计数与 NPU 数值结果应联合检查 |

### 对“target verify 已重叠”的准确解释

这里的并发是不同模型 rank 之间：target rank 不运行 draft 模型，可以立即执行
自己的 target forward；draft rank 不运行 target 模型。角色本地 synchronize
不等于 world barrier。相反，候选广播、共享计时 collective、判决广播仍在步末
形成会合，因此不是无限深流水线。

当前耗时最大的 target 模型 forward 已放在新候选交换之前。随后路径匹配的
`verify_tree_outputs` 和 guarded commit 仍在步末执行。不能把模型 forward 的
host 时间窗直接称作全部 verifier 算子的设备重叠率；NPU timeline 必须同时展示
draft/target kernel、通信及提交边界。共享计时使用 int64 微秒 collective，不将
CPU 支持的 float64 reduction 当作 HCCL 支持性证据。

### 同日二次复核：仍需确认的功能边界

以下是二次检查时发现的具体边界，不是性能不佳的泛泛归因。后续修复及硬件
结果应逐项追加，不能把未通过项简单改为“仅需调优”。

1. **动态 roof 缩小。** 若一个待验证 proposal 有 4 个节点，当前表项从
   `1:1 -> 4` 切换到 `1:2 -> 2`，控制器目前会安全拒绝该 oversized proposal，
   不会越过 B；但拒绝不等于服务闭环。调用方需要在调度前祖先安全地修剪/重建
   payload，并正确失效依赖其原始路径的 eager。二次复核的纯 CPU 例在 context
   从 512 变到 513 时复现该拒绝。固定 B 回归不覆盖这个动态情况。
2. **SpecRhythm 退回普通 PEARL 的条件（已修复代码条件）。** `enable_spec_rhythm=True` 但无 SLO、
   `min_gamma=gamma`、树宽深为 1 时，优化快路径仍可能被选中。显式固定 B 或
   逐步 token 回调必须强制保留控制面；否则会出现接口接受参数但预算/流式事件
   不生效。后续已在 `spec_rhythm_needs_control_plane` 条件补入显式固定 B 和
   token 回调；普通 PEARL 的非零温度、原有返回格式保持不变，路由回归已新增。
3. **采样契约（greedy 边界已显式校验）。** 本轮树 verifier 是 greedy。
   `_generate_batch_impl` 现在在分配模型缓存前拒绝非零 target/draft temperature，
   不再执行 argmax 却在结果里声称使用了用户指定温度。
   论文 6.1 只要求系统间采样参数一致，没有公开具体温度；全文也没有把实验明确
   标成 greedy。因此可以报告本项目的 greedy 验收，不能据论文推断随机树采样
   已完成，亦不能把该缺项归因于国产 NPU。
4. **离线表到运行时的连接。** profiler 已能产出带 model/TP/hardware/mode、
   epsilon 和原始测量证据的报告。运行接口目前接收裸 `batch:context -> B`
   映射；完整 profile 文件的身份校验、缺失表项处理及实际覆盖范围仍需核对。
   没有实测覆盖的 key 退回 `batch * max_gamma` 是兼容策略，不是经过 profile
   证明安全的 roof。B 固定实验和论文离线表实验必须分别标注。
5. **指标口径。** 当前代码采用从首 token 到末 token 的 decode 区间，除以
   `N_out - 1`；论文 6.1 写的是 `T_decode / N_out`。两种数值不能混用，更不能
   将一边的冷启动 E2E 吞吐和另一边的 decode Goodput 直接计算加速比。
6. **在线补入时旧请求的等待计时。** 二次复核时，循环在 `admit_available`
   补入/prefill 之前结算本轮 decode 时间；新请求的共卡 prefill 因而可能占住
   设备，却未进入仍 active 的旧请求的 TPOT。论文的独立 prefill 池实验与这种
   共卡 online 模式必须区分：后者需要把旧请求在 refill 期间的真实等待计入。

### CPU 证据与硬件待验项目

同日新增的集成测试不是独立模拟另写一套调度器，而是直接调用 native 树循环，
只替换模型计算、设备同步、通信传输及 KV 移动的设备边界，保留真实控制器、
候选选择、树 verifier、状态提交及流式消息处理。该测试集合与完整 native
测试在加入 int64 微秒计时及非零温度显式拒绝回归后，共
`155 passed, 14 warnings`（1.30s，含固定 B/stream 强制保留控制面的路由测试）。
这些 CPU 测试不等于 NPU 吞吐或时间轴通过。

```bash
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  /root/miniconda3/envs/vllm-hust-dev/bin/python -m pytest -q --tb=short \
  tests/ut/spec_decode/test_spec_rhythm_tree_loop.py \
  tests/ut/spec_decode/test_specslo_streaming.py \
  tests/ut/spec_decode/test_pearl_native.py
```

硬件验收仍应至少包括 Qwen3-0.6B TP1 + Qwen3-32B TP3 的多请求、多轮 eager /
ACLGraph token 对照；接受与拒绝并存时的 eager promotion/invalidation；变化的
active batch、上下文和预算；真实 draft/target 时间轴；同一 token/时间口径的
baseline TP4 对照。此处不预填吞吐、SLO attainment 或 Goodput 的任何通过值。

### 同日第三次追加：动态边界、严格测量表与采样器

保留上文二次复核的原始缺口描述。以下记录的是随后落地的软件修复，而不是
把历史缺口改写成一开始就完整，更不替代尚未完成的 NPU 验收。

- 动态 B 的闭环已加入代码和 CPU 回归：树 proposal 按祖先安全的拓扑前缀裁剪，
  同步裁剪物理 KV 映射并使依赖旧路径的 eager 失效；linear 分支在共享边界按
  新预算从已提交 prefix 重建当前角色状态。`test_spec_rhythm_dynamic_roof.py`
  包括实际循环的边界回归；不能据此声称新的多卡动态 B 数值实验已经通过。
- 共卡在线 refill 的时间已计入旧 active 请求，而不记入新请求首 token 之前的
  decode。确定性集成例中，req0 用 70ms 完成；新 req2 的 prefill 耗时 100ms，
  req1 随后再等待 50ms 验证，因此三个请求的 TPOT 分别为 `70/3`、`220/3`、
  `80/3` ms。旧 req1 不能只报告漏掉 refill 的 `120/3` ms。
- 新 `roofline.py` 将完整测量文档规范化为 `ProfiledRoofline`，保留 metadata
  和 evidence，并在配置端核对 target model、TP、eager/graph 模式；native
  初始化后读取实际设备型号，在 world 上同步失败标志，任一 target 型号不符
  时全部 rank 在模型加载前退出。不是将用户传入的 hardware 字段当作检测结果。
- 严格表缺少 `active_batch:context_bucket` key 时直接拒绝执行，不退回 gamma
  推导值；显式固定 B 也不能覆盖严格表。`validate_execution` 另外核对实际
  target 请求行数是否在 evidence 中：半批测量不能覆盖 merged-home 或尾批。
  老实验的裸 dict 仍可使用，但它保留的是**未测量的兼容 fallback**，不会被
  转换为 `ProfiledRoofline`，不能当作论文 5.3 的实测证据。
- 测量文档显式记录 `verification_layout=packed_tree`。strict 配置拒绝让
  linear PEARL 复用该树表，不能仅因 model/TP 相同就忽略 attention 路径差异。
  服务循环在实际选好 target 请求后核对 profile 的物理行数；排队中尚未准入
  的长 prompt 不影响当前 active batch/context 的查表。
- strict graph 表还检查真实 target 执行结果；任一 target rank 回退 eager，
  所有 rank 通过 int64 MAX 失败标志一致拒绝本轮，在新候选发布和判决提交前
  退出。draft 自己回退 eager 不会被误当作 target roof 无效。legacy 裸表不
  添加这个 collective，也不凭兼容执行声称获得严格图模式预算保障。

离线测量入口为 `examples/measure_specslo_tree_roofline.py`。它复用 TP1+TP3
native engine，但在采样期间只让 target TP3 计算，构造固定的合成 prefix，
在计时前完成 KV prefill、输入与 metadata 准备。AR 控制与树 shape 使用相同
物理 request 行数，AR 为每请求一个 root query，树为 root 加实际候选数。
计时覆盖模型 forward、greedy 输出头、必要的图 replay 管理和设备完成，不含
输入准备、prefill、untimed warmup/capture、测量结果的跨 rank 汇总。逐样本取
三个 target rank 耗时的最大值，汇总使用 HCCL 支持的 int64 微秒而非 float64。

```bash
torchrun --standalone --nproc_per_node=4 examples/measure_specslo_tree_roofline.py \
  --mode graph --batch-size 8 --verification-requests 4 --contexts 128 512 \
  --candidate-budgets 4 8 16 --output /path/target_samples.json
python examples/profile_specslo_tree_roofline.py \
  --measurements /path/target_samples.json --epsilon-relative 0.10 \
  --output /path/measured_roofline.json
# 现有推理/benchmark 入口的同名选项现在也接受完整文件：
# --spec-rhythm-roofline /path/measured_roofline.json
```

图模式必须提供真实 runner 的 capture/replay 计数，计时区间发生 capture 或
某次没有 replay 时测量失败，不生成合格表项。可以反复传入
`--candidate-counts 1,1,1,3` 等非均匀形状；默认均匀/近均匀 sweep 只证明实际
测过的候选分配，并非证明所有分配都等价。512-token context bucket 内也只
有 evidence 明列的 context 点是实测，不能将桶式查表说成每个长度均已验证。
这套离线测量是目标端容量证据，不是 GSM8K/ShareGPT 精度、端到端吞吐或
baseline TP4 的 Goodput 性能对比。

采样器的中间 JSON 明确为 `running`，错误退出为 `failed`，只有正常完成才为
`complete`；聚合器拒绝把这些中断/失败文件转换成完整测量表。`--dry-run` 只
输出未测量的形状清单，不产生 latency 或 roofline 数值。

新增采样器和严格表加载测试合计 `37 passed, 14 warnings`（CPU，0.72s），
覆盖物理 root 计数、准备工作不重复进入 forward 回调、int64 逐样本 MAX 汇总、
真实 counter 证据接入、错误 model/TP/mode/hardware 拒绝、跨 rank 一致失败、
缺 key 和未测物理行数拒绝。该结果不包含真实 NPU roofline 数值。

加入实际服务循环的 strict profile 绑定、strict graph fallback 全 rank 拒绝、
EOS/输出上限真实状态截断、动态 B 和 streaming 回归后，下面八个文件的
联合 CPU 测试为 `288 passed, 14 warnings`（2.06s）：
`test_spec_rhythm_tree_loop.py`、`test_specslo_roofline_loading.py`、
`test_measure_specslo_tree_roofline.py`、`test_specslo_streaming.py`、
`test_pearl_native.py`、`test_spec_rhythm_native.py`、
`test_spec_rhythm_dynamic_roof.py`、`test_spec_rhythm_roofline_profile.py`。
硬件数值、吞吐、SLO 和离线容量采样仍以各自真实运行产物为准，不由这个数量
推导功能全面完成。

### 最终功能冻结与 B 测量顺序

后续已完成生产文本生成所需的 V1/HTTP 生命周期和非贪心树采样，并在 Qwen3-0.6B
TP1 + Qwen3-32B TP3 的四卡 Graph 服务中验证：两个请求均动态接纳并完成，target
Graph capture/replay 为 2/59，最终 queue/inflight/failed 为 0/0/0。CPU spec decode
全量回归为 `880 passed, 13 skipped, 16 warnings`。详见
[功能冻结记录](specslo_functional_freeze_20260910_zh.md)。

用户所说的顺序是正确的：不能先拿手工 B 调吞吐，再反推论文功能有效。论文 §5.2
定义双槽执行与 guarded commit，紧接着的 §5.3 才定义通过 target-forward 实验获得
`B_roof(t)`。采样器最终采用每个 active batch 一个 fresh 进程、最大 context cache
常驻和 Graph 常驻复用；逐总候选预算枚举全部规范直方图。schema v3 绑定 FULL mask
容量、树拓扑、实际 attention backend 和四个核心源码 SHA，旧表强制失效。

功能冻结后得到的 B 表为：active/verify `8/4 -> [5,5,5,5,6,5]`、
`16/8 -> [9,9,9,9,9,10]`、`32/16 -> [17,17,17,17,17,17]`、
`64/32 -> [33,33,33,33,33,33]`，context 顺序均为
512/1024/1536/2048/2560/3072。线上 B=5 动态分配与 unprofiled tail fallback 的
真实 Graph 回归已经通过。
