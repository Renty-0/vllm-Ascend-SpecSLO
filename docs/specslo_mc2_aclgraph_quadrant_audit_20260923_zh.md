# TP3 MC2 在 ACLGraph 下的四象限审计与回归方案（2026-09-23）

## 1. 结论

本轮只做源码审计、门禁补强和 CPU 单元测试，没有启动 NPU，也没有运行占卡程序。

当前没有源码证据能够证明“图模式没有真正执行 MC2”。恰恰相反，图模式的正确语义是：MC2 的 Python adapter 只在 eager reference、图 capture 和 changed-input reference 阶段进入；稳定 replay 直接重放设备图，不再进入 Python。因此 measured window 内 MC2 Python counter 的增量为 0 是图常驻的必要现象，不能据此判断 MC2 未命中。

审计发现并修复了三个会让图模式结论不够可靠的 fail-open 点：

1. 旧 qualification profile 没有把模型的 RMSNorm epsilon 绑定到测量结果。Qwen3 当前是 `1e-6`，但若换成其他 epsilon，profile 仍可能误授权同一 shape。
2. production 已经通过 profile 判定 exact shape 后，custom op 在 capture 期异常时仍可能走 split fallback，使一个“MC2-on 图”实际上固化为普通 MatMul + all-reduce + RMSNorm。
3. `changed_input_replays=0` 是旧测量默认值；即使设为 1，唯一一次 replay 仍使用 capture 原输入。这样的结果不能证明 capture 图在 replay 时读取了更新后的 activation。

现在的行为是：epsilon 不一致、profile 没有至少两次且 delta 非零的 changed-input qualification、TP>1 无法解析 HCCL communicator，或 exact-qualified fused 调用异常，都会拒绝 production MC2；不会静默把 fallback 固化进图。

这仍不等于 NPU 四象限已经通过。最后一步必须在空闲设备上运行本文第 4 节的测试，特别是长 replay 的 HCCL 活性检查。

## 2. 逐项审计

### 2.1 Python dispatch counter

`mc2.py` 的 counter 是 worker 进程内的 host-side counter。PEARL 使用 `multiprocessing` 的 `spawn` 创建 worker，因此每个实验的新 worker 从干净 Python 全局状态开始，不继承控制器或上一象限的 counter。

四象限的正确预期是：

| 象限 | warmup/capture 后 MC2 counter | measured window counter delta |
|---|---:|---:|
| eager + off | 全 0 | 全 0 |
| eager + on | attempt = success > 0，fallback = exception = 0 | attempt = success > 0 |
| graph + off | 全 0 | 全 0 |
| graph + on | attempt = success > 0，fallback = exception = 0 | 全 0 |

最后一格的 0 表示 replay 没有回到 Python，而不是 MC2 没执行。必须同时满足：capture 前已有成功计数、graph replay delta 正确、图已 seal、measured window 没有新 capture/fallback，才能证明 resident graph 中包含 capture 时的 fused 路径。

### 2.2 profile 的 shape、数值与运行时绑定

production profile 现在绑定以下身份：

- exact `M/K/N`、dtype、weight format 和 `is_trans_b`；
- TP size、TP rank 到物理 NPU 的映射；
- CANN/HCCL 版本、`HCCL_DETERMINISTIC=true`、`HCCL_OP_EXPANSION_MODE=AIV`；
- 实际由 `ASCEND_CUSTOM_OPP_PATH` 解析到的 MC2 vendor payload 内容 SHA256，覆盖
  host OpAPI/tiling/proto 库、MC2 headers/sources、kernel manifest 与 kernel objects；
- adapter + MC2 C++/AscendC 源码 SHA256；
- target 模型 RMSNorm epsilon；
- 至少两次、delta 非零的 changed-input ACLGraph qualification。

因此源码发生变化后旧 profile 会按 source SHA fail closed，必须重新用真实层输入测量并聚合，不能把旧 profile 直接用于新的系统 A/B。

同理，只替换隔离 vendor 而不修改仓库源码时，旧 profile 现在也会按
`vendor_payload_sha256` fail closed。此前仅绑定仓库源码会把 production-v119、v122
或其他临时 kernel 错记为同一身份，虽然不直接改变执行时间，却会令 A/B 版本归因失真。

### 2.3 graph cache sealing

图 cache 的 seal 语义是严格的：seal 前每个 resident entry 必须完成 changed-input replay qualification；seal 后遇到未见 shape、disabled entry、unvalidated entry、容量不足或 eager fallback 都直接报错。正式计时窗口还必须验证：

- `aclgraph_sealed` 前后均为 1；
- entry/capture/capture-attempt counter 前后不变；
- replay delta 与 decode step 数完全一致；
- graph failure、shape/capacity fallback、disabled/unvalidated entry 全为 0。

### 2.4 capture/replay 输入

generic target graph capture 会 clone `input_ids`、positions、slot mapping、KV block table 等 resident buffer；每次普通 replay 前把当前输入 copy 到这些 buffer。首次真正变化的输入还会与 eager reference 比较，通过后才设置 `runtime_validated=True`。本轮又把 MC2 单算子 profile 的 activation changed-input 证明提升为 production 必需条件，避免模型图的 metadata 更新正确、但 custom operator 仍错误读取 capture-era activation 的盲点。

### 2.5 collective communicator

TP3 target 的 process group 在 worker 初始化时建立，直到 worker 退出才销毁；capture 前 eager reference 已先初始化 collective。`resolve_hccl_comm_name` 优先按 subgroup local rank 映射出的 global rank 查询，同时保留旧 torch-npu 的 local/no-argument 兼容路径。production TP>1 若无法得到非空 communicator，现已直接失败。

当前 custom op 显式使用 MTE HCCL server。源码注释记录了一个已经定位过的真实问题：改成 AICPU server 会使第一次 ACLGraph replay 卡死。因此不能在没有 NPU 证据时把它改回 AICPU。

Python 在设备图 replay 时不会重新解析 communicator，所以 host counter 不能单独证明 replay-time collective 活性。必须通过三 rank 同步完成、changed-input 数值一致和长 replay 无 hang 来闭环。

### 2.6 MC2 on/off 输出一致性

新 checker 会读取四份 target-only JSON，重新计算完整 `output_token_ids` 的 SHA256，而不是相信文件内的摘要；四象限必须逐 token 完全一致。同时会检查三张 target rank 的 route/counter 状态一致，避免只看 rank 0 掩盖 rank-divergent capture。

完整 token 一致性是系统生成回归，但它不能替代算子浮点门禁。因此还必须保留：真实层输入的 full-output/scaled-error qualification，以及 graph changed-input replay 对 eager reference 的检查。

### 2.7 跨层 profile 聚合门禁

聚合器现在会在 source SHA 检查之前执行可复用的 production measurement 验证，避免旧源码身份掩盖同一份证据里的真实 kernel 失败。任何 `max_scaled_* > 1`、不完整或退化的 changed-input gate、三 rank projection 非 bit-exact/证据不完整，以及输入行或跨层保守 envelope 的 p95 speedup 不达标，都会拒绝生成 production profile。结构身份和公共 gate 配置的跨文件不一致仍优先报告其准确字段。

## 3. 本轮代码变更

- `vllm_ascend/spec_decode/pearl/mc2.py`
  - profile qualification 增加 RMSNorm epsilon 和有效 changed-input replay 绑定；
  - 保留 exact shape、硬件、源码和 HCCL runtime binding。
- `vllm_ascend/spec_decode/pearl/native_model.py`
  - 把实际模型 epsilon 传入 qualification；
  - TP>1 communicator 解析失败时拒绝执行；
  - exact-qualified production 调用改为 `strict_fused=True`。
- `vllm_ascend/spec_decode/pearl/native_engine.py`
  - 模型装载阶段检查 profile TP size 和 target `rms_norm_eps`。
- `examples/measure_specslo_mc2.py`
  - changed-input qualification 默认 8 次；1 次或 delta=0 被拒绝；
  - profile 记录 `rms_norm_epsilon=1e-6`。
- `examples/aggregate_specslo_mc2_profiles.py`
  - 聚合时把 epsilon 作为 identity field，禁止跨 epsilon 合并。
- `examples/benchmark_nano_pearl_native_target_only.py`
  - JSON 记录 eager/ACLGraph 模式、resolved profile、profile SHA256、profile/model epsilon；
  - 保存计时窗口前后的逐 rank worker metrics。
- `examples/check_specslo_mc2_graph_quadrants.py`
  - 新增 CPU-only 四象限证据检查器。

对应 CPU 测试覆盖 profile fail-closed、communicator 缺失、strict fused、provenance、完整 token hash、四象限 route/counter 和 graph seal 门禁。

## 4. 最小严谨 NPU 测试方案

### 4.1 Phase 0：重新生成 profile

因为 profile 绑定源码 SHA，本轮修改后必须重新生成。attention `K=3072/N=5120` 和 down projection `K=8576/N=5120` 都应使用真实 Qwen3-32B 层输入，覆盖系统 replay histogram 中实际出现的每个 M；每个 row 使用默认的 8 次 changed-input replay。至少取早/中/晚三层，使用聚合器生成保守 envelope。

profile 的每个生产 row 必须满足：

- `changed_input_replays >= 2` 且 `changed_input_delta != 0`；
- 三 rank 的 full output numerical gate 通过；
- repeated p95 fused latency 通过 speedup gate；
- runtime binding 和当前 TP3 物理卡顺序完全一致。

### 4.2 Phase A：correctness/liveness（不计性能）

每个象限都启动全新 controller/worker，不复用 process group 或 graph：

1. eager + MC2 off；
2. eager + MC2 on；
3. ACLGraph + MC2 off；
4. ACLGraph + MC2 on。

共同条件：Qwen3-0.6B TP1 + Qwen3-32B TP3、相同物理卡顺序、相同 seed、相同真实且不重复的 prompt 列表、greedy + ignore-EOS、相同 B/P/T。不要只复制一句 prompt 100 次，否则 changed-input 覆盖弱。

先运行未 seal 的诊断轮：graph 两象限启用 `validate_every_replay`，至少 16 个 decode step，要求每次 replay 都与 eager reference 相符。该开关与 seal 互斥，所以诊断轮不能用于吞吐。

随后运行长活性轮：关闭 replay diagnostics，完成 warmup/changed-input qualification 后 seal，至少连续 1024 次 graph replay；三张 target rank 必须同步结束，无 hang、无 worker timeout、无 non-finite、无 capture/fallback/entry mutation。

### 4.3 Phase B：正式四象限性能与系统正确性

固定 B100/P100/T64 可先验证当前 M100 profile；如果真实服务目标是其他 batch，则必须先按 replay histogram 补齐 exact-M profile。每个象限至少重复 4 次，采用平衡顺序抵消温度和队列漂移，例如：

```text
round 1: eager-off, eager-on, graph-on, graph-off
round 2: graph-off, graph-on, eager-on, eager-off
round 3: eager-on, eager-off, graph-off, graph-on
round 4: graph-on, graph-off, eager-off, eager-on
```

每一 slot 启动前只检查目标卡是否空闲；不运行占卡程序，不驱逐其他用户进程。每一 slot 记录完整环境、模型元数据 hash、vendor package hash、profile hash、stdout/stderr 和前后 worker metrics。

四份单轮 JSON 生成后运行：

```bash
/root/miniconda3/envs/vllm-hust-dev/bin/python \
  examples/check_specslo_mc2_graph_quadrants.py \
  --eager-off  eager-off/result.json \
  --eager-on   eager-on/result.json \
  --graph-off  graph-off/result.json \
  --graph-on   graph-on/result.json \
  --expected-graph-on-resident-dispatches-per-rank 384 \
  --output     quadrant-gate.json
```

这里的 384 来自一张 resident entry 的三次 host forward（eager reference、capture、changed-input eager reference）× 每层 attention/down 两个 projection × 64 层。若模型、图 entry 数或启用的 projection 集合改变，必须按实际执行图重新推导，不能机械沿用 384。

### 4.4 四象限硬门禁

| 检查 | eager-off | eager-on | graph-off | graph-on |
|---|---|---|---|---|
| `enforce_eager` | true | true | false | false |
| seal | false | false | true | true |
| MC2 counter warm state | 0 | clean positive | 0 | clean positive |
| measured MC2 host delta | 0 | clean positive | 0 | 0 |
| graph replay delta | 0 | 0 | exactly T | exactly T |
| measured capture/entry delta | 0 | 0 | 0 | 0 |
| graph fallback/failure | 0 | 0 | 0 | 0 |
| 完整输出 token | 四象限逐 token 相同 |

性能报告必须分别给出：

- eager-on / eager-off：MC2 本身在非图路径的系统收益；
- graph-on / graph-off：MC2 在 resident ACLGraph 内的系统收益；
- graph-off / eager-off：图模式本身的收益；
- graph-on / eager-on：MC2 与图组合是否出现额外退化。

只有 graph-on 相对 graph-off 在多轮配对统计中稳定为正，并同时通过上述 correctness/liveness 门禁，才可以生产启用 MC2。单次 Raw 或 Goodput 波动不能作为结论。

## 5. 为什么“算子明显更快、系统收益很小”并不自动等于图 Bug

MC2 只替换 target 每层的两个局部片段，完整 step 仍包含 QKV、attention、gate-up、LM head、KV cache、调度和同步。即使局部 kernel 加速明显，Amdahl 定律也会把系统收益压小。另一方面，图 replay 本来就不增长 Python dispatch counter。

但在四象限完成前也不能排除组合 Bug。本文的门禁把两类原因分开：若 eager-on 有收益而 graph-on 没有，且 exact output/counter/seal/liveness 全通过，应进一步做设备 trace，比较图内 MC2 kernel latency、stream 排队和 HCCL/MTE overlap；若 graph-on 的 route 或 changed-input 门禁失败，则先修 correctness，不能继续解释性能。

## 6. CPU 验证结果

本轮两组相关测试合计：`256 passed`（105 个 MC2/profile/quadrant/aggregation 测试，151 个 graph replay/task/MC2 capture/sync 测试）；Ruff 和 `git diff --check` 通过。该结果只证明 host 侧门禁与证据检查器逻辑，不代表 NPU custom op、ACLGraph 或 HCCL 已完成四象限回归。
