# SpecSLO 功能冻结、B profiling 与线上消费验收

日期：2026-09-10

本文记录性能实验开始前的功能冻结状态。SpecSLO 指论文中的 SpecRhythm 实现，
不指普通 nano-PEARL。历史报告中手工设置的 `B=1`、`B=4` 只用于功能回归，不能
作为最终验证预算，也不能进入 Goodput 对比。

## 论文功能闭环

| 论文机制 | 当前实现 | 冻结证据 |
| --- | --- | --- |
| 双 batch 执行 | 请求固定归属两个逻辑 slot，target verify 与另一 slot 的 normal/eager draft 在不同 rank 组启动，完成后在受控边界汇合 | 真实 CANN 设备时间轴存在跨 NPU AICORE 交集；控制器与多轮集成回归通过 |
| Rolling Eager Continuation | continuation 记录 proposal ID、prefix epoch、父路径和 frontier；全接受才 promotion，拒绝或依赖变化时整体 invalidation | promotion、rejection、stale-last-row 故障注入和在线补入测试 |
| Individual Budget Shaping | 固定全局 roof 下先按 progress gap/urgency 满足紧急请求，再按 acceptance×draft confidence 分配剩余候选；选择保持祖先闭包 | 动态 roof 缩小、跨轮 ready 裁剪、实际 verify 输入不越 B 的回归 |
| 统一 draft workload | normal 与 eager 在同一深度合批，scratch KV 只在 guarded commit 后移动 | tree loop/KV 回滚与 compaction 回归 |
| packed-tree target | 只 materialize 实际选中的 root+candidate，使用 position、FULL ancestor mask、FIA 和全词表 head | eager/Graph 同形状逐元素一致；真实 Qwen3 TP1+TP3 smoke |
| 非贪心树采样 | target、draft 的 temperature/top-p/top-k 均进入采样；target 采样结果在 TP3 内广播，draft frontier/sibling 在 draft 组同步 | Qwen3 temperature=0.8 的四卡 Graph HTTP 回归通过 |
| 在线运行时 | 动态到达进入较轻 slot，支持完成替换、流式 guarded commit、断连/abort、失败传播和持久 worker 退出 | `qwen3-http-graph-nongreedy-lifecycle-v2.json`：2 次 live admission、59 次 Graph replay、最终 inflight=0 |
| V1/HTTP 边界 | `SpecSLOV1EngineCoreClient` 接收 `EngineCoreRequest` 并产生 `EngineCoreOutputs`；生产入口提供 OpenAI-compatible completions/chat、SSE、health、models | V1 协议、HTTP 生命周期 CPU 测试及上述真实四卡服务回归 |

V1 适配的项目范围是文本生成。多模态、LoRA、pooling、logprobs、structured
output 和 logits processors 不属于当前 SpecSLO 生成算法，入口会在入队前显式拒绝，
不会接受参数后改变语义。生产命令是 `examples/serve_specslo.py`；它复用 V1 的
request/output 协议，但没有声称所有 `vllm serve` 扩展接口都已支持。

## Ascend 算子与数值结论

- HCCL subgroup、TP1 draft + TP3 target、paged KV、FIA tree、ACLGraph capture/replay、
  tree KV compaction 和 Qwen3 非二次幂 head/MLP padding 均已在真实 NPU 跑通。
- TP3 MC2 使用 Qwen3-32B 的实际 `o_proj K=3072`、`down_proj K=8576`，覆盖
  `M=8/16/32/64/128/256`。ND/NZ 两组报告均没有合格形状：大部分存在非有限或
  超限误差，少数数值相同的形状也慢于分离路径。因此 production profile 不批准
  任何 TP3 fused shape，运行时使用已验证的 matmul + HCCL all-reduce。
- 同一 packed-tree FIA 输入下 Graph 与 eager 的 hidden/logits 逐元素相同；独立
  同形状 FIA A/B 的祖先、兄弟隔离也逐元素相同。dense/PA 与 FIA 使用不同 BF16
  内核归约顺序，全词表 raw logits 不满足逐位/严格 allclose，诊断报告保留该失败，
  不通过放宽阈值伪造成成功。生产正确性基准是同一 FIA 语义路径、有限值、分支隔离、
  采样/argmax 和 guarded commit，而不是要求不同注意力内核逐位一致。

## 功能冻结测试

- Spec decode CPU 全量：`880 passed, 13 skipped, 16 warnings`。
- 两卡 HCCL+FIA+ACLGraph：`1 passed`；两卡协议 HCCL：`1 passed`。
- Qwen3-0.6B TP1 + Qwen3-32B TP3 四卡非贪心 Graph 服务：通过；冷启动
  192.45s，在线生成 6.68s，Graph capture/replay=2/59，live admission=2，
  completed/failed/inflight=2/0/0。
- TP3 MC2：完成真实资格验证，结论为全部拒绝，不能启用。

## 最终 B 的实验协议

论文 §5.2 描述双槽运行循环；`B_roof` 的离线实验定义位于 §5.3。工程冻结后，
`examples/measure_specslo_tree_roofline.py` 才允许生成 B 证据：

1. 使用最终 Qwen3-32B TP3、Graph、packed-tree FIA、完整有限值保护及相同源码；
2. active batch 测 8/16/32/64，对应一个 verify slot 的物理请求行为 4/8/16/32；
3. 根据 HumanEval/Alpaca/CNN-DM 的实际 Qwen3 token 长度和 256-token decode，测
   context 上界 512/1024/1536/2048/2560/3072；
4. 对每个 `(active batch, physical rows, context)` 从每请求至少一个候选开始，按
   总候选数逐一递增；每个点排除 capture/warmup 后重复测量，并取三个 target rank
   每轮最慢值的 P95；
5. AR 对照使用完整 active batch（8/16/32/64 个单 token query），tree verify 使用
   当前逻辑 slot 的 4/8/16/32 个物理请求行；这是论文 §5.3 的标准 batch decode
   对照，不能把 AR 错缩成半批。选择验证 P95 不超过对应 AR P95 的
   `1 + epsilon` 的最大连续候选预算；第一次越界后停止，不能跳过坏点选择后续偶然快点；
6. 每个通过的总预算必须覆盖全部 permutation-equivalent canonical candidate-count
   histogram；第一次失败只需一个实测反例即可停止；
7. 输出 schema v3，绑定 model、TP、hardware、Graph/eager、max model length、树宽深、
   FIA/PA provenance、四个核心源码 SHA 和原始 latency evidence。缺 key、物理行不匹配、
   Graph 回退、设备不匹配或旧 schema 都不能进入投机路径。

采样时发现 CANN 自定义 FIA graph-task 在同一进程反复 `reset -> capture` 后会令 TP3
stream 停滞。最终协议不模拟这种非生产生命周期：每个 active batch 使用 fresh
四卡进程，一次分配最大 context cache，按 context 从大到小测量，AR、causal FIA 和
FULL-tree FIA graph 都常驻复用；四份 complete raw 最后由聚合器合并。旧 v1/v2
profile 已由生产入口强制失效。最终 B 是随 active batch/context 变化的实测表。

## 最终实测 B（Qwen3-0.6B TP1 + Qwen3-32B TP3）

固定条件：Ascend 910B2、Graph、tree width/depth=2/2、`epsilon=10%`（项目明确选择；
论文未给出数值）、P95、每 shape 3 次 warmup + 10 次计时 replay。B 是一个 verify
slot 内的总 candidate token 预算，不是每请求 gamma。

| active batch / verify rows | ctx 512 | 1024 | 1536 | 2048 | 2560 | 3072 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 8 / 4 | 5 | 5 | 5 | 5 | 6 | 5 |
| 16 / 8 | 9 | 9 | 9 | 9 | 9 | 10 |
| 32 / 16 | 17 | 17 | 17 | 17 | 17 | 17 |
| 64 / 32 | 33 | 33 | 33 | 33 | 33 | 33 |

聚合结果：`qwen3-tp3-graph-Broof-schema3-resident-eps10.json`；四个 batch 的 raw
文件位于同一结果目录，均为 `complete`，共 24 个 evidence key，无 target-only zero
key。线上回归 `qwen3-profile-online-graph-b8-tail-fallback.json` 已通过：实测键 `8:1`
给出 B=5，调度器实际分配 `[2,1,1,1]`，candidate sum=5，TP3 三 rank 均命中
causal FIA ACLGraph；随后未采样 tail 执行 17 轮 target-only fallback，流式 token 与
完成事件一致。含 sibling 的 FULL-tree `[1,3]` Graph/eager 同后端 hidden/logits
零误差，capture/replay=1/2。

完整口径、原始文件 SHA256 和线上消费证据见
[最终 B_roof 实测报告](specslo_final_b_roofline_20260910_zh.md)。

## 冻结后才能进行的工作

最终 B 表生成后，才进行 RPS=2/4、batch=8/16/32/64、三类请求 6:2:2 的
SpecSLO 与 TP4 target-only 同口径 decode-stage TPOT/Goodput 验收和性能调优。
任何旧固定 B、普通 PEARL、冷启动 E2E 吞吐或不同 TPOT 分母的数据都不得混入该结论。
