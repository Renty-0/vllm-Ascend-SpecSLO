# atc26v0 与 SpecSLO 功能审计

更新时间：2026-09-10

> 首次复核见 [第五章及核心功能审计](specslo_section5_audit_20260910_zh.md)。
> 首次复核发现了同一步 overlap、动态准入、跨 rank confidence 一致性、
> guarded commit 顺序和实际 verify 预算的缺口。下文历史组件清单与短 smoke
> 不能解释为 5.1/5.2 已完整实现，也不能概括为“只剩性能验证”。

> 同日后续：[核心闭环修复与回归](specslo_core_regression_20260910_zh.md) 记录了这些问题
> 的修复及真实设备重叠证据。上段保留首次审计结论；当前进展与失败项以新记录为准，
> 不把“代码已修改”自动视作“硬件验收通过”。
> 当前冻结版已再次通过 8/8 四卡在线 source/跨页、eager/Graph、B1/B4 回归；
> 严格表实测、独立数值及新版设备时间轴分别验收，Goodput 1.3× 尚未达成。
> 2026-09-10 最终功能冻结及 B 实验边界见
> [SpecSLO 功能冻结与 B profiling 前置验收](specslo_functional_freeze_20260910_zh.md)。
> 该记录取代下文“V1 生命周期、随机树采样尚未实现”的历史状态，但保留旧文字作为
> 修复轨迹。任何旧 B1/B4 仍只属于功能 smoke。
> 最新状态：文本生成范围的 5.1/5.2 已完成；功能冻结后按 5.3 得到 schema v3
> Qwen3 TP1+TP3 Graph 实测 B 表，并完成线上动态分配和未覆盖 tail fallback 回归。
> RPS/Goodput 1.3× 是下一阶段性能验收，不属于已经完成的功能结论。

## 审计范围

本审计固定检查了：

- `/root/data/reference-repos/atc26v0` 的当前提交 `6b1cebf`；
- `/root/data/atc26-paper1664.pdf` 中 SpecRhythm 的双批次、rolling eager 和
  individual budget shaping 机制；
- `vllm-ascend-hust` 的 native PEARL、V1 tree、ACLGraph 和 Ascend custom ops。

参考仓库的 README 功能列表不能等同于代码完成度。尤其是 `stspec_plan.py`、
`stspec_pipeline.py`、`stspec_mailbox*.py`、`stspec_kv_sync.py` 明确写着
scaffold/probe；其测试验证的是诊断边界和错误分类，不是跨进程 target forward、
KV 写入和 guarded commit 的生产实现。

## 功能对照

| 功能 | atc26v0 当前状态 | Ascend-SpecSLO 状态 | 验证方式 |
| --- | --- | --- | --- |
| Draft/target 分离与异构 TP | nano-PEARL 运行时 | native HCCL worker 已实现 | PEARL native 单测、TP1+TP3 smoke |
| 个体候选预算 | PEARL 的固定 gamma 只作为参考 | 实测全局 `B_roof` 下按 acceptance EMA + SLO urgency 动态分配；`gamma` 仅是单请求候选上限，不是 B | `test_spec_rhythm_native.py` / strict roofline 线上消费回归 |
| 双批次节奏 | ST-Spec probe | native SpecRhythm pipeline 已执行 draft/target 两角色 | native engine profile counters |
| rolling eager continuation | probe 元数据 | proposal lifecycle、full-accept promotion、reject invalidation 已执行 | controller/native 单测 |
| 非二次幂 TP | padding 实验功能 | Q/KV/MLP/vocab padding 与逻辑裁剪已实现 | config/weight loader、NPU smoke |
| tree verification | V1/PEARL 基础树 | fixed-shape tree budget、draft spine/top-k proposal、target ancestor-mask forward、设备 verifier、proposal-id rollback/compaction 已接入 SpecRhythm 主循环；固定 shape 可走 target ACLGraph，动态/捕获失败回退 eager | tree、tree-kv、coordinator、native graph、native SpecRhythm 单测；910B2 短 smoke 已通过，长 workload/多模型矩阵仍待验收 |
| CUDA Graph/FlashAttention | CUDA-only | NPUGraph/ACLGraph + FIA/paged attention | graph runtime guard、能力脚本 |
| HCCL mailbox | 不适用于普通 vLLM scheduler | native envelope 带 proposal/request/epoch/width/confidence | HCCL protocol 单测、NPU smoke |
| PEARL-2 distillation | 没有完整训练器 | acceptance-weighted KL/CE、JSONL trace collator、teacher rollout、梯度裁剪、checkpoint API 和训练示例已完成 | `test_pearl_distill.py` |
| draft temperature | README TODO | `NativeSamplingParams.draft_temperature` 已实现；非零 draft 自动避开 draft graph | native engine 静态检查与 API |
| continuous batching/chunked prefill | README TODO | native admission、prefill chunk、完成替换、抢占已实现 | native engine counters/profile |

## 算子审计

| 算子/后端能力 | 代码入口 | 当前状态 |
| --- | --- | --- |
| `npu_fused_infer_attention_score` | `pearl/native_graph.py`、`native_model.py` | native eager/graph 已接入；能力脚本检查导出 |
| `_npu_paged_attention` | `pearl/native_graph.py` | native paged path 已接入；无 FIA 时 fallback |
| `_npu_reshape_and_cache` | `pearl/native_model.py` / `DeviceOperator` | 128-token page 写入已接入 |
| `npu_rotary_embedding` | `pearl/native_model.py` | Qwen2/Llama production RoPE fallback 已接入 |
| `qkv_rmsnorm_rope` | `pearl/native_model.py` / `DeviceOperator` | Qwen3 BF16 条件融合已接入 |
| `matmul_allreduce_add_rmsnorm` | `csrc/*mc2*`、编译 fusion pass、`pearl/mc2.py` | custom op、meta 和 fallback 已有；native PEARL 默认不强制启用，避免未验证 CANN 上改变数值/稳定性 |
| HCCL subgroup/all-reduce | `pearl/topology.py`、`native_engine.py` | draft/target/verification/correction group 已接入 |
| tree KV compaction | `spec_decode/tree_kv.py` | 接受路径 compaction plan、NPU scatter 更新与 CPU fallback 已接入 |

运行 `examples/check_specslo_capabilities.py --tp-size 3` 可在目标容器中输出
ACLGraph、FIA、paged attention、RoPE、tree 和 MC2 的实际导出状态。能力为 false
时，代码仍会使用显式 fallback，不会静默调用不存在的算子。

## 与论文机制的对应关系

SpecRhythm 的双 batch 和 rolling eager 位于
`pearl/native_engine.py::_generate_spec_rhythm_decode`：target 验证当前 ready
proposal，同时 draft 生成另一 home batch 的 proposal；完整接受后将 staged eager
提升为 ready，拒绝则使其失效。budget shaper 依据 progress gap 和
`acceptance_ema * draft_confidence_ema` 排序，并使用一个全局
`verification_roof`，保证普通与 eager proposal 的总候选数不会超出 target 单步
预算；显式 `verification_budget=B` 时，先完成 urgency 分层再填充剩余候选容量，
并通过 `unused_verification_tokens` 暴露由于 per-request/draft cap 导致的未使用槽位。

树状路径由 `pearl/tree.py` 和 native engine 提供：

1. `SpecRhythmTreeCoordinator` 将每个 request 的标量预算转换成 width/depth；
2. `select_tree_candidates` 在评分选择时闭包包含祖先，避免发送无父节点的分支；
3. `build_tree_attention_mask` 生成 CANN/V1 约定的 blocked=True mask；
4. 目标侧提供 native tree forward：唯一 cache position、显式 ancestor mask，并保留
   每个候选节点的 target 输出；设备 verifier 沿实际接受分支动态选择 frontier/bonus，
   接受路径可生成 KV compaction plan。

这使策略和设备树基础设施可以组合。`spec_rhythm_tree_width/depth > 1` 时，native
SpecRhythm 主循环会：

1. 从同一个固定 `B` 预算生成 request-local `TreeSpeculationPlan`；
2. draft 沿 spine 逐层产生 top-k sibling，并用 tree mask 写入完整固定 shape 的 KV；
3. HCCL 只交换 active candidate nodes，target 对固定 shape 做 forward，按
   `candidate_budget` 截取 target predictions；
4. device verifier 返回 accepted path/bonus，target 和 draft 两侧按 proposal id 做
   rollback/promotion，并调用 tree KV compaction；树 eager 额外记录父树主干和 draft
   frontier，只有两者都与 target verdict 对齐才 promotion。

固定 shape 的 target forward 会复用 `NativeACLGraphRunner.run_target_greedy`，capture/replay
失败时回退 eager；动态预算只影响 active prefix，不改变 graph shape。

## 明确未宣称完成的内容

以下项目不是代码缺失，而是必须在指定 CANN/固件/驱动和真实工作负载上继续验证：

- MC2 custom kernel 是否稳定超过生产 `matmul + HCCL all-reduce`，以及其 TP3
  的数值误差和 graph replay 行为；
- tree proposal 的 native draft/target、固定 shape ACLGraph、分支 KV 提交和拒绝后
  回滚已经接入代码；Qwen2.5-0.5B + Qwen2.5-14B、TP1+TP3 的 910B2 短 smoke
  已通过数值生成、tree verifier、KV compaction、HCCL 双组播和 ACLGraph replay。
  仍需在每个支持模型、长上下文、多请求和生产 workload 上完成端到端吞吐回归；
  短 smoke 不等同于完整性能验收；
- 文本生成范围的 V1 `EngineCoreRequest`/`EngineCoreOutputs` 适配、持久双模型
  HCCL worker 和 OpenAI-compatible HTTP/SSE 生命周期已完成。多模态、LoRA、
  pooling、logprobs、structured output 和自定义 logits processors 不属于
  当前 SpecSLO 文本生成算法范围，生产入口会显式拒绝，不会静默改变语义；
- PEARL-2 的 teacher rollout、JSONL 数据管线、蒸馏 loss、optimizer step 与 checkpoint
  格式已经可运行，仍需大规模训练和权重质量回归；
- CANN 各版本动态 graph bucket 的内存上限、长上下文和多租户抢占矩阵。

这些限制已在能力脚本和主工作记录中写明。任何性能报告都必须注明模型、TP、
batch、gamma、是否 warm-up、CANN/驱动版本和端到端计时口径。

## 功能冻结追加（2026-09-10）

- 文本生成范围内的 V1 `EngineCoreRequest`/`EngineCoreOutputs` 适配、持久双模型 worker、
  OpenAI-compatible HTTP/SSE、动态 admission、abort、failure propagation 和 shutdown
  已完成；真实 Qwen3 四卡服务回归最终 `inflight_requests=0`。
- 树 verifier 已支持非贪心 target/draft sampling；temperature/top-p/top-k 不再被拒绝，
  Qwen3 temperature=0.8 Graph 实机回归完成。
- TP3 MC2 已完成真实矩阵资格验证，结果是无任何形状同时满足数值与性能条件，故生产
  路径明确回退；这项属于“验证后否决”，不再是“尚未验证”。
- strict roofline 升级为 schema v3，绑定 max model length、树宽深、实际算子 backend
  与四个核心源码 SHA。历史 B1/B4/v1/v2 表均不可作为最终 B。功能冻结后的最终
  B 已按论文 §5.3 实测：active/verify 8/4、16/8、32/16、64/32 在六个 512-token
  context 桶分别得到 `5/5/5/5/6/5`、`9/9/9/9/9/10`、全 17、全 33。
