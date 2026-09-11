# vLLM-Ascend-SpecSLO 工作记录

更新时间：2026-09-10

> 2026-09-10 复核更正：第 6.5 节“核心机制已经闭合”的结论超出了短 smoke
> 能证明的范围。当前明确存在树运行循环的功能缺口，详见
> [第五章及核心功能审计](specslo_section5_audit_20260910_zh.md)。历史记录保留，
> 最新完成度以该审计为准。

> 同日实现进展：上述缺项已开始逐项修复，最新软件变更、成功/失败的 NPU 证据和
> 尚待验收内容见 [核心闭环回归记录](specslo_core_regression_20260910_zh.md)。
> 历史 PEARL 性能不能替代 SpecSLO 验收；本轮仍未宣称 1.3× Goodput 达标。

## 1. 项目范围

本项目的目标是在昇腾 NPU 上实现论文和参考实现中的 SpecSLO 思路，
并复用此前 nano-PEARL 迁移中形成的跨模型投机解码、KV cache、图执行和
性能分析基础设施。代码以 `vllm-ascend-hust` 为主仓库；vLLM 核心调度器
的配套改动保存在 `patches/vllm-hust/`，因为核心仓库和 Ascend 插件是两个
独立的 Python 包，不能简单地把两个 Git 历史合并成一个安装包。

参考材料：

- ATC 论文：`/root/data/atc26-paper1664.pdf`。
- 参考代码：`rzwang22/atc26v0`，复现时固定在提交 `6b1cebf`。
- 相关上游：nano-PEARL 及 vLLM/vLLM-Ascend 的对应实现。

论文中的核心机制是双批次重叠执行：一批请求由 draft 推进，另一批请求由
target 验证；调度器按照 SLO 紧迫度、接受率和设备 roofline 预算动态选择
继续 draft、交给 target 验证或回退到自回归路径。验证成功时提升已接受的
前缀，失败时只保留接受前缀并使被拒绝后缀失效，同时允许新请求进入空出的
位置。

## 2. 已完成的迁移工作

### 2.1 nano-PEARL 原生运行时

- 实现 `PEARLConfig`、`PEARLEngine`、采样参数、请求队列和 `generate`、
  `AR_generate`、`bench_generate` 接口。
- draft/target worker 在同一个 HCCL world 中启动，使用独立的模型组和
  verification/correction 通信组，支持 `1TP + 2/3TP` 等异构 TP 布局。
- 支持 Qwen2、Qwen3、Llama 的模型加载和 safetensors 权重路径。
- target logits 在比较和采样前按 draft vocabulary 裁剪，覆盖 Qwen2.5
  0.5B/14B 这类词表大小不同的组合。
- 完成 pre-verify、gamma draft、target verify、rollback/correction 的
  原生执行链路，并记录每个请求的接受 token 数和 target 修正 token。

### 2.2 Ascend/NPU 适配

- 使用 HCCL 替代 NCCL，统一 world、rank、device 和 subgroup 拓扑，加入
  proposal id、request id、home rank、epoch、width、confidence 的自描述
  envelope，避免旧消息、错位请求和变宽 proposal 被误消费。
- 使用 CANN fused-infer attention、paged attention、FIA 和
  `_npu_reshape_and_cache` 写入 KV；KV page 固定为昇腾后端要求的 128 token，
  支持懒分配、完整 page 前缀复用、slot/block table 和请求回收。
- 使用 `torch.npu.NPUGraph`/ACLGraph；统一 draft shape，target 验证行数按
  bucket 规范化，图捕获达到 `max_aclgraph_entries` 后对新形状回退 eager。
- 对 Qwen3 BF16 标准 RoPE 自动复用生产 `qkv_rmsnorm_rope`，Qwen2/Llama
  unfused 路径复用生产 `npu_rotary_embedding`，不支持的架构保留 fallback。
- 针对 TP3 增加任务队列、HCCL AIV expansion 和确定性归约的可控默认值；
  保留 TP2/TP4 的显式环境变量覆盖。
- 增加 CPU/NUMA/IRQ affinity、prefill chunk、连续批处理、完成行 padding、
  full/half graph bucket 和 SLO/分阶段 profiling 统计。
- 保留 TP3 MC2、融合 FFN、QKV NZ、权重预取、通信 overlap 等实验入口，便于
  后续在真实 CANN 版本上替换成经过验证的 kernel。

### 2.3 vLLM 核心层配套

- 在 vLLM V1 增加 tree verification、SpecRhythm 元数据、词表映射、请求
  字段、调度器输出和 speculative 配置字段。
- 对通用 vLLM 路径增加树结构和节奏控制的数据结构及单元测试；真正的跨
  draft/target HCCL 执行仍由 Ascend 原生 PEARL engine 负责。
- 这些改动以可审阅补丁形式放在 `patches/vllm-hust/`，应用目标是
  `/root/data/vllm-hust` 对应的 `codex/nano-pearl-ascend-migration` 分支。

### 2.4 文档、示例和测试

- 提供 native target-only、native speculative、串行 speculative、连续批处理、
  profiling、TP3 collectives/MC2 等 benchmark 示例。
- 记录 graph bucket、paged KV、生产 RoPE、prefill chunk、warmup 和
  `elapsed_time`（包含 prefill + generation）的实验语义。
- 增加 PEARL、SpecRhythm、树 KV、树设备索引、ACLGraph、EAGLE proposer、
  vocab crop 和 AscendC 配置测试。

## 3. 无法直接从 CUDA/vLLM 搬运的部分

下面这些不是简单改 import 或后端名称就能完成的迁移点，因此做了专门设计。

| CUDA/上游假设 | 昇腾上的问题 | 当前设计 |
| --- | --- | --- |
| CUDA Graph、FlashAttention、Triton KV 写入 | CUDA kernel、stream/event 和 graph update API 与 CANN 不兼容 | NPUGraph/ACLGraph + FIA/PA + CANN cache op，并按 shape bucket 捕获 |
| NCCL 和跨 engine CUDA stream | HCCL rank/group、通信时序和设备可见性不同 | 单一 HCCL world、显式 subgroup、带 epoch 的 envelope 和同步点 |
| GPU block table/KV layout | vLLM-Ascend 的 page 和 slot 约束不同，page size 不是任意值 | 128-token page pool、prefix reuse、请求级 table 适配不同 attention backend |
| 一个 vLLM scheduler 直接拥有两套 worker | 通用 V1 worker 没有跨模型 HCCL group 边界 | 通用路径只传 metadata；可执行的双模型流程放在 native PEARL engine |
| 任意 TP 和均匀 head 切分 | Q/KV head、MLP、词表不能被 3 整除时 shape 不合法或通信浪费 | 参数/head/MLP/vocab padding、逻辑行裁剪；接受率和 padding 成本需单独评估 |
| 动态 shape graph | CANN graph capture 需要稳定 shape，频繁 capture 会耗尽内存 | 逻辑 proposal width 与物理 graph shape 分离，target 行数 bucket 化，超限 eager fallback |
| CUDA 异步输出和 KV 提交 | 拒绝后缀可能已写入设备 cache，旧结果可能晚到 | proposal/epoch 校验、commit/rollback 边界、失效后缀清理和状态快照 |
| PEARL-2 训练/蒸馏流水线 | 参考仓库仅有静态训练接口，缺少 Ascend 训练 kernel/数据管线 | 已补齐 teacher rollout、JSONL trace、acceptance-weighted loss、训练 step 与 checkpoint；真实训练质量仍需验证 |

## 4. 已完成的自研设计

1. **原生双模型拓扑**：在一个进程组内隔离 draft/target TP，统一设备发现、
   HCCL 初始化、verification/correction group 和 rank 映射。
2. **自描述通信协议**：每轮携带 request、proposal、epoch、逻辑宽度和置信度，
   接收端拒绝过期、错请求和不匹配宽度，解决异步 HCCL 下的状态污染。
3. **SpecRhythm 控制面**：记录接受率 EMA、SLO urgency、draft/verify 成本，
   按预算和队列状态选择 gamma、继续 draft 或 target 验证。
4. **设备侧树基础设施**：实现树节点索引、祖先路径、唯一 KV 位置映射、native
   target tree forward、显式 ancestor mask、验证结果回写和接受路径 KV compaction
   plan；通用 V1 tree verification 已有单元测试。
5. **Ascend graph/cache 适配**：物理 graph bucket、128-token page、完成行
   padding、prefix reuse、生产 RoPE 和 fused attention fallback 已接入 native
   runtime。
6. **动态离线批处理**：prefill queue、chunked prefill、连续 admission、请求
   完成替换和 SLO/goodput 统计已实现，适合论文三类 workload 和离线回归。

### 4.1 论文一致的 SpecRhythm workload

已将论文 Section 6.1 的三类请求单独落地为可复用 manifest，不再把 GSM8K
或 ShareGPT 作为 SpecRhythm 的默认数据源：

| 类别 | 数据集 | SLO TPOT | 混合比例 |
| --- | --- | ---: | ---: |
| tight / coding | HumanEval | 40 ms | 6 |
| normal / chat | Stanford Alpaca | 50 ms | 2 |
| loose / summarization | CNN/DailyMail | 150 ms | 2 |

数据文件分别为 `reference-repos/atc26v0/benchmark/data/HumanEval.jsonl`、
`/data/datasets/specslo/stanford_alpaca/alpaca_data.json` 和
`reference-repos/atc26v0/benchmark/data/CNNDM.jsonl`。生成器为
`examples/prepare_specslo_workload.py`，对 10 的倍数使用严格的 6:2:2 配额；
其他请求总数采用最大余数法得到最近整数配额，并在 `.meta.json` 保存实际配额、
SLO、gamma、seed 和到达源。已生成严格 36/12/12 配额的
`/root/data/specslo-workloads/rps4_mix6-2-2_p60_t256_exact.jsonl`；64 请求
测试对应的实际配额为 38/13/13（因整数约束无法严格等于 6:2:2）。
没有论文生产 trace 时使用固定 seed 的 Poisson 到达；这会标记为
`synthetic_poisson`，不能冒充论文的生产时间戳。

验收矩阵固定为 RPS 2 和 4、batch 8 到 64；每个请求保留
`arrival_offset_sec`、类别、SLO 和 gamma。TPOT 只从真正进入 decode 后开始计时，
admission wait 单独作为调度债务参与 urgency，不计入 decode TPOT。

## 5. 仍需设计或验证的内容

- **TP3 MC2 的实机验证**：custom AscendC MC2、meta、编译 fusion pass、communicator
  解析和 `pearl/mc2.py` dispatch/fallback 已完成；仍没有在目标 CANN/固件版本上
  证明稳定地超过生产 all-reduce + matmul。
- **SpecRhythm 与树状投机解码的实机回归**：策略、native target forward、唯一
  cache position、KV compaction plan 和 rejection-safe 数据结构已完成；仍需在真实
  Qwen/Llama 模型上压测分支提交、回滚和数值一致性。
- **通用 vLLM 服务路径**：`SpecRhythmScheduler` 与
  `PearlDualModelScheduler` 已提供 admission、双 batch 并行回调、preempt/reactivate、
  global roofline 和 verification commit；仍需绑定上游 V1 scheduler 的 worker 生命周期。
- **PEARL-2 训练/蒸馏**：`pearl/distill.py` 已提供 teacher rollout、JSONL trace
  loader/collator、acceptance-weighted KL/CE、梯度裁剪、optimizer step 和 checkpoint；
  仍需大规模训练、teacher/student 权重产出和质量回归。
- **生产级动态 shape graph（运行时 guard 已完成，版本矩阵待验证）**：native
  graph 已按 shape bucket 捕获、回放、首轮 eager 对照和容量 fallback；仍需按
  CANN 版本和真实到达分布建立 capture/replay 兼容矩阵。
- **性能和稳定性回归**：需在固定 NPU 型号、驱动/CANN、模型量化配置下重新测
  batch、gamma、接受率、端到端延迟、SLO goodput、长上下文和多租户抢占。
- **扩展覆盖面**：多模态、LoRA、structured output、更多 tokenizer/vocab 映射
  以及异常退出后的 HCCL 资源回收仍需补齐。

## 6. 验证记录

本次迁移收尾执行了以下检查：

- Ascend 重点单元测试：`223 passed, 13 skipped, 14 warnings`。
- bridge/vocab 单元测试：`28 passed, 14 warnings`。
- vLLM 核心 tree/SpecRhythm 测试：`10 passed, 14 warnings`。
- `compileall` 和 `git diff --check` 均通过。
- NPU smoke：Qwen2.5 0.5B + 14B、TP1+TP3、SpecRhythm 单请求和动态 admission
  均完成生成，并观察到 eager promotion/接受计数。
- 参考 atc26v0 测试：`109 passed, 1 skipped, 1 failed`。唯一失败项是
  `test_real_probe_only_fields_are_not_seeded_in_base_trace_defaults`，原因是
  参考仓库当前测试期望与其 trace 默认字段实现不一致；该文件未被本次 Ascend
  迁移修改，也不影响 NPU 推理运行时。

本轮收尾新增：

- SpecRhythm roofline、PEARL-2 distillation（含 JSONL loader/collator）、tree
  coordinator、scheduler 和 MC2 fallback 单元测试：相关回归合计 `140 passed`。
- `examples/check_specslo_capabilities.py --device cpu --tp-size 3` 可运行并输出
  JSON 能力矩阵；在 NPU 上会额外报告 ACLGraph、FIA、paged attention、RoPE 和
  MC2 custom op 的导出状态。
- `compileall` 和 `git diff --check` 通过。

本次继续实现并回归：

- native tree target forward：每个分支使用唯一 cache position，显式 ancestor mask
  强制走 dense correctness path；增加设备侧 batch verifier、bonus token 分离和
  接受路径 KV compaction/rollback API。
- 通用 `PearlDualModelScheduler`：基于 SpecRhythm schedule 并行提交 draft/target
  worker 回调，并提供 verification commit 生命周期入口。
- PEARL-2 teacher rollout：支持批量 prompt、temperature、EOS 截断、padding mask，
  以及 `collect_pearl_teacher_trace.py` JSONL CLI。
- TP residual MC2：增加 HCCL communicator 名称兼容探测、`enable_mc2` 配置和
  fused-op 失败自动 fallback；默认关闭，需在目标 CANN 上显式开启验证。
- 本轮相关单测为 `132 passed, 14 warnings`；完整 `tests/ut/spec_decode` 还受容器
  缺少 `numba` 影响，ngram proposer 收集失败，其余可收集用例通过。

参考仓库逐文件审计见
[`atc26v0_feature_audit_zh.md`](atc26v0_feature_audit_zh.md)。该审计把上游
scaffold/probe 与真正可执行的 nano-PEARL 功能分开记录，避免把探针通过误报为
生产迁移完成。

本次只做功能迁移和验证，没有宣称新的吞吐提升；后续性能报告必须注明模型、
TP、batch、gamma、warmup、CANN/驱动版本和端到端计时口径。

### 6.1 SpecRhythm 目标回归（2026-09-08）

本轮先用 40 请求、`max_tokens=32` 的短 manifest 验证端到端链路，模型为
Qwen3-0.6B draft（TP1）+ Qwen3-32B target（TP3），ACLGraph、paged attention
均开启，四张物理卡映射为 `0,3,6,7`。结果如下，均为冷启动、包含 prefill 和
generation 的 e2e wall time：

| 配置 | e2e 吞吐 | 接受率 | SLO 达标率 | e2e Goodput |
| --- | ---: | ---: | ---: | ---: |
| PEARL gamma=4，prefill chunk=8 | 94.12 tok/s | 77.17% | 47.5% (19/40) | 44.71 tok/s |
| PEARL gamma=4，prefill chunk=40 | 94.08 tok/s | 77.38% | 60.0% (24/40) | 56.45 tok/s |
| PEARL gamma=2，prefill chunk=8 | 93.53 tok/s | 83.73% | 27.5% (11/40) | 25.72 tok/s |
| vLLM-Ascend target-only TP4 | 247.11 tok/s | - | 未在线计时 | - |

target-only 这次通过 `VLLM_ENABLE_V1_MULTIPROCESSING=0` 和单线程环境变量
绕过容器已有的 PyTorch 多进程线程池崩溃；它以离线一次性提交所有 prompt，
没有消费 arrival timestamp，也没有逐请求 SLO 计时，因此只能作为吞吐参考，
不能用于宣称 1.3x Goodput。当前结果尚未达到图片中的 80% SLO attainment 和
1.3x Goodput 目标；主要瓶颈是 TP3 target compute、每轮 verdict/broadcast、
以及 draft-target exchange，ACLGraph capture/replay 本身未失败。

本轮已验证的可复现改动：

- 放宽 prefill chunk 上限到连续队列容量，允许一次 packed prefill，同时保持
  decode graph 的 batch bucket 不变；短 workload 的 SLO 达标率由 47.5% 提升到 60%。
- 将 admission wait 从 `decode_elapsed_ms` 中分离，wait 仍计入 urgency 的调度债务，
  但不再污染 decode TPOT；这使 SLO 统计与论文定义一致。
- 增加 `--spec-rhythm-auto-eager-tokens`，按请求 gamma 限制 eager 预算，并在结果
  中输出按类别的 attainment/goodput/mean TPOT。

后续回归必须使用同一份 manifest、同一到达时间、同一 max token 和相同计时口径，
分别跑 RPS 2/4 与 batch 8/16/32/64；在 baseline 也接入在线 arrival/SLO 计时前，
只报告吞吐，不报告伪 Goodput 加速比。

### 6.2 图片验收目标与 2026-09-09 回归

图片给出的 SpecRhythm 验收条件为：tight、normal、loose 三类请求的 TPOT 上限
分别为 40/50/150 ms；同一 batch 内请求比例为 6:2:2；RPS 在 2 和 4 之间变化；
batch 覆盖 8 和 64；相对原始 vLLM-Ascend 的 SpecRhythm 达标率至少 80%，
Goodput 至少 1.3 倍。当前实现使用 Qwen3-0.6B draft（TP1）+ Qwen3-32B
target（TP3），baseline 为 Qwen3-32B target-only（TP4），物理卡为 Ascend
910B2，计时包含 prefill 和 generation，warmup 不计入 e2e。

本轮使用论文三类 workload 的 RPS4、B64、64 请求长输出 manifest，结果如下；
该 manifest 的实际类别配额为 coding/chat/summarization=38/13/13，严格比例
回归应改用 60 请求的精确 manifest。
baseline 文件为 `/root/data/specslo-workloads/smoke-baseline-rps4-b64-t256.json`，
PEARL 文件均为 `/root/data/specslo-workloads/` 下对应 JSON：

| 配置 | e2e 吞吐 (tok/s) | 接受率 | SLO 达标率 | e2e Goodput (tok/s) | 相对 baseline Goodput |
| --- | ---: | ---: | ---: | ---: | ---: |
| 原始 vLLM-Ascend target-only TP4 | 629.53 | - | 82.81% (53/64) | 521.30 | 1.00x |
| SpecSLO/SpecRhythm TP1+TP3，gamma=4（批量 CPU materialize） | 416.375 | 54.22% | 20.31% (13/64) | 84.576 | 0.162x |
| SpecSLO/SpecRhythm TP1+TP3，gamma=2 | 414.274 | 66.83% | 20.31% (13/64) | 84.149 | 0.161x |
| SpecSLO/SpecRhythm TP1+TP3，gamma=4，priority burst=2 | 414.563 | 54.61% | 20.31% (13/64) | 84.208 | 0.162x |
| SpecSLO/SpecRhythm TP1+TP3，gamma=8 | 302.115 | 38.45% | 20.31% (13/64) | 61.367 | 0.118x |

因此图片中的 80% attainment 和 1.3x Goodput **尚未达成**。当前 best SpecSLO/SpecRhythm
吞吐仅为 baseline 的 0.657x，Goodput 为 0.162x；不能把普通吞吐、SLO 达标率
和 Goodput 混写成加速比。短输出的独立 sanity workload（同一 B64、RPS4、
`max_tokens=32`）在最新 continuation-device 路径下为 102.766 tok/s、56.25%
attainment，输出哈希为 `94f88c86174f830b9b497c6ae60e9f9fb641715a18edc95ef01e9a337d3e5f5b`
，与修改前一致；文件为
`/root/data/specslo-workloads/smoke-pearl-rps4-b64-t32-continuation-device.json`。

#### 本轮实际验证的优化

- **draft continuation 保留在设备侧**：draft rank 已在本地将 proposal 写入
  `draft_states`，`_broadcast_device_round_result` 不再把每轮完整 continuation
  拷回 CPU，只同步 verdict；target rank 仍按需要 materialize continuation。短回归
  输出一致。
- **proposal 矩阵批量 materialize**：draft worker 每轮只调用一次
  `detach().cpu().tolist()`，再在 Python 侧拆分各 request，避免按 request 触发
  多次 stream synchronize。最新长回归为 416.375 tok/s（Goodput 84.576 tok/s），
  相对旧 gamma=4 结果约 2%--3% 波动级改善，未改变总瓶颈。
- **固定 home-batch draft ACLGraph**：对 paged-attention 的稳定 B32 home batch
  捕获 gamma-step graph，短回归 replay 正常、shape fallback 为 0；e2e 仍约
  103 tok/s，因此保留为可选路径，不宣称收益。
- **SLO priority burst**：加入有限 burst 和 urgency 选择；在 B64 长 workload
  中吞吐与达标率都没有实质改善，默认关闭。

#### 已否决或受硬件阻塞的方向

- **TP3 fused matmul+all-reduce/MC2**：实机 probe 调用
  `npu_mm_all_reduce_base` 被 CANN 直接拒绝：`Rank size 3 is not supported by
  socversion id 2201; A2 supports rank size 1,2,4,8...`。代码保留 capability
  probe 和失败 fallback，但当前固件没有可用的 rank-3 fused kernel，不能把 probe
  通过写成性能优化。
- **gamma=8**：target verify 和 draft 开销增加，吞吐降至 302.115 tok/s，接受率
  38.45%，明确否决。
- **CPU verdict**：可以减少 draft 侧 verdict 的小块 NPU 同步，但 target verify
  反而增加；短测试总耗时仅约 0.3% 波动，不设为默认。
- **直接强制 draft full-graph/FIA**：FIA 或错误的 full-graph shape 会导致 graph
  fallback 增多或 e2e 变慢；当前只保留 paged-attention 固定 home-batch guard。

#### 目前真正的瓶颈和下一步

长回归的 target leader phase 为 target compute 21.327 s、target verify
10.151 s、target-to-draft broadcast 2.957 s；draft worker 仍累计 28.259 s。
这些阶段在一次 round 内由 draft/target 两个 worker 并行启动，但 round 末尾仍在
HCCL verdict/state 边界汇合，尚未形成多个 round 同时在飞的深流水线。要达到图片
目标，必须继续完成并在实机验证：

为避免把累计 worker 计时与单轮阶段混为一谈，另做了同一 workload 的
`profile-only --profile-decode-steps=3`。profile 文件为
`/root/data/specslo-workloads/smoke-pearl-rps4-b64-t256-bulkcpu-profile3.json`，
前三个稳定 decode step 的平均阶段耗时如下：

| 阶段 | 每 step 平均 |
| --- | ---: |
| Draft compute | 39.163 ms |
| Draft -> Target communication | 0.846 ms |
| Target verify（target compute + verdict） | 34.277 ms |
| Target -> Draft communication | 0.266 ms |
| wait / sync / state update | 13.932 ms |

其中 Target compute 为 33.847 ms，target verdict 为 0.430 ms，wait/sync 为
13.875 ms，state update 为 0.057 ms。profile-only 只运行三步并截断输出，不能
替代上表中的 416.375 tok/s 完整 e2e 结果。

1. 基于 HCCL async work/独立 stream 的跨 round mailbox，使下一批 draft 在上一批
   target verify 尚未完成时继续推进，并验证多 round 的状态提交顺序；
2. 针对 Ascend 910B2/CANN 版本提供可用的 TP3 MC2 或等价 rank-3 all-reduce
   kernel，并验证 matmul、归约和 KV 提交的同步边界；
3. 在 RPS2/RPS4、B8/B64 上用同一 manifest 同时跑 baseline 和 PEARL，完成
   80% attainment 与 1.3x Goodput 的全矩阵回归。

在上述两项硬件/执行模型工作完成前，继续调 gamma、CPU verdict 或 graph bucket
只能带来局部波动，无法合理声称达到图片中的验收目标。

### 6.3 2026-09-09：arrival-aware admission 与 TP3 硬件验证

本日新增的实现和复测记录见 [`specslo_regression_20260909_zh.md`](specslo_regression_20260909_zh.md)。
核心变化是将在线到达的后续请求从“每个请求单独 prefill”改为 ready 集合的一次
packed prefill，并取消首个 arrival 前对未来请求的整批预填充。128 请求测试中实际
使用 89 次 packed admission 覆盖 128 个请求，acceptance 从 42.94% 提升到 75.24%，
但 E2E 吞吐仍为 101.20 tok/s，低于 vLLM-Ascend TP4 graph 的 108.08 tok/s。

为验证 TP3 target 的矩阵/归约瓶颈，开启 `VLLM_ASCEND_PEARL_ENABLE_TP3_MM_ALL_REDUCE=1`
做了实机 probe；A2 CANN 返回 `Rank size 3 is not supported by socversion id:2201`，
随后代码自动回退普通 HCCL。该结果确认当前硬件无法直接使用 rank-3 fused MC2，不能
把这条路径算作有效优化。新增 `spec_rhythm_stable_graphs` 和
`spec_rhythm_max_target_batch` 仅作为显式实验开关，动态 FIA 与 target cap=32 的实测
均没有收益，默认策略保持固定 paged graph 和 merged-home。

本轮没有达到图片要求的 80% attainment/1.3x Goodput。原因已在独立报告中量化：
arrival-inclusive workload 的请求到达跨度约 31.6 秒，4096 token 的 E2E 上界约
129.6 tok/s，而 1.3 倍 graph baseline 已是 140.5 tok/s；此外 TP3 target forward、
target verdict 以及按轮次的 HCCL/state 边界仍是主要服务开销。后续必须使用更长的
稳态 workload，并实现跨 home 的真正异步 mailbox/overlap，才能对图片目标做有意义的
验收。

### 6.4 2026-09-09：树状 SpecRhythm 主循环接入

为修正此前“树组件存在但主循环仍是线性 proposal”的缺口，本轮新增了显式树模式：

- `spec_rhythm_tree_width` / `spec_rhythm_tree_depth` 配置和 CLI 参数；`width=1,
  depth=1` 保持线性 PEARL，只有大于 1 才启用树路径。
- 固定全局验证预算 `B` 先由 `SpecRhythmBudgetShaper` 分配，再由
  `SpecRhythmTreeCoordinator` 转换为每 request 的 dependency-closed
  `TreeSpeculationPlan`。ACLGraph padding 节点不会计入 active budget。
- draft 沿 spine 逐层生成 top-k sibling，使用 ancestor mask 写入完整固定 shape；
  HCCL envelope 只传 active candidate nodes。
- target 使用相同 plan 做 tree forward；target graph capture/replay 失败时回退
  dense eager；设备 verifier 保留每个候选节点的 target 输出，沿实际接受分支动态
  选择 frontier/bonus。
- tree metadata 将 RoPE 的逻辑深度位置与 KV 的唯一物理 slot 分开，ancestor mask
  按物理 slot 建立，避免 sibling 分支互相读错 KV。
- ready proposal 和 ahead-of-turn eager proposal 按 proposal id 分开保存，拒绝会
  invalidate；树 eager 还会记录父树主干和 draft frontier，只有依赖一致才 promotion，
  两侧均执行 accepted-path KV compaction。
- `PearlDualBatchResult` 现在返回 draft/target 各自耗时及真实 overlap window，避免
  把两个并发回调仅凭总耗时误判为串行。
- 通用 `SpecRhythmScheduler` 现在也使用 projected wait、SLO urgency、priority
  burst、ready-home 合并和 eager acceptance 门控；双 worker adapter 在每个并发
  service step 后回写 cycle elapsed，后续预算不会继续使用静态 TPOT。

本轮 CPU 相关回归为 `144 passed`，并通过 `py_compile` 和 `git diff --check`。

### 6.5 2026-09-09：910B2 端到端核心功能验收

在 NPU 3、4、6、7 上完成了短端到端 smoke（Qwen2.5-0.5B TP1 + Qwen2.5-14B
TP3；NPU 0、1、2、5 上的外部进程未触碰）：

| 路径 | 配置 | 实际结果 |
| --- | --- | --- |
| 线性双批 | `B=2, gamma=2, max_tokens=8` | 正常生成，接受率 1.0，ACLGraph capture/replay 无失败 |
| 树 eager | `B=4, width=2, depth=2, max_tokens=2` | 正常生成，tree accepted path、KV compaction、双组 HCCL verdict 均完成 |
| 树 ACLGraph | 同上，启用 `VLLM_ASCEND_SPECRHYTHM_TREE_GRAPH=1` | `aclgraph_target_entries=1`、captures/replays 均成功、失败捕获为 0 |
| SLO rolling eager | 两请求，`B=4, gamma=4, TPOT=1000ms` | 两请求均达标；`eager_proposals=5`、`promoted=5`、`invalidated=0` |

随后在同一 910B2、TP1+TP3 配置补做 Qwen3-0.6B + Qwen3-32B 树 ACLGraph smoke：
`tree_nodes=4`、`accepted_nodes=2`、`aclgraph_target_entries=1`、
`aclgraph_captures=3`、`aclgraph_replays=3`、`aclgraph_failed_captures=0`，正常生成
完成。Qwen3 输出中的 `<think>` 是模型自身的 chat-template 内容，不代表 graph
fallback；这条结果证明树路径不是只对 Qwen2.5 结构成立。

这次验收同时修复并验证了四个真实运行问题：native cache 元数据缺失、ACLGraph
capture 中 `context_lens.item()` 触发 host sync、target leader 未参加 correction
group 导致 HCCL 初始化死锁，以及 tree result 缺失 `num_acc_tokens`。因此论文图示的
三条核心机制（dual-batch、rolling eager、individual fixed-budget shaping）已经
闭合到 native Ascend 执行路径；长上下文、多请求、Qwen3/其他结构和 Goodput/1.3x
矩阵仍属于后续性能验收，不在本节短 smoke 结论中冒充完成。

### 6.6 2026-09-10：核心闭环更正、批量树 FIA 与独立数值验收

**6.5 的短 smoke 不足以证明核心功能完成，其结论由本节及同日审计更正。**
后续长一些的回归发现真实 ready 预算超限、在线准入缺失、跨 rank confidence
不一致和树 draft/verify 同步顺序问题；已经分别修复并建立对应反例测试。
当前不再用普通 PEARL 的结果证明 SpecSLO，也不把 CPU 组件通过写成 NPU 完整验收。

本轮工作分成三个层次：

1. 调度/状态：约束实际送入 target 的祖先闭合候选集合；双槽在线补入；normal/eager
   draft 按层合批；先验证全部依赖和物理 KV 映射，再统一提交、promotion 和流式交付。
2. Ascend 适配：复用 vLLM-Ascend 的 FULL-mask FIA（`sparse_mode=1`，
   `inner_precise=2`）作为树算子；保留 2D mask 独立 dense 参考，同时构造请求维度
   的 4D FIA mask。只对 mask 补齐，不增加假 query。ACLGraph 的 mask、长度和页表
   真正更新，tree/linear 的缓存键和算子契约相互隔离。
3. 验收：增加同一实际 hidden 的 LM head 对照、同一实际 prefix KV 的祖先路径
   对照、真实失败现场快照，以及独立 CPU float64 attention oracle。测试观察器的
   同步和拷贝开销不能计作正式吞吐结果。

新 FIA 接口的实卡原型：32/32 有限输入、24/24 有限兄弟 KV 扰动、34/34 动态
Graph 回放通过；24/24 人为 NaN 注入会污染结果。这揭示缓存零初始化及异常值
fail-stop 的必要性，不使用 `nan_to_num` 隐藏有效节点异常。联合 CPU 测试已达
507 项通过；冻结快照 `specslo-fia-snapshot-lp9zS0` 的整模型回归随后取得 **8/8
通过**：eager/Graph、B=1/4、GSM8K/跨页四种组合，输出全部一致，Graph 校验失败和
容量回退均为零；跨页阶段复用已有图，没有新增 capture。后续有限值异常 guard 与
严格预算表来源校验属于该快照之后的改动，需要另外联合回归。

此前尝试 FP32 SDPA 没有解决原始跨页漂移，已经否决其“修复完成”结论。旧在线
观察点的已接受分支与同 prefix KV 的祖先 teacher 逐元素相同，并出现真实 BF16
logit 平局；这排除了该观察点的 head tie-break/树 mask 错误，但不等于历史不同
batch 的前缀计算已经一致。诊断配置差异与原始失败均保留，不追改历史结果。

详细证据、仍待验收项及结果路径见
[本轮核心回归记录](specslo_core_regression_20260910_zh.md)。1.3× Goodput 尚未达成。

### 2026-09-10 功能冻结与最终 B 前置条件

最新实现已补齐文本生成范围的 V1 EngineCore 协议、OpenAI-compatible HTTP/SSE
生产生命周期、实时接纳/取消/故障传播，以及 target/draft 非贪心树采样。Qwen3-0.6B
TP1 + Qwen3-32B TP3 四卡 Graph 实机回归通过，Graph capture/replay=2/59，两个请求
均在运行期接纳，服务结束后 inflight=0。spec decode CPU 全量为
`775 passed, 16 warnings`。

TP3 MC2 已按模型真实形状实测并全部拒绝：当前 CANN/910B 环境不存在同时满足数值和
性能资格的 fused shape，production 因此使用 matmul + HCCL all-reduce。跨 dense/PA
与 FIA 的 BF16 raw logits 差异保留为诊断失败；同一 FIA packed shape 的 Graph/eager
以及独立重复执行逐元素一致，不能用放宽阈值改写硬件归约差异。

最终 B 不采用本文件历史固定值。论文 §5.2 是双槽 runtime loop，§5.3 才规定离线
profiler：在不同模型、active batch、context 范围下扫描总候选数，取 target verify
延迟仍处于标准 batch decode 的 `epsilon` 范围内的最大连续预算。采样器已经支持单次
模型加载的 batch 8/16/32/64 矩阵、逐候选精确扫描、P95 target-rank MAX 和首次越界
停止；当时 strict schema v2 绑定 FULL-mask 最大长度和树拓扑（现已由 6.7 的 schema v3
取代），旧 B1/B4 报告会被拒绝。

完整冻结清单与最终实验协议见
[SpecSLO 功能冻结与 B profiling 前置验收](specslo_functional_freeze_20260910_zh.md)。
只有生成最终 B 表后，才允许开始 RPS/Goodput 性能测试与调优。

同日后续：新增全模型有限值 sticky guard，实卡 target rank 2 有效 head 权重 NaN
注入时四 rank 均在首 token 之前停止，恢复权重后仍不允许复用污染状态。第 5.3 节补上
`a_need=0` 不新分配 eager、W 按 `a_need × benefit` 排序；最新联合 CPU 572 项通过。
严格 FIA/Graph profile 已用全部真实样本得到 active=1 的 B=1、active=2 的 B=4，
可覆盖容量 2、上下文不超过 512 的尾部预算变化。profile 测量不等于在线吞吐/Goodput
达标；同 TP3 对 dense/PA 的全词表 logits 容差检查仍有失败，独立同 FIA 对照继续进行。

包含全部保护和 gap/W 修复的冻结版 `5K2rjm` 已再次完成四卡 8/8 在线 source/跨页、
eager/Graph、B1/B4 联合回归，全部输出一致，Graph failed/shape/capacity fallback
为零。补充 draft rank 0 最后一层 MLP NaN 注入也按预期在首 token 前被四 rank 拒绝。
这些确认的是上述有限条件下的功能与异常行为，不代替不同算子全词表数值、当前设备
重叠比例、完整在线 profile 和性能矩阵。

### 6.7 2026-09-10：schema v3 最终 B 与生产消费闭环

本节取代 6.6 末尾“最终 B 前置条件”和历史 active=1/2 B1/B4 smoke；旧数字不再进入
生产。顺序严格为：先完成文本生成范围的 §5.1/5.2 工程闭环与实卡回归，再按 §5.3
测 B，最后才进入 RPS/Goodput 性能阶段。

本轮新增和确认：

1. strict profile 升级为 schema v3，绑定 `native_engine.py`、`native_model.py`、
   `native_graph.py`、`tree.py` 的 SHA256，以及实际 AR/verify attention backend；源码
   或算子路径变化会令旧 B 自动失效。
2. 标量 B 的候选扫描改成逐预算枚举全部 permutation-equivalent canonical
   histogram。通过预算必须全部形状通过；第一次失败可由一个实测反例确定。
3. AR 基线修正为完整 active batch 的标准单 token decode；tree verify 仍只执行当前
   logical slot 的物理请求，避免把 AR 错缩成半批后虚增 B。
4. 线性候选树使用标准 causal FIA 快路径；真正含 sibling 的树保留 FULL-mask FIA。
   Qwen3 TP1+TP3 实卡中，causal Graph/eager 以及 sibling `[1,3]` FULL-tree
   Graph/eager 的 hidden/logits 都为零误差，后者 capture/replay=1/2。
5. 首次 FULL-tree capture 补齐 task-update stream 到 replay stream 的依赖。离线采集
   不再反复 reset CANN graph-task：每个 active batch 使用 fresh 进程，一次构建最大
   context cache，按 context 从大到小复用 resident Graph；该生命周期与生产一致。
6. 未测 active/context 或未测 logical-home 物理行数不从 gamma 推断 B，而是该轮执行
   target-only，同时同步 target/draft KV，并分别记录 unprofiled fallback 计数。

最终实验固定 Qwen3-0.6B TP1 + Qwen3-32B TP3、Ascend 910B2、Graph、tree 2×2、
P95、`epsilon=10%`（项目选择，论文未公布数值），每 shape 3 次 warmup 和 10 次计时
replay。context 依次为 512/1024/1536/2048/2560/3072，得到：

| active / verify rows | B 列表 |
| --- | --- |
| 8 / 4 | 5 / 5 / 5 / 5 / 6 / 5 |
| 16 / 8 | 9 / 9 / 9 / 9 / 9 / 10 |
| 32 / 16 | 17 / 17 / 17 / 17 / 17 / 17 |
| 64 / 32 | 33 / 33 / 33 / 33 / 33 / 33 |

24 个 evidence key 均有完整原始样本，无 zero key。线上 Graph 验收命中 `8:1` 的
B=5，实际按请求分为 `[2,1,1,1]`，候选总数正好为 5；随后 batch tail 未覆盖时执行
17 轮 target-only fallback。流式输出、finish、TP3 rank 一致性和 Graph 命中全部通过。

证据目录：`/root/data/nano-pearl-benchmark-results/20260910-specslo-functional-freeze/`；
核心文件为 `qwen3-tp3-graph-Broof-schema3-resident-eps10.json`、
`qwen3-tree-graph-e2e-after-ordering-fix.json` 和
`qwen3-profile-online-graph-b8-tail-fallback.json`。spec decode CPU 全量为
`880 passed, 13 skipped`。

由此，当前文本生成范围的论文功能与 B 生产消费链已经完成。下一阶段是性能验收，
不是继续用手工 gamma 补功能：按 RPS=2/4、batch=8/16/32/64 和 6:2:2 三类 SLO
workload 对比 TP4 vLLM-Ascend，统计 TPOT 达成率与 Goodput。1.3× 尚未在本节宣称。
TP3 fused MC2 已经实测否决，生产使用数值合格的 matmul+HCCL；跨 dense/PA 与 FIA
raw BF16 logits 仍作为跨后端诊断差异保留，不影响同 FIA Graph/eager 的接受标准。

### 6.8 2026-09-11：程序级 profiling 检查点

进入算子调优前，本轮先建立低扰动 host timeline、同步阶段 profile 和 CANN device
trace 三层证据，并修复首次尾部 Graph shape 会重复执行完整 32B eager reference 的
运行时校验 bug。修复后 Qwen3-0.6B TP1 + Qwen3-32B TP3 在固定 B8 短回归中的三次
中位吞吐为 167.785 token/s，相对生产 vLLM-Ascend TP4 target-only 的
141.663 token/s 为 1.1844x；输出与 native TP3 target-only 完全一致，Graph 三类
fallback 和额外 runtime validation replay 均为 0。1.3x 尚未达到。

新细分表明：稳态 scheduler 的约 2.216 ms 中约 2.014 ms 用于 tree plan/ticket
构造；state update 的约 3.072 ms 中约 2.182 ms 用于 commit consensus。target
compute 约 33.888 ms、draft 约 22.861 ms，host 窗口重叠约 22.802 ms，但仍有约
11 ms target 尾部未被普通 draft 覆盖。CANN trace 证明存在真实设备 overlap，同时将
target MatMul 和 TP3 HCCL all-reduce 确认为后续算子层热点。

简单增大 B10/B12/B16 和跳过 FULL FIA 的二维 mask copy 均未提升性能，后者已撤销。
当前优先继续拆解 tree plan 构造、commit consensus 和 rolling eager 尾部覆盖；发生
显著执行架构变化后才重新按论文流程校准最终 B，而不是每次源码修改都重测。

完整配置、数据、阶段定义、证据路径和下一步问题见
[SpecSLO 程序级 Profiling 与性能优化记录](specslo_profiling_and_optimization_20260911_zh.md)。

### 6.9 2026-09-11：短回归吞吐达到 1.3x

在不改变 B=8、tree width/depth=2/2、TP1+TP3 拓扑和 Graph 硬门禁的前提下，本轮针对
程序级 profiling 的两个新增结论完成了以下优化：

1. spine-first 拓扑及可见坐标缓存，mask 祖先关系改为一次索引写；CPU 小树唯一性检查
   不再调用 `torch.unique`；
2. 已完整物理打包的 identity tree 不再重建，proposal publish 不再创建逐请求 NPU
   index/destination tensor，也不再做 KV self-copy；
3. normal/eager plan 复用上限 64 项的只读 CPU geometry LRU，预算变化只创建轻量计划
   视图；
4. prefill 保留逐层 KV 非有限值检查；完整 tree decode 改在模型输出和真实 logits 边界
   检查，并继续在全局 commit consensus 之前失败。decode commit 不再聚合 admission 已经
   投票通过且 decode 不再更新的逐层 flag。

同卡三轮 E2E 吞吐为 183.447、184.405、184.858 token/s，中位 184.405 token/s；相对
vLLM-Ascend Qwen3-32B TP4 target-only 141.663 token/s 为 **1.3017x**。三轮输出 SHA256
均为 `3204949b4f0aecb1b4a5398994b38d008c21bb30163c03073c245bb46d74e3b1`，Graph
failed/capacity/shape fallback 均为 0。稳定 cycle 中位由 v30 的 43.209 ms 降至
37.456 ms，scheduler 由 2.216 ms 降至 0.429 ms，tree plan/ticket 由 2.014 ms 降至
0.211 ms，target/draft host window 由 33.888/22.861 ms 降至 30.561/20.613 ms；真实
overlap 仍为 20.461 ms。

采用的同卡证据为
`tree-b8-B8-cached-plans-3run-v35.json`。最终代码因原 NPU 1--3 被外部 PID 占用，另在
NPU 4--7 完成一次功能 smoke：输出 hash 一致、84 次 Graph replay、三类 fallback 为 0；
其吞吐不与旧卡位 baseline 混算。width/depth=2/3、B=12 探针仅 164.856 token/s，已
否决，不写入生产默认值。

上述 1.3017x 仅回答固定 8 prompt x 16 output token 的短吞吐回归。论文式 RPS=2/4、
batch=8/16/32/64、严格/常规/宽松 6:2:2 的 TPOT attainment 与 Goodput 验收仍是独立
阶段，不能用本节替代。
