# vLLM-Ascend-SpecSLO 工作记录

更新时间：2026-09-18

> 当前结论：除第 6.12 节的 60 请求验收外，2026-09-18 已按作者规则生成并完成
> 463 请求、RPS=4、batch 上限 64、固定 `gamma=4` 的四卡验收。相对同四卡 TP4
> target-only + coalesce4 baseline，SpecSLO TP1+TP3 的 decode raw 为 **1.0048x**，
> 论文口径 Goodput 为 **1.5681x**，TPOT 达成率为 **95.03%**；正式计时窗口 graph
> capture、runtime validation 和所有 fallback 增量均为零。详细证据见
> [作者工作负载 KV-ready 优化报告](specslo_author_workload_kv_ready_optimization_20260918_zh.md)。
>
> 第 6.1--6.11 节是按日期保留的历史检查点，其中“尚未达标”“下一步”等措辞只描述
> 当时状态，不能覆盖第 6.12 节。2026-09-10 的功能缺口审计见
> [第五章及核心功能审计](specslo_section5_audit_20260910_zh.md)，修复过程见
> [核心闭环回归记录](specslo_core_regression_20260910_zh.md)。历史 PEARL 结果始终
> 不能替代 SpecSLO 验收。

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
- 保留 TP3 MC2、融合 FFN、QKV NZ 和权重预取等实验入口。draft/target 模型阶段
  overlap 已进入正式路径；跨模型 HCCL 与 target TP3 all-reduce 的并发在当前
  CANN/HCCL 上会形成 stream wait cycle，故采用计算结束后的安全通信边界。

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
3. **SpecRhythm 控制面**：通用路径记录接受率 EMA、SLO urgency、draft/verify 成本，
   按预算和队列状态选择 gamma、继续 draft 或 target 验证；第 6.12 节的独立验收路径
   按用户约束固定每请求 `gamma=4`，不消费离线 B 表。
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
- **SpecRhythm 与树状投机解码的扩展实机回归**：Qwen3 短树和 fixed-gamma linear
  路径已完成实卡回归；仍需覆盖长多请求树分支/回滚、Llama、随机采样和更长上下文
  的数值压力矩阵。
- **通用 vLLM 服务路径**：`SpecRhythmScheduler` 与
  `PearlDualModelScheduler` 已提供 admission、双 batch 并行回调、preempt/reactivate、
  global roofline 和 verification commit；仍需绑定上游 V1 scheduler 的 worker 生命周期。
- **PEARL-2 训练/蒸馏**：`pearl/distill.py` 已提供 teacher rollout、JSONL trace
  loader/collator、acceptance-weighted KL/CE、梯度裁剪、optimizer step 和 checkpoint；
  仍需大规模训练、teacher/student 权重产出和质量回归。
- **生产级动态 shape graph（运行时 guard 已完成，版本矩阵待验证）**：native
  graph 已按 shape bucket 捕获、回放、首轮 eager 对照和容量 fallback；仍需按
  CANN 版本和真实到达分布建立 capture/replay 兼容矩阵。
- **性能和稳定性扩展矩阵**：RPS=4、batch 上限 64、固定 gamma4 正式点已完成；
  RPS=2、batch=8/16/32、动态 B/gamma、长上下文和多租户抢占仍需独立回归。
- **扩展覆盖面**：多模态、LoRA、structured output、更多 tokenizer/vocab 映射
  以及异常退出后的 HCCL 资源回收仍需补齐。

## 6. 验证记录

下列小节按发生时间保留验证记录。最新完整投机解码 CPU 回归为
`1179 passed, 13 skipped, 16 warnings`；修改/新增 Python 的 Ruff、
`git diff --check`、正式 runner `bash -n` 和新示例 `py_compile` 均通过。

早期迁移收尾曾执行以下检查：

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
- 该阶段相关单测为 `132 passed, 14 warnings`；当时完整
  `tests/ut/spec_decode` 曾受容器缺少 `numba` 影响。该阻塞后来已经消除，最新完整
  回归结果以上述 1179 passed 为准。

参考仓库逐文件审计见
[`atc26v0_feature_audit_zh.md`](atc26v0_feature_audit_zh.md)。该审计把上游
scaffold/probe 与真正可执行的 nano-PEARL 功能分开记录，避免把探针通过误报为
生产迁移完成。

该早期阶段只做功能迁移和验证，没有宣称新的吞吐提升；后续性能报告继续必须注明
模型、TP、batch、gamma、warmup、CANN/驱动版本和端到端计时口径。

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
不能用于宣称 1.3x Goodput。当时结果尚未达到图片中的 80% SLO attainment 和
1.3x Goodput 目标；主要瓶颈是 TP3 target compute、每轮 verdict/broadcast、
以及 draft-target exchange，ACLGraph capture/replay 本身未失败。

本轮已验证的可复现改动：

- 放宽 prefill chunk 上限到连续队列容量，允许一次 packed prefill，同时保持
  decode graph 的 batch bucket 不变；短 workload 的 SLO 达标率由 47.5% 提升到 60%。
- 将 admission wait 从 `decode_elapsed_ms` 中分离，wait 仍计入 urgency 的调度债务，
  但不再污染 decode TPOT；这使 SLO 统计与论文定义一致。
- 增加 `--spec-rhythm-auto-eager-tokens`，按请求 gamma 限制 eager 预算，并在结果
  中输出按类别的 attainment/goodput/mean TPOT。

当时确定后续回归必须使用同一份 manifest、同一到达时间、同一 max token 和相同计时口径，
分别跑 RPS 2/4 与 batch 8/16/32/64；在 baseline 也接入在线 arrival/SLO 计时前，
只报告吞吐，不报告伪 Goodput 加速比。

### 6.2 图片验收目标与 2026-09-09 回归

图片给出的 SpecRhythm 验收条件为：tight、normal、loose 三类请求的 TPOT 上限
分别为 40/50/150 ms；同一 batch 内请求比例为 6:2:2；RPS 在 2 和 4 之间变化；
batch 覆盖 8 和 64；相对原始 vLLM-Ascend 的 SpecRhythm 达标率至少 80%，
Goodput 至少 1.3 倍。当时实现使用 Qwen3-0.6B draft（TP1）+ Qwen3-32B
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

因此在 2026-09-09 该轮中，80% attainment 和 1.3x Goodput **尚未达成**。当时 best SpecSLO/SpecRhythm
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
HCCL verdict/state 边界汇合，尚未形成多个 round 同时在飞的深流水线。当时提出的
候选改进和实机验证方向为：

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

当时判断上述硬件/执行模型工作是达标前置条件。第 6.12 节后来以计算阶段 overlap、
安全 HCCL 边界和 SLO 调度通过 1.3x，证明 TP3 MC2 和跨 communicator 通信并发不是
固定 gamma4 正式点达标的必要条件；本段只保留为历史诊断。

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

该轮没有达到图片要求的 80% attainment/1.3x Goodput。当时原因分析为：
arrival-inclusive workload 的请求到达跨度约 31.6 秒，4096 token 的 E2E 上界约
129.6 tok/s，而 1.3 倍 graph baseline 已是 140.5 tok/s；此外 TP3 target forward、
target verdict 以及按轮次的 HCCL/state 边界仍是主要服务开销。当时推断需要更长
稳态 workload 和跨 home 异步 mailbox。第 6.12 节后来完成正式长 workload，并在不让
跨模型 HCCL 与 TP3 collective 并发的安全边界下达标，因此该 mailbox 推断不再作为
必要条件。

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
- `PearlDualBatchResult` 返回 draft/target 各自耗时及 rank-local host overlap
  window，避免把两个并发回调仅凭总耗时误判为串行；物理设备 overlap 需由 CANN
  trace 单独证明。
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
[本轮核心回归记录](specslo_core_regression_20260910_zh.md)。截至该历史检查点，
1.3× Goodput 尚未达成。

### 6.6 2026-09-10 功能冻结与最终 B 前置条件

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
对于论文动态预算路径，只有生成最终 B 表后才允许开始对应的 RPS/Goodput 性能测试。
第 6.12 节是用户后续明确要求的固定 gamma4 独立验收，不消费或冒充该动态 B 表。

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
fallback 和额外 runtime validation replay 均为 0。该 profiling 检查点尚未达到 1.3x。

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
0.211 ms，target/draft host window 由 33.888/22.861 ms 降至 30.561/20.613 ms；
rank-local host timestamp overlap 为 20.461 ms。物理设备 overlap 只由独立 CANN
trace 证明，不能由该 host 数字替代。

采用的同卡证据为
`tree-b8-B8-cached-plans-3run-v35.json`。最终代码因原 NPU 1--3 被外部 PID 占用，另在
NPU 4--7 完成一次功能 smoke：输出 hash 一致、84 次 Graph replay、三类 fallback 为 0；
其吞吐不与旧卡位 baseline 混算。width/depth=2/3、B=12 探针仅 164.856 token/s，已
否决，不写入生产默认值。

上述 1.3017x 仅回答固定 8 prompt x 16 output token 的短吞吐回归，不能用本节替代
论文式 TPOT/Goodput 验收；后续第 6.12 节已完成其中 RPS=4、batch 上限 64 的正式点，
完整 RPS/batch 矩阵仍未完成。

### 6.10 2026-09-14：固定 gamma=4 串行 full-window 性能重构检查点

本阶段按新的实验约束暂不消费离线 B 表：每个被选择的请求固定分配
`gamma=4`，draft 使用一条四 token 自回归链，不把四个候选伪装成深度不足的树。
本节记录的是当时代码和 CPU 合同检查点；检查时八张 NPU 均由其他容器任务占用，
所以该检查点尚未写入新的 NPU 吞吐或 1.3x 达标结论。后续实测见第 6.12 节。

已完成的关键路径修改如下：

1. 新增独立的 linear full-window 协议。target 对每个 proposal 执行
   `[committed root, d1, d2, d3] -> [d1, d2, d3, d4]`，一次验证完整四 token；
   不再沿用旧 PEARL 在 rejection 后先做一 token pre-verify、再重建宽窗口的状态机。
   full accept 保留合法 rolling-eager child，任意 rejection 原子删除被拒绝后缀和
   eager child，并在 draft/target 两个模型侧提交同一 correction frontier。
2. full-window proposal 改为紧凑异步 HCCL envelope。verification 与 continuation
   是同一四-token 窗口，只发送一份；64 行时消息由旧自描述布局的 961 个 int64
   降为 257 个（含一个 timing scalar）。draft(B) 与 target verify(A) 的模型阶段先在
   两个独立设备组上重叠。当前 CANN/HCCL 若在 target forward 前 post receive，会与
   TP3 all-reduce 形成跨 communicator stream wait cycle；因此各 rank 先完成本地 stream
   synchronize 和 Gloo coordination，随后再提交并等待 compact HCCL broadcast。
3. target 对旧 proposal 的 forward/verdict 与 draft 对下一 proposal 的生成并发。
   verdict 移到新 proposal rendezvous 之前，关键路径由 `max(D,T)+V` 收紧为
   `max(D,T+V)`；host timeline 新增 compact submit/wait 边界，不能把完整 in-flight
   区间误报为纯通信耗时。
4. 新增固定 gamma=4 greedy verdict Triton-Ascend kernel，每行一次写出
   `(accepted_prefix_length, correction_token_id)`，替代 `eq/all/argmax/where/gather/stack`
   多个小算子；只在 full-window greedy 路径启用，旧动态 gamma、随机采样和 CPU
   verdict 均保持原实现。正式计时前必须用 warmup 触发首次 Triton JIT。
5. proposal 在 target forward、verdict 和 correction 中只打包一次二维 tensor，
   删除相同 token 的重复 `cat + stack`；固定 greedy 不再为置信度做逐轮 D2H。
6. rolling-eager 的 W 估计改为下一周期读取已完成的 draft NPU event，并与上一周期
   的真实 logical token 数配对。这样 event 查询不再位于当前 proposal HCCL submit
   之前的同步关键路径。
7. 增加控制面和边界保护：full-window 即使没有单请求 SLO 字段也强制进入
   SpecRhythm service；partial gamma/eager cap 和小于 gamma 的 draft budget 在创建
   ticket/collective 前失败；父 proposal 全接受后已经达到 `max_tokens` 时不生成无用
   eager child；B>64 或 gamma>8 的 paged-attention full-window 自动使用因果 stepwise
   路径，除非显式进入后端资格验证。
8. tiny-batch target-only fallback 现在先原子预检整批 frontier、epoch、ticket 和
   payload，再统一失效 proposal、截断 draft eager 尾并提交精确 target token；在线
   fallback 每周期继续检查新到达请求，不再等长请求完成后才 admission。

当前 CPU 回归覆盖 mismatch 0/1/2/3/full accept、完整 proposal 输入顺序、异步消息
布局、warmup/steady/eager promotion/rejection、缩批 fallback、在线 refill、非法预算在
collective 前失败、旧协议隔离和 public CLI 路由。当时列出的后续硬门禁包括 Triton
kernel/oracle、packed target 数值一致性、TP3 通信无死锁、ACLGraph 零回退、profiling
以及 60 请求正式 TPOT/Goodput；这些项目随后由数值回归、图封存门禁和第 6.12 节的
RPS4/B64 正式结果补齐。

### 6.11 2026-09-14：ACLGraph 计时前资格验证与封存

为避免把首次 capture、changed-input 数值验证或隐式 eager fallback 混入吞吐计时，
native graph runner 增加了可审计的两阶段生命周期：先在未计时窗口重放完整真实
workload，直到一整轮同时满足 capture=0、runtime validation=0、fallback=0、
unvalidated=0；随后封存 graph cache，再进入正式计时。封存状态下遇到 missing key、
未验证/已禁用 entry、shape 不支持或 logical-row expansion 会立即报错，不允许静默
capture、验证或回退。

冷启动调度可能产生后续完整 workload 不再访问的 entry。严格循环只在“一整轮没有
任何新 capture，但仍有未验证 resident entry”时，安全清除这些未验证 entry，并逐项
reset graph、精确归还 capture budget；已验证 entry 永不被该操作清除。清除后必须再跑
完整 workload：若被清除 shape 实际仍是热路径，它会重新 capture，因而不可能误过
fixed-point 门禁。worker/API 已提供 unseal、prune 和 seal 命令，默认非严格运行行为
保持不变。

四卡 Qwen3-0.6B TP1 + Qwen3-32B TP3、固定 gamma=4、40 请求 × 32 token 的 910B2
短验收在第 4 次未计时重放达到 fixed point 并成功 seal。正式计时区间中 draft rank 0
为 185/185 次 draft graph replay，target rank 1--3 各为 185/185 次 packed-FIA generic
graph replay；四 rank 的 capture、capture attempt、runtime validation、failed/capacity/
shape fallback 增量全部为 0，resident unvalidated/disabled entry 也均为 0。对应证据为
`/root/data/nano-pearl-benchmark-results/20260914-specslo-fixed-gamma4-full-window/runs/short-packedfia-coalesce2-sealed-prune-20260914T2022Z/result.json`。
这项短结果只证明计时窗口是纯 graph replay，不能单独替代正式 60 请求验收；随后
第 6.12 节已经完成正式 Goodput/1.3x 门禁。

### 6.12 2026-09-14：固定 gamma4 正式 Goodput 验收通过

正式合同为 Qwen3-0.6B draft TP1 + Qwen3-32B target TP3，对比原生
vLLM-Ascend Qwen3-32B TP4 target-only；使用同四张 910B2、同一严格 36/12/12
manifest、RPS=4、batch 上限 64、60 请求、每请求 256 token，TPOT 约束依次为
40/50/150 ms。候选固定每请求 `gamma=4`、`serial_linear`、tree width/depth=1/1，
不消费离线 B 表。E2E 从 arrival origin 开始，包含 enqueue、worker IPC、prefill 和
decode；输入已 tokenize，候选与 baseline 的 warmup 均不计入正式窗口。

三次原生 TP4 baseline 的原始吞吐为 623.554/620.904/622.519 token/s，论文 TPOT
达成为 36/34/34（共 60），Goodput 为 374.132/351.845/352.761 token/s。三次当前
源码 SpecSLO 候选分别得到：

| 候选 | 原始吞吐 | paper TPOT 达成 | Goodput | 相对最高 baseline Goodput |
| --- | ---: | ---: | ---: | ---: |
| current repeat 1 | 573.285 tok/s | 55/60 = 91.67% | 525.511 tok/s | 1.4046x |
| current repeat 2 | 587.930 tok/s | 56/60 = 93.33% | 548.735 tok/s | 1.4667x |
| host-profile 500 | 578.372 tok/s | 54/60 = 90.00% | 520.535 tok/s | 1.3913x |

因此“最差候选 / 最好 baseline”的保守 Goodput 比值为 **1.391311x**，且候选最差
论文口径 TPOT 达成率为 **90%**，同时越过 1.3x 和 80% 门禁。原始吞吐中位比值只有
0.9291x，故结论严格限定为 SLO-aware Goodput 提升，不能称为 raw throughput 加速。
收益主要来自 tight 请求达成由 baseline 的 11--12/36 提升至 30--32/36；normal 和
loose 在候选三次均为 12/12。

三次正式候选都先运行完整 workload 至 graph fixed point，再 seal cache。计时窗口
每 rank 分别有 544、541、528 次 replay，capture attempt、capture、runtime
validation、failed/capacity/shape/eager fallback 增量全部为零。rank 0 全部进入 draft
full-chain graph，rank 1--3 全部进入 target generic packed-FIA graph。因此图不回退是
timed counter 的硬门禁，不是由命令行开关推断。

500-cycle 低扰动 host profile 中有 479 个 dual cycle，463 个是无在线 prefill 的纯
稳态 dual cycle。纯稳态均值为：cycle wall 42.466 ms、draft host window 36.006 ms、
target host window 14.223 ms、两者交集 14.217 ms、verdict 0.464 ms、T→D 1.009 ms、
state update 0.256 ms。两路 host model window 的 overlap 占较短 target window
99.96%。`compute coordination` 的 23.662 ms 主要是短 target 等待长 draft 到达安全
通信点，已经与 draft window 重叠，不能再次串行相加。compact submit/wait 分别为
0.429/0.135 ms，从 submit 至 exchange 完成的完整 D→T host envelope 为 1.138 ms；
它包含 materialize/publish，不是纯 HCCL wire latency。rank-local host timestamp 不能
单独替代 CANN device-kernel trace。

随后以静态 B8/P8/T32、8 条不同 GSM8K prompt 补做当前 fixed-gamma 路径的全 rank
CANN trace。profiler active 3 个 cycle 均为 draft4 + target4，四 rank 各 27 次 sealed
graph replay且零 capture/validation/fallback。仅统计官方 Ascend Hardware 设备计算事件
后，draft rank 0 对三个 target rank 的 strict AI Core overlap 分别为
10.538/10.632/10.653 ms；AI Core + MIX_AIC 为 16.571/16.572/16.630 ms；全设备计算为
24.766/24.746/24.715 ms。三个 target rank 均保存真实 MatMul--MatMul 同时执行实例。
证据为结果目录下 `validation/device-overlap-fixedg4-retry-5OxoEK/device-overlap.json`。
该短 trace 证明采样三步的物理 kernel overlap，不参与正式 Goodput，三个 target pair
也不能相加或外推为全程利用率。

静态 B8/T32 数值诊断进一步确认 eager oracle、update-first、replay-first 三次和
validate-every-replay 的 token/hash/轮数/接受统计完全一致；replay-first target 每 rank
25 replay、0 fallback，强制校验为 25 validation、0 failure。该诊断只作数值正确性
证据，不参与吞吐结论。

采用的关键优化及审阅位置：

- fixed full-window 与原子提交：`native_engine.py:521-651,6382-6421,10011-10189,10492-10647`；
- 双 home 计划/并行执行：`spec_rhythm.py:561-677,748-777,813-863` 和
  `native_engine.py:6142-6218,6453-6532,6557-6562`；
- HCCL 安全边界/compact transport：`native_engine.py:6650-6754` 和
  `fixed_greedy_transport.py:35-117,169-356`；
- fused verdict：`ops/triton/spec_decode/fixed_greedy_verdict.py:22-111`；
- draft/target replay-first：`native_graph.py:862-901,1070-1100,2194-2275`；
- graph qualification/seal：`native_graph.py:589-757` 和
  `benchmark_nano_pearl_speculative.py:683-800,1384-1481,1534-1794`。

完整逐次表、profiling、provenance、限制和证据路径见
`/root/data/nano-pearl-benchmark-results/20260914-specslo-fixed-gamma4-full-window/README.md`。
最新完整 `tests/ut/spec_decode` 为 **1179 passed, 13 skipped, 16 warnings**。本次只完成
RPS4/B64 固定 gamma4 正式点；RPS2、B8/16/32、动态 B、随机采样和冷启动仍不能由该
结果外推。

### 6.13 2026-09-20：TP3 target 动态 FIA graph-task 刷新优化与端到端否决

本轮继续拆解 Qwen3-32B TP3 target。CANN device profile 除 MatMul、HCCL all-reduce
和 FIA 设备热点外，还暴露出一个主机瓶颈：因果 FIA 图的 KV 长度每轮变化，64 层各自
捕获的单算子 handle 必须在 replay 前逐一调用 graph-task update。原实现由一条 update
stream 串行刷新并为每层 record 一个 ExternalEvent。

新增的 opt-in 路径只对 target worker 的因果 FIA changed-input replay 生效：两条 update
stream 和常驻线程池刷新不相交 handle；event4 配置让 4 个连续 handle 共享事件，且
完整事件组只能属于一条 stream，组内最后一个 handle 完成后才能释放。PA、draft、
FULL-tree FIA 和混合任务仍走原路径。异常会失败关闭，不静默改用未经资格验证的图。

64-handle、B32/Q5/context128 独立实卡探针中，event1 串行 update+replay 为
13.103 ms，event1 parallel2 为 9.649 ms，event4 parallel2 为 7.207 ms；后者相对
event1 串行为 1.818x，所有输出逐元素一致。更宽事件组的 micro 最低值是 event16 的
6.113 ms，但没有继续进入生产：event4 已足以证明端点不再受 target 单独限制。

128 请求饱和门中，两次 event1 串行对照的 target 最慢 rank 折算中位为
25.01 ms/round；event4 parallel2 为 21.70 ms/round，下降约 13.2%。三个 target rank
均完成 64/64 stable graph eager/graph 数值验证，失败为 0；failed/shape/capacity
fallback 均为 0。其 raw=913.461 tok/s，仍处于不同进程接受轨迹造成的历史波动范围，
并未随 target 子阶段同比提升。

在同一 463 请求 Poisson/RPS4/6:2:2/40-50-150 ms 在线 workload 上，同源码串行
event1 得到 raw=922.338、Goodput=525.912 tok/s、达成率 57.02%、target 最慢 rank
77.682 s；event4 parallel2 得到 raw=920.820、Goodput=489.248 tok/s、达成率
53.13%、target=70.912 s。即 target 缩短 8.72%，但 raw 和 Goodput 分别下降约
0.16% 和 6.97%。tight 达成从 72/271 降到 56/271。图回退和数值验证均为零，故这不是
正确性 bug，而是 target 加速后 draft/协议成为关键路径、rolling-eager 可隐藏窗口和
接受轨迹随之变化。

结论：保留该 TP3 原语和 telemetry 供 profiling/后续流水线重平衡使用，但生产安全默认
仍为 event1/worker1，不把 target 子阶段加速伪报成端到端优化。下一轮若追求 raw 或
Goodput，必须优化 draft/通信/调度重平衡，或把 target 节省显式转化为额外有效验证，
而不是继续扩大 target 事件组。完整原始数据、逐候选表和代码索引见
`/root/data/nano-pearl-benchmark-results/20260920-tp3-target-optimization/README.md`。本轮最终
完整 `tests/ut/spec_decode` 为 **1581 passed, 13 skipped, 17 warnings**。

### 6.14 2026-09-21：TP3 FIA 首组短门与稳态分组解耦

继续分析 6.13 的 target 子阶段后确认，统一增大 event group 不是可持续的优化方向：
event8/event16 虽把 64 层 changed-input graph-task update 的主机提交时间继续压低，
却必须等组内 8/16 层全部更新完成才能释放第一个设备事件，推迟了前部 transformer
layer 的 replay。128 请求饱和门中，event8 和 event16 的 target 分别降至
20.239 和 19.809 ms/round，但 correction/verdict wait 升至 9.656 和
9.950 ms/submit，E2E/round 为 71.539 和 72.446 ms，均不优于 event4 的
71.501 ms。两条大分组路径因此被判定为端到端负优化，并已从生产允许集合撤销。

为同时获得早期释放和稳态分组收益，新增了 opt-in 的混合事件门：第一组只包含 2 个
FIA layer handle，后续完整组仍为 4 个，形成 `[2, 4, 4, ...]`。实现没有删除任何
per-layer `graph_task_update_begin/end`，只改变事件边界；一个完整事件组仍只由一个
update stream 负责，防止跨 stream 的组内乱序。新开关为
`VLLM_ASCEND_PEARL_TARGET_FIA_TASK_PREFIX_EVENT_GROUP_SIZE`，当前只允许 0/1/2，
且必须小于稳态 event group。默认值 0 保持原生产行为。

两次独立的 128 请求饱和正式回归结果为：

| 配置 | raw tok/s | rounds | target ms/round | verdict wait ms/submit | E2E ms/round |
|---|---:|---:|---:|---:|---:|
| event4 + parallel2 对照 | 888.155 | 516 | 21.590 | 8.962 | 71.501 |
| prefix2 + event4 v121 | 900.314 | 513 | 21.318 | 8.020 | 70.948 |
| prefix2 + event4 v122 | 869.128 | 539 | 20.661 | 8.467 | 69.948 |

混合门两次中位相对 event4 对照把 target 缩短 2.78%、verdict wait 缩短 8.02%、
E2E/round 缩短 1.47%；但 raw 中位为 884.721 tok/s，尚未超过 event4 的
888.155 tok/s，Goodput 中位也未形成稳定收益。因此该实现保留为已通过数值、图回退
和协议门禁的 TP3 opt-in 原语，不改生产默认，也不把子阶段改善写成吞吐提升。

验证覆盖 32 个精确 request-count shape 的 changed-input replay，stable graph
failure/shape/capacity fallback 均为 0；相关五个单测文件合计 **631 passed**，
`py_compile`、Ruff、`git diff --check` 和运行脚本语法检查均通过。代码入口为
`vllm_ascend/envs.py`、`vllm_ascend/spec_decode/pearl/native_graph.py`，测试位于
`tests/ut/spec_decode/test_native_graph_task_update.py`。完整逐轮原始数据和保守结论见
`/root/data/nano-pearl-benchmark-results/20260920-tp3-target-optimization/README.md`。

同一候选随后进入作者 463 请求 Poisson/RPS4 在线门。结果为 raw=915.597 tok/s、
Goodput=466.697 tok/s、SLO 达成率 50.97%，target 最慢 rank=69.806 s；相对 event4
对照，target 又缩短 1.56%，但 raw/Goodput 分别下降 0.57%/4.61%。tight 达成请求由
56 降到 47。全程 0 graph fallback/failed capture/shape fallback/capacity fallback。
因此混合门只保留为 profiling 原语，在线门正式否决将其设为默认，并停止继续搜索更宽
或更复杂的 FIA event 边界。

随后还实现并严格 A/B 了“精确 attention 分片 + 逆向 FFN 平衡”：去掉 TP3 第 9 个
padding KV group，并让 attention-light rank 承担更宽 FFN。相对同源码 NZ5/event1
均匀分片对照，该候选把 target 最慢 rank 的归一化耗时降低 3.76%--4.09%，E2E/round
降低 2.10%--3.08%；但完整执行多出 21--26 个 decode round，raw 从 895.310 降到
878.664/879.400 tok/s。精确分片改变 BF16 分块/归约顺序，输出 hash 与对照不同；虽然
三次图门禁均为零失败/零回退，也不能把单次 Goodput 上升当作稳定收益。该组合代码已
撤回，只保留 v124/v125/v126 原始目录和报告数据作为否决证据。

### 6.15 2026-09-21：910B2 rank-3 MC2 专用融合内核

CANN 9.0 的公开 `torch_npu.npu_mm_all_reduce_base` 在 910B2 上仍拒绝 rank size 3，
因此本轮没有把 TP2/TP4 API 强行套到 TP3，而是在已有 AscendC MC2 算子内增加了
TP3 小窗口专用路径。排查时发现旧构建只依赖 `.done` 标记、不跟踪 AscendC 头文件依赖；
此前若只修改头文件并执行普通 `make`，设备 `.o` 实际没有重编译。因此旧 v13 报告中的
**1.1308x 不能作为当前源码证据**。本轮改为删除该算子的 `.done/.o/.json` 后强制设备
重编译，并保存每版设备对象 SHA256；以下数字均来自真正包含对应源码的对象文件。

最终路径的执行协议为：

1. AIC 使用 CANN/Catlass 的 unit-flag/FFTS 协议，将各 rank 本地 MatMul 写入 HCCL
   symmetric window；AIV 通过 `WaitEvent` 等待本轮完整 producer epoch；
2. TP3 只清零 payload-ready/read-complete 两个连续 phase 的 6 个 source slot，并用一次
   24-byte DMA 代替通用路径的 21 个逐槽写；一次跨 rank barrier 保证所有 destination
   完成清零后才允许 producer 发布；
3. 40 个逻辑 AIV core 各自读取 rank 0/1/2 window，按固定顺序直接归约到本 rank 输出，
   不再执行 owner-reduce 后的第二次 all-gather；
4. full epilogue 在加入 residual 前显式物化 BF16 all-reduce 边界，再以 FP32 执行
   RMSNorm。这与 split `BF16 HCCL all-reduce + npu_add_rms_norm` 的舍入语义一致；
5. read-complete phase 阻止下一层覆盖仍被 peer 读取的 window。末尾重复 reset/barrier
   已删除，下一次调用开头统一清零即可维持相同安全边界。

该直达路径仅对 TP3、`M<=160` 生效；其他拓扑/大窗口保留通用实现。接口新增显式
`projection_only` attr 且默认 false，旧调用语义不变。`projection_only + native
AddRMSNorm` 的两段执行不作为生产候选；生产资格只接受单次 custom kernel 的完整
`MatMul + rank-3 AllReduce + Add + RMSNorm`。

最终在保存的 Qwen3-32B TP3 真实层输入上测试，精确 shape 为
`M=64, K_local=3072, N=5120, BF16, ND`。21 组交替 A/B、每组 50 次 ACLGraph replay
的结果如下：

| 指标 | split baseline | rank-3 fused | 加速比 |
|---|---:|---:|---:|
| p50 | 0.109393 ms | 0.104421 ms | 1.0476x |
| p90 | 0.110591 ms | 0.105052 ms | 1.0527x |
| p95 | 0.110936 ms | 0.105304 ms | **1.0535x** |
| mean | 0.109803 ms | 0.104388 ms | 1.0519x |

21 个配对样本全部为正收益，单组范围为 1.0353x--1.0865x。Norm 最大绝对误差
0.00390625、Add 输出最大绝对误差 0.0078125，分别通过 0.05/0.01 门限。相同真实输入
连续 5 次 changed-input ACLGraph replay 的三个 rank 输出始终一致，没有 stale window
或图回退。

生产 dispatch 继续采用 source hash、硬件、精确 M/K/N、dtype、weight format、数值误差
和重复 p95 联合资格门；profile 还绑定具体 epilogue。最终可消费 profile 为
`/root/data/tp3-mc2-debug-20260921/qwen3-m64-k3072-n5120-tp3-real-final-qualified-v57.json`，
绑定源码 SHA256 `f01fa4846bdce6f2b7b2b90406c54439fb8fda42c828a3a5ce4c9afc5fd3398d`。
主要代码位于
`csrc/mc2/matmul_allreduce_add_rmsnorm/op_kernel/matmul_allreduce_add_rmsnorm_aiv_kernel.h`、
`op_host/matmul_allreduce_add_rmsnorm_{def,tiling}.cpp`、`vllm_ascend/spec_decode/pearl/mc2.py`
和 `examples/measure_specslo_mc2.py`。该结果证明真实层 M64 热点的算子级可用性，不等同于
SpecSLO 全模型或在线 Goodput 已获得 1.0535x；端到端收益仍须用生成回归单独确认。

### 6.16 2026-09-22：TP3 direct-window 生命周期与同步消融

本轮继续优化 6.15 的 910B2 rank-3 融合算子，测试口径仍是保存的 Qwen3-32B
TP3 真实层输入：`M=64, K_local=3072, N=5120, BF16, ND`。split baseline 为
`linear + dist.all_reduce + npu_add_rms_norm`，候选为单次 custom
`MatMul + rank-3 AllReduce + Add + RMSNorm`；性能数字来自捕获后的算子级
ACLGraph replay，每轮包含 21 组交替 A/B。以下结论只证明该热点算子的可用性和时延，
**不等同于 SpecSLO 全模型吞吐、在线 SLO 达成率或 Goodput 已获得相同比例提升**。

首先确认了两项平台边界。CANN 的公开 `npu_mm_all_reduce_base` 仍拒绝 TP3，不能直接
复用 TP2/TP4 融合路径；尝试接入 native HCCL 的版本约为 0.15 ms，也慢于约 0.10 ms
的 split/direct-window 路径，因此本轮继续采用自定义 symmetric-window 协议。另一个
关键根因是 OpAPI 曾把 HCCL server 生命周期切到 AICPU：即使设备 kernel 字节相同，
首次 ACLGraph replay 仍会挂起。v86 恢复历史 MTE server 后恢复存活，p95 相对 split
达到 1.0177x；因此 MTE 生命周期是后续所有版本的共同前提，而不是性能可选项。

随后按单变量方式进行了如下消融：

| 版本 | 单变量修改 | 关键结果 | 决策 |
|---|---|---:|---|
| v86 | 恢复 MTE HCCL server | p95 1.0177x | 保留，解决首图挂起 |
| v87 | 复用 `add_out` 的 BF16 物化结果 | p50 0.9714x | 否决，向量依赖链反而变长 |
| v88 | reduce tile 从 512 扩到 2560 | p95 0.9936x | 否决，循环减少未抵消 UB/MTE 代价 |
| v89/B1 | 复用 reset 末尾的本地 `SyncAll` | p95 1.0357x | 保留 |
| v90/B2 | 删除 epilogue 后的 `PIPE_ALL` | p95 0.9911x | 否决并恢复屏障 |
| v91/B3 | ready/read-complete 的 publish 与 poll 并行 | 三轮 p95 加速比中位 1.0714x | 保留 |
| v92/B4 | control 路径与 AIC 重叠 | p50 1.0064x | 收益不足，否决 |
| v93 | 在设备内完成 RMSNorm 标量广播 | 首轮 p50/p95 1.0746x/1.0726x | 保留 |
| v94/B5-A | 融合一组同步边界 | fused p50 为 0.101683/0.101638 ms | 慢于 v93 的 0.101374/0.100934 ms，否决 |
| v96 | 跳过 terminal 同步 | p50 1.1062x，但 fused 退到 0.099214 ms | 慢于 v95，否决 |
| v97 | 跳过 self-slot 同步 | p50 1.0504x、p95 0.9225x | 尾部负优化，否决 |

本轮最终算子候选 v95 保留 v89/B1、v91/B3 和 v93 的设备内 RMSNorm 路径，并将
reset 的末端 barrier 合并进随后 CrossRank 同步的末端 barrier；它没有删除
`PIPE_ALL`，也没有采用 v87 的 add-out 复用或 v88 的宽 tile。三次独立测量为：

| 轮次 | split p50 | fused p50 | p50 加速比 | p95 加速比 | 正收益配对 |
|---|---:|---:|---:|---:|---:|
| run 1 | 0.111758 ms | 0.097735 ms | 1.14348x | 1.14455x | 21/21 |
| run 2 | 0.109670 ms | 0.098953 ms | 1.10831x | 1.10141x | 21/21 |
| run 3 | 0.112143 ms | 0.097600 ms | 1.14901x | 1.30789x | 21/21 |

run 3 的 p95 split baseline 含明显尾点，不能把 1.30789x 当作稳定收益；按预先使用的
三轮 p95 加速比中位数，v95 的保守算子级结果是 **1.14455x**。三轮标准真实输入的
Norm 最大绝对误差均为 0.00390625，Add 输出最大绝对误差均为 0.0078125，分别通过
0.05/0.01 门限。32 次 changed-input replay 的三个 rank 均未出现 stale-rank
组合；该扩展诊断逐轮放大输入，Add 的绝对 BF16 ULP 会随幅值增长，因此它用于检查
window 新鲜度，不能继续套用标准输入的固定 0.01 绝对误差门限。

为避免只在 M64 上获得收益后误分发到其他请求规模，又对 v95 做了 synthetic M sweep：

| M | p50 加速比 | dispatch 决策 |
|---:|---:|---|
| 1 | 0.7634x | 拒绝，fallback |
| 8 | 0.8962x | 拒绝，fallback |
| 16 | 0.8579x | 拒绝，fallback |
| 32 | 1.0595x | 可进入精确 profile |
| 64 | 1.0933x | 可进入精确 profile |
| 128 | 1.1346x | 可进入精确 profile |
| 160 | 1.0964x | 可进入精确 profile |

这些 synthetic 数字不能替代相应真实层输入的复测；dispatch 继续按硬件、TP size、
精确 M/K/N、dtype、weight format、epilogue、源码 hash 和误差门联合匹配，M1/8/16
等负收益形状必须回退 split 路径。当前最终 AIV header SHA256 为
`e3aeef239ebc4ef32ca1c59654f5b460a8993aaa037acc8d980a692ea4763272`，v95 profile
绑定的完整 source hash 为
`711b5590ddfa7cfae9f2a885686ddbc89f34a8428416f494960b04e831f573fd`。

原始性能、正确性、构建和 synthetic sweep 证据均位于
`/root/data/tp3-mc2-debug-20260922/`，其中 v95 的三轮正式算子结果为
`qwen3-m64-sync-b1-b3-vector-rms-b5b-v95-custom{,-run2,-run3}.{json,log}`，
changed-input 证据为
`qwen3-m64-sync-b1-b3-vector-rms-b5b-v95-correctness{-32replay,}.log`，M sweep 为
`qwen3-m-sweep-sync-b1-b3-vector-rms-b5b-v95-synthetic.{json,log}`。v95 应作为后续
全模型和在线 Goodput 回归的算子候选；在这些端到端验证完成前，不把 1.14455x 外推为
SpecSLO 的最终吞吐或 Goodput 提升。

### 6.17 2026-09-22：rank-3 MC2 生产接线、fail-closed 门禁与适配层回归

6.16 的 v95 只证明了设备算子本身可用，本轮进一步完成 Native PEARL/SpecSLO 的 target
配置与 Python production adapter 接线，并修复三个会使“有算子收益”退化为“运行时静默
走 split”的生产问题。这里的 NPU 证据覆盖算子和 adapter；尚未把它表述为完整
NativeQwen decoder/model-runner 端到端回归。

第一，MC2 现在只写入 TP>1 的 target model config；TP1 draft 无条件关闭 MC2 并清除
profile，避免 draft 每层进入资格判断和 fallback。显式对 target TP1 开启 MC2 会在配置
阶段报错。第二，Qwen3-32B 的 8 个 KV head 若启用 TP3 exact/light-rank 分片，会形成
`3/3/2` 的 rank-local K；此时两个 rank 可能命中 K=3072 profile，而第三个 rank 因
K=2048 回退，最终在不同 collective 上互相等待。现在该组合在任何模型 forward 前统一
拒绝；默认的均匀 padding TP3 仍保留 `K_local=3072`，可安全进入融合路径。第三，能力
探测不再只看 `torch.ops` schema，而会检查当前 `ASCEND_CUSTOM_OPP_PATH` 或 wheel bundled
vendor 中是否同时具有 OpAPI library、MC2 header 和 kernel manifest。旧的 gather-only
vendor 会明确返回 unavailable；A2/A3 wheel 打包也会对已声明的 MC2 payload 做 exact-SoC
完整性检查，缺件时直接中止构建。

为避免异常后日志仍显示“profile qualification passed”，adapter 现在为每个进程维护
`fused_attempt/fused_success/fallback/exception` 四个计数，并把真实异常写入独立去重日志。
这些计数不增加任何 rank collective；ACLGraph replay 不重入 Python，因此计数表示 capture/
eager host dispatch 次数，而不是设备 replay 次数。计数已接入 worker `graph_metrics()`，
可在完整 benchmark 结果中直接审计 target 是否真正命中融合算子。

最终先将旧 build artifact 可恢复地移出，再从当前源码强制重编译 v98 vendor；源码与安装
payload 的 AIV header SHA256 均为
`e3aeef239ebc4ef32ca1c59654f5b460a8993aaa037acc8d980a692ea4763272`。随后用该 vendor 和
当前 adapter 源码生成 identity-bound profile，最终 adapter/kernel source hash（覆盖
`vllm_ascend/spec_decode/pearl/mc2.py` 与 `csrc/mc2/**`）为
`ce42bd11c2305e828e7afb31e6cb95332a3f252529b95d4f6e3298ad356aa119`。资格工具新增
changed-input graph replay 门，并采用逐元素
`abs(error) / (atol + rtol * abs(reference)) <= 1`；旧的纯绝对误差 profile 仍兼容，新的
生产 profile 同时记录绝对误差和最大容差占用比。性能门从接近零的正收益收紧为至少
`1.05x`。

真实 Qwen3-32B 层输入、`M=64/K_local=3072/N=5120/BF16/ND`、21 组交错 A/B、每组
50 次 graph replay 的最终正式复测结果为：

| 指标 | split baseline | rank-3 fused | 加速比 |
|---|---:|---:|---:|
| p50 | 0.109519 ms | 0.101659 ms | **1.07732x** |
| p95 | 0.110479 ms | 0.103756 ms | **1.06480x** |
| mean | 0.109723 ms | 0.101687 ms | **1.07902x** |

21/21 个配对样本均达到 `>=1.05x`，最差单对为 1.05451x，中位配对为 1.07999x。
同一 source hash、同一协议共完成三轮，未挑选最好结果：

| 轮次 | p50 加速 | p95 加速 | mean 加速 | 最差配对 | `>=1.05x` 配对 |
|---|---:|---:|---:|---:|---:|
| v100 candidate | 1.09633x | 1.26703x | 1.10747x | 1.06672x | 21/21 |
| v100 final（profile/smoke 使用） | 1.07732x | 1.06480x | 1.07902x | 1.05451x | 21/21 |
| v100 repeat-3 | 1.10580x | 1.10004x | 1.10638x | 1.09216x | 21/21 |
| 三轮中位数 | **1.09633x** | **1.10004x** | **1.10638x** | **1.06672x** | **21/21** |

candidate 的 p95 被 baseline 两个尾部点 0.127925/0.142330 ms 抬高，因此正式 profile 和
主表采用更保守的 final 轮；三轮中位数仅用于跨运行复核。

数值门使用 `atol=0.01, rtol=0.01`。静态 eager/graph 的 Norm/Add 最大绝对误差为
0.00390625/0.0078125；8 轮 changed-input replay 后为 0.005859375/0.03125，对应最大
容差占用比 0.42674/0.71429，均小于 1。三 rank 的输出差异为 0。新增的生产 adapter
smoke 不直接调用裸 op，而是通过
`matmul_allreduce_add_rmsnorm_or_fallback(..., use_fused=True, strict_fused=True)` 执行
一次 eager、一次 graph capture、8 次静态 replay 和 8 次 changed-input replay；每 rank
计数都是 `attempt=2, success=2, fallback=0, exception=0`。这证明本次三 rank adapter
smoke 的 ACLGraph 没有静默回退；完整 Native target graph 仍需单独端到端确认。

关键审阅位置：

- TP3 direct-window kernel：
  `csrc/mc2/matmul_allreduce_add_rmsnorm/op_kernel/matmul_allreduce_add_rmsnorm_aiv_kernel.h`；
- target-only 与非均匀分片门禁：
  `vllm_ascend/spec_decode/pearl/native_engine.py`；
- profile、vendor 预检、异常与计数：
  `vllm_ascend/spec_decode/pearl/mc2.py`、`vllm_ascend/custom_op_package.py`；
- wheel payload 完整性：`setup.py`；
- 带多输入数值/性能门的 profile 生成器：`examples/measure_specslo_mc2.py`；
- 严格生产回归：`examples/check_specslo_mc2_production_adapter.py`。
- 完整命令与环境清单：`docs/specslo_mc2_v100_run_manifest_zh.md`。

最终 profile 位于
`/root/data/tp3-mc2-debug-20260922/qwen3-m64-production-v100-scaled-final-qualified.json`，
严格回归日志位于
`/root/data/tp3-mc2-debug-20260922/qwen3-m64-production-v100-scaled-final-adapter-smoke.log`，
资格日志位于
`/root/data/tp3-mc2-debug-20260922/qwen3-m64-production-v100-scaled-final-qualified.log`，重编译日志位于
`/root/data/tp3-mc2-debug-20260922/tp3-v98-final-source-build.log`，开发测试 vendor 位于
`csrc/build/mc2-test-install-tp3-production-v98/`。当前源码树中历史 bundled vendor 仍是
gather-only；开发运行必须先 source 该 v98 vendor，正式 wheel 则由新增的打包门禁保证
重新构建后才允许产出。上述 **1.06480x 是一个 decoder-layer attention 输出投影热点的
算子级 p95 收益，不是 TP1+TP3 全模型吞吐或 Goodput 收益**。完整四卡 TP1+TP3
model-runner/在线回归尚未执行，不能从本节数字外推。

### 6.18 2026-09-22：MC2 完整 model-runner 命中与在线 TP1+TP3/TP4 回归

本轮补齐了 6.17 尚缺的完整模型与在线系统证据，并修复了一个会污染端到端结果的
dispatch 问题。旧接线只要打开 MC2，就会令所有 attention output shape 进入
`mc2.py`；未命中 profile 的 shape 虽然保持数值正确，却走 adapter 内的手写
`F.linear + dist.all_reduce + RMSNorm` fallback，绕过 Native PEARL 原有的 row padding、
NZ weight、可用的 MatMul+AllReduce 和 `npu_add_rms_norm`。因此一个只资格化 M64 的
profile 会意外拖慢其他所有 shape。

现在 `NativeRMSNorm.can_forward_mc2()` 会在 decoder 分支前同时检查 vendor capability、
projection bias 和精确 profile 资格。只有真正有资格进入融合 kernel 的 shape 才调用
`forward_mc2()`；其他 shape 原路执行 `NativeRowLinear` 和 native AddRMSNorm。相应测试覆盖
qualified、unqualified、vendor unavailable 三类分支。benchmark 还将
`fused_attempt/success/fallback/exception` 四项 MC2 counter 纳入 worker 测量快照，令
“参数已开启”和“完整 model-runner 确实命中”可以被区分。

首先使用 Qwen3-32B TP3 production model-runner 做 M64 短测：64 个请求、每请求 4 个
token、ACLGraph。MC2 与 split 路径的输出 SHA256 完全一致；三个 target rank 均记录
`attempt=192, success=192, fallback=0, exception=0`。测量区间每个 target rank 的 graph
replay 从 4 增至 8，capture/failed capture/shape fallback/capacity fallback 均不增长。
MC2 host counter 在捕获后保持不变是预期行为，因为设备 graph replay 不会重入 Python。
该短测的 MC2/split E2E 吞吐比为 0.9835x，只用作“真实模型命中、输出一致、图重放”证明，
不作为吞吐收益结论。

随后在同一源码、同一四张物理卡 `4,5,6,7` 上完成在线正式回归。共同合同为：作者
463 请求 Poisson/RPS4 workload、tight/normal/loose=`271/96/96`、TPOT SLO=`40/50/150 ms`、
B64、固定输出 256 token、online prefill coalesce1、固定 gamma4、图模式测量区间禁止回退。
candidate 为 Qwen3-0.6B TP1 draft + Qwen3-32B TP3 target；baseline 为 Qwen3-32B TP4
target-only。warm-up 与图资格验证不计入测量时间。baseline 第一次启动命中了共享目录的
旧 AOT cache，并在初始化阶段因 device/meta 不匹配失败；该次没有结果且不计入性能。正式
baseline 使用隔离 fresh cache 重编译并通过 graph gate。

| 配置 | Raw tok/s | 论文口径 SLO 达成 | Goodput tok/s | Raw / TP4 | Goodput / TP4 |
|---|---:|---:|---:|---:|---:|
| TP4 target-only baseline | 929.934 | 188/463（40.605%） | 377.597 | 1.0000x | 1.0000x |
| SpecSLO TP1+TP3，MC2 off | 920.218 | 235/463（50.756%） | 467.065 | 0.9896x | **1.2369x** |
| SpecSLO TP1+TP3，M64 MC2 on | 919.838 | 220/463（47.516%） | 437.072 | 0.9891x | **1.1575x** |

两组 candidate 都达到 graph qualification fixed point。MC2-off 的三个 target rank 的四项
MC2 counter 全为 0，测量期每 rank graph replay 增加 1935，所有 capture/fallback delta
均为 0。MC2-on 的三个 target rank 各有
`attempt=256, success=256, fallback=0, exception=0`，测量期每 rank graph replay 增加
1871，所有 capture/fallback delta 同样为 0。这证明 on/off 对照不是由 eager 图回退造成。
两次运行前后 HEAD 和 working-tree patch 逐字节一致。

MC2-on 相对 off 的 Raw 比为 0.99959x（-0.041%），在单次运行噪声尺度内；Goodput 比为
0.93578x（-6.42%）。两次 candidate 的输出 hash、接受轨迹和 decode round 不同，而
Goodput 又在 40/50 ms 阈值处离散跳变。因此不能把单次 6.42% 直接解释成 kernel 的稳定
负收益，但它已经足以说明：**M64 算子正收益尚未转化成系统级收益，当前 profile 不能成为
生产默认。** 在完成多次重复和更多真实高覆盖 shape 的资格化以前，系统性能候选应保持
MC2 off；MC2 保留为 fail-closed 的实验路径。

本轮也没有达到最终目标：当前最佳候选 Goodput 为 TP4 baseline 的 1.2369x，尚低于 1.3x；
Raw 为 0.9896x，尚未超过 baseline。下一步应先对当前最佳 on/off/baseline 做重复测量，
再采集完整 model-runner 的真实 shape 覆盖，只有在真实输入、数值门和整机 A/B 均为正的
shape 才可加入 MC2 profile；同时继续压缩 draft/sync/state-update 开销和 tight 请求尾延迟。

完整报告与可审计运行材料位于
`/root/data/nano-pearl-benchmark-results/20260922-specslo-mc2-modelrunner-online-tp1tp3-vs-tp4/README.md`；
正式 on/off/baseline 的 result、gate、日志、源码 diff/hash 与 workload/profile SHA 均保存在
该目录的 `runs/` 子目录。复现入口为同目录的 `run_candidate.sh` 和 `run_baseline.sh`。

### 6.19 2026-09-22：MC2 真实 shape 覆盖、数值轨迹与理想全命中系统诊断

6.18 证明了 M64 MC2 能在完整 model-runner 中命中，但没有回答三个关键问题：在线运行
究竟主要使用哪些 M、融合与 split 的完整生成轨迹是否一致、以及 target 算子收益在 100%
命中时能否转化为系统收益。本轮补齐了 replay shape telemetry、完整生成一致性检查和严格
eager 真实输入采集，并基于真实 Qwen3-32B 层输入完成扩展资格测试。结论是：**MC2 仍不应
成为生产默认；但现有证据不能判断 target 是否已退出设备关键路径。** 旧 host timeline
只能证明提交窗口 overlap，后续 NPU Event 在另一合同下反而显示 target 可消费窗口更长；
因此 draft/target 的设备关键路径归属必须用同一当前合同的 CANN/Event 重新确认。

#### 6.19.1 三项诊断能力

1. ACLGraph runner 现在分别记录 generic/target replay 的 entry-key 和物理 token-row
   直方图；只计 replay，不把 capture 混入。worker 暴露累计计数，benchmark 在测量前后做
   按 rank 的深层差分，因此结果可以直接回答“正式测量区间每个 M 重放了多少次”。当前
   SpecSLO target 实际走 generic graph runner，所以应读取
   `aclgraph_generic_replay_token_rows_histogram_delta`，而不是空的 target 专用字段。
2. `examples/check_specslo_mc2_generation_consistency.py` 会重新计算完整 token-row SHA256，
   检查运行合同，统计发生差异的请求数和逐位置 token 数，并定位最早差异；若源 artifact
   带 target logits/top1/top2，还能继续检查 argmax margin。旧 artifact 没有保存 logits，
   因而本轮只作 token 级诊断，不虚构 margin。
3. 真实层输入采集是严格 opt-in 的诊断路径：必须同时设置
   `VLLM_ASCEND_PEARL_MC2_CAPTURE_DIR` 和
   `VLLM_ASCEND_PEARL_MC2_CAPTURE_ROWS`，且只允许 enforce-eager。探针在每个 TP rank、每个
   请求 M 只原子写入一次 `activation/weight/residual/gamma`，输出可直接交给
   `examples/measure_specslo_mc2.py --input-dir`；默认配置零采集，ACLGraph capture 中会明确
   拒绝，性能测试前必须移除两个环境变量。这避免了把图捕获期间的 host I/O 或 synthetic
   输入误当成生产证据。

#### 6.19.2 作者 463 请求在线 workload 的真实 replay shape

在 6.18 的 MC2-off 合同上重新运行作者 463 请求、RPS4、B64、固定输出 256 token、
TP1+TP3、gamma4 和在线 prefill coalesce1。三个 target rank 的直方图逐项一致：每 rank
`1,938` 次 generic replay、`50` 个实际 M，capture/shape/capacity/eager fallback 的测量增量
均为 0；draft rank 的该 generic target 直方图为空。完整分布如下，格式为 `M:replay 次数`：

| M 范围 | 每 target rank 的实际分布 | 小计 | 占比 |
|---|---|---:|---:|
| 4--32 | `4:23, 8:24, 12:11, 16:10, 20:34, 24:5, 28:11, 32:30` | 148 | 7.64% |
| 36--64 | `36:13, 40:16, 44:35, 48:19, 52:14, 56:23, 60:37, 64:29` | 186 | 9.60% |
| 68--100 | `68:23, 72:35, 76:48, 80:52, 84:51, 88:72, 92:64, 96:74, 100:63` | 482 | 24.87% |
| 101--128 | `101:1, 104:80, 108:70, 112:63, 116:60, 120:45, 124:43, 128:156` | 518 | 26.73% |
| 133--325 | `133:3, 165:8, 197:26, 229:14, 261:42, 293:37, 325:46` | 176 | 9.08% |
| 357--645 | `357:33, 389:32, 421:24, 453:36, 485:7, 517:17, 549:19, 581:34, 613:75, 645:151` | 428 | 22.08% |

原 M64 profile 实际只覆盖 `29/1,938=1.50%` 的 target graph replay，解释了为何算子级正收益
难以在整机吞吐中出现。扩展后的 M64/76/84/88/92/100 profile 按该次直方图最多覆盖
`327/1,938=16.87%`；这是 exact-M replay 覆盖率，不等于端到端时间占比，也不能据此线性
外推系统收益。原始直方图和完整运行合同位于：

`/root/data/nano-pearl-benchmark-results/20260922-specslo-mc2-modelrunner-online-tp1tp3-vs-tp4/runs/candidate-tp1tp3-fixed256-no-mc2-shapehist-run1/result.json`。

#### 6.19.3 完整生成轨迹与 off/off 可重复性

一致性工具对 6.18 的在线 MC2-off/on 结果重新计算的 SHA 与 artifact 内保存值一致。两者
共有 `397/463` 个请求不同，逐位置差异 `56,136` 个 token；最早 completion offset 差异在
请求 116 的第 4 个 token。该证据位于：

`/root/data/nano-pearl-benchmark-results/20260922-specslo-mc2-modelrunner-online-tp1tp3-vs-tp4/mc2-generation-consistency.json`。

但是，原 MC2-off 与本轮 shape-telemetry MC2-off 的跨次在线复跑同样出现
`400/463` 个请求、`56,066` 个逐位置 token 差异。因此在线 Poisson 到达、异步双 batch
调度和阈值判定下，生成/接受轨迹会随跨次时序分叉，不能把单次 off/on 输出不同全部归因于
MC2 数值误差。该对照在
`/root/data/nano-pearl-benchmark-results/20260922-specslo-mc2-modelrunner-online-tp1tp3-vs-tp4/mc2-off-repeat-generation-consistency.json`；
由于 checker 的主合同原本要求 off/on 开关不同，它将 off/off 文件标记为非可比，以上
请求/token 数是其中独立的完整输出直接比较结果，而不是把该文件误报为通过。

相反，后述静态 M100 的两个 MC2-off 复跑输出 SHA 都是
`f315ab463016abba3c17232eaaec6463a75a5601ce16e2bd2e28cfe77ae8e345`，所有 token 完全一致。
这说明在固定 shape、固定输入和无在线调度分叉时，split 路径本身可重复；MC2-on 的 SHA 为
`11360561d381d346096dd72513db2add878ea04ff11fa868844fb4d0261aaf6c`，与 off 有
`34/100` 个请求、`628` 个逐位置 token 差异。后续逐 row 审计发现差异具有固定 block
结构，不能再只归因于正常 BF16 融合舍入；详见 6.19.5。这是当前 MC2 晋级的 correctness
blocker，仍需逐层、逐 row logits margin 才能定位第一个数值翻转。

#### 6.19.4 真实输入资格结果

严格 eager 采集得到 M76/80/84/88/92/96/100/104/124 的三个 rank payload；每个 payload
均为真实 Qwen3-32B attention output projection 输入，而非 synthetic tensor。采集目录和
采集运行分别为：

- `/root/data/tp3-mc2-debug-20260922/qwen3-real-inputs-dominant-shapes-v102/`；
- `/root/data/tp3-mc2-debug-20260922/qwen3-real-inputs-dominant-shapes-v102-capture-run.{json,log}`。

这些被采样 payload 的算子级数值测试都通过 `atol=0.01, rtol=0.01` 的 Norm/Add 容差及
changed-input replay 门；性能仍按 p95 至少 `1.05x` fail-closed。21 组正式交错 A/B
结果为：

| M | p50 加速 | p95 加速 | mean 加速 | 决策 |
|---:|---:|---:|---:|---|
| 64 | 1.07732x | 1.06480x | 1.07902x | 保留为实验算子候选（6.17 结果） |
| 76 | 1.08205x | 1.07916x | 1.08020x | 保留为实验算子候选 |
| 80 | 1.01153x | 0.99875x | 0.99234x | 拒绝 |
| 84 | 1.07389x | 1.06668x | 1.07292x | 保留为实验算子候选 |
| 88 | 1.07872x | 1.07619x | 1.08049x | 保留为实验算子候选 |
| 92 | 1.13012x | 1.13395x | 1.12934x | 保留为实验算子候选 |
| 100 | 1.10848x | 1.11566x | 1.13081x | 保留为实验算子候选 |

M96/104/124 在 7 组真实输入 screening 的 p95 分别只有 `0.76213x/0.92675x/0.94711x`，
因此未浪费 NPU 进入 21 组正式门，实验候选/算子资格 profile 继续拒绝这些 shape。逐 shape
原始 JSON/log 位于
`/root/data/tp3-mc2-debug-20260922/qwen3-real-dominant-shapes-formal-v103/` 和
`/root/data/tp3-mc2-debug-20260922/qwen3-real-dominant-shapes-screen-v102/`。合并后的 exact-M、
identity-bound、fail-closed **实验候选/算子资格 profile** 为：

`/root/data/tp3-mc2-debug-20260922/qwen3-m64-m76-m84-m88-m92-m100-production-v104-qualified.json`。

需要特别说明：artifact 文件名历史上含 `production`，但该 profile **没有完成生产晋级**。
当前采集器对每个 M/rank 只保存首次命中的 payload，然后从 pending rows 删除该 M；所有
decoder layer 又共享同一个采集器。因此 v104 很可能只采到了同一层，而不是覆盖 Qwen3-32B
的 64 层。审计证据是旧 M64 与新 M76/M84/M100 的 rank0 `weight` SHA256 都为
`434ef094b177040609801a1536a6c0beace9d05ca7e2f26797509351365ed2ff`，`gamma` SHA256 都为
`93d7d7b7a322934951d8e918e695fb9dddcdf88586e6d2f0e4686a0b1303da87`。而当前 profile key
绑定硬件、TP、M/K/N、dtype、weight format、epilogue 和源码 hash，却不绑定 layer，命中后
会应用于全部 64 层。也就是说，上表证明的是“一个真实层 payload 上的算子资格”，不是
“64 层真实输入分布均已资格化”。

按每个已保留 M 的 p95 单层差值、64 层以及 6.19.2 的真实 replay 次数作纯乐观外推，
新增 M76/84/88/92/100 的总节省只有 `219.08 ms`，M64 约 `12.48 ms`，合计约
`231.56 ms`；相对该次约 128.9 s E2E 仅约 0.18%。这个上界还假设 64 层都获得被采样层的
收益、忽略调用开销，并把可能与 draft overlap 的 target 局部节省全部算成墙钟收益，因而
不能作为预期加速比。后续必须至少采集首/中/尾层和观测最坏层，或逐层覆盖全部 64 层；
资格门要么绑定 layer，要么采用跨层最坏值。最终生产门还必须加入完整 model-runner 的
logits margin/token 等价性，而不能只依赖单层 Norm/Add 容差。

#### 6.19.5 M100 理想全命中系统 A/B

为了消除在线 shape 稀疏和调度分叉，进一步构造 target-only、B100/P100/T64、图模式的
M100 诊断；它不是 SpecSLO 双 batch Goodput 测试。测量区间每个 target rank 都有 64 次
M100 generic replay，图 fallback 为 0；MC2-on 在图建立时每 rank 记录
`attempt=192, success=192, fallback=0, exception=0`，随后设备 replay 不重入 Python，因而
这是调度覆盖意义上“精确 shape 100% 命中”的理想上界对照，不预设生成正确性通过。

| 配置 | inference tok/s | E2E tok/s | 输出 SHA |
|---|---:|---:|---|
| MC2-off 第 1 次 | 1645.416 | 1597.570 | `f315ab46...e345` |
| MC2-off 第 2 次 | 1676.665 | 1626.220 | `f315ab46...e345` |
| MC2-on | 1652.150 | 1602.705 | `11360561...1aaf6c` |

on 相对 off1 的 E2E 只是 `1.0032x`，而 off2/off1 自身已有 `1.0179x` 波动；on 相对 off2
为 `0.9855x`，相对两次 off 吞吐中点为 `0.9943x`。因此不能挑选 off1 宣称 MC2 有稳定
系统收益。

更重要的是，这次 100 行输入使用相同 prompt，输出却呈现可复现的 row/block 模式。两个
MC2-off 运行各自都只有两种序列：主序列 98 行，row 16 和 66 为另一序列；两次分布和 token
逐字一致。MC2-on 同样只有两种序列，但分布变为 68+32：row 0--15 和 50--65 两个连续
16-row block 使用第三种序列，其余 68 行与 off 的 98 行主序列一致。34 个 off/on mismatch
中，32 个恰好集中在这两个 16-row block，另外两个正是 row 16/66。相同 prompt 的 off
路径自身在 row 16/66 产生异序列，已经提示原 graph/task 的 row-specific 可疑行为；on 的
固定 16-row block 翻转进一步表明这不是可以直接接受的普通 BF16 随机舍入。**因此 M100
即使 100% exact-shape 命中，其当前 A/B 仍是 correctness blocker，只能用来否定稳定系统
收益，不能晋级。** 下一步必须做 eager/graph × MC2-off/on 四象限，并记录逐层、逐 row
logits/top1/top2 margin 和首个 token 分叉，区分 graph/task row mapping 问题与融合算子
数值问题。

完整结果位于
`/root/data/nano-pearl-benchmark-results/20260922-specslo-mc2-static-m100-ideal-v104/`，其中
`mc2-off.json`、`mc2-off-repeat.json`、`mc2-on.json` 保存输出、计数和图直方图，
`generation-consistency.json` 保存 off/on token 差异。

旧 profiling 只能提供 overlap 线索，不能解释为已确定的设备关键路径。2026-09-14 的正式
host timeline（本轮没有重测）记录 cycle wall `42.466 ms`、draft host-submit window
`36.006 ms`、target host-submit window `14.223 ms`，两者交集 `14.217 ms`。这些 rank-local
时间戳没有加入 NPU synchronize，只证明 Python/worker 的异步提交窗口几乎完全重叠，不能
证明 target 设备执行已被 draft 完全隐藏。该 host 证据位于
`/root/data/nano-pearl-benchmark-results/20260914-specslo-fixed-gamma4-full-window/PROFILING.md`。

同一时期技术报告后续使用跨 cycle NPU Event 修正了设备口径：在 P128/T256、排除
mixed-target-prefill 的 454 个 cycle 中，四步 draft device window 为 `33.493 ms`，target
forward + device verdict 的可消费窗口为 `37.256 ms`，后者反而长 `3.763 ms`。该结果的
batch/输出长度与本轮 M100 不同，且 target 数值含 verdict、不能分离纯 target forward，
所以也不能直接证明 MC2 位于关键路径；但它足以否定“host target 只有 14.223 ms，因此
target 一定不关键”的强推断。证据位于
`/root/data/vllm-ascend-hust/docs/specslo_fixed_gamma4_optimization_report_20260915_zh.md`
的 13.1 节。当前“target 局部收益可能被 dual-batch overlap 吸收”只能作为待验证假设，
必须在同一当前源码、同一 B/P/T、同一图合同下用 CANN trace 或 device Event 对 MC2-off/on
直接验证。

M100 是 target-only 诊断，本身没有 draft 可以隐藏 target 收益；其未出现稳定增益应首先
由更直接的证据解释：单算子收益占全图比例很小的 Amdahl 稀释、off/off 波动已大于理论
信号、固定 row/block 输出 correctness blocker、单次 on 样本以及不完整的 source/env
合同。它不能作为“draft 遮住 MC2”的证据。

因此生产默认继续保持 MC2-off，v104 仅保留为实验候选/算子资格 profile，MC2 只保留为
严格 profile 控制的实验路径。TP1 四步 full-chain draft 仍值得独立优化，但不再把
`36.006 ms` host-submit window 称为已经确认的设备关键路径；在决定 draft 与 target 的
优化优先级前，应先完成当前合同下的 draft/target CANN/Event 分界和 MC2-off/on target
尾部 A/B。同时必须先关闭上述跨层泛化和固定 row/block 输出两个 correctness 缺口。

### 6.20 2026-09-23：MC2 epsilon 根因、bit-exact 修复与 full-chain taskless 门禁

对 M100 做 eager/graph × MC2-off/on 四象限后，图路径已经可以排除：graph-on 的三个
target rank 均为 `fused_attempt=192, fused_success=192, fallback=0, exception=0`，图捕获、
changed-input qualification 和 128 次 replay 都没有失败或容量/shape fallback。输出分叉却
同时出现在 eager-on 与 graph-on，并始终随 MC2 开关变化，因此不是 ACLGraph 捕获或 replay
错误。该四象限原始结果位于
`/root/data/nano-pearl-benchmark-results/20260923-specslo-mc2-aiv-global01-v112/four-quadrants/`。

随后把 custom epilogue 拆到 reduction、均值、`rstd`、gamma 前后舍入逐项核对。TP3
projection 已与 `HCCL_DETERMINISTIC=true`、`HCCL_OP_EXPANSION_MODE=AIV` 的
`BF16(BF16(rank0 + rank1) + rank2)` 全局树逐 bit 一致；真正差异来自 RMSNorm 的
`epsilon`。custom `rstd` 在 `epsilon=0/1e-6/1e-3` 三次调用中完全不变，并且恰好与 native
`epsilon=0` 一致。将 host tiling 的固定 PP 槽直接写成 `1e-6` 后，AIV 输出立即恢复与
native bit-exact，证明 reduction、sqrt/div、gamma 顺序、tiling 槽传输和 AIV 读取均正确；
故障点是 CANN 9.0 MC2 executor 的 optional float attr 在 tiling context 中暴露为 0。

生产修复采用显式、fail-closed 合同：

1. full epilogue 只序列化已资格化的 `epsilon=1e-6`；
2. Torch adapter 对 non-projection 的其他 epsilon 在进入 ACLNN 前直接 `TORCH_CHECK`，上层
   可捕获并退回 split matmul+HCCL+native RMSNorm，不能静默忽略调用参数；
3. projection-only 不执行 custom RMSNorm，仍允许任意 epsilon；
4. MC2 profile 继续绑定 source SHA、CANN/HCCL 版本、HCCL deterministic/AIV 模式和
   TP rank 到物理 NPU 映射。

v119-production 的 fresh-process 数值门结果为：真实 Qwen3-32B M100/K3072/N5120 的
norm 与 add_out 最大误差均为 0；8 次 changed-input replay 仍为 0；projection 在 3 个 rank
上各 `512000/512000` 个 BF16 元素逐 bit 一致。21 组受干扰样本的中位数为 split
`0.8245 ms`、custom `0.6724 ms`，即约 `1.2261x`；但采样末段物理 NPU6/7 被外部容器接管，
split/custom 均出现毫秒级尖峰，所以该轮只采信 correctness 和中位趋势，不作为正式 p95
或系统性能结论。证据位于
`/root/data/nano-pearl-benchmark-results/20260923-specslo-mc2-epilogue-diagnostics-v113/real-m100-production-v119/result.json`。

TP1 fixed-gamma4 draft 的代码审计同时确认：当前已经是一次 host 调用、一个 ACLGraph 捕获、
一次 graph replay 内串联四个自回归 forward，并非四次独立 decode 提交。host CANN
PagedAttention 仍需每次 replay 刷新 `4 步 × 28 层 = 112` 个 task；device-position PA 路径
则能走 taskless replay，删除这些 update/event gate。benchmark 已新增 measurement-minus-
warmup 的 task-update/taskless/PA workspace 净计数，report provenance 也绑定
`device_paged_attention.py` 的 SHA256，避免把预编译期工作或未跟踪源码变化混入正式结果。

独立 device-PA 正式数值门使用 GQA=2、head-dim 128、跨 128-token KV page、tile64。
eager 对 CPU float64 参考实现最大绝对误差为 `2.585243e-4`，首次 graph replay 与 eager
逐 bit 一致。随后只原地修改 device position、保持 query 不变并在不重捕获的情况下 replay，
对参考最大误差为 `1.958870e-4`，相对旧输出最大变化 `0.0402832`；再修改 query 的第三次
replay 对参考最大误差为 `2.387244e-4`。这比同时修改 query/position 的旧检查更严格：输出
变化不能再由 query 掩盖，能够直接证明图内读取动态 position，而非复用捕获时长度或发生
图回退。证据位于
`/root/data/nano-pearl-benchmark-results/20260923-specslo-device-pa-single-npu-qualification/official-position-only-result.json`。

本节仍不宣称系统收益。下一步必须在同一组空闲物理卡上完成 attention 与首/中/尾层
down-proj 的跨层最坏值资格化，再做完整 graph model-runner 的 MC2 AB/BA 复跑；随后对
host-PA 与 taskless device-PA 做至少三轮交错 A/B，要求输出/verification-rounds 等价、
测量段零捕获/零 fallback，并用净计数证明 host 路径为 `112 × replay` task update、device
路径为 `taskless == replay` 且 task update/workspace 均为 0。生产默认仍保持两个实验开关
关闭，直至数值和系统门同时通过。

### 6.21 2026-09-23：MC2 系统收益为何被稀释，以及 v119 新门禁

对已有 v105 CANN trace 做逐 rank、逐 step 重算后，可以排除“MC2 在 graph replay 中根本
没有执行”这一假设。三个 target rank 的三个有效 decode step 中，每 step 精确出现 64 个
MC2 MIX，同时各减少 64 个 MatMul、HCCL AllReduce 和 AddRMSNorm；shape/capacity/eager
fallback 均为 0。单层被替换窗口的设备时间为：split 三个 kernel 合计 `123.262 us`，两段
launch gap 合计 `21.596 us`，完整窗口 `144.858 us`；MC2 MIX 为 `134.026 us`。所以局部净
收益约 `10.832 us/层`，64 层约 `0.693 ms/step`，确实进入了 model-runner。

但该版 custom kernel 的纯执行时间反而比三个 split kernel 合计慢 `8.73%`，收益仅来自少
两个 launch gap。完整 trace 中 64 层模型主体只缩短 `0.396 ms/step`；第一层之前的 rank
到达/embedding collective 前缀却增加 `1.087 ms/step`，最终 kernel span 反而增加
`0.704 ms/step`。此外 profiler 本身使 graph-off/on decode 分别增加 `27.14%/20.20%`，甚至
会改变 A/B 快慢方向。因此 trace 只用于证明路由和解释局部机制，正式吞吐必须使用无
profiler 的交错 ABBA。

历史版本只融合 attention `o_proj`。按 v112 的单算子 p95，64 层理论节省仅
`1.090 ms/step`，相对 `57.597 ms/step` baseline 的理想上限约 `1.0193x`。旧 v107a 的
layer-0 down-proj 再贡献约 `1.552 ms/step` 后，attention+down 的乐观估计也只有约
`1.0481x`；它仍是有 epsilon 数值缺陷、且只覆盖一个 down 层的旧数据，不能当生产结果。
要达到系统 `1.3x`，约需从同一 baseline 节省 `13.292 ms/step`，当前双 shape 估计只覆盖
该需求的约 `19.9%`。结论是 MC2 是必要的 TP3 局部优化，但单靠现有 kernel 不可能承担
SpecSLO 的全部 1.3x 目标；后续仍需 full-chain draft、taskless device PA、双 batch overlap
和 rolling eager 的系统收益。

为防止再次把 warm-up capture 或污染样本误认为生产收益，本轮新增两层 fail-closed 门禁：

1. `benchmark_nano_pearl_native_target_only.py` 支持在唯一 batch point 的 warm-up 后封闭
   graph cache；测量阶段若发现新 shape/capture 必须失败，不能动态补图；
2. `aggregate_specslo_mc2_profiles.py` 可先对同 shape 的首/中/尾真实层做保守 envelope，
   再把互不重叠的 attention/down exact shape 合成生产只加载一次的 profile。跨组重复
   shape、路径、内容 hash、input source 或 source/runtime identity 不一致都会拒绝。

新的 v119 真实层 runner 位于
`/root/data/nano-pearl-benchmark-results/20260923-specslo-mc2-v119-real-layer-qualification/`；
system ABBA runner 位于
`/root/data/nano-pearl-benchmark-results/20260923-specslo-mc2-v119-system-ab/`。两者都不运行
占卡程序、不清理其他进程，并在每次启动前要求指定物理卡被 `npu-smi` 明确报告为空闲。
当前其他容器已占用所有可用三卡组合，因此这里只完成 runner、schema 与 CPU 门禁，尚未
产生 v119 的系统吞吐结论。

### 6.22 2026-09-23：真实深层数值树、v122/v123 隔离候选与 vendor 身份门禁

v119 的新一轮真实层诊断覆盖 attention `M100/K3072/N5120`，以及 Qwen3-32B down
projection 的 layer 0/31/63 `M100/K8576/N5120`。该轮执行期间仓库源码被并行任务修改，
所以四份结果的 source SHA 不一致，整轮不能作为正式 qualification；runner 已正确拒绝
聚合。但每个独立 fresh-process 子运行仍暴露出两个可重复定位的问题：

- attention：split P95 `0.138955 ms`，fused P95 `0.152742 ms`，当前 fused kernel 未过
  P95 性能门；
- down layer 0：split/fused P95 为 `0.177383/0.154593 ms`，数值门通过；
- down layer 31：三 rank projection 各有 `65/512000` 个 BF16 mismatch，最大误差
  `0.001953125`，added scaled error `1.5625`；
- down layer 63：三 rank projection 各有 `33/512000` 个 mismatch，最大误差 `0.125`，
  added scaled error `50`。

因此 v119 的全局固定 `01→2` BF16 归约树只是在 attention/layer0 输入上偶然逐 bit 一致，
并不等价于当前 deterministic HCCL。已有 HCCL probe 证明 M100/N5120 的 flat output 以
64 elements 为单位分成六段，树依次为
`01→2, 02→1, 12→0, 01→2, 02→1, 12→0`，边界为
`85376, 170688, 256064, 341376, 426688, 512000`。

为隔离 correctness 与性能变量，已生成两个不修改 production tree 的 vendor：

1. `v122-hccl-chunk-tree-correctness-probe`：只在 v119 上恢复六段 HCCL 树，保留 v119
   epsilon、native RMS、同步协议和行映射；M=1..160 的边界/tile CPU 覆盖及 CANN 离线
   编译通过。
2. `v123-hccl-tree-row-balance-probe`：在 v122 上再叠加 physical-core-first 行映射；
   M100 从 17/20 个活跃物理 Core、最大 6 行/核，变为 20/20 个 Core、严格 5 行/核。
   组合 ownership/segment/tile CPU 穷举及独立 CANN 构建通过。

两者都尚未通过设备端 layer31/63 exactness、changed-input ACLGraph 和 ABBA P95 门，故未
合入生产、未宣称吞吐收益。下一步顺序固定为：先以 v122 证明深层 projection 三 rank
mismatch 全部归零，再比较 v122/v123，只有 correctness 相同后才把 row-balance 的差值归因
为性能收益。

本轮还修复了 profile provenance 缺口。旧 profile 的 `source_sha256` 只覆盖工作树源码，
在不改源码、仅切换 `ASCEND_CUSTOM_OPP_PATH` 的隔离 vendor 时无法区分实际 kernel。现在
runtime binding 新增 `vendor_payload_sha256`，内容绑定实际 active vendor 的 OpAPI、
tiling/proto 库、MC2 headers/sources、manifest 和 kernel objects；profile 加载与每次
qualification 都会 fail closed。production-v119、v122、v123 在同一主源码下得到不同 digest，
防止后续把候选 kernel 的结果错误归到 production-v119。

关于“graph 模式是否跳过 MC2”，当前证据仍不支持该假设：旧 device trace 中 graph-on 每
step 精确出现 64 个 MIX，并各减少 64 个 MatMul/HCCL/AddRMSNorm。稳定 replay 不重入 Python，
所以 measurement MC2 host counter delta 为 0 是预期语义。新的四象限门禁不再只看
`fallback=0`，还要求 capture resident dispatch 为 attention-only `192/rank` 或
attention+down `384/rank`、封图后 counter/entry 不变、完整 token 一致，并最终用设备 DAG
确认 custom kernel 数量。这样可以区分“profile 未命中而在 adapter 前走 split”和“图中真正
录入并 replay MC2”。

### 6.23 2026-09-23：生产 AIV oracle 更正、bounded replay 与本地投影隔离

6.22 节的六段 HCCL 树结论来自**没有设置** `HCCL_OP_EXPANSION_MODE=AIV` 的探针，不能作为
SpecSLO 生产 oracle。生产合同固定为 `HCCL_DETERMINISTIC=true` 与
`HCCL_OP_EXPANSION_MODE=AIV`。新的 full-world 探针使用四个进程：global rank 0 仅作为
TP1 draft 协调者，global rank 1/2/3 在物理 NPU 5/6/7 上组成 target TP3 subgroup。对真实
Qwen3-32B layer-31 down projection 的 M100/K8576/N5120 输出，5 次重复在三个 target rank
间逐 bit 一致，整个 512000-element 张量唯一精确树均为
`BF16(BF16(rank0 + rank1) + rank2)`；不存在六段切换。证据位于
`/root/data/nano-pearl-benchmark-results/20260923-specslo-mc2-world4-tree-probe/target567-down-l31-m100.json`。
因此基于非 AIV 六段 oracle 的 v122 不能用于生产；生产通信方向回到 v119 的全窗口
`01→2`，后续只在明确绑定的 target 5/6/7 物理映射上资格化。

旧的 1024 次 changed-input graph 门禁还有一个独立口径错误：每次 replay 都在上一轮输入上
继续加 `delta`，rank 2 到最后一轮相对真实激活累计接近 `+48`，把 layer-31 的小误差放大为
`added=2.0`。现在每轮先恢复真实输入，再按 `0,+1,-1,+2,-2` 循环施加有界 offset；相邻
replay 仍必定改变静态输入地址的内容，但任何 rank 都不会偏离真实输入超过两个增量。对应
CPU 单测覆盖 1024 次 offset 的有界性与相邻变化。重测后 down 的 changed-input 最大
`norm/added` 误差分别降为 `0.0009765625/0.0078125`，与初始真实输入同阶，不再随 replay
次数增长。

在生产 target 5/6/7 上，v119 M100 operator-only ABBA 的当前结果为：

| shape | split 中位数 | fused 中位数 | 中位加速 | full-output 数值 |
|---|---:|---:|---:|---|
| attention K3072 | 0.134605 ms | 0.131428 ms | 1.0242x | norm/add 均逐 bit exact |
| down K8576 layer31 | 0.170687 ms | 0.150784 ms | 1.1320x | norm 0.0009766，add 0.0078125 |

原始结果位于
`/root/data/nano-pearl-benchmark-results/20260923-specslo-mc2-production-target567-v119/`。
这解释了为什么算子已经更快、系统收益仍可能很小：attention 热点只缩短约 2.4%，down
热点约 13.2%，进入 64 层完整 forward 后还会被其余算子和调度同步按 Amdahl 定律稀释。
这些数字仍只是 captured operator replay，不是完整 model-runner 或 Goodput 结论。

为了判断 down 的剩余差异是否由 graph、HCCL 或本地 MatMul 引起，新增 single-source
隔离门：每次只保留一个 TP rank 的真实 activation，其余两个 rank 输入全零；普通
`torch.linear + HCCL` reference 在第一次 MC2 launch 前全部物化；eager 与 graph 都执行
A/B/A/B 有界静态输入更新；每个 source 后插入轮换的全零 sentinel；同时要求三 rank 共识、
同状态跨 replay hash 稳定以及 A/B 输出确实变化。结果如下：

- attention K3072：24/24 个非零用例逐 bit exact，8/8 个 zero sentinel exact；
- down K8576：24/24 个非零用例均为
  `local_matmul_or_prepublication_candidate`，8/8 个 zero sentinel exact；
- down 的三个 source rank 在 state A/B 下分别有 `320/140`、`136/161`、`97/196` 个
  mismatch，最大误差不超过一个 BF16 ULP；eager 与 graph 完全一致。

证据位于
`/root/data/nano-pearl-benchmark-results/20260923-specslo-mc2-local-isolation/`。这排除了
graph 读取旧输入、跨 replay 漂移、rank 间发布不一致以及 ordinary HCCL 丢失单一 source；
范围已经缩小到 Catlass 本地 MatMul 或写入 symmetric window 前的 publication 路径。
attention 的 K3072 可整除当前 Catlass L1 K256，而 down 的 K8576 存在 128-element 尾块，
所以先构建隔离的 K128 tiling 对照；长期候选是改用 CANN 官方
`Mc2MatmulBaseKernel/MatmulV3` 生成本地 projection，同时保留已经证明正确的 rank-3 AIV
`01→2` reduction。两条路径都必须先过 single-source bit-exact、world4 target subgroup、
changed-input graph 和 ABBA 延迟门，再进入完整 model-runner。

此外，MC2 静态路由启动已删除 target-HCCL 与 WORLD-HCCL 的交叠 admission collective，
改为所有 rank 同序创建的专用 world Gloo group，并用一次 `all_gather_object` 汇总 runtime、
manifest、target group identity、digest 与所有本地异常。冻结后 route 丢失不再回到动态
qualification；四类输入张量必须匹配已资格化的 contiguous layout；显式启用 MC2 却没有
任何 qualified production row 会在 graph runner 创建前 fail closed。该改动的相关 CPU
回归为 608 项通过，但 full-world 真机 admission、current-source profile 和完整四象限仍需
在 v124 数值方案确定后重跑。生产默认继续保持 MC2-off。

### 6.24 2026-09-24：v131 图内 chained epilogue 与三次系统 A/B

为直接回答“MC2 是否在图模式下失效”，本轮把 rank-3 native epilogue 接到显式
`int64[64,4]` mailbox chain state，并将 allocation-free chained ABI 纳入 source hash 和
qualification。图模式只有在 profile 同时证明 changed-input bit-exact、当前物理 TP 映射、
当前 adapter/vendor/source，以及 `standalone_chained_flush=true` 时才允许进入；eager 模式
不会误入仅为 ACLGraph 设计的 chain。完整模型每个 epilogue 都复用预分配 state 并
`flush=True`，因此没有 Python 侧零张量分配，也不会把未完成 READ_DONE 带出 graph replay。

whole-model deferred READ_DONE 也做了 64 层、128 次 epilogue 的真机验证。数值、changed
input 和最终 mailbox 归零都通过，但性能不稳定：target 0/1/2 上 deferred 相对 fully-flushed
的 p95 为 `1.000878x`，target 1/2/4 上为 `0.994943x`。production profile 因此明确不纳入
`deferred_read_done_chain`；默认只保留逐 epilogue flush 的安全路径。

在物理 NPU `0,1,2,4`（draft TP1 + target TP3）上，以 Qwen3-32B target-only、M100、
100 prompts、每请求 64 output token、100 warm-up prompts（不计时）进行了三次
fresh-process ACLGraph off/on A/B。结果如下：

| 重复 | off Raw | on Raw | Raw 比 | off E2E | on E2E | E2E 比 | decode 比 |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 1605.344 | 1662.000 | 1.035292x | 1560.365 | 1612.254 | 1.033254x | 1.024814x |
| 2 | 1628.731 | 1643.182 | 1.008872x | 1583.140 | 1594.402 | 1.007114x | 0.997828x |
| 3 | 1616.532 | 1652.342 | 1.022153x | 1572.029 | 1603.382 | 1.019944x | 1.009940x |
| 中位数 | 1616.532 | 1652.342 | **1.022153x** | 1572.029 | 1603.382 | **1.019944x** | **1.009940x** |

六次运行的 6400-token SHA256 完全一致。每次 on 的三个 target rank 均有
`chained attempt/success/flush = 384/384/384`，fallback/exception 为 0；每 rank 只有一个
M100 graph entry、127 次 replay，graph 已 seal，eager/shape/capacity fallback 全为 0。
因此 MC2 已被实证录入 resident graph，当前小收益不是 graph 回退造成的。

旧设备 trace 对同一 M100 forward 的归因也与系统结果相符：split epilogue 约
`83.28 us/op`，MC2 epilogue 约 `73.72 us/op`，局部约 `1.13x`；但该段仅占 target kernel
duration sum 约 21.29%。Amdahl 上限约为完整 forward 的 2%--4%，三次中位数 Raw
`1.022x` 正落在该区间。下一阶段如果继续提升 TP3，必须减少 MatMul 本体，或找到能够跨映射
稳定获益的通信/计算重叠；只继续压同一个 epilogue launch 不可能承担系统级 1.3x。

完整原始数据与说明位于
`/root/data/nano-pearl-benchmark-results/20260924-specslo-mc2-v131-deferred-chain/MC2_V131_GRAPH_AB_ZH.md`。

### 6.25 2026-09-24：v133 single-response mailbox 候选

v131 的 response record 已用同一次 32-byte DMA 发布 producer、echoed request 及两个互补
校验字段，但 `ChunkReadyRendezvous()` 仍连续远端读取两次并要求两次完全相同。v133 删除
第二次读取，只在四字段不变量成立且 echoed request 等于当前 invocation request 时接受；
request/response/receipt/read-done 四阶段、唯一 invocation token 和 BF16 归约树均未改变。

在同一 target 0/1/2 映射上，attention fused P95 从 v131 的 `0.133173 ms` 降至
`0.107631 ms`，相对 split 从 `1.062418x` 提升为 `1.263700x`；down fused P95 从
`0.168121 ms` 降至 `0.143363 ms`，相对 split 从 `1.039987x` 提升为
`1.208094x`。两个 shape 的初始结果和 64 次 changed-input graph replay 均逐 bit exact。
在 target 1/2/5 上也重复通过 64 次 changed-input，P95 分别为 `1.243242x/1.250120x`。

完整 model-runner 的前两组无明显外部干扰 A/B 中，Raw 分别提升
`1.038703x/1.034361x`，E2E 提升 `1.036410x/1.032332x`，decode 提升
`1.029541x/1.025151x`；输出 token SHA 完全一致，三个 target rank 均有 chained
`384/384/384`、127 次 resident graph replay、零 fallback/exception。第 3 组开始出现外部
负载，第 4 组 off Raw 更从约 1635 跌到 772 tok/s，故后两组原样保留但不参与收益统计。

本轮同时修复 non-chained `forward_mc2()` 重建 tuple 而破坏 legacy object identity 的兼容
问题，并把 profile builder 改为要求 attention/down measurement 的 source SHA 与当前
adapter、model wiring 和 kernel source 完全一致。由于该兼容修复改变了 source hash，已有
v133 profile 会被 runtime 和 builder fail closed；必须在空闲卡上重做 1024 次 changed-input
资格后才能晋级。qualification/system runner 现已加入逐物理卡 `npu-smi proc-mem` 预检，
指定卡不被明确报告为空闲就拒绝启动，且不使用占卡程序。

详细数据位于
`/root/data/nano-pearl-benchmark-results/20260924-specslo-mc2-v133-single-response-target125/MC2_V133_SINGLE_RESPONSE_ZH.md`。

当前源码的 CPU 定向回归为 `583 passed`，Ruff 和 `git diff --check` 通过。旧 v133
measurement 也已实际送入当前 profile builder，因 source SHA 不匹配被拒绝且没有写出
profile；因此在新一轮真机资格化完成前，生产路径继续 fail closed，而不是沿用陈旧档案。

随后在用户明确允许共享低负载卡后，target 4/5/7 完成 attention/down 各 1024 次
changed-input graph replay，均为三 rank 逐 bit exact，并生成 source/vendor/adapter/物理映射
闭环的新 profile。过程中发现 raw measurement 未携带 `source_sha256`，安全 builder 因字段
缺失拒绝；已修复测量脚本后完整重测，没有修改旧 JSON。共享 A/B 三组的 Raw 比分别为
`1.074178x/0.980414x/0.912317x`，方向矛盾且绝对吞吐仅 775--965 tok/s，故不用于性能归因。
但所有输出 SHA 相同，三个 target rank 每次均为 chained `384/384/384`、127 次 resident
graph replay、零 fallback/exception，进一步证明完整模型图内命中和数值正确。原始数据位于
`/root/data/nano-pearl-benchmark-results/20260924-specslo-mc2-v133-single-response-target457-shared/`。
