# SpecSLO 当前优化与实现状态

更新时间：2026-09-09

> 本文为当时的优化记录。2026-09-10 已根据论文与新增树主循环重新审计，
> 发现实际 verify 超预算、树模式同步依赖和动态准入等功能缺口。
> 最新状态见 [第五章及核心功能审计](specslo_section5_audit_20260910_zh.md)。
> 其中“必须跨多轮异步才符合论文”的旧解释已更正：论文要求同一步内并发。

这份摘要只记录当前代码和实机回归能够证明的内容。吞吐探针、功能路径和
论文目标分别列出，避免把一次较快的实验结果误写成已经完成的 SpecSLO。

## 一、已经落地的优化

### 1. 执行与缓存路径

- draft/target 支持异构 `TP1 + TP3`，使用同一 HCCL world，并隔离
  verification/correction 通信组。
- proposal 携带 request、proposal、epoch、home、width 和 confidence，接收端
  会拒绝过期或错位消息，减少回滚后的状态污染。
- target 使用 Ascend paged attention、128-token KV page、slot/block table 和
  prefix/page 回收；FIA 作为显式 fallback。
- ACLGraph/NPUGraph 使用稳定 shape bucket；图容量或 shape 不满足时回退 eager，
  不在运行时无限 capture。
- Qwen3 BF16 复用生产 `qkv_rmsnorm_rope`，Qwen2/Llama 复用生产 RoPE 路径；
  词表不同时在 logits 比较前做逻辑裁剪。
- TP3 的任务队列、HCCL AIV expansion、确定性归约、CPU/NUMA 绑定和阶段计时
  已加入可控开关。

### 2. SpecRhythm 控制面

- online-prefill 不再提前计算未来请求；ready 请求以 packed prefill 一次进入
  服务，减少每请求一次 target prefill 的启动开销。
- acceptance EMA、draft confidence、progress gap、urgency 和
  `verification_roof` 已接入 budget shaper，支持按紧迫度分配 normal/eager
  proposal。
- rolling eager 的完整接受会 promotion，拒绝会 invalidate；proposal epoch
  校验和 KV 状态更新在同一提交边界内完成。
- 支持 merge-ready-homes、priority burst、target verification 行数上限、稳定
  graph、连续 batch、抢占和完成行替换。
- 已生成论文三类 workload 的可复用 manifest：HumanEval/Alpaca/CNN-DM，比例
  按 `6:2:2`，并保存 arrival、SLO、gamma 和 seed 元数据。
- 增加显式 `spec_rhythm_verification_budget=B` 配置；设置后 shaper 的 target
  verification envelope 固定为 B，不再由 `batch_size * gamma` 推导。分层 urgency
  分配后会填充剩余可用候选容量，并在 worker counters 中记录
  `unused_verification_tokens`；只有请求上限或显式 draft-token cap 不足时才会低于 B，
  不再静默把 gamma 当作 B。

### 3. 数值与回归保护

- stepwise target 路径修复了“旧 attention metadata”问题；每步输出 clone，避免
  graph replay buffer 覆写已经提交的 token。
- normal/eager proposal 的 prefix epoch、request id 和 home 路由均有校验；拒绝
  后缀不进入 committed frontier。
- 小规模验证结果：`p8, gamma=4, max_tokens=32` 的 SpecRhythm 输出与 native
  target-only 为 `8/8` 一致；`p64, gamma=1` 为 `64/64` 一致。长路径的 packed
  `p64, gamma=4` 仍有少量行与 native target-only 不同，因此没有将它标为最终
  correctness gate。

## 二、当前实机结果的正确解读

### 已观察到的性能收益

此前的 merged-home、固定 graph、gamma=4 快速探针达到约 `312.7 tok/s`，其 e2e
Goodput 约 `112.4 tok/s`；这是一个“控制面/执行路径探针”，不能直接作为最终
论文结果，因为 p64 长 speculative rollback 仍有输出差异，且在线 baseline 的
请求级时间戳还不完整。

### 最近的 SLO 批量上限实验

在 Qwen3-0.6B（TP1）+ Qwen3-32B（TP3）、64 请求、gamma=4、max_tokens=32 的
论文混合 manifest 上：

| 配置 | e2e 吞吐 | SLO 达成率 | e2e Goodput |
| --- | ---: | ---: | ---: |
| merged-home，早期快速探针 | 约 312.7 tok/s | 35.9% (23/64) | 112.39 tok/s* |
| target cap=32 | 约 279.6 tok/s | 20.3% (13/64) | 56.79 tok/s |
| target cap=16 | 约 258.4 tok/s | 20.3% | 52.49 tok/s |

\* 早期快速探针仍有 p64 packed rollback 的数值差异，且使用的 baseline 时间窗与
后续 rerun 不同，只能作为性能上界探针，不能作为最终验收结果。

因此“单纯把 target forward 切小”已经被实验证明不是当前 SLO 瓶颈，默认不启用
该限制。紧/常规请求仍然主要受 TP3 target forward、verdict、broadcast/state
边界影响。

## 三、尚未完成、不能冒充已完成的部分

1. **跨 round 的异步 mailbox 深度**：native 两个 worker 已在同一轮分别启动
   draft(home B) 和 target verify(home A)，并在 HCCL verdict 边界汇合；仍未实现
   多个 round 同时在飞的无阻塞 mailbox/stream，因此当前是“一轮内 overlap”，不是
   完整的深流水线。
2. **TP3 MC2 kernel**：910B2 当前 `socversion=2201` 拒绝 rank-3 fused
   `matmul+all-reduce`，代码已自动回退普通 HCCL；没有证据表明该硬件上可以直接
   获得 TP3 fused kernel 收益。
3. **树状 SpecSLO 的生产链路**：fixed-shape tree budget、draft spine/top-k proposal、
   ancestor mask target forward、设备 verifier、proposal-id rollback/promotion 和两侧
   KV compaction 已接入 native SpecRhythm 主循环，并由
   `spec_rhythm_tree_width/depth > 1` 显式启用。树 verifier 现在保留每个候选节点的
   target 输出并沿实际接受分支选择 bonus；树 eager continuation 记录父树主干和
   draft frontier，只有依赖一致才 promotion。Qwen2.5-0.5B + Qwen2.5-14B、TP1+TP3
   的 910B2 短 smoke 已完成：树模式生成、ACLGraph capture/replay、KV compaction、
   target/correction 两组 HCCL verdict 和 eager promotion 均通过。尚未完成的是在
   所有 Qwen/Llama 结构、长上下文、多请求及生产 workload 上的端到端数值和吞吐矩阵；
   短 smoke 不替代完整性能验收。
4. **公平 Goodput 基线**：当前 target-only baseline 的旧服务对象没有完整的每请求
   first/last token 时间戳；在补齐 baseline metrics 前，不能把离线吞吐或不稳定的
   online 输出直接换算成论文的 1.3x Goodput。

## 四、后续讨论建议

下一步需要先确定验收优先级：

1. 以逐 token 数值一致为硬门槛，先把 p64 gamma=4 的差异缩到 0，再测性能；或
   允许 packed path 作为独立性能探针，safe stepwise path 作为 correctness path。
2. 明确 SLO 计时是 decode-stage TPOT 还是包含 arrival/admission 的 e2e Goodput；
   两种口径的目标和上界不同，不能混用。
3. 若优先达成论文目标，工程重点应放在异步 mailbox、target verify 与 HCCL
   broadcast overlap，以及按 tight/normal/loose 分层的 urgent target 服务，而
   不是继续堆叠 gamma 或缩小 target batch。
4. 在同一 manifest、同一 arrival trace、同一 warmup 口径下，为 baseline 和
   SpecSLO 都记录 per-request timestamps、accepted tokens、TPOT 和 Goodput，之后
   再给出 80% attainment 与 1.3x 的结论。
