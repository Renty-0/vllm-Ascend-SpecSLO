# SpecSLO 核心闭环修复与实机回归（进行中，2026-09-10）

本记录承接 `specslo_section5_audit_20260910_zh.md` 的缺项审计。SpecSLO 是论文
SpecRhythm 的 Ascend 实现，不把普通 PEARL 的吞吐成绩记作 SpecSLO 成绩。
本轮 native greedy 核心闭环的四卡功能、独立祖先数值、严格表在线消耗与设备重叠
验收已取得通过证据；这不是全部产品能力或性能目标完成。此前要求的 1.3× Goodput
仍未达成，以下区分实现、测量与待验。

## 本轮落实的核心机制

- 实际 ready 验证集合按总候选 B 重新打包；不是只限制新增 draft，也不是按下一轮 gamma
  估算本轮验证量。动态 B 缩小时，树提案祖先安全修剪，线性提案保留已提交输出后重建。
- 逐请求紧迫度/接受率的初步分配后，跨请求按真实节点置信度做两阶段、祖先闭合的树细化。
  target 物理输入仅为选中节点加每请求的 root，不重新补回被丢弃的树节点。
- target(A) 模型计算在 draft(B) 候选交换前执行；normal/eager draft 按层合并计算。
  W 根据实际角色计算时间校准，未校准时不额外安排 eager，探索后丢弃的节点也计入 draft 成本。
- 双槽遵守在线到达和容量限制；完成后补入较轻槽，不再把全部输入直接当 active。
  补入 prefill 占用的时间计入旧 active 请求，不计入新请求首 token 之前的 decode 时间。
- 完整验证后才提交；eager promotion 检查父提案 ID、epoch、完整依赖路径及 frontier。
  被拒绝的 eager scratch KV 不成为已提交前缀；非连续物理 KV 页采用真实槽映射搬移。
- `PEARLEngine.generate(on_token_commit=...)` 在 guarded commit 后逐步交付；回调失败排空工作
  进程消息后报告错误，不遗留消息破坏下一次请求。该接口不是完整 HTTP 服务实现。
- 最后一轮交付按 EOS/max_tokens 截断；完成请求不再 promotion。原 TPOT 口径保留，论文
  `T_decode/N_out` 另外输出 `paper_tpot_ms`，不把两种达成率混用。
- 离线测量、聚合和严格表加载已连接：校验模型、TP、硬件、模式、批量/上下文及实际 target
  请求数。未测量 key 不伪装为经过 profile 的 gamma 回退。
- 第 5.3 节新 eager 分配增加 `a_need > 0` 条件：仅 urgency 比率高但没有进度欠账，
  不再从剩余预算获得新的 eager draft。W 窗口按 `a_need × acceptance benefit` 截断；
  normal 的剩余预算优化、已生成且依赖合法的 continuation 验证/promotion 不受此门控影响。
  此修复联合 17 个文件的 465 项 CPU 测试通过；最终冻结版本的 NPU 联合验收另记。

## 已完成的硬件测量

环境：Ascend 910B2；物理卡 0、2、3、4；Qwen3-0.6B TP1 + Qwen3-32B TP3。
原始 JSON/log/trace 保存在：
`/root/data/nano-pearl-benchmark-results/20260910-specslo-core-regression/`。

以下是 **FP32 masked-attention 实验前**的功能样本，不是最终性能成绩；GSM8K 前 8 条，
batch=4、max_tokens=64、固定 B=8、树宽 2 深 3、warmup=0，均输出 512 tokens：

| 路径 | E2E 输出吞吐（tokens/s） | 验证结果 |
| --- | ---: | --- |
| vLLM-Ascend TP4 eager baseline | 22.103 | 独立 target-only 对照 |
| SpecSLO TP1+TP3 eager | 6.769 | 完成；未取得对 baseline 的加速 |
| SpecSLO TP1+TP3 graph | 8.242 | 与该 eager 的全部输出 token 相同 |

两种 SpecSLO 模式相对 baseline 均有 2/8 请求分歧：请求 5 的 token 27、请求 6 的
token 61（均从 0 计）。不能仅凭张量并行布局不同就将它们归因于 BF16。

- 同 TP3 的 normal/scratch 逐节点因果路径核验：56/56 argmax 一致；修改自己或父树的
  非依赖兄弟节点，受保护节点 logits 全部逐元素不变。
- 该核验的全词表 `atol=0.01, rtol=0.001` 未全部通过，结果文件明确标为失败。
  重新建立 prefix KV 与原始逐步生成的 KV 不是同一份快照，不能据此宣布原分歧已解释。
- 增加增量 teacher-forcing prefix 后，两个分歧位置分别观测到候选 1052/1449 的
  BF16 logit 同为 38.0、候选 594/752 同为 34.5。存在平局敏感性证据，但仍需分离
  head tie-break 与前缀计算误差，不能据此放宽所有 logits 检查。

## 图边界失败及修复

4 条请求、在线容量 2、B=1/4、32 tokens、prompt 长度 127/128/255/256 的回归发现：
跨页时 draft rank 0 的一次 final KV hidden 图失败。原始 context 为 256–260，图 bucket
为 512；graph 与同 bucket eager **完全相同**，与原始 eager 的 hidden 最大误差为
0.1875（仅第 4 行受影响），没有 NaN。因此这次证据指向 padding 改变 attention 数值，
不是图回放本身不同于同形状 eager。

现已补齐失败时恢复 eager KV（不只返回旧 hidden），并让 masked dense tree attention
在 eager/graph 两条路径都使用 FP32 SDPA、输出写回原 dtype。原校验阈值与参考路径不变。
普通 paged/FIA 未修改。冻结快照 `specslo-core-snapshot-t65xc9` 的完整回归已经完成，
**FP32 实验没有解决问题**：B4 eager 的边界请求 0 在输出 23 出现分歧；B1/B4 graph
边界测试仍触发校验失败。graph 与同 bucket eager 仍完全相同，但原始形状与 bucket
的 hidden 最大差异达到 0.4375。B 是候选验证预算，实际在线容量为 2，不能称作 batch=4。

进一步诊断没有支持“直接归因于 NPU 精度”的结论：

- 26 组、76 行真实 NPU greedy 算子测试中，`max`、`argmax`、FP32 argmax 与显式
  最小 tie ID 均相同；尚未发现 tie-break 算子错误。
- 两组独立 SDPA 实验各 960 个 case；当前 bool-mask FP32 SDPA 在 40 种唯一输入
  形状下写回 BF16 后均 padding-invariant，未复现真实模型的较大漂移。
- 手写 FP32 attention、additive mask 在独立样本中反而出现 BF16 差异，不作为修复。
- 正在抓取真实模型首个分歧层的 Q/K/V 与实际在线 KV，而不是继续放宽验收容差。

## 批量 FIA 树 attention 原型

复用本地 vLLM-Ascend 的 FULL-mask 契约：TND、paged KV、`sparse_mode=1`、
`inner_precise=2`、四维请求掩码。只对 mask 外形补齐，不给 query 增加假节点。
实机原型 `tree-fia-operator.json` 的 32 个正常样例全部通过独立 CPU float64 oracle，
有限大值兄弟 KV 扰动不影响受保护节点；34 次 Graph 回放全部与同形状 eager 逐元素相同，
包括变更上下文、mask、非连续页和 query 分配 `[2,5]→[5,2]`，Q=1 也覆盖 TND/BNSD。

原型整体仍标为失败：被屏蔽 KV 人为注入 NaN 后会传播 NaN。正常有限值通过不代表任意
异常缓存均被 mask 隔离；生产接入必须初始化未写入缓存，不应使用 `nan_to_num` 隐藏
有效节点异常。这是算子原型证据，尚不能当作整模型 FIA/Graph 回归通过。

### 整模型 FIA 回归：8/8 通过

随后冻结 `specslo-fia-snapshot-lp9zS0`，在 Qwen3-0.6B TP1 + Qwen3-32B TP3
四卡上运行 `fia-lp9zS0-boundary-v2.json`：在线容量 2、排队请求 4、每请求 32 tokens，
验证预算 B=1/4、eager/Graph，以及 GSM8K source/127、128、255、256 跨页输入。
**8/8 组通过全部检查**，包括输出 token 一致、到达准入、容量、实际验证预算、
流式交付与图校验。此前 FP32 dense 实验失败的 B4 边界输出此次与 B1 完全一致。

| Graph 样例 | draft capture / replay | 每个 target rank capture / replay | 图校验失败 |
| --- | ---: | ---: | ---: |
| B1 source | 2 / 211 | 1 / 70 | 0 |
| B1 边界 | 0 / 217 | 0 / 69 | 0 |
| B4 source | 4 / 161 | 4 / 46 | 0 |
| B4 边界 | 0 / 164 | 0 / 51 | 0 |

所有上述样例的图容量/形状回退也为零。B4 source 记录 eager promotion 15 次、
invalidation 5 次；B4 边界 promotion 2 次、invalidation 3 次，均完成合法提交。
这次不再仅依赖“开启 Graph”开关作为证据；也不把 source 后的缓存复用阶段计作独立
冷启动性能成绩。独立数值参考、更新后的严格预算表及最后异常 guard 仍需联合验收。

冻结实现 SHA256：

- native_engine：`fe24e5114a2d589c1c4ce2283282d3afcb4e3de72e53e0a538b1f0b484638809`
- native_model：`37e2b3e21c873b9f23aae06f40ad0dde37bde11493e6f2a0a00f6d40115c9906`
- native_graph：`6cbd09880a160d9e11c172a0ae5d4d5fecbcffcd082d15d6492ff0cb24bd0e2b`

### 独立数值检查仍不能宣布全部通过

`fia-lp9zS0-tree-numerics.json` 用相同 TP3、normal/scratch 的两组真实前缀，
对照 dense/普通 PA 的祖先路径：56 组比较中 55 组 argmax 相同，只有 1 组全词表
满足原定 `atol=0.01, rtol=0.001`，因此文件整体仍为失败。dense 对照 27/28
argmax 一致，PA 对照 28/28 一致；不能用 argmax 一致代替全词表精度验收。
失败 argmax 位点的 FIA 是 752:34.5、594:34.25，dense 是两者同为 34.5。
这不等于证明历史分歧仅由 tie-break 引起。非依赖兄弟节点扰动仍全部逐元素不变。

下一步独立 oracle 不复用生产 mask 构建器，并拆成同形状 FIA（严格逐元素相等）和
祖先子集 FIA（不同 query 形状、保留原精度容差）两类，分离 mask/KV 错误与计算形状差异。

该独立验收随后完成：`fia-5K2rjm-independent-AB-v3.json` **通过**，所有 target
rank 的 A/B 失败票均为零。使用前述两组真实 token 前缀，但明确构造新的非连续
祖先闭合选择 `[0]` / `[0,1,3,5]`，物理 q=[2,5]；不是把原失败 case 重命名成成功。

- A：normal/scratch × 独立 eager、生产 Graph、独立 Graph 六组，hidden、全词表
  logits、所有层 query KV、已提交前缀与 scratch 依赖 KV 全部逐元素相同。
  独立 oracle 从 CPU 父节点列表遍历，不调用生产 mask builder；同时检查实际生产 metadata。
- B：normal/scratch 各 7 个祖先子集，共 **14/14**，不同物理 query 数 1–4 下，
  hidden/logits 的最大绝对误差均为 **0**；既满足原容差，也没有 argmax 差异。
- 所有候选 token 扰动后，两请求 root 的输出仍逐元素不变，真实前缀 KV 未被修改。

此诊断的 Graph 范围是 Transformer hidden；LM head 在相同形状下于图外执行。
包含 greedy head 的整段服务 Graph 证据来自另列的 8/8 回归。两者不能混写。
v1 启动配置和 v2 诊断树拓扑曾错误，分别在模型计算前/实际数值比较前被拒绝；日志保留，
已用真实配置构造、生产 build/pack/scratch/metadata 的 CPU 回归防止复发。
上述通过排除了本次同 FIA 对照中的 mask/祖先路径错误，不撤销 dense/PA 跨后端失败，
也不保证任意输入、不同 TP/后端的 BF16 输出始终逐位相同。

### 异常值提交屏障：真实目标 rank 故障注入通过

冻结 `specslo-fia-guarded-v2-Cn7D2n` 的
`fia-Cn7D2n-nonfinite-target-rank2.json` 已完成，三个阶段均通过：

- 正常阶段 2 条请求、`max_tokens=1`，实际交付 2 个首 token。
- 在 target rank 2 的有效 `lm_head.weight[0,0]` 注入 NaN 后，四个 rank 均报错，
  输出与流式回调均没有交付 token。检查覆盖首 token，不能被“没有 decode step”绕过。
- 原权重精确恢复后，已污染缓存的 sticky 状态仍阻止下一次提交；普通 release 不清除污染。

生产保护覆盖 Q/K/V、attention、最终残差/归一化和有效词表 logits；KV 初始化为零，
不使用 `nan_to_num` 隐藏有效输入异常。此项不是随机采样数值验收，也不是吞吐成绩。

## 实测 roofline 的约束

真实模型、active batch=2、target 实际请求数=1 的 Graph 采样已完成。上下文 256 时，
AR 中位时延约 38ms，1/2/4 候选约 57/72/102ms；上下文 512 时约 38/63/79/113ms。
以相对 AR 增量不超过 10% 为约束，当前分离 dense 路径没有合格候选预算。
聚合程序明确失败退出，没有生成虚假的合格表，也没有用 gamma 回退冒充测量结果。

上述是旧 dense 路径。新增 FIA + 完整异常保护的冻结版本 `Cn7D2n` 实测中，
active batch=2、target 实际请求数=1，上下文 256 的 AR/1/2/4 候选中位数分别约
41.56/43.63/44.15/44.73ms。上下文 128/256/512 的每形状 7 次样本均保留，
按 P95、相对 AR 增量不超过 10% 聚合得到 **`2:1 → B=4`**；键的第二项是
`ceil(context/512)`，不是上下文 token 数。全部计时 forward 都实际回放 Graph，
无计时内 capture。表记录真实 FIA/AR 后端及 native model 源文件 SHA256。

在线尾部 active batch=1 的首轮 7 次样本中，上下文 512 的 Q=1 P95 超过约束，
因此初版 `fia-Cn7D2n-roofline-online-cap2.json` 只含 `2:1`，将 `1:1` 记录为未合格。
随后预先固定每形状追加 50 次，合并原始 7 次与全部新增样本；不删除离群点、
不提高 epsilon、不把 batch=2 的预算复制给 1。新表
`fia-Cn7D2n-roofline-online-cap2-all57.json` 得到 **`1:1 → B=1, 2:1 → B=4`**，
无未合格 key。batch=1 每形状 57 个样本、batch=2 每形状 7 个样本，数量不混写。
这使容量 2、上下文不超过 512 的尾部动态预算有了实测依据；实际在线消耗仍另行验收。
这些是离线容量探测，warmup/capture 不计入探测窗口；并非用户 E2E/Goodput 计时结果。

随后给 active=2 补测 `[3]` 候选形状及配对 AR，三个 context 点各 7 次。
聚合保留全部四份原始文件，最终 `fia-Cn7D2n-roofline-online-cap2-full-shapes.json`
仍为 `1:1 → 1, 2:1 → 4`；active=2 的 `[1]/[2]/[3]/[4]` 均有实测证据。
本实验 collector 和实际服务的 `max_model_len` 均为 1024；目前 profile v1 身份校验
未绑定 FULL-mask 的这一容量维度，不能将本结果推广为任意 mask 容量的性能保证。

最新调度冻结目录为 `specslo-fia-guarded-v3-5K2rjm`，模型/图源码与上述采样一致，
engine SHA256 为 `2697ea9996676061be32bc00a8610a7e18839a2c2f69e55ec3848f9e4a86eff5`。
该版本包含 CPU tree plan 与第 5.3 节 gap/W 修复；24 个相关文件联合 **572 项 CPU 测试通过**。

### 完整保护与调度修复合入后的再次实机回归

`fia-5K2rjm-boundary-final.json` 已正常退出，**8/8 组再次通过**。与前述 lp9zS0
同样的容量 2、请求 4、B=1/4、T32、source/跨页、eager/Graph 条件，输出 token 全部
与 eager B1 参考一致；加入完整 finite guard、CPU tree plan 和 gap/W 修复没有破坏结果。

| 新冻结版 Graph 样例 | draft capture / replay | 每 target rank capture / replay |
| --- | ---: | ---: |
| B1 source | 2 / 220 | 1 / 70 |
| B1 跨页 | 0 / 218 | 0 / 69 |
| B4 source | 4 / 161 | 4 / 46 |
| B4 跨页 | 0 / 162 | 0 / 51 |

所有图校验失败、形状回退及容量回退均为 0。B4 Graph source 的 eager
promotion/invalidation 为 15/5，跨页为 1/2。B4 eager 本次没有触发 eager continuation，
这是实际时延/W 门控的结果，不伪造覆盖；合法 promotion/拒绝覆盖来自 Graph 样例。

另在 `fia-5K2rjm-nonfinite-draft-rank0.json` 将 draft 最后一层 MLP 的有效权重
注入 NaN，正常、污染、恢复权重但保留污染状态三个阶段均通过预期断言。
异常阶段四 rank 全部在首 token 前拒绝提交；与 target rank 2 head 注入互补。

### 严格表驱动的动态在线验证通过

`fia-5K2rjm-strict-online.json` 已完成，4 条合成请求、在线容量 2、输出限制
8/10/12/14、错峰到达、相同 40/50/150ms SLO。实际消费上述完整实测 profile，
不是固定标量 B，也不使用 gamma 回退补齐缺 key。

- 三个 target rank 各执行 **15 次实际验证**，key、候选形状、context 和图执行序列一致。
- 实际出现 `2:1 → B=4` 与 `1:1 → B=1`，候选形状 `[4]/[1]`，根节点额外计入
  物理 query；尾部安全修剪 **3 个未提交候选**，没有丢失已提交输出。
- target 每 rank capture/replay=2/15，draft=2/50，failed/shape/capacity fallback 全 0。
- 3 次 prefill 接纳全部 4 请求，容量、arrival、实时 stream 和终止事件检查通过。
- 在线观测结束、移除观察器后，用同模型、同 FIA、相同 token 输入和输出限制另跑固定 B1
  参考，**4/4 输出完全一致**。参考不进入在线预算证据，也不是计时前隐藏 warmup。

这里明确使用 512-token 桶查表，实际在线 context 并非恰好等于采样点；报告保留
`actual_context_was_sampled` 区分精确采样与桶内覆盖，不称作每个长度都单独实测。

## 真实设备重叠

使用官方 CANN 绝对时间轴，对两组已完成的 eager-B1 trace 离线解析。严格 AI_CORE
事件交集分别为 0.409/0.342ms；包含 MIX_AIC 时为 1.620/1.010ms。确实存在不同 NPU
的计算重叠，但 Cube 覆盖率很低，不等于高效流水线。

详细口径、原始 kernel 例子、时间校准限制与复现脚本见结果目录的 `DEVICE_OVERLAP.md`。
不能用几百毫秒的 host span 交集替代设备计算交集，也不能把修复前 trace 归到修复后路径。

### 当前 FIA / Graph / CPU plan / gap-W 版本的真实时间轴

新目录 `fia-graph-overlap-QMmSGj/` 保存本次四 rank 原始 trace、复制后官方离线解析结果、
源码 SHA 和 `device-overlap.json`。一个持久 Graph engine 先 source 后跨页；采第二段
3 个 profiler active step。该跨页 case draft replay=162、每 target rank=51，
没有新 capture、没有图校验失败/回退。Graph-only 采集未另跑 eager，对照明确标 skipped；
数值依据是前述独立回归，不以采集脚本的 skipped 项当通过。

下表均使用约 **221.17ms 的共同设备观察窗口**，只取 `Ascend Hardware` 的真实
kernel 区间，每设备先合并多 stream，再求交；排除 CPU/HCCL/HCOM/等待/拷贝。

| target 物理 NPU | 纯 AICORE 交集 ms | AICORE+MIX_AIC 交集 ms | 后者占 draft Cube busy | 后者占 target Cube busy | 全设备计算交集 ms |
| --- | ---: | ---: | ---: | ---: | ---: |
| 2 | 3.340 | 4.774 | 44.50% | 11.16% | 11.063 |
| 3 | 3.332 | 4.747 | 44.26% | 11.15% | 11.034 |
| 4 | 3.359 | 4.768 | 44.45% | 11.21% | 11.085 |

draft 的 Cube busy=10.727ms，target 约 42.55–42.76ms；纯向量与 MIX 分类、
实际重叠 `aclnnMatmul_MatMulCommon_MatMulV2` 样例均在 JSON 中。
不同 target rank 的交集不能相加。CANN 使用绝对时间轴，不平移各 rank 首事件，
没有独立测量跨设备时钟误差上界。结论是**当前版本确有设备计算 overlap**，但 target
Cube 大部分时间仍未被 draft Cube 覆盖，也不能将旧 eager/B1 与新 Graph/B4 当成
仅改变一个开关的性能 A/B，或从这个片段宣称 Goodput 1.3×。

## 尚在验收的内容

1. 已完成：最后冻结版 8/8、同 FIA 独立祖先 logits、两角色异常注入、真实 B4→B1
   表驱动在线输出对照、四 rank Graph 设备时间轴。覆盖范围和配置见各节，不能泛化到任意模型。
2. 跨 dense/PA 后端的全词表误差仍保留为失败；上述同 FIA 通过不保证不同后端逐位一致。
3. 论文 workload 的 RPS2/4、容量8/16/32/64、同口径 TP4 baseline/SpecSLO Goodput
   与 80% TPOT 达成率仍须重新测量和优化，不能复用旧 PEARL 成绩。
4. 当前入口是 native SpecSLO API；完整 vLLM V1 HTTP 双模型 worker 生命周期集成不属于
   本次 native 服务循环回归，不能对外描述成所有 vLLM serving 接口均已无缝支持。
5. 运行证据、文档和仓库提交的对应关系仍需收尾记录。

树 verifier 目前明确限定 greedy，非零 target/draft temperature 在分配缓存前报错；
普通 PEARL 原有采样路径不受影响。论文未公开具体实验温度，不能把 greedy 验收称作
随机树采样完整迁移，也不能将这个软件缺项归因于 NPU。

## 功能冻结追加：V1、随机树与 schema v2

上段是当时的准确历史状态。随后树路径已实现 target/draft temperature、top-p、top-k
采样及跨 rank token 同步，不再限定 greedy。真实 Qwen3 TP1+TP3、temperature=0.8、
ACLGraph HTTP 生命周期回归通过；报告为
`20260910-specslo-functional-freeze/qwen3-http-graph-nongreedy-lifecycle-v2.json`。
其 worker 冷启动 192.45s，在线生成 6.68s，Graph capture/replay=2/59，两个请求各有
一次 live admission，最终 completed/failed/inflight=2/0/0。

同时完成文本生成范围的 `SpecSLOV1EngineCoreClient`、OpenAI-compatible HTTP/SSE、
disconnect abort 和持久 worker shutdown。TP3 MC2 按真实 Qwen3-32B 形状完成验证，
所有形状因非有限/超限误差或慢于分离路径被拒绝。dense/PA 与 FIA 的 BF16 raw logits
严格 allclose 失败继续保留；同一 FIA packed shape 的 Graph/eager 和独立 A/B 则逐元素
一致，不能混淆这两种断言。

严格 roofline 现在是 schema v2，额外绑定 `max_model_len/tree_width/tree_depth`。
此前本文件记录的 B1/B4 只证明控制面和动态缩小逻辑，不能复用为最终 `B_roof`。
最终 B 必须在所有工程冻结后，按论文 §5.3 对 batch/context/候选总数做真实 NPU sweep；
完整协议见 [功能冻结记录](specslo_functional_freeze_20260910_zh.md)。
