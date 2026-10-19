> 2026-10-18 16:30 UTC：266完整实验与独立审计完成，Raw649.08、Goodput414.96，296/463达标（tight104/271），三路target均463/463逐token相等；原预算修正执行2398次，本轮没有触发混合B3/4升级，因此不能把收益归因于升级。仍不及正确257版本Goodput554.76，未达到644.77目标，不采用为最终候选。267缩短KV范围＋4096哨兵的256组合成测试中128组长上下文不满足所需不变性，且长上下文更慢，暂不接入。268成对收益估计仅HOST原型；269主链4＋侧链1..4的B1..8目标端协议与独立参考已通过1252组HOST检查（83632节点），尚未NPU验证、未实现真实draft侧链、没有新Goodput结果。继续准备独立目标端数值实验270。

> 2026-10-18 15:23 UTC：继续动态 B＋树状优化。266 已冻结并启动：恢复实际218预算修正，融合分支 KV 原始字节拷贝；模型仍为 BF16、M120/S24/KV2560。160组实际预算修正与227组主机协议检查通过，完整模型与463请求逐token验证进行中。265三维缓存三路拷贝对照全部344组一致、融合较快；264未发现掩码10/11的稳定明显惩罚。以上算子结果不能当作完整Goodput收益。历史260输出全一致但Goodput397.95且预算修正审计失败，未采用。目标仍未达到。

# SpecSLO 数据格式 逐请求 TPOT 和优化汇总

<!-- ROOT_CURRENT_DYNAMIC_TREE_STATUS -->
14:50 UTC 最新进度：目标仍未达到。ROOT260完整463测量为 Raw **644.24**、Goodput **397.95 token/s**，**286/463** 达标（tight **94/271**，normal与loose各 **96/96**），低于ROOT257的554.76。三张target均463/463输出与独立同后端own-AR完全一致，原启动/组件、新鲜227节点与封图检查完成，独立完整计算审计通过。但收尾监督检查失败：新LIVE预算代理绕过了原cost218混合B3/B4修整器，实际full形状调用记录缺失，旧修整行为也未执行。因此执行状态保留失败，未采用，不能把退化单独归因于扩大B5。下一完整候选必须保留实际原修整器及计数闭包，不能补写计数伪装执行。

ROOT262复制融合原型在Triton/BiShengHIR的16位指针转换编译处失败，无算子正确性或耗时结论；退出及环境/源码核验通过。新ROOT263已冻结2480成员并启动独立合成K/V字节复制测试，使用现有batch_memcpy的标量地址转uint8指针方式；核验两种行宽下全部172种Native分叉映射、全部65536种16位数据模式、同缓冲修改/恢复ACLGraph重放及80组交替图计时。该测试不加载真实模型、不执行attention/解码，无Goodput或神经正确性资格，未接入完整模型。环境未更新，所有新结果在私有目录。

ROOT257仍是完整执行与审计通过的M120参考：Raw **686.78**、Goodput **554.76**、374/463达标，三张target各463/463输出一致，仍低于强MC2 TP4的1.5倍门槛 **644.77**。当前树B4父路径未实现eager晋级，不计rolling-eager收益。最终同卡三组九轮仍待完成。


截至 2026-10-18 07:17 UTC，ROOT237 已完成完整463请求和独立同后端自回归：实测 Goodput/Raw 均为 **640.17 token/s**、耗时185.149秒，TPOT达标463/463；但只有 **305/463条输出完全一致，158条存在token差异**，因此结果**未合格、未采用，目标未达到**。三张target卡的独立对照一致，AR期间四rank的21项计数增量均为0；四个进程正常退出、四卡空闲，前后冻结环境与源码检查通过，独立HOST计时和完整AR审计已完成且无审计错误；整轮仍因输出差异保留失败。159个同实例小案例通过不足以推出全量正确。下一步ROOT238仅做真实第1个多请求prefill组的三份独立前缀与逐查询数学诊断，不计Goodput、不继承正确性资格。最终同卡配对仍未执行。

07:29 的 ROOT239 已在原参考分组中复现请求19的358/773分叉：三张target卡第一层query及全部可见K/V均逐位相等，但attention输出分别出现约0.000122、0.000122、0.0000305的差异，后续层继续累积。这只定位该案例，尚不能推广到158条全部差异。该诊断四进程正常退出且环境/源码核验通过，但复制的汇总检查错误地要求不同TP分片的全部误差统计完全相同，执行状态保留失败，逐卡原始数值报告另存。ROOT240正在测试固定80个单查询attention段、4096执行长度和原精确可见掩码；未分配尾页映射到专用零页，保持实际候选预算与权重/采样；HOST1600查询行检查通过，随后该小范围设备对照完成：三张target卡的55个节点选词、hidden和全部KV均逐位相等，第19个请求在原参考分组也一致；原比较器未放宽。预先构造元数据的模型+greedy图交替局部计时中，固定q1约54.9ms、原q5约39.7ms，两边都重复固定上下文，不能外推完整Goodput；该方案数值更稳定但此项局部更慢，未采用。ROOT241正在测试保留原16组×5查询、固定4096执行长度的线性小原型，HOST400查询行检查通过，仍无完整Goodput或全负载正确性资格。

08:04进度：ROOT241固定q5/4096和ROOT242固定q5/2560均在真实前缀小对照中，55节点选词、hidden与全部KV逐位一致。局部交替模型+greedy图中位数分别约44.0ms对40.2ms、41.5ms对39.8ms；两边预建元数据且重复固定上下文，不能换算完整Goodput。ROOT243的11516个支持预算直方图及131个实际树计划的CPU检查保持真实输入、位置、KV写槽和祖先可见顺序；8424个超物理容量的预算直方图和48个原节点形状明确不支持，不计通过。ROOT244正在设备上检查111个支持形状，独立前缀包含1505和2555长度，线性/分叉各有独立图键；完整463请求、正式接入和最终同卡配对仍未完成。ROOT237四份退役大JSON已无损压缩约52MiB，原trace/源码未改，恢复索引保留。

以下为此前诊断记录：


截至 2026-10-18 06:27 UTC，本轮目标**尚未达到**。ROOT224 完整计时 Goodput/Raw 均为 **638.13 token/s**，463/463 达标，墙钟 185.74 秒；独立同后端 AR 为 **323/463 条完全一致、140 条存在 token 差异**。进程池退出快照记录 all_exited=false，随后原有清理确认四卡空闲，执行状态保留为失败；已完成计算的源码、计时、整数协议与参考历史经独立审计核对，未采用。强 MC2 TP4 历史参考 429.84，对应 1.5 倍线 **644.77**；当前比值约 **1.485×**。固定 γ4 历史参考 501.64，最终同卡配对与稳定性比较仍待完成。

ROOT224 显式改变 target 的 BF16 最高分平分处理：每卡最高分候选不超过4个时，用同一 BF16 hidden/权重的 FP32 标量点积进一步比较；超过上限保留原选择。树形与独立 AR 使用同一规则，draft、BF16 权重和模型主体未改，无量化。此前 ROOT223 原型在3个选择点取得一致结果，独立选词图额外耗时为40行约0.416毫秒、80行约0.511–0.521毫秒；完整实验仍有140条差异，不能把原型结论推广为全量修复。此前 ROOT218 为630.89、462/463达标、独立AR有177条差异；两个版本的生成历史不同，不能将差异条数变化等同于逐条修复。详见 [ROOT224 结果及失败退出记录](../budget-calibration-20261015/budget-diagnostics/ROOT_V224_TERMINAL_RESULT_AND_HANDOFF.json)。

ROOT225 因诊断只在 eager 端取整 KV 长度而失败，记录保留；实际 FULL 路径使用精确长度，不能借 KV 桶跳过相邻上下文参数更新。ROOT226 的已完成观测确认：相同精确元数据下 eager/graph hidden 和查询 KV 逐位一致；请求129的两条受控历史亦一致。但请求0第148位置在不同 FULL 形状下仍为678与279，均非最高分平分，单改平分规则不足以修复。该轮在新增 generic causal 局部计时图的原生验证处停止，没有可用时延；原始门槛未放宽。ROOT228 的正向部分结果为：固定80行和单查询段 FULL 后，两个新鲜前缀上下文的各变体 hidden/查询KV逐位一致，包括此前148位置统一选279；请求129的两条受控历史亦一致。新增q5 FULL对照计时图未通过原生验证，因此该轮仍记失败，没有可用时延，完整第四个上下文的两路比较未保留。ROOT229和ROOT230随后完整完成，独立HOST审计均通过：固定80行时，四个受控上下文的hidden与查询KV一致，两条受控历史的完整共同前缀KV亦一致。小对照显示，在两个新鲜前缀中，40/80行切换仍会改变中间结果；固定80行后q5 FULL和q1 FULL相同。已有图的局部调用中位数为q1约85.82ms、q5 FULL约36.57ms；时延含图重放/参数更新，预先构造元数据，排除LM-head、调度及prefill，不能换算完整Goodput。原采样规则未改，FP32平分细化仅为未安装的观察器。随后ROOT232扩展到真实分叉的独立节点检查：前2个线性案例通过，首个N1/B5/anchor0案例的选词一致，但hidden未通过原Native误差门槛，保留失败，KV比较尚未执行，根因未完全证明。ROOT233在保存的元数据尚未补齐原生mask宽度时停止，未完成任何神经案例，分叉修复尚未验证。CPU集成已修正为保存真实Native forward使用的最终元数据对象；ROOT234已完整通过155个独立节点案例，原生门槛不变、全部案例hidden及128组KV逐位一致，48个独立前缀受保护，退出与前后环境检查通过；独立HOST审计完成。ROOT235已通过原启动及组件检查，同一实例159个案例（包括遗漏形状）全部hidden及128组KV逐位一致，48个独立前缀受保护；但私有函数包装未复制keyword-only默认参数，进入预热时缺少max_attempts，整轮保留为失败，未计时、无Goodput。ROOT236已恢复默认参数并再次通过全部原检查和159个数值案例，但实际预热首轮暴露draft rank的target调用原本返回None，私有观测器误读State.last导致失败；未计时。失败及清理保留。ROOT237已启动：保留默认参数修复，draft恢复观测器所捕获的原forward回调，target观测器和模型计算代码不变。HOST已复现旧错误，并验证draft None协议与target原结果/记录；原门槛仍全部重跑；有效每轮N≤16、验证候选C≤64，保持原全局draft预算192、模型/采样/到达/SLO与计时口径；仍需通过原启动/组件、两轮零新增图检查和完整独立AR，测试将分叉可见KV规范为连续顺序、按层复制到专用scratch尾页；主链保留原表，分叉使用独立FIA查询段。线性图掩码容量9、带分叉复制的图容量10，真实查询仍为80行，原采样规则不变，无量化；原布局另作负面对照，绝不计为通过案例。ROOT234的203个CPU可见KV顺序/页边界/原Native ABI检查通过；ROOT235补齐形状后共207个通过。79行情况仅补typed root占位，真实候选预算不减少。完整463请求和独立AR尚未完成，目标仍未达到。ROOT227冻结未运行，因其计时路径会重复ROOT226失败而被替代；ROOT219启动合批和较大活跃批次方案仍未运行。

ROOT218 实际没有成功复用提前生成的后续草稿：现有机制只允许B3父路径，主要B4工作受到排除，不能把当前性能归功于已实现的rolling-eager收益。ROOT213旧记录为631.18，亦未取得独立全量AR一致性资格；其code worker TTFT中位数6.99秒，loose中位数93.59秒、P90 115.40秒，这些排队代价不能省略。下面主表为2026-10-15历史汇报数据，以官方三轮中位数325.26计算的1.551×不等于达到本轮较强基线与最终同卡配对目标。
<!-- /ROOT_CURRENT_DYNAMIC_TREE_STATUS -->

官方 vllm-ascend 主表采用三轮 Goodput 中位数 **325.26**，三轮范围 **281.38–409.20 tokens/s**。该中位数恰为第 2 轮实测值，Raw、SLO 和 TPOT 也全部使用第 2 轮。详见 [连续三轮确认报告](official-confirm3/REPORT_zh.md)。

北京时间 2026 年 10 月 15 日，主表使用同一组物理卡 1、2、3、5、同一份 463 请求。SpecSLO 使用最新串行复测，vllm-ascend 使用新三轮 Goodput 的中位数对应轮次，vllm-hust-ascend 使用既有 MC2 ON 单轮。每行 Raw、SLO 达标率与 TPOT 均来自该行同一份原始结果。

## 数据格式

输入为 UTF8 JSONL，每行一个请求，字段为 request_id、arrival_offset_sec、category、slo_class、slo_tpot_ms、per_request_gamma、max_tokens、prompt、source_dataset、source_index。

coding/tight/HumanEval 共 271 请求；chat/normal/GSM8K 共 96；summarization/loose/CNNDM 共 96。chat 是负载分类名，实际来源是数学问答。request_id 全部唯一，源样本有 307 个不同值。

所有方法重放同一份 Poisson 4 requests/s、seed 0、120 秒窗口的到达轨迹，在线 prefill、coalesce1、额外等待 0 ms。每请求生成 256 token、忽略 EOS，共 118528 token。tight 的 SLO 从原始 31.56 ms 受控修改为 40 ms；normal 为 50、loose 为 150 ms。

字段、真实样例、输入长度、来源及哈希见 [数据格式说明](DATASET_FORMAT_zh.md)，机器可读结构见 [JSON Schema](request-row.schema.json)。

## 本轮三种方法的运行条件

| 方法 | 模型与分卡 | 代码版本 | 环境 | MC2 |
| --- | --- | --- | --- | --- |
| 串行 SpecSLO | 0.6B TP1 draft 加 32B TP3 target | 冻结 HUST 源码与原串行 gamma4 auto 生产配置 | Python3.11、CANN9.0、torch-npu2.10.0 | v149 TP3 路由开启 |
| 官方 vLLM Ascend | Qwen3-32B TP4，自回归 | v0.23.0.post1 `1cdb8c4`，vLLM v0.23.0 `0fc695f` | Python3.12、CANN9.1、torch-npu2.10.0.post4 | 普通 Matmul AllReduce 开启 |
| HUST vLLM Ascend | Qwen3-32B TP4，自回归 | 插件冻结 lock `ac3efecb`，核心 `933842ab` 加归档补丁 | Python3.11、CANN9.0、torch-npu2.10.0 | 普通 Matmul AllReduce 开启 |

全部 BF16 greedy、无量化，模型从公共目录读取。两种 TP4 基线均使用默认 HCCL、max_num_seqs=64、max_num_batched_tokens=640，禁用计时中的逐次图事件文件日志，实际模型四 rank 各选择 128 个 MC2 投影模块。普通 MC2 不等于另一个 enable_fused_mc2 选项；后者为 0。SpecSLO 保留原有 AIV 和专属优化，没有强行统一各方法内部策略。

## 按请求平均 TPOT 的中位数和 P90

每个请求先计算自身的平均 TPOT，再在类别内计算分位数；包含所有完成请求，包括超过 SLO 的请求。P90 使用排序后索引 (n−1)×0.9 的线性插值。这不是全部相邻 token 间隔的 P90。两种 runner 均记录首 token 之后的 decode 时间：基线用首末 token 时间戳，SpecSLO 用 native 的首 token 发布后累计时钟。463 条记录全部独立重算通过。

以下主表沿用 paper TPOT：decode 时间毫秒 / 256。单元格为中位数 / P90，单位 ms/token。三种方法均对应 MC2 ON 的本轮结果；vllm-ascend 的普通 Matmul AllReduce 开启。第一列为方法，其余列为请求类型。

| 方法 | 代码 tight（271 请求） | 数学问答 normal（96 请求） | 摘要 loose（96 请求） | 全部（463 请求） |
| --- | ---: | ---: | ---: | ---: |
| SpecSLO | 45.33 / 53.54 | 37.27 / 45.77 | 58.55 / 70.78 | 45.52 / 59.08 |
| vllm-ascend | 53.05 / 58.11 | 52.60 / 57.74 | 53.49 / 57.61 | 53.05 / 57.93 |
| vllm-hust-ascend | 51.01 / 55.29 | 51.14 / 55.43 | 51.09 / 54.85 | 51.09 / 55.29 |

常见的 observed TPOT 使用同一时间 / 255：

| 方法 | 代码 tight（271 请求） | 数学问答 normal（96 请求） | 摘要 loose（96 请求） | 全部（463 请求） |
| --- | ---: | ---: | ---: | ---: |
| SpecSLO | 45.51 / 53.75 | 37.41 / 45.95 | 58.78 / 71.06 | 45.70 / 59.31 |
| vllm-ascend | 53.26 / 58.33 | 52.80 / 57.96 | 53.70 / 57.83 | 53.26 / 58.16 |
| vllm-hust-ascend | 51.21 / 55.50 | 51.34 / 55.65 | 51.30 / 55.07 | 51.29 / 55.50 |

## SLO 达标数和整体吞吐量

Raw 为全部生成 token / 整轮耗时；Goodput 仅计入平均 paper TPOT 达标请求的 token / 同一整轮耗时。总体及各类单元格为达标请求数 / 总请求数（达成率），tight、normal、loose 阈值分别为 40、50、150 ms/token。SpecSLO 行对应本轮串行版本。

提升比 = SpecSLO Goodput / 本行方法 Goodput。它是表中选用结果的描述性倍率；串行本次为单轮，官方为三轮中位数，HUST 为既有单轮，尚非多轮交替配对的稳定加速比。

| 方法 | MC2 | Raw tokens/s | Goodput tokens/s | 总体 SLO 达成 | tight 达标 | normal 达标 | loose 达标 | SpecSLO 相对本方法提升比 |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| SpecSLO | ON | 923.22 | 504.48 | 253/463（54.64%） | 64/271（23.62%） | 93/96（96.88%） | 96/96（100.00%） | 1.000×（+0.0%） |
| vllm-ascend | ON | 923.90 | 325.26 | 163/463（35.21%） | 34/271（12.55%） | 33/96（34.38%） | 96/96（100.00%） | 1.551×（+55.1%） |
| vllm-hust-ascend | ON | 922.56 | 320.80 | 161/463（34.77%） | 25/271（9.23%） | 40/96（41.67%） | 96/96（100.00%） | 1.573×（+57.3%） |

vllm-ascend 的 MC2 ON 证据：启动时 VLLM_ASCEND_ENABLE_MATMUL_ALLREDUCE=1；实际模型四 rank 各选择 128 个 MC2 投影模块。见 [启动环境](../upstream-baseline-20261014/runs/mc2-on-confirm3-20261015-r2/launch-environment.json) 和 [模型模块记录](../upstream-baseline-20261014/runs/mc2-on-confirm3-20261015-r2/model-evidence.json)。VLLM_ASCEND_ENABLE_FUSED_MC2=0 是另一项融合开关，不代表普通 MC2 关闭。

三类来自不同数据源，输入长度不同，类别间延迟差异不能全部归因于调度机制。整体结果来自多个优化的组合，尚无逐项收益的因果隔离。官方三轮 Goodput 范围为 281.38–409.20，主表采用中位数 325.26；延迟变化跨过 SLO 阈值会放大达标数和 Goodput 的差异，底层运行时诱因尚未证明。

主表逐请求完整数值见 [per-request-tpot.csv](current-report/per-request-tpot.csv)，分组统计见 [tpot-by-class.csv](current-report/tpot-by-class.csv)，吞吐量、SLO 达成率及倍率见 [throughput-slo.csv](final/throughput-slo.csv)，当前选用轮次、完整精度与结果哈希见 [summary.json](current-report/summary.json)。原三方法首轮汇总保留在 [历史 tpot-summary.json](final/tpot-summary.json)。

## 当前 SpecSLO 优化

### 1. 论文部分

双 batch pipeline、rolling eager、SLO 驱动的调度。

本轮实验设置：固定 gamma=4，采用串行线性完整链，关闭 linear bonus token。固定 gamma=4 属于实验设置。

### 2. 性能优化

**批处理与资源调度优化：** ready home 合并、跨 bucket eager、idle residual eager、尾部 padding repair。

**图执行与主机调度优化：** 封存图、固定四步 draft chain graph、提前提交当前 target、replay-first、两路 attention task 更新。

**算子与数据布局优化：** TP3 MC2 与 chained epilogue、NZ14、融合 verdict。

**通信、状态管理与缓冲优化：** 紧凑 transport、proposal/continuation 与 scheduler/accounting 缓冲复用；原子提交与失败回滚用于保障状态一致性。

树形 B3 是另一条尚未胜出的实验路径。本轮没有树形、动态 gamma 或可选 B；4 线程更新和 bucket320 等未采用。官方优势移植、新 CANN/TorchNPU 环境的优化调查继续暂缓。本轮只做用户要求的重测和汇报整理。

参数、作用与边界见 [当前优化清单](CURRENT_SPECSLO_OPTIMIZATIONS_zh.md)。原 518.09 轮的实际计数作为历史机制证据保留在 [serial-optimization-evidence.json](serial-optimization-evidence.json)，最新串行计数见本次原始结果和 mechanism-gate。

## 本轮结果文件

- SpecSLO：本次新增的串行 gamma4 单轮复测。`/root/SpecSLO/tpot-report-20261015/serial-repeat-once/measurement-output/runs/candidate-on-serial-repeat-20261015-r1/result.json`，SHA256 `80eb54f4951e7a59f29ad7cd0c304265859af71521cd96d17faefa701dd2920d`。
- vllm-ascend：新三轮 Goodput 中位数对应的第 2 轮；Raw、SLO 和 TPOT 均取同轮。`/root/SpecSLO/upstream-baseline-20261014/runs/mc2-on-confirm3-20261015-r2/result.json`，SHA256 `7cf4fa810cd70ac93f5f911bd00fe3a052eae1acd0b7a91563033f9473adc94b`。
- vllm-hust-ascend：既有 HUST MC2 ON 单轮。`/root/SpecSLO/upstream-transfer-20261015/runs/tpot-by-class-20261015-r1/result.json`，SHA256 `349b66804ae72754a7bd791b0e449762dffdbf5fa1a88e4bc8a7606a0557cfb1`。

## 官方 MC2 ON 追加复测

本节保留此前追加复测记录。所有轮次 MC2 ON；当前主表的选用规则见结果文件说明。

| vllm-ascend 轮次 | Raw tokens/s | Goodput tokens/s | 总体达标 | normal 达标 |
| --- | ---: | ---: | ---: | ---: |
| 最初测量 | 925.66 | 429.84 | 215/463 | 92/96 |
| 汇报重测 r1 | 923.42 | 277.23 | 139/463 | 30/96 |
| 本次原配置复测 r2 | 924.35 | 393.30 | 197/463 | 82/96 |
| CPU 绑核修正对照 | 923.85 | 297.31 | 149/463 | 31/96 |

逐类 TPOT、原配置复测及独立 CPU 绑核修正对照见 [MC2 ON 复测与差异分析](OFFICIAL_MC2_VARIATION_zh.md)。

## 官方原配置 MC2 ON 连续三轮确认

| vllm-ascend 轮次 | Raw tokens/s | Goodput tokens/s | 总体达标 | normal 达标 |
| --- | ---: | ---: | ---: | ---: |
| 新测第 1 轮 | 924.19 | 409.20 | 205/463 | 82/96 |
| 新测第 2 轮 | 923.90 | 325.26 | 163/463 | 33/96 |
| 新测第 3 轮 | 923.96 | 281.38 | 141/463 | 32/96 |

三轮 Goodput 中位数 325.26，均值 338.61 tokens/s。全部逐请求统计与原始结果见 [连续三轮确认报告](official-confirm3/REPORT_zh.md)。

## 串行 gamma4 本次复测

本次 Goodput 504.48，上次 518.09 tokens/s，变化 -2.63%。本次仅新增一轮正式测量，详见 [串行单轮复测报告](serial-repeat-once/REPORT_zh.md)。

## 双 batch 设备时间线诊断

本次单独重放原泊松轨迹的前 64 个请求，每请求仍生成 256 token。保留固定 gamma=4、TP1 draft 加 TP3 target、MC2 ON、双 batch 与 rolling eager 的生产设置，使用同一组物理卡 1、2、3、5。原推理源码未修改；完成 3 轮图资格预热后，采集 profiler 第 257、258、259 步。诊断开启 profiler 和主机日志，其耗时不替换本文正式 Goodput / TPOT 数据。

下表使用 draft 卡 1 与 target TP3 的卡 2。计算窗口是该步所选图内矩阵乘 / attention kernel 从首次开始到最后结束的跨度，包含中间间隙；实际同时计算则先分别合并 kernel 活跃区间，再求时间交集。通信、事件等待、拷贝、独立 prefill 等图外操作均不计入交集。单位为 ms。

| 采样步 | draft 计算窗口 | target 计算窗口 | 两侧共同时间跨度 | 实际 kernel 同时计算 |
| --- | ---: | ---: | ---: | ---: |
| 257 | 25.94 | 59.50 | 59.50 | 4.86 |
| 258 | 26.13 | 113.14 | 113.14 | 3.90 |
| 259 | 25.68 | 81.52 | 81.52 | 4.83 |

3 步内，所统计草稿 kernel 的累计活跃时间为 20.70 ms；与 target 三张卡分别同时计算 13.60、13.55、13.62 ms，占该部分草稿 kernel 活跃时间的 65.68%、65.43%、65.80%。因此本次采样证明存在设备端实际计算重叠。该比例只针对所选 kernel，不能解释为整轮时延降低比例。

第 257 步，target 验证 20 个请求，draft 同时为另一 home 的 14 个普通请求生成草稿，并处理 13 行 eager 延续；普通草稿请求与当前 target 请求集合不相交。后两步的普通 draft home 继续交替。当前配置也包含 ready home 合并和 rolling eager，本次未做关闭流水线的对照，不能据此给出双 batch 的独立加速比。

时间戳采用 profiler 换算后的同一主机时钟，仅使用一个公共显示原点，没有分别把各 rank 归零。四 rank 的 CANN 时钟换算偏移差为 0.64 微秒；区间交集另用独立端点扫描算法重算通过。原源码锁检查前后均通过，模型进程已退出。

![双 batch 设备计算重叠时间线](dual-batch-overlap/device-overlap-timeline.png)

可放大矢量图见 [SVG 时间线](dual-batch-overlap/device-overlap-timeline.svg)；完整数值见 [逐步时延与交集](dual-batch-overlap/overlap-by-cycle.csv)、[kernel 时间戳](dual-batch-overlap/device-kernels.csv)、[汇总与具体重叠实例](dual-batch-overlap/overlap-summary.json)、[请求分组证据](dual-batch-overlap/sampled-scheduler-plans.json) 和 [时钟校准证据](dual-batch-overlap/profiler-clock-evidence.json)。复现条件与所有诊断改动见 [run-plan.json](dual-batch-overlap/run-plan.json)。

## 串行收益与预算调度追加实验

本节使用当前物理卡 2、5、6、7，独立于前文主表的卡组 1、2、3、5。保留原 463 请求、泊松到达和 256 token 输出，五种方法均为 TP1 draft 加 TP3 target、MC2 ON、BF16、完整 gamma4 窗口。每种方法只计时一轮，预热排除。此处仅比较同一卡组的 SpecSLO 原调度与预算调度，不与前文不同卡组的官方基线计算新倍率。

已对每类 24 个不同来源请求、共 216 个上下文标定 n=1…8。训练样本的 g(4) 为代码 3.06、数学问答 3.31、摘要 2.60 个有效 token；类别曲线在留出请求上的预测误差未优于统一曲线，因此调度还使用在线实际交付量修正。n 是 target 查询位置数，n=0 表示本轮跳过；g 包含接受草稿和纠正 token。

成本扫描在当前 gamma4 范围选用 B_T≤128、即 target 最多 32 请求；更宽窗口成功测到 B_T=256，这不是机器绝对最大预算的证明。四版原型按完整窗口授权 0 或 4 个投机位置，原有 AR fallback 保留；尚未实现混合的可变 n 协议。

| 方法 | Raw tokens/s | Goodput tokens/s | 代码达标 /271 | 数学达标 /96 | 摘要达标 /96 | 相对原调度 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 固定 gamma4 原调度 | 921.66 | 501.64 | 61 | 95 | 96 | 1.0000× |
| 预算原型 1 | 919.92 | 480.82 | 54 | 92 | 96 | 0.9585× |
| 预算原型 2 | 922.32 | 452.19 | 49 | 82 | 96 | 0.9014× |
| 动态全局预算原型 3 | 888.82 | 466.49 | 74 | 96 | 73 | 0.9299× |
| 带 SLO 余量保护的预算原型 4 | 914.29 | 495.65 | 62 | 93 | 96 | 0.9881× |

第一版采用类别优先级和实测接受率初值；第二版保留原接受率初值，按排除准入等待后的 TPOT 缺口判断当前可行性，再按类别排序，并保留久等请求的服务机会。第三版保留原紧迫度排序，在 tight 有风险且全部能放入 24 行时，将 B_T 从 128 收紧为 96，并在收紧时优先 tight；不以本轮预测不可行为由排除请求。完整结果包括未胜出的版本，当前主表与生产配置不受本节原型影响。

第四版增加摘要余量保护：摘要的预测平均 TPOT 达到其 SLO 的 80% 时，恢复 B_T=128 和原紧迫度排序，避免高优先级收益伴随摘要大量失约。

标定曲线、留出误差、成本条件、预测不可行与最终 SLO 的对照，以及源代码和复现记录见 [串行预算实验报告](../budget-calibration-20261015/serial-v1/REPORT_zh.md) 和 [全部对照数值](../budget-calibration-20261015/serial-v1/budget-trials-comparison.json)。单轮结果尚不能证明稳定收益；所有路径相对 AR 的逐 token 等价资格仍未完成。

### 固定全局预算与变长 n 的后续修正

上表四版原型只控制完整 gamma4 窗口的准入，不能作为逐请求混合 n 预算方案已经测试的证据。后续实验固定逻辑 target 验证预算 B_T=128，并实际执行每请求 n=1…8；同轮 sum(n_i) 不超过 128，预算不足时优先满足高优先级请求。128 是起始候选值，尚不是机器绝对最大预算或已证实的最优值。

实验同时把一轮新 normal/eager 草稿的逻辑工作限制为 128；draft 与 target 的实际 padding 工作分别统计。变长 draft 使用单步 PA 图，请求生成够自己的 n 后退出后续步骤；target 使用隔离 scratch KV 的有界变长 FIA 图。类别 g(n) 是期望交付量，预测当前不可行不计为最终 SLO 失败。

当前状态：已按用户要求暂停。固定全局 B_T=128、逐请求 n=1…8 的实验代码已接入，12项 CPU/通信检查通过；NPU 图初始化后，在预热的变长 Gloo 通信分支遇到未初始化变量 gloo_proposal_timing，尚未完成正式性能回放，不能报告新的 Goodput。错误、复现条件及接续步骤已保存，NPU 进程已全部退出。

具体协议、实验状态和复现条件见 [实际变长预算实验目录](../budget-calibration-20261015/serial-v1/variable-budget-v1/RUN_PLAN.json)。生产配置与前文基线主表保持原记录。

## 动态树形预算后续进展（独立实验）

已继续进入真实分叉树、逐请求候选预算 1–8、全局 target 查询上限 128（含 root）的实验。最新完整原轨迹结果为 Raw 345.81、Goodput 72.45 tokens/s，相对本卡组固定串行 gamma4 的 501.64 为 0.1444×，尚未胜出。此前变长串行原型的暂停记录属于历史检查点，当前工作重点已转入动态树形预算。

三类 g 曲线已完成 36 上下文 pilot；缓存分配、主机页表打包和有界树形图已实测。后续成本扫描确认树形 target 的动态 attention 更新仍先于图提交，尚未走串行路径的 replay-first。下一步先验证该执行路径，再考虑 FULL 并行更新、prefill 与 draft 修复成本。详细结果、边界和复现入口见 [动态树形实验报告](../budget-calibration-20261015/tree-v1/REPORT_zh.md)。本节不替换前文正式基线表。

暂停补记：用户再次要求暂停。树形 target replay-first 配对成本小测已完成，B=6 的动态 target 耗时从 86.88 ms 降到 51.84 ms，三张 target 卡真实预测 token 一致；这不是端到端 Goodput 倍率。短服务回放被主动中断，尚无新的完整 Goodput，正式动态树形仍为 72.45。接续入口与未完成事项见上述动态树形实验报告。

## 动态树形预算本次继续优化

已恢复优化，当前在同卡2/5/6/7上运行 `tree-replay-pa-exactprefill-censor-full463-r6`：原463请求泊松轨迹、256输出、BF16、MC2 ON，投机验证查询预算上限128（含root）。组合使用 target replay-first、draft PA 单叶修复、48档常见原始形状的 target prefill 图，以及末尾截断样本不更新内在 g 的修正。最新完整动态树形结果在本轮结束前仍为此前的72.45，不能用下面短诊断替换。

| 32请求×128输出的饱和诊断 | 端到端秒 | Raw tokens/s | Goodput tokens/s | 达标请求 |
| --- | ---: | ---: | ---: | ---: |
| CPU transport，原target执行 | 10.6568 | 384.36 | 180.17 | 15/32 |
| 增加树形target replay-first | 8.1497 | 502.59 | 471.18 | 30/32 |
| 再增加PA修复与延迟末叶KV | 7.7588 | 527.92 | 511.42 | 31/32 |

这些是饱和短诊断，不与完整泊松轨迹串行501.64直接判断胜负。原图数值、finite、KV与封存门槛保留；独立KV重建参考9种提交几何组合通过，但两轮短诊断有4请求的输出序列不同，完整AR等价资格仍未完成。

带padding的prefill图在严格数值对照中失败，已拒绝。改为原始形状后，4档查询数的target prefill小测通过，约215–224 ms降至41–79 ms；只对常见singleton形状启用，其余保留原路径。完整原始replay直方图保留，树验证MC2全链覆盖与prefill分别审计。更深树的收益标定入口和按预算限制探索深度原型仅通过CPU检查，尚未启用。详细记录见[动态树形实验报告](../budget-calibration-20261015/tree-v1/REPORT_zh.md)。


续跑完整负载 r6 已完成：463 个请求、Poisson 4 rps、seed 0、每请求 256 token，MC2 开启。Goodput **132.73 token/s**，Raw **495.58 token/s**，论文口径 SLO 达标 **124/463**（tight **2/271**，normal **26/96**，loose **96/96**）。经过 10 轮完整预热达到图固定点后测量，测量期零捕获、零运行时验证、零图回退；源文件、冻结基线前后校验以及退出清理通过。独立审计保留完整原始图直方图，仅将明确标记的精确形状 prefill 从树验证 MC2 覆盖统计中单列。

相对上一轮完整树形 72.45 有提升，但仅为同卡固定串行 γ=4（501.64）的 **0.265×**，未达到优化目标；短饱和负载的 511.42 不能替代这个完整结果。全负载逐 token AR 一致性仍未完成资格验证。结果见 `budget-calibration-20261015/tree-v1/full-r6-summary.json`。下一阶段已启动隔离目录 `tree-depth8-v2` 的深度 8 收益标定，验证按实际 B 截短 draft 链及合并元数据准备；尚无这一版的在线性能结果。


深度 8 的受控短诊断已完成：在相同 32 请求×128 输出、饱和到达下，Goodput 366.62、Raw 419.00、达标 28/32；深度 4 同范围为 511.42、527.92、31/32。新图资格门槛通过，但更深 draft 的计算开销超过收益，暂不采用深度 8。完整泊松树形结果仍是 r6 的 132.73 / 495.58；未超过固定串行 γ=4。详细证据见 `budget-calibration-20261015/tree-depth8-v2/short-r4-summary.json`。


完整候选 `depth4-arena-prefill201-full463-r1` 已完成，独立审计通过。原463请求、Poisson4rps、seed0、每请求256输出、卡组2/5/6/7、MC2开启。预热4轮后图封存，测量期零捕获、零运行时验证、零回退；实验源、冻结基线和退出设备空闲检查通过。Goodput **216.07**，Raw **562.04**，E2E **210.89s**，达标 **178/463**（tight **2/271**、normal **80/96**、loose **96/96**）。逐请求paper TPOT均值分别 **53.23 / 46.46 / 60.68ms**。

相对上一轮完整树形132.73，Goodput提高约62.8%；Raw从495.58提高约13.4%。主要新增达标来自normal，tight仍只有2个达标。在线prefill停顿约80.01s→52.60s；这说明扩大覆盖确有收益，但仍不够，尚未超过同卡固定γ=4的501.64。全负载AR token等价仍未完成资格验证。完整证据见 `budget-calibration-20261015/tree-prefill-arena-v4/full-r1-summary.json`。

prefill首token和故障投票改用CPU/Gloo的独立成对NPU对照已完成，完整token与故障控制检查通过，但耗时差异不足约1%，暂不合入。随后启动 `tree-packed-budget-v7/b3-qualified-packing-full463-r1`：每请求实际B为1–3，树仍2×4，全局root-inclusive查询上限128；使用现有全部合格MC2行档位，在真实规模已合格时允许零dummy行。16请求各root+3候选的物理图由96行减为64行。CPU检查覆盖2048个可达请求数/查询数及21组真实token/mask/KV几何，NPU完整结果待出。


## 完整 463 请求正式结果：较小 B 与细化验证填充

本轮采用单请求 B=1..3、宽 2 深 4，实际 target 查询（含 root）全局上限 128；MC2、原始泊松到达序列、输出长度和四张卡不变。改动同时涉及 B 上限与填充策略，属于组合实验，不能据此分离两项收益。

| 配置 | Raw (token/s) | Goodput (token/s) | tight 达标 | normal 达标 | loose 达标 |
|---|---:|---:|---:|---:|---:|
| 上一轮：B≤8，201 种 prefill 图与共享缓冲 | 562.04 | 216.07 | 2/271 | 80/96 | 96/96 |
| 本轮：B≤3，细化 MC2 验证填充 | 558.33 | 205.00 | 2/271 | 72/96 | 96/96 |
| 同卡固定串行 γ=4 对照 | 921.66 | 501.64 | 61/271 | 95/96 | 96/96 |

本轮 Goodput 较上一轮低 5.12%，未超过固定串行 γ=4。7 轮完整负载预热后封存，第 8 轮正式测量，正式阶段零捕获、零运行时验证、零回退；独立审计通过，源文件和冻结基线校验通过，进程已清理。全负载 AR/token 一致性仍未取得资格，不能据此宣称生产正确性。

下一步独立测试 canonical eager：把未来草稿的 frontier 作为 PA 单查询，再与普通请求共用草稿链图；延迟 draft KV 搬移并在需要时从已提交前缀修复。保留原有 primary-path、frontier、proposal identity 和 epoch 判定。源代码原本已按 primary-path 长度判定 full acceptance，不存在必须接受所有分支节点才可复用的限制；本轮没有放宽该判定。

本轮原始结果：`/root/SpecSLO/budget-calibration-20261015/tree-packed-budget-v7/measurement-output/runs/candidate-on-b3-qualified-packing-full463-r1/result.json`；汇总：`/root/SpecSLO/budget-calibration-20261015/tree-packed-budget-v7/full-r1-summary.json`。


## canonical eager 与按图容量估算窗口：验证进度

canonical eager 的独立 PA 参考检查已通过 19 组，每组 4 个请求。18 组覆盖前缀 127/128/255、纯链或带分支父草稿，以及父草稿完整接受后继续、父草稿拒绝、frontier 不匹配；另 1 组验证 3 个 eager 请求与 1 个不同前缀的普通请求合批。frontier、提前草稿 top-2、裁决后修复所得 top-2 均与重新建立原始 prompt 的逐 token PA 参考一致，数值故障标志为 false。冻结源和实验源校验通过，四卡清理为空闲。该检查不是全负载 AR 一致性资格。

新增的 packed window 按实际 PA 图容量估计草稿耗时，将 eager 的额外开销与普通图耗时分别记录。首次最多一行主动探测会产生实际耗时，不能称为隐藏开销；后续只按观测的图耗时和 frontier 额外开销估计可重叠容量。CPU 边界、超时退让、观测重置和异常输入检查通过。正在进行 32×128 饱和诊断；其结果不能与 463 请求泊松负载混比。


32×128 饱和短测正式结果：Goodput/Raw 544.9542 token/s，32/32 达标（tight 16/16、normal 8/8、loose 8/8），E2E 7.5162 秒，7 轮预热后封存，独立审计通过。实际 B 为 1/2/3/4；测量发生 1 次 eager frontier，耗时模型观测额外开销约 8.396 ms，该草稿因原有依赖检查失效，成功复用 0 次。因此不能把此次性能差异归因于 eager 复用，也不能把短测结果与完整泊松 463 请求的串行 γ=4 501.6356 直接比较。正在追加全量测试。

短测输出补充核查：与旧 r5 的相同 32 个 prompt 相比，17/32 请求的 token 序列有差异，尚未确认统一根因。因此 544.95 是当前配置实测值，不能视为已隔离的优化收益，也不能宣称全负载 AR 一致性通过；详见 `budget-calibration-20261015/tree-packed-window-v9/short-output-comparison.json`。


完整 463 请求 canonical packed window r1 在第 4 轮预热被 KV 检查阻止：`SpecRhythm compacted tree is missing its preflighted KV mapping`。没有封存、没有正式结果，不得将前三轮预热耗时计作 Goodput。根因为新 canonical deferred eager 成功复用后，下一次验证的 `rows_to_compact` 仍只按旧 `ticket.eager` 区分，错误要求未物化草稿的 KV 映射。已有 preflight 正确表示无该映射；修复必须让该消费分支识别显式 deferred 标记，不能伪造映射，也不能跳过 target/legacy eager 的真实检查。源与冻结校验通过，失败组进程已清理。

KV 消费分支修复验证：v13 通过新增 25 组 NPU 独立逐 token PA 参考，v14 主动调度的完整协议回归通过；封存测量内真实消费 24 个复用草稿，覆盖 1 个原先错误要求映射的非平凡分支。未伪造映射或强制接受，target/legacy 映射检查保留。该主动调度回归不作为性能收益。v13 正常调度的全量测试重新开始。


## 修复后的正常调度全量正式结果

463 请求、原始泊松轨迹、每请求 256 token、同卡 MC2 on，实际单请求 B=1..4，全局验证查询上限128（含root）。本轮未主动安排 eager。12 轮完整预热后封存，第13轮正式测量，独立审计、源码/冻结/清理检查通过。

| 方法 | Raw (token/s) | Goodput (token/s) | tight 达标 | normal 达标 | loose 达标 |
|---|---:|---:|---:|---:|---:|
| 上一轮完整树形最好结果 | 562.04 | 216.07 | 2/271 | 80/96 | 96/96 |
| 本轮预算≤4、细化填充、packed window、canonical eager修复 | 584.01 | 238.40 | 4/271 | 89/96 | 96/96 |
| 同卡固定串行 γ=4 | 921.66 | 501.64 | 61/271 | 95/96 | 96/96 |

本轮 Goodput 较上一轮树形最好结果高10.33%，但仅为串行对照的0.475倍，未达到目标。tight 平均 paper TPOT 为51.24 ms，高于40 ms要求；normal 43.65 ms、loose 58.54 ms。本轮测量只有1次 eager失效、0次成功复用，不能把提升归因于 eager复用；B上限、填充和窗口等组合变化尚未做独立收益拆分。全负载 AR/token 一致性仍未取得资格。

后续先验证保留四步主链、按真实联合概率选择并继续展开一条侧分支的候选，逐档预算1..8重新测三类g(n)。已通过128组CPU结构检查和96组独立前缀模型/真实KV写入记录检查；96组NPU侧分支与裁决后修复参考检查正在启动。它尚未用于任何完整负载性能结论。

原始结果：`/root/SpecSLO/budget-calibration-20261015/tree-canonical-commit-v13/measurement-output/runs/candidate-on-canonical-commit-full463-r1/result.json`；汇总：`/root/SpecSLO/budget-calibration-20261015/tree-canonical-commit-v13/full-r1-summary.json`。


## 预选 B 与保主链分支：本轮标定

独立NPU KV检查96组（每组4请求）通过，覆盖前缀127/128/255、四种分支位置、实际B=5..8、主链或侧链提交后重建。发现并修正改写祖先后仍保留旧后代KV记录的问题；多token FIA修复曾与逐token PA参考出现一个第二候选token差异，因此此实验候选改用原生PA逐token修复，96组均精确一致。没有据此解释全部历史输出差异，全负载AR资格仍未通过。

随后完成36上下文×8预算的实测收益标定，三张target卡的全部token/裁决记录一致，测时区间零捕获、零运行时验证、零回退。每类12个上下文来自4个请求的3个偏移，属于相关pilot样本，尚无held-out泛化结论。B≤4保留主链；B5保留四步主链并添加最大联合概率的一个次选token；B6..8继续展开该分支。

| 类别 | g(1) | g(2) | g(3) | g(4) | g(5) | g(6) | g(7) | g(8) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 代码 tight | 1.92 | 2.75 | 3.33 | 3.83 | 3.92 | 4.00 | 4.08 | 4.17 |
| 问答 normal | 1.92 | 2.58 | 3.25 | 3.83 | 4.08 | 4.33 | 4.58 | 4.83 |
| 摘要 loose | 1.67 | 2.17 | 2.50 | 2.67 | 3.00 | 3.25 | 3.42 | 3.58 |

6请求小批次中，draft当前proposal中位耗时B4/B5/B6/B7/B8为24.27/24.39/31.68/36.19/41.22ms，target为39.64/37.59/37.85/39.54/39.79ms。前者不含下一轮提交后KV修复；独立参考中B6..8改写分支后若回到主链，还需额外修复上下文。不能用单独g提高推算完整Goodput。先测试无需额外draft pass的B≤5，并保持全局候选96、含root真实query128；物理padding另统计。

预算分配器补充跨平段的完整增益跳跃：当g(n+1)暂不增加而更大的n仍有收益时，按优先级×净收益/新增预算分配，不因一格平段永久停住；10项CPU检查（含500组随机边界）通过。新候选尚无完整463请求结果，最新正式树形仍是238.40，串行对照501.64。

标定证据：`budget-calibration-20261015/tree-adaptive-side-prototype-v15/adaptive-KV-summary.json`、`adaptive-side-gain-v1-summary.json`、`adaptive-side-gain-v1-raw-gain-model.json`。

标定边界补记：B8在一个摘要上下文中的target主链预测与B4..7不同，导致选择的接受路径也变化；原生数值门仍通过，但不能把B8全部增益归因于分支加深。B4与B5未出现此处跨预算预测差异；B1..5的180条标定记录已通过原生depth4计划的positions、parents、mask、物理slot以及裁决结果一致性CPU检查。下一次完整测试使用B≤5，实际物理批次耗时仍重新测量。

在线候选接入前，241组NPU canonical eager独立参考检查全部通过，覆盖parent主链长度1..4、child预算1..4完整叶子接受、拒绝、frontier不匹配、次选分支及一次普通/提前草稿混批；75组CPU上下文KV检查使用原生tail5 catchup且包含混合预算。仅提前执行所需draft上下文参考通过，不能替代全负载AR资格。正在运行 `tree-backbone-budget-v17/backbone-B5-PA-context-full463-r1` 完整463请求原轨迹测试，新结果尚未产生。


## B≤5 保主链候选的完整正式结果

原463请求、Poisson4rps/seed0、每请求256token、同卡MC2 ON。6轮完整预热，第5/6轮连续零捕获及运行时验证后封存，第7轮正式测量；独立审计通过，测量期零捕获/验证/回退，源码与g模型未变化，冻结前后校验与退出清理通过。

| 方法 | Raw (token/s) | Goodput (token/s) | tight达标 | normal达标 | loose达标 |
|---|---:|---:|---:|---:|---:|
| 上一轮树形B≤4 | 584.01 | 238.40 | 4/271 | 89/96 | 96/96 |
| 保四步主链、最高联合概率单分支B≤5 | 585.81 | 239.13 | 3/271 | 90/96 | 96/96 |
| 同卡固定串行γ=4 | 921.66 | 501.64 | 61/271 | 95/96 | 96/96 |

本轮E2E202.33s，仍189/463达标。Goodput仅+0.31%，差异来自E2E缩短约0.62s；不能认定稳定改进，尚未达到超越串行目标。tight/normal/loose平均paper TPOT为51.01/43.52/58.63ms。实际B1..5均出现，但约98.995% proposal为B5，当前全局上限并未频繁约束预算；不能将此结果宣称动态B已经有效提升。全量上下文中B5在线观测交付均值为tight3.29/normal3.85/loose2.83，与小pilot有差距，三类曲线仍只是预测先验。全负载AR/token一致性资格仍未完成。

另完成仅top2选择/编码的NPU微基准：显式.float()与softmax(dtype=float32)的有序候选、encoded置信度、changed-input和图回放检查一致，但图内耗时无实益（16行约182μs，32行约349μs每次），暂不采用；它不含LM投影和完整模型，不能据此解释全部draft差距。下一步验证混合prefill与一次PA打包多后缀修复。

结果/独立审计/汇总：`budget-calibration-20261015/tree-backbone-budget-v17/measurement-output/runs/candidate-on-backbone-B5-PA-context-full463-r1/`、`tree-backbone-budget-v17/full-r1-summary.json`。


## 混合 prefill 与打包 PA 修复：尚未通过资格检查

混合 target 探针 `tree-mixed-prefill-v18/mixed-target-kv-v1` 在首组检查失败：同一次 target 中放入两组真实树验证、128 个 prompt token 和独立 scratch 行，共 165 行；图执行通过原生执行门，但全部 hidden 与逐请求 eager 参考的原生容差比较失败。因此尚未进入精确 token、后续 KV 或封存轨迹检查。当前没有数值误差分布，不能认定是 BF16、NZ14 或任何特定算子的原因；该路径没有用于新的正式 Goodput 结果。

打包后缀修复探针 `tree-packed-PA-repair-v21/packed-PA-repair-v1` 通过了前缀127、四请求等长后缀1/2/4的前三组检查（原生 changed-input、all-hidden、后续 top-2 一致），在异长 `[1,2,3,4]` 的第4组 all-hidden 比较失败。15组几何检查仅完成3组，后续96组分支提交检查尚未开始，整体资格未通过。前三组的五次配对计时中位数如下，仅描述已通过的子范围：

| 四请求后缀长度 | 打包 PA（ms） | 逐步 PA（ms） |
|---|---:|---:|
| 1 / 1 / 1 / 1 | 5.22 | 5.17 |
| 2 / 2 / 2 / 2 | 5.38 | 23.16 |
| 4 / 4 / 4 / 4 | 5.48 | 45.82 |

两次失败均已保留原始报告与源码快照，实验源码和冻结基线校验通过，进程全部清理。正在分别补充同批次 eager/graph 与独立参考的误差定位；保持原始容差和资格门，不把子范围耗时改善当作完整负载收益。最新完整树形正式结果仍为 **Goodput 239.13**，串行对照 **501.64**。

预算的物理成本审计还发现：B≤5 混批时只要存在B4/B5，所有请求仍执行四步 draft；正式阶段低B仅省去2个真实 PA query。98.67%的草稿链前仍发生修复。预算选择、真实计算步数与修复成本需要一起优化。现有正式结果没有设备时间线，不能由不同 rank 的累计 phase 耗时相加推断精确 overlap。证据：`tree-backbone-budget-v17/BUDGET_PHYSICAL_COST_AND_OVERLAP_AUDIT.json`、`tree-packed-PA-repair-v21/NPU_DIAGNOSTIC_FAILURE.json`。

首case定位补充（v22）：同M165的 mixed graph、mixed eager，以及逐请求隔离后用另一独立scratch补到M165的 eager，全部hidden和全层query KV均逐位一致；四种执行模式的已提交prefix KV均未改变。与小M5/5/128/27逐请求参考相比，742913/844800个hidden元素超出原容差，max绝对差4.0、RMS0.0410；10个tree verifier token仍一致，prompt中间位置有2处argmax不同。该case为partial prompt，不能把空的first-token检查当作最终prompt token验证。三个target rank的诊断一致。

这说明首case的差异随物理M及其计算路径变化出现；M165另有16个NZ布局候选模块，现有对照不能再区分NZ、MC2与矩阵乘形状各自的原因。原始跨M资格门仍失败，未来KV/全分支尚未通过。下一步采用独立且匹配M的参考继续验证数学隔离、原始1e-3容差、exact token、原生changed-input与KV提交，保留既有失败归档。证据：`tree-mixed-prefill-diagnostic-v22/NUMERICAL_DIAGNOSTIC_SUMMARY.json`。

后续资格进展：v24的线性树partial/final两组通过matched-M/N的全hidden、全query KV、精确token、原生提交与后续query检查；首个真正分支case2在独立known-path重建处仍失败，未封存，不能启用在线混合prefill。该重建除M/N外仍改变了真实请求Q/KV长度，正在独立控制这些变量定位，尚无最终根因。

打包PA修复v25成功完成15/15几何组及原有96/96分支/提交后参考检查。其独立参考从原初始prefix逐token eager PA重算，只把每步物理行数固定为候选档位；异长后缀的原生1e-3 hidden与exact next-top2均通过，未复制候选KV。原先small-M参考在最后单请求步骤出现的hidden及top-2差异继续保留，不宣称跨M等价。资格通过后，12项新增范围的五次采样均零捕获/验证/回退，异长四请求一次修复中位约5.94–6.04ms。实验源与冻结校验、退出清理通过；尚未接入新的全量性能候选。证据：`tree-packed-PA-matched-oracle-v25/NPU_QUALIFICATION_SUMMARY.json`。


## 四阶段 draft：修复与 root 合批的独立实测

v27将已知 committed suffix 与当前已知 root 放进一次原生PA图，从root hidden计算原top-2，再接原生三步草稿图。四步主链、候选/概率编码和后续验证不变；模型阶段从“修复一次+主链四次”的5次减为4次。root top-2在图外，计入完整调用耗时。无修复、eager、短链或超容量时使用原FastV5，绝不截断已知后缀。

18组新几何检查及原96组正确性参考全部通过，覆盖N1/4/8/15/16、混合后缀长度、跨页、hidden、encoded概率、token、ledger、全层KV与原生changed-input/finite门。随后在prefix=127的三个计时fixture中，独立对真实新旧方法交替各采5次，所有30次完整调用的捕获/验证/回退增量为0，方法内部输出稳定，跨方法token、置信度及ledger相同。该耗时范围不覆盖真实在线长prefix（本次g标定context长度92…2131）：

| 正常请求N / 需修复请求R | 原FastV5（ms） | 四阶段root-seed（ms） | draft调用加速 |
|---|---:|---:|---:|
| 4 / 4 | 25.26 | 21.90 | 1.154× |
| 15 / 8 | 28.10 | 25.78 | 1.090× |
| 16 / 8 | 27.59 | 24.58 | 1.122× |

该计时包含元数据、图外root head、最终encoded回读和本地完成fence，不包含外部模型prefill或独立oracle。原96检查附带的旧计时有3项发生冷图捕获/验证，不能用于成本结论；上表独立配对30次均通过计时门。该结果不能直接换算成完整Goodput；新的463请求候选正在准备，完整树形最新正式值仍为239.13。证据：`tree-root-seed-draft-v27/NPU_QUALIFICATION_AND_PAIRED_COST_SUMMARY.json`。

补充审计修正：旧独立审计脚本检查未加`_delta`后缀的测量增量字段，缺失默认0；新脚本将要求实际字段存在并检查正确后缀。已对B17与同卡串行结果重新逐字段核查，四个rank六项实际增量均存在且为0，旧正式数值未因此失效。证据：`budget-diagnostics/FORMAL_NATIVE_DELTA_FIELD_REAUDIT.json`。


局部提交验证补记：v29在原case2真实分支布局下，完成root零接受、primary与side三条路径，各配offpath0/1，共6组独立proof；三个target rank的全hidden、query KV、提交后canonical KV及future hidden/token均逐位一致。参考从初始prefix独立重算稀疏source KV，再自行保存源值并index-copy，未使用actual KV作为参考。此结论仅为相同稀疏物理布局下的局部提交正确性；v24/v28的dense known-path重建失败保留，完整AR等价性仍未通过，混合prefill尚未在线启用。证据：`tree-sparse-local-commit-v29/NPU_LOCAL_COMMIT_SUMMARY.json`。

v30已启动完整463请求root-seed候选：保留B17的Poisson轨迹、B≤5、g先验、MC2与原封存协议，只接入已通过v27的四阶段draft实现。正式Goodput尚未产生。


v30全量测试未产生有效正式结果：7轮完整预热后，第6/7轮连续零捕获/验证并封存；第8轮正式阶段约59秒遇到未缓存的target FULL图键（16个物理请求段、M80、mask 16×1×9×4096），被原sealed禁止补捕获门拒绝。未启用回退、未放宽门，不能用预热耗时报Goodput。源码未变、冻结后校验与进程清理通过，原始证据保留于`tree-root-seed-online-v30/FAILED_FORMAL_SUMMARY.json`。最新有效全量树形Goodput仍239.13，对照串行501.64。下一步在新版本预先覆盖可达物理图键，同时继续独立g标定。


## B5 同物理布局的 g 标定：72 上下文与留出检验

v32成功完成原18组几何、原96组分支/KV回归后，采集24个独立请求×3位置=72上下文。每类训练/留出各4个独立请求、12个上下文，同请求所有位置不跨组；四rank标签完全相同。每批标签经过原映射、结构/finite投票、compaction及实际B5下一轮draft修复后才保存。下表为每轮有效交付token均值，按n=1…5排列：

| 请求类型 | 训练 g(1…5) | 留出 g(1…5) | B5留出逐上下文MAE（token） |
|---|---|---|---:|
| 代码 tight | 1.92 / 2.58 / 3.17 / 3.67 / 3.75 | 1.75 / 2.42 / 3.08 / 3.50 / 3.67 | 1.25 |
| 数学问答 normal | 1.83 / 2.58 / 3.17 / 3.75 / 3.92 | 1.75 / 2.50 / 3.00 / 3.50 / 3.67 | 1.35 |
| 摘要 loose | 1.83 / 2.17 / 2.25 / 2.33 / 2.58 | 1.83 / 2.58 / 3.33 / 3.92 / 4.17 | 1.94 |

侧分支共13/72次多交付1个token（18.06%）；各类/分组的请求聚类bootstrap区间下限仍为0，样本不足以证明稳定侧支收益。摘要B5训练2.58、留出4.17，类均值存在较大泛化差异；三条类均值曲线可作先验，尚不能直接作为逐请求可行预算保证。正在检验只用过去观测的个性化g预测。

这些g1…4由同一次B5物理forward的原target successor与accepted path截断推导，并以冻结原host verifier逐项核查；并非独立较小M的B4执行。实际B5后续修复45个请求需1个token、27个不需修复，不能据此推断B4反事实修复成本。本轮没有Goodput或合格时延结论，完整AR资格仍未完成。证据：`tree-B5-path-counterfactual-v32/NPU_CALIBRATION_SUMMARY.json`、`CURVES_TRAIN_HELDOUT.csv`。


预测器补记：只用过去真实g做请求级更新的留出MAE约1.098→1.063，但请求聚类区间跨0、代码/数学两类未改善，未采用。完整draft之后已有的原始joint confidence预测MAE约0.617，较类均值1.098更好；额外isotonic模型的改善未有稳定证据，side预测变差且出现互斥概率约束违例，因此拒用该模型。当前轮confidence只能用于target refinement研究，不能冒充草稿前预分B或省draft计算；用过去confidence预估下一上下文也没有稳定改进证据。见`budget-diagnostics/V32_PREQUENTIAL_GAIN_ANALYSIS.json`、`V32_ISOTONIC_GAIN_ANALYSIS.json`、`V32_LAGGED_CONFIDENCE_GAIN_ANALYSIS.json`。


## 混合普通修复与 eager 的 draft 调用：V35 局部结果

25 组新混合几何的独立 hidden/KV/top-2/置信度/frontier/ledger 检查通过；旧路径 241 组 eager 与 96 组普通 KV 回归通过。两条完整零发现轨迹后封存，30 次计时调用均零捕获、零运行时验证、零回退。源码、冻结基线与退出清理检查通过。

| 普通 / eager 请求数 | 原 F16 中位耗时 (ms) | 混合路径中位耗时 (ms) | 原耗时 / 新耗时 | 完整输出与状态等价 |
|---|---:|---:|---:|---|
| 15 / 1 | 38.18 | 35.46 | 1.0766× | 否：eager 置信度不同 |
| 16 / 1 | 38.40 | 37.08 | 1.0357× | 否：eager 置信度不同 |
| 15 / 2 | 38.48 | 35.77 | 1.0758× | 是 |

每种方法每组 5 次交替计时，范围为 draft 整段调用及 NPU 完成，包括 PA、head、tail4 与最终 D2H，不包括前缀建立、父草稿构造及参考验证。全部成本样本的前缀长度仅为 127，不能直接推广到真实长上下文。前两组 token、root、frontier 和 ledger 相同，但 eager 置信度不同，因此仅记录原始耗时比，不宣称等价结果加速。第三组完整签名一致，局部等价加速约 1.0758×。

尚未获得新混合路径的在线提交、拒绝、promotion 与后续 child 消费的完整 NPU 资格，也没有本版本在线 Goodput。全负载 AR/token 一致性仍未通过。详细证据见 `budget-calibration-20261015/tree-root-seed-mixed-eager-v35/NPU_QUALIFICATION_AND_PAIRED_COST_SUMMARY.json`。


## V40 初始化失败：MC2 资格字段契约错误

V40 采用 44 个固定 target 解码图档位，CPU 几何与源码检查通过。设备端 201 个 prefill 图验证成功；进入解码图资格检查时，新 helper 错把 `_mc2_deferred_row_counts` 当作普通全链 MC2 可用行集合，导致初始化拒绝。冻结 profile 没有 `deferred_read_done_chain`，该专用字段按原生实现必为空；普通 chained flush 的资格来自 `_mc2_chained_row_counts`。这是新增检查的字段契约错误，不能据此认定 MC2 实际不可用。日志未分别打印其他前置条件的运行值，因此不推断其他条件也已通过。

本轮没有完成任何完整 463 请求预热，没有封存或正式 Goodput。源码、gain model、冻结基线前后校验通过，进程已清理，四卡空闲。失败版本保留不改；下一隔离版本将正确检查全链可用行、128 条路由、逐形状资格及实际 dispatch，并在昂贵 prefill 准备之前执行前置检查。


## V44：MC2 检查修复通过，图准备触及显存预留线

修正 MC2 字段后，四 rank 实际 preflight 成功；三 target 各 128 路由、22 个全链行档位与冻结 profile 身份全部一致，201 个 prefill 图验证成功。第 26/44 个 target fixture（FULL M88/S17）完成四 anchor 和第二见证的数值/key 检查，各 target 实际 MC2 success=1024、fallback/exception=0；随后 device free 降到约 5.80–5.86 GiB，触发 6 GiB 预留门槛。44 图整体尚未 qualified，draft66 未开始，没有完整 463 请求预热或正式 Goodput。源码、gain、冻结基线和清理检查通过。

该时点 allocated 较 prefill 完成后增加约 17.51 GiB，其中 Q/O retained 仅约 1.304 GiB，不能将全部增量归因于 Q/O。源码确认 FULL FIA workspace 只复用已有的逐维更大容量，升序捕获会累积多个容量 owner；实际 FULL workspace 字节尚未记录，仍需实测。下一隔离版本先捕获覆盖全家族的最大容量，并增加 workspace 实际占用统计，保留全部 44 图的原数值、MC2 与显存门槛。


## V46：最大 FULL 图优先捕获，完整 root-seed＋提前 target 正式结果

V46 在不改 44 个解码图键、数值/MC2 检查或显存门槛的条件下，先捕获 FULL M165/S38/mask9，再准备其余档位。三张 target 卡实际各只保留一份 FULL workspace（251 MiB），pool 与 task 引用的全部 FIA workspace 合计约 572.1 MiB，未发现额外 task-only storage。201 个 prefill、44 个 target、66 个 draft 档位资格通过；target 图准备后剩余显存约 11.06–11.10 GiB。最大图首次捕获的约 10.7 GiB allocated 跳变尚未全部归因，不能把全部显存改善都归因于这 251 MiB。

完整 463 请求、Poisson 4 req/s、seed0、每请求 256 输出、相同四卡与 MC2 on。实际 3 轮完整预热，第 2/3 轮连续零新增后封存，第 4 轮正式测量；以下统一采用报告的 paper TPOT/Goodput 口径：

| 方法 | Raw (token/s) | Goodput (token/s) | E2E (s) | 代码达标 | 数学问答达标 | 摘要达标 | 总达标 |
|---|---:|---:|---:|---:|---:|---:|---:|
| 原 B17 树形 | 585.81 | 239.13 | 202.33 | 3/271 | 90/96 | 96/96 | 189/463 |
| V46 树形：root-seed＋提前 target | 608.71 | 259.00 | 194.72 | 8/271 | 93/96 | 96/96 | 197/463 |
| 同卡固定串行 γ=4 | 921.66 | 501.64 | 128.60 | 61/271 | 95/96 | 96/96 | 252/463 |

V46 相对 B17 的本次 Goodput 高 8.31%，Raw 高约 3.91%；只有串行 γ=4 的 0.5163×，尚未达到超过串行的目标。本次为多个改动组合的一次正式观察，不能归因于某一个优化，也未证明稳定重复收益。actual root-seed 调用 2359 次，三 target 各提前提交 2379 个合格验证批次；这些 host 累计耗时不等于设备 overlap 的精确测量。

独立正式审计通过，源与 gain 未变、冻结基线前后校验通过、进程清理后四卡空闲。完整负载 AR/token 等价性仍未通过，局部资格不替代此限制。observed TPOT 的另一口径得到 256.37 token/s、195/463，不与上表混用。证据：`budget-calibration-20261015/tree-largest-first-bounded-v46/full-r1-summary.json`、其正式 `result.json` / `independent-tree-audit.json` 与 `budget-diagnostics/V46_TARGET_INVENTORY_MEMORY_AUDIT.json`。

正式后审计另确认：84 项实际测量增量字段（21 项×4 rank）全部存在且为 0，124 个冻结源及 120 个运行快照源重新 SHA 一致。单独 online prefill 在 target leader 上累计 52.8468 s、418 批、463 请求，已含在整轮 decode/E2E 内，不能被初始 prefill 的微小字段掩盖；eager promotion/invalid/KV move 均为 0。三 target 累计 target phase 约 112.17–112.22 s、draft 约 99.86 s，不能跨 rank 相加。详见 `budget-diagnostics/V46_FORMAL_NATIVE_AND_PERFORMANCE_AUDIT.json`。

接下来先验证真实混合 prefill：V45 源已冻结，含原 partial 复跑、N16 final101 和 N15 partial100→final2 的独立递进参考。4 项新增 CPU 检查已实际通过，独立核对 29 冻结源、25 测试输入与 170 依赖哈希后，已启动四卡 NPU 诊断；尚未启用在线生产路径，也没有本诊断的 Goodput。动态 B 还需处理物理档位问题：V46 的 N16/B4 当前被映射到 causal M128/S38，而 B5 是 FULL M96/S16；不能把预算数减少直接视为计算量减少。


## V45：真实 mixed prefill 的 partial、final 与递进续传资格通过

实际执行原 V42 partial 的 false/true/false 三变体，并新增 N16 final101（97 行 head）、N15 partial100（90 行 head）→final2（91 行 head）。全部保持真实 FULL M197/S17/mask128；三 target 的全 197 行 hidden、64 层 query KV、精确 tree/最终首 token 通过原容差与原生 changed-input/finite/MC2 检查。N15 独立参考只从原始 token 预填至 2199，再分别实际计算 phase1，为 real 与独立 scratch prompt 各自补齐到 2299，随后检查最后 2 token；没有复制候选 KV，也未用 dense 重填 2299 替代递进历史。

draft rank 的实际 chunk101 与 chunk100/2 均经独立 hidden 和 28 层 query KV 参考验证；原 finite、CPU fence、target-owned WORLD first-token 协议通过，partial 不生成首 token，prompt 状态未提前发布。各 rank 两条完整 trace 的 12 项原 native 增量均存在且为 0 后封存。最终 target reserved 约 44.54–44.56 GiB、peak 约 42.36 GiB、free 约 14.89–14.92 GiB，源码/冻结基线/清理复核通过。

这是具体 forward/递进 chunk 的局部设备资格，尚未执行在线 tree controller commit 或请求发布，不能视作在线混合 prefill 已启用，也没有本诊断的 Goodput。全负载 AR/token 等价性仍未通过。原证据：`budget-calibration-20261015/tree-mixed-prefill-progressive-v45/context-cost-sweep-v1/mixed-progressive197-v1/` 与 `cost-receipt-mixed-progressive197-v1.json`。下一步同时补齐 promoted child 消费资格及 B4/B5 真实物理成本。


V43 promoted-child-consume 探针补记：本轮在首个 setup 因调用不存在的 `paired_mixed_root_seed.fixture_sources` 失败（实际 helper 位于其 `v27` 导入）；原 18 组完成 0、新 3 组未进入。这是诊断脚本接口错误，尚未触及本轮 target 数值/提交检查，不能视为算法失败或取得了局部资格。失败源码保留，前后冻结校验和退出清理通过；下一隔离版本修正接口，并补充实际执行 setup 的 CPU 契约检查。


V48 修复探针接口后实际通过旧 case0…3（observed target、valid promotion、frontier mismatch、parent reject），在 case4（prefix127、same-token changed-ancestor 反事实）被原生 tail entry 的 `runtime_validated=False` 拒绝。后续 independent hidden/KV 对照尚未进入，不能称数值 mismatch；新 3 个 child 消费 case 仍未执行。源码、冻结基线和清理通过。已保留失败证据并准备明确的 tail changed-input 诊断，禁止赋资格位或跳过门。


V47 实际通过 N16/B5 与 N16/B4 各 false/true/false 共 6 组资格；首个 N15/B4 的独立 tail allhidden 原比较失败，未执行 fixed-point、seal 或成本计时。已保留失败及清理证据。源码确定参考每步重置 dummy 输入，而真实链会递推其 top1；这是一条待实测验证的参考契约差异，尚不能认定失败只在 dummy 行。新隔离版本将补完整 physical 输入递推与逐step/row诊断，继续比较所有行。

已通过的 N16 原输入实际交付为 B4 合计56、B5合计57 token，仅1/16请求多交付1；该数据来自各自真实 target forward/verifier，而非B5路径截断。仅16个请求、output offset96的一批，且成本未 qualified，不能推断稳定类曲线、最优B或Goodput。见 `budget-diagnostics/V47_N16_ACTUAL_G4_G5_PARTIAL_EVIDENCE.json`。


## 2026-10-15 暂停交接：当前检查收尾

按用户“测完这个先暂停”的要求，当前已启动的 V50 CPU 检查结束：7 项中 4 通过、2 失败、1 错误；未在暂停后修复或重跑。V49 首轮为 7 通过、1 错误，随后准备阶段因 V46 凭据字段契约失败，未启动第二轮；V51 仅保存未测试的参考修复源码，CPU/NPU 均未启动。三个准备版本都不可直接启动设备实验。失败记录、源码哈希和未完成事项已保存，当前无 CPU 测试或模型进程，物理卡 2/5/6/7 实查均空闲。

最新有效全量树形仍为 V46：Goodput **259.00 token/s**、Raw **608.71 token/s**、达标 **197/463**；同卡固定串行 γ=4 为 **501.64 token/s**，树形仅为其 **0.5163×**，尚未超过串行，完整 AR/token 等价性也仍未通过。后续先补齐真实 B4/B5 成本、promoted child 消费资格和 FULL 图两线程更新诊断，再考虑在线混合 prefill；全部等待用户恢复后进行。

统一恢复入口：[暂停检查点](../budget-calibration-20261015/PAUSE_CHECKPOINT_20261015.json)。分支交接：[V49](../budget-calibration-20261015/tree-FULL9-FIA2-v49/HANDOFF_PAUSED_V49.json)、[V50](../budget-calibration-20261015/tree-root-seed-online-mask9-v50/PAUSED_HANDOFF_V50.json)、[V51](../budget-calibration-20261015/tree-pre-B4-B5-cost-v51/PAUSED_HANDOFF.json)。


## 2026-10-16 恢复：真实 B4/B5 成本与预算瓶颈

恢复时，旧 CANN9.0/ATB 绝对路径缺失导致 V51 首次在模型启动前被原锁校验拒绝。私有保留运行环境实际完整，13 个关键文件 SHA 与原锁全部相同；仅恢复两个缺失路径的符号链接，未下载、重装、改 CANN9.1/latest 或重写旧 lock。原完整校验重新通过（10496 个冻结文件、15 个外部文件、212 个包）。后续可用根目录 `restore-retained-runtime-aliases.py` 先只读检查，确有缺失时加 `--restore` 恢复；仍需原完整 verifier 通过。

V51 真实设备测试完成：四 rank 原 9 项资格通过，包括原失败的 N15/B4 场景；完整 physical dummy 输入递推参考修复后，所有原比较门均通过，未缩小比较行或修改容差。两条完整零发现轨迹后封存，每方法 5 个交替配对样本合格。采用 draft rank 的整段 sequential local round＋下一轮 proposal（包括修复）墙时：

| 方法 | 物理 target | 整段中位耗时 (ms) | 16 请求实际交付 token | 交付/整段耗时 (token/s) |
|---|---|---:|---:|---:|
| 预选 B4 | causal M80/S16 | 144.689056 | 56 | 387.04 |
| 预选 B5 | FULL M96/S16 | 145.294474 | 57 | 392.31 |

B5 中位成本高约 0.4184%，这批交付量高约 1.7857%，按交付/中位成本计算约高 1.36%。5 对成本差有正有负；这里只是一个真实 N16 上下文批次，不能推断稳定人群收益、混合 B 的单请求边际成本、生产双 batch overlap 或全负载 Goodput。V51 局部 B4 用 M80/S16，当前 V46 生产 N16/B4 仍映射到 M128/S38；两者不是同一个物理执行。正式完整树形 Goodput 仍为 V46 的 259.00，对照串行 501.64，目标尚未达到。

预算分布审计确认：V46 单 rank 的 36027 条实际 proposal 中，B5 为 35574 条（98.7426%）；`current_forecast_unfit=17358` 是重复调度窗口计数，不是独立请求数。现有策略满足最低预测需求后仍继续购买正预测增益，N16 全 B5 仅用 80/96 候选预算，因此几乎填到 B5。后续需同时考虑整批 FULL/causal 族及实际 M/S 成本，不能只凭三类请求的均值曲线或认为减少候选必然更快。

证据：`budget-calibration-20261015/budget-diagnostics/V51_ACTUAL_LOCAL_COST_AND_QUALIFICATION_AUDIT.json`、`V51_ACTUAL_B4_B5_BATCH_COST_EVIDENCE.json`、`V46_B5_SATURATION_AND_PRE_B_BATCH_PLAN.json`、`V46_B4_B5_PHYSICAL_MAPPING_STATIC_AUDIT.json`。

当前继续独占测试 V53 的 FULL96/S16 两线程 FIA 参数更新；V52 原 18+3 promoted-child 协议诊断已完成 CPU/源码审计，等待设备顺序执行。两者尚无新的性能或在线资格结论。


V52 后续草稿协议设备检查完成：四 rank 原 18 项加新增 3 项 promoted-child 实际验证/消费/提交检查全部通过；源码、原环境锁、退出清理和四卡空闲复核通过。实测原 case4 的四次 root 改动都产生相同 tail seed=13，原生 changed-input 门没有观察到新 tail 输入；在未发布未来槽内，以同 key seed14 完成一次原生 changed-input 数值校验（validation1、changed1、failure0），随后完整真实方法恢复原 token、置信度、ledger 和全部 query KV，保护前缀保持一致。这是局部协议资格，未启用在线路径，也未完成全负载 AR 等价或 Goodput；该纯数值 scope 不包含两轮零发现/封存/成本测量。

V53 FULL 两线程诊断仅完成 prefix127 的 serial/parallel2 两项资格；prefix128 的旧 key replay 在 MC2 检查中被拒绝。该检查仍错误要求 capture/eager 阶段的 host dispatch 正增长，但失败时原始 delta 未存，尚不能确定实际触发的是哪项计数条件，也不能据此断言 fallback 或数值失败。未进入零发现/封存/计时。V53 失败源保留，新 V56 分开 cold capture 与 existing replay 的计数契约，并在门前保存所有原始计数；独立 eager reference 每次仍要求真实 MC2 正增长。


V56 FULL 两线程更新正式局部成本：不采用。修正诊断计数口径后，三 target 各 6 组数值资格、2 条完整零发现轨迹、原生封存及每 prefix/mode 5 次成本采样均通过。以下仅列 rank1 的完整 target 调用＋设备完成等待中位数，不跨 TP rank 合并样本：

| prefix | 原串行参数更新 (ms) | 两线程参数更新 (ms) | 两线程增加 (ms) |
|---:|---:|---:|---:|
| 127 | 45.211915 | 47.127884 | 1.915969 |
| 128 | 44.785843 | 47.001432 | 2.215588 |
| 255 | 45.255181 | 47.546838 | 2.291657 |

另两 target 同方向；各 rank/prefix 的 5 个配对样本均为两线程更慢，中位差约 1.85–2.29 ms（4.10–5.11%）。因此不接入当前版本，也不为此扩大生产资格。2016 个必需 native 增量字段全部存在且为零，parallel 调用均实际 1 次/64 task，serial 为 0；原 token、hidden、全层 KV、MC2、源码和清理检查通过。证据：`budget-calibration-20261015/budget-diagnostics/V56_ACTUAL_FIA2_WHOLECALL_AND_NATIVE_AUDIT.json`。

下一项为 V57：同 12 个真实上下文（每类 4 个），比较当前 primary4＋四个 terminal-side 候选池的 B4…8。该候选池与旧单支延伸到 depth7 的树不同，旧 g6…8 不可继承；新代码尚在独立参考与 CPU 契约准备，未启动设备标定，不能给出新收益曲线或 Goodput。


V57 设备标定更新：实际完成 1/15 组资格，第二变体的下一轮 proposal 被新增的首步完整 physical hidden 比较门拒绝；原 before-tail real hidden、后三步完整 hidden 和 encoded token/置信度门通过。该新增观测在 head/tail 结束后读取 resident entry 输出，诊断差值异常大；输出缓冲生命周期问题尚未实证确认，不能据此断言候选算法数值失败。尚未进入两轮零发现、封存或成本采样，未形成 B4…8 收益/耗时结果。47 个冻结源、42 个实际运行输入、原环境锁和清理检查通过，四卡空闲。失败证据保留于 `budget-calibration-20261015/budget-diagnostics/V57_RESIDENT_FIRST_HIDDEN_FAILURE_AUDIT.json`。隔离 V58 将在实际首步返回后、head/tail 前取完整快照，并保留全部原比较门后重测。


## 2026-10-16：当前 terminal-side 候选池的 B4…8 标定完成

V58 在实际首步 `run_tree_hidden` 返回后、head/tail 前立即复制完整输出，修正 V57 的过晚观测；该 full snapshot 仅用于资格检查，成本分支直接调用原方法。原 before-tail real、全 physical hidden、后三步 hidden、精确 token/置信度、全层 KV、提交和下一轮修复门均保留。四 rank 各 15 项资格、两条完整零发现轨迹、原生封存及每预算 5 个交替采样全部完成，原环境锁/源码/退出清理通过。当前四卡空闲。

以下是同一批 12 个真实请求、每类 4 个、output offset96 的局部观察；耗时包括完整 sequential round 加下一轮 proposal/修复，包含原 root diagnostic 和 tail observer，原上下文预填与独立参考不计入。此项不包含生产双 batch overlap。

| 预选预算 | 实际 target M/S | 整段中位耗时 (ms) | 12 请求交付 token | tight 平均交付 | normal 平均交付 | loose 平均交付 |
|---|---|---:|---:|---:|---:|---:|
| B4 | causal 60/12 | 148.265 | 40 | 2.75 | 4.00 | 3.25 |
| B5 | FULL 72/12 | 148.002 | 41 | 3.00 | 4.00 | 3.25 |
| B6 | FULL 84/12 | 149.208 | 43 | 3.00 | 4.25 | 3.50 |
| B7 | FULL 96/12 | 149.055 | 43 | 3.00 | 4.25 | 3.50 |
| B8 | FULL 116/14（108 真 query＋8 scratch） | 151.156 | 43 | 3.00 | 4.25 | 3.50 |

在这个样本中，B6 比 B4 多交付 7.5%，中位成本高约 0.636%；交付/中位局部耗时比约 1.0682。B7/B8 没有再增加交付，B6/B7 的微小耗时差也不能解释为稳定收益。这不是全负载 Goodput 提升；不同预算有不同 M/packing/attention family，各自同 M 数值资格不能证明跨 M 的 target token 等价。每类仅 4 请求，尚不足以确认三条通用 g 曲线。

下一项 V59 将使用已有 24 个独立请求各 offset32/96/192 的 72 个真实上下文，分成 6 个 N12 cohort，分别执行 B4…8 实际 forward/verdict/commit/nextrepair 的完整数值检查，验证候选增益是否普遍存在。它是纯数值 g 标定，不重复 V58 成本测量，也不继承在线或性能资格；同一请求的三个 offset 不是三个独立样本。最新有效完整树形仍为 V46 的 Goodput **259.00 token/s**，同卡固定串行 γ=4 **501.64 token/s**，尚未超过串行。

数据：`budget-calibration-20261015/budget-diagnostics/V58_ACTUAL_CURRENT_POOL_LOCAL_GAIN_AND_COST.json`；原始记录与收据：`tree-terminal-pool-gain-v58/context-cost-sweep-v1/terminal-pool-gain-early-first-v1/`、`cost-receipt-terminal-pool-gain-early-first-v1.json`。

V58 后审计补充：1680 个必需 native 增量字段逐项存在且为 0；195 份 MC2 原始报告通过。30 份首步/后续诊断的 real/PAD、tail 原比较全部通过；48 个冻结源、42 个 CPU 输入及 44 个实际运行输入重新 SHA 一致，运行快照与清理检查通过。独立审计：`budget-calibration-20261015/budget-diagnostics/V58_ACTUAL_TERMINAL_POOL_G_NATIVE_AUDIT.json`。


rolling 窗口的下一隔离实验 V61 已完成 12 项 CPU 合约检查并冻结，尚不能启动全量。V46 正式观测中，target worker 每次平均约 36.974 ms 提交、4.395 ms 本地调度、5.799 ms 剩余等待，整段约 47.169 ms；草稿 normal16 的最终整段成本 EMA 为 42.395 ms。现有窗口估计将草稿整段成本与 target 提交后的残余比较，可能漏掉两独立 worker 的 CPU 并行时间。V61 只改为使用完成原 late fence 后观测的 target 整段 worker 墙时作预测，并保留付费首 probe、实测额外开销和全部原正确性门；该值不是 NPU 耗时，也不能直接证明 eager 已被隐藏。

V61 的原 B5、44 target 图、gain、到达/SLO 计时与内存/MC2/提交机制不变；需先完成 V62 的真实 FastV5 大混合批次独立数值与提交资格。V52 原 18+3 使用的是 V35 mixed 方法、normal1+eager1，不能替代 FastV5 normal15+eager1 等组合的完整证明；V35 虽实际运行过 FastV5 的 15+1、16+1、15+2 并得到同方法稳定性/零发现，仍缺各自独立全 physical hidden、KV 和高 N 子草稿消费闭环。当前 V59 正在进行 72 上下文数值 g 标定，V61/V62 均未启动 NPU，全量 Goodput 仍无新结果。


## 2026-10-16：72 上下文的 g 数值记录完成，摘要回传失败

V59 四 rank 均实际完成 30/30 数值案例并保存原始记录，无数值门错误；结束时 `calibrate_base.worker` 读取新报告缺失的 `gain` 字段触发 `KeyError`，因此子进程和主管收据最终为失败，`completed.json` 不存在。此处保留原失败收据，不将整项标为成功，也不为补一个摘要字段重复执行 30 个 NPU 案例。以下仅使用失败实验中已经完成的原始数值标签；源码、原环境锁和退出清理复核通过，四卡空闲。后续 driver 会补上实际 worker 最终回传契约的 CPU 检查。

24 个独立请求各取 output offset32/96/192，共 72 个相关上下文；每类 8 个独立请求、24 个上下文，五种预算各有 72 个实际交付标签：

| 预算 | 72 上下文交付总数 | 全部平均 g | 代码 tight 平均 g | 数学 normal 平均 g | 摘要 loose 平均 g | 接受 terminal 侧支的上下文数 |
|---|---:|---:|---:|---:|---:|---:|
| B4 | 248 | 3.4444 | 3.5833 | 3.6250 | 3.1250 | 0 |
| B5 | 261 | 3.6250 | 3.7083 | 3.7917 | 3.3750 | 13 |
| B6 | 267 | 3.7083 | 3.7917 | 3.8750 | 3.4583 | 19 |
| B7 | 270 | 3.7500 | 3.8750 | 3.9167 | 3.4583 | 22 |
| B8 | 269 | 3.7361 | 3.8750 | 3.9167 | 3.4167 | 22 |

这是当前 primary4＋四个 terminal-side 候选池的实际 g，不继承旧 depth7 曲线，也不包含新成本、零发现封存或在线 Goodput 资格。B7 相对 B4 的交付量在该样本高约 8.87%；预算需求与物理档位成本也增加，不能将其解释为性能提高 8.87%。B8 与 B7 的侧支接受覆盖相同。

跨布局 token 等价存在明确反例：`rps4p0-seed0-000162`、offset96，B7 的 M96/S12 输出 `[11,1340,594,803]`，B8 的 M116/S14 输出 `[11,1340,3855]`，第三 token 已分歧，g 从 4 变 3。两者各自的 matched-M 独立数值/KV 门通过；尚未确定这两个布局哪一条等于 dense AR，不能把 B8 下降仅归因于侧支无收益，更不能宣称跨布局精确 token 等价或全负载 AR 等价通过。该分歧的统一数值原因未证明，模型精度仍为原 BF16。

三条类别均值不能直接作为每请求确定性 g：例如摘要 B4 在原 train/heldout 的均值分别为 2.3333/3.9167；各类单上下文 g 均覆盖 1…5。原 train/heldout 请求身份虽跨 offset 保持分离，但 V58 pilot 已观察其中 heldout 请求（代码3/4、数学2/4、摘要3/4），此处的预测误差只能描述，不能称未触碰测试集的泛化结果。后续分配应把类别曲线作为先验，并结合每请求已观测历史、真实 physical bucket 成本与预测不确定性，不能把类别均值不足推导为该请求必定无法达标。

原始记录：`budget-calibration-20261015/tree-terminal-pool-context-gain-v59/context-cost-sweep-v1/terminal-pool-context-gain-v1/rank*-context-gain.json`；失败收据：`tree-terminal-pool-context-gain-v59/cost-receipt-terminal-pool-context-gain-v1.json`；可复算分析脚本与结果：`budget-diagnostics/analyze_v59_gain.py`、`budget-diagnostics/V59_ACTUAL_G_CLASS_SPLIT_OFFSET_ANALYSIS.json`。最新有效全量树形 Goodput 仍为 **259.00**，固定串行 γ=4 为 **501.64 token/s**。

V59 后审计：四 rank 共 120 个实际数值记录、1440 个 native 字段齐全；840 项禁用 fallback/failure/disabled 检查全为 0（资格阶段捕获/校验允许正增量）。180 份 MC2 原始报告、60 份首步/尾步/PAD 诊断通过；50 个冻结源、44 个 CPU 输入、46 个实际运行输入及快照哈希复核一致，清理通过。主管最终回传失败状态保留。证据：`budget-calibration-20261015/budget-diagnostics/V59_ACTUAL_NUMERICAL_AND_REPLY_FAILURE_AUDIT.json`。


### 2026-10-16：V62 启动配置检查失败，保留失败记录

V62 的 18 项 CPU 合同通过，但没有覆盖真实 PEARLConfig 构造；实际启动被原配置检查拒绝：`gamma=5`、`spec_rhythm_max_eager_tokens=8` 超出允许上限。失败发生在模型创建前，实际数值和协议记录均为 0，没有新的 Goodput 结果。原环境前后检查通过、72 个冻结文件未变、子进程已清理且四卡空闲。

后继 V64 使用与 V46 实际生产版本相同的框架配置 `gamma=8`、eager 上限 5，实际计划和 ticket 仍固定 B5，验证范围仍为 27 个数值记录和 12 个协议记录；新增直接构造真实配置的 CPU 合同。V62 冻结源和失败 receipt 保留，V63 窗口改动的完整性能实验仍等待实际数值与协议验证。审计：`budget-calibration-20261015/budget-diagnostics/ROOT_V62_ACTUAL_CONFIG_STARTUP_FAILURE.json`。


### 2026-10-16：V64 混合草稿数值与父子协议验证完成

配置修正版 V64 的四个 rank 均实际完成 **27 个数值记录 + 12 个协议记录**，错误列表为空，并正常返回原 worker 的四条 `calibrated(rows=0,timings=0)` 消息。三个混合组合为正常请求 15+提前草稿 1、16+1、15+2；数值前缀为 127/128/255，协议前缀只测 128。检查覆盖原 FastV5 的 repair PA、frontier PA、完整四步草稿，以及独立参考的全部物理行 hidden、token/置信度、全部层查询 KV 和 ledger。

协议保留真实父批裁决，另外用明确标注的控制输入覆盖有效晋升、父拒绝及相同 frontier token 但祖先已变的失效情形。有效晋升后，owner 15、owner 16 以及两者同时晋升均进入真实 target 验证、提交和下一轮修复，未用合成孩子裁决替代。父批物理形状为 FULL96/S16 和 FULL116/S22，孩子为 FULL40/S8；每个形状和实际分区都通过自身独立检查。

运行前后原环境锁检查通过，65 个准备文件、59 个 CPU 输入和 61 个运行输入哈希全部匹配，源快照内容匹配，子进程已清理且四卡空闲。此实验仅证明上述局部数值和协议范围，**没有测 cost、完整 Goodput、生产双 home overlap 或全负载 AR/cross-M 等价**；重复源是受控数值 fixture，不是收益样本。后续 V63 仅改变 target 窗口的预测口径，B5 策略保持原生产版本，完成真实证据绑定和最终检查后才运行完整性能对照。审计：`budget-calibration-20261015/budget-diagnostics/ROOT_V64_ACTUAL_NUMERICAL_PROTOCOL_SOURCE_CONFIRMATION.json`。


补充范围说明：V64 三组真实 observed 父裁决均没有自然晋升；`valid_promotion` 通过受控父裁决验证晋升后的真实孩子验证、提交和修复。因此该结果不证明生产中会有足够的自然晋升。V63 已启动完整 463 请求实验，实际 Goodput 和生产计数仍待正式测量。


### V59 三类经验 g(B) 曲线（描述性先验）

已按同一请求先合并三个上下文段，再画每类 8 条请求均值细线和类别均值粗线，共 24 个请求、360 个原始 context-budget 标签：

- [tight / coding](../budget-calibration-20261015/budget-diagnostics/v59-request-mean-curves-20261015/tight-coding.svg)
- [normal / chat](../budget-calibration-20261015/budget-diagnostics/v59-request-mean-curves-20261015/normal-chat.svg)
- [loose / summarization](../budget-calibration-20261015/budget-diagnostics/v59-request-mean-curves-20261015/loose-summarization.svg)

图、CSV 和生成脚本均保存于私有诊断目录，总计约 68 KiB。类别曲线是已有样本的描述性先验，不能直接替代逐请求 g(i,n)，不能外推到 conditional eager B3 或 Goodput。图中保留 V59 最终返回失败、旧 pilot 已见部分 heldout 样本及 B8 跨布局 token 冲突的实际限制。

### 2026-10-16：V63 完整窗口预测对照完成，尚无明显性能收益

V63 在原 B≤5 策略上，仅将 target 调度窗口预测从提交后的剩余等待改为整个 target worker 的墙钟时间；它不是设备计算时间，也不能证明这段时间全部可重叠。463 请求、泊松到达、原 SLO、MC2、模型分卡和预算上限均保持原对照。

| 方法 | Goodput（token/s） | Raw（token/s） | 全量耗时（s） | SLO 达标 |
|---|---:|---:|---:|---:|
| 固定串行 γ=4 参照 | 501.64 | 921.66 | 128.60 | 252/463 |
| V46 原树形 B≤5 | 259.00 | 608.71 | 194.72 | 197/463 |
| V63 整段 target worker 窗口预测 | 259.36 | 609.55 | 194.45 | 197/463 |

V63/V46 为 **1.00137×**，V63/固定串行 γ=4 为 **0.51702×**；单次约 0.14% 的差异不足以证明稳定提升。两版达标构成相同：tight 8/271、normal 93/96、loose 96/96；2379 次 target 周期及主要图形状分布也相同。完整启动 44 target / 201 exact-prefill / 66 draft 检查通过，两条连续完整 zero trace 后原生封存，正式 84 项禁止捕获、校验、回退计数全部存在且为整数 0。源、快照和环境锁匹配，所有自有进程已退出，四卡空闲。

实际仅执行 **1 个 eager 请求**，晋升 1、失效 0；两次 bootstrap offer 不能解释成执行了两次。唯一 eager 观测的附加开销 EMA 为 **17.645 ms**，正常 16 行桶最后 EMA 为 **40.883 ms**。当前路径混入 eager 时会从 normal 的 packed PA＋三步尾链，切到 repair PA＋frontier PA＋四步尾链。数据说明仅扩大预测窗口并未带来足够 eager 工作；没有逐周期 admission 记录，不能将其归因为唯一一个已证实原因。生产没有独立的“晋升孩子已验证/提交 token”计数，不能由晋升次数推导该计数。

独立在线 prefill 累计 **53.148 s / 418 批 / 463 请求**，其中 377 批只有一个请求。该项单独记录，不能把整个性能差距都归给窗口预测。下一候选 V65 尝试 packed PA＋三步尾链、正常 B≤5 / eager 恰 B3；先做自身 matched-M 数值、父 B5→孩子 CAUSAL B3 验证/提交/修复检查，再决定是否进入完整性能实验。三类经验 g(B) 不视为 conditional eager B3 的已校准收益。

证据：[V63 全量摘要](../budget-calibration-20261015/tree-worker-window-qualified-B5-v63/full-r1-summary.json)、[正式图与源审计](../budget-calibration-20261015/budget-diagnostics/V63_FORMAL_NATIVE_ZERO_SEAL_SOURCE_AUDIT.json)、[逐请求指标复算](../budget-calibration-20261015/budget-diagnostics/ROOT_V63_ACTUAL_RESULT_CHECK.json)、[窗口与耗时审计](../budget-calibration-20261015/budget-diagnostics/V63_ACTUAL_WINDOW_TELEMETRY_AUDIT.json)。全负载 AR 与跨布局精确 token 等价仍未取得资格。

### 2026-10-16：V65 四段混合草稿资格通过，准备 V66 全量对照

V65 将正常 B5 与 eager B3 的首轮 repair/root/parent-leaf 一起做 packed PA，再执行三步尾链。正常请求保留完整四步 primary；eager 只生成三步 primary，原八槽接口中的第 4、8 槽保持不可发布的零占位。正常候选池置信度按 8 项平均，eager 按实际计算的 6 项平均；没有把尚未计算的第四步 token 或 KV 算成有效工作。

统一 CPU **22/22** 通过，随后四 rank 均实际完成 **27 项数值 + 12 项协议检查**。数值覆盖正常/eager 15+1、16+1、15+2，前缀 127/128/255 的原输入、变更输入和恢复输入；全部物理行 hidden、精确 token/置信度、draft 全部实际 28 层查询 KV、ledger 和有效前缀检查通过。父批仍为真实 B5 FULL96/S16 或 FULL116/S22；B3 孩子在自己的 **CAUSAL40/S8** 分区下完成 target 全部 64 层 KV、真实裁决、原提交和下一轮修复检查，没有借用旧 FULL 孩子的资格。

三组 observed 父裁决均未自然晋升；有效晋升由明确标注的局部受控父裁决覆盖，分别晋升 owner 15、16、15+16，随后孩子全部使用真实 target 结果消费。它证明该局部协议路径可用，不能据此估计生产晋升率或 conditional eager B3 的 g。实验没有成本、封存性能或 Goodput 结果。71 个冻结输入、70 个 CPU 输入、72 个运行输入和快照全部匹配，108 份 target MC2 原始记录无 fallback/exception；前后环境锁通过，进程已清理且四卡空闲。独立审计见 [V65 原始门与清理审计](../budget-calibration-20261015/budget-diagnostics/V65_ACTUAL_BOUNDED39_NATIVE_MC2_SOURCE_CLEANUP_AUDIT.json)。

V66 在线候选已完成 **16 项 CPU 检查**，绑定 206 个 V65 实际证据文件并冻结源码。它在创建 ticket 前落实 normal B≤5、eager 只取 0/3；只对上述三种组合、正常请求各缺 1 个 token、父 B5 等条件满足的工作启用新路径，其余可选 eager 不准入，已发布父草稿保留。正常 g 的在线更新与 conditional eager 观测分开；成功提交后另记孩子实际消费数和提交 token 数。首版保留 V65 的诊断 clone/hook，其开销计入完整计时。

V66 仍须重新通过原 44 target / 201 prefill / 66 draft 启动检查、连续两条完整 zero trace、原生封存和正式 84 项零增量检查，才能报告完整负载性能。当前最新有效全量树形仍为 **259.36 token/s**，固定串行 γ=4 对照为 **501.64 token/s**，尚未达到目标。


### 2026-10-16：V66 全量完成，实际未准入 eager B3

| 方法 | Goodput（token/s） | Raw（token/s） | 全量耗时（s） | SLO 达标 |
|---|---:|---:|---:|---:|
| 固定串行 γ=4 参照 | 501.64 | 921.66 | 128.60 | 252/463 |
| V63 树形 B≤5 | 259.36 | 609.55 | 194.45 | 197/463 |
| V66 有界 normal B≤5 / eager B3 | 254.62 | 601.48 | 197.06 | 196/463 |

V66/固定串行 γ=4 为 **0.50758×**，V66/V63 为 **0.98175×**，没有达到优化目标。SLO 达标构成为 tight **5/271**、normal **95/96**、loose **96/96**。沿用同一 463 请求、泊松到达、输出长度、SLO、MC2 和分卡；方法各自的图缓存容量上限不同，串行参照为 80、树形为 512，因此不称所有配置完全相同。

正式轮有 158 次窗口 bootstrap offer，实际计划 64 个 B3 eager 请求；准入 **0**，其中 53 次普通/eager 请求数组合不在已验证范围、11 次正常请求 repair 条件不符。晋升、孩子实际消费和提交 token 数均为 0；conditional eager g **没有样本**，不能记成 g=0，也不能用这轮结果判断 V65 新四段计算路径本身的成本或收益。每次实际计划只有一个 eager；现有记录未保存相应普通请求数，无法精确还原被拒绝的组合。

独立在线 prefill 为 **53.395 s / 422 批 / 463 请求**。新增准入协调累计耗时在 draft 为 **2.278 s / 64 次**（约 35.60 ms/次）、target leader 为 **0.273 s**（约 4.26 ms/次）。这说明主机到达时间存在差异，不能直接解释成相同长度的 NPU overlap 损失。源码顺序确认为 target 各 rank 先返回提交函数、再进入准入 Gloo、draft 才开始模型调用；被拒绝的 eager 也支付该等待。

原 201 prefill / 44 target / 66 draft 启动检查全部通过，G2−G1 与 G3−G2 为连续两条完整零发现轨迹，随后 G4 原生封存计时；正式 84 项禁止捕获/校验/回退字段齐全且全部为整数 0。124 个实际运行输入、123 个冻结输入、107 个 CPU 输入、206 个实际资格绑定和快照字节均匹配，环境锁前后通过，自有进程全部退出，四卡空闲。早期 r1 在模型启动前因缺少三个空归档目录失败；记录保留，仅补回原空目录且未修改冻结源码，r2 才产生上述结果。

后继 V67 保持当前数值范围、预算和验证门槛，新增有上限的普通/eager/拒绝原因及缺失 token 数统计，并将原 early-target 提交条件移动到 CPU 规划/准入之后、ticket 创建之前。它会改变 target 启动时刻及窗口观测区间，净收益必须实测；暂不扩大未经验证的组合，也不将类别 g(B) 曲线外推成 conditional eager B3 收益。

证据：[V66 完整摘要](../budget-calibration-20261015/tree-packed-eager3-bounded-online-v66/full-r2-summary.json)、[正式原生门与源码审计](../budget-calibration-20261015/budget-diagnostics/V66_FORMAL_NATIVE_ZERO_SEAL_SOURCE_AUDIT.json)、[30 项指标与对照复核](../budget-calibration-20261015/budget-diagnostics/ROOT_V66_ACTUAL_RESULT_CHECK.json)、[实际准入和耗时](../budget-calibration-20261015/budget-diagnostics/V66_BOUND_EAGER3_ONLINE_ACTUAL_TELEMETRY_AUDIT.json)。完整负载 AR/cross-M 精确 token 等价仍未取得资格。


### 2026-10-16：V67 明确拒绝组合，整体后移 target 无性能收益

| 方法 | Goodput（token/s） | Raw（token/s） | 全量耗时（s） | SLO 达标 |
|---|---:|---:|---:|---:|
| 固定串行 γ=4 | 501.64 | 921.66 | 128.60 | 252/463 |
| V63 | 259.36 | 609.55 | 194.45 | 197/463 |
| V66 | 254.62 | 601.48 | 197.06 | 196/463 |
| V67 新增有界统计并整体后移 target 提交 | 232.66 | 579.14 | 204.66 | 186/463 |

V67 的 tight/normal/loose 达标数分别为 **2/271、88/96、96/96**，仍未达到目标。正式轮 185 次 shaper 前非空 eager offer，分配后实际检查 95 个 B3 eager，全部被拒：92 次组合不支持、3 次正常 repair 条件不符；晋升和真实消费均为 0，conditional eager g 仍无样本。offer、分配后计划和实际计算分开计数，没有把被拒工作当收益。

新统计全部未溢出、无未知 ledger 或查找错误。95 次检查中的普通行共出现 repair0 **451 行**、repair1 **456 行**，没有 repair2..4；正式最频的具体计数组合为 N13/E1、7 行 repair0＋6 行 repair1，共 7 次。第一轮预热较常见的是 N14/E1 和 N9/E1，说明同负载回放中的窗口选择随时序变化；这些只作受控资格范围的选择依据。记录没有逐行预算和 owner 顺序，不能声称复现了真实全 B5 的逐行输入。

准入 Gloo 的 draft 侧累计等待降到 **0.0555 s / 95 次**，但总耗时比 V66 增加 **7.602 s**。把整个提交条件后移会影响没有 eager 的周期，可能损失 CPU 规划与 target 的重叠；这是有源码依据的候选原因，尚无单变量消融证明。后继生产方向是：window offer 为空时恢复 shaper 前早启 target；offer 非空时保持准入协调后提交。该改动尚未计入上述结果。

最终 CPU **21/21** 通过，109 个本地源和 3 个外部参照均匹配；首次测试有一项损坏 ledger fixture 不成立的失败，日志、收据和源前像已保留，修正仅发生于未冻结测试代码。随后 128 个冻结文件、129 个实际运行输入、206 个实际资格绑定及全部归档字节复核一致。原启动 201/44/66、两条完整零发现轨迹、原生 seal 和正式 84 项整数零检查全部通过，环境锁前后通过、四卡空闲。

下一局部资格 V68 计划保留原 V65 core 字节，使用 draft 内部规范排列：repair1 正常行在前，repair0 在后，eager 最后，再将所有行相关输出恢复为调用顺序。受控新范围为 N9/E1 的 repair1 数 3、5，以及 N13/E1 的 repair1 数 6；需要独立的真实源、各 owner 不同前缀、same-M hidden/精确 token/置信度、draft28层 KV、target64层 KV、原父子提交与下一轮修复验证。CPU 排列证明不替代新 NPU 数值和协议记录，当前没有 V68 结果。

证据：[V67 完整摘要](../budget-calibration-20261015/tree-eager3-admission-observation-v67/full-r1-summary.json)、[完整原生门与源码审计](../budget-calibration-20261015/budget-diagnostics/V67_FORMAL_NATIVE_ZERO_SEAL_SOURCE_AUDIT.json)、[30 项指标复算](../budget-calibration-20261015/budget-diagnostics/ROOT_V67_ACTUAL_RESULT_CHECK.json)、[正式准入分布](../budget-calibration-20261015/budget-diagnostics/V67_FORMAL_ADMISSION_OBSERVATION_AUDIT.json)。全负载 AR/cross-M 精确 token 等价尚未取得资格。


### 2026-10-16：V68 异长前缀资格在长度 255 被原 allhidden 门拒绝

V68 最终 CPU **25/25** 通过；实际 NPU 测试在四个 rank 各完成 **18/27** 条数值记录后失败，父子协议为 **0/12**，没有成本或 Goodput 结果。已通过记录覆盖 prefix anchor 127、128 下三个受控组合及 changed=false/true/false，draft 为真实 28 层；这不构成整个新范围已取得资格。

首个本地错误来自 draft rank0：prefix anchor **255**、N9/E1、正常 repair1 数 **3**、changed=false，原 `Fresh same-M canonical full-physical allhidden gate failed` 拒绝。该门合并检查首步完整物理 hidden、首步真实行 hidden 和后三步完整 hidden，失败文件未保存分项统计；现有证据不能区分具体步骤、行或误差原因，也不能直接归因为低精度、PAD、页边界或 tensor 生命周期。其余 rank 通过原协调门接收到失败，不称各自发生独立数值失配。

81 个冻结文件、76 个本地＋5 个外部 CPU 输入、206 个 V65 实际绑定以及 82 个运行输入/归档逐字节一致。child exit1，环境锁前后为0，源码未变，自有进程已退出，四卡空闲。失败版本保持原样；诊断后继 V69 仅细化原门失败的分项与有界标量日志，原 rtol/atol、exact token/confidence、全层 KV、生命周期和资源门槛保持。通过诊断与后续资格后才考虑生产整合。

证据：[V68 失败与完整源码/归档审计](../budget-calibration-20261015/budget-diagnostics/V68_ACTUAL_ALLHIDDEN_FAILURE_SOURCE_CLEANUP_AUDIT.json)、[原始失败收据](../budget-calibration-20261015/tree-canonical-binary-repair-eager3-v68/cost-receipt-packed-eager3-canonical-binary-qualification-v1.json)。当前最佳有效全量树形 Goodput 仍为 V63 **259.36 token/s**，同卡固定串行 γ=4 为 **501.64 token/s**，目标尚未达到。


### 2026-10-16：V69 定位 repair1 第二查询差异，仍保留原拒绝

V69 最终 CPU **34/34** 通过，运行中仍在 prefix255/N9/E1/R3/changed=false 被原门拒绝；四 rank 各 **18/27** 数值记录、**0/12** 父子协议，没有新 Goodput。诊断仅发生在原门已判失败之后，原三比较对象、rtol=atol=1e-3 和原异常均保持。

| 对比 | 原门结果 | 最大绝对差 | RMS | 超出原相对＋绝对容差的元素数 |
|---|---:|---:|---:|---:|
| 首步完整物理 hidden 与逐步参考 | 失败 | 0.5 | 0.016637 | 2585 |
| 首步真实行 hidden 与逐步参考 | 失败 | 0.5 | 0.018457 | 2585 |
| actual graph 首步与 fresh 同 packed eager | 通过且逐值一致 | 0 | 0 | 0 |

首步差异仅在 repair1 的第二查询：owner5@250/context251、owner3@252/context253、owner1@254/context255；这三行分别有 881、879、825 个超容差元素，其余首步行和 PAD 一致。即刻 observer 与 core 保留的首步 hidden 也逐值一致，所有比较值有限。tail0/1/2 的差异随后在这三 owner 传播。公开候选 token、first token 和每一步 tail 输入 token 都精确相同，但 seed 编码有 5 项、tail 编码有 17 项置信度不同，故不能因为 token 一样就宣布整个资格通过。

第三对照从真实 token 独立重建缓存，原 FastV5 重新生成父历史，之后按相同 packed 布局执行原 eager model；它与 actual graph 全 16 行和 13 个真实行逐值一致。这支持进一步检查逐步参考的 repair-root 执行几何。该第三对照的父历史仍用 Native FastV5，而旧参考父历史用 eager chain，不能单凭它把全部差异归因某个内部 tiling；后三个 normal repair owner 的差异及下一步独立 parent 控制将继续用于隔离原因。

90 个冻结源、80 本地＋88 外部 CPU 输入、206 实际 V65 绑定、91 运行及归档文件全部一致。child exit1、前后环境锁0、源码未变、自有进程退出、四卡空闲。后继 V70 仅针对该失败 case，独立 fresh source/eager parent 后比较：原逐步 root、增加真实只读 eager leaf 的 root、保持全部真实行位置但其他行只读的 root，以及移除 eager 后的 normal-only packed。所有控制都保留原失败，不充作性能或完整资格结果。

预算性能侧另有静态方案：服务 B64 可使 N16 的真实分支混合落入现有 FULL80/S16，而全 B4 会走 CAUSAL128/S38；B72 为 FULL88/S16，B80 与 B96 在 N16/B≤5 下都最多实际使用80候选。服务预算应与原硬件 Q128、candidate96 和 44 图启动域分开，原 need/优先级/g 曲线及约53秒独立 prefill 仍需单独评估，以上不是实测提速。

证据：[V69 原始诊断](../budget-calibration-20261015/tree-canonical-binary-repair-diagnostic-v69/context-cost-sweep-v1/packed-eager3-canonical-binary-diagnostic-v1/rank0-hidden-failure-p255-N9-R3-changed0.json)、[完整源码/归档/清理审计](../budget-calibration-20261015/budget-diagnostics/V69_ACTUAL_HIDDEN_FAILURE_SOURCE_ARCHIVE_AUDIT.json)、[预算物理映射方案](../budget-calibration-20261015/budget-diagnostics/GLOBAL_B64_B72_B80_B96_PHYSICAL_POLICY_READONLY_PLAN.json)。


### 2026-10-16：V70 单 case 确认上下文组合敏感性，准备五阶段修复

V70 最终 CPU **43/43** 通过；本轮只对 prefix255/N9/E1/R3/changed=false 做一次原数值尝试和四个失败后控制，保留 case 内 F/T/F 原生预热。原 firstphysical/firstreal/tail 三门仍失败，数值通过记录 **0**、协议 **0**。这一轮是独立冷启动单样本，不能当作原27＋12资格或四个独立rank样本。

| 控制 | 最大 context | repair root hidden / 全28层 root KV 的有限值对照 |
|---|---:|---|
| C0：独立原逐步补算 | 255 | 与逐步 reference 相同；与 actual 不符 |
| C1：C0 加真实只读 E leaf | 261 | 与 actual 相同；与 reference 不符 |
| C2：完整 actual 行描述，仅 root 可写 | 261 | 与 actual 相同；与 reference 不符 |
| C3：normal-only packed，不读 E leaf | 256 | 与 actual 相同；与 reference 不符 |

四个控制分别从真实 token 独立重建、用 `old.chain` eager parent，未借 candidate/reference KV 构造历史。C0 对 reference 的 root hidden 和56个 K/V数组最大差均0；C1/C2/C3对 actual 也最大差0。两组之间 root hidden 最大差 **0.5**、2585 个超容差元素；root KV 有 **124312** 个超容差元素，首个超门层为 layer1 key。所有输出有限，committed prefix 哈希保护通过；56个只读 KV数组使用 uint8 真正字节比较，四控制均通过。C3 明确未物化、未读取 E leaf，只保护真实 parent4slots。

因此仅移开 eager 不足以恢复原 reference；normal-only 已含 context256，现象支持批次上下文/描述组合会影响结果。尚未证明具体 CANN 内核选择或唯一底层根因，不能把原 gate 改为通过。输出最大差0在此称有限数值逐值相同，未宣称输出 raw bits 已比较。

98冻结文件、84本地＋97外部CPU输入、206实际底层绑定和99运行/归档文件全部匹配；child1、环境锁前后0、源码未变、所有自有进程退出、四卡空闲。新候选 V72 将正常 missing-token 与当前 root 保留为两次原 M16/M20 PA，再接既有 outside head＋tail3，共五模型阶段。它会增加首段计算，必须先重新完成27数值＋12父子协议，并逐次检查完整物理 hidden/PAD及组装结果、exact confidence、全层KV和下一轮修复；后续成本与Goodput仍待测。

证据：[V70 四控制原始记录](../budget-calibration-20261015/tree-repair-root-context-control-v70/context-cost-sweep-v1/packed-eager3-repair-root-context-control-v1/rank0-hidden-failure-p255-N9-R3-changed0.json)、[独立完整归档审计](../budget-calibration-20261015/budget-diagnostics/V70_ACTUAL_ROOT_CONTEXT_CONTROL_SOURCE_ARCHIVE_AUDIT.json)、[控制结果复核](../budget-calibration-20261015/budget-diagnostics/V70_ACTUAL_REPAIR_CONTEXT_CONTROLS_AUDIT.json)。最佳有效全量树形 Goodput 仍为259.36，对照串行501.64 token/s，目标未达成。


### 2026-10-16：V72 两段首步方案通过限定范围数值和提交协议检查

V68–V70 保留为失败及定位记录。V72 把混合 normal repair 的首次草稿计算拆为两次有先后依赖的 PA 调用，并立即保存各次返回的隐藏状态；后续仍使用原有三步 tail graph。实际执行是 **两次首步 PA、共五个模型阶段**，不再把拼装后的隐藏状态称为某一次 Native 图的输出。

| 检查项 | 本轮实际结果 |
|---|---|
| 数值范围 | N9/E1/R3、N9/E1/R5、N13/E1/R6；前缀 127、128、255；原输入／改输入／恢复输入，每 rank 27 项全部通过 |
| 首步检查 | 54 份独立调用记录逐次通过完整物理隐藏状态、PAD、元数据和保存副本检查；四个 rank 的共享记录不重复计算为独立样本 |
| 草稿结果 | 原有精确 token、精确置信度、28 层 query KV、ledger、源前缀保护及恢复检查均通过 |
| 提交协议 | 前缀 128；真实 target 裁决、受控有效提升、父拒绝、同 token 但祖先变化，共 12 项／rank 通过；包含真实子树消费和下一轮 repair |
| Target / MC2 | 三个 target rank 各 36 份实际 MC2 记录；128 路原门、64 层 KV／压实检查通过，fallback／exception 为 0 |
| 完整性与资源 | CPU 41/41；冻结源 94、运行源／归档 95 项逐项一致；620.931 秒，私有输出约 15.8 MB；子进程退出 0，四卡空闲 |

这轮只证明上述限定范围内的正确性；**尚未证明该五阶段方案的成本、线上覆盖率、双 batch overlap 或 Goodput**，也不扩大为全工作负载的 AR／跨形状 token 一致性结论。Native target-hidden 的 `validated_real_row_count` 原值仍为 `None`，实际有效行数另记，未伪造验证字段。实际结果见 [V72 完整审计](../budget-calibration-20261015/budget-diagnostics/V72_ACTUAL_BOUNDED39_SPLIT_FIRST_SOURCE_MC2_AUDIT.json) 与 [root 复核](../budget-calibration-20261015/budget-diagnostics/ROOT_V72_ACTUAL_BOUNDED_RESULT_CHECK.json)。

接下来 V74 在同一组 16 个真实历史上比较四种预选预算：12B5+4B1（FULL80/S16）、4B5+12B4（FULL84/S16）、16B5（FULL96/S16）、16B4（原映射 CAUSAL128/S38）。预算 64 和 68 的原 allocator／shaper／硬件上限区分已分别通过 7/7 CPU 检查，但尚未运行其线上版本。V74 将保留原有 root 诊断和 tail 观测，仅把额外独立参考与完整 KV 检查放在计时外；局部绝对耗时不能直接冒充 V63 关闭诊断时的生产耗时。

**完整负载指标未更新**：当前同卡固定串行 γ4 Goodput 为 501.6356，最好完整树形结果仍为 V63 的 259.3551 token/s；动态 B 树形超过串行的目标尚未达到。


### 2026-10-16：独立 prefill 墙钟和当前提升空间的口径复核

V63 的 **53.147953 秒**来自 418 次在线 admission 内部的同步 prefill 调用，包含 target leader 的主机墙钟和必要广播。源码在这些调用期间给已有 active 请求累计 decode 等待，因此它确实是需要优化的阻塞区间；它不等于纯 target kernel 时间，也不是顶层几乎为零的初始 prefill 字段。串行的约 1.132 秒只统计其独立／idle prefill，另有 594 个 mixed target chunk；两者不能直接相减为可回收收益。

在达标请求仍为 197 个的理想假设下，即使删除 V63 全部这 53.148 秒，Goodput 也只约 **356.90 token/s**，仍低于串行 501.64。要达到串行结果，既需要减少墙钟，也需要让更多请求达标；现有日志没有每批 active IDs 和逐请求 prefill 等待累计，不能把 tight 的超时逐个归因到这 53 秒。实际定义和数字见 [prefill 时钟审计](../budget-calibration-20261015/budget-diagnostics/V63_SERIAL_PREFILL_WALL_CLOCK_AND_GAIN_BOUND_AUDIT.json)。V75 正并行准备混合 prefill 的原子提交整合源码，尚未测试或进入线上，不能计入当前性能。


### 2026-10-16：V74 四预算局部实验在裁决接口处停止，尚无成本结果

V74 统一 CPU 检查最终 **19/19** 通过（20.8248 秒，无 NPU 初始化），随后四卡实际运行 170.2705 秒，在首个 `all_B5_C80` 组的裁决通信接口停止。错误为 `Tree verifier output has an invalid token shape`：测试调用把候选预算 B=5 传给原 verifier 的 `max_depth`，生成 `[16,6]` 输出和 `[16,5]` 接受数组；真实树的 depth=4，原 transport 要求 `[16,5]` 与 `[16,4]`。这是测试集成的 ABI 错误，**不是本轮已证明的数值／精度失败**。

首组实际冷图改输入／恢复 witness 两项、三个 target rank 共六份 MC2 raw 通过，但没有任何完整资格项、固定点 trace 或成本样本完成；不能据此给出预算 64／68／80 的胜负。133 份冻结文件、134 份运行源／归档、46 份外部依赖及基线检查一致；子进程退出 1 后全部清理，四卡空闲。原失败 CPU attempt（测试误改 frozen metadata）与 NPU 失败现场分别保留。详见 [实际失败审计](../budget-calibration-20261015/budget-diagnostics/V74_ACTUAL_HOST_VERDICT_ABI_FAILURE_AUDIT.json)。

新 V76 将按真实 `max(plan.depth)` 生成裁决数组，并在原通信前让所有 rank 先完成 leader 输出形状检查；新增真实 verifier→transport 的 CPU 回归。V74 冻结源不再改动，各预算的实际候选数、物理映射、Native／MC2／数值门均保留。尚无新的 Goodput 结果。


### 2026-10-16：四种预算布局的完整局部成本对照（V76）

实际运行已完成，四个 rank 均通过 12 项资格检查、两轮覆盖全部四种方法的零发现检查，以及 20 次耗时采样（每种方法 5 次）。检查包括独立重建首轮和下一轮历史、全部 28 层 draft / 64 层 target 查询 KV、完整物理行 hidden、精确 token/confidence、源前缀保护，以及移除额外资格观察器后的完整路径恢复对照。MC2 原始记录 192 份均通过，fallback/exception 为 0；源码与归档一致，作业结束后四张卡空闲。

这里的耗时是**本地顺序执行一轮草稿、验证与提交，再实际生成下一轮草稿的主机墙钟时间**；源历史预填充和独立参考检查在计时外，原 root 诊断与常驻 tail 观察器仍保留。它不是设备 overlap 时间，也不是完整请求流的 Goodput。每种方法重复使用同一组 16 条真实历史，不能当作 5 组独立请求样本。 四个预算向量是预先设定的控制组，本轮没有运行新的在线动态预算分配器，也没有据此确定全局最优 B。

| 预算分配（16 个请求） | 候选预算 C | 真实查询 Q / 物理行 M / 序列行 S | 耗时中位数（ms） | 最小 / 最大（ms） | 实际交付 token 合计 |
|---|---:|---:|---:|---:|---:|
| 12 个 B5 ＋ 4 个 B1 | 64 | 80 / 80 / 16 | 161.83 | 161.26 / 164.75 | 52 |
| 4 个 B5 ＋ 12 个 B4 | 68 | 84 / 84 / 16 | 162.76 | 162.30 / 165.09 | 57 |
| 全部 B5 | 80 | 96 / 96 / 16 | 164.67 | 163.48 / 165.55 | 57 |
| 全部 B4 | 64 | 80 / 128 / 38 | 172.42 | 171.67 / 173.61 | 56 |

在这组历史上，B68 混合方案与全 B5 交付相同的 57 个 token，局部速度约提高 1.18%；B64 混合方案虽稍快，但少交付 5 个 token。全 B4 会进入原有 CAUSAL128 布局，额外包含 48 个 scratch 查询，预算更小却更慢。六组两两布局比较在共同有效 token 位置没有冲突，输出长度差仍单独保留；这些结果不提供跨布局通用 g 拟合或完整 AR 一致性证明。

**下一步重点：混合 prefill 的真实两轮原子提交资格。**仅缩减这部分预算的已测局部收益不足以填补当前最佳已测树形 Goodput 259.36 与同卡串行 γ4 的 501.64 之间的差距。V75 在准备将新请求 prompt 分块放入现有 target 工作，并保留原 tree verifier、共识、KV 提交与首 token 发布顺序；V77 的灵活固定形状布局仅完成 CPU 合同，均尚无新的完整 Goodput 结果。

证据：[V76 完整资格与 MC2 审计](../budget-calibration-20261015/budget-diagnostics/V76_ACTUAL_FOUR_FAMILY_QUALIFICATION_COST_AUDIT.json)、[耗时与逐请求 g](../budget-calibration-20261015/budget-diagnostics/V76_FOUR_FAMILY_COST_AND_DELIVERED_TOKENS.json)、[根代理独立 20 项核对](../budget-calibration-20261015/budget-diagnostics/ROOT_V76_ACTUAL_FOUR_FAMILY_RESULT_CHECK.json)。


### 2026-10-16：混合 prefill 集成启动核对修复中

V79 实际四卡完成原有 201 项 prefill、8 项切换检查、44 项 target 和 66 项 draft 检查后，在启动库存审计停止，尚未执行新增的五个混合阶段。错误来自审计集成：Native 的 `run_tree_hidden` 将 prefill 图保存在 `target_entries`，审计却按 `entries` 查找；此外原启动 receipt 中“保留 24 项”的字段是固定声明，这次诊断启动并未实际调用原版 draft tree 的 24 项检查。不能据此宣称图缺失、数值失败或混合路径通过。

本轮 781.28 秒，私有目录约 6.6 MB；冻结源码、外部依赖和归档逐项一致，基线环境锁前后通过，所有自有进程退出，四卡空闲。失败现场保持冻结。新 V80 将按实际 Native 映射核对完整 key 集合，并实际执行原版 24 项检查后再进入混合路径。[实际失败复核](../budget-calibration-20261015/budget-diagnostics/ROOT_V79_ACTUAL_STARTUP_AUDIT_FAILURE_CHECK.json)。当前完整树形／串行 Goodput 仍为 **259.3551／501.6356 token/s**，尚无新性能结果。


### 2026-10-16：V80 混合 prefill 完成首遍与一遍零发现，完整资格仍未结束

V80 修正启动库存映射，并实际调用原版 draft tree 的24项检查；统一主机回归131/131通过，无模型/NPU初始化。实际四卡启动库存为draft 90项（66＋24）、每张target 245项（201 prefill＋44 target），原201项prefill、8项切换及MC2检查均通过。另修正了诊断观察器直接比较CPU context长度与NPU position的问题，模型计算和原数值门槛未改。

四个rank均完成N15的101/1两阶段、N16的97/97/8三阶段，以及四个真实中止/回退控制。实际记录覆盖独立hidden、28/64层查询KV、原tree裁决与提交、游标推进、仅最终首token发布、重复发布保护及发布后N1草稿检查。原汇总审计错误地把全部边界事件当作纯ledger列表；新只读审计分别校验全部同步事件、租约、提交和ledger，原始文件和失败断言保持不变。用实际Python3.11环境对四卡首链逐项复核通过，另有11项主机审计回归通过；system Python3.12的AST哈希差异尝试单独保留，未改源码哈希断言。

随后完成一遍覆盖整条路径的重复检查：每个rank的21项真实Native发现/验证/fallback计数增量全部为0，四卡提交/发布签名与共同边界事件一致。但第二遍重复检查尚未完成时触及本轮2400秒上限，实际2404.45秒结束，未封存图库存。因此**原作业状态为失败（诊断超时），不能称完整资格通过，更不能称线上或性能通过**。候选私有目录约71.5 MB；167份冻结本地源、656份外部输入和实际源码归档逐项一致，环境锁结束检查通过，自有进程清空、四卡空闲。

V81正在独立目录准备真实请求的live descriptor、host metadata和激活/释放跟踪，后续还需实际从cursor0运行完整生成循环，保留原裁决、有限性、压实、时钟及首token发布顺序，并完成无诊断观察器的完整请求流检查。当前没有新Goodput；完整树形最好值仍为259.3551，固定串行γ4为501.6356 token/s，目标未达到。

证据：[V80实际结束及完整性复核](../budget-calibration-20261015/budget-diagnostics/ROOT_V80_ACTUAL_TIMEOUT_PARTIAL_QUALIFICATION_AUDIT.json)、[首遍只读审计](../budget-calibration-20261015/budget-diagnostics/v80-offline-report-audit-v1/FIRST_CHAIN_RESULT_RUNTIME_PY311.json)、[第一遍完整零发现复核](../budget-calibration-20261015/budget-diagnostics/ROOT_V80_ACTUAL_FIRST_ZERO_TRACE_AUDIT.json)、[线上接入方案](../budget-calibration-20261015/budget-diagnostics/V80_LIVE_METADATA_IMPLEMENTATION_PLAN_zh.md)。


### 2026-10-16：V81 实际生成循环首轮失败，V82 修复状态转换

V81 完成真实请求元数据、页租约跟踪、混合计算接入和原始回退分支的连接，统一主机检查 **206/206 通过**。新路径采用实际页归属和写入位置的结构检查，去掉每轮完整 KV 抓取；它保留原 Native 有限值投票、tree 裁决、KV 提交、旧 active 集计时和最终首 token 发布顺序。另修复了纯草稿轮次误触发回退的问题。

实际四卡实验使用原启动配置和 MC2：各 target 完成201项 prefill及44项 target 检查，draft 完成66项显式检查及原版24项检查。随后通过原始 `generate_batch` 运行一条已有请求和一条后到请求，后到 prompt 为129 token，计划从cursor0分成128＋1两段。作业在首轮的 `original-draft-chunk-preflight` 失败：`submitted()` 会将不可变的 RESERVED 记录替换成 SUBMITTED 记录，边界检查仍要求旧记录的对象地址相同，因而四个rank同时拒绝。这是接入检查错误；本轮没有完成游标提交、独立AR对照或Goodput测量。

本轮实际781.57秒，候选私人目录约7.45 MB；185份候选源及CPU/freeze归档、外部输入均保持一致，环境锁前后检查通过，自有进程清空、四卡空闲。V81失败版本永久冻结。V82在独立副本修复这次状态转换，继续严格核对请求标识、nonce、租约、页、prompt、游标及chunk，并要求共同提交点使用实际当前SUBMITTED记录；新增回归覆盖两种角色的0→128→129真实事务转换和错误状态拒绝，正在执行统一主机检查，随后重跑相同设备实验。

当前完整树形最好Goodput仍为 **259.3551**，同卡固定串行γ4为 **501.6356 token/s**。新路径尚未提供完整负载的性能收益证据。[V81实际失败及清理复核](../budget-calibration-20261015/budget-diagnostics/ROOT_V81_ACTUAL_FIRST_LIVE_LOOP_FAILURE_AUDIT.json)、[V81主机检查与源码冻结复核](../budget-calibration-20261015/budget-diagnostics/ROOT_V81_FINAL_CPU_AND_SOURCE_PIN_AUDIT.json)。


### 2026-10-16：V82 首次从cursor0完成真实混合提交与有界AR对照

V82修复不可变事务记录切换，统一主机检查 **210/210通过**。真实四卡再次完成原有启动检查，然后在原始`generate_batch`中实际执行后到129-token prompt的 **0→128→129** 两段提交，两个实际预算均为B5。四rank逻辑轨迹一致，首token只发布一次且在旧active集计时之后，没有回退或中止。两请求实际输出32＋8＝40个token，与候选缓存全部释放后、fresh allocation上的独立eager大模型自回归结果完全一致。

四个worker均保存`bounded-diagnostic-complete`原始记录。原作业最后因主机汇总脚本把外层worker记录交给内层probe审计，出现`target_leader_rank`缺失，监督状态仍保留为失败。根代理使用同一保留Python3.11审计器，对四份真实内层记录、所有冻结源码/外部输入、实际归档、原工作进程退出和四卡空闲重新独立核对，8项检查全部通过，接受上述N1有界结果；未修改原记录或原作业状态。

本轮781.51秒，候选私人目录约8.94 MB；原环境锁前后通过、自有进程清空。**这不是完整463请求资格或Goodput结果，也不能推广为所有N/B组合及整体AR等价。**完整树形最好Goodput仍为259.3551，同卡固定串行γ4仍为501.6356 token/s。

下一版V83将修复外层汇总适配，并使用原manifest前48个Poisson请求做未计分的实际批次/预算/上下文覆盖轨迹，保持原到达间隔、SLO与预算策略，选取后续混合路径的实际热点分区。[V82四rank原始数据独立复核](../budget-calibration-20261015/budget-diagnostics/ROOT_V82_ACTUAL_N1_RAW_RECORD_AUDIT.json)、[后续性能实验顺序](../budget-calibration-20261015/budget-diagnostics/V82_NEXT_PERFORMANCE_STEPS_zh.md)。


### 2026-10-16：V83 覆盖诊断在请求执行前停止，V84 修复计数读取

V83修复外层worker汇总适配，并完成原48条Poisson请求的只读批次记录器；完整主机回归 **232/232通过**，未初始化模型/NPU。记录器保留原target函数的路径判断、调用参数和返回值，在实际原decode入口重建新到达原点，不安装新的mixed/live路径。

真实四卡完成模型及原启动准备后，在`Poisson-coverage-policy-before`阶段停止，**尚未执行48条请求**。新增检查要求`graph_metrics`已经包含预算计数，但draft上的原Python `Counter`在首次target调用前尚未出现对应条目；该Counter的缺省零语义与Native必须显式存在的图库存计数不同。这是诊断读取错误，不能解释成模型或预算分配失败。V84将在独立目录读取实际原policy Counter并交叉核对已发布指标，保持原Native计数和数值检查不变。

原作业保留failed：实际751.64秒，候选私人目录约8.05 MB；192份本地源和1208份外部输入保持一致，原环境锁前后通过，自有进程清空、四卡空闲。未产生新的覆盖统计或Goodput，完整树形最好值仍为 **259.3551**，固定串行γ4为 **501.6356 token/s**。[实际失败及清理复核](../budget-calibration-20261015/budget-diagnostics/ROOT_V83_ACTUAL_POLICY_COUNTER_FAILURE_AUDIT.json)。


### 2026-10-16：V84 原48条Poisson覆盖完成，V85准备混合路径输出对照

V84改为读取实际原`TreeBudgetPolicy.counts`，验证原指标函数和target记账函数共用同一policy；已发布指标仍须逐项一致，Native显式计数要求不变。实际完整CPU回归 **235/235通过**。四卡作业正常结束，原始worker记录及冻结审计全部通过；完成原manifest前48请求（tight29、normal10、loose9），每个256 token，共 **12,288 token**。真实decode入口重新生成leader一致的时间原点，保留原Poisson偏移、prompt、SLO和预算策略。

| 原路径覆盖项 | 实际结果 |
|---|---:|
| target验证次数 | 331 |
| 16请求批次 | 125次，37.76% |
| 16请求且全部B5 | 83次，25.08% |
| 大于16请求批次 | 本轮0次 |
| 不同N/B向量 | 65种 |
| B5候选提案 | 3272/3686，88.77% |
| 实际/物理查询量 | 21608/23148 |
| padding查询 | 1540，物理查询量的6.65% |
| 独立prefill批次 | 30，其中18次单请求 |
| prompt查询总量 | 20958（每请求平均436.625） |

本轮采用原V63树形路径，**没有安装新mixed/live路径，也没有测量新Goodput或进行独立AR对照**。最大16仅描述这48条实际轨迹；不能据此证明完整463负载都不会超过16。最常见的混合数值验证目标是16请求、全部B5，其次为9请求、全部B5。作业总811.85秒包含原模型/图库存准备，不能当成生成耗时或用于换算吞吐。候选私人目录约11.74 MB，194份冻结源和1408份CPU外部输入逐项一致，原环境锁前后通过、自有进程退出、四卡空闲。[原始覆盖与完整性审计](../budget-calibration-20261015/budget-diagnostics/ROOT_V84_ACTUAL_POISSON48_AUDIT.json)。

V85将保护前缀的逐token物理slot集合改为物理page上的保护offset上界，保留原租约、别名、容量、事务和写入顺序检查。真实CPU descriptor与冻结旧函数的完整`writes` tuple对照已通过；测试校验值多录一位的失败及源码归档保留，修正后对照通过。单独的合成主机微测中，16×2301-token前缀的保护结构构建中位时间为旧 **9.000ms**、新 **0.232ms**；这只覆盖该数据结构的构建，不代表完整footprint、NPU或Goodput收益。[主机微测原始数据](../budget-calibration-20261015/budget-diagnostics/V85_PREFIX_PROTECTION_HOST_MICRO_ATTEMPT1.json)。

V85已完成专项CPU检查17/17、最终完整CPU检查249/249（29模块），未初始化模型/NPU；198份候选源、1610份外部输入及实际源码归档独立核对通过，随后永久冻结。真实四卡随后完成原48请求的候选生成，共12,288 token；真实记录包含27次混合chunk提交（N16有11次）、4次混合首token发布，以及4次fallback-request→abort→fallback-reset。新混合路径从真实cursor0开始，没有采用fixture前缀；其他不支持的组合保留原始回退。以上属于诊断轨迹，尚不能视为全部输出正确性通过。

候选缓存完全释放、原类和实例绑定恢复后，使用fresh allocation和独立自身预测的eager AR进行对照。已完成前两个请求，三个target rank的独立参考结果逐token一致，但均与候选存在分歧：

| 原请求索引 | 独立参考完成token | 首次分歧（从1计数） | 全256 token中的不同数量 |
|---|---:|---:|---:|
| 0 | 256 | 251 | 4 |
| 1 | 256 | 35 | 180 |

第0请求的首个分歧已定位到普通target验证轮（cycle156，N16、M92/S16、owner0的B5），是当轮第5个输出；混合plan最后在cycle146。第1请求的首个分歧则位于cycle21的混合验证根节点：N15且全部B5，模型M197/S19、实际LM head行数90，配合owner19的prompt游标1058→1159。两案都不能仅凭不同token断言是BF16近分；前者还需区分早先KV状态与正常图数值路径，后者还需核对混合attention及对应head形状。

在确认分歧后，根代理对监督进程发送SIGINT，让其执行原finally清理，避免继续耗费完整48×256逐步对照。**原作业状态保留failed/KeyboardInterrupt**；实际997.54秒，私人候选目录约16.83 MB，源码和外部输入保持一致，原环境锁前后通过，自有进程清空、四卡空闲。不是完整48AR资格、线上资格或新Goodput结果。下一轮先针对请求0/1的首次分歧做同前缀、实际logits/hidden/KV及图/eager/物理M的受控对照，再继续性能优化。当前完整树形／固定串行γ4 Goodput仍为 **259.3551／501.6356 token/s**，目标未达到。


### V86：两个首次分歧的参考侧分数诊断（进行中）

已从真实记录导出两个案例，并用既有公共区域的tokenizer重新核对原48 prompt的长度与哈希，全部一致。原始缺失的完整draft token行、parents以及case0 proposal IDs均明确标为缺失，未制造重放数据。案例文件与原始失败证据均保留私人空间。

V86保留原生模型、MC2、NZ配置及`enforce_eager=False`，直接调用模型进行独立自回归，不安装额外图库存构造包装。只生成请求0的前251个、请求1的前35个输出；先逐token检查是否重现V85已保存的独立AR，再记录末步真实top8、指定两个token的分数/差值，以及同一hidden下full-logits与生产greedy head的对照。CPU合同检查10/10通过，4份新源和1813份外部输入已冻结；真实四卡诊断已正常完成，205.08秒、私人目录约1.48 MB，原环境锁前后通过、自有进程清空、四卡空闲。

这一步仅提供参考侧数据；candidate实际logits和KV仍未捕获。即使参考侧分差较小，也不能据此证明所有差异都是舍入导致，更不能把CPU通过或分数观察当成完整树形正确性及Goodput收益。

两个参考序列均逐token重现，三个target rank结果一致；同hidden下full-logits和生产greedy head也一致。真实参考侧末步分数如下：

| 请求 | 参考token及分数 | 原候选token在参考hidden上的分数 | 分差 |
|---|---|---|---:|
| 0 | 8365：34.50 | 7196：34.25 | 0.25 |
| 1 | 7629：41.75 | 525：41.50 | 0.25 |

两个候选均是参考侧第二名，但这不是candidate实际logits。下一步V87固定同一个参考hidden，将LM head实际投影行数控制为1、90、92、128，分别保存所有行的argmax及两个指定token分数，检查单独改变投影形状是否足以翻转。case0缺失原树parents，不能凭空指定其实际frontier行；case1的mixed root对应head第0行。该控制仍不等于候选KV或完整模型重演。[V86实际独立审计](../budget-calibration-20261015/budget-diagnostics/ROOT_V86_ACTUAL_REFERENCE_AR_MARGIN_AUDIT.json)。


### V87：固定参考hidden的LM head行数对照完成

13/13 CPU检查通过，5份新源及1835份外部输入冻结。实际四卡运行205.14秒正常结束，原环境锁前后通过、自有进程清空、四卡空闲；私人候选目录约2.20 MB。独立HOST审计核对全部真实输入、CPU源码归档、V86父证据及14份输出，全部通过。

| LM head实际行数 | 请求0所有行argmax | 请求1所有行argmax | 两例指定token分差 |
|---|---|---|---|
| 1 | 8365 | 7629 | 各0.25 |
| 90 | 8365 | 7629 | 各0.25 |
| 92 | 8365 | 7629 | 各0.25 |
| 128 | 8365 | 7629 | 各0.25 |

每例311行均实际验证重复输入hidden相同，full-logits和生产greedy一致，分数与M1 base相同；两个原参考序列再次逐token重现，三target报告一致。**在固定参考hidden的这项控制中，仅改变LM head行数没有重现任何翻转。**这不证明实际candidate hidden或KV正确，也没有确定V85差异根因。

下一步应捕获候选实际进入head的hidden、原始投影输出分数，以及同一逻辑前缀的KV，再与fresh独立AR比较；不能用参考侧小分差直接宣称全部由BF16舍入导致。当前没有新完整Goodput，树形最好259.3551仍未超过固定串行γ4的501.6356 token/s。私人空间当前约72G已用、125G可用，所有诊断留在私人区域。

证据：[V87实际独立审计](../budget-calibration-20261015/budget-diagnostics/ROOT_V87_ACTUAL_HEAD_ROW_CONTROL_AUDIT.json)、[下一轮候选hidden/KV捕获条件](../budget-calibration-20261015/budget-diagnostics/V87_NEXT_CANDIDATE_HIDDEN_KV_CAPTURE_zh.md)。


### V88/V91：候选数值捕获与 causal80 成本对照准备（2026-10-16晚）

V88最终31模块、265项CPU检查全部通过，204份本地源和1825份外部输入绑定；root独立核对实际CPU源码归档后冻结。首次启动在环境校验阶段停止，模型未启动：容器重启丢失两处旧运行库路径别名。13个保留文件逐项匹配原锁定SHA后仅恢复symlink，未安装或升级运行库；原失败记录保留。第二轮环境校验通过，正在四卡2/5/6/7原图预热，随后捕获实际candidate hidden/生产local logits/同前缀KV，并执行251+35步fresh独立自身递推AR。该轮未计分，没有新完整Goodput。

单独的V91成本候选补充causal M80/S16，并保持旧B4 causal128/S38、新B4 causal80/S16和原B5 FULL96/S16三组身份。37项HOST检查及root源码/归档审计通过，尚未启动NPU，排在V88之后；原策略和独立same-M参考不变。V90动态receipt误计入源码集合的问题已在新V91修复，旧版本保持冻结。

并行源码审计确认固定γ4与树形均有双batch流水；固定γ4生产配置允许64活跃请求，当前树形为32，树形eager成功复用极少，均需单独验证影响。容量64仅构造合法性不等于稳态KV页足够，不能提前宣称性能收益。最终目标仍是完整原463负载下超过固定γ4的501.6356，并超过较强MC2基线429.8443的1.5倍，即644.7664 token/s；正式倍率还需相同资源的真实配对。

证据：[V88最终CPU与源码审计](../budget-calibration-20261015/budget-diagnostics/ROOT_V88_FINAL_CPU_AND_SOURCE_PIN_AUDIT.json)、[V91启动前审计](../budget-calibration-20261015/budget-diagnostics/ROOT_V91_PRE_NPU_SOURCE_AND_HOST_AUDIT.json)、[树流水成本审计](../budget-calibration-20261015/budget-diagnostics/TREE_PIPELINE_COST_AUDIT_V88_zh.md)。


### V88实际抓取完成，原任务因最终日志预留失败；恢复证据已独立审计

更新上段运行状态：实际候选48×256=12288输出、三个target各251+35步fresh自身递推AR均完成，4份candidate和8份AR恢复gzip已保留。最后缩进JSON主报告的保守预留估算触发32MiB门，原任务FAILED、四份final report缺失均保留，不改写为完成。实际输出目录5,225,939B，其中.pt2,543,466B；独立审计确认完整恢复证据，唯一审计错误为缺final report，所有正式资格和Goodput标志仍为false。环境前后校验0、源码与外部输入未变、worker正常关闭、自有进程清空、四卡空闲。

| 实际案例 | 同前缀对照 | 实际分数/差异 | 证据边界 |
|---|---|---|---|
| 请求0 | 首差index85；输入1852、位置170；normal16×B5、FULL9 M96/S16、head96 | candidate11/30均35.25，AR30=35.5、11=35.25；hidden RMS差0.06413293、max1.5 | 实际首差已从旧250改变；该位置KV只有64层四通道样点与整row统计，没有完整向量或全前缀KV |
| 请求1 | 前35个输出全部相同；pinned位置298均选7629 | hidden RMS差约0.07923；完整singleposition KV存在可测微差 | rank1 layer0 value已差6.1035e-05，虽稀疏样点首差layer1；不能称全部KV首差是layer1 |

现证据确认candidate进入输出层前hidden已不同，并捕获了真正的近分/同分选择；不能再只用LMhead行数解释，也不能据此确认所有差异根因、证明全前缀KV正确，或宣称造成Goodput偏低。计算仍为原BF16，未引入量化。V91三组设备成本对照正在运行，尚无新完整463 Goodput。

证据：[V88实际恢复审计](../budget-calibration-20261015/budget-diagnostics/ROOT_V88_ACTUAL_CANDIDATE_CAPTURE_candidate-hidden-kv48-v2_AUDIT.json)、[真实HOST执行说明](../budget-calibration-20261015/budget-diagnostics/ROOT_V88_ACTUAL_CANDIDATE_CAPTURE_candidate-hidden-kv48-v2_AUDIT_HOST_EXECUTION.json)。


### V91实际成本对照完成：补图有小幅收益，尚未形成完整Goodput优势

三组均使用相同16条真实前缀、原草稿/裁决/提交/下一轮草稿协议；每组5次旋转配对同步调用。完整实际资格及原始数据独立审计通过，作业712.35秒正常结束，源码/外部输入及环境锁前后保持一致，四卡已清空。表中耗时是每次调用四rank最大值的中位数；交付token为整批16请求合计。

| 方法 | target中位耗时(ms) | 局部整轮中位耗时(ms) | 实际交付token |
|---|---:|---:|---:|
| B4，原causal M128/S38 | 58.30 | 174.01 | 56 |
| B4，新增causal M80/S16 | 48.82 | 164.41 | 56 |
| B5，原FULL9 M96/S16 | 49.05 | 166.23 | 57 |

新增M80使B4的target约快16.3%、局部整轮约快5.5%；但B5耗时相近且多交付1 token，不能据此直接把线上全部预算改成B4。局部整轮包含串联的两次草稿、参考fixture协议和同步，不是线上双batch重叠时延，不能据此推算完整463 Goodput。下一轮分别检验generic stable/staged参数准备，以及将active容量32提升到与固定γ4相同的64；当前完整树形最好值仍为259.3551，目标644.7664尚未达到。

另对旧固定γ4做同标准复核：两份真实结果的首个86-token prompt均与V88 owner0完全一致，和fresh M1自身递推AR的前251输出比较，首次分歧分别为index148（678/279）及index137（13/382）。因此固定γ4也尚未证明与单行独立AR逐token完全一致；树形与固定γ4必须使用一致的正确性标准。这项事实不能替树形通过资格，也没有确定数值差异的统一根因。

证据：[V91实际独立审计](../budget-calibration-20261015/budget-diagnostics/ROOT_V91_ACTUAL_CAUSAL80_COST_AUDIT.json)、[实际三组耗时与交付量](../budget-calibration-20261015/budget-diagnostics/V91_CAUSAL80_COST_AND_DELIVERED_TOKENS.json)。


### V97：实际两份hidden的输出层2×2控制完成

只读取公共模型的原BF16输出层权重，未加载完整32B、没有KV/分布式通信/图捕获。三个独立词表分片使用相同leader hidden作为受控输入，分别执行candidate/reference hidden × M1/M96，各3次重复。实际22.03秒正常结束，36份完整词表分片输出由独立HOST审计逐项核对，私人证据约4.52MB。

| 实际hidden来源 | M1结果 | M96结果 |
|---|---|---|
| candidate，V88 index85 | 11=30=35.25，选11 | 11=30=35.25，选11 |
| fresh AR reference，同前缀 | 30=35.50、11=35.25，选30 | 30=35.50、11=35.25，选30 |

固定hidden后改变M，并未改变本例竞争token分数或最终选择；切换hidden来源即可复现V88实际翻转。其他logits仍有小差异：candidate有70个、reference有61个M1/M96分数不同，因此不能说全部输出逐bit一致。这项直接输出层控制不重演原完整target图，也未测原另外两TP的hidden相等性；上游attention、历史KV和异步执行的差异来源仍未区分。

[实际独立审计](../budget-calibration-20261015/budget-diagnostics/ROOT_V97_ACTUAL_HEAD_CONTROL_hidden85-head2x2-v1_AUDIT.json)。没有新完整463 Goodput，性能优化继续推进V95图执行与V96并发64。


### V95：图参数staging的独立成本对照没有明显整体收益

实际三臂对照正常完成，791.60秒，私人目录约27.51MB。31项HOST、9次封存前F/T/F与1次封存后staged独立资格、封存前后各两条完整零发现轨迹，以及171份MC2原始记录、全64层target/28层draft KV、原裁决/提交/下一草稿均通过独立实际审计。三臂保持相同16条真实前缀、B4及M80/S16；generic执行如实记入generic，未借用旧target资格。

| 方法 | target中位耗时(ms) | 局部整轮中位耗时(ms) | 16请求交付token |
|---|---:|---:|---:|
| 原target80 | 48.644 | 164.498 | 56 |
| generic materialized80 | 50.682 | 167.196 | 56 |
| 封存后generic staged80 | 48.259 | 164.273 | 56 |

表为每次同步调用跨rank最大值，再取5次中位数，共15次同步实验。staged整轮中位只降低0.14%，5个配对仅3个改善，没有形成可宣称的整体收益，暂不纳入生产优化；三臂实际输出相同。计时仍是固定前缀局部整轮，不能据此换算线上Goodput。V95沿用V91普通mixedgraph flag1，不能把其启动和缓存行数直接套到保留flag0的V96完整运行。V96实际flag0下两种角色缓存逻辑行均527、tree scratch463..526，旧532只是flag1的保守上界。

[完整实际审计](../budget-calibration-20261015/budget-diagnostics/ROOT_V95_ACTUAL_STABLE_CAUSAL_COST_AUDIT.json)、[逐次成本数据](../budget-calibration-20261015/budget-diagnostics/V95_STABLE_CAUSAL_COST_AND_DELIVERED_TOKENS.json)。完整463性能目标仍未达到，继续测试并发64与大批次混合prefill。


### 2026-10-17 V96：扩大活跃请求容量的完整反例

V96 在原 463 条泊松请求、256 输出 token、40/50/150 ms SLO 下完成一次完整测量。保持 target 请求上限 32、候选预算 C=96、root-inclusive Q=128、KV 544，将活跃请求上限从 32 增至 64；实际峰值确为 64。当前结果不采用，不能称作树状方案的提升。

| 方法/变动 | Raw（token/s） | Goodput（token/s） | 总耗时（s） | tight 达标 | normal 达标 | loose 达标 |
|---|---:|---:|---:|---:|---:|---:|
| V96，活跃容量64、原S19混合路径 | 585.0044 | 73.2835 | 202.6105 | 0/271 | 3/96 | 55/96 |

完整463请求均输出256 token；正式请求流 Native 21项捕图/验证/回退计数增量均为零。实际引用KV页峰值470/544，收尾引用页归零，源码和外部输入前后不变，四进程正常退出、四卡空闲。局部same-M独立KV检查与完整工作负载审计是不同层面的证据；本轮尚未获得全请求独立AR逐token等价资格。外部独立终态审计另行记录，原始supervisor结果不改写。

原S19混合路径仍只支持N≤16，较大实际验证批次占比很高：普通target调用1477次，其中N>16有1378次；224次混合预留中止，独立prefill297批，实际refill累计64.5243秒。扩大活跃容量后吞吐没有相应增加，逐请求TPOT升高：tight均值84.25ms、normal79.83ms、loose141.62ms。不能凭单轮把全部下降归因于某一个因素；记录支持继续分别验证S35混合路径和cap3/4/5预算对照。

原始记录：`budget-calibration-20261015/tree-capacity64-runtime-v96/local-receipt-capacity64-full463-v1.json`，逐请求与token事件：`local-output/capacity64-full463-v1/rank1-full463-actual-trace.json.gz`。Goodput分母为实际generate入口至返回的主机时间，含真实请求到达与完整输出，启动与预热排除；Native内部时钟另存，两者不可混用。


V96外部复核补记：首份外部审计拒绝了旧HOST schema（把完整boundary事件表误当成仅2/3条draft物化记录）。保留原拒绝报告；后续独立HOST补充审计逐项核对完整事件、真实物化记录和原数值/事务检查，通过局部五阶段same-M资格。该补充不授全请求AR或正式性能资格。另用既有token提交事件重算墙钟首token至末token/256，达标34/463、辅助Goodput42.9593；历史Native累计clock结果为58/463、73.2835。两口径分列，不混称同一时延。

V98新增S35局部设备尝试失败：原库存完成后，首个warmup scope末尾的重复发布检查发生未定义`engine`引用，四rank均停止。缺少完整scope/chain数值证据与成本样本，不能授S35数值资格或报告提速。原失败收据、原源码、四rank错误、清理记录均保留，前后锁校验通过、四卡已释放；新V104将用`hooks.engine`修复并增加真实函数路径HOST回归，V98冻结文件不修改。

实际g的补充诊断（尚未替换模型）：V88两条coding请求中153个同上下文B5预测切片，逐token重放与实际输出一致，截断主干得到g1..5约1.693/2.229/2.608/2.882/3.124；这些不是新cap的设备执行，不能当三类held-out曲线。V96完整请求流另保留434个未参加混合验证的owner，normal调用/候选账本与Native计数相符；排除末尾输出截断后，tight类已选择预算的观测均值g3/4/5约2.893/3.071/3.192。该观测有预算选择偏差，均值之差不能直接当同上下文的预算边际收益。诊断源与统计均位于`budget-calibration-20261015/budget-diagnostics/derive_v88_same_context_gain.py`、`derive_v96_selected_gain.py`及对应JSON。

### 2026-10-17：预算与并发对照、局部失败及主机优化准备

V102 packed eager 0/1 修复局部实验保留为 **failed**：实际260.33秒，六种packed场景的18份F/T/F数值记录在四rank均已保存，但首个旧Fast对照在观察padding时失败。其真实repair请求从owner7开始，旧校验器却硬编码owner0的KV表；这项校验失败不构成模型计算错误的证据，也不构成完整资格通过。旧对照、协议检查、零发现轨迹、封存及成本样本均未完成。另有CANN诊断JSON的重复session键被冻结HOST解析器拒绝，导致原收据缺少部分退出后检查字段；原收据和原文件保持不变。后续V103启动前的真实四卡空闲证据另存，不回填V102缺失字段。实际取证见`budget-calibration-20261015/budget-diagnostics/ROOT_V102_ACTUAL_FAILED_PACKED_EAGER01_FORENSICS.json`，修复将在新V107目录验证。

V103三组原463请求对照已全部完成，卡2/5/6/7、每请求256输出、原泊松到达与40/50/150ms SLO保留。原作业2527.71秒、四rank共享一次模型和原图库存；每组分别预热、两轮零发现，统一封存后各测一轮。32项HOST及后续独立实际审计通过，三组计时均21项Native计数零增量，源锁、双运行库脚本和退出四卡空闲均通过。四rank是同一轮的观测，不能作为四次重复实验。cap3/4属于线性控制，不能用它们宣称分支树形目标已实现。

| 方法／控制配置 | 主机完整负载时间 s | Raw token/s | Native累计时钟 Goodput | 既有token事件墙钟 Goodput | Native口径tight/normal/loose达标 |
| --- | ---: | ---: | ---: | ---: | --- |
| active32 / cap3 | 262.0693 | 452.2773 | 103.5451 | 98.6609 | 1/271、9/96、96/96 |
| active32 / cap4 | 244.2568 | 485.2598 | 144.6347 | 128.9135 | 2/271、40/96、96/96 |
| active64 / cap3 | 188.5558 | 628.6096 | 134.4111 | 134.4111 | 0/271、3/96、96/96 |

三组均未采用，分支树形最好完整Goodput仍为259.3551，固定γ4为501.6356。active32/cap3的2848次普通target调用中，2674次使用causal M128/S38；N16有2248次且均使用该图，有效Q仅59–64，全程物理查询padding比例51.27%。其正常owner的实际g约2.733，target周期数和每周期耗时同时高于V63；不能从causal的CPU更新线程数推断整体更快。独立prefill的原主机阻塞时间仍有61.52秒，是后续混合S35需要解决的重要瓶颈。详细证据见`budget-calibration-20261015/budget-diagnostics/ROOT_V103_ACTUAL_ACTIVE_CAP_CONTROLS_active32-64-cap3-4-full463-v1_AUDIT.json`和`V103_ACTUAL_ACTIVE32_CAP3_COMPLETED_CASE_ONLY_ATTRIBUTION.json`。外部审计首次将draft family误当Native mapping分布，拒绝日志保留；改为从原源码导出60 draft keys及54 target keys后重新严格审计通过，原实验文件未改。

实际V96调用的离线形状分析发现：1477次普通target调用中，31请求、127有效查询行导致165行/S38图的情况有234次；理论上减少一个候选可选择128行/S32图。共有347次调用在减少不超过四个候选后有更小的合法图形状。这里只执行了原HOST形状映射，尚未测量减少候选后的token损失或性能收益，不能据此报告Goodput提高。源与原始统计见`budget-calibration-20261015/budget-diagnostics/derive_v96_physical_budget_cliffs.py`和`V96_ACTUAL_PHYSICAL_BUDGET_CLIFFS.json`。

另一个尚未安装到生产路径的HOST优化是：每次attention任务列表更新仅对相同FULL掩码ABI校验一次，仍保留原检查函数、全部设备调用和事件。固定CPU核的微测试中，M96/S16的64次原检查约1.684ms，缓存版约0.221ms；M165/S38约3.766ms降至0.296ms。真实原Native updater/checker AST加显式HOST设备桩的10项测试已通过，覆盖异常长度、掩码、页表、跨轮重验与跨线程回退；模拟中的256条设备调用/事件记录完全相同。以上是CPU证据，设备整轮耗时与完整Goodput收益仍未证明。


2026-10-16 18:08 UTC 数值定位补充：V105 在全新、完全相同的 170-token KV 前缀上，按 FULL96 graph → FULL96 eager → graph → M1 PA eager → graph 顺序重复三轮，共 15 次。三种模式的当前 root 均选 token 30；同 M 的 graph/eager 全部 root logits、hidden 与所有 64 层 query KV 逐字节相同。FULL96 与 M1 PA 的 query KV 从第 1 层起存在差异，但当前 token 相同。此实验仅说明该 fresh-prefix 反事实没有重现 V88 旧候选历史中的 root token 11；没有重建旧 KV 历史，不能据此宣称全部输出一致性已解决。完整执行、源文件及环境检查通过，外部复核继续。随后启动 V108，单独测量 FULL96 的原始逐层参数校验与一次更新内缓存校验结果的整次 target 延迟，尚无新 Goodput 结论。


2026-10-16 18:15 UTC 局部性能对照：V108 已完成（390.414 秒，原环境/源文件前后相同，四卡退出空闲）。FULL96/S16 的每个模式均通过三个 prefix 的独立相同物理形状全 hidden／64 层 query KV 与精确 token 检查，并完成两个零新增图／验证轨迹、封存后的五组成对计时。target leader 的 prefix127、128、255 整次 target 中位时延，原始→校验缓存分别为 44.6231→44.5644、44.6198→44.6947、45.0581→45.1357 ms，未见明显收益，因此 V113 不集成该改动。HOST 校验微测试提速没有转化为此设备场景的整体收益。随后启动 V110，独立测试仅 target 主机默认 CPU DeviceContext 清理的时延；draft 保留原模式，没有新完整 Goodput。


2026-10-16 18:24 UTC 局部性能：V110 已完成（340.304 秒、原环境/源前后相同、自有进程退出、四卡空闲）。仅 target 清理由原 Native 构造引入的默认 CPU DeviceContext，draft 保留原状态；三个 prefix 均完成原 Native 冷捕获／变更输入／恢复、MC2、独立同 M 全 hidden／64 层 query KV／精确 token 检查，两完整零新增轨迹及封存后五组成对计时。target leader prefix127/128/255 原始→清理后的整次 target 中位时延为 44.8401→42.1219、44.9132→42.1518、45.2304→42.5693 ms，降低约 6.06%／6.15%／5.88%。这是局部整次 target 收益，不能等同完整 Goodput 提升；V113 将在它自己的实际调用范围与 warm/zero 检查后再验证完整效果。默认CPU与未设置默认设备并非所有公开工厂 API 都等价，真实 Meta 张量反例已保留；本轮源级调用范围已单独核对。


2026-10-16 18:40 UTC 继续执行：V113 在启动前实际 outer action 检查被拒（exit2，未启动 NPU）；冻结源码／CPU／完整归档与真实失败证据全部保留。新 V116 仅修复入口、加入真实 shell 参数传递回归及新路径证据，算法和 paired 阶段逻辑保持 V113 字节不变。15 项 HOST、ROOT 独立源码与 3361 个归档成员逐字节检查、真实 launcher HOST 测试及 retained Python3.11 readiness 通过。现已启动同一进程中的首次 S35 21 场景／26 阶段资格，以及原版本和 target CPUcontext 清理＋normal 大批次 cap4 组合的各一次完整463请求测试。两臂保留同一泊松到达、模型、设备和40/50/150ms SLO；尚无新完整 Goodput，最好树形仍259.3551，固定串行仍501.6356，目标未达。


### 2026-10-16 V116 实际失败记录与继续修复

V116 在物理卡 2、5、6、7 实际运行 813.95 秒后，于正式计分前失败。三个 target rank 的 201 个 prefill 图与 44 个 target 图，以及 draft rank 的 86 个专用图和保留的 28 个 generic 图均已有实际完成记录；四 rank MC2 启动检查通过。第一组 `V101-S19-N15-progressive` 被独立 draft 参考脚本残留的 `N∈{17,32}` 限制拒绝，随后 draft 单独异常与 target 的下一共同阶段错位。没有任何 S19/S35 scope 完成，也没有产生完整 463 请求结果或新 Goodput。

冻结源码与外部输入均未改变，进程全部退出，四卡清理后均空闲。独立失败审计：`/root/SpecSLO/budget-calibration-20261015/budget-diagnostics/ROOT_V116_ACTUAL_FAILURE_AUDIT.json`，SHA256 `6657b49fc5fc3915f9165bbb4a9cf0a1d172b9244e2d2df2ee2fb3fba44459ba`。V119 将保留原测试范围和数值检查，修复请求数声明与共同阶段错误传播后重新验证。历史最佳树形 Goodput 和目标达成状态均不更新。


### 2026-10-16 V117 混合 attention 局部对照：更快但实际输出不同，未采用

物理卡 2、5、6、7，实际完整运行 285.21 秒。最小 Native 构造，只建立 M197/S19、S35 的 FULL 与 causal 共 4 个 hidden 图。每种路径自身的独立同 M 全197行 hidden、64层 query KV、精确 head/verdict、自然 changed-input 门均通过；两轮 zero、统一原生 seal 和各场景5组交替成本样本完成。计时为全部 target rank 中最慢 rank 的完整 target 调用耗时，准备/恢复/哈希在共同阶段完成后才开始计时；下面是5组样本的中位数，并非完整 Goodput。

| 场景 | FULL197（ms） | causal197（ms） | 耗时降低 | 跨路径实际交付一致性 |
|---|---:|---:|---:|---|
| N16 B4 partial | 76.38 | 72.24 | 5.41% | 1个owner不同 |
| N24 B1..4 heterogeneous | 82.71 | 76.61 | 7.37% | 1个owner不同 |
| N32 B3 partial65 | 85.88 | 79.10 | 7.90% | 2个owner不同 |
| N32 B3 final Q1 | 82.57 | 77.26 | 6.43% | 2个owner不同 |

四组均有真实 normal verifier 输出差异，不是 dummy 或中间 prompt token 差异：owner15 的实际输出第4 token 为 FULL 8298、causal 7628；N32两组另有 owner28 的第2 token 为 FULL 3910、causal 2747。接受节点索引保持相同，最终 prompt 首 token（Q1场景为151667）相同；跨路径 hidden/KV 原容差比较未通过。因此本版本没有获得替换 FULL 的数值资格，也未采用到全463生产路径。尚未证明差异根因，也未证明任一路径相对于独立 AR 的优劣；后继仅对实际差异位置做独立逐token参考诊断，原门槛保持。

源/CPU/双归档、MC2实际路由与dispatch delta、计时共同阶段顺序、prefix全系列保护及进程/卡清理均经独立原始文件审计：`/root/SpecSLO/budget-calibration-20261015/budget-diagnostics/ROOT_V117_ACTUAL_S19-S35-FULL-causal197-v1_AUDIT.json`，SHA256 `9d59299ebc005acb3769586bc3b3cf8e3e688660d3c74ce8f1739b75889ce157`。原四卡均已空闲，未生成新完整 Goodput。


### 2026-10-16 继续优化：资格汇总修复与 Top2 算子负结果

当前仍未达到动态预算树形超过固定串行 γ=4，以及较强 MC2 TP4 基线 1.5× 的目标；本节没有新的完整 Goodput。当前 NPU 作业是 V120 的普通 target 图容量对照组件。

V119 实际运行 3413.35 秒，四 rank 共保存 104 份阶段文件，对应 26 个阶段。独立 HOST 审计逐份重算了已保留阶段的数值、MC2、提交证据；最后的整体结构校验失败，错误为 `Original reservation source/capacity/epoch packet differs`。完整 chain 和生命周期汇总因原程序先校验、后保存而没有落盘，不能补造其 nonce、allocation epoch 或页租约，也不能据 104 个阶段文件追认完整资格。正式 463 请求、预热/zero trace 均未开始。源码显示生产者保留 tuple/int 键，而汇总检查器用于 JSON 读取后的 list/str 键；V129 将先保存完整 gzip，再从真实保存的字节读取并执行原校验。模型、原协议和数值检查阈值保持原样。

- V119 独立失败审计：`budget-calibration-20261015/budget-diagnostics/V119_ACTUAL_FAILURE_INDEPENDENT_AUDIT_ATTEMPT1.json`，SHA `a17c9bf423a7eb5132489e559777755c19426796a657c9342b4b887d3c9e29b9`。
- V125 的辅助诊断入口另有 shell 目录引用错误，加载模型前退出 127；原失败保留，V130 补实际入口贯通检查。它没有产生设备资格或性能结果。

V127 仅测 TP1 单算子：词表 151936、FP32 logits，比较原 `softmax + topk(2) + 整数编码` 与 CANN fused TopKSoftmax V2。实际完成 8 个自有图的 F/T/F、两遍零新增捕获及 16 个输入案例，每例 10 对交替 NPU event 计时。真实维度支持，但融合版本在 16 个案例的中位数全部更慢，不采用。

| 固定 seed random，行数 | 原算子设备中位数 ms | 融合 V2 设备中位数 ms |
|---|---:|---:|
| 1 | 0.123550 | 0.371700 |
| 16 | 0.202650 | 0.376200 |
| 32 | 0.368480 | 0.382560 |
| 64 | 0.658800 | 0.708320 |

这不包含 BF16→FP32 转换、LMHead、KV 或完整 draft。16 个案例的 Top2 token IDs 相同；8 个案例的概率 FP32 位不同，3 个 random 案例共 9 行的 1e6 整数编码不同，不能称完整 draft 等价。独立实际审计：`budget-calibration-20261015/budget-diagnostics/ROOT_V127_ACTUAL_TOP2_COMPONENT_AUDIT.json`，SHA `424b9ecffb8a8a5cec3dfc70be29308e7108b6fb76359b0e37b1c7af994a9d9b`。实际源码归档 29 个成员逐字节核对，原进程已退出、四卡空闲后才启动下一个 V120 作业。

后续重点是恢复树形 rolling-eager：旧完整 fixed γ4 的提前草稿复用计数为 10344，V63 树形为 1，四 rank 的重复计数未相加；这不是单项消融或唯一根因证明。V123 已有独立 draft/target 预算 HOST 原型，V126 正准备 parent B3/child B3 的真实协议组件，其中 N32/E1 的 retained selected 候选数为 99，target 单次仍不超过 C96/Q128/N32。192/64 工作行端点尚无实际资格。V128 则准备仅清理 draft 主机 CPU default context 的独立成本对照，尚无设备性能结论。


### 2026-10-16 target 图布局的实际组件收益

V120 在同一组四张卡上完成：16 请求、59–64 个真实 query 的六个场景，旧布局 M128/S38 改为已有的 M96/S32；各场景五组交替测量，先取每组三个 target rank 的最大耗时，再取中位数。旧耗时 56.63–57.11 ms，新耗时 51.46–51.96 ms，减少 8.64%–9.34%。两种 owner 顺序下，六个场景所有真实 query token 均一致；各自通过独立同 M 参考、全部 64 层 query KV、原 MC2、两轮零新增图与 seal 校验。跨 M 的全部 hidden/KV 未直接比较，不宣称逐位相同或全工作负载 AR 一致。

这只是 target 组件结果，没有新的完整 Goodput。完整结果和独立审计见 `budget-calibration-20261015/budget-diagnostics/V120_ACTUAL_INDEPENDENT_MAPPER_COST_AUDIT_ATTEMPT1.json`。另一个 V130 teacher PA 诊断因重复阶段的 rank 标签分歧失败，产物完整保留，未采用其性能或正确性结论；修复另建 V133。V126 B3 父草稿/后续草稿组件正在实际 NPU 资格检查，后续完整候选计划见 `budget-calibration-20261015/budget-diagnostics/NEXT_SPLIT_BUDGET_FULL_METHOD_PLAN_zh.md`。

### 2026-10-16 22:40 UTC 后续实际进展

本节更新前述“正在准备/运行”的状态，原始失败记录和已有 Goodput 表格保留。目前仍没有新的完整 463 请求 Goodput，目标尚未达成。

| 实验 | 实际结论 | 当前决定 |
|---|---|---|
| V134：parent B3 / child B3，包含 N32/E32、全局 selected 192 端点 | 4 组真实数值与协议场景、两轮 zero 和 seal 通过。受控 promotion 分支通过；实际 `observed_target` 分支自然草稿晋级为 0，因此不能据此证明在线 rolling 收益。 | B3 父子路径可进入完整实验，在线收益待测。 |
| V136：draft CPU context 清理 | 两组 normal 场景的实际 hidden、encoded 和 56 个 query KV 逐位一致。对照前后两侧原版本，候选中位耗时分别降低 6.47%、7.53%。固定前缀采用 task-update skip；不能推广为全部动态 attention 更新或完整 Goodput 的收益。 | 完整候选启用此项，并保留两角色 factory 源码范围检查。 |
| V135：N27–30 的 target 大形状映射 | 组件耗时降低 23.03%–23.29%，但四组均有一个真实 query 的 token 不同：旧布局 7628、新布局 8298。各自同 M 参考通过，跨布局替换资格未获得。 | 不采用这四个新映射；只保留 V120 已通过的六个小形状映射。 |
| V132 → V138：完整候选入口 | V132 在图清单完成后因 `dataclasses.asdict` 复制 MC2 的 `mappingproxy` 失败，没有进入完整测量。V138 使用 MC2 原生序列化接口保存真实配置；新增 8 项 HOST 通过，父版本 23 项只按 SHA 复用，未伪称重跑。当前已越过原失败点并保存首组四 rank 结构校验记录。 | V138 是当前唯一 NPU 作业，继续原结构资格、warm65、两轮 zero、seal、full463；尚无性能结论。 |

当前候选在 normal 请求数大于 8 时采用每请求 B 上限 3，小批次上限 5；draft 全局上限 192，target 每次仍受 N32/C96/Q128 限制。这个配置下 draft 总预算通常不构成稀缺约束，不能直接把结果归因于“预算不足时优先保障高优先级”；原 target admission 在 ready rows 或候选数超限时仍有真实竞争。`g` 是按请求类型和 B 建立的在线曲线，再按单个请求剩余输出剪裁，并非为每个请求单独拟合一张曲线。完整负载结束后，需要结合实际授予、晋级、作废及窗口计数判断收益。

另需区分生成候选与选中节点：当前实际启用的 `backbone_selection.py` 固定 B1–B4 取前 B 个主链节点，B5 取四个主链节点及一个侧分支。因此当前 B3 确实是三层主链，FastV5 shared3 生成的三个侧候选不会进入该 B3 验证。三主链全部接受后，下一普通轮通常仍需补写末端 primary 的 KV；eager 则提前支付 frontier 查询。应计入 repair、frontier 和 padding，不能由“四步改三步”推断 draft 必然节省 25%。当前完整候选是在高并发下主要使用 B3 线性子树、小批次允许 B5 真实分支的混合策略；真实分支执行量及其整体效果仍待完整结果确认。

实际审计依据：[V134 协议](../budget-calibration-20261015/budget-diagnostics/ROOT_V134_ACTUAL_SHORT_PARENT_PROTOCOL_AUDIT.json)、[V136 draft 对照](../budget-calibration-20261015/budget-diagnostics/ROOT_V136_ACTUAL_DRAFT_CPU_CONTEXT_AUDIT.json)、[V135 负结果](../budget-calibration-20261015/budget-diagnostics/V135_ACTUAL_INDEPENDENT_MAPPER_COST_AUDIT_ATTEMPT1.json)、[预算竞争源码核查](../budget-calibration-20261015/budget-diagnostics/V138_SLO_SCARCITY_SOURCE_REVIEW.json)、[B3 查询成本核查](../budget-calibration-20261015/budget-diagnostics/V138_B3_BONUS_KV_REPAIR_COST_ACCOUNTING_SOURCE_EVIDENCE.json)。

### 2026-10-16 23:35 UTC：V138 完整负载失败，继续修复准入边界

V138 实际运行 3863.19 秒。原 21 个 scope、26 个阶段的完整链已保存并通过原汇总检查；warm65、zero65-1、zero65-2 三条完整轨迹也已通过原审计。随后 full463 在 `original-generate` 阶段失败，四 rank 同报 `Draft row bounds exceeded before allocation`。没有完整 463 请求 artifact，也没有新的有效 Goodput。进程全部退出、四卡空闲，源代码、外部输入和运行环境检查保持通过。独立失败审计：[ROOT_V138_ACTUAL_FAILED_ROWBOUNDS_AUDIT.json](../budget-calibration-20261015/budget-diagnostics/ROOT_V138_ACTUAL_FAILED_ROWBOUNDS_AUDIT.json)，SHA `7e3672896da7819b6807dc8006e53cf85f8f2bd5298233f142ad38548d4147c9`。

实际 guard 将 normal 候选数也限制为 32，但滚动接收和完成请求后，某个逻辑 home 的候选数不保证始终不超过 32。使用真实 HOST controller 和 typed allocator 构造的 N33/E31 场景可以复现相同错误；32+32、逻辑 owner ID 为 100–163 的控制组则正常通过，排除了“逻辑 ID 不能超过 64”的解释。该 guard 在 padding 和实际 draft metadata 生成之前执行。本次实际失败的精确 N/E、round 未保存，不能把构造例子当作现场还原。[HOST 复现证据](../budget-calibration-20261015/budget-diagnostics/V138_NORMAL_ROW_BOUND_HOST_COUNTEREXAMPLE.json)。

V140 正在准备真实 normal 准入上限：实际送入 draft 的 normal 仍至多 32，超额完整请求保留等待；按现有类别优先级择取，同类中记录延后次序，避免新增上限反复跳过尾部请求。eager≤32、总工作行≤64、target N32/C96/Q128、原预算数学、KV 和数值门保持。新版本还将在既有计时外快照中保留实际分类型/分 B 的收益计数，并只在异常路径保存当轮已有的 stream、几何和 HOST 状态；不读取前一代的陈旧 worker counters 冒充本轮数据。

V139 单点布局诊断另在加载模型前因外层配置中的 `torch.dtype` 无法序列化而退出，没有新增设备资格或性能结果；原失败保留，修复另建 V141。以上失败和组件结果均不更新 Goodput 主表，当前目标仍未达到。


**2026-10-16 23:47 UTC：V140 修复版原完整流程已启动，结果待实际完成。** V138 完整负载的 normal 行上限异常已通过显式准入处理：每次至多32行 normal，超出部分按类别优先级、延期尝试年龄与 owner ID 选择，保持原请求/home/KV 映射和 target N32/C96/Q128。另在原 phase 边界留存本代实际 class/B 的 rounds/delivered 与准入记录，异常只保留部分证据而不生成完整 GP。13项新 HOST 通过；父23+8项只按 SHA 继承。ROOT 独立核验343个本地输入、4752个外部输入与5097个归档成员后启动同一原流程。四rank实际 MC2 preflight 通过，三个target实际完成201/201 prefill图，目前继续decode图/原资格检查；完整463及新Goodput尚未产生。

本轮目录为 `budget-calibration-20261015/tree-normal-admission-fix-v140`，tag `normal-admission-v140-one-full463-v1`，ROOT实际session `22405`；独立启动前审计为 `budget-diagnostics/ROOT_V140_PRE_NPU_SOURCE_AND_HOST_AUDIT.json`（SHA `06732c8d458fe2a066742b8c2d46ec1a803dd264b73dbc5e5e79220567d95ad6`）。原463条Poisson seed0/256输出/SLO40/50/150ms不变。后续先核完整原Native与既有stream墙钟两套口径，不能以warm/zero/局部算子结果代替。


## 2026-10-17 01:07 UTC：V140 完整负载结果与后续方向

V140 已完成原 463 请求、Poisson seed0、每请求 256 输出 token，实际四卡仍为 2/5/6/7。启动资格、原 21 scope/26 phase、三段 65 请求、两次零新增发现与图封存均通过；四个 worker 正常退出，所有页面归还，前后锁验证及四卡空闲检查通过。它是一轮完整候选实验，尚未进行新的同卡交错基线对照。**性能没有改善，不能作为达标结果。**

| 指标 | 原 Native TPOT /256 | 现有首末 token 事件 wall /256 |
|---|---:|---:|
| Goodput（token/s）|2.73493|2.34422|
| 达标请求 /463 |7|6|
| Raw（token/s）|180.89597|180.89597|
| 完整 generate 耗时（s）|655.22741|655.22741|

原 Native TPOT 的 tight/normal/loose 达标数为 1/1/5，事件 wall 则为 0/1/5。原 Native 各类逐请求 TPOT 中位数/P90（ms）分别为 tight 322.05/408.95、normal 248.86/328.86、loose 500.37/665.40。两种时钟继续分列，不改 SLO、分母或到达时间。

本轮 normal overflow=0，所有非空 normal 均有正预算；因此不能把退化归因于 V140 的 32 行延期逻辑。B3 的实际平均交付为 tight 2.77455（24546 次）、normal 3.13824（7697 次）、loose 2.39857（9918 次），与 warm65 接近。这些是自适应策略下的相关样本，不是随机控制实验；B1/B2 无观测，不能填 0 或宣称完整三条 g 曲线已确定。实际三类 g 的 PNG/SVG/CSV 已保存。

主要路径变化是 mixed prefill 从 warm65 的 21/318 轮升到 full463 的 799/1488 轮；其中 791 轮 M197/S35、507 轮实际 prompt 增量只有 65 token。所有 mixed 都有真实 prompt，mixed dummy 仅占物理 query 的 4.70%，不是空 padding 导致。普通 FULL 轮数反而为 85（warm65 为 87）；full 中 B5 的实际 owner-round 仅 353/43018，不能把 884 个 FULL 轮都当成真实分叉。target worker 平均耗时从 73.85 升到 281.17 ms/轮，但未保存逐调用 kernel wall，尚不能把全部差额唯一归因到某个算子。rank0 的 target/join 326.90 s 嵌在 draft 421.04 s 内，两者不得相加。

实际 eager 生成 25686 行，晋级 10819、失效 14867；完整运行的发布行数与验证候选数两条守恒都推出消费 10819，但没有逐 ticket 的消费时刻直接记录。下一步优先单独验证生产 reservation 关闭后走原生 standalone whole-prompt prefill 的效果，保持预算、模型、SLO、Poisson、计时和原资格门不变。原 E-window local-ready 方案保留为独立后续候选，暂不同时叠加。

证据：[完整独立审计](../budget-calibration-20261015/budget-diagnostics/ROOT_V140_ACTUAL_NORMAL_ADMISSION_REVISION2_AUDIT.json)、[完整成本摘要](../budget-calibration-20261015/budget-diagnostics/V140_ACTUAL_FULL463_COST_SUMMARY.json)、[实际 g 数据](../budget-calibration-20261015/budget-diagnostics/V140-actual-g-normal-admission-v140-one-full463-v1-full463/actual-observed-g.csv)。第一次外部审计误将 physical scratch owners 与 real budgets 等长比较，失败记录保留；新外部脚本严格校验物理形状后按 real N 前缀统计，未修改冻结实验或原始回执。


2026-10-17 01:10 UTC 补充诊断：V141 单点 M165/S38→M128/S32→原形状实际完成，三张 TP 卡的全词表生产 token 均为 7628→8298→7628。原形状 7628/8298 的 BF16 logit 同为 19.125，紧凑形状中 8298=19.25、7628=18.875；还原后 hidden、logits 和该位置 64 层 K/V 均逐字节一致。跨形状最终 hidden maxabs=0.8125，最早观察到当前点 K/V 差异在第 1 层；未保留逐层 hidden，不能据此唯一归因 NZ14 或某一算子。各自同形状的 Native 独立资格、两次零新增、封存仍通过，说明同形状资格不能替代跨形状等价。V135 的额外四个紧凑分区继续不采用，无完整负载性能声明。[单点实际 CPU 独立分析](../budget-calibration-20261015/budget-diagnostics/ROOT_V141_ACTUAL_SINGLE_POINT_CPU_REVISION2_AUDIT.json)。首次外部分析被其可选 NVML import guard 阻断，原记录保留；新副本以 ModuleNotFoundError 阻止可选模块导入并让 CPU Torch 正常加载，未查询设备、改变数值门或修改冻结源。


2026-10-17 01:16 UTC：V143 已开始实际完整试验。唯一生产语义变化是在每个 fresh live hook 实例上拒绝创建 mixed prefill reservation，随后继续原生整段 prompt prefill；环境中的 mixed flag 和原混合资格 actor 不变。原生成控制器逐字节不变，不调整动态 B、g、窗口、MC2、模型精度、Poisson、SLO 或计时，也没有新增 clock、collective、设备 fence 或逐轮 observer。7 项新 HOST 实际通过（8.757s），父13/8/23仅按SHA复用；352本地/5097外部源与5451归档成员由 ROOT 再核。仍须完成实际201/44/86/28启动、原21scope26phase、warm65/两次zero/封存/完整463及独立审计。当前没有新Goodput或达标声明。[启动前复核](../budget-calibration-20261015/budget-diagnostics/ROOT_V143_PRE_NPU_SOURCE_AND_HOST_AUDIT.json)。


2026-10-17 02:06 UTC：V143 四 rank 已各保存原 26 个阶段，尚在完整 chain/lifecycle 汇总，不能据阶段文件直接追认整体验证或报告 Goodput。下一项 V144 局部变量 B4 组件已冻结：两组 N32 的真实候选数分别为 C96/Q128/M128/S32 与 C95/Q127/M165/S38；使用独立同时租用的 candidate/reference bank，原 full-M greedy 输出即时保存、原 hidden/KV matcher、typed publication 和 compiled verifier/commit/nextdraft。其本地配置为 maxseq32/queued64/storage96/block128/pages544，不能代替完整生产 storage527 资格。13 项新增 HOST 通过，ROOT 与独立代码复核通过；实际设备运行仍未开始。

V144 计划每组四轮真实探索及两次 fresh-history 完整固定重放，共 8 次探索提交、16 次重放提交，另 6 次末轮后 nextdraft。自然补 KV 路径只有在实际走完 publication→target→commit 时才计为 mandatory 覆盖；仅末次 nextdraft 遇到 repair 不授整体组件资格。复核发现的连续 TP 调用间异常错位已通过所有 rank 的固定子阶段边界修复，诊断同步不作为生产性能证据。原始初始 root 是源数据中的 teacher-forced token，后续草稿和提交真实生成，不能称全工作负载独立 AR 验证。此组件没有测速或新的 Goodput。证据：[ROOT V144 源码/归档检查](../budget-calibration-20261015/budget-diagnostics/ROOT_V144_PREPARED_SOURCE_HOST_AUDIT.json)。


## 2026-10-17 02:20 UTC：V143 整段预填充对照已完成，目标仍未达到

原463请求/Poisson seed0/256输出/SLO40、50、150ms不变，原21scope26phase、warm65、两次zero、封存、full463均通过；源与外部输入未变，自有进程退出、页面归还、四卡空闲及原环境复核通过，独立终态审计通过。下面是一次完整候选结果，尚无新的同卡交错基线对照，历史最好树形仍为259.3551；本轮没有超过固定γ4的501.6356。

| 指标 | 原Native /256 | 首末token事件墙钟 /256 |
|---|---:|---:|
| Goodput（token/s）|147.67029|144.68705|
| 达标请求/463 |99|97|
| tight/normal/loose达标数 |1/9/89|1/8/88|
| Raw（token/s）|690.61962|690.61962|
| 完整generate耗时（秒）|171.62559|171.62559|

原Native逐请求TPOT中位数/P90（ms）为：tight 76.83/87.98、normal 60.83/69.86、loose129.95/144.95。事件墙钟为77.51/88.74、61.45/70.53、131.04/146.24。差异按原定义分列，不改统计口径。

生产mixed调用实际为0；standalone prefill为397次、463个owner、203850个真实prompt query，target leader原HOST refill区间为50.65672秒。target1561轮中1410causal、151FULL，1106轮N32；target总HOST区间85.99878秒，draft92.21399秒内嵌target仅0.59343秒。原先mixed join大幅减少，但这些HOST区间仍不能当纯kernel成本，也不能相加推导总时长。tight时延仍接近SLO两倍，后续需要核对预算分配、target批容量及预填充停顿的结构性差异。

本轮42902个验证owner-round中B3=41951、B4=21、B5=646；B5才含真实侧分支，因此不能将大部分高并发B3执行描述成真实分叉。E生成25051行、晋级10735、失效14316，完整发布行/验证owner和候选数两条守恒均推出实际消费10735行（32205候选）；它是有前提的整体推导，并非逐ticket直接计数。

实际逐请求token事件与target记录按owner+prefix精确一一匹配42902次，无缺失/歧义；由原事件独立重算g的rounds和delivered与保存状态完全一致，再与三类图的7个观测点CSV逐项核对。B1/B2仍无可纳入g更新的非截断观测，保持空白，不能宣称三类完整预算曲线已确定。原g3为tight2.77758、normal3.14371、loose2.40734。

证据：[完整独立审计](../budget-calibration-20261015/budget-diagnostics/ROOT_V143_ACTUAL_STANDALONE_PREFILL_AUDIT.json)、[实际成本摘要](../budget-calibration-20261015/budget-diagnostics/ROOT_V143_COMPLETED_FULL463_COST_SUMMARY.json)、[逐请求精确重建](../budget-calibration-20261015/budget-diagnostics/ROOT_V143_RETAINED_STREAM_BUDGET_IDENTIFIABILITY.json)、[实际g图及CSV来源](../budget-calibration-20261015/budget-diagnostics/V143-completed-actual-g-standalone-prefill-v143-one-full463-v1/provenance.json)。CANN运行期fusion诊断原字节已另存并绑定回执SHA。随后才启动V144两变量B4局部设备资格，ROOT session88144；其结果待实际完成，不计为性能结果。


2026-10-17 02:31 UTC：V144 局部变量B4资格实际完成并通过独立终态复核。两组真实预算场景均自然出现repair，并完成后续typed publication、target、原协议commit；8次探索、16次完整固定重放、6次末轮nextdraft齐全，两次本地zero原21字段增量为0。只观察到PA40/44，两组均有已commit的mandatory repair覆盖。原MC2/独立同M全hidden和query KV/原协议检查通过，源未变，进程退出、页归还、四卡空闲。实际运行约406.24秒；它是maxseq32/queued64/storage96的局部正确性证据，没有新Goodput，不能替代full527进程自己的资格。[V144独立实际审计](../budget-calibration-20261015/budget-diagnostics/ROOT_V144_ACTUAL_VARIABLE_B4_AUDIT.json)。V146将仅提高high-N cap至4并在本full实例补齐两组资格；V147另准备大prompt mixed局部资格，二者暂不叠加。

**固定γ4基线批次范围纠正：实际target最大也是32，不能解释成tree32对fixed64。** fixed的两个home各32是活动容量；测量实际target1932轮、43221 owner-round，平均22.37请求/轮，V143为1561轮、42902 owner-round，平均27.48请求/轮。两者实际prompt SHA一致、相对到达间隔逐项相同；fixed以首次到达归零，V143保留manifest首次0.4651518秒偏移，原计时不事后扣除。完整target真实query量（含prompt/fallback）约376817对376340，几乎相同。固定更快不能归因为少处理prompt或target cohort更大。

fixed测量实际draft normal32880+E24378=57258 work rows、1933次四步链；其物理bucket/queries只保留构造以来累计，不能拿累计960536充作本轮。V143 normal32167+E25051=57218 work rows，调用计数4434=1561 chain+1376 repair+1376 frontier+121 rootseed；三步链不等于整体forward或等待更少。fixed draft HOST区间65.7012秒排除后置staged prefill40.1058秒；tree draft92.2140秒的定义含本地同步/CPU选树等，不能把差值当纯设备收益。fixed E的normal+promotion与实际验证行差3，未保留足够证据闭合消费守恒，不强填；V143消费10735有自身两条精确守恒。[真实范围对照](../budget-calibration-20261015/budget-diagnostics/V143_VS_FIXED_GAMMA4_PREFILL_BATCH_ADVANCE_READONLY.json)。


2026-10-17 02:45 UTC：V146 的 high-N cap4 源已冻结，新增9项 HOST 在框架绑定前、后各执行一次（7.076s及16.119s），父套件只按 SHA 复用。随后独立复核发现新增库存守卫与其fake-engine测试共同误分了 Native 缓存族：target 的201个prefill与44个decode实际均在target_entries（21/26后为247），不是entries201/target46；draft实际draft_entries60、target_entries54，总114，也不是draft86/target28。ROOT在supervisor归档阶段中断，模型子进程尚未创建，未执行模型、组件或full463，没有新Goodput。原冻结源、失败回执和部分归档全部保留；原环境前后复核通过、四卡空闲。

此错误是新检查器与测试夹具对缓存分类的共同误解，不是数值或性能失败。V148将只修正确切缓存族及其HOST来源回归：直接调用原startup key producer/原库存验证，并对已完成V143真实库存做比对、加入旧错误分类的反例。预算、算法、数值门、原21/26与计时不因此改动。[V146中断的独立分类记录](../budget-calibration-20261015/budget-diagnostics/ROOT_V146_ABORTED_BEFORE_MODEL_NATIVE_NAMESPACE.json)。


2026-10-17 03:10 UTC：V148 修正真实 Native dictionary 分类后，10项新增 HOST 实际通过（15.206s用例、28.812s包装总时长），377本地/6007外部源前后SHA一致；ROOT与独立peer复核后冻结，6390成员/73,134,846B归档逐字节检查通过。生产仍仅是high-N normal单请求cap3→4，总normal96、global192、targetC96/Q128/N32不变，shared4继续defer E。首次tag在原运行库校验前置失败，模型子进程未创建：两旧版兼容目录link缺失，保留运行库15个外部文件与原lock的SHA全部相同。ROOT只恢复两个目录symlink，未安装/修改二进制或软件包；原10496文件/15外部/212package verifier随后通过。失败tag和日志保留，冻结代码未变，以新tag high-normal-cap4-v148-restored-paths-full463-v2重试。

该新tag已实际启动模型进程，4rank MC2预检查通过，正在原201prefill准备/验证阶段；后续原44/86/28、21scope26phase、ownfull527可变B组件、warm65/两zero/封存/full463都仍待完成。当前没有新GP或达标声明。[ROOT冻结源审计](../budget-calibration-20261015/budget-diagnostics/ROOT_V148_PREPARED_SOURCE_HOST_AUDIT.json)、[环境路径恢复原字节证据](../budget-calibration-20261015/budget-diagnostics/ROOT_RUNTIME_COMPATIBILITY_ALIASES_RESTORED_20261017T0302.json)、[首次tag前置失败分类](../budget-calibration-20261015/budget-diagnostics/ROOT_V148_TAG1_RUNTIME_ALIAS_FAILURE.json)。


### 2026-10-17：高并发 normal cap4 完整对照（V148）

在相同 463 请求、原 Poisson 到达序列与 40/50/150 ms SLO 下，提高高 N 的 normal 预算上限至 4；保持 target C96/Q128/N32、全局预算 192、原 standalone prefill 和 MC2 路径。以下为一次完整运行，Native paper 时钟采用原运行外层区间，不能用模型内层时间替代分母。

| 方法 | Raw token/s | Native Goodput token/s | stream Goodput token/s | tight 达标/271 | normal 达标/96 | loose 达标/96 |
|---|---:|---:|---:|---:|---:|---:|
| V143：高 N cap3 | 690.62 | 147.67 | 144.69 | 1 | 9 | 89 |
| V148：高 N cap4 | 787.22 | 204.03 | 197.23 | 2 | 22 | 96 |

表中达标数采用 Native paper TPOT；V148 的 stream 达标数为 2/18/96。V148 的 Native TPOT 中位数（tight/normal/loose）为 68.64/55.58/98.99 ms。Goodput 比 V143 增加约 38.2%，仍低于已保留最佳树形结果 259.36，也低于固定 γ=4 的 501.64；未达到较强 MC2 TP4 基线 1.5 倍的要求。

实际预算为 B1/B2/B3/B4/B5 = 2625/8555/13134/17777/568 个 owner rounds，仍包含真实树分支；实际 eager 行数为 0。提高 normal 预算同时触发了当前 shared-steps 门，推迟 E，因此本轮不能被解释为“normal4 与 rolling E3 的联合收益”。

原 21 scope/26 phase 与后置生命周期、同一 full527 进程内新增 B4 组件、warm65 与两次 zero65 均完成，正式阶段原 Native 新增/失败计数为零。独立外审通过；仅支持已运行的同 M 数值、KV、协议及记录检查，不宣称全部请求与独立 AR 输出已逐 token 比较。旧外审 revision2 的 tuple/list 比较失败保留，revision3 仅修正 JSON 序列化口径。

证据：[完整外审](../budget-calibration-20261015/budget-diagnostics/ROOT_V148_ACTUAL_FULL_CAP4_REVISION3_AUDIT.json)、[成本与预算分析](../budget-calibration-20261015/budget-diagnostics/ROOT_V148_COMPLETED_FULL463_COST_SUMMARY.json)、[逐请求流与 g 观测重建](../budget-calibration-20261015/budget-diagnostics/ROOT_V148_RETAINED_STREAM_BUDGET_IDENTIFIABILITY.json)。

下一步先实测 V149 repair/frontier 合并的完整前置处理成本；其固定 prefix 成本不等同动态 prefix 时延或全链路 Goodput。V151 正准备 normal4＋parent3/child3 的真实 shared4 局部资格，尚未打开生产 deferE 门。


### 2026-10-17：repair/frontier 合并局部耗时（V149）

同进程、封存图后各 11 对 AB/BA；计时包含 metadata、实际模型/head 与真实 frontier token 读回，准备及完整前缀哈希不在计时内。

| 局部规模 | 原两次调用 ms | 合并一次调用 ms | 原/新中位数 | 时延下降 |
|---|---:|---:|---:|---:|
| R1/E1 | 7.9812 | 5.8722 | 1.359× | 26.42% |
| R32/E32 | 10.3439 | 9.1968 | 1.125× | 11.09% |

两个端点全部逐对样本均更快，分别与独立同 M 参考的数值门通过，frontier token 相同。跨 M 比较中，小端点 real hidden 和全部 query KV 不满足原 matcher；大端点这些比较通过。此处报告的 crossM_behavior_equal 字段仅表示 frontier token 相同，不代表小端点的 hidden/KV 与旧路径相同。

本轮只有固定 prefix，未保留实际 TTU task/skip 计数；尚未运行完整 child chain、target verdict、commit/consume/nextdraft 或 463 请求。上述结果不能作为动态 prefix 或 Goodput 提升结论，暂未采用到完整方法。

证据：[逐对耗时与 252 项 Native 零差分复算](../budget-calibration-20261015/budget-diagnostics/ROOT_V149_ACTUAL_PURE_PRELUDE_COST_AUDIT.json)。后续 V151 已开始检查 normal4＋parent3/child3 的真实 shared4 局部协议。


### 2026-10-17：normal B4 与 eager B3 共用四步草稿（V151）

本轮局部数值检查正常结束（904.14 秒）：parent3、normal 可变 B≤4、child3 共用真实四步草稿链；独立同形状 hidden、编码、KV、失效后的后续草稿和两次完整重放通过，336 项原 Native 计数差分均为零。

但本次固定前缀中，32 个 parent 的接受数分布每次均为 0 个：8 请求，1 个：8 请求，2 个：16 请求，没有请求完整接受三个父草稿，因此没有自然 eager 提升和消费。结果仅为 `passed_subset`，生产 eager 门仍关闭；不能据此声称完整协议已验证或 Goodput 提升。下一步使用 V143 完整运行保留的真实 p128 前缀验证自然提升分支，不强制接受或修改 target 输出。

证据：[V151 原始终态记录](../budget-calibration-20261015/tree-normal4-parent3-shared4-eager-v151/receipt-normal4-parent3-shared4-v151-local-v1.json)、[真实前缀筛选](../budget-calibration-20261015/budget-diagnostics/P151_NATURAL_INVALIDATION_AND_V143_FULL_B3_SOURCE_CANDIDATES.json)。


### 2026-10-17：大块混合预填充启动检查（V153）

已准备完整 2301-token prompt 的 512×4＋253 连续分块数值实验，配置为 64 并发、463 排队、527 存储行；target 物理 M645，普通请求仍为 N32/C96/Q128。第一次启动在模型构造前被源码路径检查误拒：PyTorch 的 `torch.classes.__file__` 是相对伪路径 `_classes.py`，检查器将它解析成了项目目录下的文件。四个 worker 均未进入模型构造，原环境前后校验均通过，进程已清理、四卡空闲。

旧源和失败记录保持不变，后续使用新版本修复路径检查。这次没有运行数值或性能实验，不能据此判断大块预填充效果；M645 的新 MC2 profile 也尚未生成。证据：[启动失败分类](../budget-calibration-20261015/budget-diagnostics/ROOT_V153_PREMODEL_ORIGIN_FALSE_REJECTION_CLASSIFICATION.json)。


### 2026-10-17：自然 eager 分支与跨页预分配修复

V154 使用旧完整运行保存的真实请求前缀，实际 target 裁决后自然进入 eager 提升所需的分支；随后在 real33/physical36 的 padding 检查中因缺少 `source_epoch` 导入而退出。该轮没有完成子草稿消费，不能记作完整协议通过。V157 仅补当前只读缓存接口的导入，并提前保留真实 gate 行；已重新冻结，当前正在设备验证。

V156 修复路径检查后成功构造模型，但第一次 M645 混合预填充在计算前被原 KV 检查拒绝。已用实际 Native 缓存分配器在 HOST 复现根因：预分配只请求了末页，中间逻辑页没有分配，跨页 dummy 请求引用到空页。后续新版本将逐页分配，再重跑完整 2301-token 分块验证；该失败尚不能说明大块预填充的性能。

上述均没有新 Goodput。当前保留的最佳完整树形 Goodput 为 259.36，固定 γ=4 对照为 501.64，目标仍未达到。证据：[V154 失败分类](../budget-calibration-20261015/budget-diagnostics/ROOT_V154_POSITIVE_BRANCH_PADDING_DEPENDENCY_FAILURE.json)、[V156 实际分配器复现](../budget-calibration-20261015/budget-diagnostics/ROOT_V156_UNALLOCATED_DUMMY_BLOCKS_FAILURE_AND_HOST_REPRO.json)、[V157 冻结复核](../budget-calibration-20261015/budget-diagnostics/ROOT_V157_PREPARED_SOURCE_HOST_AUDIT.json)。


2026-10-17 05:26 UTC 补充：V157 终态为失败（215.89 秒）。已保留四个 rank 完全相同的 32 行真实 gate：16 个父请求接受全部三个候选且满足 eager 有效条件，其余接受数为 2 个：8 请求，1 个：4 请求，0 个：4 请求。尚未 commit 提升或消费子草稿；独立同 M hidden 检查之后的 PA 元数据合取检查失败。源码显示该检查跨重新分配缓存比较原始物理页号，后续 V161 将分别核各自真实映射并保留逐字段标记，维持独立 KV/输出检查。此失败没有新 Goodput。[V157 终态与原始 gate](../budget-calibration-20261015/budget-diagnostics/ROOT_V157_TERMINAL_REFERENCE_METADATA_FAILURE.json)。

V158 逐逻辑页分配修复已完成五项 HOST、独立源码与冻结复核，并于 05:26 UTC 启动完整 2301-token/M645 局部设备实验。HOST 使用真实 Native 分配器及原 allocator 主体，模型写入和设备栅栏为桩：原 whole/P1 夹具分别为 154/150 页并缺 dummy 中间页，修复后均为 158 页且所需页完整；这些是 HOST 夹具数字，不是设备显存高水位或吞吐结果。[V158 冻结复核](../budget-calibration-20261015/budget-diagnostics/ROOT_V158_PREPARED_SOURCE_HOST_AUDIT.json)。


2026-10-17 05:34 UTC 补充：V158 终态失败（308.51 秒）。四个 rank 的第一遍 whole2301 均完成 5 块，原数值、KV 与真实 normal/prompt 提交门通过；尚未完成两遍完整 zero。随后 final-Q1 场景在原 manager 初始前缀采纳接口中缺少 `local_qualification_scope.scopes()` 而退出，尚未执行该场景的新 target 计算。V164 将补当前 scope 的兼容接口。环境前后校验通过、源未变、进程已清理。[V158 完整首遍及 Q1 接口失败记录](../budget-calibration-20261015/budget-diagnostics/ROOT_V158_WHOLE_PASS_AND_P1_SCOPE_INTERFACE_FAILURE.json)。

V161 修正跨重新分配缓存直接比较物理页号的问题，保持独立模型、全 hidden/encoded/KV 与原协议门，并于 05:34 UTC 开始新的自然提升/消费设备验证。当前仍无新 Goodput。[V161 冻结复核](../budget-calibration-20261015/budget-diagnostics/ROOT_V161_PREPARED_SOURCE_HOST_AUDIT.json)。


### 2026-10-17：自然 eager 提升、消费与后续草稿通过（V161）

V161 局部设备验证正常完成（321.30 秒），ROOT 独立复核四个 rank 的 3891 个共同阶段及 84 项原 Native 计数减法。32 个真实父请求中，16 个自然满足条件，随后其子草稿全部真实消费并通过原 target/verdict/commit 与独立后续草稿检查；另外 16 个正确失效。独立同 M 的 union M36/F32 全 physical hidden、完整四步 encoded512、56 项 query KV、父草稿和 ledger 检查通过。

实际记录证实：候选与重建参考的物理页表、槽位确实不同；各自十字段及本分配内的 owner/slot/table/padding 核对全部精确相等，跨分配逻辑语义相同。修复针对原检查器错误要求跨分配页号一致的问题，并未移除独立输出或 KV 检查。

本轮只有一个固定前缀、R1/E32 的局部组件，没有两次完整 zero、生产重叠或 463 请求 Goodput；不能把16个请求的局部结果当作吞吐提升。源未变，原环境前后校验通过，进程退出且四卡空闲。[ROOT 独立实际复核](../budget-calibration-20261015/budget-diagnostics/ROOT_V161_ACTUAL_POSITIVE_CONSUME_AND_METADATA_AUDIT.json)。

V164 已于 05:41 UTC 启动新的 whole2301/Q1 与完整 zero 检查；活动窗口 32/24/16 和真实 shared4+合并 prelude 的完整负载执行层正在分别准备，后续用原 463 请求与原时钟实测。

### 2026-10-17 05:55 UTC：V164 大形状完整提示验证完成

V164 四卡实测完成（718.57 秒）：完整 2301-token 提示按 512-token 分块推进，以及独立 2300→2301 的末尾单 token 场景，各执行探索一轮和完整零增量复测两轮。每卡完成 18 次真实提示/normal 提交、6 次后续 draft；72 个阶段记录与四卡 2347 个共同边界一致。两轮复测的 336 项 Native 计数差分全部为零，完整 hidden/KV 与独立同形状参考比较通过，原始源文件未变、工作进程已退出。这是局部数值与协议验证；不是完整 463 请求性能结果，也未获得 M=645 定制 MC2 性能资格。下一步先实测活动窗口 32/24/16 对完整负载 Goodput 的影响。

原始核对记录：[ROOT_V164_ACTUAL_WHOLE_PROMPT_AND_TWO_ZERO_SWEEPS_AUDIT.json](../budget-calibration-20261015/budget-diagnostics/ROOT_V164_ACTUAL_WHOLE_PROMPT_AND_TWO_ZERO_SWEEPS_AUDIT.json)。


### 2026-10-17 06:29 UTC：V166 完整 32 请求窗口已保存，统计阶段退出

V166 的 32 请求活动窗口已完成原 463 请求、每请求 256 token 的推理，并保存四个 rank 的完整 trace。随后 HOST 统计函数缺少 `sha` 名称而退出，整个作业状态为失败；24、16 窗口尚未执行。冻结源不改，后续以新版本修复统计入口。

保留 trace 的离线原口径重算：Raw 591.70 token/s，Goodput 244.09 token/s，达标 191/463（tight 5/271、normal 90/96、loose 96/96）。这不是成功通过整个作业的正式候选，未替换最佳结果。2544 次 target 验证中，有 1884 次在 N16/Q80 的因果路径执行了物理 128 行图，提示有可验证的 padding 开销；减少物理行数仍需新的数值资格和完整性能实测。

[保留完整 trace 的 ROOT 重算](../budget-calibration-20261015/budget-diagnostics/ROOT_V166_RETAINED_FULL32_ANALYSIS.json)。

四卡保留 trace 已另行通过原完整输出及 warm/zero 计数差分审计，重算结果一致；这不改变整个 V166 作业失败、24/16 窗口未跑的状态。[独立终态核对](../budget-calibration-20261015/budget-diagnostics/V166_INDEPENDENT_RETAINED_FULL32_TERMINAL_AUDIT.json)。


### 2026-10-17 06:56 UTC：V163 因果 mask 检查误拒

V163 终态失败（1085.15 秒）。原 201 个 prefill、44 个 target 和 114 个 draft 图检查及同进程可变预算组件完成；ROOT 独立复核其真实四周期裁决、提交、后续草稿与两轮 Native 零增量。进入首个新增混合路径时，三张 target 卡在共同元数据预检查处被拒，尚未执行该路径的 target 计算，也未运行完整 463 请求计时。

已确认一处足以触发拒绝的检查器错误：冻结 Native 源码创建的因果 mask 是 `int8[2048,2048]`，新检查却要求 `bool`。后续 V172 将只修正此检查，并同时覆盖候选与独立参考两个实际调用入口；FULL 分支的布尔 mask 规范保留。旧源码和失败记录保持原样。环境前后校验通过，源未变、进程已清理、四卡退出空闲。V167 减少 padding 的独立候选随后启动，性能待实测。

[ROOT 终态与源代码核对](../budget-calibration-20261015/budget-diagnostics/ROOT_V163_ACTUAL_CAUSAL_MASK_DTYPE_PREFLIGHT_FAILURE.json)。


2026-10-17 07:18 UTC 续跑记录：V167 在真实新 M80 计算前的独立前缀准备阶段失败。real16 适配遗漏 `DualBank.prefix_ready` 中的 `i%32`，32 个物理 owner 索引 16 个 prompt 导致四卡共同 `IndexError`；原 full527 B4 数学/协议组件通过，新 M80 和 full463 未完成，无新 Goodput。ROOT 终态审计见 [ROOT_V167_ACTUAL_BANK_PREFIX_INDEX_FAILURE.json](../budget-calibration-20261015/budget-diagnostics/ROOT_V167_ACTUAL_BANK_PREFIX_INDEX_FAILURE.json)。冻结 V170 同类遗漏已静态确认，禁止直接运行；分别准备 V173/V174 最小修复及实际生命周期 HOST 检查。V172 cached causal mask int8 修复源已通过 ROOT/peer 审查、封存和归档校验，开始独立设备资格及 full463；尚无性能结果。

2026-10-17 07:22 UTC 启动记录更正：ROOT 最初误调用 V172 内层 `launch172.sh`，未经过外层 supervisor。该次 v1 仍在 constructor 预热时已由 ROOT 终止，全部自有进程清理，保留 [启动纠正记录](../budget-calibration-20261015/budget-diagnostics/ROOT_V172_DIRECT_INNER_LAUNCH_ABORTED_BEFORE_QUALIFICATION.json)，不计资格/性能。随后从冻结 `guard172.py` 正确入口启动新 v2 tag；原 source/CPU/冻结未变。


### 2026-10-17 08:38 UTC：V172 完整463保留结果与后续选择

V172（`live645-int8-v172-full463-v2`，原 active64、C96/eager96/global192）已在四个工作进程上完成原463请求；原warm65、两次zero65和封存后的full463均保留，Native的21项禁止计数增量全部为零。完整测量耗时397.540917932秒。工作进程正常关闭，但主进程随后在 `audit172.audit_component` 读取 `execution['used_aclgraph']` 时报 `KeyError`：NativeGraphExecution 的该成员是由 mode 派生的 property，`dataclasses.asdict` 不会保存它；这与图执行失败含义不同。监督任务因此以1退出，不能记为正式成功。事后原始基线校验通过，源文件未变，四卡均空闲，自有进程无遗留。

独立HOST重算的诊断结果：Raw 约298.152956 token/s，Native论文口径Goodput为4.507712085 token/s，达标7/463（tight 0/271、normal 0/96、loose 7/96）。TPOT总体中位数191.328476ms、P90 317.009826ms。实际调用包含381次645行mixed target和1103次普通target，合入117903/203850个prompt query。rank1 target阶段HOST边界累计261.678836秒；该时间包含等待，不能与其他角色耗时相加当作设备计算总量。扩大物理图并合入预填充没有带来本轮性能收益，本方案不采用，也不替换既有最优有效树形结果。完整数学/KV/协议原始数据仍在独立审计中。证据：`../budget-calibration-20261015/budget-diagnostics/V172_INTERIM_INDEPENDENT_RETAINED_FULL463_METRICS.json`。

08:35 UTC 已通过ROOT自持锁的 `guard169.py` 启动V169（tag `parent4-v169-full463-v1`），验证线性parent4后续child3草稿的自然就绪、发布、消费与原完整测量；它未叠加645行mixed方案。后续优先对比active16及high-normal cap3，后者保留低N下真实B5分支，不把线性B3当成树分支收益。当前没有新达标结论。


### 2026-10-17 08:47 UTC：V169 外围统计竞态与独立177修复

V169第一次运行 `parent4-v169-full463-v1` 在启动预填充阶段、模型组件验证之前被监督进程终止（189.191866秒），原因是磁盘统计 `os.walk` 列出 `startup-progress/prefill-rank2.tmp` 后，该临时文件被原子替换，随后的 `stat` 抛出FileNotFoundError。此轮没有性能结果。原冻结源未改，事后原10496文件/15外部文件/212包校验通过，自有进程无遗留，四卡空闲。记录：`../budget-calibration-20261015/budget-diagnostics/ROOT_V169_ACTUAL_STORAGE_TMP_STAT_RACE_FAILURE.json`。

原冻结V169及其他候选文件保持原样，新增单独审读、实际HOST7项通过、独立peer与ROOT审读并打包的177监督层修复。它只允许注册的原候选 `actual` 输出树内 `.tmp` 文件在stat时真实ENOENT；普通文件、不可变源码、其它异常及符号链接继续拒绝。磁盘扫描仍是并发写入下的近似统计，未宣称某一瞬间的精确原子快照。原guard按原文件位置加载，P/G、模型启动命令、独占运行锁、基线校验、资源界限和自有进程清理保持原实现；177自身来源在调用前后独立校验，并记录原receipt的SHA。

08:47 UTC 已以 `parent4-v169-full463-v2-overlay177` 重新启动同一模型源码；入口为 `budget-diagnostics/v177-root-safe-storage-overlay/run177.py --epoch 169 --action linear-parent4-child3-local --tag parent4-v169-full463-v2-overlay177`（经原serial runtime环境启动）。完整命令与修复档案位于 `ROOT_POST_V169_TMP_RACE_SAFE_STORAGE_GUARD_QUEUE.json`。本修复不构成模型性能优化收益。


### 2026-10-17 08:52 UTC：V176 源码已固定，尚无设备性能结论

新增V176只将较大 normal draft cohort（admitted N>8）的每请求候选上限由4改为3；低N仍保留真实B5，当前target候选上限96、normal/eager各96和全局192保持原配置。这里的N指生成普通草稿时的批次，不是后续灵活合批后的target N，不能断言target N>8就全部为B3。它仍按原g/need和优先级选择预算。

实际控制器HOST检查中，32个完整B3 proposal可一起装进96候选的target批次；B4只能装下24个完整proposal，其余请求延后。实际shape检查也验证了normal16与eager16可各取B3，使用原三步路径；原B4配置在该例中会因共享四步条件延后eager。它们是代码路径和容量证据，尚不等于设备时延或Goodput收益。原114 draft/44 target/201 prefill初始化、当前完整527范围的B4数学/KV/verdict/协议与后续草稿检查、实际65预热/两zero/封图/full32均保留，没有借用旧模型进程的数值资格。

V176已有4项实际HOST检查、独立peer和ROOT源码审读以及162成员的复现档案，计划在当前V169试验结束并清理后运行。入口及顺序已更新至 `budget-diagnostics/ROOT_POST_V169_TMP_RACE_SAFE_STORAGE_GUARD_QUEUE.json`；冻结SHA为 `5e2bfc3ae1ce9bb9d0553f2ed2ec3b3457a9f6799b1820cc4d800e18df8de319`。


### 2026-10-17 09:02 UTC：V169 第二次启动后独立bank导入遗漏；开始V176

V169第二次 `parent4-v169-full463-v2-overlay177` 已越过原初始化，但在第一个P3代表例 `N1-E1-R1/allocate-independent-real-banks` 失败：rank0运行从 `bank_v149.py` 适配的 `build_bank` 时缺少 `hashlib` 全局绑定，四rank共同边界拒绝继续。此轮没有完成新组件或full463，没有性能数值。177磁盘统计修复层自身前后来源一致；原冻结源未变、原基线事后校验通过，自有进程无遗留、四卡空闲。原失败记录和源码保留，不在原冻结文件内补导入；新179将针对这条真实bank调用补最小HOST检查后单独审读。

09:02 UTC已启动封存V176（`high-normal-cap3-v176-full463-v1`，原 `guard176.py` 自持锁），优先验证高N候选上限3、低N真实B5和原parent3/shared3复用。当前仍处于初始化，没有新Goodput结果。


### 2026-10-17 09:38：cap3 完整结果与后续紧凑图测试

V176（高并发 normal cap3、active32，低并发保留真实 B5，原 C96/E96/全局192）已完整完成并通过当前组件、两轮零增量与原463请求独立审计：Raw 550.9989 token/s，Goodput 224.9218 token/s，耗时215.1148秒；达标189/463（tight 6/271、normal 87/96、loose 96/96）。源码未变，四卡已清理。该结果低于此前 full32 保留诊断的244.0908，也低于已完成树形最好结果259.3551，未采用，未达成优化目标。

提前生成机制确已执行：生成24817个后续候选，提升为可用后续草稿10471个，失效14346个。原目标布局仍有133401/306452物理行属于padding；draft模型提交7382次，对照full32为5062次。以上计数说明需要继续降低实际执行开销，不能据此把任一开销直接等同于可回收时间。

独立审计：[V176_INDEPENDENT_TERMINAL_FULL463_AUDIT.json](../budget-calibration-20261015/budget-diagnostics/V176_INDEPENDENT_TERMINAL_FULL463_AUDIT.json)。随后启动V173的M80/S16布局及real16 bank修正版，先重新通过自身数值、KV及提交/下一轮校验，再进行完整测量；V180则在HOST侧准备常见uniform B3的三档紧凑图（N9–16/M64、N17–20/M80、N21–24/M96）。新布局尚无性能结果，正式对照表不变。


### 2026-10-17 09:56：M80/S16 完整测量与三档 B3 图

V173（沿用原 cap4/低N真实B5、active32、C96/E96/192；仅针对N16、B4连续链采用M80/S16新图，并修正real16 bank）已完整完成：Raw **635.7311 token/s**、Goodput **291.0907 token/s**、耗时 **186.4436秒**，212/463达标（tight 21/271、normal 95/96、loose 96/96）；整体逐请求TPOT中位数46.3908ms、P90 55.4793ms。原当前组件、新M80组件、两零增量及完整463原审计通过；源码与storage-only overlay均未变，worker已退出，四卡空闲，原baseline前后检查通过。完整独立终态复核另行保存，尚不宣称全负载AR等价或正式采用。

该单轮高于此前树形最好完整值259.3551，也高于旧full32保留诊断244.0908；后者的整job曾因metrics错误失败，且不同运行历史不能视为逐步配对因果实验。新值仍低于固定串行gamma4的501.6356，更未达到较强MC2基线的1.5倍目标644.7664。

随后启动V180：保持V176的cap3/低N B5及预算/active32，只对真实uniform B3连续计划采用N9–16/M64/S16、N17–20/M80/S20、N21–24/M96/S24三档图。源码及原guard已独立审查并封存；实际当前数学、六个新Native对象、协议及性能仍须新进程验证。每N至少验证三个连续推进的prefix上下文；六个边界N另做三次fresh完整轨迹及两zero，共28条轨迹、102个协议cycle和28个next，不应理解成每个N都有三次fresh独立重放。


V173独立终态复核已通过（2026-10-17 10:00），记录见[V173_INDEPENDENT_TERMINAL_FULL463_AUDIT.json](../budget-calibration-20261015/budget-diagnostics/V173_INDEPENDENT_TERMINAL_FULL463_AUDIT.json)。当前树形最好完整测量更新为291.0907 token/s，仍未正式采用、未达成目标。组件12个上下文比较中，新旧M的全部80个head及原裁决一致；这是已测组件范围，不能扩展为全负载AR等价。完整轨迹2557次target调用中，1874次（73.29%）使用M80/S16，padding物理行比例14.49%；Eager生成/提升/失效均为0，说明本轮收益不能归因为rolling-eager。与此前166 full32保留诊断比较详见[V173_COMPLETED_PERFORMANCE_BREAKDOWN.json](../budget-calibration-20261015/budget-diagnostics/V173_COMPLETED_PERFORMANCE_BREAKDOWN.json)。


### 2026-10-17 10:40 UTC：三档 B3 图完整结果，继续优化 draft

V180 已完整完成并通过原审计及独立终态复核。原463泊松到达、256输出、SLO、充时口径和active32不变；当前全527范围B4组件、新三档图的102次协议cycle/28次next、两轮完整zero65及封图后的full463均通过，源码前后未变，worker正常退出、四卡清理。

| 方法 | Raw token/s | Goodput token/s | 完整耗时 s | tight 达标 | normal 达标 | loose 达标 |
|---|---:|---:|---:|---:|---:|---:|
| V176：cap3，原 target 布局 | 551.00 | 224.92 | 215.11 | 6/271 | 87/96 | 96/96 |
| V180：cap3，三档紧凑 B3 图 | 548.16 | 223.76 | 216.23 | 6/271 | 87/96 | 96/96 |

V180逐请求TPOT总体中位数54.9001ms、P90 66.9967ms；TTFT中位数40.0953秒、P90 77.9150秒。真实2197/2618次target调用采用新三档布局。六个局部代表例中，含原padding、metadata、图执行及完成等待的单次target调用中位数降低约8.5%–12.9%；但完整测量没有改善。

角色HOST边界计时揭示了后续方向：rank1 target累计133.435→119.857秒，而exchange累计13.426→27.615秒；rank0 draft累计131.743→133.932秒，实际draft提交7382→7397次，Eager生成24817→24948个。该现象支持target变快后等待draft的瓶颈转移，不能据此宣称已证明每一项等待的因果来源，也不能把不同角色的HOST耗时相加当成设备计算量。当前最好树形完整结果仍为V173的291.0907，V180不采用，目标未达成。

证据：[V180独立终态审计](../budget-calibration-20261015/budget-diagnostics/V180_INDEPENDENT_TERMINAL_FULL463_AUDIT.json)、[完整性能拆解](../budget-calibration-20261015/budget-diagnostics/V180_COMPLETED_PERFORMANCE_BREAKDOWN.json)。后续V181已从固定源码启动，验证parent3/child3三步路径中repair与frontier合并；原capture/commit、当前数学、真实正负提升/消费/下一轮、warm/两zero/seal/full463均重新执行。V182准备紧凑target与该draft优化的同进程组合；V183准备持久metadata缓冲。它们尚无新完整性能结果。


### 2026-10-17 11:02 UTC：V181 独立参考的设备索引错误

V181已越过原201+44/114初始化和当前原B4组件，在新增第一个N1/E1/R1例的独立元数据检查中失败：`pure181.independent_frame`把NPU索引用于CPU `context_lens` 的 `index_select`，四rank共同拒绝继续。此轮未完成新数学/协议资格，也未跑full463，没有新Goodput；不能从这次检查器错误判断生产优化有效或无效。作业实际1072.03秒以1退出，并未触发7200秒超时，外围通用child-success断言不应被解释为超时。源码前后未变，worker已关闭、自有进程无遗留、四卡空闲，原10496文件/15外部文件/212包前后校验通过。

原181和已封存182均保持原字节。182复用同一独立参考，已禁止启动；新187仅修正每个被置换张量按自身device构造索引，并增加旧函数必败的混合设备HOST回归后再审查和重跑。183持久metadata候选已独立审查并封存，但尚未运行，当前优先获得draft路径的完整结果。185仅预填充凑批候选仍在源码准备，尚无性能结论。

证据：[V181失败及清理记录](../budget-calibration-20261015/budget-diagnostics/ROOT_V181_ACTUAL_ORACLE_MIXED_DEVICE_INDEX_FAILURE.json)。


### 2026-10-17 11:16 UTC：索引修复版187启动，预填充与 eager 瓶颈继续核查

187已独立审查并冻结，11:11 UTC启动新进程，当前仅处于设备初始化与原精确预填充图库存准备阶段，尚无新完整Goodput。相对失败181，模型相关改动仅是独立oracle为各张量在其自身设备上构造重排索引，保留CPU context_lens；生产解码、预算与时钟不变。冻结SHA为`4617c299705f0cfea149594adc360a69d31402798b3e926b4af596747d89389d`。181原失败和已完成B4证据另经独立终态审计，原文件保留。

已完成的173/180和固定γ4均关闭SLO-aware admission和target-SLO-first。固定γ4真实使用512-token分块及staged/mixed预填充；当前树版本在decode间同步执行完整prompt。173/180分别记录427/412次whole-prefill，leader HOST helper累计48.907/50.696秒；这些HOST累计时间不能直接从整轮wall扣除。预填充185候选尝试实际空位出现后按4请求/150ms门限合批，源码及4项HOST就绪，尚待独立审查和当前设备数值验证；该门限不是TTFT上限，也不修改到达时间或TPOT定义。

173的eager=0并非预算为0：18532次父B4行被only-B3条件排除，余55行父B3又被共享四步链条件推迟。176/180恢复E3后分别生成24817/24948行，晋升10471/10500，约58%失效；该比率是提前草稿行的失效率，不是token错误率。cap4+shifted E3仅完成只读可行性分析，尚未实现或测量，不能宣称会提升Goodput。


### 2026-10-17 11:35 UTC：187第二处观察器错误；下一主测选择cap4合批190

187在1089.732秒后退出1，四个worker均关闭、无残留进程，原基线pre/post10496文件、15外部文件及212包均通过。独立重开原B4的124份raw审计通过（96个协议cycle、24次next、336个零计数差分）。新路径首代表流程走到第二代表，但没有保留完整单代表数值资格文件；N1/E3/R1内部padding用例在真实union hidden返回并clone后的诊断观察器失败：`observer187.inspect_indices`仍用NPU索引重排CPU `context_lens`。这与上一处独立oracle错误是不同调用点；不是超时，也没有证据表明union模型数学不一致。没有进入自然晋升、warm/zero/seal或full463，因此没有新Goodput。冻结187/181及全部失败证据保持原样，另开188修复观察器，当前不优先重跑。

下一主测190以已完成的较快173为生产父源，保留高并发normal cap4/低并发真实B5、active32、C96/E96/global192和N16因果M80/S16；只加入已审185的空位合批4/150策略。新增2/4-owner完整prompt、同N首token、原真实callback及真实四步next的独立hidden/KV/encoded8/ledger验证已完成源码和HOST准备，当前等待最终独立审查、ROOT冻结和设备验证，性能未知。只开启SLO准入排序的189已独立审查并冻结（253个源码归档成员），尚未上卡。更小M197、cap64并按实际空间裁减的真实chunk游标版本192在准备源码，尚无设备或性能结论。


### 2026-10-17 12:07 UTC：cap4 合批190完整测量；改测真实小块预填充192

190已完成原463请求、每请求256输出、原到达/SLO/Native计费时钟的完整测量。当前完整B4、M80与新增2/4-owner whole-prompt/同N首token/真实callback/nextB4数学及状态检查通过；两轮完整zero65、封图、full463和原guardian审计均通过，源码未变、worker关闭、四卡清理，原baseline pre/post通过。独立终态复核正在进行，尚不宣称全负载独立AR等价。

| 方法 | Raw token/s | Goodput token/s | 完整耗时 s | tight 达标 | normal 达标 | loose 达标 |
|---|---:|---:|---:|---:|---:|---:|
| 173：cap4，原准入 | 635.73 | 291.09 | 186.44 | 21/271 | 95/96 | 96/96 |
| 190：cap4，空位4/150合批 | 581.79 | 221.16 | 203.73 | 9/271 | 71/96 | 96/96 |

190的逐请求Native TPOT中位数50.6046ms、P90 60.5220ms；TTFT中位数38.7176秒、P90 68.9185秒。该单轮Goodput低于173，不采用；不能由组件通过或合批机制本身推断性能改善。客户端首尾事件墙钟另算的Goodput为216.1295，不与原Native口径混用。

证据：[190完整结果与清理记录](../budget-calibration-20261015/budget-diagnostics/ROOT_V190_CURRENT_FULL463_RESULT_AND_HANDOFF.json)。12:07 UTC启动192原自持锁guardian，优先验证真实64-token分块游标和M197/S19或S35：每块P=min(remaining,64,197-Q-2×scratch_count)，FULL mask仍为128；active+reserved上限32、原全局预算192、到达/输出/时钟不变。独立历史、四个当前交易、原commit/最终一次发布/后续draft、两完整zero和full463均须新进程通过。192只有源/HOST资格，尚无设备或性能结果；其实际继承日程是单warm65→两zero65→seal/full，不是最多八轮预热。191仅SLO准入排序版也已封存，暂未上卡。


190独立终态审计已通过，见[190独立审计](../budget-calibration-20261015/budget-diagnostics/V190_INDEPENDENT_TERMINAL_FULL463_AUDIT.json)。它逐项重开271个源码归档成员及当前raw，复核原pre/post10496文件/15外部文件/212包、全部worker退出与四卡空闲，并重新执行原full463/256输出及当前组件、两zero、准入事件审计。与173的完成轨迹对比，预填充427→284次，但leader HOST helper累计48.907→52.170秒；853次deferred，释放原因275次timeout、8次size、1次empty，cohort大小分别为153×1、91×2、32×3、8×4。M80使用1874→611次，占比73.289%→23.312%；旧M128/S38调用550→1891次，padding物理行32719→104008。target累计HOST120.260→131.508秒，draft提交5083→5202次。这些是完成轨迹的实际变化，不能把不同角色HOST时间相加当纯设备时长或断言单一因果。


### 2026-10-17 12:42 UTC：后续小图候选封存与统计口径澄清

194已完成源码、4项HOST检查及独立审查，并封存516个选定归档成员，尚未上卡。它以190为唯一生产父源，只为真实N16、Q70～79、连续线性B1～4请求增加M84/S18档位，避免这些请求落入原M128；Q80仍使用原M80/S16，预算与4/150准入不变。新84图的独立当前模型检查、完整重放和full463必须实际重新通过，不能由源码就绪推断性能。整体跨形状兼容性同时取决于原80和新84的结果。193的融合裁决候选也已封存，尚未上卡；它以173为父源，不能把两者当成已合并的优化版本。

更正前文的事件名称：190的216.1295来自rank1 Native worker的原`on_token_commit`回调，由worker内`perf_counter`记录，并非main或外部客户端收到token的时刻。它应称为**worker提交事件Goodput**；数值保持不变，与原Native计费Goodput 221.1558分别报告。历史固定γ4的501.6356采用Native计费TPOT；官方TP4的429.8443来自engine-core首尾token时间戳，且原harness使用FINAL_ONLY，没有保存外部客户端逐token事件。二者不是已统一的时延观察点，不能据此直接宣称最终1.5倍提升。501.6356与644.7664继续作为历史筛选门槛；最终比较仍需相同卡、相同原始到达、明确统一观察点的新三组交替测量，并分别保留原生口径。

另更正到达时间的解释：固定γ4结果中第一个`arrival_offset_sec=0`是输出时减去最早到达时间的展示归一化。实际提交仍使用原manifest offset，首个为0.46515177776630584秒；不能由展示中的0推断它换了泊松到达序列或时钟起点。现有历史文件保持原样，以这次源码核查作补充说明。

证据：[194 ROOT源码审查](../budget-calibration-20261015/budget-diagnostics/ROOT_V194_CAUSAL84_VARIABLE_Q_SOURCE_AND_GUARD_REVIEW.json)、[194封存复核](../budget-calibration-20261015/budget-diagnostics/ROOT_V194_PREPARED_SOURCE_HOST_AUDIT.json)、[后续三组对照与评分桥接规范](../budget-calibration-20261015/budget-diagnostics/final-pairing-scoring-bridge-v2/FINAL_NINE_RUN_SCORING_BRIDGE_SPEC.json)。当前192仍在NPU进行长prompt分块的当前数学与状态资格检查，尚未产生新full463结果。

上述时钟名称以后续[观察点与现成固定γ4 witness复用核查](../budget-calibration-20261015/budget-diagnostics/final-pairing-scoring-bridge-v2/STREAM_OBSERVATION_EDGE_AND_EXISTING_FIXED_WITNESS_REUSE.json)为准：fixed与树可复用同一原Native worker提交回调；官方TP4应额外保留原engine-publication时间戳和原始token事件，不能把HOST接收时刻当成engine时间戳。现成fixed witness的13个静态源、8份HOST证据及21个归档成员已只读核验，尚不是新完整测量。三者的外部最终批次完成耗时可另行观测；FINAL_ONLY下没有官方客户端中间token流，不能构造零TPOT或据此判SLO全部通过。

已完成轨迹的独立批次核查如下。请求轮数指一个真实请求参与一次target验证，排除scratch及首token；平均批次=请求轮数/调用次数。

| 已完成方法 | target调用次数 | 真实请求轮数 | 平均真实批次 | 超过16请求的调用 |
|---|---:|---:|---:|---:|
| 固定γ4 | 1932 | 43221 | 22.37 | 混合调用未保留完整逐次真实N，不能用物理bucket代替 |
| 树173 | 2557 | 38711 | 15.14 | 0 |
| 树180 | 2618 | 43018 | 16.43 | 1566 |
| 树190 | 2621 | 39269 | 14.98 | 0 |

173的请求轮数比固定γ4少10.44%，调用次数却多32.35%，因此下一优先候选为已有冻结171：测试active64、normal128/eager64/global192能否实际形成home32及M165大批次。171未叠加173的M80，小批次仍走旧mapper，结果不能拆分归因于单项参数。180虽然已出现更大批次，但每请求轮交付token下降，完整性能仍更低，故“大批次一定更快”尚未成立。192必须先完成并独立审计、清理后才能启动下一轮；若其结果改变瓶颈，再调整顺序。

每请求轮交付量也要注明bonus：173实际为3.04991，固定γ4为2.72974，但树含主链全接受后的额外target token，固定版本关闭该项。仅对173同一轨迹扣除可见的主链额外token后为2.71225；这个算术视角不等于关闭bonus重跑。173/180/190中B5仅占约1.2%～1.6%的请求轮，现有全量日志不足以重建同前缀B5对B4的分支因果增益，不能把表面交付量差异全部归功于树。

证据：[下一次设备实验决策](../budget-calibration-20261015/budget-diagnostics/NEXT_NPU_INFORMATION_GAIN_AFTER_192_COMPLETED_EVIDENCE_DECISION.json)、[批次、bonus与输出截断的独立核查](../budget-calibration-20261015/budget-diagnostics/COMPLETED173180190_BATCH_OWNER_AND_BONUS_ACCOUNTING_READONLY_DIAGNOSIS.json)。


### 2026-10-17 13:01 UTC：已保存独立AR参考的实际反例与测量工具就绪

只读重开V85的两条完整独立own-AR参考（每条256 token、三TP检查点一致、从独立prompt和自身上一预测递推），与已完成173/190及固定γ4的对应原prompt输出逐token核对。两条prompt token hash、模型/词表/贪心/忽略EOS/256输出、核心Native及原基线锁身份匹配；旧参考与树候选的完整配置仍有maxseq32→64、timeout1800→6000差别，完整历史继承环境没有单独dump，不能扩大为任意环境的golden。首分歧如下，位置从0计数。

| 已完成候选 | 原请求0首分歧 | 原请求1首分歧 | 与这两条canonical AR输出完全一致 |
|---|---:|---:|---|
| 固定γ4（原501.6356轮） | 148 | 34 | 否 |
| 173 | 137 | 34 | 否 |
| 190 | 205 | 199 | 否 |

这已是针对既定direct eager单query PA参考的明确反例，而非“尚未做任何输出对照”。参考执行方式与候选batched graph/FIA及物理M不同；目前没有证明哪一路更准确、same-M实现错误或具体差异根因，不能仅用“低精度”解释全部差异。当前结果继续不采用，whole-workload AR资格为False。有限same-M hidden/head/KV、原提交与重放检查不能替代完整独立AR等价；后续共同前缀诊断仅用于定位，也不能替代自身递推的全量对照。192当前输出尚未用于这一对照。

固定γ4原result也实际保留463×256个无损token IDs；全输出hash及首行副本已重核。固定全prompt摘要与173保存的原prompt列表一致，前两prompt分别匹配V85参考；原HEAD加已保存patch重建的6个核心/benchmark源与锁定源一致。固定串行版本同样有上述逐点差异，说明这类差异并非只在树中被观察到；这仍不能证明全部差异属于同一原因、排除缓存问题或证明任意数值模式更准确。

证据：[两条旧独立参考与173/190的实际核查](../budget-calibration-20261015/budget-diagnostics/V85_TWO_OWN_AR_REFERENCES_VS_COMPLETED173190_BOUNDED_REUSE_AUDIT.json)、[固定γ4无损输出与同一参考的核查](../budget-calibration-20261015/budget-diagnostics/FIXED501_TWO_LOSSLESS_COMPLETIONS_VS_V85_OWN_AR_BOUNDED_AUDIT.json)、[完整463独立AR的复用边界与最小资格计划](../budget-calibration-20261015/budget-diagnostics/WHOLE463_INDEPENDENT_AR_GREEDY_REFERENCE_REUSE_AND_MINIMUM_QUALIFICATION_READONLY_PLAN.json)。

TP4原engine-publication事件witness的7个静态源、14个只读依赖、39份HOST证据及46个归档成员，已通过独立peer和ROOT源码复核。保留FINAL_ONLY及原时钟；实际新运行须逐请求验证全部463×256 raw事件重建，以及首尾时间戳严格等于原metrics。它只有源码/HOST资格，尚未绑定新实际外层launcher或运行模型，不能转写成性能/设备资格。见[ROOT源码复核](../budget-calibration-20261015/budget-diagnostics/ROOT_FINAL_OFFICIAL_PUBLICATION_WITNESS_SOURCE_HOST_REVIEW.json)。

后续评分桥接的171选择器另有一处已在实际运行前纠正：应选择window64、调用`audit171.audit(directory,(64,))`；原v2及其生成脚本写成32的字段保留并停用，冻结171本身无改动。192/193/194的active32选择器已逐源复核。见[选择器纠正记录](../budget-calibration-20261015/budget-diagnostics/FINAL_PAIRING_CANDIDATE_SELECTORS_SOURCE_CORRECTION_REVISION1.json)。


### 2026-10-17 13:34 UTC：192完整负结果；启动冻结active64候选171

192于13:29:22 UTC完整结束，guardian状态complete/`passed_full`，总作业4877.76秒。原B4与四个当前197交易、三遍完整历史（含2301-token的36分块连续游标）、当前独立hidden/head/KV与原提交检查通过；warm65、两zero65及sealed full463均执行。原基线pre/post10496文件/15外部文件/212包通过，源码未变，worker全部关闭、自有进程无遗留，四卡空闲。组件通过没有带来端到端性能提升。

| 已完成方法 | Raw token/s | Goodput token/s | 完整负载耗时 s | tight达标 | normal达标 | loose达标 |
|---|---:|---:|---:|---:|---:|---:|
| 当前最好树173 | 635.73 | 291.09 | 186.44 | 21/271 | 95/96 | 96/96 |
| 小分块197树192 | 161.53 | 3.49 | 733.78 | 0/271 | 0/96 | 10/96 |

192逐请求Native TPOT中位数190.7063ms、P90 243.3710ms；worker首提交回调TTFT中位数260.0016秒、P90 539.1001秒。实际有1247次normal target、1322次mixed target；性能退化的具体来源待分项核查，不能仅凭较小M、调用次数或组件检查通过推断有效吞吐。本版本不采用，最好完整树结果仍为173的291.0907；目标未达到，whole-workload AR仍未通过。

证据：[192完整结果及清理记录](../budget-calibration-20261015/budget-diagnostics/ROOT_V192_CURRENT_FULL463_RESULT_AND_HANDOFF.json)。独立终态审计最终REV4已通过：重开225归档成员，重新执行原audit并与completed/supervisor核对；复算387次原Native整数裁决、11286个请求轮，以及672个组件和252个后续零差分计数。三次旧auditor适配错误（CPU字段、draft/target角色字段、二维整数carrier切片）及失败结果全部保留；修订仅涉及HOST审计脚本，四次检查的原始raw manifest相同，模型作业始终complete。见[192独立终态审计REV4](../budget-calibration-20261015/budget-diagnostics/V192_INDEPENDENT_TERMINAL_FULL463_AUDIT_REVISION4.json)。这不授予whole-workload AR资格。

13:33 UTC，ROOT通过原冻结177存储修复入口启动171，唯一NPU会话94519。它采用active64、normal128/eager64/global192、target单轮上限32；当前只是启动检查/模型初始化阶段，尚无新性能结果。高预算C97～128的新M165/S34或S36路径须在本次新进程重新完成独立数学、原提交、next及两zero检查，再跑原463请求。171不含173的M80，不能将其结果作为仅改active参数的消融。192已终态、清理及原始校验通过，且独立来源和完整性能复算已通过；剩余附加HOST审计与新候选独立进行，不传递192模型资格。


### 2026-10-17 13:44 UTC：192退化拆解与独立输出诊断封存

192与173的target调用2569/2557、请求轮38664/38711及实际接受收益接近；实际物理target查询量却为411830/225840，即1.824倍。192不含173的M80，另有1153次旧M128/S38及1322次M197混合事务；185个mixed prompt被切成1322块，触发更多原draft prompt forward、四rank状态共识和显式提交fence。rank1 target累计HOST时间120.26→499.12秒，rank0 draft累计94.55→484.42秒；不同角色不能相加，缺少逐family/任务更新时间拆分，现有数据不能归因于单一MC2、通信或硬件问题。见[192已完成性能拆解](../budget-calibration-20261015/budget-diagnostics/V192_COMPLETED_PERFORMANCE_BREAKDOWN.json)。

共同前缀诊断已由ROOT和独立peer完成源码复核，封存114个选定源码/证据成员，尚未上卡。它比较四个已知首分歧位置的独立own-AR、相同前缀PA、eager FIA与自然graph FIA，并保留完整80行LM head后再选root的实际形状及top2分数。它仅用于定位数值与执行路径差异，不能代替完整463条自身递推AR对照，也不授予生产或性能资格。退出信号清理问题已修复并通过3项新HOST检查，原4项HOST证据未重跑。见[ROOT诊断源码复核](../budget-calibration-20261015/budget-diagnostics/ROOT_CANONICAL_AR_COMMON_PREFIX_DIAGNOSTIC_REV3_SOURCE_REVIEW.json)。当前171仍是唯一NPU作业，原constructor预热尚未结束。


### 2026-10-17 13:59 UTC：rolling-eager零生成的门槛与g曲线实测边界

173的rolling continuation生成、消费均为0，但2557次正常target均保留early submit，不能据此说正常双batch没有重叠。原前置条件后有18532次parent B4机会，全部被仅允许parent B3的资格门槛排除；余下55行B3也因同批存在四步normal而被shared4保护全部延期。30次paid-probe offer说明不能归因为整轮始终无窗口。开放B4/shared4需要新的真实前缀、target裁决、KV、promotion/失效/回滚和own-next验证，直接删除过滤条件不能授予资格。

当前动态B确实执行SLO缺口→最小可行n→tight/normal/loose优先分配，但之后的正收益heap常把候选填到cap4/低N cap5。g的单位是每轮有效交付token，含纠正和全接受bonus；它没有除以耗时，heap也没有计算整批bucket、shared steps、repair的真实延迟成本。减少B同样可能更慢：N16/Q80命中M80，而Q77会落回M128。

旧pilot对三类各12个同前缀context、每个B1～8分别重新prefill/draft/target/verdict，当前取其中B1～5；并非由一次完整树简单截断。173完整负载在非终端B4上获得更多在线样本，编码类的预测3.8333更新到3.0840。B1/B2尚无当前非终端样本，不能把其旧prior称为完整负载实测曲线。

| 请求类型 | pilot g(B1) | pilot g(B2) | pilot g(B3) | pilot g(B4) | pilot g(B5) | 173最终B4后验 | 当前B4非终端样本数 |
|---|---:|---:|---:|---:|---:|---:|---:|
| tight/编码 | 1.9167 | 2.7500 | 3.3333 | 3.8333 | 3.9167 | 3.0840 | 21897 |
| normal/聊天 | 1.9167 | 2.5833 | 3.2500 | 3.8333 | 4.0833 | 3.6412 | 6526 |
| loose/摘要 | 1.6667 | 2.1667 | 2.5000 | 2.6667 | 3.0000 | 2.6094 | 9097 |

后续两个独立可测方向是：严格新资格下的parent4→child3/shared4小规模eager，以及保留进度fallback的“只分最小所需B”对照；二者均尚无性能结论。当前171先完整测完，195仅并行准备171大批次与173精确M80的源码整合，不叠加上述策略，也不会自动运行。证据：[rolling与预算的只读核查](../budget-calibration-20261015/budget-diagnostics/ROLLING_E0_AND_CURRENT_DYNAMIC_B_COMPLETED_EVIDENCE_READONLY_DIAGNOSIS_REVISION2.json)。


### 2026-10-17 14:05 UTC：171完整结果——吞吐提高，SLO退化

171已完成原463×256输出、当前B4与new165组件、warm65/两zero65/sealed full，guardian complete/passed_full；pre/post锁定校验通过，源码未变，worker退出/自有进程清理/四卡空闲。独立终态HOST审计已按固定receipt SHA授权，正在执行。

| 完整候选 | Raw token/s | Goodput token/s | 完整负载秒 | tight达标 | normal达标 | loose达标 |
|---|---:|---:|---:|---:|---:|---:|
| 最好树173 active32 | 635.73 | 291.09 | 186.44 | 21/271 | 95/96 | 96/96 |
| 171 active64/C128 | 749.07 | 169.88 | 158.23 | 2/271 | 7/96 | 96/96 |

171确实实际达到active64，1402次target中1182次真实N>16，平均真实批次27.5927；目标调用次数减少但逐请求SLO变差。本版不采用，最好完整树Goodput仍为291.0907，目标未达到。逐请求Native TPOT中位数76.5266ms/P90 92.1435ms；TTFT中位数10.7134秒/P90 24.4057秒。四rank实际页峰[449, 448, 448, 448]，pool544，最终引用页均为0；这是本轮实际覆盖，不代表任意64长请求都装得下。

171仍不含173精确M80，且两者并发/预算同时变化，不是单项消融。下一步先运行已冻结共同前缀输出诊断；195整合源仅并行准备，是否上卡依据本轮分布和诊断决定。证据：[171完整结果与清理记录](../budget-calibration-20261015/budget-diagnostics/ROOT_V171_CURRENT_FULL463_RESULT_AND_HANDOFF.json)。


171独立终态审计一次通过，见[171独立审计](../budget-calibration-20261015/budget-diagnostics/V171_INDEPENDENT_TERMINAL_FULL463_AUDIT.json)：140源码归档成员与177 overlay、原audit/completed/supervisor一致、72次Native整数裁决/2304请求轮、336组件零差分、两zero及full463时钟/无损输出/page和清理均通过。这个结果依然是性能负结果，不采用。14:07 UTC在171清理和独立终态审计后，ROOT启动唯一NPU会话80825运行已冻结共同前缀诊断；这不是新的Goodput测量。


### 2026-10-17 14:18 UTC：共同前缀诊断完整通过；启动193融合裁决对照

诊断完整作业310.2075秒，原基线pre/post校验通过，源码未变、worker全部退出、自有进程清理、四卡空闲；独立终态审计一次通过。下表是四处既定共同前缀下的预测token ID，所有own-AR均从原prompt按自身预测递推，并复现旧reference至该位置；三个TP观测一致。

| 诊断路径 | 173请求0/137 | 173请求1/34 | 190请求0/205 | 190请求1/199 |
|---|---:|---:|---:|---:|
| 独立逐token own-AR | 382 | 7629 | 576 | 382 |
| 重新一次执行whole-prefix PA | 13 | 525 | 576 | 382 |
| prefix-minus-root再单root PA | 13 | 525 | 3411 | 382 |
| 真实N16/M80 eager FIA | 382 | 525 | 3411 | 382 |
| 自然graph hidden首次/恢复 | 382 | 525 | 3411 | 382 |
| 自然graph greedy首次/恢复 | 382 | 525 | 3411 | 382 |
| 历史生产候选该位置 | 13 | 525 | 3411 | 13 |

本次四处eager FIA与graph hidden首轮全部16个root一致；改变owner0输入后预测变化，恢复后完整80个greedy token恢复一致。纯PA前缀重建也能产生分歧，因此不能只归因于树裁决或图重放；这仍未证明某个具体算子、精度或KV根因。该cohort是合法重建，历史packed proposal与完整KV构造历史未保留，不能把本表当历史逐call重放；尤其第1、第4处生产token未由此cohort复现。四处诊断不能替代全部463条自身AR等价。证据：[ROOT诊断结果](../budget-calibration-20261015/budget-diagnostics/ROOT_CANONICAL_COMMON_PREFIX_CURRENT_RESULT_AND_HANDOFF.json)、[独立审计](../budget-calibration-20261015/budget-diagnostics/CANONICAL_COMMON_PREFIX_INDEPENDENT_TERMINAL_AUDIT.json)、[四案例原始对比](../budget-calibration-20261015/budget-diagnostics/CANONICAL_COMMON_PREFIX_FOUR_CASE_RETAINED_COMPARISON.json)。

14:15 UTC，ROOT在上述诊断清理及独立审计后启动唯一NPU会话89848测试已冻结193：以最好173为唯一父源、active32不变，仅在exact N16/B4/Q80/M80/S16上用原Triton融合BONUS裁决替代HOST分散裁决，B5/异构等保留原路径。当前仍在模型初始化；当前算子、真实consume/fallback、同M数学/提交/ownnext、两zero及full463须重新通过。195整合源与最终9轮执行层仅完成source/HOST及独立peer，ROOT审查待做；最终执行层tree adapter在源码中硬禁用，不能由JSON启用。


### 2026-10-17 14:28 UTC：单变量后续对照与source-only整合封存

171与173已完成轨迹中，各类请求相邻target参与间隔基本均为全局call数2；171各类callback burst平均约234ms，173约145ms。Loose仍在150ms阈值内，tight却普遍超过40ms；当前没有把loose松弛变成tight额外服务频率的可见证据。两轮decode_timeline均未保存每请求CPU时序，不能将整体HOST时间直接归因于某一请求或某次prefill。SOURCE核查中TARGET_SLO_FIRST影响ready packing，burst按home而非按class；当前tree没有传入generic线性路径的goodputEDF/remaining-service开关。现成189仅开SLO admission、target0/coalesce1-0/其余173不变，是193结果不改善后的优先单开关对照，尚未上卡或证明收益。见[服务频率与最小SLO调度核查](../budget-calibration-20261015/budget-diagnostics/V171_V173_CLASS_SERVICE_AND_MINIMUM_SLO_SCHEDULING_READONLY_DIAGNOSIS.json)。

195已经ROOT+peer源码复核并封存240个选定成员，三份父源码档案P148/P160/P171逐成员原位验证，未复制父binary archive；当前仍未上卡。它只给171补exactN16/Q80的M80，小批次路径在本轮171仅14/1402次，不能认为这一整合足以修复大N SLO退化。其当前165/80组件、三proof六entry、两zero及完整负载须另新进程通过。见[195 ROOT封存](../budget-calibration-20261015/budget-diagnostics/ROOT_V195_PREPARED_SOURCE_HOST_AUDIT.json)。ROOT首个准备辅助脚本遗漏projection显式appended_text而在冻结前拒绝；辅助脚本和拒绝记录保留，补全该声明重构后通过，候选源未改。

最终9轮执行层也已ROOT+peer完成当前硬禁用scope的源码/HOST复核，105个归档字节成员验证通过；它仍缺真实赢家tree adapter及新proofschema，既有源码和JSON都不能启用模型。原三组对照尚未执行，未产生新baseline或1.5倍结论。见[ROOT source-only审查](../budget-calibration-20261015/budget-diagnostics/ROOT_FINAL_NINE_RUN_SUPERVISOR_BINDER_DISABLED_SOURCE_REVIEW.json)。


### 2026-10-17 14:40 UTC：193组件失败并清理；189单开关对照开始

193在M80当前流程pass0/cycle0、融合BONUS算子验证前触发同设备ABI检查错误，四worker一致失败退出。它没有运行full463，没有Goodput结果；不能把失败填成0，也不能据此评价融合裁决的性能。ROOT保留原B4证据、四rank错误和M80 partial；原基线pre/post校验、源码不变、自有进程清理、四卡空闲均通过。实际子进程exit1，guardian的“within7200”错误文字不是超时证据。独立失败审计已授权。见[193失败记录](../budget-calibration-20261015/budget-diagnostics/ROOT_V193_CURRENT_COMPONENT_FAILURE_AND_HANDOFF.json)。

只读源码发现`engine.device`是未指定index的`npu`描述符，而真实输入和输出张量位于带index设备；当前合取式直接比较三者，会拒绝合法真实张量。修复将使用新版本，193冻结源和失败证据保留。

ROOT完成189当前253成员源码、HOST保留证据和归档预检后，于14:39启动唯一NPU会话80092。189以173为唯一生产父源，只打开SLO优先接纳，target SLO packing仍关、coalesce保持1/0，active32及原到达/负载/输出要求不变。当前模型初始化尚未确认，尚无新性能结果。共同前缀历史隔离v2已完成源码/HOST及独立peer审查，尚未ROOT冻结或上卡。

193[独立失败审计](../budget-calibration-20261015/budget-diagnostics/V193_INDEPENDENT_TERMINAL_FAILURE_AUDIT.json)已通过，377归档成员与当前source字节、四rank失败共识、原B4通过以及清理均核实；没有新的完整负载或性能结果。共同前缀历史隔离v2已ROOT复核并封存92选定成员，仍未上卡：它将让16个owner独立逐token递推，保留相同过去KV，比较不同attention与同hidden不同LM-head行数，以缩小四个已知分歧的来源范围；不能作为新的生产资格或Goodput。见[v2 ROOT源码复核](../budget-calibration-20261015/budget-diagnostics/ROOT_CANONICAL_AR_PREFIX_HISTORY_ISOLATION_V2_SOURCE_REVIEW.json)。


### 2026-10-17 14:50 UTC：completed轨迹给出的prefill等待边界

173的427次prefill中426次有旧active，包含12744次“旧请求经历一次prefill”成员关系；171对应383/384次、22313成员关系。原leader累计refill HOST分别48.9072/52.5306秒，这些HOST累计时间不能直接当NPU kernel时间或从新实验wall扣除。

ROOT利用原同步helper位于上次token callback之后、本组root callback之前的源码顺序，将每次helper耗时约束在对应callback间隙内，并保留原累计refill时间。逐请求活动成员关系给出以下贡献边界；数值是各请求边界的中位数，不是精确逐次耗时，也不代表所有边界能同时取到。

| 已完成版本/类型 | 原Native charged TPOT中位数(ms) | prefill对该TPOT贡献的下界中位数(ms) | 上界中位数(ms) |
|---|---:|---:|---:|
| 173 tight | 46.37 | 5.99 | 12.56 |
| 173 normal | 39.69 | 4.25 | 10.81 |
| 173 loose | 54.43 | 8.34 | 14.90 |
| 171 tight | 77.12 | 21.36 | 27.51 |
| 171 normal | 64.55 | 17.26 | 23.41 |
| 171 loose | 89.99 | 25.46 | 31.61 |

这支持进行“active整组完成后再接纳”的单变量对照；并非预测将表中等待移除后就能达到某个Goodput。新的多owner prefill、缩小尾批、padding、TTFT和总wall都会变化，候选197正在独立源码/HOST准备，尚未上卡。189仍为唯一NPU作业。证据：[条件时序边界及复算脚本](../budget-calibration-20261015/budget-diagnostics/COMPLETED_REFILL_CHARGE_CALLBACK_GAP_BOUNDS_ROOT.json)、[最小cohort可行性核查](../budget-calibration-20261015/budget-diagnostics/EMPTY_ACTIVE_COHORT_WHOLEPROMPT_PREFILL_MINIMUM_READONLY_FEASIBILITY.json)。


### 2026-10-17 15:04 UTC：当前完整测量与后续独立候选

189同进程原B4、M80组件及warm/两zero已完成，sealed full463正在运行，暂无终态Goodput。196设备描述符修复已ROOT+peer源码复核并封存602选定成员，3项新HOST和旧193拒绝回归、15类错误设备/类型/形状/mask反例均通过；它尚未上卡，不继承193失败为数值或性能资格。见[196源码封存](../budget-calibration-20261015/budget-diagnostics/ROOT_V196_PREPARED_SOURCE_HOST_AUDIT.json)。

197整组drain后接纳以173为唯一生产父源，只在原admit_available接收控制消息后增加active非空即返回；原到达、过滤、即时root/token发布、TTFT/TPOT/全wall、463×256和target≤16均保留。它已通过4项新HOST，独立peer收尾中、ROOT审查中，尚未冻结或运行。新增当前模型门会代表性验证2/32多owner原Native eager prefill的完整hidden、56draft/128target queryKV、同N首root和真实next；原planner只选N1/16下一批，额外组件不submit target，原173本engine完整target裁决/commit/ownnext门依然必跑。不声称全部prompt/cohort组合穷举。

树形与固定501的[草稿工作核查](../budget-calibration-20261015/budget-diagnostics/V173_VS_FIXED501_FASTV5_SPINE_AND_DRAFT_WORK_SOURCE_DIAGNOSIS.json)确认173的B4占37713/38711请求轮，预算≤4固定选spine、B5才添旁枝；实际2557次chain及2526次packed root seed，包含13117个已提交后缀repair query。当前confidence还影响E预测，不能直接删softmax/top2。树路径总提交path+frontier，linear bonus配置不控制它；尝试无BONUS需要新提交合同、g与分支/后续状态验证，尚无该消融结果。原Native具有M80/S16/full9 greedy key，但这不授予新B4分叉或hidden诊断的当前资格，真正分叉还须独立数学/commit-next/g与成本实验。


### 2026-10-17 15:07 UTC：189完整结果小幅提高；继续197

189完成同进程原B4/M80组件、warm与两zero、sealed full463，guardian complete/passed_full，原基线pre/post通过、源码不变、worker闭合、自有进程清理、四卡空闲。独立终态审计已按固定receipt SHA授权，正在复核。

| 完整候选 | Raw token/s | Goodput token/s | 完整负载秒 | tight达标 | normal达标 | loose达标 |
|---|---:|---:|---:|---:|---:|---:|
| 173原最好树 | 635.73 | 291.09 | 186.44 | 21/271 | 95/96 | 96/96 |
| 189 SLO优先接纳 | 630.16 | 299.43 | 188.09 | 34/271 | 90/96 | 96/96 |

本轮Goodput提高约2.86%，是单次观测；Raw略低，尚未超过固定γ4的501.6356，也没有达到较强MC2 TP4的1.5倍。Native charged TPOT整体中位45.6866ms/P90 60.3845ms；三类中位分别45.1587/40.9538/60.0688ms。TTFT中位1.7026秒、P90 89.7427秒，原到达和worker首root时钟均保留，不能隐藏排队代价。未采用、whole463 own-AR和正式1.5倍资格仍为False。证据：[189完整结果](../budget-calibration-20261015/budget-diagnostics/ROOT_V189_CURRENT_FULL463_RESULT_AND_HANDOFF.json)。

197已ROOT+peer源码复核并封存294选定成员，原父归档逐成员原位核验，未重旧HOST套件。ROOT于15:06:50启动唯一NPU会话93758；它保留173原生产预算/拓扑，只测active整组结束后再接纳的独立变化，并重新执行新增多owner prefill数值门、原target协议、warm/twozero/full463。此时模型初始化尚未确认，暂无新结果。见[197 ROOT封存](../budget-calibration-20261015/budget-diagnostics/ROOT_V197_PREPARED_SOURCE_HOST_AUDIT.json)。

189[独立终态审计](../budget-calibration-20261015/budget-diagnostics/V189_INDEPENDENT_TERMINAL_FULL463_AUDIT.json)首次执行通过，未修改审计条件：253来源/归档、504组件zero与252正式zero/full、144原整数verifier调用、全部463原到达/输出和清理均核。页峰四rank均377、最终0；2570次target，M80为1898次。按原worker callback首尾间隔/255判定则为213达标、GP289.8986（仍除原generate wall），这是另一历史口径，不是final9共同窗口评分。Native charged/256仍为220达标、GP299.4258。TTFT tight中位0.9717秒/P90 2.1834秒；loose中位89.47秒/P90 106.13秒，SLO优先接纳的排队代价必须与收益同时呈现。

ROOT的refill贡献条件边界已由[另一代理独立复核](../budget-calibration-20261015/budget-diagnostics/COMPLETED_REFILL_CALLBACK_GAP_BOUNDS_INDEPENDENT_READONLY_REVIEW.json)，926行逐项重算差为0。复核支持同leader perf_counter和原callback/helper顺序，公式成立；依然没有逐次helper起止或逐rowcharge原始记录，因此不能改称实测device耗时或直接预测新Goodput。


### 2026-10-17 15:38 UTC：197完整测量显著提高TPOT达成；继续缩短验证窗口

197通过当前进程原B4/M80、多owner eager prefill/next检查、warm/两zero和sealed full463，guardian complete/passed_full，原基线pre/post、源码不变、worker退出、自有进程清理与四卡空闲均通过。独立终态HOST审计按明确receipt首次执行通过，未修改审计脚本。294归档/来源、504原组件zero、336新增组件zero、252正式zero/full与144次原整数verifier核实；新增多owner为24个实际role-case，未声称穷举所有prompt/cohort组合。[ROOT结果](../budget-calibration-20261015/budget-diagnostics/ROOT_V197_CURRENT_FULL463_RESULT_AND_HANDOFF.json)、[独立终态审计](../budget-calibration-20261015/budget-diagnostics/V197_INDEPENDENT_TERMINAL_FULL463_AUDIT.json)。

| 方法/完整候选 | Raw token/s | Goodput token/s | 完整负载秒 | tight达标 | normal达标 | loose达标 |
|---|---:|---:|---:|---:|---:|---:|
| 固定γ4历史同卡结果 | 921.66 | 501.64 | 128.60 | 61/271 | 95/96 | 96/96 |
| 189 SLO优先接纳 | 630.16 | 299.43 | 188.09 | 34/271 | 90/96 | 96/96 |
| 197 active整组drain后接纳 | 604.39 | **597.86** | 196.11 | **266/271** | **96/96** | **96/96** |

197以173为唯一生产父源，active32、target≤16、C96/E96/global192与原gamma/topology/budget策略不变，只在active非空时延后新admission/prefill。16个实际cohort覆盖全部463，大小为1×1、32×13、18×1、28×1；没有旧active被新prefill打断，未删除原到达或即时root/token callback，deferred_polls为0。它是接纳策略的收益，Raw略低，不能解释为矩阵算力或接受率翻倍；本轮也非旧请求轨迹不变的反事实。四rank页峰均260/544、最终0，target3178次，其中exactN16/Q80为1696次。

| TPOT/TTFT | tight | normal | loose | 全体 |
|---|---:|---:|---:|---:|
| 原Native charged TPOT中位/P90(ms) | 34.26/37.61 | 28.67/32.24 | 39.51/41.76 | 34.26/39.94 |
| TTFT worker首root减原arrival中位/P90(s) | 34.64/63.79 | 38.63/60.69 | 38.25/63.88 | **36.48/63.77** |

排队代价必须与收益并列。使用原worker实际首尾publication间隔/255则456达标、GP595.2499；Native charged/255与/256均458达标、GP597.8606。它们仍除本轮原generate wall，不是final9共同origin到last-publication评分。历史Native charged与官方MC2 TP4 clock不同：本轮相对固定γ4约1.192×，相对较强历史MC2 TP4 429.8443约1.391×，**尚未达到644.7664目标或正式1.5倍资格，未采用、whole463 own-AR仍False**。

后续优先评估197上的独立四查询B4 spine路径：真实四步draft与四比较不省略，target只查询root/d1/d2/d3，raw accepted0..4显式转换成tree path最多3加target frontier，最大交付仍4。这与同前缀B3+BONUS输出相同，但第四draft步真实写好d3 KV，可能减少下一轮committed-past repair；不能把第四候选算作额外交付，亦不能把20%query减少直接当20%GP。197当前可见主链BONUS13040token约占decode输出11%，去掉它会增加循环数。新私有count4 preflight、新g策略域、M64同M完整hidden/head/KV、原commit/ownnext与full测量仍须通过，当前只有[只读可行性](../budget-calibration-20261015/budget-diagnostics/TREE_B4_NO_BONUS_FOUR_QUERY_FIXED_WINDOW_MINIMUM_SOURCE_FEASIBILITY_REVISION2.json)，200尚在设计。198真正B4分叉math/g诊断已SOURCE/HOST就绪、独立peer中；199四步后续草稿诊断仍在实现，均未上卡。


### 2026-10-17 15:48 UTC：后续诊断与四查询的零边际限制

ROOT于15:39:48启动共同前缀历史隔离v2，唯一NPU会话11592；92成员源归档预检通过，当前四rank已在从原prompt独立递推16个owner历史。该诊断无生产/Goodput计分，不继承finite数学门为whole463 own-AR，终态后另审计。

198真正B4分叉对照已ROOT+peer封存320选定成员，4项必要HOST通过、五次attempt完整保留，仍未上卡。[ROOT来源封存](../budget-calibration-20261015/budget-diagnostics/ROOT_V198_PREPARED_SOURCE_HOST_AUDIT.json)。它比较六个真fork与两个线性控制，以及三类各12相关输入前缀的旧spine/新joint选择；包含独立sameM数学与原commit/next，实际自然fork或side命中不足须报subset。诊断计时包含greedy返回clone/shape-key-flag检查与同步，old三次再joint三次，既非ABBA也非纯生产成本，不可当作GP。

[200四查询最小计划](../budget-calibration-20261015/budget-diagnostics/V200_B4_FOUR_QUERY_NO_BONUS_MINIMUM_IMPLEMENTATION_PLAN.json)尚未实施。其new B4最大交付4，在相同前三草稿与四target预测下，交付数与原B3+BONUS逐点相同，因此稳态g4相对g3的边际为0，原“购买完整正收益跳跃”分配器可能根本不选B4。不能通过样本噪声、旧5token g或强制B4掩盖这个限制。后续同时只读评估保留原B4 BONUS、仅额外执行最后一个KV-only draft模型步的方案：Native支持5步且最后一步无LMhead，可能在不改g和target协议的条件下减少下一轮missing d4 repair；尚未形成候选或性能证据。


### 2026-10-17 16:05 UTC：197后的两个独立性能候选进入实现

200四查询方案继续保留为只读计划，未创建候选。completed197[精确覆盖核查](../budget-calibration-20261015/budget-diagnostics/V200_EXACT_N16_ALL_B4_NO_E_COMPLETED197_STATIC_BOUNDARY.json)显示exactN16/allB4/noE为1696调用、27136 owner-rounds；静态少27136个target物理query的同时，会少9809个可见BONUS。新g4与同heads旧B3交付相同，不能把这种消融宣称为新的最优动态B策略。

ROOT已批准两个互不混合的源码/HOST/guard实现，均以冻结197为唯一生产父源、尚未上卡：

| 候选 | 核心变化 | 原样保留 | 当前证据与限制 |
|---|---|---|---|
| 201 normal B4 final-KV-only5step | 仅合法N16/allB4/noE且无missing的draft多一个模型步消费d4写KV，不执行第五head | 原四head/raw8/g、target五query与BONUS、所有域外root_seed修复 | 新steps5/final-kv/confidence/M16 key须自然验证；第五hidden/全部56KV不能借只验证前四输出的Native标记。[最小计划](../budget-calibration-20261015/budget-diagnostics/V201_NORMAL_B4_FINAL_KV_ONLY_FIVE_STEP_MINIMUM_SOURCE_PLAN.json)。 |
| 202 N9..15 allB4尾批M80 | 原pad函数将这七类causal真实尾批由M128/S38缩为M80/S16，dummy均为5query | 真实查询/预算/g/BONUS、真B5/full9、其他mapper分支 | 573调用物理query由73344降至45840，静态少27504；这占全target物理Q10.521%，不能当耗时或GP预测。复用现G80/H80与M80profile，但七个新realN+dummy usage仍须独立sameM/current lease/128KV/原commit-next/两zero/full。[只读计划](../budget-calibration-20261015/budget-diagnostics/V202_N9_TO15_UNIFORM_B4_EXISTING80_PADDING_MINIMUM_READONLY_PLAN.json)、[原轨迹计数](../budget-calibration-20261015/budget-diagnostics/V202_COMPLETED197_SMALL_TARGET_GEOMETRY_READONLY_COUNTS.json)。 |

199 bounded P4→child4/shared4诊断在独立peer发现rollback前prefix-only空ledger被误作完整父/child源快照。REV2先在common stage真实恢复parent/shared4源，再进入原nextproof快照，原严格ledger相等门不变；旧拒绝次序和新next3边界均由必要HOST真实执行，旧ready与全部source/CPU/HOST249-member archive保留。ROOT+peer封存752成员，尚未上卡、生产/full资格保持False，默认planner缺mixed-child16覆盖仍应报subset。[199 ROOT封存](../budget-calibration-20261015/budget-diagnostics/ROOT_V199_PREPARED_SOURCE_HOST_AUDIT.json)。guard实际749 pre-peer输入与ready754输入区别已显式绑定；唯一P154旧tar4,163,298字节作为选定依赖嵌入新tar，P148/P160/P173大档案仍原位核验，未将该例外掩作所有binary都未复制。

共同前缀历史隔离v2仍是唯一NPU会话11592，前三case完成、第四case独立历史运行中；无终态数值归因、无新GP结果，后续按固定receipt独立审计。当前完整最好仍197 GP597.8606，目标644.7664与正式final9资格均未达成。


### 2026-10-17 16:20 UTC：历史隔离v2完成，仍保留own-AR限制

共同前缀历史隔离v2完整完成，四个case各16个owner先从真实prompt独立按原单行AR递推；每个target rank实际9200次自身模型/单行head请求，draft仅参与同步。原基线pre/post通过、源码不变、自有进程清理、worker退出与四卡空闲均通过。独立HOST终态审计按明确receipt SHA首次通过，未修改审计条件：[独立审计](../budget-calibration-20261015/budget-diagnostics/CANONICAL_PREFIX_HISTORY_V2_INDEPENDENT_TERMINAL_AUDIT.json)、[ROOT四case比较](../budget-calibration-20261015/budget-diagnostics/ROOT_CANONICAL_PREFIX_HISTORY_V2_TERMINAL_RESULT_AND_COMPARISON.json)。

| case | own-history PA单行root | 同past eager FIA80 root | 同一份80行hidden仅换单行head | hidden/greedy图恢复输入后的root | 历史生产candidate root |
|---|---:|---:|---:|---:|---:|
| P173 request0 step137 | 382 | 382 | 382 | 382 | 13 |
| P173 request1 step34 | 7629 | 525 | 525 | 525 | 525 |
| P190 request0 step205 | 576 | 576 | 576 | 576 | 3411 |
| P190 request1 step199 | 382 | 382 | 382 | 382 | 13 |

每条受控路径都保持同rank全部128份past KV指纹，root/future槽不计入past。第二case说明在相同受保护own-AR历史下，PA单行和FIA80模型路径仍可选出不同token；在同一份FIA80计算出的hidden上，只把LMhead从80行改为1行没有消除这个差异。四case同hidden不同head形状的BF16 logits有数值差，但这里argmax均相同，不能据此把所有差异归因到head、量化或历史KV损坏。其他三个历史candidate差异没有被本受控own-history组合复现，它没有重放当时真实packed输入/历史；whole463 own-AR、生产数值无损和正式性能资格仍False。

201新候选在冻结前发现父permit身份绑定错误：把197身份传给原P148 diagnostic action，原checker会在模型前拒绝。独立审查已落blocker；旧源码/HOST/ready保留后最小REV2修复并补实际原permit正负HOST，再审查封存。202继续准备七个尾批的新实际usage数值门。203获准单独实现197的empty-active调度加原SLO优先接纳0→1，其余配置/时钟/g不变，尚无新模型结果。当前完整最好仍为597.8606，未达到644.7664目标。


### 2026-10-17 16:35 UTC：201 REV2封存并启动；202修订域外回退

201启动许可身份问题已由实际原P148 checker/activate_local_diagnostic正负例验证修复。修订后的当前源同次5项HOST与冷guard检查通过，原4项旧revision未拼接到当前计数；9次attempt、旧218文件快照与失败全部保留。独立peer通过、ROOT按531选定成员封存并启动唯一设备会话40675，tag为`final-kv5-v201-full463-v1`。截至本条仍在模型加载/构造阶段，没有新数值资格或GP。[201 ROOT封存](../budget-calibration-20261015/budget-diagnostics/ROOT_V201_PREPARED_SOURCE_HOST_AUDIT.json)、[明确启动记录](../budget-calibration-20261015/budget-diagnostics/ROOT_V201_EXPLICIT_GUARDIAN_LAUNCH.json)。后续须先过当前原B4/M80、多owner prefill、独立第五hidden/全部56KV及三trace/两zero，再warm/seal/full。

202尚未封存。`install_pad`的N9..15/allB4过滤漏了causal条件，但进一步核源码确认：冻结父`target_layout.plan_family`本身已拒绝B4真fork，不能据此声称发生了父生产回归。窄wrapper在单独generic Native full9 padding ABI下存在范围过滤缺口；修订须保留当前父生产拒绝语义，generic carrier测试不授新生产资格。已保留旧ready/source/CPU后授权最小REV2补family filter和实际原predicate/fork域外HOST，原check_frame数学严门不放松。同时把scope报告的数值variant分类与component已计算的合并结果对齐，避免只透传原173标签。[202独立阻断](../budget-calibration-20261015/budget-diagnostics/V202_INDEPENDENT_FULL9_FALLBACK_SOURCE_BLOCKER.json)。203仅原SLO优先接纳的独立候选3HOST及冷guard通过，仍在source peer中、未封存或上卡。

202边界适用范围已更正：[ROOT层级说明](../budget-calibration-20261015/budget-diagnostics/ROOT_V202_FULL9_BOUNDARY_SCOPE_CLARIFICATION.json)。原独立阻断记录03172…保留供审计，不再把其过强的父生产域外回归前提当成已证明结论。


### 2026-10-17 17:09 UTC：201目录竞争失败闭环，203已进入模型检查

201完成原构造/201预填充/44个target与114个draft入口、原B4/M80和197多owner检查后，在新组件输出目录的创建阶段失败。原代码让四个进程同时执行`mkdir(exist_ok=False)`，一个成功、三个FileExistsError，随后共同失败；尚未进入第五KV写入的数值比较、正式full或GP测量，不能记为GP=0或性能回退。原源码和失败原样保留；原baseline前后检查、源码一致、自有进程清理及四卡空闲通过。[ROOT失败记录](../budget-calibration-20261015/budget-diagnostics/ROOT_V201_CURRENT_FAILURE_RESULT_AND_HANDOFF.json)、[独立失败审计REV2](../budget-calibration-20261015/budget-diagnostics/V201_INDEPENDENT_TERMINAL_FAILURE_AUDIT_REVISION2.json)、[ROOT审计补充](../budget-calibration-20261015/budget-diagnostics/ROOT_V201_FAILURE_AUDIT_COMPLETED_ADDENDUM.json)。首次独立HOST审计遇到fresh Python tuple与JSON list的结构比较差异；旧审计保留，REV2仅在比较时做JSON canonical roundtrip，未放宽任何来源、数学或测量门。

204以冻结201为源码派生、197为唯一生产父源，仅将该目录创建改为draft leader单写，保留四进程原共同stage及预存目录拒绝。当前6项HOST同次通过，其中真实四进程共享文件系统重现旧三失败、新全通过、预存一失败并四进程共同拒绝；HOST中执行原stage4函数但其同步/collective是明确的HOST carrier，不是设备验证。独立peer通过，ROOT冻结准备中，未上卡。202与198冻结源码也发现同类目录竞争，保持原档案不改且不再启动；分别在独立新205、206修复。205的7项HOST及冷归档检查通过，独立peer中；206仍为有界数学/收益诊断，不能授生产或GP资格。

203已于16:56:12作为唯一NPU会话60458启动，精确tag为`empty-cohort-slo1-v203-full463-v1`。它只在197 empty-active调度上启用原SLO优先接纳，target SLO first仍关闭，g、预算、模型、MC2、原463到达及输出/计时规则不变。截至本条已完成真实模型预填充，正在进行原算子和计算图资格检查，尚无新的完整GP。[203 ROOT封存](../budget-calibration-20261015/budget-diagnostics/ROOT_V203_PREPARED_SOURCE_HOST_AUDIT.json)、[启动记录](../budget-calibration-20261015/budget-diagnostics/ROOT_V203_EXPLICIT_GUARDIAN_LAUNCH.json)。接纳重排可能增加部分请求的原到达TTFT，结果须同时报告各类TTFT与Native charged Goodput。当前完整最好仍197 GP597.8606，目标644.7664、whole463 own-AR和正式final9资格均未达成。


17:15更新：204按315选定成员、205按968选定成员均通过ROOT+独立peer封存，仍未上卡。[204封存](../budget-calibration-20261015/budget-diagnostics/ROOT_V204_PREPARED_SOURCE_HOST_AUDIT.json)、[205封存](../budget-calibration-20261015/budget-diagnostics/ROOT_V205_PREPARED_SOURCE_HOST_AUDIT.json)。205第7项HOST只执行实际actor目录lambda和真实四进程共享FS/common-outcome carrier，未执行原stage4函数体/NPU fence；原stage源码/调用/位置由整源投影保持，实际现场门不变。它与204 HOST执行原stage4体的范围不同。所有旧尝试均保留；205初次冷检查的日期路径误替换已通过原launch_argv/四路径检查防回归。若203不达目标，ROOT倾向下一个单独实测205，仍须等203终态、清理、基线和精确独立审计；并非自动启动。私人区约75G已用、121G空闲，新204/205共约20MB。


### 2026-10-17 17:28 UTC：203完整完成，GP613.70，保留摘要类等待代价

203唯一模型会话60458正常退出0，原baseline前后10496/15/212通过、源码不变、自有进程清理且四卡空闲。独立终态审计使用预先检查的原脚本，首次完整PASS/errors为空，HOST会话29737退出0：[独立审计](../budget-calibration-20261015/budget-diagnostics/V203_INDEPENDENT_TERMINAL_FULL463_AUDIT.json)、[ROOT完整结果](../budget-calibration-20261015/budget-diagnostics/ROOT_V203_CURRENT_FULL463_RESULT_AND_HANDOFF.json)、[ROOT独立审计补充](../budget-calibration-20261015/budget-diagnostics/ROOT_V203_INDEPENDENT_AUDIT_COMPLETED_AND_NEXT_SELECTION.json)。273份raw文件保留，原463到达与每请求256token全部完成，当前原B4/M80、独立多owner和全两zero/seal门通过。

| 方法 | Raw token/s | Native charged Goodput token/s | tight达标 | normal达标 | loose达标 | 相对固定γ4 | 相对历史较强MC2 TP4 |
|---|---:|---:|---:|---:|---:|---:|---:|
| 197 empty-active，原入场 | 604.39 | 597.86 | 266/271 | 96/96 | 96/96 | 1.192× | 1.391× |
| 203 empty-active，原SLO优先入场 | 619.05 | 613.70 | 267/271 | 96/96 | 96/96 | 1.223× | 1.428× |

203生成区间191.4683s，共459/463达标；相对197单轮观察GP高2.65%，尚不能认为已建立稳定因果增益。原Native charged/256与worker-publication/255本轮均459达标、同一generate wall下GP相同，但它们的TPOT分布不同，仍是不同口径，不称client delivery或正式final9。16个空active cohort仍为1×1、18×1、32×13、28×1，target3055轮，其中当前M80使用1768次，物理Q256796、真实Q194095。

| 方法 | tight原到达TTFT中位数/P90 秒 | normal原到达TTFT中位数/P90 秒 | loose原到达TTFT中位数/P90 秒 |
|---|---:|---:|---:|
| 203 | 6.614 / 11.800 | 29.056 / 42.208 | 96.802 / 121.019 |

loose最大TTFT139.108s；203把紧SLO请求优先服务，明显增加摘要类等待，不能只报告TPOT或GP而隐去这个代价，当前不采用该策略。全体TTFT中位数10.439s/P9097.457s，Native TPOT中位数33.998ms/P9041.459ms。目标644.7664仍未达到，距此需再高约5.06%GP；此差额不是性能预测。Whole463独立own-AR、adoption和正式final9的1.5x仍False。下一轮单独选205：唯一生产父197、原SLO入场关闭、仅合法小尾批M128→M80和目录单写修复，实际开始时间另记。

206有界真分叉诊断已独立peer+ROOT封存247成员，[封存证据](../budget-calibration-20261015/budget-diagnostics/ROOT_V206_PREPARED_SOURCE_HOST_AUDIT.json)，尚未上卡/无生产issuer或full资格。207/208仍在SOURCE设计：先控制normal cap4→3沿原mapper，再研究合法N16/B3/M64，不能把预算策略与图变化混为单一归因。197原P3候选出现90次但selected/promoted为0，新cap3可能扩大该域，必须补当前原非union P3/shared3→child3的真实状态/KV/commit/next资格；normal3证明不外推到E。


17:28:30 UTC：ROOT已明确启动205，唯一自锁guardian会话17979，tag=`small-tail-single-writer-v205-full463-v1`，无outer flock。启动前203完整终态、独立审计及模型/HOST会话关闭0均已核验；205从source/归档/原baseline测前检查开始，当前尚无205数值或GP。[205启动记录](../budget-calibration-20261015/budget-diagnostics/ROOT_V205_EXPLICIT_GUARDIAN_LAUNCH.json)。原入场SLO0，不叠加203；后续须真实重跑当前原201/44/114、B4/M80、N2/N32和七个新N9..15 usage的完整数值/协议/两zero，再进入full。


### 2026-10-17 17:38 UTC：B4旧轨迹按类的交付边际（不是新B3测量）

[仅HOST算术记录](../budget-calibration-20261015/budget-diagnostics/COMPLETED_197_203_B4_DELIVERY_MARGINAL_HOST.json)使用197/203独立终态审计绑定的实际raw；按真实owner将target call与非首token回调一一关联，每个后继context均校验为前context加实际交付长度，463请求均核256输出。只统计未结束、未被输出长度截断的B4轮次：

| 旧轨迹 | tight B4每轮平均交付 / 第5token频率 | normal平均交付 / 第5token频率 | loose平均交付 / 第5token频率 |
|---|---:|---:|---:|
| 197 | 3.0783 / 33.92% | 3.6403 / 52.48% | 2.6231 / 21.57% |
| 203 | 3.0626 / 33.70% | 3.6524 / 52.62% | 2.6059 / 21.32% |

这些是旧B4被分配到的实际历史上的条件统计，不是随机请求样本。把旧回调交付裁到4只能作为旧heads的算术proxy，不是实际B3的g、不包含新M64或变化后的未来历史，亦不能推算GP；未安装任何新g。normal的第四候选贡献较高，后续仍需真实B3与计算成本实验，不能凭此统计直接改策略。

源码核查补充：当前生产每home逻辑批次16，原cfg实际`max_target_batch=32`且允许ready-home合批；历史“target16”的简写不能解释成生产最大验证批次16。207/208仍为SOURCE预备，原父源码不改；207拟以逐N形状的受控同M数学与独立自然P3/shared3协议两层验证区分数值覆盖和真实晋升，不声称每个N都有自然positive。


### 2026-10-17 17:53 UTC：后续缓存机会仅做HOST可行性，未启用

[提示前缀HOST分析](../budget-calibration-20261015/budget-diagnostics/ROOT_PROMPT_PREFIX_CACHE_FEASIBILITY_HOST.json)核197实际tokenized input：463请求/307种提示/额外156个完全重复请求，共203850提示token。按原16个cohort从冷缓存、只在cohort完整prefill后发布缓存、至少留1个未缓存token，简化独立LRU缓存128/256个32token块可复用提示token约31.54%/32.48%；不限制容量为41.68%。这是没有真实active/decode页压力的理想模型，不是Native544页池实测、NPU模型测试或时延/GP预测，未创建或启用缓存候选。

原NativePrefixCache的实际纯HOST allocator中，已用公开allocate/activate/release API复现“前块被回收重分配、后块仍命中，却报告cached_tokens=2”的2token块微例；它只证明命中计数不总等于连续已物化前缀的元数据边界，没有执行KV张量/数值输出。未来若做缓存必须先修连续命中、已完成prefill发布、共享块不可写、引用计数和页复用，并补实际prefix/完整KV/root/ownnext及suffix形状预热/两zero；禁止复用测量前warm数据或未来请求。当前原实验均关闭缓存。

[MC2基线设置核对](../budget-calibration-20261015/budget-diagnostics/ROOT_PREFIX_CACHE_BASELINE_SETTING_ADDENDUM.json)确认历史较强MC2 TP4的真实result中enable_prefix_caching=False；若后续启用缓存，需单列缓存工程收益、保留原off对照并评估正式MC2基线缓存设置，不能把缓存机会计作动态B或树算法已测收益。205仍是唯一设备实验，当前新七usage第一组N14已完成，尚未full计分。


### 2026-10-17 18:02 UTC：205在组件汇总校验失败，无新Goodput

205在新七种N的三组实际cycle/next结束后，`audit_usage205.py`第36行把target rank原应为None的draft记录当作字典，触发TypeError并共同停止；没有进入warm/twozero或full463。336份cycle、84份next和四份component保留，但不将已写记录当作组件最终资格，也不记GP=0。总用时1878.61s，小于7200s上限，外层通用“within7200”异常不代表超时。原baseline前后10496/15/212、源码不变、worker退出、自有进程清理和四卡空闲均通过；预先审查的独立失败HOST审计首次通过/errors为空。[205失败记录](../budget-calibration-20261015/budget-diagnostics/ROOT_V205_CURRENT_FAILURE_RESULT_AND_HANDOFF.json)、[独立失败审计](../budget-calibration-20261015/budget-diagnostics/V205_INDEPENDENT_TERMINAL_FAILURE_AUDIT.json)。冻结205原样保留，另建209仅修按role汇总及真实失败raw正负HOST；新引擎仍须重新跑全部实际数值/协议门，不能借用本轮资格。当前无模型运行，最好完整结果仍203 GP613.6995，未达到644.7664或正式1.5倍目标。


18:03:28 UTC：ROOT已明确启动冻结206，唯一自锁guardian会话20089，tag=`truefork-single-writer-v206-bounded-gain-v1`；205终态失败审计和清理已先闭合。本轮先测同B4下真分叉与主链选择的实际交付/状态/成本，不跑463或签发生产issuer，不把诊断HOST计时当GP。三类前缀使用teacher-forced输入，各arm随后真实独立提交和ownnext；候选数相同不代表其接受率或成本相同，必须实际验证。209尾批校验修复、207cap3控制继续SOURCE/HOST并行准备，均未上卡。[206明确启动](../budget-calibration-20261015/budget-diagnostics/ROOT_V206_EXPLICIT_GUARDIAN_LAUNCH.json)。


### 2026-10-17 18:15 UTC：207源码封存，完整执行仍暂缓

207经过独立peer与ROOT审查封存697选定成员，六项当前HOST同次通过，原父档案原位核验且未重旧套件。当前源码包括normal真实三步/raw6/canonical8与独立N16/M96数学门，以及受控全N17..32/allB3的E参数域和独立自然P3/shared3原协议门；所有设备门均尚未执行。实际四角色reader已用明确synthetic JSON执行正负例，不能当浮点数值证明。[ROOT封存](../budget-calibration-20261015/budget-diagnostics/ROOT_V207_PREPARED_SOURCE_HOST_AUDIT.json)、[独立源码审查](../budget-calibration-20261015/budget-diagnostics/V207_INDEPENDENT_SOURCE_PEER_REVIEW.json)。

源审查同时明确：合法B1/B2 normal与E3混合，或总N<17的E批次，仍可能出现且会在模型前硬拒；没有关闭E或强制预算掩盖这个缺口。故封存不授权直接跑full，full463_execution_authorized=False，未产生GP。208仍为design-only。206为唯一NPU会话20089，原201预填充已全部完成、继续原图/算子门；209按角色的校验修复当前三项HOST及真实424旧raw/15反例通过，独立peer进行中，未上卡。


### 2026-10-17 18:20 UTC：209角色校验修复封存，等待重新设备验证

209以冻结205为source donor、197为唯一生产父源；20个whole-source和四guard投影由peer/ROOT复核，actor/target/mapper/协议/预算/到达与计时无计算delta，实际修复仅按role读取cycle/next draft证明并补互斥None门。当前三项HOST同次通过；424份授权旧raw重放通过，15个缺字段、物理head、原整数/Native零差分/matcher反例全部拒绝。独立peer另核252轮target整数BONUS裁决和1176零差分。旧205七项仅为原位验证的donor证据，不拼进当前三项；两个失败尝试与96份源/日志保留。[ROOT封存](../budget-calibration-20261015/budget-diagnostics/ROOT_V209_PREPARED_SOURCE_HOST_AUDIT.json)、[独立peer](../budget-calibration-20261015/budget-diagnostics/V209_INDEPENDENT_SOURCE_PEER_REVIEW.json)。

ROOT封存1529选定成员，原205968及上游档案只原位验证；尚无209模型或GP，旧raw回归不转移数值资格。ROOT倾向206有界分叉诊断终态并清理/审计完成后，下一轮完整测209，仍须单独明确启动，不自动排队执行。当前唯一NPU仍206会话20089；207/208完整性能路径暂缓于已披露的E域缺口。


18:24 UTC：206原201/44/114、原B4与causalM80当前组件通过，开始新增sixfork/twolinear第一组。新增actor记录仍待三组与终态独立审计，不是生产/GP资格。209未来独立终态auditor已ROOT审源码通过，[审查](../budget-calibration-20261015/budget-diagnostics/ROOT_V209_FUTURE_TERMINAL_HOST_AUDITOR_SOURCE_REVIEW.json)：除当前来源绑定、增加按role严格检查、首receipt读取前绑定门外，原独立整数/多owner/full统计和时钟函数AST保持；其旧424raw/12反例HOST不作209模型证明。终态入口尚未执行，须未来明确terminal tag及receipt SHA。


### 2026-10-17 18:45 UTC：V206 同预算 B4 真分叉诊断结束

原始前后锁校验、源文件一致性、worker 退出与独立终态审计均通过；8 个布局（6 个分叉、2 个线性对照）完成 3 遍，1344 个 Native 几何零差检查通过。自然选择仅观察到 9 行分叉、0 行接受侧分支，状态保留 `passed_subset`，不授予生产启用或全工作负载输出等价资格。

| 请求类别 | 旧 B4 主链平均交付 token | 当前 joint B4 平均交付 token | 每臂测量行数 |
|---|---:|---:|---:|
| 代码 | 3.8333 | 3.7500 | 12 |
| 问答 | 3.9167 | 3.9167 | 12 |
| 摘要 | 2.6667 | 2.6667 | 12 |

每类来自 4 个请求身份的 3 个前缀位置，彼此相关，不能当作 72 个独立请求。诊断时间含同步与检查、采用先旧后新的固定顺序，不能当作生产设备时间或 Goodput。未安装新 g 曲线、未跑完整 463 请求，也没有新的 Goodput。当前分叉选择未显示收益，继续核对 raw8 候选中被漏选的侧分支机会。

证据：`budget-calibration-20261015/budget-diagnostics/ROOT_V206_CURRENT_BOUNDED_RESULT_AND_HANDOFF.json`；独立审计 SHA256 `59563da961e533aa63a2ca4c864ee6577a94317ce44f8902f1a06f3e45e568a5`。下一完整测试选择已冻结的 V209，重新执行本进程全部数值资格检查，不继承 V206 资格。


2026-10-17 18:45 UTC：ROOT 已在上一模型与 HOST 审计会话关闭后，显式启动 V209 完整测试（session 82303）。冻结 1529 个归档成员；只测试原生产父版本 197 的小批次尾部 causal 执行扩展，保留原请求、预算策略及 g 曲线。V205 的失败保留，V209 按实际角色检查各卡审计字段，并重新执行本进程数值资格检查。当前尚未产生 V209 完整结果。


### 2026-10-17 20:42 UTC：容器重启中断 V209 首轮

恢复后原 session82303 不存在，child1756454 及模型进程均已消失，4 卡空闲。旧回执仍为 running，最后进度停在原始201项预热的第1项；无 completed/worker-exit、无完整463结果。保留原回执、日志与进度，不归类为数值失败、不记录 Goodput 0。ROOT 另写中断观察证据，并用新 tag 重启同一冻结源码的完整测试，重新执行所有当前进程资格检查。


2026-10-17 20:44 UTC：V209 v2 在模型启动前被原锁校验拒绝，原因仅为重启后丢失 `/usr/local/Ascend/cann-9.0.0`、`/opt/specslo-cann` 两个路径链接。运行现有 `restore-retained-runtime-aliases.py --restore` 后，15 个外部文件全部符合原 SHA；没有安装/下载、没有修改容器默认环境或锁。保留 v2 失败回执，模型从未启动。已显式启动同源码 v3，session19465，全量校验和本进程数学验证重新执行。


### 2026-10-17 21:15 UTC：预算3／64行诊断源码冻结，尚未启动

已有父版本的16请求/B3实际执行为96行、32段，其中32行为填充；降低候选预算不能直接视为降低设备成本。新 ROOT210 仅准备正常请求 B3 的64行/16段有界数值诊断，原构造、环境、扩展、201/44/114预热及当前 B4/M80 检查保留，新 G64/H64 必须自然验证，原提交与下一轮缺失后缀修复必须通过，完成后撤销临时映射。尚无生产启用、完整463、耗时、g曲线或Goodput。

三组实际 HOST 检查分别覆盖写入/审计/映射、spawn函数与源检查边界、原监管异常与清理控制流；全部为显式主机载体或替身，不是设备资格。新文件没有独立代理源码复审声明；原捐赠库的复审只作为源码依赖证据。1738成员归档及真实guard冷启动的源码/归档路径检查通过，冷检查耗时12.59秒，未查询设备或启动模型。冻结证据：`budget-calibration-20261015/budget-diagnostics/ROOT_V210_PREPARED_BOUNDED64_SOURCE.json`。当前唯一模型仍为 V209 v3/session19465；下一模型由ROOT在它完整关闭并审计后显式决定。

简单 target 概率校准的按请求身份留出检查，旧head条件计数为122，对照主链实际125；该估算不是新设备 g，未接入生产。源数据仅12个请求身份的36个相关前缀，仍需独立身份实测。


### 2026-10-17 21:29 UTC：V209完整结果未提升，转入B3/64行诊断

V209 v3 完成并通过独立审计（session19465、worker及两个HOST审计均正常退出；原前后锁校验通过、自有进程清空、四卡空闲、源码未变）。Goodput **580.90494**、Raw **596.36139 token/s**，整轮 **198.75197秒**，达标 **451/463**（tight259/271、normal96/96、loose96/96）；实际首末token发布时间口径、分母256的Goodput为577.04083。低于原197的597.86060和既有最高203的613.69947，不采用、不计为目标完成。

逐请求对比197与209这两份实际完整结果，463条/118528个输出token全部相同；11条tight的SLO状态发生变化，净减少7条达标。这只证明这两轮输出一致，不证明canonical own-AR、TP4或所有输入等价。物理target查询从261420减至233916，整轮仍变慢。发现209常规回退路径每次调用`parent_module`重读454072字节源清单并算SHA，300次HOST微测每调用中位数约2.007ms，缓存模块属性约0.407µs；这是可移除的热路径开销，尚无NPU对照证明其解释全部变化。

已显式启动 ROOT210 有界 B3/Q64/M64/S16 数值诊断（session20545）；原构造与新tag原始记录保存在199允许的private actual根下，199源码和既有记录均未修改。它不启用生产64映射、不跑完整463、不测Goodput；须先完成新形状隐藏态/完整head/KV、原提交、真实下一轮d3修复与两轮零新增检查。源代码没有独立代理复审声明，已完成四份整源码反向投影、两份launcher精确投影、三组实际HOST检查及真实guard冷源码/归档检查。

结果证据：`budget-calibration-20261015/budget-diagnostics/ROOT_V209_V3_INDEPENDENT_NEGATIVE_RESULT_AND_HANDOFF.json`；当前启动记录：`budget-calibration-20261015/budget-diagnostics/ROOT_V210_EXPLICIT_BOUNDED64_GUARDIAN_LAUNCH.json`。


### 2026-10-17 21:51 UTC：B3／64 行有限路径验证与概率校准准备

ROOT210 的设备子进程正常退出，normal B3、16 请求、64 个真实 query 的 hidden/head/KV、原始提交及下一轮补修检查文件齐全。原 guardian 在结束后导入 `base207` 时缺少路径而失败；保留原失败 receipt，未改写为成功。单独补足 HOST 审计加载路径后，独立核查通过（36 次原始 Native 整数裁决、576 个 owner、完整元数据和槽位核对）。这仅验证有限路径，不提供性能、全负载或生产资格。详见 `ROOT_V210_CURRENT_BOUNDED64_RESULT_AND_HANDOFF.json`。

下一项为已有公共 0.6B draft 的 BF16 概率校准诊断：463 请求的 118528 个历史 target 标签，按 307 组不同提示分组，248 组训练、59 组留出；相同提示不跨划分。先比较 HF 与 Native 的 144 个实际前缀预测，再用真实 draft 自回归四步概率评估留出候选选择，不能使用失配之后的 target 前缀预测概率来选树。结果仅为历史 golden 条件下的代理，必须再以 Native 新验证和完整 Goodput 检验；本阶段尚无新 g 或 Goodput。源码与 6 项 HOST 检查已封存，模型不下载、不迁移。


### 2026-10-17 21:54 UTC：分组留出概率校准诊断完成（不接入）

现有公共 0.6B 模型的 HF eager BF16 诊断完成：118528 标签；59 组提示留出，原始输出与衍生文件均在私人目录。相同 Native 前缀的 144 次预测中 top1 一致 144 次、顺序 top2 一致 139 次，概率平均绝对差 0.00857。当前分箱校准的留出 Brier 从 0.28653 变为 0.28837，log-loss 从 0.50215 变为 0.50552，均略差，故不接入生产。

177 个留出提示前缀使用 draft 自身四步自回归概率选候选，再与历史 target 后缀核对：固定四节点主链有效 token 代理总数 593，直接 draft 概率选子树为 603，分箱校准选子树为 604。这里是历史 golden 条件下的计数代理，未运行新 target/FULL9/KV/提交，不能当作新 g 曲线或 Goodput；此处 B1..5 是四层 raw8 子树选择的代理预算；当前生产 B5 保留四个主链节点和一个侧枝，并非连续五步主链。选择规则不同，不能把这里的代理直接当作生产 g5。独立 HOST 标签、分组、885 项预算裁决复核通过；基线前后核查、源码复核和四卡空闲检查通过。详见 `ROOT_HF_TARGET_RANK_CALIBRATION_TERMINAL_AUDIT_AND_DECISION.json`。

后续优先做单变量主机开销实验：ROOT211 对上一轮 ROOT209 的实际 dispatcher 保持函数体、预算、形状、模型、到达与时钟不变，只将每次退回父 mapper 时的 SHA/454KB JSON 读取改为启动前校验一次后缓存模块；前后完整冻结源检查保留。尚无此修复的完整模型测量。


### 2026-10-17 22:16 UTC：缓存单变量完整测试运行中；扩大 Native 接受率样本已准备

ROOT211 当前 guardian session 59951、子进程 148623，源码 SHA `36543d7cb41c947141c78f6e11bc70d5b054c573edd9c0086fcb1002eed4d526`，归档 SHA `5bcdb6660fdd0e9fdea6da758e34bd38a6aed24c68cb5f13fe656202c13b99d2`。继承的全部参数、MC2、预算、到达、图、计时保留，只缓存已校验的父 mapper 模块；当前已进入原 B4 提交/KV 检查，没有正式性能结果。

ROOT212 是后续有界诊断候选，未运行模型：对全部 59 组留出提示，取输出位置 32、96、192，共 177 个前缀（代码 84、数学问答 45、摘要 48），重新运行 Native 固定 B4 主链与原始概率选树两种策略。共 12 对真实 16 行批次；末组填充请求不计入 g。输入只有历史前缀，不含未来 target 后缀或 HF 候选概率；最大独立候选/参考缓存用页 400/544。原始 206 的 8 种布局、完整零重复、独立 hidden/KV、整数裁决、提交与 own-next 不变，只扩展样本来源和计数校验。7 项 HOST 检查通过；281 成员源码归档与实际 guard 启动前源码检查通过，没有新 Native g 或 Goodput。详见 `ROOT_V212_SOURCE_FROZEN_UNRUN.json`。

另保存 active40 与 realN17..20 的 M100/S20 稠密图可行性推导，仅为备选设计。当前 463 输入的任意连续 40 请求在保守输出增长与 64 scratch 页下上界为 338/544 页，但任意 40 请求子集上界为 602/544；只有严格 FIFO 空 active 队列及顺序不变时才可使用连续区间上界，不能将其当作通用容量保证。尚未实现或运行此设计，不能推导性能收益。


### 2026-10-17 22:57 UTC：缓存单变量负面结果与下一轮准入测试

ROOT211 完整 463 请求完成，Goodput 510.9032、Raw 600.3761、394 请求达标；独立审计复算通过，前后环境与冻结源码检查、四卡清理均通过。首次 HOST 审计仅因 JSON 将元组转为列表而拒绝；原失败记录保留，新审计只统一表示后复核通过，没有改模型代码或 receipt。该配置未采用。

当前 ROOT213 开启已有 SLO-aware admission=1，target-SLO-first=0；唯一模型 guardian session 4684，实际 IO 留在原209私人目录，标签 `cache-slo1-root-v213-full463-v1`。源码 SHA `828af749e5d164399078a078e8074f4f774e2dbbb2ee90af88e6d82f2e1d9e96`。准备在预算选择中加入图容量成本：只考虑已具备四步计算、16 个正常请求混合 B3/B4 的情形，在所有原预算上限内将其补齐为 B4，从 128 行图回到已有资格的 80 行图；尚未实现或测量，不宣称收益。详见 `ROOT_V211_CURRENT_FULL463_RESULT_AND_HANDOFF.json`。


### 2026-10-17 23:08 UTC：预算图成本修复已封存，独立 FIA 自回归核查准备中

ROOT214 预算修复源码和 11 项 HOST 检查通过，独立冷启动源码检查通过，没有模型执行。源码 SHA `52252f8f00d1413e3340952bb94c302ff65023f38f06f413df1ebf3125c2a714`，1619 成员归档 SHA `54a3d15ee4ebfeef713950912c8d8933ae5aed762225af2265a85cf87f50386e`。原分配器仍计算 g、需求和优先级，随后只在原上限允许时把 16 请求混合 B3/B4 补齐到全 B4；候选数会改变，但最大 draft 步数原本已经为 4。没有新增 Native 图或算子。

ROOT211 与 ROOT209 的 201/463 输出序列不同、69 个 SLO 判定翻转，逐请求首差异和实际图容量已保存到 `ROOT_V209_TO211_FULL_OUTPUT_AND_GEOMETRY_DIFFERENCES.json`。不能把全部差异直接归因于低精度。ROOT215 正在准备相同 FIA 后端的独立自回归核查：完整性能计时结束后，以新缓存重做原 prefill 分组，每次只消费 q5 因果段的第一个 target 输出，后续输入只来自自己的历史；另外改变四个未来占位输入，检查第一 query 不受影响。它与 PA 或官方后端的输出等价关系仍需分别说明，当前没有新设备一致性资格。

当前唯一模型仍是 ROOT213，guardian 4684、子进程 246087；201 个 prefill 图完成，继续原 target/draft 与有限路径检查，尚无新 Goodput。


### 2026-10-17 23:11 UTC：实际在线 g 轨迹复算与独立 AR 候选封存

用 ROOT211 每个 target 调用、各请求的实际 token 回调和终止范围规则，重建全程 g 观察值，与当前完整生成结束的三类/B 后验计数及有效 token 总数逐项完全一致。发现 627 次 16 请求混合 B3/B4；首次位于 target 第 270 次（从 0 计），代码类估计 g3=2.91628、g4=2.89510，normal 和 loose 仍认为 B4 更好；当批 16 个请求均剩 255 token，因此不是尾部输出上限导致。实际 shape 调用可能早于当前 target 提交，故这不是直接捕获全部 shaper 输入，也不宣称统一解释所有回归或输出差异。见 `ROOT_V211_ACTUAL_GAIN_EVOLUTION_MIXED16_DIAGNOSIS.json`。

ROOT215 独立 AR 候选完成最终 14 项 HOST 测试与冷启动源码复核：源码 SHA `5a23d88da7a54b61b615221396398258fe81f96d054b0690840307c3ef501561`，1669 成员归档 SHA `dffaac24106e7dc8cb5268ef977902744a3793165b5daabb2760c9821ef20a6e`。性能部分与 ROOT214 相同，AR 在性能时钟之后，以原 prefill 分组、新 KV 和 target 自身历史生成全部 463×256 token；三个 target rank 独立轨迹核对，保留 80 行 head、抽样完整元数据、未来输入 canary、所有实际不一致。未启动模型，不能提前给它输出一致性资格。当前唯一模型仍为 ROOT213。


### 2026-10-17 23:22 UTC：较早的批次分岔与 N9–16 图成本修复

进一步按顺序核对 ROOT209 与 ROOT211：首个请求的 81 次 target 元数据和 token 回调块完全一致；第二个 prefill 分组则分别是 18 与 17 请求，同一 home 变成 9 与 8 请求，跨越当前 per-request cap4/cap5 阈值。首次不同 target 位于从 0 计第 81 次：旧路径 N9、混合 B3/B4、M128/S38，新路径 N8、B5、M48/S8；全部 463 请求的首 token 一致。不同大小的回调块不等于在此处 token 序列已不同；后续逐请求首 token 差异另存。该证据说明在线到达相同仍会因实际运行时间改变准入分组，不能直接将整轮差异解释为单一浮点或竞争问题。见 `ROOT_V209_TO211_FIRST_DISCRETE_DIVERGENCE.json`。

ROOT216 将预算图成本修复扩展到当前已有独立设备资格的 N9–16 混合 B3/B4；一次性准备原 mapper 的 92 种成本，不在热循环读文件。N9 部分混合路径为 M96，其余为 M128，因此分别只记录拟议减少的 16 或 48 个物理 query，不能把该计数当成实际节省的设备时间。原 caps、角色/global 上限、context 和四步最大深度保持约束；原不可变 DTO 重新校验。15 项 HOST 检查与冷启动源复核通过，1710 成员封存源码 SHA `80416aecc96a38ca7c238001337cdfdb260864a979964feac0ff6215872e677a`，归档 SHA `1e803de643bfd8c2b705433334da717d2d9b00eedbc376532831adba506e15f9`，尚未启动模型。完整性能之后保留 ROOT215 同样的独立 FIA80 自回归核查。

当前唯一模型仍是 ROOT213，正在原七种真实请求数的完整协议重复检查。完成本轮终端审计和清理后，优先测 ROOT216。ROOT214/215 继续保留为未运行的源码候选，不把它们写成已验证性能。


### 2026-10-17 23:39 UTC：ROOT213 独立复算通过，继续测试图成本修复

ROOT213 完整结果：Goodput 631.177198、Raw 632.543383、wall 187.383195 秒，462/463 达标；tight 仅请求 000043 的 TPOT 40.04134 ms 略超 40 ms。三类 TPOT 中位数/P90 分别为 tight 33.194/37.405、normal 28.420/32.017、loose 40.272/42.875 ms。原生计时/256、观测/255、实际 callback wall 等口径全部保留。Eager 提升与失效计数仍均为 0，不能把本轮改进归因为实际 rolling-eager 提升。

与历史强 MC2 TP4 的单点比值约 1.4684×，仍低于 1.5×；整负载独立 AR 与最终配对实验仍未完成，未采用。worker TTFT：tight 中位数 6.990 秒/P90 11.471 秒；normal 27.134/41.027 秒；loose 93.586/115.401 秒。

独立 HOST v2 继承的辅助函数仍要求 SLO0，因此在最后的完整统计处拒绝；原结果保留。v3 只把该断言改为本轮明确启用的 SLO1，保留 target0、coalesce0 和全部原时钟、整数、元数据与清理断言。新独立审计 SHA `701f0d5eb3e27e645eb70bd3076146097fec34975e061cc2c51725179fbbcb62`。运行 receipt SHA `a336538bb1a0dc2a27f31c2c67f9a43472172ae19dd1df4c6fdd372213ac6f2a`，所有模型及冻结源未改写。见 `ROOT_V213_CURRENT_FULL463_RESULT_AND_HANDOFF.json`。

当前 ROOT216 已实际启动，guardian 53684，标签 `cost-aware-N9to16-root-v216-full463-AR-v1`，源码与归档沿用前述 1710 成员封存版本。性能时钟之后才执行独立参考，尚无新结果。


### 2026-10-18 ROOT217 槽位映射候选（未运行 NPU）

ROOT216 仍在独立资格验证中，当前没有新 Goodput。ROOT217 仅减少 target KV 槽位预检查中的逐请求 int32→int64 转换和拼接，直接读取已经按同一顺序打包的原始槽位张量；所有范围、别名、形状、布局和提交检查保留。已通过 21 项实际主机检查、28 个真实 Torch CPU 张量案例和 3 个错误输入案例，冷启动源码闭包核验通过。源码已封存（1745 个成员）；NPU 运行时还必须先与原实现逐次比较，尚未获得设备正确性或性能资格，未采用。

另对 ROOT213 原始记录按提示词分组留出进行了前缀条件接受率估计：留出 69 组、8545 轮，原分类在线估计 MAE 1.4407，前缀模型 MAE 1.4469，没有改善，未接入生产。原始记录重算的估计时间点与实际调度选择时刻存在限制，不能据此宣称提高 Goodput。证据：`budget-calibration-20261015/budget-diagnostics/ROOT_V213_GROUP_HELDOUT_PREFIX_HAZARD_GAIN_HOST.json`。


### 2026-10-18 ROOT216 完整计时后验证脚本失败

原始完整 463 请求计时得到 Goodput **631.76**、Raw **633.12**、462/463 达标（tight270/271，normal96/96，loose96/96），仍低于644.77。后续独立 AR 脚本误把原生仅输出主 rank 保存的 `generate_results` 当作每个 rank 都保存；rank0/2/3 读取 `None` 后退出，rank1 随后遇到对端退出。因此这是验证脚本的角色处理错误，不是已观察到 token 不一致，也不是7200秒超时，整轮状态为失败，未取得完整 AR 资格，未采用。独立参考没有完成首个 cohort。四卡已空闲、自有进程清空，源码及前后环境核验通过。冻结的ROOT217继承同一错误，保持未运行；下一版单独修复读取结果的角色与全 rank 阶段错误传播。


### 2026-10-18 ROOT218 实际启动

保留ROOT217的生产槽位预检查优化，单独修复后置独立AR与独立审计的结果读取角色：所有rank核对原始输入和cohort身份，读取带字节数和SHA校验的rank1完整结果，仅供比较；参考输入仍仅来自自身提示词和自己生成的历史。候选读取放入原始全rank阶段，失败时同步报告，避免其他rank继续进入新阶段。28项实际HOST检查、28个真实Torch CPU案例与3个错误输入案例通过；另用不加载真实Torch/模型的合成HOST驱动走完4rank完整参考循环及独立审计，不据此授予NPU正确性。

源目录：`budget-calibration-20261015/tree-packed-KV-preflight-rank-aware-own-AR-full-v218`。冻结SHA `fa58f2866e8fed383e2d1ed1103b2fcb72a731aab3d65060e3a7c3e350a4d260`；归档SHA `163960145a58f20f71241ce8f55c4db25abf3bc832fa44041ac1f1a7b547032d`，1874个成员。实际tag：`packed-KV-rolefix-root-v218-full463-AR-v1`。冷启动源码检查通过后已显式启动，未自动串联实验，未加入启动合批策略。当前NPU资格、完整463性能、独立AR及最终配对结果均待实际完成。

### 2026-10-18 ROOT218 全量独立AR与终审结果

Goodput **630.89**、Raw **632.26**，462/463达标（tight270/271、normal96/96、loose96/96）。槽位映射新旧实现的实际组件比较通过，正式计时使用新路径；本轮没有观察到整体提速。后置参考从自己的提示词和自己生成的历史独立生成全部463×256 token，三target rank记录一致、21项原生发现/失败计数增量全为0。结果286条完全相同、177条不相同；差异包含链式causal80/128和树形FULL40/44/48路径，不能统一归因于浮点精度，更不能称已修复。

终审验证了源码、环境、整数协议、记录与时钟重算及参考历史，`same_backend_exact_output_qualified=false`。实际进程清空、四卡空闲、前后10496/15/212核验通过。保留所有原始结果与复现条件。下一步做相同前缀、logits/KV和重复执行诊断；ROOT219虽已通过34项HOST和28个CPU案例，仍未封存、未运行，暂不继续性能叠加。证据：`budget-calibration-20261015/budget-diagnostics/ROOT_V218_CURRENT_FULL463_RESULT_AND_HANDOFF.json`。


### 2026-10-19 ROOT316 完整测量、独立 AR 与暂停记录

本轮采用单次映射准备、按请求因果收益预测，以及 v311 实测 target 分支成本约束。保留原 463 请求、泊松到达、BF16、物理卡 2/5/6/7、全局预算 192 和原输出计时。正式负载要求实际使用分支，成本规则可以跳过大批次分叉；较大批次数值能力仍完整检查。

| 方法 | Raw token/s | Goodput token/s | 达标请求 | tight / normal / loose 达标 |
| --- | ---: | ---: | ---: | --- |
| 当前最好且完整 AR 通过的树形 v257 | 686.78 | 554.76 | 374/463 | 182/271、96/96、96/96 |
| v299 | 579.42 | 254.04 | 203/463 | 13/271、94/96、96/96 |
| 本轮 v316 | 611.70 | 288.01 | 218/463 | 27/271、95/96、96/96 |

固定串行 gamma4 的对照 Goodput 为 501.64，本轮仅为其 0.574 倍；也低于已有树形 v257 和 644.77 的目标阈值，故不采用。tight 的 TPOT 中位数/P90 为 45.50/51.42 ms，高于 40 ms 门槛，导致多数 tight 请求不达标。相较 v299，本轮实际分枝调用从 1087 降到 229、总墙钟从 204.56 降到 193.77 秒，但这不是单变量实验，不能把变化全部归因于某一项。

三个 target 各 408 个独立数值案例、45 个真实 draft 案例、20 个分枝后连续性案例及全部原组件检查完成；独立参考以新 KV、原 prefill 分组和自身历史生成 463×256 token，三个 target 均为 463 条完全一致、0 条差异。独立终审通过，源码未变，前后 10496 个冻结文件、15 个外部文件及 212 个包核验通过，自有进程已退出、四卡空闲。

按照用户“把这个方案测完后先暂停”的要求，本轮完成后暂停，不再启动下一项 NPU 实验。M165/N32 与角色预算 128/64 的原型、接续计划继续保留，尚无相应设备端模型资格；最终三块九轮比较仍为 0/9，目标尚未完成。

结果：`budget-calibration-20261015/budget-diagnostics/ROOT316_FINAL_FULL_RESULT.json`；独立审计：`ROOT316_TERMINAL_INDEPENDENT_FULL_AUDIT.json`；接续记录：`ROOT_20261019_ACTIVE316_COST311_PASS_HANDOFF.json`。

<!-- ROOT_OFFICIAL021_STABILITY_20261019 -->

## 2026-10-19：官方0.21基线重复测量与继续优化

按本轮要求，只复测官方 vLLM Ascend 的0.21版本线，没有启动 HUST 基线。准确版本为官方插件 **v0.21.0rc1 / 80610e44**、配套 vLLM **v0.21.0 / ad7125a4**；MC2 ON，Qwen3-32B TP4 BF16。三轮均重新启动引擎，使用原463请求和同一泊松到达轨迹；每轮四个实际worker均选择128个MC2投影模块，时间戳、达标数和共同发布时钟统计独立复核通过。

| 方法/版本（均MC2 ON） | 三轮Goodput token/s | Goodput中位数 | 全距/中位数 | Raw中位数 token/s |
| --- | --- | ---: | ---: | ---: |
| 官方0.21 | 274.32、372.91、287.63 | 287.63 | 34.28% | 920.37 |
| 官方0.23（历史三轮） | 409.20、325.26、281.38 | 325.26 | 39.30% | 923.96 |

全距/中位数 =（三轮最高值－最低值）÷三轮中位数。0.21的Goodput波动略小，但仍明显不稳定：normal的TPOT中位数为 **56.19、48.20、57.04 ms/token**，对应达标 **29、68、28/96**，跨过50 ms的SLO阈值。0.21未通过预先规定的稳定条件，因此**不采用它替换原官方参考，也不据此降低最终性能目标**。

0.21为Python3.11/CANN9.0，0.23历史记录为Python3.12/CANN9.1，可见物理卡编号不同。这是两组实际配置的波动比较，不能把全部差异归因于框架版本；最终性能对照仍须在当前同组卡上重新测量。

完整逐轮达标数、来源和统计见 [官方0.21与0.23波动报告](../official-v021-baseline-20261019/STABILITY_COMPARISON_V021_V023_zh.md) 及 [完整精度JSON](../official-v021-baseline-20261019/STABILITY_COMPARISON_V021_V023.json)。

树形优化继续按原计划推进：已冻结v317的M165/S33、最多32个真实请求的独立数值与成对成本实验；546个HOST情形通过了token顺序、KV可见性和复制映射检查，设备实验现已启动，尚无新设备资格或完整负载Goodput结论。此后仍需真实draft、原组件、预热封图和463请求独立AR检查。

提高**树形自身Raw**作为额外方向，与已有树形版本比较，不要求超过自回归基线Raw。回看v316实际记录，正式2279次target调用中1809次（79.38%）为完整gamma4主链；这是高频路径统计，不是已确认的耗时根因。后续可检查这些调用的执行成本，以及有效批次利用率、实际draft/target重叠。当前最好的已校验树形仍是v257：Raw686.78、Goodput554.76；v316未采用。

<!-- ROOT_TREE_M165_20261019 -->

### 2026-10-19：提高树形 Raw 的扩大批次试验（进行中）

- 正式基线重测使用官方插件 `v0.21.0rc1 / 80610e4` 配合官方核心 `v0.21.0 / ad7125a`，没有重测 HUST 基线。三轮 Goodput 为 274.32、372.91、287.63，仍未达到预先声明的稳定条件，因此不据此降低最终目标。
- 原优化顺序继续：先验证较大有效批次，再接入普通预算 128、提前生成预算 64、全局预算 192，之后重新完成全量正确性与性能评估。提高 Raw 是新增尝试；不要求树形 Raw 超过自回归基线的 Raw。
- M165 / 33 个 Q5 分组、最多 32 个真实请求的方案，主机端几何、KV 祖先关系检查已完成。修正了原生对照缓存写入算子仅接受 int32 索引的调用问题；144 项设备原始位检查保留候选 int32/int64 两种索引，覆盖两个 KV 宽度各全部 65,536 个 BF16 位模式，独立检查通过。
- 新模型检查在首个分支用例失败：三个 target rank 的六个 token 预测一致，但分支查询的隐藏状态和 KV 未通过原有容差检查。该方案未被采纳，没有可报告的 Raw 或 Goodput。保留失败记录：`tree-causal80-small-cohort-role-aware-audit-full-v209/receipt-M165-N32-root-v318-B1to8-546-math-cost-v1.json`。
- 控制试验 v320 把备用 NZ 权重切换阈值从 161 改为 166，但三个 target rank 的首个分支用例仍失败，差异统计与 v318 一致，不能将问题归因于 NZ。后续 v322 保留全部原检查，记录前四层中间值及前缀副本的原始位一致性，诊断结论见后文。当前 MC2 profile 确实包含 M165，不能宣称该形状没有 MC2。
- 新角色预算已通过 192 个实际绑定原预算修复器的主机检查，以及 28 个包含提前生成预算的守恒检查；仍没有新性能结论。
- 已完成全量正确性检查的最佳树形版本仍为 v257：Goodput **554.76**、Raw **686.78**。后续候选必须通过同样完整的检查后再比较。

<!-- END_ROOT_TREE_M165_20261019 -->

<!-- ROOT_FIA324_M160_20261019 -->

### 2026-10-19：分组数值诊断与 M160 后续试验

- v322 逐层诊断显示：第一层 attention 的独立 side 根查询先出现差异；进入该 attention 前相关输入一致，候选前缀副本的全部 128 个 KV 张量逐位一致。该证据缩小了问题范围，不能单凭它认定完整根因。
- v324 直接调用当前 FIA 算子，不加载模型，给各组的根查询完全相同的 Q、KV、掩码和页表。本轮 216 项检查中：24 组的 72 项、32 组的 72 项全部逐位一致；33 组的 72 项中有 24 项不同，发生在长度 1505、2555 的长前缀。掩码容量 10/11、精度模式 1/2 均复现差异。相同输入恢复后的整张输出仍一致，因此这里观察到的是对分组位置敏感，并非重复执行随机改变。
- 统一 Q11 的 v323 设备试验在参考路径的旧形状校验处中止，未产生新数值结论。v325 已修正并验证实际编译引用；鉴于独立算子试验仍揭示 33 组问题，尚未启动 v325 大模型试验。
- 后续扩大批次改为 **M160／32 个 Q5 分组／最多 32 个真实请求**，普通候选仍拟为 128、提前生成 64、全局 192。主、侧分组总数上限相应为 32。
- v326 已通过 532 项主机端 token 顺序、物理 KV、完整可见祖先和 span/alias 检查；原 408 项预算及锚点全部保留，新范围覆盖 N25～32。另通过了 144 项 160 行设备缓存原始位检查，以及实际编译参考校验器引用检查。v326 完成的 410 个模型用例在三个 target 进程上均逐位一致；随后 N25 分支用例触发实际安装策略的旧 128 验证量上限，中止且未测组件成本。v329 已将实际策略实例绑定为候选 128、包含根节点的验证量 160，并从头重跑全部 532 项；尚无新的 Raw／Goodput。
- 现有 MC2 精确形状记录包含 M165，但**不包含 M160**。M160 需要新的实际输入算子验证；不能把当前配置中的 MC2 开关当作 M160 已使用融合路径的证明。

<!-- END_ROOT_FIA324_M160_20261019 -->

<!-- ROOT_CAPACITY329_20261019 -->

### 2026-10-19：实际预算绑定修复与 Raw 后续尝试

v329 的主机预检查调用实际安装器和实际记账回调：复现旧验证量上限错误后，绑定同一个策略实例，全部 532 个预算用例通过；超过候选、请求、分组、分支和单请求预算的输入均拒绝，原策略类默认值未改。四个实际进程在加载模型前均记录候选上限 128、验证量上限 160、最多 32 组和 12 个侧分支，构造配置同为候选 128。本轮已完成全部 532 项模型检查，三个 target 进程的 hidden、全部 128 个 KV 张量、greedy token 和图/即时执行对照均逐位一致；16 组组件耗时、后置核验及退出检查现已通过独立复核。此前 v326 的 410 项结果未用于代替本轮检查。

新增 Raw 原型 v330：保持 FIA 为 160 行、32 组，只对 attention 输出投影与 FFN 下投影补 5 个零行，调用**实际 165 行**的已有 MC2 路径，再截取原 160 行结果。该做法不宣称现有 profile 已认证 160 行；暂不启用 160 行的跨层延迟同步。新增补齐、计算和存储开销可能抵消收益，需独立正确性及成对时延验证。当前仅为私有实例适配原型，尚未运行大模型，也没有性能结论。

普通预算 128、提前生成预算 64、全局预算 192 的主机原型已同步采用最多 32 个物理分组，并通过实际原预算修复器的守恒检查；尚未因此宣称完整在线调度已接入。当前最佳有效结果与最终目标保持不变。

<!-- END_ROOT_CAPACITY329_20261019 -->

<!-- ROOT329_COMPONENT_RESULT_20261019 -->

### 2026-10-19：M160／32 组 target 组件检查完成

v329 完整通过三个 target 进程各 532 个数值用例，覆盖原 408 项及新增 124 项；hidden、全部 128 个 KV 张量、greedy token、图/即时执行对照均逐位一致。每个 target 的 144 项原始位缓存写入检查通过，96 份独立已提交前缀在计时后仍一致。前后 10496 个冻结文件、15 个外部文件和 212 个包核验通过，自有进程已退出、四卡空闲。

| 真实请求 N | B=8 时侧分支请求数 | B=4 target 耗时 ms | B=8 target 耗时 ms |
| ---: | ---: | ---: | ---: |
| 4 | 1 | 71.72 | 79.12 |
| 12 | 12 | 72.43 | 82.67 |
| 24 | 8 | 72.38 | 82.00 |
| 28 | 4 | 72.52 | 81.17 |
| 31 | 1 | 72.70 | 80.22 |
| 32 | 0 | 72.42 | — |

每格先取每个 target 进程 20 次同步耗时的中位数，再取三个进程中位数的中位数；三个进程共同组成同一 TP3 执行，不能当作三轮独立实验。侧分支列表示对应 B=8 配置中分叉的请求，其余请求保持 B=4；B=4 对照中的请求均不分叉。耗时包括 target 的 pad、元数据、模型和同步，不包含真实 draft 生成。

本轮 M160 缺少精确 MC2 形状记录，实际使用普通投影路径；没有测量完整负载 Raw／Goodput，也未采纳为最终版本。下一轮 v331 保持 FIA160／32 组，只把两类投影补成实际 165 行以复用已有 MC2 路径，再从头验证全部 532 项和 16 组耗时。该轮已通过主机检查并启动，尚无设备结论。当前最好且完整独立 AR 通过的树形仍为 v257：Raw **686.78**、Goodput **554.76**；最终目标阈值 **644.77** 保持不变。

独立复核：`budget-calibration-20261015/budget-diagnostics/ROOT329_INDEPENDENT_TARGET_MATH_AND_COST_AUDIT.json`。

<!-- END_ROOT329_COMPONENT_RESULT_20261019 -->

<!-- ROOT331_REJECTED_AND333_CURRENT_20261019 -->

### 2026-10-19：非链式投影桥接未采用，转入原 chained-flush 接口的完整试验

v331 保持 FIA160／32 组，投影补成实际 165 行，运行非链式 epilogue。保留的三个 target 记录均含 532 个逐位一致的数值用例和 16 组、每组 20 次轮换耗时。32 请求／B=4 的 target 中位数为 **81.18 ms**，高于普通路径 v329 的 **72.42 ms**，未观察到提速。两轮为分别启动的组件测量，尚无完整负载 Raw／Goodput。

v331 在诊断总量达到预设 2.25 GiB 上限后被监控器以 SIGTERM 终止；正常退出文件未完成，原回执保持 failed，不能记为完整验收通过。前后源码／环境核验通过、自有进程清理完成、四卡空闲。最终数值文件保留，仅做明确标记的观察记录，不用于新版本的外部成本校准。

清理了旧 M165 编译缓存和 8 份重复数学检查点，共回收 **243,536,719 字节（约 232 MiB）**。检查点已逐字节验证可由保留的最终报告重建；保留原 SHA、字段顺序和重建条件。没有删除最终数值报告，也没有向公共区域转移代码、结果或缓存。

新完整负载源版本 v333 已通过 15 项主机检查并封存。FIA 仍为160行／32组；使用 v334 的实际165行、原 chained-flush ABI，每个投影均 flush，不启用 deferred-read-done。先用原构造配置完成全部原有控制检查，再将同一策略实例切换至候选128、含根节点验证量160；实际 LIVE 上限32、普通128／提前生成64／全局192。正式计时前会在同一个模型实例上重新测量16组 target 成本并绑定预算策略，保留全部532数值检查、真实draft45、分支后恢复20、预热封图、完整463请求与独立AR对照。

这些主机检查和源封存不构成新的设备或性能结论；当前最好且完整 AR 通过的树形仍为 v257（Raw686.78、Goodput554.76），目标阈值644.77不变。

源条件：`budget-calibration-20261015/tree-M160-C128-F64-global192-full-v333/SOURCE333_RECEIPT.json`；失败记录复核：`budget-calibration-20261015/budget-diagnostics/ROOT331_REJECTED_RUN_RETAINED_COMPONENT_OBSERVATIONS.json`；清理重建清单：`ROOT329_331_REDUNDANT_CHECKPOINT_AND_OLD_CACHE_REGENERATION_20261019.json`。

<!-- END_ROOT331_REJECTED_AND333_CURRENT_20261019 -->
