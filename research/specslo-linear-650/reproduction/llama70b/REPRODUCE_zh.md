# Llama 70B / 1B 的 SpecSLO 1+3 复现记录（`llama-b64-vchain`）

**当前有效结论仅针对 b64。** 官方 vLLM Ascend v0.21.0 baseline 三轮 Goodput 均值为 **122.58 token/s**；同输入的 SpecSLO 双 batch 候选两轮为 **230.42、225.01 token/s**，均值 **227.71 token/s（1.858×）**，两轮实际活跃峰值均为 64。b32 仅完成一轮官方 baseline（Goodput 83.62 token/s）；第二轮按用户要求中止，b16/b8 未启动，故这些档位没有冻结基线和 1.5× 结论。详情见 [BATCH_SWEEP_zh.md](BATCH_SWEEP_zh.md)。

当前发布候选为 b64、**双 batch rolling eager（`auto`）**、固定 γ=4、target 单轮上限 16、tight/normal 各最多 8 个活跃请求、KV 640 页且入场保护线 639 页。该目录的 [`run_candidate.sh`](run_candidate.sh) 固定这些参数，`gate.py` 使用同目录冻结的三轮官方 baseline 均值与输入合同审计结果。确认物理 NPU `1,2,3,4` 空闲后，从仓库根目录以新标签运行：

```bash
SPECSLO_RESULT_ROOT=/root/SpecSLO/llama-pair-20261126/matched-results \
bash research/specslo-linear-650/reproduction/llama70b/run_candidate.sh llama70b-b64-repro-new
```

自动门禁通过后，再运行 `python research/specslo-linear-650/reproduction/llama70b/audit_active_peak.py <result.json> --output <active-peak-audit.json>` 独立核对实际峰值。原始大型 `result.json` 留在本机的各运行目录，未加入 Git；下文保留其路径和 SHA256。历史命令中的 `/root/SpecSLO/llama-pair-20261126/gate_llama.py` 与此目录的 `gate.py` 检查同一冻结合同，后者已对 r21/r22 的原始结果重审通过。**以下按时间排列的实验记录保留了当时的“下一轮”计划；这些不是当前运行队列，现已按用户要求暂停测试。**

复现依赖本机 CANN 9.0、Python 3.11、torch/torch-npu 2.10.0 和仓库上层 [通用复现说明](../../REPRODUCE_zh.md) 的扩展构建与环境脚本。`run_candidate.sh` 调用仓库内的 `run.sh`、`body_candidate.sh` 和源码；模型、请求清单、编译产物和 CANN 不随 Git 分支分发。使用不同设备或软件版本时应记录实际差异，不能把数值偏差直接解释为代码回归。

官方 baseline 的复现入口也已随分支保存：`run_baseline_v021.sh`、`run_baseline_v021_inner.sh` 和 `freeze_baseline.py`。它们使用 **另外检出的**官方 vLLM Ascend 源码 `80610e4438dba05011b05f89fc45d91e96992671`、vLLM 核心 `ad7125a431e176d4161099480a66f0169609a690`，以及 `/root/SpecSLO/official-v021-baseline-20261112/harness/examples/benchmark_nano_pearl_target_only.py`（SHA256 `70b06cbfcd746e9dc5fc5aa99deb26735aa7f13f5d27566fbb93df30d7f0bb29`）；这些外部源码和 harness 不在本分支中。环境入口 `/root/SpecSLO/budget-calibration-20261015/serial-v1/runtime.sh` 的 SHA256 为 `107a11a58764eabe4f8b6e5d752d9234cf67f9e898d81a8619e47e684e83af7c`。原 baseline 环境记录为 vLLM 0.21.0、torch/torch-npu 2.10.0、transformers 5.5.4。脚本在启动前核验两个 commit，并为每轮使用独立编译缓存。原三轮的运行环境、原始结果路径和 SHA256 已写入本目录的 [冻结 baseline JSON](baseline-v021-llama70b-b64-three-run-mean.json)。在同一机器补测时，按新标签执行：

```bash
SPECSLO_VISIBLE_DEVICES=1,2,3,4 bash research/specslo-linear-650/reproduction/llama70b/run_baseline_v021.sh new-b64-r4 64
```

运行前须确认上述外部目录、共享模型和请求清单存在；输出写到本目录被 Git 忽略的 `baseline-results/runs/`。不要以新一轮结果替换已冻结的三轮均值。

目标模型位于公共区 `/data/shared_upload/models/Meta-Llama-3.1-70B-Instruct`，草稿模型位于 `/data/shared_upload/models/Llama-3.2-1B-Instruct`。70B 的 30 个 safetensors 分片均已能打开，索引中的 723 个 tensor 键均存在；1B 权重 SHA256 为 `1ff795ff6a07e6a68085d206fb84417da2f083f68391c2843cd2b8ac6df8538f`。两模型的 `tokenizer.json` SHA256 均为 `79e3e522635f3171300913bb421464a87de6222182a0570b9b2ccba2a964b2b4`。70B `config.json` 与索引 SHA256 分别为 `fa6e9124e4621df77aecf96fbfaf7975814013d2d5ab1c972e965000588a9749`、`2abe0910e23770a30ccf9b1b91804c64831c47f9c98defaa5293aa999433fc2b`。

测试使用物理 NPU `1,2,3,4`：1B draft TP1 占一张卡，70B target TP3 占三张卡。请求清单与 Qwen b32/b64 对照完全相同，SHA256 `dc58dc5af1c5b28c6944f8f3faadeaae2dae34298c07b3ac84745d367c67d662`；每请求 256 输出 token，SLO 为 40/50/150 ms，按引擎内部 paper TPOT 计时。由于 Llama 70B 没有该 Qwen MC2 profile，本轮 MC2 OFF，不能直接把其吞吐与 Qwen 的 MC2 ON 结果当成同模型比较。

## 2026-11-26 固定的 70B b64 baseline

官方 vLLM Ascend v0.21.0、70B TP4、MC2 OFF、`max_num_seqs=64`，在相同的物理 NPU `1,2,3,4` 上独立跑三轮。三轮均使用同一请求清单和同一输入 token（SHA256 `2327fc4be1069c109fda2450767b5e0e5230a64ae7c6ba74fe38ea22cc754631`），463 个请求各输出 256 token，按引擎内部 paper TPOT 计算 Goodput。

| 轮次 | Raw (token/s) | Goodput (token/s) | tight / normal / loose 达标 |
| --- | ---: | ---: | --- |
| r1 | 593.46 | 123.05 | 0 / 0 / 96 |
| r2 | 594.05 | 123.17 | 0 / 0 / 96 |
| r3 | 586.07 | 121.52 | 0 / 0 / 96 |

**后续固定基线为三轮 Goodput 的算术平均 122.58062376295965 token/s，1.5× 门槛为 183.87093564443947 token/s。** Raw 算术平均为 591.1961333567741 token/s。原始结果保存在本机 `/root/SpecSLO/llama-pair-20261126/baseline-results/runs/`，其 SHA256 和合同见本目录的 `baseline-v021-llama70b-b64-three-run-mean.json`；上文命令是分支内可检出的复现入口。

比较 SpecSLO 时必须加 `SPECSLO_PROMPT_TOKENIZER_MODEL=target`：1B 和 70B 虽使用相同 tokenizer 词表，但 chat template 不同。下文 `production-results/...-r2` 是发现差异前的历史试验，其输入 token SHA256 为 `31bc330452ab7391202f0cd618957cd4a63fa2c7ad1a6a903313edfb140a4e6e`，不能当成与上述 baseline 完全同输入的公平对照。输入对齐后的新试验保存在 `matched-results/`。

短测 `runs/smoke-b8-16tok-kv128-graphs32` 已通过：8 个请求全部输出 256 token（请求清单中的 `max_tokens=256` 优先于启动参数，因此目录名中的 `16tok` 不是实际输出长度），图资格达到固定点，正式阶段 shape/capacity fallback 均为零。短测 Raw 为 157.53、Goodput 为 137.84 token/s，仅用来验证模型加载、1+3 通信、Llama 3 RoPE 和图路径，不作为 b64 成绩。初次短测的 4 个图条目不足，后改为 32；70B 从共享存储加载超过 HCCL 默认 120 秒建连等待时，将 `HCCL_CONNECT_TIMEOUT` 设为 900 秒。

正式 b64 的当前运行目录为 `production-results/runs/candidate-off-llama70b-b64-dynamic-b128-r2/`。源码身份是 `SpecSLO-Linear-650@8fd4c4adcd5ffd28a70b16b70bc72dfc8bd6c162` 加该目录的 `working-tree.patch`（SHA256 `8a1f34f0278f04d05da23b1db1f063b4ce8d82fe4e314a0ca03512c5dcc40fad`）。启动合同见同目录 `contract.env`。核心参数：b64、target 单次上限 16、KV 512 页、图条目 32、γ=4、全局验证预算 B=128、normal 活跃上限 8、eager 类别偏置 0.1、loose 入场 tight 门槛 6。B=64 的先前运行在生成前因不足以覆盖双 batch 的常规草稿而失败，没有性能结果。

在四张卡均空闲、源码与依赖一致时，使用新标签复测：

```bash
cd /root/SpecSLO/specslo-linear-650-publish
SPECSLO_RESULT_ROOT=/root/SpecSLO/llama-pair-20261126/production-results \
SPECSLO_VISIBLE_DEVICES=1,2,3,4 SPECSLO_BATCH_SIZE=64 \
SPECSLO_DRAFT_MODEL=/data/shared_upload/models/Llama-3.2-1B-Instruct \
SPECSLO_TARGET_MODEL=/data/shared_upload/models/Meta-Llama-3.1-70B-Instruct \
SPECSLO_MC2_MODE=off SPECSLO_HCCL_CONNECT_TIMEOUT=900 \
SPECSLO_NUM_KVCACHE_BLOCKS=512 SPECSLO_ADMISSION_KV_GUARD_PAGES=496 \
SPECSLO_MAX_ACLGRAPH_ENTRIES=32 SPECSLO_MAX_TARGET_BATCH=16 \
SPECSLO_TARGET_NZ_MODE=default SPECSLO_LOOSE_ADMISSION_TIGHT_GATE=6 \
SPECSLO_NORMAL_ACTIVE_CAP=8 SPECSLO_EAGER_CLASS_FLOOR_BIAS=0.1 \
SPECSLO_VERIFICATION_BUDGET=128 SPECSLO_PROMPT_TOKENIZER_MODEL=target \
bash research/specslo-linear-650/run.sh llama70b-b64-dynamic-b128-new 0.4
```

主脚本的 Qwen 机制门要求 MC2 ON，不能原封不动用于这轮 Llama MC2 OFF 结果。正式测量写出 `result.json` 后，应独立核查 `enable_mc2=false`、463 请求、每请求 256 token、图资格固定点和零图回退，并运行 `audit_active_peak.py` 重建实际活跃峰值。若机制门因 MC2 条件失败，保留原始结果并明确标记门禁不适用，不可把脚本退出码当成性能测量结果。

## 2026-11-26 正式结果

`candidate-off-llama70b-b64-dynamic-b128-r2/result.json` SHA256 为 `e49c5e1d0ba962a2b69f0493beb987f2d85b168e754ebfea6a782480f630ada3`。463 个请求全部输出 256 token；Raw **421.28 token/s**，Goodput **66.42 token/s**，达标 **73/463**（tight **0/271**、normal **24/96**、loose **49/96**）。E2E 测量耗时 **281.35 s**。图资格达到固定点，正式阶段 shape/capacity fallback 均为 0；独立 `active-peak-audit.json` 状态为 `pass`，重建活跃峰值 **64**。预算确实生效：draft rank 记录额外 eager 延后 tight 1 行、normal 0 行、loose 4 行。

该模型在这组 40/50/150 ms 阈值下的平均 TPOT 分别为 tight **114.66 ms**、normal **68.92 ms**、loose **164.57 ms**，因此吞吐与 Goodput 差距很大。这个结果只能描述 70B+1B、MC2 OFF 的当前配置，不能用于声称 Qwen3-32B 的 650 token/s 目标已经达成或未达成。主脚本最后返回 2，是因为 Qwen 专用门禁只报告 4 条 MC2 ON 条件未满足；测量 JSON、零图回退和活跃峰值审计仍完整保存。

## 同输入固定 γ=4 对照

`matched-results/runs/candidate-off-llama70b-b64-matched-fixed4-r4/result.json` 使用 70B target 的 chat template，输入 token SHA256 与三轮 baseline 相同；b64、1B TP1 + 70B TP3、γ=4、B=128、每轮 target 最多 16 个请求。Raw **422.08 token/s**，Goodput **56.52 token/s**，相对冻结的 baseline 为 **0.461×**。达标 **62/463**（tight 0、normal 17、loose 45）；1B 草稿的 token 级接受率 **57.22%**，定义为 target 接受草稿 token 数除以验证草稿 token 数。图资格达到固定点，正式阶段 shape/capacity fallback 均为 0。

原 Llama 门禁将 `Path("auto")` 和字符串 `"auto"` 比较，误报“Expected the dual-batch rolling mechanism”；修正这一类型错误后，对已落盘的同一测量重审，`gate-fixed.json` 为 `pass`，测量数据没有重算或改写。后续优化统一对照三轮 baseline 的 **122.58 token/s** 平均值，达标线为 **183.87 token/s**。

增大每轮 target 验证上限的第一轮 `matched-results/runs/candidate-off-llama70b-b64-matched-target32-b128-g4-r5/` 保持 KV 512 页、B=128、γ=4，将请求上限从 16 改到 32；图预热时 target 卡显存不足，日志记录仅余约 46 MiB、下一次分配需要 90 MiB。这轮没有正式性能测量，不能纳入 Goodput 对照。

上述 KV 384 页尝试在初始化时因小于 463 个排队请求加图暂存行的结构性下限而被拒绝，也没有性能结果。基于同输入 γ=4 对照中估计的 KV 峰值 452 页，后续改用 KV 480 页、入场保护线 464 页验证 target 上限 32；这仍保留对外 b64 请求上限。

KV 480 页的 `...-target32-b128-kv480-r7/` 仍在较大图的注意力预热中 OOM（需要 90 MiB，target 卡当时约有 64 MiB 空闲），没有正式性能结果。动态 γ 的可变宽度路径会关闭固定 γ 使用的混合 target 图；下一组改测 γ=2–6、B=128、target 上限 32，以同时验证预算分配和显存路径。

动态 γ 首轮 `...-dyng2to6-t32-b128-r8/` 越过图预热，但在资格回放中 draft 卡 KV cache 的 512 页耗尽（61 个活跃行，压力估计 478），因此没有正式结果。1B draft 卡仍有大量显存，70B target 卡接近显存上限；为避免增加 target 的 KV 常驻内存，源码加入可选 `VLLM_ASCEND_PEARL_DRAFT_KVCACHE_BLOCKS`，下一轮仅将 draft KV 增至 1024 页，target 保持 512 页。默认值 0 仍沿用原统一页数。

`...-dyng2to6-t32-b128-dkv1024-r9/` 完整通过图资格并测得零图回退，但 Raw **249.89**、Goodput **8.64 token/s**，仅 **16/463** 达标（全为 loose），1B token 接受率 **44.36%**。相对同输入固定 γ=4 的 56.52 Goodput 明显退步；target 阶段约 308 秒、refill 约 125 秒。这是一个有效的负面实验，不纳入最优方案。下一组固定 γ=2、target 请求上限 32、B=64，让验证 token 的理论上限仍为 64，同时检验更短草稿和更宽请求覆盖。

固定 γ=2、target 上限 32、B=64 的 `...-fixedg2-t32-b64-r10/` 在图预热时 OOM，没有性能结果。异常前 target 卡约有 115–118 MiB 空闲，仍无法稳定分配新的 90 MiB 注意力缓冲；下一轮将 target KV 降为 480 页、draft KV 设为 1024 页，其余保持不变，以验证较小 KV 预留能否让图路径通过。

`...-fixedg2-t32-b64-kv480-r11/` 在图预热时出现 NPU vector core 异常并触发 HCCL 退出，日志没有明确 OOM 证据；同样没有正式性能结果。后续回到已通过的每轮 target 请求上限 16，单独测试固定 γ=2、B=64，避免把无效的 32 请求图当成可行路径。

固定 γ=2、target 上限 16、B=64 的 `...-fixedg2-t16-b64-r12/result.json` 完整测得 Raw **328.91**、Goodput **13.50 token/s**、达标 **19/463**（tight 0、normal 1、loose 18），明显低于固定 γ=4。token 接受率虽升至 **72.20%**，验证轮数却从 2858 增至 **4264**，跨卡回传约 **156.92 秒**，抵消了单次草稿缩短的收益。图资格固定点和 shape/capacity 回退均为正常；本地门禁原先误把 `aclgraph_draft_pa_parallel_update_fallback_replays_delta`（图内 task 更新策略回退）当作图执行回退，修正后对同一结果重审的 `gate-fixed.json` 为 `pass`。下一轮回到固定 γ=4，单独开启 target replay-ahead。

固定 γ=4、target 上限 16、B=128，仅开启 target replay-ahead 的 `...-matched-g4-replayahead-r13/result.json` 完整测得 Raw **415.29**、Goodput **56.51 token/s**、达标 **63/463**（tight 1、normal 21、loose 41），相对同输入 r4 的 Goodput **56.52** 没有实际提升。2792 次提前回放尝试中，仅 143 次满足协议、前沿不变和资源条件并成功提交、消费；其中 1770 次因前沿可变而拒绝。因此该优化在目前调度下覆盖率不足。图资格固定点和 shape/capacity 回退仍为正常，机制门 `pass`。下一轮保持固定 γ=4，开启 target SLO 优先，单独验证按 tight 优先挑选 target 请求是否提升达标数；`run.sh` 固定开启的 drain FIFO 仅在不可变尾段生效。

仅开启 target SLO 优先的 `...-matched-g4-target-slo-first-r14/result.json` 测得 Raw **420.33**、Goodput **17.25 token/s**、达标 **19/463**（tight 3、normal 0、loose 16），比 r4 更差。图资格固定点、正式阶段零 shape/capacity 回退，独立门禁 `gate-fixed.json` 为 `pass`；`active-peak-audit.json` 重建的实际活跃峰值仍为 **64**。测试时曾编辑同一个运行脚本，导致结果写出后 shell 包装层报退出码 127，且原 `run.log` 被截断；原始 JSON 已完整保存并独立核查。这组负面结果只用于筛选方向，不能提升为发布成绩。下一轮在相同的固定 γ=4 配置下，增加默认关闭的 tight 活跃上限 16，并允许 loose 填充其余活跃名额，直接验证是否能让 tight 连续获得验证机会。

`audit_active_peak.py` 按逐请求到达偏移、入场等待和解码持续时间重建半开活跃区间；它对历史 `production-results/...-r2` 重建出的峰值 64 与原 `active-peak-audit.json` 完全一致。复核新结果可运行 `python audit_active_peak.py <result.json> --output <active-peak-audit.json>`。

默认关闭的 tight 活跃上限设为 16，放开 loose 入场，并开启 target SLO 优先的 `...-matched-g4-tightcap16-r15/result.json` 测得 Raw **419.29**、Goodput **84.22 token/s**、达标 **93/463**（tight 52、normal 26、loose 15），1B token 接受率 **57.04%**。tight 逐请求 TPOT 中位数降至 **46.42 ms**，比无活跃上限的 r14 更接近 40 ms。输入 token SHA256 与冻结基线相同，图资格固定点和零回退门禁 `pass`；活跃峰值审计仍为 **64**。虽然比 r4 的 56.52 提升，尚未超过基线平均 122.58，更未达到 1.5× 的 183.87。下一轮改用脚本支持的 target 单轮上限 8 与 tight 活跃上限 8，检验更小验证批次能否进一步降低 TPOT。原先尝试 12/12 被启动脚本的合法档位检查拒绝，没有占用 NPU，也没有性能结果。

全程 target 上限 8、tight 活跃上限 8 的 `...-matched-g4-tightcap8-t8-r16/result.json` 测得 Raw **280.21**、Goodput **126.49 token/s**、达标 **209/463**（tight 185、normal 22、loose 2），1B token 接受率 **57.52%**。tight 逐请求 TPOT 中位数 **36.85 ms**，验证轮数增至 5311，E2E **423.00 s**；图资格经过 4 次回放达到固定点，正式阶段 shape/capacity 回退均为零，机制门 `pass`。相比冻结基线平均仅为 **1.032×**，未达 1.5×。活跃峰值审计重建为 **62**，虽然配置的总 batch 上限为 64，但还不能声称实际峰值达到 64。下一轮保持 tight/在线 target 上限 8，让收尾阶段恢复 target 上限 16，同时将 KV 入场保护线从 496 页调至 512 页，以验证能否补足活跃峰值并缩短尾段。

首次在线 8 / 收尾 16 的 `...-matched-g4-tightcap8-online8-tail16-r17/` 在资格回放前被容量校验拒绝：KV 总量与入场保护线同为 512 页，没有留下至少一页空余，故无性能结果。重启的 `...-matched-g4-tightcap8-online8-tail16-kv528-r17b/` 将 KV 总量设为 528、保护线仍为 512；这仍保持对外 batch 上限 64，正式性能需待测量完成。

上述 `...-matched-g4-tightcap8-online8-tail16-kv528-r17b/result.json` 已完整测得 Raw **288.40**、Goodput **127.07 token/s**、达标 **204/463**（tight 178、normal 19、loose 7），E2E **410.99 s**、1B token 接受率 **57.31%**。在线上限触发 4639 轮，尾段恢复 16 触发 414 轮；比全程 8/8 只略微提高 Raw 和 Goodput。图资格固定点、零 shape/capacity 回退、门禁 `pass`，输入 token SHA256 与冻结基线相同；实际活跃峰值 **63**，尚未到 64。loose 的 TPOT 中位数约 **812.61 ms**，说明早入场而长期等 target 是主要问题。下一轮把 loose 入场推迟到 tight 活跃和已到达待入场请求都清零后，并适度增大 KV 可用页数以争取实际峰值 64；这同时改变调度和容量，需用完整新结果评估，不把二者拆成单因素效应。

延后 loose 到 tight 空档入场的 `...-matched-g4-tightcap8-online8-tail16-loosehold-r18/result.json` 测得 Raw **270.21**、Goodput **152.90 token/s**、达标 **262/463**（tight 240、normal 6、loose 16），E2E **438.65 s**、1B token 接受率 **57.56%**。tight 的 TPOT 中位数进一步降至 **32.80 ms**；normal、loose 中位数分别为 **67.99**、**221.20 ms**。图资格固定点和正式阶段零回退，门禁 `pass`，输入 SHA 与冻结基线一致。实际活跃峰值只有 **55**，normal 和 loose 的等待首 token 时间也显著增加；因此仍未达 batch 实际峰值 64 或 Goodput 1.5×。下一组默认关闭的 normal 硬活跃上限 8 与 loose/normal 入场门槛 0，让 tight、normal、loose 按优先级分阶段获得连续服务；KV 总量 640 页、入场保护线 624 页用于检验是否能恢复实际峰值 64。该策略可能进一步增加排队等待，须同时报告。

normal 硬活跃上限 8、loose/normal 入场门槛 0、在线 target 上限 8 的 `...-matched-g4-staged-classes-kv640-r19/result.json` 测得 Raw **275.32**、Goodput **152.83 token/s**、达标 **257/463**（tight 236、normal 7、loose 14），E2E **430.50 s**、1B token 接受率 **57.48%**。图资格固定点和零回退门禁 `pass`；独立活跃峰值审计为 **64**。但相对 r18 的 Goodput 没有提升，normal 与 loose 仍大量不达标。计数器显示 loose 因 tight 门槛被延后 307277 次，而 loose/normal 门槛延后为 0 次；前一门槛已覆盖后一门槛的作用。这一轮不能证明 normal 分阶段入场有效。下一轮保留 b64、tight/normal 活跃上限 8 和两个入场门槛，将在线 target 单轮请求上限从 8 调至 16，验证是否能让 normal 和 loose 获得更多验证机会。

在线 target 上限 16 的 `...-matched-g4-staged-fulltarget16-r20/result.json` 测得 Raw **294.45**、Goodput **231.49 token/s**、达标 **364/463**（tight 237、normal 87、loose 40），E2E **402.54 s**、1B token 接受率 **57.49%**。相对冻结基线三轮平均 Goodput **122.58** 为 **1.888×**，超过 1.5× 门槛 **183.87**。输入 token SHA 与基线一致，图资格固定点且正式阶段零 shape/capacity 回退，门禁 `pass`。与 r19 相比，normal 达标数 7→87、广播阶段 189.32→165.11 s、总解码轮数 5461→5053；但不能把整个提升只归因于广播变化。独立活跃峰值审计为 **63**：配置允许 b64，实际尚未出现同时活跃 64 个请求。下一轮仅将 KV 入场保护线从 624 提到 639（KV 总量 640），检验是否能达到实际峰值 64 且保持性能收益；之后还需独立复测稳定性。

复测 r20 时在四张 NPU 空闲、当前源码与依赖就绪的前提下使用**新的运行标签**，不要覆盖原始结果：

```bash
cd /root/SpecSLO/specslo-linear-650-publish
SPECSLO_MC2_MODE=off SPECSLO_VISIBLE_DEVICES=1,2,3,4 SPECSLO_BATCH_SIZE=64 \
SPECSLO_DRAFT_MODEL=/data/shared_upload/models/Llama-3.2-1B-Instruct \
SPECSLO_TARGET_MODEL=/data/shared_upload/models/Meta-Llama-3.1-70B-Instruct \
SPECSLO_RESULT_ROOT=/root/SpecSLO/llama-pair-20261126/matched-results \
SPECSLO_MAX_ACLGRAPH_ENTRIES=32 SPECSLO_NUM_KVCACHE_BLOCKS=640 \
SPECSLO_MAX_TARGET_BATCH=16 SPECSLO_ONLINE_TARGET_MAX_BATCH=0 \
SPECSLO_TARGET_NZ_MODE=default SPECSLO_LOOSE_ADMISSION_TIGHT_GATE=0 \
SPECSLO_LOOSE_ADMISSION_NORMAL_GATE=0 SPECSLO_ADMISSION_KV_GUARD_PAGES=624 \
SPECSLO_NORMAL_ACTIVE_CAP=8 SPECSLO_NORMAL_ACTIVE_HARD_CAP=8 \
SPECSLO_TIGHT_ACTIVE_CAP=8 SPECSLO_EAGER_CLASS_FLOOR_BIAS=0.1 \
SPECSLO_GAMMA=4 SPECSLO_MIN_GAMMA=4 SPECSLO_VERIFICATION_BUDGET=128 \
SPECSLO_TARGET_SLO_FIRST=1 SPECSLO_HCCL_CONNECT_TIMEOUT=900 \
SPECSLO_PROMPT_TOKENIZER_MODEL=target \
SPECSLO_PRODUCTION_GATE_SCRIPT=/root/SpecSLO/llama-pair-20261126/gate_llama.py \
bash research/specslo-linear-650/run.sh llama70b-b64-matched-g4-r20-repro-new 0.4
```

复核 `result.json` 时运行 `python /root/SpecSLO/llama-pair-20261126/audit_active_peak.py <result.json> --output <active-peak-audit.json>`，并保留 `contract.env`、`working-tree.patch`、`gate.json` 与模型/输入哈希。r20 结果的 SHA256 为 `89bbb13234c32e3e79e57ad6598c80ca83f9ecfab293a5bc64928d85666137b2`。

仅将 KV 入场保护线从 624 改为 639 页的 `...-matched-g4-staged-fulltarget16-guard639-r21/result.json` 测得 Raw **294.71**、Goodput **230.42 token/s**、达标 **362/463**（tight 238、normal 84、loose 40），E2E **402.18 s**、1B token 接受率 **57.30%**。相对冻结基线为 **1.880×**。输入、图资格固定点、正式阶段零回退门禁均为 `pass`；独立活跃峰值审计为 **64**。与 r20 的 231.49 token/s 很接近，说明提高保护线补齐实际峰值后仍保留 1.5× 以上的提升。这两轮仅差一个入场参数，不是完全同配置重复试验；同参数复跑结果见下段。r21 结果 SHA256 为 `933e4d2f678b84328c0e4d913020421dea88052a638db8db1cbb2ebadfc63250`。

同参数复跑 `...-matched-g4-staged-fulltarget16-guard639-r22/result.json` 测得 Raw **293.46**、Goodput **225.01 token/s**、达标 **355/463**（tight 237、normal 80、loose 38），E2E **403.90 s**、1B token 接受率 **57.06%**。图资格固定点、正式阶段零 shape/capacity 回退、门禁和实际峰值 **64** 均通过；结果 SHA256 `60742c2c8ee0bc9d957fe389180167c6c44f8b00c397579da92d96056e594a00`。r21/r22 同配置 Goodput 均值 **227.71 token/s**，相对冻结 baseline 三轮均值 **1.858×**；最差单轮 **1.836×**。两轮达标数相差 7 个，说明 TPOT 门槛附近仍有波动，不能保证逐位复现。建议以这组实际峰值 64、两轮都超过 1.5× 的配置作为当前 70B b64 发布候选，r20 留作保护线 624 的消融对照。
