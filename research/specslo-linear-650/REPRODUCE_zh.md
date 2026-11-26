# b64 固定 γ=4 与动态 γ=0/4 的复现步骤

本文对应 `SpecSLO-Linear-650` 分支中的**可检出源码**，不是 `SpecSLO-Tree` 分支里的补丁归档。目标是按同一代码、输入、运行合同和计时定义重新测量；Goodput 是实测值，不保证跨机器逐位相同。本分支的通用环境重建步骤仍见 `/root/SpecSLO/REPRODUCTION_RUNBOOK_zh.md`；以下步骤针对当前 910B2、CANN 9.0、Python 3.11 环境。

## 1. 固定源码和本机依赖

```bash
git clone --branch SpecSLO-Linear-650 git@github.com:Renty-0/vllm-Ascend-SpecSLO.git /root/SpecSLO/specslo-linear-650-publish
cd /root/SpecSLO/specslo-linear-650-publish
git rev-parse HEAD
git status --short
```

记录实际提交；2026-11-13 三轮原实验的源码由 `SpecSLO-Chain@b8f3ad9e` 加本分支保存的代码增量构成。本分支直接包含这些源码。运行需要以下未存入 Git 的资产，必须逐项核对：

| 资产 | 当前机器路径或要求 | 已记录 SHA256 |
| --- | --- | --- |
| 请求清单 | `/root/data/specslo-baselines/nano-pearl-ascend-v1/runtime/manifest.jsonl` | `dc58dc5af1c5b28c6944f8f3faadeaae2dae34298c07b3ac84745d367c67d662` |
| MC2 profile | 同目录 `mc2-profile.json` | `ad9e0648810ed697679ee84b8bbc2939ea14bfb687d0215d27f403ad5526db1c` |
| MC2 vendor | 同目录 `mc2-vendor/op_api/lib/libcust_opapi.so` | `4b80b486b07efbc1a4e8c3d6747cd98abf5f482b112add8767c35f966be8e937` |
| Ascend 扩展 | 工作树 `vllm_ascend/vllm_ascend_C.cpython-311-aarch64-linux-gnu.so` | `f9499c28abdcfb4070f03c72ad94baea2d556ba088b8c2dbf966c9e63c1c6f23` |
| Ascend kernels 动态库 | 工作树 `vllm_ascend/libvllm_ascend_kernels.so` 和 `vllm_ascend/lib64/libvllm_ascend_kernels.so`，两处均需要 | `a8ae3257147cfed087bf3f289577068592aaae8eeeb390f2b921f73b178fb185` |
| 模型 | `/root/data/shared-models/Qwen3-0.6B` 和 `Qwen3-32B` | 使用上层手册的模型文件清单校验 |

当前机器可从之前的同源码实验目录复用已编译扩展与 kernels 动态库。若工作树中已存在对应文件，先核对 SHA，并跳过对应的 `ln -s`：

```bash
ln -s /root/SpecSLO/specslo-b64-kv640-coarse-20261113/source/vllm_ascend/vllm_ascend_C.cpython-311-aarch64-linux-gnu.so vllm_ascend/vllm_ascend_C.cpython-311-aarch64-linux-gnu.so
mkdir -p vllm_ascend/lib64
ln -s /root/SpecSLO/specslo-b64-kv640-coarse-20261113/source/vllm_ascend/libvllm_ascend_kernels.so vllm_ascend/libvllm_ascend_kernels.so
ln -s /root/SpecSLO/specslo-b64-kv640-coarse-20261113/source/vllm_ascend/lib64/libvllm_ascend_kernels.so vllm_ascend/lib64/libvllm_ascend_kernels.so
sha256sum /root/data/specslo-baselines/nano-pearl-ascend-v1/runtime/manifest.jsonl
sha256sum /root/data/specslo-baselines/nano-pearl-ascend-v1/runtime/mc2-profile.json
sha256sum /root/data/specslo-baselines/nano-pearl-ascend-v1/runtime/mc2-vendor/op_api/lib/libcust_opapi.so
sha256sum vllm_ascend/vllm_ascend_C.cpython-311-aarch64-linux-gnu.so
sha256sum vllm_ascend/libvllm_ascend_kernels.so vllm_ascend/lib64/libvllm_ascend_kernels.so
```

若扩展或 kernels 动态库不存在、SHA 不同、CANN/torch-npu 改版或换机器，必须按上层手册重建扩展和 MC2 vendor/profile，不能复用旧二进制后把结果当成同一合同。运行脚本默认使用 `/root/miniconda3/envs/vllm-hust-dev`、持久 CANN 9.0 环境和该机上的 vLLM-HUST 核心；这些依赖不在本分支中。当前环境中 `vllm` 报告版本 `0.23.1.post1.dev669+g933842ab0.dirty.empty`，`torch` 与 `torch-npu` 均为 `2.10.0`，`transformers` 为 `5.14.1`；vLLM 核心实际补丁身份与完整重建步骤见上层手册，不能仅凭版本字符串代替源码核验。两处 kernels 动态库若缺失，MC2 身份检查会在权重加载完成后失败；当前运行脚本会在启动前检查它们。

## 2. 选择卡并运行

先执行 `npu-smi info` 确认可用的四张**物理**卡和空闲状态。原三轮合同使用物理卡 `0,1,2,7`；2026-11-19 当前容器可见 `2,3,5,7`，映射器将其映射为容器设备 `0,1,2,3`。设备不同须在结果中注明，不能称为同卡重复。

```bash
cd /root/SpecSLO/specslo-linear-650-publish
SPECSLO_VISIBLE_DEVICES=2,3,5,7 \
SPECSLO_LOOSE_ADMISSION_TIGHT_GATE=6 \
bash research/specslo-linear-650/run.sh gate6-b64-r1 0.4
```

更换物理卡时只修改 `SPECSLO_VISIBLE_DEVICES`，保持四卡且不要改变卡顺序。每次重跑必须使用新的标签，如 `gate6-b64-r2`、`gate6-b64-r3`。`run.sh`、`body_candidate.sh`、环境加载、设备映射、结果机制门和活跃峰值审计脚本均在本目录；模型、扩展和 MC2 资产仍从上述路径读取。

运行合同为：Qwen3-0.6B TP1 draft、Qwen3-32B TP3 target、MC2 ON、固定线性 γ=4、463 个原始在线到达请求、每请求最多 256 token；全局活跃上限 64，单个 target home 上限 32；KV 640 页、入场预留门槛 624 页、loose 入场 tight 门槛 6、mixed target prefill 图桶 `128,256,384,512`。SLO 阈值为 tight/normal/loose `40/50/150 ms`。使用**引擎内部 paper TPOT**判定每个请求，Goodput 为达标请求的输出 token 总数除以整轮 E2E 时长；不能与主进程接收计时或 HTTP 客户端计时混算。

## 3. 核对结果

以第一轮为例，正式结果路径为 `research/specslo-linear-650/results/runs/candidate-on-gate6-b64-r1/`。先检查 `contract.env`、`input.sha256`、`git-head.txt`、`run.log`、`mechanism-gate.json` 和 `result.json`。图资格固定点必须成立，正式阶段 shape/capacity fallback 必须为零，机制门必须通过，并且全部 463 个请求各输出 256 token。随后重建实测活跃峰值：

```bash
run_dir=research/specslo-linear-650/results/runs/candidate-on-gate6-b64-r1
python research/specslo-linear-650/scripts/audit_active_peak.py \
  "$run_dir/result.json" "$run_dir/active-peak-audit.json"
python - "$run_dir/result.json" <<'PY'
import json, sys
doc = json.load(open(sys.argv[1]))
r = doc['results'][0]
p = r['slo']['paper']
print('batch_limit:', r['batch_size'])
print('raw:', r['e2e_throughput_tokens_per_second'])
print('goodput:', p['goodput_tokens_per_e2e_second'])
print('attained:', p['attained_requests'], '/', p['constrained_requests'])
print('by_class:', p['by_class'])
print('graph_fixed_point:', r['graph_qualification_fixed_point'])
print('fallbacks:', r['aclgraph_shape_fallbacks'], r['aclgraph_capacity_fallbacks'])
PY
```

原实验三轮 Goodput 为 `664.56 / 657.21 / 662.75 token/s`，Raw 为 `736.10 / 735.00 / 742.99 token/s`，tight 达标为 `226 / 222 / 221`（分母 271），normal/loose 均为 `96/96`，推断活跃峰值每轮都为 64。原始 JSON、输入哈希与门禁结果保存在 `SpecSLO-Tree/research/specslo-linear/kv640-coarse4-b64/evidence/gate6-r1..r3`。新轮次应逐项对比，保留不同卡号、源码提交、扩展 SHA 和运行时间；若数值变化，先核对合同及设备条件，不改写历史成绩。

### 2026-11-19 当前容器复测

物理卡改为 `2,3,5,7` 后，标签 `gate6-b64-retest-20261120-r2` 的完整 `result.json` 得到 Raw **740.72**、引擎内部 Goodput **681.53 token/s**，426/463 请求达标，其中 tight **234/271**、normal **96/96**、loose **96/96**；E2E 为 **160.016 s**。标签中的 `20261120` 只是运行标签，日志及请求时间戳对应 **2026-11-19 UTC**。输入 SHA 与上文一致，图资格固定点成立，正式阶段 shape/capacity fallback 均为 0。对这份结果补执行 `main-production-check-gate.py` 和 `audit_active_peak.py` 均通过，推断实际活跃峰值为 **64**。结果与两份审计文件位于 `results/runs/candidate-on-gate6-b64-retest-20261120-r2/`，SHA256 为 `8796e3508d5b5380bc30bdc15d49d0a56e225ee3f132b2890559a89b391273c2`。

该轮在 `result.json` 写完后，运行脚本因运行中被编辑而以 127 退出，导致 `run.log` 最后被错误消息覆盖；这是**测量后的封装故障**，不能把它描述为整条脚本成功退出。上述结果由完整 JSON 和补执行的门禁支持。运行期间不得修改正在执行的 shell 脚本。

随后同一物理卡、同一输入与固定 γ=4 合同执行 `gate6-b64-clean-20261119-r3`，整条 `run.sh` **退出码 0**，完整 `run.log` 保留，`mechanism-gate.json` 为 `pass`。正式结果 Raw **741.37**、引擎内部 Goodput **675.72 token/s**、E2E **159.877 s**；422/463 达标，tight **230/271**、normal **96/96**、loose **96/96**。图资格固定点成立，正式阶段 shape/capacity fallback 均为 0，单独运行 `audit_active_peak.py` 得到实际活跃峰值 **64**。`result.json` SHA256 为 `3a4787442e0591f1f08f2f06797f33d8ade19bf7fd3ea6787b01220ef29ecfa3`。该次完整复测比上一轮低 5.81 token/s，因此应保留两次观测，不把单次 681.53 当作稳定值。

另一次标签为 `dynamic-g0g4-residual-tightonly-r1` 的完整运行得到 Goodput **678.68**、Raw **739.36 token/s**，tight **233/271**、normal/loose 均 **96/96**，实际活跃峰值 **64**，`result.json` SHA256 为 `7d2f63750987119b11ab3b0023ed4702c52d8ecdf9f46ad3a7b5bea88e7e0e14`。这个标签**不能当作动态 γ 结果**：当时 `run.sh` 将传入的 `SPECSLO_RESIDUAL_EAGER_TIGHT_ONLY=1` 覆盖为 0，`contract.env` 和 worker 指标均记录 0，因此它是第三次固定 γ=4 重复测量。后来已修正脚本透传；后续动态实验须先检查 `contract.env` 和运行指标确认开关生效。

### 后续候选与未采用的结果

`gloo-buffer-reuse-gate6-r1` 仅增加 `VLLM_ASCEND_SPECRHYTHM_GLOO_PROPOSAL_REUSE_SOURCE_BUFFERS=1`，其余仍为上述固定 γ=4 合同。该轮脚本退出码 0，机制门、图资格固定点和实际活跃峰值 64 均通过；但 Goodput **668.00**、Raw **738.14 token/s**、E2E **160.576 s**，tight **227/271**、normal/loose 均 **96/96**，比同设备三次默认配置复测更低。worker 记录源缓冲区分配 43 次、复用 2845 次，因此开关确实生效；本轮不采用。原始 `result.json` SHA256 为 `cb0a5c56eedd55321411a1ea1849b219721000d27b450a05bc1e6c8a733cfb7b`。运行该候选须在上文命令前增加该环境变量，并使用独立标签，不能把它与默认配置的结果混作一组。

`tight-home-40-24-gate6-r1` 标签得到 Goodput **690.35**、Raw **741.61 token/s**、E2E **159.825 s**，tight **239/271**、normal/loose 均 **96/96**，机制门和活跃峰值 64 通过；`result.json` SHA256 为 `555c996a9186c80409ec63f28c1e292d5681c9b339f04258161a9135dd16f95e`。但这轮虽然设置了 `SPECSLO_TIGHT_HOME_FRACTION=0.625`，命令行未启用 `--spec-rhythm-slo-home-partition`，worker 明确记录分区 `requested=0`、实际 home 容量仍为 **32/32**。因此它是第四次默认策略的重复测量，不能作为 40/24 的改进证据。脚本现已加入独立的 `SPECSLO_SLO_HOME_PARTITION=1` 开关；真正的 40/24 试验必须同时设置它和 `SPECSLO_TIGHT_HOME_FRACTION=0.625`，并核对 worker 报告的 home 容量确为 40/24。

真正启用分区的 `tight-home-40-24-enabled-r1` 在相同物理卡和清单上得到 Goodput **685.14**、Raw **737.72 token/s**、E2E **160.668 s**，tight **238/271**、normal/loose 均 **96/96**。worker 报告分区 requested/enabled 均为 1，容量确为 **40/24**，两 home 分别接纳 317/146 个请求并发生 46 次跨 home 接纳；图资格、机制门和实际活跃峰值 64 均通过。`result.json` SHA256 为 `ac74caba500468f218b902b3a736d79328b14fbef961c270a982257453e01c24`。单轮低于上面的未启用分区复测，不能据此主张分区带来稳定收益。

`budget112-g0g4-gate6-r2` 保持线性固定窗口图执行，但加 `SPECSLO_VERIFICATION_BUDGET=112`，每轮被选中的请求仍使用 γ=4，预算不足的请求本轮 γ=0、后续重新选择。这是动态 **0/4** 分配，不能称为逐请求 γ=1–4。完整结果 Goodput **661.13**、Raw **739.38 token/s**、E2E **160.307 s**，tight **222/271**、normal/loose 均 **96/96**。worker 报告 168 轮发生预算推迟、累计 514 个 normal-draft 行被推迟，均为 loose；图资格和机制门通过，但实际活跃峰值为 **63**，低于当前要求的 64，Goodput 也明显下降，因此不采用。`result.json` SHA256 为 `cd9c2b27cdbced0933addfda5ee983b2e688f1ed89f0407475289df9f2b0e6d3`。同标签 `r1` 在正式测量前因实验计数器初始化位置错误触发 `KeyError`，没有有效结果；`r2` 修复后完整运行。

更偏向 tight 的 `tight-home-48-16-enabled-r1` 得到 Goodput **670.12**、Raw **731.75 token/s**、E2E **161.978 s**，tight **232/271**、normal/loose 均 **96/96**。worker 确认分区启用且容量为 **48/16**，发生 62 次跨 home 接纳；图资格、机制门及活跃峰值 64 均通过。`result.json` SHA256 为 `235eb66b3155f00cfc21c1e269c543cf790b0b54098fbfefe71fdd8ef629faba`。该单轮比 40/24 和默认配置均低，不继续增大 tight home 比例。

标签 `target-fia-workers4-gate6-r1` 得到 Goodput **694.34**、Raw **742.45 token/s**、E2E **159.645 s**，tight **241/271**、normal/loose 均 **96/96**，图资格、机制门、活跃峰值 64 通过；`result.json` SHA256 为 `672af5753b91837c28f07b30df0f9739b4b8aa04bb7e3abc364df0681f16c0f4`。但当时 `run.sh` 把传入的 `SPECSLO_TARGET_FIA_TASK_UPDATE_WORKERS=4` 覆盖为 2，worker 指标 `aclgraph_target_fia_task_update_workers=2`，所以这仍是默认 2 线程的第五次重复测量，不能归因于 4 线程。脚本后来已改为透传环境变量；真正的 4 线程实验须核对 worker 指标为 4。

修正透传后的 `target-fia-workers4-actual-r2` 得到 Goodput **648.98**、Raw **734.67 token/s**、E2E **161.336 s**，tight **217/271**、normal/loose 均 **96/96**。worker 指标确认为 4 线程，图资格、机制门和实际活跃峰值 64 通过；`result.json` SHA256 为 `ba6342a696aa3ede60839c31c6419f718c121707bcb9ec4ae9ac00a216053e5e`。虽然 target 阶段累计时间从默认复测的约 89.70 s 降到 84.77 s，整轮时长及 Goodput 反而变差，不采用 4 线程。

设置 `SPECSLO_NORMAL_ACTIVE_CAP=8` 的 `normal-cap8-gate6-r1` 得到 Goodput **694.81**、Raw **739.53 token/s**、E2E **160.275 s**，tight **243/271**、normal/loose 均 **96/96**；worker 确认 normal cap=8 且限流发生 1364 次，图资格、机制门和实际活跃峰值 64 均通过。`result.json` SHA256 为 `8b08e63a2643458703187ffedc5095ae93d9d06c8dbc20dc98f72cd157585a6e`。这只是一轮候选，Goodput 与默认配置最高单轮 694.34 很接近，尚不能声称稳定提升。

随后仅把限流判断中每个候选重复扫描待入场队列的实现改为每次决策缓存类别计数，`normal-cap8-cached-pending-r2` 得到 Goodput **687.82**、Raw **740.61 token/s**、E2E **160.042 s**，tight **238/271**、normal/loose 均 **96/96**，normal 延期计数 1323，图资格、机制门、实际活跃峰值 64 均通过；`result.json` SHA256 为 `33ef187328f74f0213e169abd3512998e14a864384c328dea4f7fc670f463a1c`。两轮生成输出哈希不同；核对另外两次未改调度的固定配置重复测量，输出哈希也不同，因此不能把跨轮 token 差异单独归因于这个缓存改动。0.23 s 的 E2E 缩短没有转化为 Goodput 提升，不能据此认定缓存扫描是有效优化。

更严格限制 normal 活跃请求数的 `normal-cap4-cached-pending-r1` 使用 `SPECSLO_NORMAL_ACTIVE_CAP=4`，其余保持固定 γ=4、loose 入场门槛 6。复现命令为 `SPECSLO_VISIBLE_DEVICES=2,3,5,7 SPECSLO_LOOSE_ADMISSION_TIGHT_GATE=6 SPECSLO_NORMAL_ACTIVE_CAP=4 bash research/specslo-linear-650/run.sh normal-cap4-cached-pending-r1 0.4`；重跑时必须换一个未用标签。该轮 Goodput **672.21**、Raw **725.48 token/s**、E2E **163.378 s**；tight **248/271**、normal **85/96**、loose **96/96**，总计 **429/463** 达标。worker 确认 normal cap=4，发生 22375 次 normal 入场延期；图资格固定点、机制门、零 shape/capacity fallback 和实际活跃峰值 64 均通过。`result.json` SHA256 为 `1bad26422351bcb81673c05373b4cd17c40129fdab5e9f4729c2e4dab5296284`。这个限流强度虽然增加了 tight 达标数，却牺牲了 normal 和 Raw，不能采用。

`budget120-g0g4-normalcap8-r1` 在 normal cap=8 下把每轮草稿验证预算设为 **120 token**；每个入选请求仍用 γ=4，未入选请求本轮 γ=0。复现命令为 `SPECSLO_VISIBLE_DEVICES=2,3,5,7 SPECSLO_LOOSE_ADMISSION_TIGHT_GATE=6 SPECSLO_NORMAL_ACTIVE_CAP=8 SPECSLO_VERIFICATION_BUDGET=120 bash research/specslo-linear-650/run.sh budget120-g0g4-normalcap8-r1 0.4`，重跑须更换标签。这轮 Goodput **675.89**、Raw **739.81 token/s**、E2E **160.215 s**；tight **231/271**、normal/loose 均 **96/96**。预算使 192 次 loose 草稿行延期，normal 入场延期 2265 次；图资格、机制门、零 shape/capacity fallback 和实际活跃峰值 64 均通过。`result.json` SHA256 为 `7b1cad8efb70b91924732a8dc82eb31a5974002e0973c164686eff5bc64efe9f`。该方案没有改善 Goodput，不能称为逐请求 γ=1–4 动态分配。

首次逐请求 γ=1–4 的 `dynamic-g1-4-b120-prefillonly-cap8-r1` 使用 `SPECSLO_MIN_GAMMA=1`、B=120、normal cap=8，但脚本仍向图审计声明启用了 mixed-target 图。可变宽度调度走的是通用 target 图，正式阶段该 mixed-target 图调用计数为 0；`--require-no-graph-fallback` 因启用图未被调用而拒绝该轮，**没有有效 `result.json` 或 Goodput**。失败日志保存在该运行目录的 `run.log`。随后脚本改为仅在固定全窗口 γ=4 时启用 mixed-target 图，可变宽度路径关闭这组未使用的图并继续要求实际执行路径图资格固定点及零回退；须用新标签重新测量。

`dynamic-g1-4-b120-generic-cap8-r2` 使用上述修正启动，但在正式阶段前被主动终止，没有结果。终止依据是前一轮第三次图重放完成至门禁报错相隔约 5 分半钟，而该阶段包含正式 463 请求测量；即使门禁修正，它与当前约 160 秒的固定 γ=4 实测路径相差过大，不适合优先冲击 700 token/s。该判断只是路径筛选，**不是**这轮的吞吐测量；可变宽度通用图的 Goodput 尚未得到有效结果。中断后确认 NPU 显存释放，转测正常长度上限 6 的固定图调度。

`normal-cap6-cached-pending-r1` 使用 `SPECSLO_NORMAL_ACTIVE_CAP=6`、loose 门槛 6、固定 γ=4；复现命令为 `SPECSLO_VISIBLE_DEVICES=2,3,5,7 SPECSLO_LOOSE_ADMISSION_TIGHT_GATE=6 SPECSLO_NORMAL_ACTIVE_CAP=6 bash research/specslo-linear-650/run.sh normal-cap6-cached-pending-r1 0.4`，重跑须换标签。Goodput **689.82**、Raw **735.92 token/s**、E2E **161.061 s**；tight **243/271**、normal **95/96**、loose **96/96**，总计 **434/463** 达标。normal 入场延期 5882 次；图资格、机制门、零 shape/capacity fallback 和实际活跃峰值 64 均通过。`result.json` SHA256 为 `b60d3329cf7ad3249c36e0126ee846b758307ecc826a234a50df34405f3e53ae`。单轮未超过 cap=8 的 694.81，也未达到 700，不能把更小的 cap 作为有效改进。

跨 2026-11-19/20 UTC 的 `gate4-normalcap8-r1` 保持固定 γ=4、normal cap=8，把 loose 入场 tight 门槛从 6 改为 4。复现命令为 `SPECSLO_VISIBLE_DEVICES=2,3,5,7 SPECSLO_LOOSE_ADMISSION_TIGHT_GATE=4 SPECSLO_NORMAL_ACTIVE_CAP=8 bash research/specslo-linear-650/run.sh gate4-normalcap8-r1 0.4`，重跑须换标签。Goodput **653.15**、Raw **720.01 token/s**、E2E **164.619 s**；tight **228/271**、normal/loose 均 **96/96**。图资格、机制门、零 shape/capacity fallback 和实际活跃峰值 64 均通过；`result.json` SHA256 为 `29f9a937dcaf48d42e324c6ccea9fa986d5a8abd8bc8cb2a900262b98c222a70`。过度延后 loose 使 tight 和总吞吐都变差，不采用。

`acceptance-floor05-normalcap8-r1` 恢复 loose 门槛 6、normal cap=8，把 rolling-eager 接受率门槛从 0.4 调到 0.5；复现命令为 `SPECSLO_VISIBLE_DEVICES=2,3,5,7 SPECSLO_LOOSE_ADMISSION_TIGHT_GATE=6 SPECSLO_NORMAL_ACTIVE_CAP=8 bash research/specslo-linear-650/run.sh acceptance-floor05-normalcap8-r1 0.5`，重跑须换标签。Goodput **654.50**、Raw **737.31 token/s**、E2E **160.757 s**；tight **219/271**、normal/loose 均 **96/96**。图资格、机制门、零 shape/capacity fallback 和实际活跃峰值 64 均通过；`result.json` SHA256 为 `e942f72add008762e1a5b840a549752f99ad11cb65c90955bb80d72453dbdea7`。worker 记录 W 窗口内实际接纳 eager 行 **14888**，而 0.4 门槛的 cap=8 轮为 **20407**；更高门槛大幅减少 eager 草稿，却使 tight 达标减少，不采用。

`acceptance-floor03-normalcap8-r1` 保持其余合同，只把接受率门槛降到 0.3；复现命令为 `SPECSLO_VISIBLE_DEVICES=2,3,5,7 SPECSLO_LOOSE_ADMISSION_TIGHT_GATE=6 SPECSLO_NORMAL_ACTIVE_CAP=8 bash research/specslo-linear-650/run.sh acceptance-floor03-normalcap8-r1 0.3`，重跑须换标签。Goodput **666.42**、Raw **739.94 token/s**、E2E **160.187 s**；tight **225/271**、normal/loose 均 **96/96**。W 窗口内实际接纳 eager 行升至 **27218**；图资格、机制门、零 shape/capacity fallback 和实际活跃峰值 64 均通过。`result.json` SHA256 为 `ccfcc7dbbd5649f45ba52e1fdbcbac2c9dedcb8fc004b536ccad0204e8d87829`。0.3 与 0.5 均低于 0.4 的 cap=8 轮；增加或减少 eager 工作都不能单独解释 40 ms tight 达标。

`fia-workers3-normalcap8-r1` 保持接受率门槛 0.4、loose 门槛 6、normal cap=8，只把 target attention task 更新线程数设为 3；复现命令为 `SPECSLO_VISIBLE_DEVICES=2,3,5,7 SPECSLO_LOOSE_ADMISSION_TIGHT_GATE=6 SPECSLO_NORMAL_ACTIVE_CAP=8 SPECSLO_TARGET_FIA_TASK_UPDATE_WORKERS=3 bash research/specslo-linear-650/run.sh fia-workers3-normalcap8-r1 0.4`，重跑须换标签。三个 target rank 的 worker 指标都确认实际线程数为 3。Goodput **666.42**、Raw **738.16 token/s**、E2E **160.572 s**；tight **226/271**、normal/loose 均 **96/96**。target 阶段累计约 **85.34 s**，但整轮收益没有提高；图资格、机制门、零 shape/capacity fallback 和实际活跃峰值 64 均通过。`result.json` SHA256 为 `c9e285ea3722ea1e8a5a45316b128adf8ffea148245a52dbe3bed3864d91fc4e`，不采用。

`eager-class-bias005-cap8-r1` 在原 0.4 接受率门槛上设置 `SPECSLO_EAGER_CLASS_FLOOR_BIAS=0.05`，使 rolling-eager 候选门槛按类别成为 tight **0.35**、normal **0.4**、loose **0.45**；未通过门槛的候选本轮不额外生成 γ=4 草稿，默认 bias=0 保持原行为。复现命令为 `SPECSLO_VISIBLE_DEVICES=2,3,5,7 SPECSLO_LOOSE_ADMISSION_TIGHT_GATE=6 SPECSLO_NORMAL_ACTIVE_CAP=8 SPECSLO_EAGER_CLASS_FLOOR_BIAS=0.05 bash research/specslo-linear-650/run.sh eager-class-bias005-cap8-r1 0.4`，重跑须换标签。Goodput **684.17**、Raw **738.39 token/s**、E2E **160.522 s**；tight **237/271**、normal/loose 均 **96/96**。worker 记录 bias 和四类门槛，W 窗口内接纳 eager 行 **21850**；图资格、机制门、零 shape/capacity fallback 和实际活跃峰值 64 均通过。`result.json` SHA256 为 `a73954e650c63e8e8b3ffd3cce9f1a9bd6a895fcfe043b6d9aeabfbb26ca92eb`。单轮未超过无偏置 cap=8 的 694.81，尚无有效提升证据。

### 2026-11-20：动态 γ=0/4、B=128 的 b64 结果

在物理 NPU `2,3,5,7`、相同 463 请求清单和 40/50/150 ms 引擎内部 paper TPOT 口径下，基于固定 γ=4 图路径增加三项参数：normal 活跃上限 8、按类别调整 rolling-eager 接受率门槛，以及每轮全局验证候选预算 `B=128`。基础门槛仍为 0.4，类别偏置 0.1 使 tight/normal/loose 的门槛分别为 0.3/0.4/0.5。正常推进草稿优先保留；预算不足时，额外 eager 草稿按 SLO 类别、预计进度缺口和紧迫度取舍。被延后的请求本轮 γ=0，获准请求仍使用完整 γ=4，**不是**逐请求 γ=1–4。全局活跃上限仍是 64，单次 target 上限仍是 32；`B=128` 恰好覆盖 32 条请求各 4 个候选。

```bash
cd /root/SpecSLO/specslo-linear-650-publish
SPECSLO_VISIBLE_DEVICES=2,3,5,7 \
SPECSLO_LOOSE_ADMISSION_TIGHT_GATE=6 \
SPECSLO_NORMAL_ACTIVE_CAP=8 \
SPECSLO_EAGER_CLASS_FLOOR_BIAS=0.1 \
SPECSLO_VERIFICATION_BUDGET=128 \
bash research/specslo-linear-650/run.sh dynamic-g0g4-b128-classbias010-cap8-new 0.4
```

标签必须换成未用过的值。该命令的其他参数由 `run.sh` 和 `body_candidate.sh` 固定，尤其是 MC2 ON、Qwen3-0.6B TP1 draft、Qwen3-32B TP3 target、γ 最大值 4、KV 640 页、最多 256 输出 token/请求、预编译草稿图以及严格图资格门。运行后按上文的方法检查 `contract.env`、`result.json`、`mechanism-gate.json`、`active-peak-audit.json` 和各输入 SHA。

| 轮次 | Goodput token/s | Raw token/s | tight /271 | normal /96 | loose /96 | 实际活跃峰值 | 额外 eager 延后：tight / normal / loose |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `dynamic-g0g4-b128-classbias010-cap8-r1` | 700.73 | 744.13 | 244 | 96 | 96 | 64 | 6 / 64 / 987 |
| `dynamic-g0g4-b128-classbias010-cap8-r2` | 709.11 | 741.13 | 251 | 96 | 96 | 63 | 1 / 8 / 948 |
| `dynamic-g0g4-b128-classbias010-cap8-r3` | 701.79 | 741.85 | 246 | 96 | 96 | 64 | 0 / 8 / 871 |

三轮 Goodput 中位数 **701.79 token/s**，三轮均超过 700。第二轮配置上限仍为 64，但逐请求区间审计推断实际峰值为 **63**，不能写成峰值 64；第一、三轮的实际峰值为 64。三轮脚本退出码均为 0，图资格固定点成立、正式阶段 shape/capacity fallback 为 0、机制门为 `pass`，所有 463 个请求均输出 256 token。正常草稿预算延后计数均为 0。三轮原始 `result.json` SHA256 依次为 `cce9f01a3b206c81fa8c8b718afa6518324d83c49da4cc2d8425151bd3c0d35c`、`fc4991b75c16929ab27e06a1bbcd57dce29984945e781634cc7cae92be20053e`、`f94d185e2b73887c4f1eedb1e52dd98819f650cc54f89b90803ece22616711e7`。原始结果在本机 `research/specslo-linear-650/results/runs/candidate-on-<标签>/`，该目录被 Git 忽略；上述哈希和命令用于核验与复测，GitHub 分支不包含模型权重或大体积结果 JSON。

作为消融，固定 γ=4、相同类别偏置但**不设置 B** 的两轮 Goodput 为 703.13 和 695.73 token/s，中位数 699.43；tight 为 247、242，峰值均为 64。两轮原始 JSON SHA256 为 `1ed95163f4d6ba77bc6622fa6603d424f4639695576d81ba99eabfbab2dc53d1` 和 `73e4cd94faecd169dfa20d3a251cb8e7d4cc57c14d026467912975eabfeab81c`。另一个 B=192 候选得 682.14 token/s，但额外 eager 延后计数为 0，因此该轮**没有验证实际动态预算取舍**，不能据此推断 B=192 的收益或损失；其 JSON SHA256 为 `9505c1dc54919b689ddaa8fb4c4eef20ea257ac45be24f8b6463aebc256e3ae3`。

复测时还应做一次**预算确实生效**的审计；只设置 `B=128` 但所有延后计数为 0 的运行，不能称为动态分配效果。将 `run_dir` 改成自己的标签：

```bash
run_dir=research/specslo-linear-650/results/runs/candidate-on-dynamic-g0g4-b128-classbias010-cap8-new
python - "$run_dir/result.json" "$run_dir/mechanism-gate.json" <<'PY'
import json, sys
r = json.load(open(sys.argv[1]))['results'][0]
gate = json.load(open(sys.argv[2]))
w = next(item for item in r['worker_aclgraph_metrics'] if item['rank'] == 0)
assert gate['status'] == 'pass'
assert r['graph_qualification_fixed_point'] and not r['aclgraph_shape_fallbacks']
assert not r['aclgraph_capacity_fallbacks']
assert r['batch_size'] == 64 and r['num_prompts'] == 463
assert w['worker_spec_rhythm_eager_class_floor_bias'] == 0.1
assert w['worker_spec_rhythm_normal_active_cap'] == 8
assert w['worker_spec_rhythm_budget_deferred_normal_rows'] == 0
deferred = {name: w[f'worker_spec_rhythm_budget_deferred_eager_{name}_rows']
            for name in ('tight', 'normal', 'loose')}
assert sum(deferred.values()) > 0
assert all(w[f'worker_spec_rhythm_allocated_gamma_{name}_{depth}_rows'] == 0
           for name in ('tight', 'normal', 'loose', 'unclassified')
           for depth in (1, 2, 3))
print('Goodput:', r['slo']['paper']['goodput_tokens_per_e2e_second'])
print('eager deferred:', deferred)
PY
python research/specslo-linear-650/scripts/audit_active_peak.py \
  "$run_dir/result.json" "$run_dir/active-peak-audit.json"
```

## 4. 2026-11-26 b32/b64 同合同对照的复现

这两轮在物理卡 `1,2,3,4` 上运行，使用同一份 463 请求清单（SHA256 `dc58dc5af1c5b28c6944f8f3faadeaae2dae34298c07b3ac84745d367c67d662`）、每请求 256 输出 token、引擎内部 paper TPOT 和 40/50/150 ms 阈值。SpecSLO 的 target TP=3、draft TP=1，MC2 ON，基础线性 γ=4；全局验证预算 B=128 会在预算不足时延后额外 eager 草稿。两轮仅把**全局活跃 batch 上限**从 64 改为 32；单次 target 上限始终是 32，不可把这两个上限混为一谈。

原始测量使用提交 `8fd4c4adcd5ffd28a70b16b70bc72dfc8bd6c162` 加上本目录 [`reproduction/batch-sweep-20261126.patch`](reproduction/batch-sweep-20261126.patch)，补丁 SHA256 为 `b0c1c38cca0ef2a7a74776c4787f2e467d4f89536864f6f4a7aac27b8dcce0ec`。该补丁把 `SPECSLO_BATCH_SIZE` 接入启动脚本；**只检出上述提交而不应用补丁时，b32 命令仍会以 b64 运行**。为避免随后 b16 实验的代码修改混入对照，应在独立工作树重建原始源码：

```bash
cd /root/SpecSLO/specslo-linear-650-publish
git worktree add --detach /root/SpecSLO/repro-batch-20261126 8fd4c4adcd5ffd28a70b16b70bc72dfc8bd6c162
cd /root/SpecSLO/repro-batch-20261126
git apply /root/SpecSLO/specslo-linear-650-publish/research/specslo-linear-650/reproduction/batch-sweep-20261126.patch
git diff --binary | sha256sum
```

最后一条应输出上述补丁 SHA。按第 1 节建立并校验三处 Ascend 扩展/kernels 动态库链接；环境、模型、MC2 profile 和 vendor 必须与第 1 节相同。确认四卡空闲后分别用新标签运行，不能同时占用同一组 NPU：

```bash
SPECSLO_VISIBLE_DEVICES=1,2,3,4 SPECSLO_BATCH_SIZE=64 \
SPECSLO_LOOSE_ADMISSION_TIGHT_GATE=6 SPECSLO_NORMAL_ACTIVE_CAP=8 \
SPECSLO_EAGER_CLASS_FLOOR_BIAS=0.1 SPECSLO_VERIFICATION_BUDGET=128 \
bash research/specslo-linear-650/run.sh batch64-repro-new 0.4

SPECSLO_VISIBLE_DEVICES=1,2,3,4 SPECSLO_BATCH_SIZE=32 \
SPECSLO_LOOSE_ADMISSION_TIGHT_GATE=6 SPECSLO_NORMAL_ACTIVE_CAP=8 \
SPECSLO_EAGER_CLASS_FLOOR_BIAS=0.1 SPECSLO_VERIFICATION_BUDGET=128 \
bash research/specslo-linear-650/run.sh batch32-repro-new 0.4
```

原始轮次 b64 的 Raw/Goodput 为 **740.05/692.10 token/s**，tight/normal/loose 达标 **241/96/96**；b32 为 **705.87/673.85 token/s**，达标 **250/96/96**。原始 `result.json` SHA256 分别是 `dbd7139004b81fa42f08c334dabf4e25f3bef15f082264e735b35711c1f5f44a` 和 `af16d04abb9dead9b10fee8d58f29bf2c8ffdfd9ab12adc753ee5e13d63d10dc`。两轮都通过图资格固定点、正式阶段零 shape/capacity fallback、机制门和活跃峰值审计，实际活跃峰值分别为 64 和 32。重新运行应保留全部原始输出，比较同合同下的波动范围；不要要求输出 JSON 的哈希或 Goodput 与原轮次逐位一致。

核查新轮次时，至少检查 `contract.env` 中的 `batch_size`、`target_max_batch=32`、`verification_budget=128`、`normal_active_cap=8`、`eager_class_floor_bias=0.1`，并运行第 3 节的统计与 `audit_active_peak.py`。b32/b64 比值所用的官方 vLLM Ascend v0.21.0 TP4 自回归基线是**另一组运行**；其结果和数据位置见 `/root/SpecSLO/comparison-batches-20261126/REPORT_zh.md`，不能用 SpecSLO 命令代替基线。

### 同源码 b64 新复跑

2026-11-26 在独立工作树运行 `batch64-repro-20261126-r2`，`working-tree.patch` 的 SHA256 与原始补丁一致。新结果 Raw **740.97**、Goodput **689.75 token/s**，tight/normal/loose 达标 **239/96/96**；相较原始 b64 的 Raw **740.05**、Goodput **692.10**、**241/96/96**，Goodput 低 **0.34%**，tight 少 2 个请求。463 个请求均输出 256 token，图资格固定点、正式阶段零 shape/capacity fallback、机制门和活跃峰值审计均通过，实际峰值 **64**。新 `result.json` SHA256 为 `18af66944e65f0381503ae86a8bd7e45c82832cd35585f5e748f7a079989b379`。这证明在同源码、同输入和同设备合同下可复现 b64 的性能区间；数值不要求逐位一致。

## 5. Llama 3.1 70B / 3.2 1B 的 b64 迁移

70B TP4、MC2 OFF 的官方 vLLM Ascend v0.21.0 baseline 独立测三轮并冻结 Goodput 均值 **122.58 token/s**。SpecSLO 使用 1B TP1 draft + 70B TP3 target、固定 γ=4、同一 463 请求与 40/50/150 ms 引擎内部 TPOT 合同；实际活跃峰值达到 64 的同配置两轮 Goodput 为 **230.42、225.01 token/s**，相对冻结 baseline 分别为 **1.880×、1.836×**，均超过 1.5×。固定 γ=4 的 1B token 级接受率约 **57%**。完整命令、冻结基线摘要、门禁、峰值审计及原始结果哈希见 [`reproduction/llama70b/REPRODUCE_zh.md`](reproduction/llama70b/REPRODUCE_zh.md)。
