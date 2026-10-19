# SpecSLO-Tree 优化代码快照

本分支保存正在这台 Ascend 机器上开发的动态预算与树状投机解码源码。
`SOURCE_MANIFEST.json` 记录每个文件的原始位置、SHA256 和字节数；
`snapshot/root/` 保持原实验目录结构。源码保留原样，未改动正在使用的运行环境。

## 当前结果与验证范围

| 版本 | 实现及验证范围 | 当前结论 |
| --- | --- | --- |
| 固定串行 gamma4 | 双 batch、rolling-eager 和已有算子、计算图优化 | 对照 Goodput 501.6356 token/s。 |
| v257 | Q5/M120 树形验证；463 请求均与独立自回归参考一致 | 当前最好树形结果：Raw 686.7756，Goodput 554.7604 token/s。 |
| v287 | 根节点分叉，动态 B=1…8；完整 463 请求和三个 target rank 的独立自回归校验通过 | Raw 510.2503，Goodput 206.0838；性能退化，未采用。原 guard 的预算上限审计过期，保留原失败记录和独立修正审计。 |
| v290 | 先为正常请求保留主链容量，再分配分叉预算 | HOST 测试通过，已集成 v299，完整负载收益待测。 |
| v294 | 按请求估计条件接受概率，处理未展开分支的观测截断 | HOST 测试通过，模型超参数和实际性能尚未验证。 |
| v295 | 所有层的连续前缀合并复制，当前 K/V 双写 | 独立 NPU 算子测试通过；不能据此宣称端到端提升。 |
| v291 | 将 v295 路径接入真实 target 模型 | 初始化时缓存检查失败，尚无模型数值或时延结果，不能采用。 |
| v296 | 按实际缓存张量解析具体 NPU 编号，再执行原有严格缓存检查 | 三个 target rank 各 408 组数值检查及成对 target 时延验证通过；尚无完整负载 Goodput。 |
| v299 | 组合主链容量保留、按请求估计收益、连续前缀复制与 K/V 双写 | 完整模型检查及三个 target rank 的 463 请求独立 AR 校验通过。Raw 579.4179、Goodput 254.0428；tight 13/271、normal 94/96、loose 96/96 达标。性能不采用，保留完整负面结果。 |
| v304/v305 | 保留一次完整映射校验，直接生成复制区间与当前 K/V 别名，省去无用缓冲 | 417 个映射场景及 408 组真实 HOST prepare 回调检查通过；已纳入 v311 冻结输入。HOST 描述准备约从 5–6 ms 降到 1–2 ms，不能作为模型提速结果。 |
| v307 | 仅将 v299 的准备路径替换为 v305 | 13 项 HOST 集成检查通过，尚未运行完整模型实验；根据 v299 的负面结果，先测 v311 的新准备路径成本，再结合分支成本判断。 |
| v308 | 以连续区间生成分支复制与补齐索引，保留原布局检查 | 472 个真实 HOST 布局场景与原结果完全一致，6 项坏输入检查通过；HOST 布局准备小幅减少约 0.25 ms，尚未集成或测试 NPU。 |
| v309 | 只有预测的优先级加权进度足以抵消分支耗时，才购买分支 | 16 个实际类型分配案例及 3,072 次合成因果观测测试通过，保留优先级和硬预算；当前暂用 v296 的 target 成本代理，尚未集成或测试 NPU。 |
| v311 | 对 v305 准备路径重新做真实 target 数值与成对成本测量 | 5 项 HOST 预检通过，源码已冻结，真实模型实验正在运行；不测真实 draft 成本或完整负载 Goodput。 |

最终目标仍是动态预算树形方案超过固定 gamma4，并超过打开 MC2 的官方
TP4 基线 1.5 倍。当前参考阈值为 644.7664 token/s，尚未达到。

## 主要代码入口

所有目录均相对于 `snapshot/root/SpecSLO/budget-calibration-20261015/`。

| 目录 | 内容 |
| --- | --- |
| `tree-static-q5-M120-SLO-first-full463-own-AR-v257/` | 当前已验证的最好树形方案。 |
| `tree-dynamic-root-alternate-eager-compat-full-v287/` | 动态根分叉、预算控制、真实 draft、完整负载与独立 AR 校验。 |
| `tree-spare-capacity-budget-host-v290/` | 保留主链容量的预算分配原型与 HOST 检查。 |
| `tree-request-conditional-risk-gain-host-v294/` | 请求条件收益模型及 HOST 检查。 |
| `tree-prefix-precopy-dual-cachewrite-host-v288/` | 原始字节复制与 K/V 双写内核。其旧的大网格前缀内核不可直接启动，见 v293 失败记录。 |
| `tree-contiguous-prefix-span-copy-host-v292/` | 连续区间压缩与安全网格前缀内核。 |
| `tree-safe-grid-span-dual-cache-probe-v295/` | 原始字节一致性和成对轮换时延验证。 |
| `tree-all-layer-prefix-dual-cache-target-cost-v291/` | 保留原失败的模型集成诊断；设备比较问题在 v296 修复。 |
| `tree-concrete-device-span-dual-cache-target-cost-v296/` | 修复设备比较后的独立模型诊断，不继承前驱的数值验证资格。 |
| `tree-capacity-request-risk-span-full-v299/` | 当前冻结的完整实验，需重新通过所有模型与独立 AR 校验。 |
| `tree-single-validated-prefix-descriptors-host-v304/` | 不可变映射校验与直接区间描述，HOST 原型。 |
| `tree-single-validated-prefix-prepare-host-v305/` | 精简 HOST prepare 回调，尚未执行 NPU 模型验证。 |
| `tree-capacity-risk-single-prepare-full-v307/` | 后续完整实验候选，尚未启动 NPU。 |
| `tree-range-packed-branch-layout-host-v308/` | 连续区间索引构造的 HOST 原型与完整差分检查，尚未接入运行版本。 |
| `tree-measured-branch-cost-eligibility-host-v309/` | 成本约束的分支选择原型，旧成本代理不能替代新路径实测。 |
| `tree-single-prepare-target-math-cost-v311/` | 当前冻结的 408 案例与 11 组、20 轮成对 target 成本实验。 |
| `budget-diagnostics/` | 独立审计、逐类收益校准与最终比较脚本。 |

运行时依赖还保存在 `snapshot/root/data/vllm-ascend-hust/`；
历史冻结的 `native_engine.py` 等依赖位于 `snapshot/root/data/specslo-baselines/`。
这些路径与 manifest 一一对应。仓库顶层插件没有自动切换到这些实验原型。
配套 vLLM 核心的 14 个本地修改文件也已保存，包含调度、投机元数据、验证和对应测试；
它们与冻结运行时中的对应文件逐字节一致，映射记录在 `companion_core_sources`。

## 复现约束

实验使用原生 Python 3.11、CANN 9.0、torch-npu 2.10.0；BF16，无量化。
Qwen3-0.6B TP1 draft 与 Qwen3-32B TP3 target 使用物理卡 2/5/6/7。
核心 vLLM 与本地插件提交锚点记录在 manifest 中。
负载为原 463 请求、泊松到达 4 请求/秒、seed=0、120 秒窗口、每请求 256 输出 token；
三类 TPOT SLO 为 40/50/150 ms。结果必须使用共同发布时钟，并通过全部请求的独立 AR 校验。

此提交保存代码和必要的轻量审计摘要，不是原实验目录的逐字节全量备份。
模型、数据集、历史输出、张量抓取、图缓存、编译产物和源码压缩归档未上传。
原 guard 依赖的历史证据和数据 fixture 由 manifest 中的 `omitted_evidence` 列出，
因此仅克隆仓库不能直接通过原冻结 guard；恢复时需要按原条件重新生成对应 fixture、
证据与新的冻结清单，不能绕过校验或将旧资格标记继承给新代码。
模型和数据集仍使用公共存储，其他实验数据保存在容器私有区域。

## 检查快照

在仓库根目录执行：

```bash
python3 -B research/specslo-tree/verify_snapshot.py
```

此检查只核对保存文件的 SHA256 与 Python 语法，不导入模型、不运行 NPU，
也不表示尚未验证的原型已经通过端到端实验。
