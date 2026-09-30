# SpecSLO 最终实验：新容器从零复现总入口

> 这是交给新容器及其 Codex 的第一份文档。目标是从空容器取得源码、恢复历史上下文、重建
> 环境和模型、重跑同一实验，并判断是否复现了旧容器当前最优检查点。不要在读完“结果
> 口径”和“不可自动恢复边界”之前宣称复现成功。

## 1. 先回答：私人 Git 仓库能不能下载后直接运行

**不能单独运行。** 私人仓库保存的是集成了 SpecSLO、PEARL 和 TP3 MC2 的
`vllm-ascend-hust` 插件，不是完整 vLLM。最终运行栈为：

```text
宿主机 Driver/Firmware + CANN 9.0.0
├── vLLM-HUST @ 933842ab + 本私库保存的 specslo-core.patch
├── vLLM-Ascend-SpecSLO 私库 @ SpecSLO-Chain
│   ├── SpecSLO/PEARL Python 与 C++ 源码
│   ├── rank-3 MC2 自定义 CANN 算子
│   ├── benchmark、资格化、正式 A/B 脚本
│   └── workload、结构化结果和编译回滚包
├── Qwen3-0.6B（draft TP1）
└── Qwen3-32B（target TP3；baseline TP4）
```

私库已经保存 vLLM-HUST 的核心修改补丁，因此不需要访问另一个未提交工作树；但仍要克隆
`vLLM-HUST` 的固定基线并应用补丁。`atc26v0`、`eurosys27` 和上游 nano-PEARL 是论文、
workload 与设计来源，不是最终二进制的运行时依赖。精确 workload 已经放入私库恢复包，
所以复现实验不需要重新随机生成数据。

## 2. 所有下载入口及固定版本

| 用途 | 地址/分支/提交 | 是否运行必需 |
| --- | --- | --- |
| 最终插件源码、补丁、结果和文档 | `git@github.com:Renty-0/vllm-Ascend-SpecSLO.git`，分支 `SpecSLO-Chain` | 是 |
| 性能源码检查点 | 私库提交 `31b3cde3296c00867e8d9e23558598debed276a6`；其后到本文提交只有文档、补丁和迁移资产变化 | 是，分支头已包含 |
| vLLM 核心 | `git@github.com:vLLM-HUST/vllm-hust.git`，基线 `933842ab0fdd53ee47dee8148bca1671d7d6da61` | 是 |
| vLLM 核心修改 | 私库 `patches/vllm-hust/specslo-core.patch` | V1/HTTP 必需；统一恢复时必须应用 |
| Catlass 子模块 | 私库 submodule 固定 `41bf90da655bba3c66d0acd7e00abe33960ecfd6` | 编译自定义算子时需要 |
| Codex 历史 | 同一私库孤立分支 `codex-conversations-backup-20260924`，当前头 `ca06fca23d7b935139d28d2eff04f0868de168d6` | 理解历史、继续调优需要 |
| quickstart 管理器 | `git@github.com:intellistream/vllm-hust-dev-hub.git`，基线 `a61b3183ea8394f16059ae4084eeb19c18aa9251`，私库有补丁 | 可选，不是正式 runner 的运行依赖 |
| workload 原始生成器 | `git@github.com:rzwang22/eurosys27.git`，提交 `22c196bdf571e84887e5573607bd8e734faa0625` | 仅溯源；精确 workload 已归档 |
| 早期论文代码 | `https://github.com/rzwang22/atc26v0` | 仅设计审计，不是最终运行依赖 |
| Qwen3 模型 | ModelScope ID `Qwen/Qwen3-0.6B`、`Qwen/Qwen3-32B` | 是，权重未放普通 GitHub |

源代码浏览入口：

- `https://github.com/Renty-0/vllm-Ascend-SpecSLO/tree/SpecSLO-Chain`
- `https://github.com/Renty-0/vllm-Ascend-SpecSLO/tree/codex-conversations-backup-20260924`

这是私人仓库。新容器必须由用户重新配置 GitHub SSH 权限；仓库不保存 SSH key、Codex
token、GitHub token 或代理凭据。

## 3. 下载代码前必须完成的准备

### 3.1 硬件与系统

1. aarch64 容器；
2. 至少 4 张同节点、进程空闲的 Ascend 910B2，每张约 64 GiB；
3. Driver `26.0.rc1`、`ascendhal 7.35.23`；
4. CANN toolkit 9.0.0，内部版本 `V100R001C10SPC001B250`；
5. HCCL、`/dev/davinci*`、足够的共享内存和 CPU/NUMA 权限；
6. 功能复现至少预留约 90 GB，恢复全部历史大资产建议 400 GB；
7. 不使用占卡脚本。运行前检查目标 NPU，若有别人的任务则等待或联系管理员。

系统版本不匹配时，只能恢复源码，不能直接使用 GitHub 中的旧 `.so` 或 CANN vendor。

### 3.2 网络与账号

- 配置可读取私人仓库的 SSH key，并用 `ssh -T git@github.com` 验证；
- 能访问 vLLM-HUST 和模型镜像；外网不稳定时优先 ModelScope；
- 安装 `git`、`zstd`、`rsync`、`sha256sum`、aarch64 Miniconda、CMake、Ninja，且
  GCC/G++ > 8、支持 C++17；
- 若要求模型 bit-exact，先从异机备份复制旧权重。单靠当前上游模型 ID不能保证未来下载的
  revision 永远不变。

### 3.3 给新容器 Codex 的停止条件

以下任一条件不满足时，Codex 应停止性能实验并报告，而不是降低门槛：

- Driver/CANN/torch/torch-npu ABI 不匹配；
- 四张目标 NPU 不空闲；
- workload SHA256 不匹配；
- 模型不完整或哈希不匹配且用户要求 bit-exact；
- vLLM-HUST 补丁冲突；
- eager/graph 数值门、图零回退门、MC2 changed-input 门未通过。

## 4. 从零下载源码

```bash
mkdir -p /root/data

git clone --branch SpecSLO-Chain --single-branch \
  git@github.com:Renty-0/vllm-Ascend-SpecSLO.git \
  /root/data/vllm-ascend-hust

git clone git@github.com:vLLM-HUST/vllm-hust.git \
  /root/data/vllm-hust
git -C /root/data/vllm-hust checkout \
  933842ab0fdd53ee47dee8148bca1671d7d6da61

git -C /root/data/vllm-hust apply --check \
  /root/data/vllm-ascend-hust/patches/vllm-hust/specslo-core.patch
git -C /root/data/vllm-hust apply --3way \
  /root/data/vllm-ascend-hust/patches/vllm-hust/specslo-core.patch

git -C /root/data/vllm-ascend-hust submodule update --init --recursive
git -C /root/data/vllm-ascend-hust rev-parse HEAD
git -C /root/data/vllm-hust rev-parse HEAD
```

不要从私人仓库的 `main` 获取代码；当前项目分支是 `SpecSLO-Chain`。运行源码位于私库，
旧实验执行时为 `c3bd269f` 加工作树 patch，随后已整理为提交 `31b3cde3`。此后分支只增加
恢复文档和归档，没有再改变 `vllm_ascend/`、`csrc/`、正式 runner 的运行语义。

## 5. 恢复 GitHub 中的 workload、结果和 runner

```bash
cd /root/data/vllm-ascend-hust/migration_artifacts/20260930
sha256sum -c SHA256SUMS
bash restore.sh /tmp/specslo-github-restore

cd /tmp/specslo-github-restore
zstd -dc /root/data/vllm-ascend-hust/migration_artifacts/20260930/\
SOURCE_SHA256SUMS.zst | sha256sum -c -
zstd -dc /root/data/vllm-ascend-hust/migration_artifacts/20260930/\
COMPILED_RUNTIME_SOURCE_SHA256SUMS.zst | sha256sum -c -

mkdir -p /root/data /data
rsync -a /tmp/specslo-github-restore/root/data/specslo-workloads/ \
  /root/data/specslo-workloads/
rsync -a /tmp/specslo-github-restore/root/data/nano-pearl-benchmark-results/ \
  /root/data/nano-pearl-benchmark-results/
rsync -a /tmp/specslo-github-restore/data/datasets/ /data/datasets/
```

恢复包包含正式 baseline wrapper；因此必须先执行本节，再运行
`tools/specslo_experiments/run_online_coalesce1_formal_ab.sh`。编译运行时包只是 ABI 一致时的
回滚材料，优先按源码重编译。

精确 workload 为：

```text
/root/data/specslo-workloads/eurosys27-rps4-seed0-author-slo40-50-150/
rps4.0_mix0.6_0.2_0.2_seed0.jsonl
```

其 SHA256 必须为：

```text
dc58dc5af1c5b28c6944f8f3faadeaae2dae34298c07b3ac84745d367c67d662
```

合同为 463 个在线 Poisson 请求、RPS=4、arrival seed=0、类别 seed=1、输出上限 256；
tight/normal/loose 数量为 271/96/96，对应 TPOT SLO 40/50/150 ms。不要重新生成“同分布”
workload 替代该文件。

## 6. Codex 聊天记录在哪里、怎么使用

聊天记录放在同一私人仓库的孤立分支，不会随源码分支一起 checkout：

```bash
git clone --single-branch \
  --branch codex-conversations-backup-20260924 \
  git@github.com:Renty-0/vllm-Ascend-SpecSLO.git \
  /root/data/codex-conversations-backup

cd /root/data/codex-conversations-backup
sha256sum -c SHA256SUMS
sha256sum -c SHA256SUMS_INCREMENTAL_20260930
sha256sum -c SHA256SUMS_TAIL_20260930
bash restore.sh /tmp/codex-conversations-restore
```

该分支包含 296 个 JSONL、三个经过 SQLite `integrity_check` 的数据库、9 月 24 日全量层、
9 月 30 日增量层和 03:12 UTC 活跃会话尾包。最终实验发生在该时间点之前，因而相关设计、
失败方案和性能推理已覆盖。它不含 `auth.json`、`.env`、`config.toml`、SSH key 或 token。

不要直接覆盖新容器的 `/root/.codex`。聊天备份不会自动进入新 Codex 的上下文；用户应先让
新 Codex 阅读本文和下面列出的技术文档，需要历史细节时再在
`/tmp/codex-conversations-restore/root/.codex` 中用 `rg` 搜索 JSONL：

```bash
rg -n "drain padding repair|v149|930.005|560.578|MC2|dual batch" \
  /tmp/codex-conversations-restore/root/.codex \
  --glob '*.jsonl'
```

建议阅读顺序：

1. 本文；
2. `docs/migration/20260930/REPRODUCTION_RUNBOOK_zh.md`；
3. `docs/specslo_optimization_resume_20260928_zh.md`；
4. `docs/specslo_work_record_zh.md`；
5. `docs/atc26v0_feature_audit_zh.md`；
6. 只有遇到争议或需要恢复失败实验上下文时，再查聊天 JSONL。

## 7. 环境和模型恢复

环境的逐条命令、CPU 测试、MC2 编译和资格化命令见
`REPRODUCTION_RUNBOOK_zh.md`。必须匹配的关键版本是：

- Python 3.11.15；
- torch 2.10.0；
- torch-npu 2.10.0；
- triton 3.5.0；
- triton-ascend 3.2.1；
- vLLM editable `/root/data/vllm-hust`；
- vLLM-Ascend-HUST editable `/root/data/vllm-ascend-hust`。

Conda explicit、environment.yml、pip freeze、系统包、CANN 和驱动快照全部在
`docs/migration/20260930/`。pip freeze 含旧 CANN 临时 `file://` 路径，不能原样盲装；这些
wheel 应由新 CANN 镜像提供。

模型可从 ModelScope 镜像下载：

```bash
mkdir -p /data/shared-models /data/.cache/modelscope
export MODELSCOPE_CACHE=/data/.cache/modelscope

/root/miniconda3/envs/vllm-hust-dev/bin/python - <<'PY'
from modelscope import snapshot_download
snapshot_download("Qwen/Qwen3-0.6B", local_dir="/data/shared-models/Qwen3-0.6B")
snapshot_download("Qwen/Qwen3-32B", local_dir="/data/shared-models/Qwen3-32B")
PY
```

下载后验证：

```bash
sha256sum -c /root/data/vllm-ascend-hust/docs/migration/20260930/\
CORE_MODEL_SHA256SUMS
```

全部通过表示与旧容器核心模型 bit-exact。若下载 revision 已变化，功能测试仍可继续，但不能
宣称“完美复现”；应从异机备份获取旧权重或明确报告哈希差异。

## 8. 当前核心实验结果：复现目标

机器可读版本为 `FINAL_EXPERIMENT_TARGET.json`。以下为同一四卡组 `3,4,5,7`、同一 workload
的一轮检查点：

| 指标 | TP4 target-only baseline | SpecSLO TP1+TP3 | 比值/变化 |
| --- | ---: | ---: | ---: |
| E2E 时间 | 127.448802 s | 127.868051 s | candidate 稍长 |
| Raw | 930.004815 tok/s | 926.955552 tok/s | **0.996721×** |
| paper Goodput | 385.660744 tok/s | 560.577871 tok/s | **1.453552×** |
| paper SLO 达成数 | 192/463 | 280/463 | +88 请求 |
| paper SLO 达成率 | 41.468683% | 60.475162% | **+19.006479 pp** |
| 输出 token | 118,528 | 118,528 | 相同 |

Candidate 具体配置：

- Qwen3-0.6B draft TP1 + Qwen3-32B target TP3；baseline 为 Qwen3-32B TP4；
- batch=64、固定 gamma=4、serial-chain draft、在线 coalesce1；
- v149 adaptive MC2、NZ14、early-current-target submit、draft PA workers=2；
- immutable-drain padding repair；
- graph shape/capacity fallback=0；MC2 fallback/exception=0；
- acceptance rate 52.6018%，平均每轮接受 4.61057 token；
- 独立 CANN profiling 中三个 target rank 的 target window 被 draft device compute 覆盖约
  77.51%/77.30%/77.08%。该 overlap 数据来自独立 profile，不冒充上表无 profiler 运行。

候选完整结构化结果恢复路径：

```text
/root/data/nano-pearl-benchmark-results/
20260929-specslo-v149-drain-padding-repair-precheck/runs/
candidate-on-v149-drain-padding-repair-target457-r1/result.json
```

候选 `result.json` SHA256：
`943cba8031561a23aef795541d1f07aebb9769886817871596317a6b4289095b`。

### 8.1 “完美复现”验收定义

先满足不可放宽的合同：

1. workload SHA 完全相同；
2. 模型哈希完全相同；若不相同只能标为功能复现；
3. 463 请求、118,528 输出 token、同四张物理卡；
4. baseline TP4，candidate draft TP1 + target TP3；
5. graph shape/capacity fallback 和 MC2 fallback/exception 全为 0；
6. 数值、changed-input、bit-exact、双 batch device overlap 门通过。

由于性能存在系统噪声，以下作为“当前检查点复刻成功”的数值窗口：

- baseline 和 candidate 的 Raw/Goodput 分别落在上表目标的 ±3%；
- candidate/baseline paper Goodput 至少 1.40×，中心目标为 1.453552×；
- candidate Raw/baseline Raw 接近 1，中心目标为 0.996721×；
- SLO 达成率接近 60.48%，且相比 baseline 提升约 19.01 个百分点；
- 独立 profile 的三个 target rank overlap 均不低于 75%。

### 8.2 不得把当前检查点误写成什么

当前最好结果**不是**三轮正式双门已经完成。它已经超过 Goodput 1.3×，但 Raw 仍比 TP4
低约 0.328%。项目后续最终门仍是三轮交错 A/B 的每一轮都同时满足：

```text
candidate Raw > baseline Raw
candidate paper Goodput >= 1.3 * baseline paper Goodput
```

新容器应先复现本节的 1.4535× 检查点，再继续补齐 Raw 和三轮稳定性；不能把“恢复环境”与
“完成最终性能优化”混为一件事。

## 9. 重跑顺序

1. 按 `REPRODUCTION_RUNBOOK_zh.md` 重建 Conda 和两个 editable 仓库；
2. 通过 CPU 控制面测试；
3. 重新从当前源码编译 v149 MC2 vendor；
4. 在新机器重新捕获真实 Qwen3-32B TP3 输入并生成 MC2 profile，不能跨机器照搬旧 profile；
5. 通过 eager/graph × MC2-off/on 数值四象限；
6. 通过 tree/chain graph 零回退和双 batch device overlap；
7. 用显式 v149 profile/vendor 参数运行
   `tools/specslo_experiments/run_online_coalesce1_formal_ab.sh --dry-run`；
8. 审核 dry-run 后执行正式 A/B；
9. 用 `FINAL_EXPERIMENT_TARGET.json` 对照新结果，保存源码、环境、卡映射、模型/workload
   哈希和完整 `result.json`。

正式启动脚本默认值仍保留历史 v134 路径，因此必须按详细手册显式设置 v149 profile 和
vendor；依赖默认值运行不算复现。

## 10. GitHub 无法替代的内容

普通 GitHub 已保存代码、补丁、聊天、结构化结果、精确 workload、环境清单和编译回滚包，
但不包含：

- 约 182.57 GB 模型权重；
- 约 114.64 GB raw profiler、trace 和 tensor capture 原件；
- 11.77 GB Conda 目录、4.83 GB Triton LLVM 和 2.81 GB 历史 Ascend 日志；
- 宿主驱动/固件、完整 CANN 和 588 GB 共享数据集挂载；
- SSH、Codex、GitHub 等认证凭据。

这些内容中，模型、环境、vendor、profile 可以按文档重新下载或生成；历史 trace/capture
原件只能从外部备份恢复。重新生成得到的是新证据，不是旧文件。完整边界见
`README_zh.md`。

## 11. 可直接交给新 Codex 的指令

```text
先阅读：
/root/data/vllm-ascend-hust/docs/migration/20260930/
SPECSLO_FINAL_EXPERIMENT_HANDOFF_zh.md

然后严格按 REPRODUCTION_RUNBOOK_zh.md 执行。源码必须来自私人仓库
SpecSLO-Chain，vLLM-HUST 必须固定到 933842ab 并应用 specslo-core.patch。
恢复 GitHub evidence archive 和固定 workload，验证 workload 与核心模型 SHA256。
不要使用占卡程序，不要沿用旧机器 MC2 profile，不要依赖 runner 的 v134 默认路径。
先完成 CPU、数值、graph 零回退、MC2 和 device-overlap 门，再跑 TP4 baseline 与
TP1+TP3 candidate。以 FINAL_EXPERIMENT_TARGET.json 为旧检查点目标，先复现
Goodput 1.453552x、Raw 0.996721x，再继续完成三轮 Raw>1 和 Goodput>=1.3x 的最终门。
任何不匹配都保留证据并报告，不得通过降低门槛或删除失败样本宣称成功。
```

