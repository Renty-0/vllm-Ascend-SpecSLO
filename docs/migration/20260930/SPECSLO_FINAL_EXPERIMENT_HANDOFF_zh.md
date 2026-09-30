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
| 冻结的复现源码、补丁、结果和文档 | `git@github.com:Renty-0/vllm-Ascend-SpecSLO.git`，标签 `specslo-reproduction-20260930` | 是，复现必须用它 |
| 后续开发分支 | 同一私库分支 `SpecSLO-Chain` | 继续调优时使用，不替代冻结标签 |
| 性能源码检查点 | 私库提交 `31b3cde3296c00867e8d9e23558598debed276a6`；其后到冻结标签只有文档、补丁和迁移资产变化 | 是，冻结标签已包含 |
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
- `https://github.com/Renty-0/vllm-Ascend-SpecSLO/tree/specslo-reproduction-20260930`
- `https://github.com/Renty-0/vllm-Ascend-SpecSLO/tree/codex-conversations-backup-20260924`
- `https://github.com/vLLM-HUST/vllm-hust/commit/933842ab0fdd53ee47dee8148bca1671d7d6da61`

这是私人仓库。新容器必须由用户重新配置 GitHub SSH 权限；仓库不保存 SSH key、Codex
token、GitHub token 或代理凭据。

截至 2026-09-30，已从一个空临时 Git 仓库实际执行并确认下面三个远端对象可下载：

```text
vLLM-HUST commit:       933842ab0fdd53ee47dee8148bca1671d7d6da61
SpecSLO frozen tag:     specslo-reproduction-20260930
conversation branch:    ca06fca23d7b935139d28d2eff04f0868de168d6
```

以后 `SpecSLO-Chain` 会因文档或代码提交继续前移；完美复现始终使用冻结标签，并以
`31b3cde3296c00867e8d9e23558598debed276a6` 识别归档的运行时代码检查点。只有完成旧检查点
复刻后，才切换活动分支继续优化。

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

下面所有输入都来自 GitHub，不依赖旧容器 `/root/data/vllm-hust` 的本地 Git 对象。私人仓库
用 SSH 下载；公开的 vLLM-HUST 同时给出 SSH 和 HTTPS 地址：

```text
SpecSLO 私库 SSH:  git@github.com:Renty-0/vllm-Ascend-SpecSLO.git
vLLM-HUST SSH:     git@github.com:vLLM-HUST/vllm-hust.git
vLLM-HUST HTTPS:   https://github.com/vLLM-HUST/vllm-hust.git
```

先验证权限和远端对象：

```bash
ssh -T git@github.com || true

git ls-remote git@github.com:Renty-0/vllm-Ascend-SpecSLO.git \
  refs/tags/specslo-reproduction-20260930
git ls-remote git@github.com:Renty-0/vllm-Ascend-SpecSLO.git \
  refs/heads/codex-conversations-backup-20260924
```

然后下载冻结的复现标签，并确认运行时代码检查点在其历史中：

```bash
mkdir -p /root/data

git clone --branch specslo-reproduction-20260930 --single-branch \
  git@github.com:Renty-0/vllm-Ascend-SpecSLO.git \
  /root/data/vllm-ascend-hust

git -C /root/data/vllm-ascend-hust cat-file -e \
  31b3cde3296c00867e8d9e23558598debed276a6^{commit}
git -C /root/data/vllm-ascend-hust merge-base --is-ancestor \
  31b3cde3296c00867e8d9e23558598debed276a6 HEAD
```

### 4.1 推荐：执行仓库自带的源码恢复器

下面这个脚本会从 GitHub 精确拉取 vLLM-HUST 基线、校验并应用补丁、恢复 Catlass 子模块，
最后用反向补丁检查证明 14 个 vLLM-HUST 修改/新增文件完整落地。它不会安装环境、模型或
覆盖非空的未知目录：

```bash
bash /root/data/vllm-ascend-hust/tools/specslo_migration/\
bootstrap_from_github.sh
```

看到 `Source bootstrap passed.` 才表示两个源码树恢复完成。脚本可以再次运行；若完整补丁
已经应用，它只校验而不会重复应用。执行本小节后，跳过 4.2 的等价手工步骤。

### 4.2 等价手工恢复：只在审核脚本时使用

本小节与 4.1 **二选一**，不能在自动恢复成功后再次手工应用补丁。

不要依赖 vLLM-HUST 当前默认分支的位置。直接从 GitHub 按完整提交 SHA 拉取并 detached
checkout，这样即使默认分支以后前移也不会改变实验基线：

```bash
mkdir -p /root/data/vllm-hust
git -C /root/data/vllm-hust init
git -C /root/data/vllm-hust remote add origin \
  git@github.com:vLLM-HUST/vllm-hust.git
git -C /root/data/vllm-hust fetch --depth=1 origin \
  933842ab0fdd53ee47dee8148bca1671d7d6da61
git -C /root/data/vllm-hust checkout --detach FETCH_HEAD

test "$(git -C /root/data/vllm-hust rev-parse HEAD)" = \
  "933842ab0fdd53ee47dee8148bca1671d7d6da61"
```

若公开仓库的 SSH 不可用，只把上面 `remote add origin` 的地址换成：

```bash
https://github.com/vLLM-HUST/vllm-hust.git
```

最后应用由私人仓库下载下来的 SpecSLO 核心补丁：

```bash

git -C /root/data/vllm-hust apply --check \
  /root/data/vllm-ascend-hust/patches/vllm-hust/specslo-core.patch
git -C /root/data/vllm-hust apply --3way \
  /root/data/vllm-ascend-hust/patches/vllm-hust/specslo-core.patch

git -C /root/data/vllm-ascend-hust submodule update --init --recursive
git -C /root/data/vllm-ascend-hust rev-parse HEAD
git -C /root/data/vllm-hust rev-parse HEAD
```

补丁应用成功后的正常状态是：vLLM-HUST 的 `HEAD` 仍然显示 `933842ab...`，同时
`git status --short` 显示 SpecSLO 修改和新增文件。补丁本身的 SHA256 应为：

```text
133d96d1645225a6c9c40db445bf660a1f576787bd59265e5d0cc239f0c6db63
```

可用下面的命令确认补丁文件没有在下载过程中损坏：

```bash
printf '%s  %s\n' \
  133d96d1645225a6c9c40db445bf660a1f576787bd59265e5d0cc239f0c6db63 \
  /root/data/vllm-ascend-hust/patches/vllm-hust/specslo-core.patch | \
  sha256sum -c -
```

无论选择 4.1 还是 4.2，最后都执行以下审计；五项必须全部成功：

```bash
test "$(git -C /root/data/vllm-hust rev-parse HEAD)" = \
  933842ab0fdd53ee47dee8148bca1671d7d6da61
git -C /root/data/vllm-hust apply --reverse --check \
  /root/data/vllm-ascend-hust/patches/vllm-hust/specslo-core.patch
git -C /root/data/vllm-ascend-hust merge-base --is-ancestor \
  31b3cde3296c00867e8d9e23558598debed276a6 HEAD
test "$(git -C /root/data/vllm-ascend-hust/csrc/third_party/catlass \
  rev-parse HEAD)" = 41bf90da655bba3c66d0acd7e00abe33960ecfd6
test -f /root/data/vllm-hust/vllm/v1/spec_decode/spec_rhythm.py
```

不要从私人仓库的 `main` 获取代码；完美复现用冻结标签，继续开发才切换
`SpecSLO-Chain`。运行源码位于私库，
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

## 9. 当前全部优化设计总账

本节用于代码审核和恢复后配置核对。状态含义如下：

- **采用**：当前 `1.453552x` 检查点实际使用，恢复时必须保持；
- **基础能力**：代码已实现并通过相应功能门，但当前 fixed-gamma4 性能点没有消费；
- **实验保留/默认关闭**：代码入口仍在，只能重新做完整门禁后启用；
- **否决/已回滚**：不得因为微基准好看就重新打开，除非设计前提发生实质变化。

### 9.1 论文核心机制与当前性能路径

| 设计 | 状态 | 当前实现和取舍 | 主要代码 |
| --- | --- | --- | --- |
| 双 batch execution pipeline | **采用** | A/B home batch 在 TP1 draft 与 TP3 target 间交替；CANN trace 已证明三个 target rank 的设备窗口分别约 77.51%/77.30%/77.08% 被 draft compute 覆盖，不以 host 同时提交冒充 overlap | `native_engine.py`、`spec_rhythm.py` |
| Rolling Eager continuation | **采用** | 每轮依据 `a_need`、urgency、接受率和 confidence 选择另一 home 的请求提前 draft；full accept 才 promotion，reject 原子丢弃后缀 | `spec_rhythm.py`、`native_engine.py` |
| individual speculative budget / 动态 B、动态 gamma | **基础能力** | 标量/逐请求预算、tree plan 和 scheduler 接口已实现；当前最终检查点为便于隔离工程瓶颈而固定 gamma=4，不能把本结果写成论文动态预算收益 | `spec_rhythm.py`、vLLM-HUST patch |
| tree speculative decode | **基础能力** | tree plan、FULL-mask FIA、KV/回滚和 graph 门已具备；当前 `SpecSLO-Chain` 最优点用线性四步链。gamma=4 下 2x2 浅树缩短连续 horizon 且增加 mask/compaction 成本，故不用于该性能点 | `pearl/tree.py`、`native_graph.py`、vLLM-HUST patch |
| fixed gamma4 serial full-window | **采用** | 一次依赖链产生 4 个连续 proposal，一次 target 验证 4 行，最多提交 4 token；不是四个独立 target decode，也不启用 bonus token | `native_engine.py`、`native_graph.py` |
| 在线 Poisson admission | **采用** | 463 请求按固定 arrival trace 陆续进入，prefill 会进入在线执行；不是预先做好全部 KV 的纯 decode 测试 | benchmark runner、`native_engine.py` |
| coalesce1 公平合同 | **采用** | baseline/candidate 都让到达请求按 coalesce1 接入；历史 coalesce4 虽提高 Goodput，但改变 TTFT/prefill 批处理，不作为最终公平收益来源 | formal runner、`api.py` |

### 9.2 当前检查点实际采用的工程优化

| 优化 | 状态与证据 | 恢复配置/边界 |
| --- | --- | --- |
| sealed ACLGraph、qualification、prune、replay-first | **采用**；消除首尾 shape 重复 eager/capture，正式窗口 shape/capacity fallback=0 | 必须先 warm-up/资格化，warm-up 不计正式 E2E；正式区间不允许捕图或回退 |
| fixed-shape full-chain draft graph | **采用**；固定四步链减少动态 bucket 和 Python plan 抖动 | gamma=4、linear bonus=0 |
| current-target early submit | **采用**；把当前已经确定的 decode target graph host 更新与 next draft 重叠 | `SPECSLO_EARLY_CURRENT_TARGET_SUBMIT=1`；只采用已验证的 current/decode 路径 |
| target FIA 连续两路更新 | **采用**；FIA1→FIA2 的一次预检使 target 局部关键段约 -5.57%，Raw 约 +0.366% | workers=2、event group=1；每层 handle/begin/end 和事件边界保留 |
| draft PagedAttention 两路更新 | **采用**；draft 局部约 -18.14%，系统 Raw 约 +0.099%，说明大部分收益被 overlap 隐藏 | workers=2；只在 pure PA、固定 gamma4 full-chain 合格形状启用，其他形状 fail-closed |
| proposal/continuation 零拷贝与 confidence elision | **采用**；复用 compact envelope 的 host continuation、proposal matrix view，固定 acceptance-only 路径不传 scheduler 不消费的 confidence | 不改变 token/acceptance 语义；由 CPU 回归和完整在线门保护 |
| fused fixed-greedy verdict | **采用**；设备侧合并 accepted-prefix/correction 计算，减少 Python/D2H 小算子链 | `fixed_greedy_verdict.py`；必须保留 changed-input/bit-exact 检查 |
| compact D→T/T→D transport + atomic commit | **采用**；B64 int64 envelope 从 961 压到 257 元素，所有 verdict/KV/epoch 成功后一次提交，失败统一回滚 | `fixed_greedy_transport.py`、`native_engine.py`；不能提前 post 形成跨 communicator wait cycle |
| scheduler/accounting 固定缓冲 | **采用**；plan 内缓存 urgency/`a_need`，proposal 用 cycle-local 固定输出，accounting 用持久 host tensor/NumPy view；属于正确的小开销优化，不宣称独立系统跃迁 | 保持排序语义和 buffer 生命周期 |
| v149 rank-3 adaptive MC2 | **采用但必须重新资格化**；按 M 选择 16/24 AIV，epilogue 用 block broadcast/zero-stride Mul；单次 MC2-on Raw 仅约 +0.0716%，局部算子收益没有等比例转为 E2E | 新机器必须重编 vendor、真实输入 capture、跨层 changed-input/P95 资格化并生成新 profile；禁止照搬旧 profile |
| NZ mode14 稀疏层大-M 双布局 | **采用于组合包**；只给 8/64 层保留大-M gate-up/down NZ 副本，小 M 保持 ND，避免全层双份 OOM | `SPECSLO_TARGET_NZ_MODE=14`、阈值 161；没有独立稳定收益声明 |
| immutable-drain free padding repair | **采用，是当前最好点的直接增量**；只复用 serial-draft graph 已存在的 padding 行，按最长剩余服务修复 opposite-home proposal，不增加 bucket | 命中 64 cycle/93 行；相对无 repair Raw +1.084 tok/s、Goodput +28.652 tok/s；`DRAIN_FIFO=1` 且 repair=1 |
| 完整图容量 | **采用** | target max batch=32、stable request bucket=1、mixed token buckets=64..512、graph entries=80、KV blocks=544；改值必须重做资格化 |
| 图/MC2/overlap 证据门 | **采用** | graph fallback=0、MC2 fallback/exception=0、四 rank 计数一致、设备 overlap 通过才允许计性能 | benchmark report 与 formal runner fail-closed |

当前检查点的关键环境值必须与 `REPRODUCTION_RUNBOOK_zh.md` 第 10.2 节命令一致；不要靠
launcher 历史默认值猜测。核心源码
审核入口为：

- `vllm_ascend/spec_decode/pearl/native_engine.py`：双 batch、full-chain、early submit、repair；
- `vllm_ascend/spec_decode/pearl/spec_rhythm.py`：SLO scheduler、`a_need`、rolling eager、预算；
- `vllm_ascend/spec_decode/pearl/native_graph.py`：ACLGraph 生命周期和 FIA/PA task update；
- `vllm_ascend/spec_decode/pearl/fixed_greedy_transport.py` 与
  `vllm_ascend/ops/triton/spec_decode/fixed_greedy_verdict.py`：通信与 verdict；
- `vllm_ascend/spec_decode/pearl/mc2.py` 与
  `csrc/mc2/matmul_allreduce_add_rmsnorm/`：生产 adapter 和 rank-3 MC2；
- `examples/specslo_benchmark_report.py`、`tools/specslo_experiments/`：证据门和正式 A/B。

### 9.3 已实现但当前必须保持关闭的方向

| 方向 | 为什么不进当前配置 | 以后什么条件下可重开 |
| --- | --- | --- |
| dynamic B/tree budget | 当前 fixed-gamma 检查点未消费，不能混入恢复实验 | 先复刻固定点，再按论文 §5.2 独立标定 B 并跑完整 RPS/batch 矩阵 |
| conservative target replay-ahead | 在线 mutable frontier/pending arrival 使覆盖率过低，preflight 成本大于收益 | scheduler 能冻结 next-plan reservation/frontier epoch 后重新设计 |
| mixed current-target early submit | 596 次真实命中但 Raw 924.462，未形成收益 | 新图更新路径显著变便宜且单变量完整门通过 |
| prefetch-ahead | 预取 16 请求后 Raw/Goodput 均下降，提前占 mixed budget 伤 tight TPOT | 有隔离 prefill 资源或新的 SLO admission 证明 |
| target input embedding replica | 单次 Raw 仅 +0.087%；完整图族额外约 1.56 GiB/rank 并 OOM | HBM 预算改变且多轮 A/B 证明收益 |
| class-preserving residual LRTF | loose 尾部前移但 Raw 仍仅 0.9957x TP4 | 有不损 tight/normal 且能缩短 makespan 的新证明 |
| linear bonus token | 虽减少 target round，但增加 25% target rows、破坏 full-chain PA 与 early-submit overlap | 先设计可携带 bonus frontier 的 continuation 协议 |
| PARD、persistent FIA staging | PARD 降低接受率；staging 虽缩短 draft 11.79%，cycle wall/Raw/Goodput 反而变差 | draft 再次成为设备关键路径且不增加 coordination 时 |

### 9.4 已否决或已回滚的方向

| 方向 | 实测原因/决定 |
| --- | --- |
| coalesce4 作为最终收益来源 | 历史 Goodput 可升，但改变在线 batching/TTFT；最终公平合同固定双方 coalesce1 |
| gamma4 的 2x2/tree2x3 浅树 | 连续预测 horizon 变短，新增 tree mask/KV compaction，短测更慢；不否定更大预算下的树功能 |
| FIA3/FIA4、event-group2/4、prefix1 | 微观 host event/submit 可下降，但线程管理或首层等待转移到 correction，系统 Raw 均不及 FIA2/event1 |
| FIA 更新线程永久绑 NPU context | 单次完整系统未证明 E2E 收益，源码和专用测试已回滚 |
| post-correction target replay | 安全覆盖约 54.3%，但 task update 串行暴露约 16 ms/次，Raw 跌至 683.793；已回滚 |
| side-stream draft prefill、Gloo terminal fallback、单独 async correction | 原工作大多已被 overlap 隐藏，Gloo/D2H 反而形成阻塞或收益不足 |
| 全局/tight-only/late-drain/target-only LRTF | 或无真实调度覆盖，或延后 tight/normal，或不能缩短最后 loose makespan；不进入默认 |
| padding repair + eager 行交换、跨 bucket W、latest-first、叠加 LRTF | 分别破坏 continuation/接受率、付出 graph 阶梯成本或损害 SLO；最终只保留“免费 padding + 最长剩余服务” |
| target NZ5、永久全模型 NZ | Qwen3-32B TP3 小 M gate-up 慢 17%--29%；NZ5 完整系统明显下降 |
| 全 64 层 dual-NZ mode11、偶数层 mode12 | mode11 OOM；mode12 仅约 +0.18% Raw 且无稳定 Goodput 因果，不作为当前模式 |
| target embedding 全图复制 | 资格化尾部 OOM；压 bucket 后收益过小，不抵消 HBM/形状风险 |
| small-M 全覆盖或 large-M MC2 强行融合 | 小 M 微基准收益未转为系统收益；M293/M645 fused 比 split 慢且数值门失败，exact-shape 不合格就走 split |
| token-row padding 到 4 倍数、TP3 attention head/light-rank padding | 设备微基准慢约 5%--10%或不均衡，且部分 BF16 输出改变 |
| core/chunk 参数盲扫、dead-mask、skip mask copy、caller-primary、fallback0 | 都只优化非关键微段或引入等待，完整系统无稳定正收益 |

上述表覆盖当前所有系统级设计族。MC2 v86--v149 的逐微版本屏障、mailbox、AIV 和数值消融
没有在这里重复几百行，完整可追溯记录在 `docs/specslo_work_record_zh.md` 第 6.15--6.24 节；
9 月 28--29 日的每次系统门、结果路径和回滚原因在
`docs/specslo_optimization_resume_20260928_zh.md`。审核时以本节状态为当前结论，以那两份文档
定位原始证据，不能从旧日期文档的“下一步”覆盖新结论。

## 10. 重跑顺序

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

## 11. GitHub 无法替代的内容

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

## 12. 可直接交给新 Codex 的指令

```text
先阅读：
/root/data/vllm-ascend-hust/docs/migration/20260930/
SPECSLO_FINAL_EXPERIMENT_HANDOFF_zh.md

然后严格按 REPRODUCTION_RUNBOOK_zh.md 执行。复现源码必须来自私人仓库标签
specslo-reproduction-20260930；继续开发才使用 SpecSLO-Chain。vLLM-HUST 必须固定到
933842ab 并应用 specslo-core.patch。
先运行 tools/specslo_migration/bootstrap_from_github.sh，并确认五项源码审计全部通过。
恢复 GitHub evidence archive 和固定 workload，验证 workload 与核心模型 SHA256。
不要使用占卡程序，不要沿用旧机器 MC2 profile，不要依赖 runner 的 v134 默认路径。
先完成 CPU、数值、graph 零回退、MC2 和 device-overlap 门，再跑 TP4 baseline 与
TP1+TP3 candidate。以 FINAL_EXPERIMENT_TARGET.json 为旧检查点目标，先复现
Goodput 1.453552x、Raw 0.996721x，再继续完成三轮 Raw>1 和 Goodput>=1.3x 的最终门。
任何不匹配都保留证据并报告，不得通过降低门槛或删除失败样本宣称成功。
```
