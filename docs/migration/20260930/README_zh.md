# SpecSLO 新容器完整恢复审计（2026-09-30）

从空容器开始的逐步恢复、重建、图/MC2 资格化和正式 A/B 验收命令见
[`REPRODUCTION_RUNBOOK_zh.md`](REPRODUCTION_RUNBOOK_zh.md)。本文保留资产边界和灾备审计，
不再用高层“恢复顺序”代替可执行手册。

需要下载到本地、并作为新容器/Codex 唯一总入口的交接文档见
[`SPECSLO_FINAL_EXPERIMENT_HANDOFF_zh.md`](SPECSLO_FINAL_EXPERIMENT_HANDOFF_zh.md)；核心结果
的机器可读目标和模型逐文件哈希分别见 `FINAL_EXPERIMENT_TARGET.json` 与
`CORE_MODEL_SHA256SUMS`。

从 GitHub 恢复两个源码树时优先运行
`tools/specslo_migration/bootstrap_from_github.sh`；该脚本已在空临时目录实测完成
`vLLM-HUST@933842ab + specslo-core.patch` 的 14 文件恢复和反向补丁校验。

## 1. 本次已经保存到私人 GitHub 的内容

私人仓库为 `git@github.com:Renty-0/vllm-Ascend-SpecSLO.git`。
完美复现使用冻结标签 `specslo-reproduction-20260930`；`SpecSLO-Chain` 是会继续前移的开发
分支，不应作为未来 bit-identical 源码身份。

| 内容 | 分支/目录 | 状态 |
| --- | --- | --- |
| SpecSLO/PEARL、TP3 MC2、测试、启动器和技术文档 | `SpecSLO-Chain` | 已推送；本审计提交完成后以远端分支头为准 |
| Codex 对话全量快照 | `codex-conversations-backup-20260924` | 2026-09-24 全量包 + 2026-09-30 增量包 + 03:12 UTC 活跃会话尾包；远端分支头 `ca06fca2` |
| vLLM-HUST 未提交的核心层修改 | `patches/vllm-hust/specslo-core.patch` | 包含 10 个修改文件和 4 个原未跟踪文件，基线 `933842ab` |
| dev-hub quickstart 修改 | `patches/vllm-hust-dev-hub-quickstart.patch` | 基线 `a61b3183` |
| 可移植实验结果和数据集 | `migration_artifacts/20260930/` | 21,937 个文件；压缩前 1,704,355,840 B，含所有 `result.json`、小型 JSON、报告、CSV/TSV、脚本、论文 workload 和 `/data/datasets` |
| 当前编译运行时回滚包 | `migration_artifacts/20260930/` | 703 个文件；压缩前 232,284,160 B，含三个自定义 CANN vendor 和 `vllm_ascend` 扩展库 |
| 环境和存储清单 | 本目录 | Conda/Pip 锁定信息、软件版本、模型/数据集/仓库/挂载清单及 672,013 个文件的路径/大小/mtime 清单 |

GitHub 上的对话包不含 `auth.json`、`.env`、`config.toml`、SSH 私钥、日志、缓存、
插件和 SQLite WAL/SHM。2026-09-30 增量中出现过的代理值已脱敏，最终认证值和 PEM 私钥
扫描均为 0。对话内容本身仍包含终端输出和环境细节，因此该仓库必须保持 private。

## 2. 为什么不能把字面意义上的“所有文件”放进普通 Git

当前仅 `/root/data` 就有 **152,559,620,096 B**，其中 benchmark 目录有
**114,643,017,728 B**；`/data/shared-models` 另有 **182,569,271,296 B**。
benchmark 中 `.pt` 张量捕获约 103.35 GB，部分单文件为 105,498,807 B，已经超过
GitHub 普通 Git 的 100 MiB 单文件上限。模型和张量权重本身也几乎不可压缩。

因此本次上传的是“可在 GitHub 中可靠保存的全部关键层”：源码、核心补丁、全部结果
`result.json`、报告、workload、项目数据集、环境锁定、文件清单和当前编译产物。下面列出的
大资产尚未进入 GitHub，若要求逐文件恢复当前容器，必须复制到异机磁盘或对象存储。

## 3. 仍须离开本服务器备份的内容

### P0：不可从 GitHub 当前内容无损重建

1. `/root/data/nano-pearl-benchmark-results` 中未进入便携证据包的 raw CANN profile、
   大型 trace JSON/JSONL、`.pt` 真实层输入捕获、`.db`、`.slice_*`、`.op_range` 和
   `.op_mark`。总目录 114.64 GB，其中大部分是后续 MC2 数值/性能复验所需的真实输入。
2. `/root/data/specslo-mc2-small-m-real-prefix-20260929`（9.81 GB）以及其余
   `specslo-mc2-*`、`tp3-mc2-debug-*` 目录。源码可以重编译，但真实输入、资格结果和
   历史失败证据不能凭源码恢复。
3. `/root/data/b_budget_910b2`（2.17 GB）、`b_budget_910b3`（2.99 GB）和历史
   profiling 原件；GitHub 便携包只覆盖结构化结果和报告，不覆盖全部原始 profiler 数据。
4. 所有没有远端提交、又未被上述两个补丁覆盖的个人文件。当前仓库审计见
   `repository_inventory.tsv`；除 vLLM-HUST 和 dev-hub 外，其余已发现仓库在检查时均干净。

### P1：可以重下载/重建，但建议异机复制

1. `/data/shared-models`（182.57 GB）。当前 SpecSLO 必需的是 `Qwen3-0.6B` 和
   `Qwen3-32B`；历史回归还使用 Qwen2.5、EAGLE/EAGLE3、PARD 等。外网不稳定时应优先
   直接复制。`Llama-3.1-70B-Instruct` 有 30 个 `.incomplete` 分片，
   `Llama-3.2-1B-Instruct` 有 2 个 `.incomplete` 文件，二者不是完整模型，不能当作
   成功备份。完整列表见 `model_inventory.tsv`。
2. `/root/miniconda3`（11.77 GB）。同架构镜像可复制，但更稳妥的方式是在新容器按
   `vllm-hust-dev.conda-explicit.txt`、`vllm-hust-dev.environment.yml` 和
   `vllm-hust-dev.pip-freeze.txt` 重建，再以 editable 方式安装两个源码仓库。
3. `/root/.triton/llvm`（4.83 GB），无外网时需要；有稳定镜像时可重新下载。
4. `/root/ascend`（2.81 GB）主要是历史 trace/log；不影响基础运行，但影响故障追溯。
5. `/root/data` 中四个派生 Qwen2.5 draft 模型和 vocab 对齐模型，合计约 3.56 GB；这些
   是历史异构 vocab/层裁剪实验的产物，不能只靠原始 0.5B 模型自动还原。

### P2：必须匹配版本，不能简单复制后假定可用

1. CANN toolkit 9.0.0，内部版本 `V100R001C10SPC001B250`；当前安装约 9.0 GB。
2. Driver `26.0.rc1`，`ascendhal 7.35.23`；固件/宿主机驱动由管理员提供，复制容器内
   `/usr/local/Ascend/driver` 不能替代宿主安装。
3. 8 张 Ascend 910B2、aarch64、Python 3.11.15、torch/torch-npu 2.10.0、
   vLLM 基线 `933842ab`。具体记录见 `runtime-packages.txt`、CANN/driver 文件和
   `npu-smi-info.txt`。
4. `/data/shared_datasets` 是只读共享挂载，当前约 588 GB，不属于本容器私有备份；必须
   让管理员确认新服务器继续挂载，不能把它算作已经迁移。

### P3：只应通过加密私有通道保存，禁止提交 GitHub

- `/root/.ssh` 私钥、Git 凭据；
- `/root/.codex/auth.json`、`.env`、`config.toml` 和插件连接凭据；
- 必须保留的 shell/private 配置。

新容器最好重新生成 SSH key、重新登录 Codex/GitHub，并只人工迁移确有必要的非认证配置。

## 4. 建议的异机完整备份命令

`/external/specslo-backup` 必须是异机挂载、对象存储网关或不会随当前服务器清空的路径；
不能放在 `/root`、`/data`、`/tmp` 或当前宿主机其他目录。

```bash
cd /root/data/vllm-ascend-hust

bash tools/specslo_migration/capture_manifest.sh \
  /external/specslo-backup/manifest

SPECSLO_MANIFEST_DIR=/external/specslo-backup/manifest \
SPECSLO_INCLUDE_CORE=1 \
SPECSLO_INCLUDE_CODEX=0 \
SPECSLO_INCLUDE_PRIVATE_CONFIG=0 \
SPECSLO_INCLUDE_COMPLETE_MODELS=1 \
SPECSLO_INCLUDE_FORENSIC_LOGS=1 \
SPECSLO_INCLUDE_OFFLINE_TOOLCHAIN=1 \
  bash tools/specslo_migration/create_portable_archives.sh \
    /external/specslo-backup/chunks 4G

bash tools/specslo_migration/verify_portable_archives.sh \
  /external/specslo-backup/chunks
```

这会保存 `/root/data`、Conda、项目数据集、所有已完成模型、历史 Ascend 日志、离线 LLVM
和 manifest，并自动跳过两个不完整 Llama 下载。原始体量约 354 GB；模型/张量通常压缩率
不高，外部目标建议至少预留 400 GB。认证配置若确实需要保留，应单独加密归档，不能混入
上述普通分卷。

## 5. 新容器恢复顺序

1. 要求管理员提供匹配的 910B2 驱动/固件和 CANN 9.0.0 镜像，先核对 ABI。
2. 克隆私人仓库的 `SpecSLO-Chain`，校验远端提交；不要从 public `origin` 猜测项目状态。
3. 克隆 vLLM-HUST 到基线 `933842ab`，应用
   `patches/vllm-hust/specslo-core.patch`；按需对 dev-hub 基线 `a61b3183` 应用 quickstart
   补丁。
4. 重建 `vllm-hust-dev`，安装 editable vLLM 和 vLLM-Ascend-HUST。
5. 运行 `migration_artifacts/20260930/restore.sh` 到临时根目录，校验归档内 SHA256；
   编译产物仅用于对照，优先从源码重编译 extension 和 CANN vendor。
6. 从异机恢复模型、raw profile、MC2 captures、历史结果和可选 LLVM；用
   `filesystem-file-inventory.tsv.zst` 检查路径、大小和 mtime。
7. 从对话分支执行其 `restore.sh` 到临时目录，检查后再合并到新的 `.codex`；不要覆盖新
   容器已经生成的认证文件。
8. 先跑 CPU 单测和 `git diff --check`，再做 MC2 changed-input/bit-exact/graph 资格测试，
   最后跑 TP1+TP3 对 TP4 的完整在线回归。旧机器的物理卡映射和 MC2 profile 不能未经
   资格验证直接晋升生产。

## 6. 当前恢复等级结论

- **代码、对话、结构化实验证据和小型项目数据集：已进入私人 GitHub。**
- **功能级复现：** 新容器具备相同 CANN/驱动，且能重下载 Qwen3-0.6B/Qwen3-32B 时，
  GitHub 内容足以重建并重新测试 SpecSLO。
- **逐文件复原当前容器：尚未完成。** 仍缺约 300+ GB 的模型、tensor capture、raw profile、
  Conda/LLVM/log 等异机副本，以及由管理员保证的宿主驱动/固件和共享数据集挂载。
- 当前服务器上的任何 `/root`、`/data`、`/tmp` 归档都不算灾备；只有复制离开该宿主机并在
  目标端再次验证 SHA256，才算完成完整备份。

## 7. 文档化重建与外部大包的边界

模型、Conda、Triton LLVM、CANN profiler 和 MC2 张量捕获可以在手册中记录来源、重建步骤
和验收门，使新容器重新生成**功能等价的新资产**。但文档本身不会包含这些大文件，也不能
从无到有恢复某个历史时点的 profiler、失败 trace、真实层输入或未记录 revision 的模型
字节。需要审计历史原件或 bit-exact 恢复时，仍必须在旧服务器清空前完成第 4 节的异机
归档。新容器自动重建的完整步骤和明确的停止条件见
[`REPRODUCTION_RUNBOOK_zh.md`](REPRODUCTION_RUNBOOK_zh.md)。
