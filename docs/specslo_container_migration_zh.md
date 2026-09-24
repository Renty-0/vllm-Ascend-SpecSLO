# SpecSLO 容器迁移与灾备记录

## 结论

`/root`、`/data`、`/workspace` 都来自当前服务器的本地 LVM/XFS 卷，不是灾备存储。把压缩包
留在这些目录中的任何一个位置，都无法应对服务器清空。完整迁移必须至少有一份副本离开
当前宿主机，例如新服务器 SSH 目录、对象存储或用户本地磁盘。

## 数据分层

### A. 不可替代，必须异机保存

- `/root/data/vllm-ascend-hust`：SpecSLO/PEARL 源码、MC2 内核、测试和技术文档；
- `/root/data/vllm-hust` 与 `/root/data/vllm-hust-dev-hub`：仍含未提交改动，manifest 会额外
  保存 bundle、binary patch、状态和未跟踪文件清单；
- `/root/data/nano-pearl-benchmark-results`：原始吞吐、Goodput、profiling 和图回退证据；
- `/root/data/specslo-workloads`：论文 workload 与到达轨迹；
- `/root/data/specslo-mc2-*`、`/root/data/tp3-mc2-debug-*`：自定义算子 vendor、输入捕获和资格档案；
- `/root/data/b_budget_910b2`、`/root/data/b_budget_910b3`：历史实验结果；
- `/data/datasets`：本项目下载和生成的数据集；
- 本文件旁的 manifest、Git bundle、Conda export 和 SHA256 文件。

### B. 建议保存，或者接受重新下载/构建

- `/data/shared-models`：约 171 GB；重新下载耗时且外网不稳定，优先异机复制；
- `/root/miniconda3`：约 11 GB；同架构、相近系统的新容器可直接复制，但仍应保留
  `environment.yml`、Conda explicit spec 和 pip freeze 作为可审计重建路径；
- 自定义 CANN `.run`/vendor 和已编译 extension：直接保存可缩短恢复时间，恢复后仍必须校验
  CANN、驱动、Torch/NPU ABI。

### C. 不应进入 GitHub，可按需归档

- `csrc/build.*`：可重建的中间产物；
- `extra-info/data-dump`：约 677 MB 的异常 dump；
- Hugging Face/pip/tokenizer cache；
- `/data/shared_datasets`：约 588 GB 的只读共享挂载，应先向管理员确认新服务器是否继续提供。

## 当前规模（2026-09-24）

| 路径 | 规模 |
|---|---:|
| `/root/data` | 约 30 GB |
| `/root/data/nano-pearl-benchmark-results` | 约 7.8 GB |
| `/data/shared-models` | 约 171 GB |
| `/data/datasets` | 约 35 MB |
| `/root/miniconda3` | 约 11 GB |
| `/root/.codex` | 约 7.1 GB |
| `/data/shared_datasets`（只读共享） | 约 588 GB |

## 恢复顺序

1. 新容器安装与旧容器兼容的 Ascend driver/CANN、Python 和系统依赖；
2. 从私人仓库 checkout `SpecSLO-Chain`，或用 Git bundle 恢复；
3. 恢复 `/data/shared-models`、`/data/datasets`、benchmark results 和 custom vendor；
4. 用 Conda export 重建 `vllm-hust-dev`，或复制环境后执行 relocation/ABI 检查；
5. 重建 `vllm_ascend_C` 与自定义 CANN operator；
6. 先跑 CPU 单测，再跑算子 changed-input graph qualification；
7. 最后进行完整 model-runner 数值与性能回归，不能直接沿用物理 NPU 映射不同的 MC2 profile。

## 工具

```bash
bash tools/specslo_migration/capture_manifest.sh /path/on/external/storage/manifest
bash tools/specslo_migration/verify_manifest.sh /path/on/external/storage/manifest
bash tools/specslo_migration/restore_environment.sh /path/to/manifest
SPECSLO_MANIFEST_DIR=/path/to/manifest \
  bash tools/specslo_migration/create_portable_archives.sh /path/on/external/storage/chunks 4G
bash tools/specslo_migration/verify_portable_archives.sh /path/on/external/storage/chunks
bash tools/specslo_migration/restore_portable_archive.sh /path/to/chunks root-data /

# 单独生成全部已完成模型、历史 trace/log 和离线 Triton LLVM；不重复核心包：
SPECSLO_INCLUDE_CORE=0 SPECSLO_INCLUDE_CODEX=0 \
SPECSLO_INCLUDE_PRIVATE_CONFIG=0 SPECSLO_INCLUDE_COMPLETE_MODELS=1 \
SPECSLO_INCLUDE_FORENSIC_LOGS=1 SPECSLO_INCLUDE_OFFLINE_TOOLCHAIN=1 \
  bash tools/specslo_migration/create_portable_archives.sh /path/on/external/storage/large-chunks 4G
```

`capture_manifest.sh` 会为三个 dirty repo 保存 Git bundle、binary patch 和状态，但不复制
未跟踪大文件或模型权重；大文件必须通过可断点续传工具
（推荐 `rsync --partial --append-verify` 或对象存储 multipart upload）单独传出服务器。
portable archive 默认分开保存 `/root/data`、Miniconda、项目数据集、Codex 状态和私密配置，
并用临时目录生成后再原子发布分卷。输出目录不能放进任何待归档目录内。
`codex-state` 包含会话历史和 `auth.json`，`private-config` 包含 SSH key；两者都只能通过
加密私有通道传输并加密落盘，恢复时应确认不会覆盖新容器已有凭据。设置
`SPECSLO_INCLUDE_KEY_MODELS=1` 可额外打包 Qwen3-0.6B/Qwen3-32B 和
Qwen2.5-0.5B/Qwen2.5-14B 四个关键模型。设置 `SPECSLO_MANIFEST_DIR` 可把已经验证的
manifest 一并打包。每个分卷以及 `ARCHIVE_INDEX.tsv` 都记录在 `SHA256SUMS` 中；校验脚本
同时检查分卷连续性、zstd 完整性和 tar 可遍历性。

`SPECSLO_INCLUDE_COMPLETE_MODELS=1` 会保存 `/data/shared-models` 中所有已完成模型，但跳过
只有 `.incomplete` 权重的 Llama-3.1-70B/Llama-3.2-1B 下载残片；不要与
`SPECSLO_INCLUDE_KEY_MODELS=1` 同时启用。`SPECSLO_INCLUDE_FORENSIC_LOGS=1` 单独保存
不可再生的 `/root/ascend` 历史 trace/log；`SPECSLO_INCLUDE_OFFLINE_TOOLCHAIN=1` 保存可选的
Triton LLVM 离线工具链。通过 `SPECSLO_INCLUDE_CORE=0`、`SPECSLO_INCLUDE_CODEX=0` 和
`SPECSLO_INCLUDE_PRIVATE_CONFIG=0` 可生成不重复核心包的大文件归档。

即使归档输出位于 `/tmp`，它仍与 `/root`、`/data` 同属当前服务器，只能作为临时 staging。
服务器清空前必须把 `chunks/` 整体复制到异机 SSH 目录、对象存储或用户本地磁盘，并在
异机再次运行 `verify_portable_archives.sh`。不能只保留当前机器上的压缩包。

该方案保存项目源码、实验数据、用户态环境和必要配置，不等同于完整容器镜像：它不会保存
宿主机 driver、完整 CANN 安装、全部 `/etc`、可重建 cache、只读 `/data/shared_datasets`
或未明确选择的全部共享模型。新容器仍须按 manifest 记录匹配系统 ABI 并执行回归验证。
