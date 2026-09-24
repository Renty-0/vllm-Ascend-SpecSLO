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
```

`capture_manifest.sh` 会为三个 dirty repo 保存 Git bundle、binary patch 和状态，但不复制
未跟踪大文件或模型权重；大文件必须通过可断点续传工具
（推荐 `rsync --partial --append-verify` 或对象存储 multipart upload）单独传出服务器。
