# SpecSLO 空容器恢复演练与最终备份缺口

## 1. 结论先行

截至 2026-09-30，**代码与小型实验资产已经可以仅凭 GitHub 从空目录恢复**，但若旧容器
现在被清除，仍不能仅靠 GitHub 保证 bit-exact、ABI-exact 的“完美复现”。清机前至少还要
把以下三项保存到异机磁盘或对象存储：

1. `/data/shared-models/Qwen3-0.6B` 和 `/data/shared-models/Qwen3-32B` 的精确副本，共
   `67,059,520,461` 字节；
2. 当前容器的**不可变 OCI image digest 和镜像导出包**，或至少完整 Conda 环境与
   CANN 9.0.0 安装介质/目录；
3. 管理员侧的宿主驱动、固件、910B2 拓扑和容器启动参数记录。驱动本身应由新宿主正确
   安装，不应把旧宿主动态库直接覆盖到新宿主。

如果只追求“源码与功能可重新构建”，第 1 项可以从 ModelScope 重下，第 2 项可以按版本
重装；但只有逐文件哈希完全匹配时才能称为 bit-exact。若要保留旧 profiler/tensor capture
原件，还需额外备份约 115 GB 历史资产；它们不是重跑最终结果的必要输入，却是完整历史
取证所需内容。

## 2. 本次真正做过的空目录演练

本次不是仅检查文档，而是按“新容器只有 GitHub SSH 权限，没有旧工作目录”的假设执行了
以下恢复链：

| 演练步骤 | 实际结果 |
| --- | --- |
| 从私人 GitHub 克隆冻结标签到空目录 | 通过；源码提交、子模块和文档清单可读取 |
| 从 GitHub 获取 vLLM-HUST 固定基线 | 通过；精确 fetch `933842ab0fdd53ee47dee8148bca1671d7d6da61` |
| 校验并应用 `specslo-core.patch` | 通过；14 个修改/新增文件完整落地，反向 `--check` 通过 |
| 固定 Catlass 子模块 | 通过；`41bf90da655bba3c66d0acd7e00abe33960ecfd6` |
| 校验并解压 portable evidence | 通过；约 1.9 GB、1,073 个恢复文件，压缩包和逐文件 SHA 均通过 |
| workload 身份 | 通过；SHA `dc58dc5a…`，463 请求、RPS=4、40/50/150 ms |
| candidate 完整结果身份 | 通过；SHA `943cba80…` |
| v149 production profile 身份 | 通过；SHA `88d70b00…` |
| TP4 baseline 完整结果 | 首轮发现缺口；现已另存 zstd，解压 SHA `2b957c1e…` 并加入 Git |
| Conda explicit dry-run | 能解析，但只覆盖 29 个 Conda 基础包，**不能重建完整 Python/CANN 栈** |
| 正式三轮 A/B runner `--dry-run` | 通过；输出 TP4↔TP1+TP3 三轮交错计划、同 workload 和双门合同 |
| 新的只读一键预检 | 轻量模式为 21 PASS / 2 WARN / 0 FAIL；完整读取 67 GB 模型后为 **22 PASS / 1 WARN / 0 FAIL** |

恢复演练没有启动推理，也没有占用或预留 NPU。`--dry-run` 明确不查询或启动 NPU；真正新机
仍必须完成数值、图零回退、MC2 资格化和吞吐回归，才能证明运行时复现。
本次 `--full-model-hash` 已确认当前两套 Qwen3 目录与仓库清单逐文件完全一致；剩余一个
warning 是卡上进程占用必须在正式测试前人工审核，预检不会替用户清理或预留 NPU。

## 3. 已经安全放在 GitHub 的内容

私人仓库：

```text
git@github.com:Renty-0/vllm-Ascend-SpecSLO.git
冻结标签：specslo-reproduction-20260930-v2
开发分支：SpecSLO-Chain
对话分支：codex-conversations-backup-20260924
```

冻结标签包含：

- SpecSLO、PEARL、图模式、双 batch、rolling eager、TP3 MC2 的源码；
- vLLM-HUST `933842ab…` 所需完整补丁及其 SHA；
- Catlass 固定子模块；
- 固定 workload、候选完整 `result.json`、v149 profile 和报告；
- 完整 TP4 baseline 原件的 zstd 压缩包；
- 编译运行时回滚包；
- 环境版本快照、模型逐文件 SHA、最终指标目标；
- 一键源码恢复器和一键预检脚本；
- 当前所有优化、否决方向、代码位置和复现顺序。

因此不需要再把 `/root/data/vllm-hust` 的脏工作树人工复制到新容器；应使用仓库中的
`bootstrap_from_github.sh` 从公开固定提交重建。也不需要为了最终 463-request 实验备份
588 GB `/data/shared_datasets`，因为精确 workload 与必要小型数据已经进入证据包。

## 4. 仍须外部备份的内容与优先级

### P0：缺失后无法保证完美复现

| 内容 | 当前规模/身份 | 为什么 GitHub 不能替代 |
| --- | ---: | --- |
| 两个 Qwen3 模型目录 | 约 67.06 GB | 上游 revision 未记录；未来同名模型文件可能变化，SHA 只能发现不同，不能恢复旧字节 |
| 当前 OCI 镜像 digest/导出包 | 容器内无法获知 | 镜像 tag 可变；CANN、系统库、编译器和 Python ABI 会影响图与算子 |
| 宿主/启动合同 | 910B2、Driver 26.0.rc1、ascendhal 7.35.23 | 容器不能自行恢复驱动、固件、设备映射、共享内存和 NUMA/HCCL 配置 |

当前容器里最接近的构建线索是
`quay.io/ascend/cann:9.0.0-910b-openeuler24.03-py3.12`，来自
`vllm-ascend@80610e4438dba05011b05f89fc45d91e96992671` 的
`Dockerfile.openEuler`；但容器内没有 Docker/Podman，也没有注入 OCI digest，故这只是
重建线索，不是当前镜像身份的证明。详情见 `BASE_CONTAINER_PROVENANCE.txt`。必须让管理员
在宿主侧查询运行容器对应的 image ID/RepoDigest。

### P1：强烈建议保存，显著降低恢复风险

| 内容 | 当前规模 | 说明 |
| --- | ---: | --- |
| `/root/miniconda3` | 约 11 GB；实际实验 env 约 3.3 GB | explicit 文件只含 29 个基础包；`pip freeze` 中多个 CANN 包指向已不存在的本地路径 |
| `/usr/local/Ascend/cann-9.0.0` 或原始安装介质 | 约 9.0 GB | 当前没有找到完整 CANN 9.0.0 原始 installer；目录备份是兜底，但优先使用官方匹配安装包 |
| `/root/.triton/llvm` | 约 4.5 GB | 可重下载/重建，但保存可避免镜像失效和工具链漂移 |
| 当前 v149 custom-op `.run`/vendor | 已有 GitHub 编译回滚包 | ABI 一致时可回滚；新机正常路径仍应从源码重编并重新资格化 |

完整环境中的关键版本是：Python 3.11、torch 2.10.0、torch-npu 2.10.0、triton 3.5.0、
triton-ascend 3.2.1、transformers 5.14.1、ModelScope 1.40.1。注意
`vllm-hust-dev.conda-explicit.txt` 不是完整环境锁；仅执行它不会安装上述全部 pip/CANN 包。

### P2：只在需要历史取证时保存

- 约 114.64 GB raw benchmark/profile/tensor capture；
- `/root/ascend` 中约 2.7 GB 历史 Ascend 日志；
- 旧 Qwen2.5 派生模型和历史 benchmark 全量目录；
- 588 GB 管理员共享 `/data/shared_datasets`。

这些内容可以帮助审核旧优化过程，但不参与当前 Qwen3 最终工作负载的最小闭环。新机上的
MC2 真实输入 capture 和 profile 必须重新生成；复制旧 profile 不能替代新机器资格化。

## 5. 清机前外部备份命令

以下命令中的 `/mnt/external/specslo-20260930` 必须是真正不会随容器清除的异机盘、NAS 或
对象存储挂载。**不要把备份写回本容器的 `/root/data` 或 `/data` 后误以为已经离机。**

### 5.1 模型：最低必做

首选可续传的逐文件复制：

```bash
BACKUP_ROOT=/mnt/external/specslo-20260930
mkdir -p "$BACKUP_ROOT/models"
rsync -aH --partial --append-verify --info=progress2 \
  /data/shared-models/Qwen3-0.6B "$BACKUP_ROOT/models/"
rsync -aH --partial --append-verify --info=progress2 \
  /data/shared-models/Qwen3-32B "$BACKUP_ROOT/models/"

cd /
sha256sum -c /root/data/vllm-ascend-hust/docs/migration/20260930/\
CORE_MODEL_SHA256SUMS
```

复制完成后在外部副本上把清单中的 `/data/shared-models` 前缀映射到
`$BACKUP_ROOT/models` 再校验，不能只比较目录大小。若外部系统更适合大文件归档，可用
`tar | zstd -T0 -3 | split -b 20G`，并为所有分片另建 SHA256 清单。

### 5.2 容器镜像：让管理员在宿主侧执行

Docker 示例：

```bash
docker inspect <container-name-or-id> \
  --format 'image_id={{.Image}} image_ref={{.Config.Image}}'
docker image inspect <image-id> \
  --format 'id={{.Id}} repo_digests={{json .RepoDigests}}'
docker save <image-id> | zstd -T0 -3 \
  > /mnt/external/specslo-20260930/specslo-container-image.oci.tar.zst
sha256sum /mnt/external/specslo-20260930/specslo-container-image.oci.tar.zst \
  > /mnt/external/specslo-20260930/specslo-container-image.oci.tar.zst.sha256
```

若平台是 containerd/Kubernetes，应由管理员用对应的 `ctr`/`crictl`/镜像仓库流程记录
immutable digest 并导出。新容器必须记录相同的设备、IPC/shared-memory、ulimit、volume、
NUMA 和环境变量配置；只有文件系统 tar 没有这些启动参数仍不完整。

### 5.3 环境兜底

如果无法导出 OCI 镜像，至少保存以下目录：

```bash
BACKUP_ROOT=/mnt/external/specslo-20260930
mkdir -p "$BACKUP_ROOT/environment"

tar -C /root -cf - miniconda3 | zstd -T0 -3 \
  > "$BACKUP_ROOT/environment/root-miniconda3.tar.zst"
tar -C /usr/local/Ascend -cf - cann-9.0.0 | zstd -T0 -3 \
  > "$BACKUP_ROOT/environment/cann-9.0.0.tar.zst"
tar -C /root -cf - .triton/llvm | zstd -T0 -3 \
  > "$BACKUP_ROOT/environment/triton-llvm.tar.zst"

sha256sum "$BACKUP_ROOT"/environment/*.tar.zst \
  > "$BACKUP_ROOT/environment/SHA256SUMS"
```

目录 tar 是同架构/同系统下的灾难恢复兜底，不等价于官方安装流程。新宿主仍应安装匹配
driver/firmware；不要用 tar 覆盖宿主 `/usr/local/Ascend/driver`。认证文件、SSH 私钥、Codex
token 和 GitHub token 不得放进普通 GitHub 或上述未加密归档，应在密码管理器或单独加密
介质中重新配置。

### 5.4 可选的离线 Git 保险

虽然远端已经推送完成，若担心私人仓库权限或远端删除，可在外部盘再保存 bundle：

```bash
git -C /root/data/vllm-ascend-hust bundle create \
  /mnt/external/specslo-20260930/vllm-Ascend-SpecSLO.bundle --all
git -C /root/data/vllm-hust bundle create \
  /mnt/external/specslo-20260930/vllm-hust.bundle --all
git bundle verify /mnt/external/specslo-20260930/vllm-Ascend-SpecSLO.bundle
git bundle verify /mnt/external/specslo-20260930/vllm-hust.bundle
```

## 6. 新容器的完整恢复次序

1. 管理员提供相同硬件、驱动、CANN、OCI 镜像和启动参数；
2. 重新配置 GitHub SSH，不从仓库恢复密钥；
3. 克隆私人仓库冻结标签 `specslo-reproduction-20260930-v2`；
4. 运行 `tools/specslo_migration/bootstrap_from_github.sh`；
5. 校验并解压 `migration_artifacts/20260930`；
6. 从外部备份恢复两个模型，或从 ModelScope 重下后做完整 SHA；
7. 重建/恢复 Conda 环境，安装两个 editable 源码树；
8. 运行：

   ```bash
   bash tools/specslo_migration/preflight_zero_container.sh --full-model-hash
   ```

9. 只有 `FAIL=0` 后才跑 CPU、eager/graph、MC2、tree 零回退和 device-overlap 门；
10. 重新编译 v149 vendor、重新 capture/资格化 profile；
11. 审核正式 runner 的 `--dry-run`，再执行三轮交错 A/B；
12. 将新结果与 `FINAL_EXPERIMENT_TARGET.json` 对照并保留完整证据。

`preflight_zero_container.sh` 是只读检查，不会运行占卡程序，也不会自动认定 NPU 空闲。卡上
是否有别人的进程必须在实验启动前查看 `npu-smi info`；本次演练时卡上确有其他容器进程，
因此没有把“8 卡可见”误写成“卡空闲”。

## 7. 怎样才叫“完美复现”

“完美”需要分层定义，不能要求两次硬件计时逐 bit 相同：

| 层级 | 验收 |
| --- | --- |
| 源码/数据 bit-exact | Git commit、补丁、submodule、workload、模型、历史 result SHA 全匹配 |
| 软件栈 ABI-exact | OCI digest 或 CANN/driver/Python/torch-npu/编译器身份一致，扩展重新通过数值门 |
| 功能 exact | 输出 token hash/接受、回滚、graph/MC2 fallback、跨 rank 一致性门通过 |
| 性能复刻 | 同卡同合同下 baseline/candidate 各指标在目标 ±3%，Goodput ratio 至少 1.40x |
| 项目最终门 | 三轮交错 A/B 每轮同时 Raw(candidate)>Raw(baseline) 且 Goodput>=1.3x |

当前 GitHub 中的旧检查点是：TP4 baseline Raw `930.004815`、Goodput `385.660744`；
TP1+TP3 candidate Raw `926.955552`、Goodput `560.577871`；Goodput `1.453552x`，Raw
`0.996721x`。它是**单轮最佳检查点**，不是已经完成三轮 Raw 双门。新容器先复刻这个检查点，
再继续优化 Raw 与三轮稳定性。

## 8. 当前还必须由用户/管理员补充的信息

在清机前，仍需拿到并离机保存：

- 当前运行容器的实际 OCI image ID、RepoDigest、导出包 SHA；
- 容器创建命令或平台任务 YAML，尤其是 device、IPC、共享内存、volume、NUMA、ulimit；
- 新服务器将提供的 910B2 型号/拓扑、driver 和 firmware 版本；
- 外部备份目标位置及至少约 100 GB 可用空间；若连历史 profile 一起保存，建议 250 GB
  以上，若保存整个 `/root/data` 和共享数据则按 400 GB 以上规划；
- 模型外部副本最终 SHA 校验结果。

若这五项没有离开即将清除的容器，文档再完整也只能做到源码级/功能级重建，不能保证原
环境 bit-exact。反过来，只要 P0 外部资产保存成功，GitHub 冻结标签和本手册已经覆盖从
空容器拉代码、恢复证据、验环境、重编算子、重跑最终实验的闭环。
