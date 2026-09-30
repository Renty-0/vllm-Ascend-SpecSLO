# SpecSLO 新容器从零复现手册

本文给出从空的昇腾开发容器恢复 SpecSLO 的可执行顺序。它与
[`README_zh.md`](README_zh.md) 的分工是：审计文档说明“保存了什么、还缺什么”，本文说明
“在新容器上按什么顺序恢复、重建和验收”。

已经实际执行过的空目录恢复步骤、OCI 镜像缺口和清机前外部备份命令见
[`ZERO_CONTAINER_REHEARSAL_AND_BACKUP_GAPS_zh.md`](ZERO_CONTAINER_REHEARSAL_AND_BACKUP_GAPS_zh.md)。

## 1. 能恢复到什么程度

| 恢复等级 | 仅依赖私人 GitHub | 还需要什么 | 结论 |
| --- | --- | --- | --- |
| 源码、补丁、技术文档、对话、结构化实验结果 | 是 | 私人仓库读取权限 | 可完整恢复 |
| SpecSLO 功能级运行 | 基本可以 | 匹配的 CANN/驱动、Qwen3 两个模型、4 张空闲 910B2 | 可重新构建并运行 |
| 可比较的 TP1+TP3 对 TP4 性能结果 | 否 | 同型号机器、重新编译/资格化 MC2、相同 workload 和运行合同 | 可重测，不能复制旧数字冒充新机器结果 |
| 逐文件、逐二进制恢复旧容器 | 否 | 异机保存的模型、raw profile、tensor capture、Conda/LLVM 等大包 | 只有外部大包才能做到 |

因此“让新容器自己复现”是可行的：从源码和固定 workload 重新生成运行环境与实验结果，
不要求备份旧模型、Conda 或 CANN 目录。模型没有私有转换，可从 ModelScope 重下并以 SHA
验证；Conda/CANN 没有私有二进制修改，可按固定版本重装。只有历史 profiler 原件、真实
张量捕获或旧容器逐字节快照在没有外部副本时无法恢复。

## 2. 不可由容器自行解决的管理员前置条件

开始恢复前，管理员必须提供以下条件；任一项不满足，都不要开始性能对比：

1. aarch64 容器，至少 4 张同节点 Ascend 910B2；正式复现推荐保留当前的 8 卡节点条件；
2. Driver `26.0.rc1`、`ascendhal 7.35.23`；
3. CANN toolkit 9.0.0，内部版本 `V100R001C10SPC001B250`；
4. 容器能访问 `/dev/davinci*`、HCCL，并有足够的共享内存；
5. 至少约 90 GB 可用空间用于两个 Qwen3 模型、源码、编译和基础实验；若恢复全部历史
   资产，按审计结果预留至少 400 GB；
6. 若实验依赖 `/data/shared_datasets`，管理员需重新挂载。当前 GitHub 恢复包不包含该
   588 GB 只读共享目录。

新容器首先保存以下检查结果，随后与本目录中的快照逐项比较：

```bash
mkdir -p /root/data/specslo-rehydrate-audit
npu-smi info > /root/data/specslo-rehydrate-audit/npu-smi-info.txt
uname -a > /root/data/specslo-rehydrate-audit/uname.txt
cat /etc/os-release > /root/data/specslo-rehydrate-audit/os-release
findmnt > /root/data/specslo-rehydrate-audit/mounts.txt

cat /usr/local/Ascend/driver/version.info
cat /usr/local/Ascend/cann-9.0.0/ascend_toolkit_install.info
```

对照文件为 `npu-smi-info.txt`、`uname.txt`、`os-release`、`mounts.txt`、
`usr-local-Ascend-driver-version.info` 和 `cann-ascend_toolkit_install.info`。驱动/CANN 不匹配时，
GitHub 中的已编译 `.so` 和 CANN vendor 只能用于取证，不能直接加载。

## 3. 获取源码并固定身份

认证信息不得从旧容器 GitHub 分支恢复。应在新容器重新配置 SSH key，然后执行：

```bash
mkdir -p /root/data
git clone --branch specslo-reproduction-20260930-v3 --single-branch \
  git@github.com:Renty-0/vllm-Ascend-SpecSLO.git \
  /root/data/vllm-ascend-hust

git -C /root/data/vllm-ascend-hust status --short --branch
git -C /root/data/vllm-ascend-hust rev-parse HEAD
```

不要 checkout 私人仓库的 `main`，也不要用活动的 `SpecSLO-Chain` 分支头代替冻结标签。
活动分支只用于复刻完成后的后续调优。复现实验前把实际提交和标签写入实验目录。

SpecSLO 的 V1 scheduler/请求/输出等核心层修改属于另一个仓库。恢复它时必须从固定基线
应用本仓库保存的补丁。推荐直接运行经过幂等和 fail-closed 检查的恢复器：

```bash
bash /root/data/vllm-ascend-hust/tools/specslo_migration/\
bootstrap_from_github.sh
```

脚本从 `git@github.com:vLLM-HUST/vllm-hust.git` 按完整 SHA
`933842ab0fdd53ee47dee8148bca1671d7d6da61` 直接 fetch，不依赖默认分支；然后校验
`specslo-core.patch` SHA、应用补丁、反向检查完整补丁并固定 Catlass 子模块。若必须手工审核，
等价命令是：

```bash
mkdir -p /root/data/vllm-hust
git -C /root/data/vllm-hust init
git -C /root/data/vllm-hust remote add origin \
  git@github.com:vLLM-HUST/vllm-hust.git
git -C /root/data/vllm-hust fetch --depth=1 origin \
  933842ab0fdd53ee47dee8148bca1671d7d6da61
git -C /root/data/vllm-hust checkout --detach FETCH_HEAD

git -C /root/data/vllm-hust apply --check \
  /root/data/vllm-ascend-hust/patches/vllm-hust/specslo-core.patch
git -C /root/data/vllm-hust apply --3way \
  /root/data/vllm-ascend-hust/patches/vllm-hust/specslo-core.patch
git -C /root/data/vllm-hust status --short
```

补丁包含旧工作树 10 个修改文件和 4 个原未跟踪文件。`--check` 失败时必须先确认基线提交，
不能用 `--reject` 静默跳过冲突。dev-hub 只在重新使用 quickstart 管理器时需要，基线与补丁
分别是 `a61b3183` 和 `patches/vllm-hust-dev-hub-quickstart.patch`。

## 4. 校验并恢复 GitHub 中的小型资产

GitHub 已包含结构化结果、报告、精确 workload、项目数据集和一份编译运行时回滚层。先
恢复到临时根，而不是直接覆盖活动环境：

```bash
cd /root/data/vllm-ascend-hust/migration_artifacts/20260930
sha256sum -c SHA256SUMS
bash restore.sh /tmp/specslo-github-restore

cd /tmp/specslo-github-restore
zstd -dc /root/data/vllm-ascend-hust/migration_artifacts/20260930/\
SOURCE_SHA256SUMS.zst | sha256sum -c -
zstd -dc /root/data/vllm-ascend-hust/migration_artifacts/20260930/\
COMPILED_RUNTIME_SOURCE_SHA256SUMS.zst | sha256sum -c -
```

校验成功后，可恢复可移植数据：

```bash
mkdir -p /root/data /data
rsync -a /tmp/specslo-github-restore/root/data/specslo-workloads/ \
  /root/data/specslo-workloads/
rsync -a /tmp/specslo-github-restore/root/data/nano-pearl-benchmark-results/ \
  /root/data/nano-pearl-benchmark-results/
rsync -a /tmp/specslo-github-restore/data/datasets/ /data/datasets/
```

不要在此时把临时根中的 `vllm_ascend_C*.so` 或 custom-op vendor 覆盖进源码树。它们绑定旧
容器 ABI，只能在第 8 节校验完全一致后作为应急回滚使用；正常路径是重新编译。

### 4.1 论文 workload 的固定身份

当前正式在线 workload 已进入 GitHub 证据包，恢复后路径为：

```text
/root/data/specslo-workloads/eurosys27-rps4-seed0-author-slo40-50-150/
  rps4.0_mix0.6_0.2_0.2_seed0.jsonl
```

其合同为 463 请求，tight/normal/loose 为 271/96/96，输出上限 256，Poisson RPS=4，
生成窗口 120 秒，到达 seed=0、类别 seed=1。作者生成代码来源为
`rzwang22/eurosys27@22c196bdf571e84887e5573607bd8e734faa0625`。本项目将 tight TPOT
从作者表中的 31.56 ms 映射为当前验收使用的 40 ms，normal/loose 为 50/150 ms。

必须检查：

```bash
sha256sum /root/data/specslo-workloads/\
eurosys27-rps4-seed0-author-slo40-50-150/\
rps4.0_mix0.6_0.2_0.2_seed0.jsonl
```

期望 SHA256：
`dc58dc5af1c5b28c6944f8f3faadeaae2dae34298c07b3ac84745d367c67d662`。不要重新随机生成
一份“同分布”文件来代替它；同分布不等于同请求或同到达轨迹。

## 5. 重建 Python 环境

### 5.1 推荐路径

优先使用与旧容器相同的 vLLM-HUST/CANN 开发镜像，在其中创建环境。Conda explicit 文件只
固定 Python 基础层；`environment.yml` 和 `pip-freeze.txt` 是完整审计参考，但含 CANN 安装时
产生的本地 wheel URL、旧源码 editable URL，不能在新容器盲目执行 `pip install -r`。

```bash
# 安装 aarch64 Miniconda 后执行；不要改写系统 HOME。
/root/miniconda3/bin/conda create -y -n vllm-hust-dev \
  --file /root/data/vllm-ascend-hust/docs/migration/20260930/\
vllm-hust-dev.conda-explicit.txt

source /usr/local/Ascend/cann-9.0.0/set_env.sh
/root/miniconda3/bin/conda run -n vllm-hust-dev python -VV
```

接着通过匹配 CANN 9.0.0 的官方/校内镜像安装 torch 2.10.0、torch-npu 2.10.0 和依赖。包名、
版本及旧环境位置见 `runtime-packages.txt`、`vllm-hust-dev.environment.yml` 和
`vllm-hust-dev.pip-freeze.txt`。其中 `asc_op_*`、`te`、`hccl`、`superkernel` 等 `file://`
依赖应由新 CANN 安装提供，不能访问旧 `/root/selfgz*` 临时路径。

### 5.2 安装两个 editable 源码树

```bash
source /usr/local/Ascend/cann-9.0.0/set_env.sh

VLLM_TARGET_DEVICE=empty \
  /root/miniconda3/envs/vllm-hust-dev/bin/python -m pip install \
  --no-build-isolation -e /root/data/vllm-hust

cd /root/data/vllm-ascend-hust
git submodule update --init --recursive
COMPILE_CUSTOM_KERNELS=1 VLLM_BATCH_INVARIANT=1 \
  /root/miniconda3/envs/vllm-hust-dev/bin/python -m pip install \
  --no-build-isolation --no-deps -e .
```

如果基础镜像已经提供全部 Python 依赖，可用
`scripts/install_local_ascend_plugin.sh /root/data/vllm-ascend-hust` 执行带来源检查的 editable
安装。禁止使用 `ALLOW_EXISTING_INSTALL_FALLBACK=1` 掩盖编译失败。

安装后验证导入来源和版本：

```bash
/root/miniconda3/envs/vllm-hust-dev/bin/python - <<'PY'
from pathlib import Path
import torch
import torch_npu
import vllm
import vllm_ascend

print("torch", torch.__version__)
print("torch_npu", torch_npu.__version__)
print("vllm", Path(vllm.__file__).resolve())
print("vllm_ascend", Path(vllm_ascend.__file__).resolve())
assert Path(vllm.__file__).resolve().is_relative_to("/root/data/vllm-hust")
assert Path(vllm_ascend.__file__).resolve().is_relative_to(
    "/root/data/vllm-ascend-hust"
)
PY
```

## 6. 恢复模型

当前 SpecSLO 正式路径只要求以下两个完整模型：

| 本地目录 | 上游模型 ID | 旧容器规模 | 旧下载 revision |
| --- | --- | ---: | --- |
| `/data/shared-models/Qwen3-0.6B` | `Qwen/Qwen3-0.6B` | 1,519,288,320 B | 未记录 |
| `/data/shared-models/Qwen3-32B` | `Qwen/Qwen3-32B` | 65,540,485,120 B | 未记录 |

网络不稳定时首选异机 `rsync --partial --append-verify` 恢复旧权重。若只能重下载，可使用
ModelScope 镜像：

```bash
mkdir -p /data/shared-models /data/.cache/modelscope
export MODELSCOPE_CACHE=/data/.cache/modelscope

/root/miniconda3/envs/vllm-hust-dev/bin/python - <<'PY'
from modelscope import snapshot_download

snapshot_download("Qwen/Qwen3-0.6B", local_dir="/data/shared-models/Qwen3-0.6B")
snapshot_download("Qwen/Qwen3-32B", local_dir="/data/shared-models/Qwen3-32B")
PY
```

旧容器没有保存这两个目录的下载 revision，但现在已经保存两套核心模型逐文件 SHA256。
ModelScope 下载后用 `CORE_MODEL_SHA256SUMS` 判断是否与旧模型逐文件一致；若上游 revision
已变化，应记录差异并重新跑数值门，只有恢复旧权重原件时才需要可选外部副本。不能臆造
revision。四个派生 Qwen2.5 draft/vocab 模型也没有足够的转换配方从基础模型自动恢复，历史
Qwen2.5 回归需要从外部包恢复它们。

模型完整性最低检查：

```bash
test -f /data/shared-models/Qwen3-0.6B/config.json
test -f /data/shared-models/Qwen3-32B/config.json
if find /data/shared-models/Qwen3-0.6B /data/shared-models/Qwen3-32B \
  -name '*.incomplete' -print -quit | grep -q .; then
  echo 'model contains incomplete files' >&2
  exit 1
fi

sha256sum -c /root/data/vllm-ascend-hust/docs/migration/20260930/\
CORE_MODEL_SHA256SUMS
```

## 7. 先过源码和 CPU 回归

```bash
cd /root/data/vllm-ascend-hust
git diff --check

/root/miniconda3/envs/vllm-hust-dev/bin/python -m pytest -q \
  tests/ut/spec_decode/test_pearl_scheduler.py \
  tests/ut/spec_decode/test_pearl_tree.py \
  tests/ut/spec_decode/test_spec_rhythm_native.py \
  tests/ut/spec_decode/test_spec_rhythm_tree_graph.py \
  tests/ut/spec_decode/test_specslo_admission.py \
  tests/ut/spec_decode/test_specslo_benchmark_metrics.py \
  tests/ut/spec_decode/test_pearl_mc2.py
```

这一步只证明控制面、tree/rolling-eager、指标和 MC2 adapter 的软件语义，没有证明 NPU
数值正确、图不回退或设备 overlap。

## 8. 重建并资格化 TP3 MC2

### 8.1 源码构建

当前生产候选来自 `matmul_allreduce_add_rmsnorm` 的 v149 自适应 epilogue 源码。建议把它
安装到独立目录，不覆盖基础 `custom_transformer`：

```bash
source /usr/local/Ascend/cann-9.0.0/set_env.sh
cd /root/data/vllm-ascend-hust/csrc

bash build.sh --pkg --soc=ascend910b \
  --ops=matmul_allreduce_add_rmsnorm \
  --vendor_name=specslo_mc2_v149_adaptive_epilogue_transformer \
  -j16 -O3

mkdir -p build/specslo-mc2-v149-install
./build/cann-ops-transformer*.run \
  --install-path=/root/data/vllm-ascend-hust/csrc/build/\
specslo-mc2-v149-install
```

实际 vendor 根应含 `op_api/lib/libcust_opapi.so`，且导出以下符号：

```bash
SPECSLO_V149_VENDOR=/root/data/vllm-ascend-hust/csrc/build/\
specslo-mc2-v149-install/vendors/\
specslo_mc2_v149_adaptive_epilogue_transformer

nm -D "$SPECSLO_V149_VENDOR/op_api/lib/libcust_opapi.so" \
  | grep aclnnAllreduceAddRmsnormGetWorkspaceSize
```

若打包器生成的 vendor 目录名不同，以 `.run` 的实际安装清单为准，并把实际路径显式传给
后续脚本；不要创建一个指向旧 v134 的伪目录。

### 8.2 资格化不能省略

MC2 profile 绑定源码 SHA、vendor payload、adapter 二进制、CANN/HCCL 和物理 rank 映射。
旧 profile
`20260929-specslo-mc2-v149-adaptive-production-target457/qwen3-32b-tp3-v149-adaptive-production.json`
可用来审计旧结果，但新编译二进制必须重新捕获真实 Qwen3-32B 层输入并资格化：

```bash
export SPECSLO_VISIBLE_DEVICES=0,1,2,3
export SPECSLO_CAPTURE_LAYERS=0,31,63
export SPECSLO_CAPTURE_ROWS=40,44,48,52,56,60,64,68,72,76,80,84,88,92,96,100,104,108,112,116,120,124,128,165,197,229
export SPECSLO_CAPTURE_ROOT=/root/data/specslo-rehydrate/mc2-captures

bash tools/specslo_experiments/capture_mc2_real_inputs.sh attention rehydrate-attn
bash tools/specslo_experiments/capture_mc2_real_inputs.sh down rehydrate-down

export SPECSLO_TARGET_DEVICES=1,2,3
export SPECSLO_MC2_VENDOR="$SPECSLO_V149_VENDOR"
export SPECSLO_QUALIFY_ROWS="$SPECSLO_CAPTURE_ROWS"
export SPECSLO_QUALIFY_ROOT=/root/data/specslo-rehydrate/mc2-qualification
export SPECSLO_MC2_PROFILE_FILENAME=qwen3-32b-tp3-rehydrate.json

bash tools/specslo_experiments/qualify_mc2_multishape.sh \
  /root/data/specslo-rehydrate/mc2-captures/runs/attention-rehydrate-attn/captures \
  /root/data/specslo-rehydrate/mc2-captures/runs/down-rehydrate-down/captures \
  rehydrate-v149
```

这里显式覆盖旧 profile 出现过的 M=40..128、165、197、229；资格工具会分别决定 attention
和 down projection 的 shape 是否进入生产 profile，而不是强迫所有 shape 使用 fused。正式在线
晋升还必须根据新机器实际 replay histogram 检查是否出现列表外 M；若出现，先补 capture 和
资格化，再用保守 exact-shape union 生成生产 profile。不能用 synthetic M64 或单层、单 shape
的成功代替跨层 changed-input 资格测试；已经证明 fused 较慢的大 M 应继续走 split 路径。

## 9. NPU 功能门和图模式门

按以下顺序验证，每一步失败都应停止性能测试：

1. eager 下 target-only 与 SpecSLO 数值回归；
2. graph 下相同输入和输出回归；
3. eager/graph × MC2-off/on 四象限；
4. tree draft 的 capture/replay，要求 shape/capacity fallback 为 0；
5. 在线 Poisson 请求、动态 prefill 插入、取消/异常传播；
6. 双 batch 的 CANN device overlap 门，而不只是 host 同时提交计数。

至少执行：

```bash
cd /root/data/vllm-ascend-hust
/root/miniconda3/envs/vllm-hust-dev/bin/python -m pytest -q \
  tests/ut/spec_decode/a2_2/test_pearl_greedy_aclgraph_hccl.py \
  tests/ut/spec_decode/a2_2/test_pearl_protocol_hccl.py
```

测试结束后在 `result.json`/门禁产物中检查：

- `graph_shape_fallbacks == 0`；
- `graph_capacity_fallbacks == 0`；
- `mc2_fallbacks == 0` 且 `mc2_exceptions == 0`；
- 四个 rank 输出/接受计数一致；
- changed-input 和 bit-exact 门通过；
- 纯 decode step 上三个 target rank 的 target device window 均被 draft device compute 覆盖。

旧 v149 证据中，两个纯 decode step 的三个 target rank 覆盖率约 77%，它是恢复后的方向性
对照，不是允许新机器跳过 profiling 的理由。

## 10. 重跑正式在线 A/B

### 10.1 固定合同

- candidate：Qwen3-0.6B draft TP1 + Qwen3-32B target TP3；
- baseline：同四张物理卡上的 Qwen3-32B target-only TP4；
- 相同 463 请求、相同 Poisson 到达时间、相同 256 输出上限；
- TPOT SLO 为 40/50/150 ms；
- 在线 coalesce1；warmup/compile 在计时区间外；
- 三轮交错顺序为 baseline/candidate、candidate/baseline、baseline/candidate；
- 每轮均要求 candidate Raw > baseline Raw，且 paper Goodput >= baseline 的 1.3 倍；
- 图和 MC2 fallback 均为 0。

### 10.2 必须显式覆盖旧默认值

当前正式脚本的默认路径仍保留历史 v134，恢复时**禁止依赖默认 profile/vendor**。在完成第
8 节后显式运行：

```bash
cd /root/data/vllm-ascend-hust

export SPECSLO_VISIBLE_DEVICES=0,1,2,3
export SPECSLO_RESULT_ROOT=/root/data/nano-pearl-benchmark-results/\
rehydrate-v149-formal-ab
export SPECSLO_MC2_VENDOR="$SPECSLO_V149_VENDOR"
export SPECSLO_MC2_PROFILE=/root/data/specslo-rehydrate/mc2-qualification/\
runs/rehydrate-v149/qwen3-32b-tp3-rehydrate.json

export SPECSLO_DRAIN_FIFO=1
export SPECSLO_DRAIN_PADDING_REPAIR=1
export SPECSLO_TARGET_NZ_MODE=14
export SPECSLO_TARGET_LARGE_M_NZ_MIN_ROWS=161
export SPECSLO_TARGET_FIA_TASK_UPDATE_WORKERS=2
export SPECSLO_TARGET_FIA_TASK_EVENT_GROUP_SIZE=1
export SPECSLO_DRAFT_PA_TASK_UPDATE_WORKERS=2
export SPECSLO_EARLY_CURRENT_TARGET_SUBMIT=1
export SPECSLO_MAX_TARGET_BATCH=32
export SPECSLO_STABLE_TARGET_REQUEST_BUCKET=1
export SPECSLO_MIXED_TARGET_GRAPH_BUCKETS=64,128,192,256,320,384,448,512
export SPECSLO_MAX_ACLGRAPH_ENTRIES=80
export SPECSLO_NUM_KVCACHE_BLOCKS=544
export SPECSLO_ENABLE_NPU_PROFILE=0

bash tools/specslo_experiments/run_online_coalesce1_formal_ab.sh \
  --dry-run rehydrate-v149
bash tools/specslo_experiments/run_online_coalesce1_formal_ab.sh \
  rehydrate-v149
```

先审查 `--dry-run` 输出中的模型、workload、卡号、profile、vendor 和结果目录，再开始实机
运行。脚本要求四张卡在启动时无进程；本项目不再使用占卡脚本。

旧机当前最好单轮候选是 Raw 926.956 tok/s、paper Goodput 560.578 tok/s、达成率
60.475%，对同卡 TP4 的 Goodput 为 1.4535x，但 Raw 仍低 0.328%。它不是“双门槛已完成”的
声明；新机器必须以三轮正式门的实际输出为准。

## 11. raw profile、tensor capture 和历史实验如何处理

这些资产可以分为两类：

- **可重新生成但昂贵：** CANN profiler、MC2 真实层 capture、Triton LLVM cache、编译 vendor。
  新容器按第 8、9 节重新跑即可，但得到的是新证据，不是历史原件；
- **不可重新生成的历史状态：** 旧时点的失败 trace、旧二进制对应 capture、历史 profiler
  数据库和中间派生模型。它们只能从异机大包恢复。

若外部备份存在，先在外部介质执行
`tools/specslo_migration/verify_portable_archives.sh`，再逐包恢复。不要把 300+ GB 大包重新提交
普通 GitHub，也不要把 SSH/Codex/GitHub 凭据混入非加密归档。

## 12. 最终验收清单

只有全部勾选后，才可以声称新容器完成 SpecSLO 功能级复现：

- [ ] 冻结标签 `specslo-reproduction-20260930-v3` 的来源和提交已记录；
- [ ] `bootstrap_from_github.sh` 输出 `Source bootstrap passed.`；
- [ ] `preflight_zero_container.sh --full-model-hash` 输出 `FAIL=0`；
- [ ] vLLM-HUST 固定基线、核心补丁和 Catlass 提交均已校验；
- [ ] Driver/CANN/torch/torch-npu/架构与快照匹配；
- [ ] 两个 Qwen3 模型完整、没有 `.incomplete`，且 `CORE_MODEL_SHA256SUMS` 全部通过；
- [ ] workload SHA256 精确匹配；
- [ ] CPU 控制面测试通过；
- [ ] eager/graph × MC2-off/on 数值门通过；
- [ ] tree graph shape/capacity fallback 为 0；
- [ ] 双 batch 的设备 overlap 门通过；
- [ ] 当前源码、当前 vendor、当前卡映射重新生成 MC2 profile；
- [ ] TP4 baseline 与 TP1+TP3 candidate 使用同卡、同 workload、同在线 prefill 合同；
- [ ] 三轮交错 A/B 的完整结果、环境、源码身份和日志已保存；
- [ ] 任何性能结论都来自新容器实测，而不是复制旧容器报告中的数字。

## 13. 当前无法自动化的边界

1. 宿主驱动、固件、CANN 和共享挂载需要管理员提供；
2. 私人 GitHub/SSH/ModelScope 登录需要用户重新授权；
3. 模型旧 revision 未记录；当前 SHA 清单可以判定下载文件是否与旧容器相同。上游若更换
   文件仍可做功能/性能重新验证，但只有恢复旧权重原件时才需要可选外部副本；
4. 原始 100+ GB profiler/tensor capture 若没有外部副本，只能重新生成，不能恢复历史内容；
5. 物理 NPU 拓扑、HCCL 状态和系统噪声变化会改变性能，旧 MC2 profile 不能跨机器直接晋升；
6. 当前最优候选尚未同时稳定满足 Raw 超 TP4 和 Goodput 三轮均 >=1.3x，因此恢复完成与最终
   性能优化完成是两个不同里程碑。
