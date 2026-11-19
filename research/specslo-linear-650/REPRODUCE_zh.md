# b64 固定 γ=4、loose 门槛 6 的复现步骤

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
| 模型 | `/root/data/shared-models/Qwen3-0.6B` 和 `Qwen3-32B` | 使用上层手册的模型文件清单校验 |

当前机器可从之前的同源码实验目录复用已编译扩展：

```bash
ln -s /root/SpecSLO/specslo-b64-kv640-coarse-20261113/source/vllm_ascend/vllm_ascend_C.cpython-311-aarch64-linux-gnu.so vllm_ascend/vllm_ascend_C.cpython-311-aarch64-linux-gnu.so
sha256sum /root/data/specslo-baselines/nano-pearl-ascend-v1/runtime/manifest.jsonl
sha256sum /root/data/specslo-baselines/nano-pearl-ascend-v1/runtime/mc2-profile.json
sha256sum /root/data/specslo-baselines/nano-pearl-ascend-v1/runtime/mc2-vendor/op_api/lib/libcust_opapi.so
sha256sum vllm_ascend/vllm_ascend_C.cpython-311-aarch64-linux-gnu.so
```

若扩展不存在、SHA 不同、CANN/torch-npu 改版或换机器，必须按上层手册重建扩展和 MC2 vendor/profile，不能复用旧二进制后把结果当成同一合同。运行脚本默认使用 `/root/miniconda3/envs/vllm-hust-dev`、持久 CANN 9.0 环境和该机上的 vLLM-HUST 核心；这些依赖不在本分支中。

## 2. 选择卡并运行

先执行 `npu-smi info` 确认可用的四张**物理**卡和空闲状态。原三轮合同使用物理卡 `0,1,2,7`；2026-11-20 当前容器可见 `2,3,5,7`，映射器将其映射为容器设备 `0,1,2,3`。设备不同须在结果中注明，不能称为同卡重复。

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
