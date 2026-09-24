# SpecSLO TP3 MC2 v100 运行清单

本文档记录 2026-09-22 对 Qwen3-32B TP3 attention output projection 热点进行的算子和
production-adapter 验证。它不是 TP1+TP3 全模型吞吐或 Goodput 报告。

## 1. 环境与身份

- 分支：`SpecSLO-Chain`
- 基础 HEAD：`d3cbad96393d5a15dc3a36ddd2628db96188f781`；测试时工作树含未提交实现
- Python：3.11.15
- PyTorch：2.10.0+cpu
- torch-npu：2.10.0
- CANN OPP：9.0.0，timestamp `20260428_134817545`
- Ascend driver：26.0.rc1
- NPU：Ascend 910B2；仅使用物理卡 0、1、2，未启动占卡程序
- active vendor：`csrc/build/mc2-test-install-tp3-production-v98/vendors/custom_transformer`
- adapter/kernel source hash：
  `ce42bd11c2305e828e7afb31e6cb95332a3f252529b95d4f6e3298ad356aa119`
- source/vendor AIV SHA256：
  `e3aeef239ebc4ef32ca1c59654f5b460a8993aaa037acc8d980a692ea4763272`

`mc2_source_sha256()` 的覆盖范围是 `vllm_ascend/spec_decode/pearl/mc2.py` 与
`csrc/mc2/**`，不包含完整 model runner、构建脚本或本测试脚本。

## 2. 资格测试命令

三轮均使用同一命令参数，仅修改 `--output` 与 tee 日志名：

```bash
source csrc/build/mc2-test-install-tp3-production-v98/vendors/custom_transformer/bin/set_env.bash
export ASCEND_RT_VISIBLE_DEVICES=0,1,2
torchrun --standalone --nproc-per-node=3 -- \
  examples/measure_specslo_mc2.py \
  --m 64 --k 3072 --n 5120 --weight-formats ND \
  --warmup-replays 20 --replays-per-sample 50 --samples 21 \
  --latency-percentile 95 --minimum-speedup 1.05 --epilogue custom \
  --norm-atol 0.01 --norm-rtol 0.01 \
  --added-atol 0.01 --added-rtol 0.01 \
  --changed-input-replays 8 --changed-input-delta 0.015625 \
  --input-dir /root/data/nano-pearl-benchmark-results/20260920-tp3-target-optimization/mc2-real-inputs \
  --output <profile.json>
```

结果：

| 轮次 | profile | p50 | p95 | mean | 最差配对 |
|---|---|---:|---:|---:|---:|
| 1 | `qwen3-m64-production-v100-scaled-candidate.json` | 1.09633x | 1.26703x | 1.10747x | 1.06672x |
| 2（正式） | `qwen3-m64-production-v100-scaled-final-qualified.json` | 1.07732x | 1.06480x | 1.07902x | 1.05451x |
| 3 | `qwen3-m64-production-v100-scaled-repeat3.json` | 1.10580x | 1.10004x | 1.10638x | 1.09216x |

每轮 21/21 个配对样本均达到 `>=1.05x`。第一轮 p95 受 baseline 两个尾部样本影响，
正式资格采用第二轮，避免择优。

## 3. 严格 adapter smoke 命令

```bash
source csrc/build/mc2-test-install-tp3-production-v98/vendors/custom_transformer/bin/set_env.bash
export ASCEND_RT_VISIBLE_DEVICES=0,1,2
torchrun --standalone --nproc-per-node=3 -- \
  examples/check_specslo_mc2_production_adapter.py \
  --profile /root/data/tp3-mc2-debug-20260922/qwen3-m64-production-v100-scaled-final-qualified.json \
  --input-dir /root/data/nano-pearl-benchmark-results/20260920-tp3-target-optimization/mc2-real-inputs \
  --repeats 8 --changed-input-replays 8 --changed-input-delta 0.015625
```

结果为 `status=passed`。三 rank 均为 `attempt=2, success=2, fallback=0,
exception=0`；8 次静态和 8 次 changed-input graph replay 不重入 Python dispatch。
静态 Norm/Add 最大绝对误差为 0.00390625/0.0078125，changed-input 为
0.005859375/0.03125；对应最大 scaled error 为 0.42674/0.71429，均小于 1，rank 间差异为 0。

## 4. 证据路径与范围

- profile：`/root/data/tp3-mc2-debug-20260922/qwen3-m64-production-v100-scaled-final-qualified.json`
- profile 日志：`/root/data/tp3-mc2-debug-20260922/qwen3-m64-production-v100-scaled-final-qualified.log`
- adapter 日志：`/root/data/tp3-mc2-debug-20260922/qwen3-m64-production-v100-scaled-final-adapter-smoke.log`
- vendor 构建日志：`/root/data/tp3-mc2-debug-20260922/tp3-v98-final-source-build.log`

当前证据证明精确形状 `M64/K_local3072/N5120/BF16/ND` 的 rank-3 fused
MatMul+AllReduce+AddRMSNorm 在该真实层 payload 及其 8 个平移输入上正确、图模式无 adapter
回退且算子级有正收益。它没有覆盖 Qwen3-32B 所有层/权重，也没有覆盖完整 TP1+TP3
model-runner、在线调度、吞吐和 Goodput；这些需要后续端到端回归。
