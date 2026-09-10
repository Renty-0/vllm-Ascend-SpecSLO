# SpecSLO 2026-09-09 回归记录

## 测试口径

- 平台：Ascend 910B2，CANN `socversion=2201`。
- 模型：Qwen3-0.6B draft（TP1）+ Qwen3-32B target（TP3）。
- baseline：同一份 request manifest 上的 Qwen3-32B target-only（TP4）。
- 负载：论文三类请求，RPS=4，128 条请求，类别为 tight/normal/loose=77/26/25，
  `max_tokens=32`，arrival trace 相同。
- SpecSLO 使用固定 paged-attention graph、gamma=4、`max_num_seqs=64`、384 个 KV
  page；warmup 不计入测量，E2E 包含到达等待、prefill 和 generation。

## 结果

| 配置 | E2E 吞吐 (tok/s) | 接受率 | SLO 达成率 | Goodput (tok/s) |
| --- | ---: | ---: | ---: | ---: |
| vLLM-Ascend target-only TP4，graph | 108.08 | - | 未提供有效在线 metrics | 未计算 |
| vLLM-Ascend target-only TP4，eager | 95.49 | - | 未提供有效在线 metrics | 未计算 |
| SpecSLO TP1+TP3，旧的提前首批 prefill | 101.18 | 42.94% | 9.38% (12/128) | 9.49 |
| SpecSLO TP1+TP3，arrival-aware packed admission | 101.20 | 75.24% | 7.81% (10/128) | 7.91 |
| SpecSLO TP1+TP3，显式 target cap=32 | 99.72 | 75.27% | 7.03% (9/128) | 7.01 |

arrival-aware 版本相对 graph baseline 为 `0.936x`，没有达到 1.3x。这个负载的
arrival offset 横跨约 31.6 秒，而 4096 个输出 token 的理论 E2E 吞吐上界约为
`4096/31.6=129.6 tok/s`；因此在 arrival-inclusive 口径下，1.3 * 108.08 =
140.5 tok/s 数学上已经超过该负载上界。要验收 1.3x，必须增加持续时间/请求数或改用
稳态服务吞吐口径，不能通过修改 Goodput 分母实现。

## 已完成的修改

1. **arrival-aware packed admission**：online-prefill 不再在首个 arrival 前预填整批
   未来请求；controller 每次选择当前 ready 的请求，并一次调用 packed prefill，保持
   draft/target 的共同 committed frontier。128 条请求本轮实际为 89 次 packed prefill，
   覆盖 128 个请求，而不是每个请求单独启动一次 target prefill。
2. **稳定 graph 默认开关**：新增 `spec_rhythm_stable_graphs`（CLI 为
   `--spec-rhythm-stable-graphs/--no-spec-rhythm-stable-graphs`）。启用时 SpecSLO
   不使用已知在动态 query shape 下不稳定的 FIA；动态 FIA 仅作为显式实验选项。
3. **target verification 行数上限**：新增 `--spec-rhythm-max-target-batch`，用于在
   SLO 实验中主动收紧单轮 target 行数。实测 cap=32 将吞吐和达成率都降低，因此默认
   不启用；参数保留用于后续按 SLO 类别做分层调度实验。
4. **TP3 fused MM/all-reduce capability probe**：开启
   `VLLM_ASCEND_PEARL_ENABLE_TP3_MM_ALL_REDUCE=1` 做过实机 probe，失败时自动回退
   普通 HCCL，避免破坏正确性。

## 已否决或硬件阻塞的方向

- 当前 A2 CANN 明确拒绝 TP3 fused kernel：`Rank size 3 is not supported by socversion
  id:2201; A2 supports rank size 1,2,4,8`。因此 TP1+TP3 无法直接复用
  `npu_mm_all_reduce_base`/MC2；这不是 Python 调度参数可以解决的问题。
- target cap=32：增加 verification rounds，E2E 从 40.47 s 增至 41.08 s，否决为默认策略。
- 动态 FIA：曾出现 graph replay shape 不稳定和显著降速；固定 paged graph 保留为
  SpecSLO 默认。

## 当前瓶颈

arrival-aware 版本的 target leader 累计阶段约为 target compute 20.59 s、target
verify 6.52 s、HCCL exchange 0.15 s、broadcast 0.19 s、refill 11.76 s；这些是
worker 累计值，不能直接相加为 E2E，但能说明主要开销在 TP3 target forward、verdict
计算和 admission/refill。当前控制流虽让 draft 与 target worker 分属不同 NPU，但
target verification、HCCL verdict 和 state update 仍以轮次为边界；论文意义上的
跨 home 真正异步 mailbox 尚未实现。

基线脚本已补充 `online_timing` 字段（wall、engine step、arrival sleep），但旧的
vLLM 输出对象没有 `metrics`，所以本轮 baseline 的 TPOT/Goodput 仍不能声称有效；
后续必须让 baseline 也提供 per-request first/last token timestamp，再进行 80%/1.3x
验收。

## 可复现实验文件

- workload：`/tmp/rps4_mix_p128_t32_batched.jsonl`
- SpecSLO arrival-aware：`/tmp/specslo-batched-admission-p128-t32-online-0909.json`
- SpecSLO TP3 fused probe：`/tmp/specslo-tp3-fused-mm-p128-t32-0909.json`
- target cap=32：`/tmp/specslo-targetcap32-p128-t32-0909.json`
- baseline graph：`/tmp/specslo-baseline-p128-t32-tp4-graph-0909.json`
- baseline eager：`/tmp/specslo-baseline-p128-t32-tp4-eager-0909.json`

本记录不把未达到的 1.3x 写成已完成；下一阶段若要继续逼近目标，应优先实现真正的
异步双 home mailbox/跨轮 overlap，或在支持 TP3 的 CANN/硬件上验证 fused all-reduce，
再进行长期稳态 workload 的 Goodput 测试。
