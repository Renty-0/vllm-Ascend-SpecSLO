# SpecSLO TP3 MC2 AIC/AIV 同步协议静态审计（2026-09-23）

## 1. 结论先行

本轮没有使用 NPU，也没有运行占卡程序。生产 `csrc/mc2` 源码未改动，因此不会让
已经生成的 v119 vendor/profile 因源码 SHA 变化而失效。

已有设备 trace 中，单个 MC2 MIX kernel 为 **134.026 us/层**，被替换的三个
kernel 的纯 duration 合计为 **123.262 us/层**，即 MC2 raw kernel 慢
`10.764 us`、`8.73%`。但原三算子之间还有 **21.597 us/层** 的两个 launch
gap，所以完整替换窗口仍从 `144.858 us` 降至 `134.026 us`，局部净快
`10.832 us/层`、`7.48%`。因此：

- 这不是 MC2 没有命中，也不是 ACLGraph 回退；
- raw kernel 的确仍有优化空间；
- 该 trace 中全 step 回归的更大来源是第一层前 prefix/首个 embedding
  all-reduce 的到达偏斜，不应把它全部归因于 MC2 kernel；
- 当前 ready/read-complete 两轮跨 rank 依赖不能直接删除；
- 当前 reset 已是每 rank 一次 24B 合并 DMA，简单改成 epoch 在现有 ABI 下既
  不能证明更快，也不能在 graph replay 中安全初始化。

本轮给出一个**未应用到生产源码**的最小实验补丁：只省去 direct TP3 中
`source == destination` 的本地 ready/done flag 写入和轮询，所有远端边、入口
rendezvous、局部 `SyncAll`、末尾 read-complete 保护均保留。CPU 穷举证明新旧
协议的“允许开始 reduce”和“允许覆盖 producer window”谓词等价。该补丁仍须在
独立 vendor 中通过 eager/ACLGraph 数值与活性门禁，不能直接生产启用。

## 2. v119 direct TP3 的实际同步链

适用分支为 `rank_size == 3 && m <= 160`。

### 2.1 AIC 侧

1. `InitFlags -> WaitEvent(12)`：等待 AIV 释放本次 MatMul；
2. AIC 对当前 split 执行 MatMul，并把结果写入本 rank 的 HCCL symmetric input
   window；
3. 每个 split 用 `FFTSCrossCoreSync<PIPE_FIX>(..., flag_idx)` 发布完成；M100 的
   small-M 路径通常只有一个 split；
4. `PipeBarrier<PIPE_ALL>()` 后退出 AIC。

### 2.2 AIV 侧

1. 用 FFTS flag 12 释放 AIC；
2. `WaitEvent(0)` 等待 AIC 的第一个/唯一 small-M split 完成；
3. 本 rank `SyncAll`；
4. `ResetTp3DirectIpcFlags(false)`：由一个 AIV core 把 ready 和 done 两个 phase
   的 `2 × 3` 个 source-owned slot 合成一次 **24B DMA** 清零；
5. `CrossRankSyncEx(FLAG_NUM, true)`：跨 rank 确认所有本地 reset 已可见；
6. `Tp3PublishAndWait(FLAG_ZERO_IDX, 1, true)`：三 source 向三 destination 发布
   payload-ready，并等待本地三个 source slot；
7. 32 个（或平台给出的全部）逻辑 AIV sub-core 分摊完整 M 行，按固定
   `rank0 + rank1 -> BF16 -> + rank2` 顺序做归约，再执行 residual/AddRMSNorm；
8. `PipeBarrier<PIPE_ALL>() + SyncAll`；
9. `Tp3PublishAndWait(FLAG_ONE_IDX, 1, true)`：发布并等待 read-complete，防止下一
   层/下一 replay 在远端读取结束前覆盖 symmetric payload window。

按 TP3 整个 cluster 计数，direct 路径每次调用有：

| 项目 | 当前值 |
|---|---:|
| 本地 flag reset DMA | 3 次（每 rank 一次，每次 24B） |
| ready/done source-owned 发布 DMA | 18 次 |
| ready/done poll lane | 18 条 |
| 有效跨 rank rendezvous | 3 轮（reset-entry、ready、done） |
| AIV `SyncAll` | 5 处 |

## 3. 为什么不能直接删除 barrier 或 reset

### 3.1 read-complete 不是冗余 barrier

MC2 的 AIC 下一层仍写同一个 rank-local symmetric payload window。若 producer
只等自己完成计算、不等两个远端 consumer 完成读取，快 rank 会覆盖慢 rank 尚在
读取的上一层 payload。仓库中的
`csrc/build.no-final-crossbarrier-rejected` 正是这一方向的历史失败构建；当前
`FLAG_ONE_IDX` 是修复这一 hazard 的必要边。

### 3.2 固定 epoch=1 必须先 reset

ready/done 当前都写固定值 1。如果不清零，graph 下一次 replay 会立即读到上次的
1，发生 stale-pass。HCCL symmetric window 的首次内容也没有在本算子的公开契约
中保证为零，不能把“通常新分配显存看起来为零”当成正确性条件。

### 3.3 单调 epoch 在当前 graph ABI 下没有免费来源

安全的单调 epoch 至少要满足：

1. 所有 consumer 在任一 producer 发布新 epoch 前先取得同一个旧 epoch；
2. 所有 rank 对当前 replay 使用同一 generation；
3. graph replay 时 generation 必须在设备侧变化，因为 host tiling/attr 已被冻结；
4. 首次调用要有可证明的初始化与 wraparound 规则。

若由每个 consumer 先读自己的六个 slot 再让 producer 原子 `+1`，仍需保留入口
跨 rank barrier 来消除 read-before-publish race，同时额外增加 GM read 与原子
写；它不再需要 24B reset，却大概率比当前“一次合并 reset”更贵。若使用
double-buffer，则还需要可靠的动态 parity，其初始化和跨 rank 共识仍是同一个
问题。没有新增 device generation ABI 前，不应生产启用。

## 4. 已实现的最小实验候选：省去 self IPC

实验补丁：

`docs/patches/specslo_tp3_mc2_elide_self_ipc_experimental.patch`

补丁没有应用到当前生产源。它只在 direct protocol 中跳过：

- source rank 向相同 destination rank 写 ready/done；
- consumer rank 轮询相同 source rank 的 ready/done。

本地 ready 已由 `AIC WaitEvent(0) + local SyncAll` 保证；本地 read-complete 已由
计算数据依赖、`PipeBarrier<PIPE_ALL>() + local SyncAll` 保证。远端 `source !=
destination` 的 12 个发布和 12 条 poll lane 完全保留。因此 cluster 计数从
`18/18` 降到 `12/12`，但 reset、三轮 rendezvous 和五处 local `SyncAll` 均不变。

这项改动的预期收益应保守看待：self slot 与远端 slot 本来由不同 sub-core 并发
执行，critical path 通常仍由最慢远端决定，所以它更可能是低个位数百分比或无显著
收益，而不是单独抹平 8.73%。它的价值是补丁很小、数值算术完全不变，适合作为
下一次同步协议 A/B 的第一步。

## 5. CPU/源码证据

新增纯 CPU 审计器：

`examples/audit_specslo_tp3_mc2_sync.py`

它对 rank size 2--8 穷举所有 local/remote 事件子集，验证：

```text
完整协议允许 reduce
== local AIC done && 所有 remote ready
== self-IPC-elision 协议允许 reduce

完整协议允许覆盖 source window
== local read done && 所有 remote consumer done
== self-IPC-elision 协议允许覆盖 source window
```

TP3 静态输出为：

- 检查谓词 48 例；
- 等价：true；
- publish DMA：18 -> 12；
- poll lane：18 -> 12；
- reset/rendezvous/SyncAll：不变。

测试文件：

`tests/ut/spec_decode/test_tp3_mc2_sync_protocol.py`

它还检查：

- 生产源码仍保持 `WaitEvent -> reset -> entry rendezvous -> ready -> compute ->
  done` 顺序；
- 实验 patch 对 publisher 和 poller 对称修改，且没有删除 `SyncAll`；
- full epilogue 仍 fail-closed 到 `epsilon=1e-6`；
- `projection_only` 分支仍存在且不进入 RMSNorm。

运行：

```bash
python examples/audit_specslo_tp3_mc2_sync.py
python -m pytest -q tests/ut/spec_decode/test_tp3_mc2_sync_protocol.py
git apply --check docs/patches/specslo_tp3_mc2_elide_self_ipc_experimental.patch
```

当前结果：`12 passed`，patch 可干净应用；未运行 NPU。

## 6. 独立 vendor 的硬门禁

只有在空闲三卡上，且不使用占卡脚本，才应把实验 patch 应用到一次性 worktree 并
构建独立 vendor。至少要同时满足：

1. full epilogue：真实 Qwen3 M100/K3072/N5120，`epsilon=1e-6`，初次执行和
   8 次 changed-input replay 均逐元素 bit-exact；
2. projection-only：三个 rank 的 projection reduction 均逐元素 bit-exact；
3. eager 与 ACLGraph 都连续多轮完成，无 timeout、hang、fallback、新 capture；
4. graph changed-input replay 确认不是 stale output；
5. baseline/candidate 使用相同设备映射做交错 AB/BA，至少 21 样本，比较 median
   和 P95；
6. 若 P95 无稳定改善，撤销候选，不改生产默认。

即使该候选通过，也只能证明 self IPC 优化有效。要让 raw MC2 kernel 本身稳定低于
123.262 us，后续更可能需要优化 replicated peer-read/vector epilogue（例如经单独
数值资格测试的 tile/pipeline）或真正的 rank-3 fused collective，而不是继续删除
必要的同步边。

## 7. 证据来源

- trace 汇总：
  `/root/data/nano-pearl-benchmark-results/20260923-specslo-mc2-m100-four-quadrants/device-trace-v105/summaries/20260923-v105-m100-run1/SUMMARY.md`
- v119 数值修复与资格结果：
  `/root/data/nano-pearl-benchmark-results/20260923-specslo-mc2-epilogue-diagnostics-v113/AVG_FACTOR_V113.md`
- 当前 direct AIV 实现：
  `csrc/mc2/matmul_allreduce_add_rmsnorm/op_kernel/matmul_allreduce_add_rmsnorm_aiv_kernel.h`

trace 来自 v105，但 v119 对该问题的生产修复是 epsilon 的 host tiling/fail-closed
传递；上述 direct TP3 flag/reset/ready/done 同步骨架没有因 epsilon 修复而改变。
因此 trace 可用于解释同步和 raw kernel 开销，但 v119 的正式性能仍必须由 fresh
vendor 的无争用 A/B 得出。
