# SpecSLO 最终 B_roof 实测报告（2026-09-10）

本报告只描述论文 §5.3 的 target-forward 容量实验及其线上消费验证，不是
RPS/Goodput 性能报告。§5.1/5.2 文本生成功能已先完成并冻结；普通 PEARL、历史手工
B1/B4 和 gamma sweep 均未用于本表。

## 实验口径

- 模型/并行：Qwen3-0.6B draft TP1 + Qwen3-32B target TP3；
- 设备/模式：Ascend 910B2、CANN 9.0.0、ACLGraph；
- 树：width=2、depth=2；
- active/verify rows：8/4、16/8、32/16、64/32；
- context：512、1024、1536、2048、2560、3072；
- 每 shape：3 次不计时 warmup，10 次计时 replay，取三个 target rank 每轮最大值的 P95；
- AR 对照：完整 active batch 的标准单 token paged-attention decode；
- 合格条件：tree verify P95 不超过 AR P95 的 `1.10`。10% epsilon 是本项目显式
  选择，论文没有公布具体数值；
- 每个总预算枚举全部规范 candidate-count histogram。通过预算必须全部形状通过，
  第一个失败预算可由一个实测反例确定。

采集器按 active batch 使用 fresh 四卡进程；进程内只分配一次最大 context cache，
按 context 从大到小测量，Graph 常驻复用。原因是 CANN 自定义 FIA graph-task 在反复
reset/capture 后出现过 TP3 stream 停滞，这种生命周期也不符合生产服务的常驻图行为。

## B_roof 结果

B 是当前 logical verify slot 内的 candidate token 总预算，不是 per-request gamma。
`gamma`/单请求 `max_gamma` 仅限制某个请求的树候选数；调度器先查本表
得到固定总 B，再根据 SLO 紧迫度与接受收益在请求间分配不同的
候选数；任何 gamma 参数都不能扩大 B。

| active / verify rows | 512 | 1024 | 1536 | 2048 | 2560 | 3072 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 8 / 4 | 5 | 5 | 5 | 5 | 6 | 5 |
| 16 / 8 | 9 | 9 | 9 | 9 | 9 | 10 |
| 32 / 16 | 17 | 17 | 17 | 17 | 17 | 17 |
| 64 / 32 | 33 | 33 | 33 | 33 | 33 | 33 |

schema v3 聚合器接受了 24 个 evidence key，没有 zero-budget key。profile 绑定实际
AR backend `paged_attention_v1`、verify backends
`fused_infer_attention_causal_v1`/`fused_infer_attention_tree_v1`，并绑定四个核心
执行源文件 SHA256；执行代码变化后旧表会被拒绝。

## 线上消费验证

8 个同时到达请求的真实 Graph 回归命中 `8:1`，查表 B=5。Individual Budget
Shaping 根据当轮状态分配为 `[2,1,1,1]`，实际 candidate sum=5、物理 query=9，
TP3 三个 target rank 均命中 causal FIA ACLGraph。请求完成造成 active batch 降到
未采样的 7…1 后，没有用 gamma 伪造 B，而是执行 17 轮 target-only fallback；target
和 draft KV 同步，流式 token、finish 事件和各 rank 执行序列均通过。

含 sibling 的 `[1,3]` FULL-tree 路径也单独通过真实 TP3：Graph/eager 同后端 hidden
和 logits 零误差，capture/replay delta=1/2。跨 dense/PA 与 FIA 的 raw BF16 logits
差异仍只作为跨后端诊断，不伪装成严格 allclose。

## 证据与校验和

原始文件保存在
`/root/data/nano-pearl-benchmark-results/20260910-specslo-functional-freeze/`：

| 文件 | SHA256 |
| --- | --- |
| `qwen3-tp3-graph-Broof-resident-b8-raw.json` | `d5dc03eebb5144e677392f00a819a52ed1e2671b78e50421ebd5a359033d21ed` |
| `qwen3-tp3-graph-Broof-resident-b16-raw.json` | `5d9edd23b828f0921b60cd95e06fcc0ea83c6591aa1b3710d74235cf4660bcb2` |
| `qwen3-tp3-graph-Broof-resident-b32-raw.json` | `149ac241e363c8e24888a35e086f99c83c2cd9f35512b09f2401603d55ac775f` |
| `qwen3-tp3-graph-Broof-resident-b64-raw.json` | `5210599ea5a5bb61c09cab556d025a16da299d33bd472044a2025c6e5396dd37` |
| `qwen3-tp3-graph-Broof-schema3-resident-eps10.json` | `e557a0e6393055d19983c4b4b1fb202198ade9c406cc9111cfde1b81952271ff` |
| `qwen3-tree-graph-e2e-after-ordering-fix.json` | `3d3229a33be5cde825f66dc2e05fe647e63e0fd6a0e57eee6cf5ca93ea5c6844` |
| `qwen3-profile-online-graph-b8-tail-fallback.json` | `90a91705adade308e4afe1dda3568ad32af74ae3c7133566288c20670a942690` |

当前 spec decode CPU 全量回归为 `880 passed, 13 skipped`。下一阶段才能使用本表
开展 RPS=2/4、batch=8/16/32/64、三类 SLO 6:2:2 的 TPOT/Goodput 对照；本报告不
声称 1.3× 已达成。
