# MT-Bench 固定 γ=4 接受率实测（2026-10-21）

本次采用现有 SpecSLO 固定串行 γ=4、双 batch / rolling-eager 路径；draft 为 Qwen3-0.6B TP1，target 为 Qwen3-32B TP3。计算为 BF16，双方 greedy（temperature=0），MC2 开启；未新增量化。

官方 FastChat MT-Bench 共80道问题、8类各10道，全部测两轮，共160轮。第二轮带入本次第一轮实际生成的回答。使用 Qwen3 默认聊天模板，未覆盖 enable_thinking；每轮最多生成256 token，允许 EOS 提前结束。本轮160/160均达到256 token上限，第二轮历史因此为第一轮受该上限截断的实际输出。该结果适用于这个长度和模板设置，不是完整回答的 MT-Bench 质量评分。

**总体草稿 token 接受率为46.0622%（29,384 / 63,792），共15,948次验证，平均每次接受1.8425个草稿 token。** 所有验证均为4个草稿候选，63,792 = 4 × 15,948。

统计口径：先汇总实际接受的草稿前缀 token 数，再除以实际验证的草稿候选数。target 纠正 token 不计入接受量；不以逐请求接受率的简单平均代替总体接受率。作为区别，逐请求简单平均为47.6175%。

| 轮次 | 请求数 | 接受草稿 token | 验证草稿 token | 接受率 | 每次验证平均接受数 |
|---|---:|---:|---:|---:|---:|
| 第一轮 | 80 | 14,742 | 31,792 | 46.37% | 1.8548 |
| 第二轮 | 80 | 14,642 | 32,000 | 45.76% | 1.8302 |
| 合计 | 160 | 29,384 | 63,792 | 46.06% | 1.8425 |

| 题型（两轮合计） | 请求轮数 | 接受草稿 token | 验证草稿 token | 接受率 |
|---|---:|---:|---:|---:|
| 代码（coding） | 20 | 3,651 | 8,016 | 45.55% |
| 信息提取（extraction） | 20 | 4,023 | 7,248 | 55.50% |
| 人文（humanities） | 20 | 3,269 | 8,620 | 37.92% |
| 数学（math） | 20 | 4,334 | 6,564 | 66.03% |
| 推理（reasoning） | 20 | 3,714 | 7,840 | 47.37% |
| 角色扮演（roleplay） | 20 | 3,306 | 8,908 | 37.11% |
| STEM（stem） | 20 | 3,680 | 7,988 | 46.07% |
| 写作（writing） | 20 | 3,407 | 8,608 | 39.58% |

总共生成40,960个输出 token。每轮80个请求一次性进入队列，系统活动请求上限64、单次 target 验证请求上限32；本次不使用泊松到达，也不测 Raw、Goodput 或 SLO 达成率。

数据源固定为 [FastChat 官方问题文件](https://github.com/lm-sys/FastChat/blob/b494d0c6b4e7935f1764f8439e75da3e66beccc7/fastchat/llm_judge/data/mt_bench/question.jsonl)，文件 SHA256：`119565adbab82227089cefdb44c8d7e2cf04dc0a0ec233634c82e7d4e2a944f7`。数据文件保存在公共区域 `/data/shared_upload/SpecSLO/datasets/MT-Bench/question-b494d0c6b4e7.jsonl`；模型复用现有公共权重，代码、日志及输出保留在私有目录。

复现入口：`run_mtbench.py`、`launch.sh`、`supervise.py`；有效配置保存在 `run-plan.json`，依赖代码及哈希保存在 `dependencies/`。运行时使用既有 CANN 9.0 / Python 3.11 / 冻结 SpecSLO 环境，不更新依赖。实际运行入口为 `python -B supervise.py`；重跑须复制到新的私有输出目录并调整 launch.sh 的 task_dir，避免覆盖既有回执和输出。

首次尝试因新入口未启用标准图资格预热，在服务检查处停止，没有接受率结果；原始代码、日志和失败回执保存在 `attempts/attempt1/`。成功的第二次尝试仅在测试配置补上原生产入口使用的 `precompile_serial_draft_graphs=True`，未修改冻结引擎或绕过资格检查。

核验：成功回执显示退出码0、运行源文件未变、四卡均空闲；前后冻结源码及环境检查均通过（10,496个冻结文件、15个外部文件、212个软件包）；独立 CPU 审计复算全部计数，并重新编码核对第二轮历史及全部提示词 token。没有额外执行完整 target 自回归等价测试，不授予新的树形或生产性能资格。原树形优化仍暂停。

证据：`summary.json`、`acceptance-by-category.csv`、`request-counters.csv`、`receipt.json`、`run-plan.json`、`pre-verifier.json`、`post-verifier.json`、`independent-audit.json`。完整逐请求 token 和输入保留在本机，不上传到 GitHub。
