# ShareGPT 固定 γ=4 接受率实测（2026-10-21，北京时间）

**128条首轮请求的总体草稿 token 接受率为37.7320%（21,308/56,472），平均每次验证接受1.5093个草稿 token。**

模型组合为 Qwen3-0.6B TP1 draft＋Qwen3-32B TP3 target，双方 BF16 greedy（temperature=0）；沿用现有固定串行 γ=4、双 batch / rolling-eager 路径和 MC2 配置，未引入量化或修改冻结引擎。每次验证4个草稿候选，完整线性窗口、bonus token关闭。

使用公共区域现有 ShareGPT_V3_unfiltered_cleaned_split.json，原始文件含94,145条对话。以 Python random.Random(0) 打乱原始行索引，依次选取前128条合格样本：至少两条消息，前两条角色为human/gpt；首条用户问题非空，参考回答至少4个token；默认聊天模板后的提示词为4～1024 token。实际检查前234条抽样索引后选满128条，提示词长度为10～979 token。抽样索引及source_id保存在selection.csv。

本次只生成首轮用户问题的回答。ShareGPT原回答只用于上述长度过滤，不作为模型生成历史或验证答案。使用Qwen3默认聊天模板，未覆盖enable_thinking。每请求最多输出256 token，允许EOS提前结束；实际128/128请求输出长度均达到256 token。故该结果适用于本次抽样、模板和长度条件。

| 指标 | 实测值 |
|---|---:|
| 请求数 | 128 |
| 接受草稿 token | 21,308 |
| 验证草稿 token | 56,472 |
| 总体接受率 | 37.73% |
| 验证次数 | 14,118 |
| 每次验证平均接受数 | 1.5093 |
| 输出 token | 32,768 |
| 输出达到256 token | 128/128 |

接受率先汇总实际接受的草稿前缀token，再除以实际验证的草稿候选数；target纠正token不计入接受量。逐请求简单平均为38.3043%，与总体加权接受率的口径不同。

| 提示词长度（token） | 请求数 | 接受草稿 token | 验证草稿 token | 接受率 |
|---|---:|---:|---:|---:|
| 4-128 | 101 | 16,576 | 45,060 | 36.79% |
| 129-512 | 21 | 3,680 | 8,904 | 41.33% |
| 513-1024 | 6 | 1,052 | 2,508 | 41.95% |

128条请求一次性进入队列；系统活动请求上限64、单次target验证上限32。此次不设置泊松到达或三类SLO，不测Raw/Goodput，也不进行ShareGPT回答质量评分。与MT-Bench两轮生成及GSM8K局部上下文校准的统计范围分别记录，不能把数据集间接受率差异当作优化收益。

数据文件：/data/shared_datasets/strict-inputs/ShareGPT_V3_unfiltered_cleaned_split.json；SHA256：35f0e213ce091ed9b9af2a1f0755e9d39f9ccec34ab281cd4ca60d70f6479ba4。数据已存在于公共区域，本次未重新下载或复制完整数据。上游版本未独立确认，使用该文件哈希固定数据身份。模型复用既有公共权重；代码、运行输出和编译缓存保留在私有目录。

复现入口为run_sharegpt.py、launch.sh、supervise.py。有效配置在run-plan.json，数据选择依据在HOST-preflight.json与selection.csv，依赖代码及哈希在dependencies/。使用既有CANN9.0/Python3.11冻结环境，不更新依赖；先以runtime.sh --environment-only运行run_sharegpt.py --prepare-only，再运行python -B supervise.py。重跑须复制到新的私有目录并调整launch.sh的task_dir，避免覆盖已有日志和回执。

核验已通过：128条源索引及提示词token独立重建；逐请求及分长度汇总计数独立复算；每次验证固定4候选；前后冻结源码和环境检查通过（10,496个冻结文件、15个外部文件、212个软件包）。成功回执显示退出码0、运行源文件未变、四卡空闲。未额外执行完整target自回归等价测试，不授予新的生产性能资格。原树形优化仍暂停。

证据：summary.json、request-counters.csv、acceptance-by-prompt-length.csv、receipt.json、run-plan.json、pre-verifier.json、post-verifier.json、independent-audit.json。完整输入和逐请求输出token留在本机；GitHub仅同步代码、配置、计数和报告。
