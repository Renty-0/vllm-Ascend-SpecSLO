# vLLM-HUST 核心补丁

Ascend 插件和 vLLM 核心是独立仓库。此目录保存本项目对应的核心层改动，
便于在另一份 vLLM-HUST 工作树复现，而不把两个项目的源码重复复制进来。

目标工作树：`/root/data/vllm-hust`，分支
`codex/nano-pearl-ascend-migration`。

应用方式（在 vLLM-HUST 工作树执行）：

```bash
git apply --3way /path/to/vllm-Ascend-SpecSLO/patches/vllm-hust/specslo-core.patch
```

补丁基于提交 `933842ab0fdd53ee47dee8148bca1671d7d6da61`，同时包含原工作树的
10 个已修改文件和 4 个原未跟踪文件，因此单个补丁即可恢复完整变更。补丁包含
speculative 配置、V1 scheduler/request/output、rejection sampler、
vocab mapping、tree verification 和 SpecRhythm 元数据及测试。Ascend native
PEARL engine 不依赖这份补丁即可运行；使用 vLLM V1 服务桥接时需要它。
