# v462：已验证的自适应M80/M120完整版本

单次完整463请求与三路独立own-AR均通过；全局上限64，实际峰值48。Raw681.6589、Goodput460.8191，313/463达标。相比此前同上限最佳Goodput提高4.93%，最终目标仍未达到。

真实运行代码保存在`../snapshot/root/SpecSLO/budget-calibration-20261015/`中的`tree-adaptive-own-history-full-v460/`、`tree-adaptive-retained-CANN-path-full-v461/`和`tree-adaptive-qualified-wait-margin-full-v462/`及其原依赖目录。新增独立检查和局部成本代码在`tree-adaptive-own-geometry-node-math-v459/`。

源码逐字节保留。`SOURCE_MANIFEST.json`中的omitted_evidence明确列出未上传的历史数据与源码压缩档；仓库克隆不是原目录全量备份，不能直接继承旧NN许可或绕过原guard。恢复时按原条件重新生成fixture和证据并封存。模型与数据集、原始张量、图缓存和完整环境均未上传。证据摘要和原始收据的无损压缩副本位于evidence目录。
