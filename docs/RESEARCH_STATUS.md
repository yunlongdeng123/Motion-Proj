# 当前研究状态

2026-11-05，Scene008 已从原生训练第 1 步切换为双实例监督，完成真实 GPU smoke、1K 继续条件、原生 10K 与 49 帧配对。[训练组件、配对评价与检查点](autoresearch/scene008_native_ownership_20261105/README.md)

在固定实测 a6 射线上，静态前方遮挡从旧双动态 0.1653 变为 0.0000。删除后可见 a6 的 RGB MAE 从 0.1906 变为 0.0187。两车串车贡献、实测深度和道路质量见同域对照。

道路保留质量与删除显露区仍有退化，人工验收尚未登记。[上一轮 10K 后接管与 PVD 对照](autoresearch/scene008_instances_20261105/README.md) 单独保存。AutoDL 电源由用户控制。
