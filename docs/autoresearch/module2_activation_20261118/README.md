# 道路资产覆盖、材质与持久化

[结果与组件](report.md) · [组件图](architecture.svg) · [结果摘要](results.json) · [资格入口](../../../scripts/worldsim_v82/module2_activation_state.py)

`run_code/activation/`保存本轮激活、来源与实际渲染检查；`run_code/marking_solver/`保存单12cm成熟求解器控制。`evidence/`保留小型协议与验收记录。完整资产、精度数组、帧和checkpoint保留远端，未进入Git。

新增资格入口为显式使用的CPU组件，默认场景不变。GPU保持开机，未自动启动199帧或后续训练。
