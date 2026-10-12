# DrivingEditor 历史排错工具

本目录保存已经产生证据的检查、控制实验、审阅及报告脚本。用户已停止定时研究；本次只交付代码，未重新执行GPU任务。

- `component_diagnostic/`：分量替换、训练/导出语义、控制续训及真实状态导出。
- `loss_audit/`：公式、深度、初始化、时间合同及V/S单变量实验准备。
- `sgcm_data_audit/`：SGCM、时间行为、输入覆盖与独立审阅。
- `preprocessing_gap_audit/`：M-detector离线适配与隔离编译验证。编译成功不等于native构造或真实scan推理通过。
- `local_report/`：报告、同步视频和本地媒体核查工具。

源文件保留历史绝对路径、配置和receipt绑定。迁移前需据[路径映射](../../../docs/autoresearch/worldsim_v82/drivingeditor_debug_20261012/r1/evidence/git_code_delivery.json)还原依赖并重新审阅计划；不能将记录脚本当作当前状态更新器重复运行。第三方完整checkout、模型和数据在仓库外，差异见[兼容补丁](../../../compatibility/DrivingEditor-v82/)。未执行候选不代表修复已采用。
