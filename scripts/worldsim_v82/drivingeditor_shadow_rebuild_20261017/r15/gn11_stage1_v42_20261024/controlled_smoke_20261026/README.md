# 完整片段道路与编辑控制

[实验报告与组件](EXPERIMENT_REPORT.md)。运行脚本是固定campaign的开发smoke，路径与预算写在脚本/远端protocol.json；不是可自动启动正式30K的队列。

[C为什么有效、固定状态反事实与后续设计](C_MECHANISM.md)。

**以用户截图选中的保背景C为准**：edit_smoke_preserve/full199；edit_visibility不是用户选择的C。后者旧分析见[保留对照](VISIBILITY_VARIANT_AUDIT.md)。

报告重建先运行 build_controlled_report.py，再运行 build_selected_c_report.py <本地报告根目录>/c_selected_mechanism；后者固化选中C、默认查看通路和历史误绑定提示。原始强透射结果保持可查。
