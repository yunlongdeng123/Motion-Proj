# v77九场景与编辑工程审核入口

- task/run：`WS-V77-EXPAND-EDIT-20260928 / r1–r7`。
- 本地完整HTML：`C:/Users/dengyunlong/Documents/Codex/2026-09-26/xia/outputs/v77-expansion/index.html`。
- 远端全部输入/中间层/PNG/视频/日志：`/root/autodl-tmp/runs/worldsim_v77/WS-V77-EXPAND-EDIT-20260928`。
- [技术报告与组件图](../../../v77/EXPANSION_EDIT_CHECKPOINT.md)、[收口](closeout.json)、[助手观察](observations.json)、[产物合同](validation.json)、[HTML验证](html_validation.json)。

| run | 已执行内容 | 配置/证据 |
|---|---|---|
| r1 | 新六例固定评测：五例生成，一例mask停止 | [登记](r1/registration.json)、[输入准入](r1/mask_admission.json)、[生成状态](r1/drive_state.json)、[隐藏邻车核对](r1/neighbor_audit.json) |
| r2 | 既有r30首10帧→冻结Ω | [登记](r2/registration.json)、[状态](r2/state.json) |
| r3 | 旧0255资产INSERT首10帧 | [登记](r3/registration.json)、[位置](r3/selected_command.json)、[渲染](r3/actor_layers/render_summary.json) |
| r4 | 真实RGB监督候选，未训练 | [登记](r4/registration.json)、[候选清单](r4/manifest.json)、[隔离/输入核对](r4/data_audit.json) |
| r5 | 目标12 shape/PBR、朝向与原位 | [登记](r5/registration.json)、[mesh放置](r5/actor_layers/placement_checks.json)、[接地测量](r5/ground_audit.json) |
| r6 | 0.5m MOVE候选筛选；未执行MOVE | [筛选](r6/registration_results.json) |
| r7 | 仅固定下移14.61cm的原位对照 | [登记](r7/registration.json)、[单变量核对](r7/validation.json) |

脚本保存于`scripts/worldsim_v77/expansion_*.py`。这些是绑定本次唯一task/run的实验脚本，不是可覆盖旧结果的生产入口；新实验需新登记和输出路径。远端GPU模块分别使用既有SAM2、DriveEditor、worldsim-v77和Hunyuan环境；不要在默认motionproj环境一键执行全部。局部Blender脚本使用本机5.2.2、CPU4线程，保留所有原始RGBA/深度/八视角和0/180控制。`r5/orientation180/`及`r7/`为分离对照，未覆盖r5。

本地页面重建：在已有媒体和观察JSON的stage上调用`build_expansion_html.py --stage <stage> --output <output>`；核对调用`validate_expansion_html.py --output <output>`。52视频1200帧解码、47图片验证通过；不是52个独立实验。静态引用、SVG、JS语法已检；浏览器交互未测试。

固定的本轮留出只表示本轮没有结果后调参。未来若诊断425/382/756，它们转为开发资源，不能继续充当独立确认集。所有human_verdict仍空；旧scene_0255 r18>r15是用户已有评价，未被本轮资产控制改写。正式训练0步；未做反传显存测试。
