# 当前研究状态

更新：2026-09-26（Asia/Singapore）。远端分支 **v77**，工作区 `/root/autodl-tmp/motion_proj_v77`。阶段：**两场景 background + 显式actor 持续迭代；R3背景参考检查已完成；用户确认目标与原位基本可用，当前优先解决DELETE背景模糊。** 仍用 frozen VGGT-Ω + SAM2 + ProPainter + Hunyuan3D-2.1；不训练Ω、不叠加生成/补景模型。完整新证据见 [PIPELINE_R3](v77/PIPELINE_R3.md)，按run查 [EXPERIMENTS](EXPERIMENTS.md)。

## 当前结论与边界

用户最新人工观察（R3审核页）：“目标选中没问题，原位 factual 问题不大，就是删除了以后背景补全就是糊的了”。记录为局部视觉反馈，不扩大为全流程/高保真通过；整体 `human_verdict: null` 保持。

`WS-V77-PIPELINE-R3-20260926/r1` 固定0230/actor22与0255/actor25，GT已知131/181帧×6相机。发现0230原GT框底高度漂移会放过车底射线；将框仅向下闭合到局部地面后，核心168点原本全部“可见”变为0。0255剩9/207视线候选，但无15cm内地面LiDAR支持。两例联合候选为0，背景参考状态为 `no_jointly_supported_core_reference`，停止据此复制所谓已观测背景。

这只是保守地面检查，不是全场景不可见证明或残影唯一因果解释。阴影、mask、照度尚未分离；没有新模型前向、训练、重建或MOVE，画面未宣称改善。原GLB形状失败/关闭路线的过早归因仍撤回，用户本地Blender形状反馈保留，人工 `human_verdict: null`。

前轮放置修正保持：0230车头180°校正、0255方向不变，相机投影已修复。R2空间拒绝保持：0230旧2m碰邻车；0255旧2m/车头前移2m的100/100帧均有静态LiDAR占据。0230前移1m暂无正占据，也不能当已证实自由空间。R2的196帧ProPainter控制仍留主要残影，不再扫长度参数。证据见 [ACTOR_COMMAND_AUDIT](v77/ACTOR_COMMAND_AUDIT.md)、[PIPELINE_R2](v77/PIPELINE_R2.md)。

## 下一次有界工作

按用户最新视觉反馈，优先解决DELETE背景模糊，当前目标、GLB与已修正放置保持固定。先分开审核ProPainter直接RGB输出、实际删除mask及最终合成，检查车身/轮胎/阴影覆盖与接缝；在固定时间窗、同权重下做一次有依据的mask覆盖控制，不再扩帧或扫长度参数。若充分覆盖后仍糊，明确当前补景方案对持续遮挡背景的能力边界，再决定补景模块的下一步，不同时增加多套模型。0255资产参考围栏问题降为次要，不抢占背景补全工作。

当前未批准任何新的MOVE。只有取得足够空间证据再渲染；道路规则/ego/转向时序仍须区别于无框碰撞。若后续需在这两个scene内更换actor，先记录原目标限制与目标选择依据，不悄悄替换分母。

任务heartbeat `v77-pipeline` 已启用，每30分钟检查并继续；先读状态和实际进程，避免重复启动。状态不变不通知，只报实质结果/失败/完成/所需用户行动。用户授权持续自主迭代，人工verdict只由用户填写。

## 资源与交付

R3为CPU诊断，主检查最终一次约10.1秒，12项针对性测试通过；本轮没有GPU作业或待完成控制器。R2控制器已退出。原模型、资产、全部前后控制保留；R3原框输出另存 `raw_box_control/`。

完整run：`/root/autodl-tmp/runs/worldsim_v77/WS-V77-PIPELINE-R3-20260926/r1`。最新本地页：`C:\Users\dengyunlong\Documents\Codex\2026-09-26\xia\outputs\v77-pipeline-r3\index.html`，含新证据图与明确标注来源的已有三栏视频；R2完整10Hz视频另有链接。未作浏览器交互验收。旧POC入口已加最新链接。

VGGT系列仍为重建基座；V7.6按 [V76-F03](research_failures/entries/V76-F03.md) 关闭。更新同一 [V77-F02](research_failures/entries/V77-F02.md)，不新增失败ID。`failure_ledger_refs: [V77-F02]`；`failure_ledger_delta: updated V77-F02`；`human_verdict: null`。
