# WS-V77-EXPLICIT-POC-20260926 / r1 审核入口

**审计更正**：本页为旧结果入口；旧MOVE未经准入，scene_0230车头放反，旧资产失败/关闭结论撤回。请先看[新目标与指令审核页](../actor_command_audit_20260926/review_link.md)及[审计报告](../../../v77/ACTOR_COMMAND_AUDIT.md)。

本地历史交付：`C:\Users\dengyunlong\Documents\Codex\2026-09-26\xia\outputs\v77-explicit-poc\index.html`。目录包含两场景各六段六相机同步采样视频（原视频、ProPainter DELETE、像素背景+GLB原位、像素背景+GLB MOVE、Ω点背景DELETE、Ω点背景MOVE）、两个最终 `actor.glb`、两份 `placement.json` 和逐时刻索引 `review_data.json`。浏览器页展示10个真实采样时刻；每段视频逐一解码验收，GLB在Blender 5.2.2重新导入。

直接看两张同一相机四栏图：[scene_0230 f005 CAM2](scene_0230_f005_cam2.jpg)、[scene_0255 f020 CAM3](scene_0255_f020_cam3.jpg)。四栏从左到右为原视频、ProPainter DELETE、GLB原位、GLB横移2m。前者无GLB的DELETE留深色补景涂抹，后者有车形残影；这两项背景观测保留。原先依据后续合成判定资产车型/身份失败的归因已撤回，详见新审计。这里四栏图保留工程错误的原始证据，不能继续作为资产方法失败依据。人工 `human_verdict` 留空。

运行与完整原始证据：`/root/autodl-tmp/runs/worldsim_v77/WS-V77-EXPLICIT-POC-20260926/r1/`。两场景技术报告及架构见 [EXPLICIT_POC](../../../v77/EXPLICIT_POC.md)，轻量结构化完成数见 [summary.json](summary.json)。
