# WS-V77-EXPLICIT-POC-20260926 / r1 审核入口

本地完整交付：`C:\Users\dengyunlong\Documents\Codex\2026-09-26\xia\outputs\v77-explicit-poc\index.html`。目录包含两场景各六段六相机同步采样视频（原视频、ProPainter DELETE、像素背景+GLB原位、像素背景+GLB MOVE、Ω点背景DELETE、Ω点背景MOVE）、两个最终 `actor.glb`、两份 `placement.json` 和逐时刻索引 `review_data.json`。浏览器页展示10个真实采样时刻；每段视频逐一解码验收，GLB在Blender 5.2.2重新导入。

直接看两张同一相机四栏图：[scene_0230 f005 CAM2](scene_0230_f005_cam2.jpg)、[scene_0255 f020 CAM3](scene_0255_f020_cam3.jpg)。四栏从左到右为原视频、ProPainter DELETE、GLB原位、GLB横移2m。前者留大片深色补景涂抹且资产车型改变；后者仍有灰色车形残影且白色资产与原车不一致。图像是代理观察，人工作出的 `human_verdict` 留空。

运行与完整原始证据：`/root/autodl-tmp/runs/worldsim_v77/WS-V77-EXPLICIT-POC-20260926/r1/`。两场景技术报告及架构见 [EXPLICIT_POC](../../../v77/EXPLICIT_POC.md)，轻量结构化完成数见 [summary.json](summary.json)。
