# 当前研究状态

2026-10-01，v77；`WS-V77-TARGET-PROTECTED-20260929/r8`已准入50train/25个receiver场景，单scene最多3例；24背景、23单保护、3密集（密集训练仅scene-0240）。7合成评测/4个隔离场景，8个已曝光真实DELETE开发窗口。固定原/r7/新数据三臂，final不用。

68候选全10帧RGB/覆盖/metric放置/depth/连续性复核，gpt-6-sol xhigh独立固定帧61pass/5uncertain/2reject；50train和7val均技术通过且AI2，人工空。M053/M064视觉树带/人行道拒绝，即使三扫描静态占据无正证据也不放行。5待定、4合格备用完整保留。

数据改为已有DEV sedan/SUV显式mesh在真实camera/metric SE3下投影，完整synthetic RGB擦除；Y永远真实。train/val共享两种形状模板，只隔离真实世界来源，不声称未见形状/跨域。Boston白天数据，真实DELETE无去车GT。原图、全部PNG/视频与旧规则保留。

已完成160步，严格恢复0 missing/0 unexpected、80有限非零梯度、峰值10.48GiB；训练配方逐字段与r7一致。同r7有效初始化、80空间attention、320×576/160步、原loss/seed6201、从原DriveEditor开始；不延续r7、不扩大模块/加权/加步数。真实原/r7对照已完成，正在完成7合成+8真实的三臂采样和HTML；不以latent loss下降认证效果。无电源/自动化操作。failure_ledger_refs: [V77-F02]；failure_ledger_delta: updated V77-F02（数据准入，模型结论待定）。
