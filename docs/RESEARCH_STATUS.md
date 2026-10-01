# 当前研究状态

2026-10-01，v77；`WS-V77-TARGET-PROTECTED-20260929/r8`已准入50train/25个receiver场景，单scene最多3例；24背景、23单保护、3密集（密集训练仅scene-0240）。7合成评测/4个隔离场景，8个已曝光真实DELETE开发窗口。固定原/r7/新数据三臂，final不用。

68候选全10帧RGB/覆盖/metric放置/depth/连续性复核，gpt-6-sol xhigh独立固定帧61pass/5uncertain/2reject；50train和7val均技术通过且AI2，人工空。M053/M064视觉树带/人行道拒绝，即使三扫描静态占据无正证据也不放行。5待定、4合格备用完整保留。

数据改为已有DEV sedan/SUV显式mesh在真实camera/metric SE3下投影，完整synthetic RGB擦除；Y永远真实。train/val共享两种形状模板，只隔离真实世界来源，不声称未见形状/跨域。Boston白天数据，真实DELETE无去车GT。原图、全部PNG/视频与旧规则保留。

160步与两套三臂45窗全部完成，严格0 missing/0 unexpected、80有限非零梯度、峰值10.48GiB；配方逐字段与r7一致。原/r7/r8合成scene等权洞内MAE为0.08703/0.05697/0.05637，r8对r7仅−1.05%、3/7例改善；保护车MAE为0.13925/0.09712/0.10176，r8对r7+4.78%、仅1/4改善。8真实DEV固定f5未见明确r8迁移增益，A022再生、A013/A048恢复损伤保留。不是视频通过率，human空。

本轮r8不推广，原模型默认和有效r7均保留。下一控制先对齐训练hole与真实生成mask的形状/面积/触边：训练平均1.62%、车辆轮廓、无触边；真实5.60%、矩形、48.75%帧触边。输入差异未证因果，不能据此说模块错误；仍保持同80张量160步和原loss，随后才单独保护加权，再考虑扩大模块。当前不加步数/新训练，不用final，无电源/自动化操作。交付模型五列+68候选数据四列逐帧HTML，视频实际解码核验见[报告](v77/TARGET_PROTECTED_DATA_CONTROL_R8.md)。failure_ledger_refs: [V77-F02]；failure_ledger_delta: updated V77-F02（数据覆盖后无稳定增量收益及输入分布假设）。
