> 2026-09-13：本目录的大表/生成资产已按清单移出 Git；需要逐例数据时先查[完整归档与恢复方法](../../../archive/2026-09/worldsim_v74_closeout_20260913/README.md)。下方历史路径保留作定位，当前目录未必包含全部旧文件。

# P1.6 结果：按问题渐进读取

先读[报告](../../../archive/2026-09/v74-0920/WORLDSIM_V7_4_H2_P16_DIAGNOSTIC_REPORT.md)或[F10](../../../research_failures/entries/V74-H2-F10.md)，不默认读取全部JSON。

| 问题 | 读取资产 / 筛选键 |
|---|---|
| 本轮结论与关闭范围 | decision.json |
| 四类最终联合指标 | control_summary.json → mode / model / family |
| 各任务全部指标、轨迹 | control_rows.json → mode / model / case；按需筛选 |
| 法向/切向替换及教师物理误差 | interpretation_summary.json → model / family |
| 累计漂移、右删失 | recovery_summary.json；完整drifts.json在runs |
| C2如何恢复 | recovery_detail_summary.json → recovery_detail.json |
| 普通控制引入什么错误 | residual_summary.json → residual_rays.json；closed_new_transitions.json |
| 同一最终表面加FIT信息 | same_parent_fit_rows.json |
| 原70条是否修好 | paired_failures.json；须同时查看新错误 |
| 启动/预算/耗时 | launch.json（历史启动）、manifest.json（启动快照）、completion.json（计算完成） |

模型原始=raw；可用控制=closed_build；closed_fit_diagnostic和same_parent_fit只能做诊断。8步终点尚未恢复=右删失。完整逐射线rows、所有表面、求解历史在runs；不将本目录长JSON粘贴进failure入口。日志无哈希/校验和/指纹。

正式结果为r2；launch.json记录原r1启动。batch_correction.json说明为何匹配原8+4前向设置，r1原资产保留在runs及历史提交132500fa。
