# 当前研究状态

更新：2026-11-22（Asia/Singapore）。

`A20-SCENE-RESIDUAL-CLEANUP-20261122`：[现状报告](autoresearch/a20_scene_residual_cleanup_20261122/report.md)、[同一HTML历史对照](autoresearch/a20_joint_reconstruction_20261122/index.html)、[恢复与选中sidecar](autoresearch/a20_scene_residual_cleanup_20261122/restore_index.json)。本轮封存为partial_cleanup_no_final_acceptance，保留原a20与默认项，14个实际组全部保留。本任务优化器更新0；120步透明度控制CANCELLED_NOT_STARTED，实际0步。主要残影来自未归属静态背景，少量来自其他动态车辆的外露杂散，当前a20主体保持。硬删暗层暴露亮紫层，再删亮紫层又暴露新黑拖影。灰条处道路Alpha约0.09，教师来源支撑不足；保护点不能继续全删。保守cleanup_candidate为959行sidecar：亮红条基本去除，冻结道路RGB参考诊断两帧小改善；1468/1500/1754/2006仅为失败或部分改善消融，r5未通过最终验收，未升级默认。a20/道路字节、真实可见误差及保留车保护保持，sidecar重载差0。当前部分完成、场景未通过；human_verdict/problem_solved=null，failure_ledger_refs=[V82-F06]，failure_ledger_delta=updated:V82-F06。GPU按用户要求保持开机。

免责声明：冻结道路float RGB和诊断圈用于定位与相对比较，不是地面真值；黑像素下降不能单独判定质量通过。原内部灰模失败保留历史，不作为本轮场景清理的完美硬门。
