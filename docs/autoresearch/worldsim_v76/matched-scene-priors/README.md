# 同场景先验与身份门禁证据

方法、architecture、分母与适配边界见[报告](../../../v76/MATCHED_SCENE_PRIORS.md)；当前执行见[状态](../../../RESEARCH_STATUS.md)。

- `dsine_encoding_gate.json` / `dsine_encoding_comparison.png`：固定官方帧20六相机编码控制。
- `scene_0230_normal_report.json` / `scene_0255_normal_report.json`：各366张原分辨率法线生成记录。
- `scene_*_sam_gate.json`：每场景固定6视图的SAM提示和区域数；是未过身份门禁的诊断，不能当已批准训练输入。
- `matched_priors_audit.json`：文件数量快照、法线抽查、保留SAM的uint8 ID合同；身份字段明确failed。背景数量是取证时快照，实时数量另查服务器。
- `sam_occlusion_audit.json` / `sam_identity_gate.png`：17个原始SAM响应和4个已确认身份反例；原始mask完整保留于AutoDL `/root/autodl-tmp/data/v76_vadgs/sam_occlusion_audit`。
- `scene_*_DYNAMIC_IDENTITY_BLOCKED.json`：两场景动态入口保护及保留路径。
- `visible_detection_protocol.json` / `visible_detection_result.json`：沿用V7.5阈值的可见二维检测一对一关联，固定6正/4遮挡开发控制。
- `visible_sam_result.json` / `visible_sam_fixed_controls.png`：CPU重新提示SAM的结果与全部十例对照；隔离sidecar，不代表完整训练输入准入。
- `matched_priors_audit_0309.json`：03:09文件分母，背景366/268；scene-0255剩余由CPU续做。保留先前审计快照，未覆盖历史数量。
- Python/Shell：精确生成与审计脚本。`run_sam_gate.sh` 只记录已执行的失败适配，当前guard会拒绝重跑。背景单独入口 `run_matched_backgrounds.sh` 可恢复未完成输出，进入主训练时退出。

完整模型权重和逐帧PNG不提交Git，保留于AutoDL持久数据盘；这份轻量归档不是完整输入备份。`failure_ledger_refs: [V76-F01]`，`failure_ledger_delta: V76-F02`。
