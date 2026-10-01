# r8 数据覆盖控制审核入口

[模型五列逐帧审核](C:/Users/dengyunlong/Documents/Codex/2026-09-26/xia/outputs/v77-target-protected-r8/index.html) · [68候选数据四列审核](C:/Users/dengyunlong/Documents/Codex/2026-09-26/xia/outputs/v77-target-protected-r8/data_review.html)

实际392视频/3920帧解码、3470引用JPG、HTML脚本语法通过；未实际验证浏览器同步播放。人工逐帧0/1/2为空，可导出CSV/JSON；助手只看固定f5，数据技术分与模型效果粗分独立。

完整run：`/root/autodl-tmp/runs/worldsim_v77/WS-V77-TARGET-PROTECTED-20260929/r8`。`evaluation/`保存45窗原生PNG/局部写回；`review/`保存HTML/视频/十帧图；`data_review/`保存68候选全帧；`assets/`为已有DEV形状投影模板，非新生成器。

新数据权重：`training/attention_step_0160.safetensors`，同目录有40/80/120/160快照及优化器。有效r7：同task的`r7/encoder_fixed_lowres/training/attention_step_0160.safetensors`。原模型：`/root/autodl-tmp/external/worldsim_v75_downstream_bench/DriveEditor/checkpoints/model.safetensors`。新训练从原权重+官方106encoder恢复开始，不延续r7。

`dataset_catalog.json`冻结50train/25scene、7val/4scene；`evaluation_plan.json`冻结15case/三臂。原模型三个旧真实窗口仅在RGB/H/窗口/seed/步数/previous条件相同后复用，不复用r6错误训练权重。所有真实例曝光DEV、无去车GT；final未使用。

r8相对有效r7洞内scene等权MAE仅−1.05%，保护车MAE+4.78%；固定f5未见明确真实迁移收益。本轮不推广r8，也不据此否定Target+Protected路线或证明模块错误。见[报告与组件图](../../../../v77/TARGET_PROTECTED_DATA_CONTROL_R8.md)、[收口](closeout.json)、[输入分布](input_distribution.json)。轻量Git记录不是完整备份。
