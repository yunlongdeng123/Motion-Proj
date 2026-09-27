# 当前研究状态

更新：2026-09-27，v77。`WS-V77-HYBRID-BG-20260927/r1` 的 DiffuEraser 最小权重已下载并核对：18个新文件共10.977GB，复用3份prior权重；独立环境、离线导入与4项CPU合同测试通过。**已按用户要求停在GPU推理之前，等待用户开启GPU并继续，不自动恢复。** 零模型实例化、零前向，完整hybrid接入及GPU兼容性尚未验证。

## 执行范围

只保留 scene_0230/actor22、scene_0255/actor25，分别固定CAM5/f18–47、CAM3/f65–94。冻结Ω、原GLB和structured editor不变。本轮不扩scene、不训练、不做MOVE、不建立定时任务。ProPainter仅作为DiffuEraser官方内部prior依赖，不恢复其独立实验。

当前默认 `configs/worldsim_v77/delete_pipeline_current.json` 继续指向 `WS-V77-DELETE-FULL-20260927/r1`；候选未替换默认。此前精确mask直接DriveEditor失败与全部视频完整保留，第三scene保留为历史诊断。

## 准备入口

- [方案、官方依赖与architecture](v77/HYBRID_BACKGROUND_PREP.md)，[唯一登记](autoresearch/worldsim_v77/hybrid_bg_20260927/registration.json)。
- 原始下载日志/清单：`/root/autodl-tmp/runs/worldsim_v77/WS-V77-HYBRID-BG-20260927/r1`。
- 模型：`/root/autodl-tmp/models/worldsim_v77_diffueraser`；环境：`/root/autodl-tmp/envs/worldsim-v77-diffueraser`。
- [文件与离线导入验证](autoresearch/worldsim_v77/hybrid_bg_20260927/preparation_validation.json)、[准备收口](autoresearch/worldsim_v77/hybrid_bg_20260927/preparation_closeout.json)。官方额外形态学、mask视频阈值、prior删除和最终模糊写回需要在接入时适配，不能把下载完成称为完整hybrid已验证。

## 下一步与边界

等待用户GPU。恢复后先做多对象protect mask与事实证据来源审查，再串行跑两例各30帧DiffuEraser，保留prior、原生/最终输出。未获factual邻车轨迹支持的新车标记 `FAIL_ACTOR_REGENERATION` 并阻断Ω；检测器对灰白轮廓会漏检，零检测不能自动准入。新模型尚无视觉结果。

本轮 `failure_ledger_refs: [V77-F02]`；`failure_ledger_delta: none`（只有准备）；`human_verdict: null`。历史关机授权不适用于本轮。
