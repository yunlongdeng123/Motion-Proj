# 当前研究状态

更新：2026-09-27，v77。`WS-V77-HYBRID-BG-20260927/r2`两场景Hybrid DELETE已完成，按用户stop rule停止该候选：0230仍有灰块/再生车形，0255仍有SUV残影；两例不进入Ω。GPU生成与语义检查已收口，没有定时任务或下一轮自动生成。

## 当前结果

scene_0230/actor22/CAM5/f18–47、scene_0255/actor25/CAM3/f65–94，各30帧。三mask + 原RGB证据 + DiffuEraser官方2-Step，seed42，每例一次。删除区真实证据覆盖0.000426%/0.27238%；车辆再生guard拦截21/30与30/30。60帧像素/来源合同通过，24视频720帧实际解码通过，不能将工程通过当质量通过。

0255围栏mask含助手14点图像提示、镂空候选选择和有界相似变换跟踪，未宣称全自动。内部prior已有残影，与最终车形关联但不证明唯一因果。保护mask不变也不保证所有邻车语义保真。

## 验收与下一步

本地审核入口：`C:/Users/dengyunlong/Documents/Codex/2026-09-26/xia/outputs/v77-hybrid-delete/index.html`。主三栏原视频/旧DriveEditor/新DELETE，附mask、evidence、内部prior及详细注释。报告：[HYBRID_BACKGROUND_RESULTS.md](v77/HYBRID_BACKGROUND_RESULTS.md)。run：`/root/autodl-tmp/runs/worldsim_v77/WS-V77-HYBRID-BG-20260927/r2`，同一失败卡V77-F02已追加。

默认仍保留旧DriveEditor FULL及已知缺陷，原Ω/GLB与所有失败输出未改。0训练、0新Ω前向、0新GLB、0新MOVE。不继续换seed/叠加成熟模型。若下一阶段进入数据路线，先验证真实背景监督可得性；本轮没有启动训练。人工verdict保持null，等待用户验收；历史关机安排不适用于本轮。
