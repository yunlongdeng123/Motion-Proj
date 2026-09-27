# 当前研究状态

更新：2026-09-27，v77。按用户最新方案执行 `WS-V77-DELETE-REPAIR-20260927/r1`：精确SAM2实例mask + 多视角/时序真实RGB证据优先 + DriveEditor residual补景 + 车辆再生guard。保留旧资产、旧两条非法MOVE与全部对照；零训练，无ProPainter、语言/RL、新生成器、定时任务。

## 当前范围

scene_0230/actor22/CAM5 f18–47、scene_0255/actor25/CAM3 f65–94、第三scene official_000/actor12/CAM0 f0–29；各3秒10Hz，固定旧失败后段及第三目标，不按新生成结果选片。第三目标为右侧黑色MPV，避开被近裁面切断的truck14。三个已曝光开发scene，不称独立测试。

当前是三场景的背景入口强控制：旧矩形对照保留；A精确mask、B证据+残余洞，两组模型/seed/窗口相同。六相机整段原始Ω缓存搜索证据；GT相机、框及LiDAR辅助准入，不称RGB-only。GT框只prompt/限制SAM范围，删除mask不是框。只膨胀3px。

## 执行与边界

登记和中间产物位于 `/root/autodl-tmp/runs/worldsim_v77/WS-V77-DELETE-REPAIR-20260927/r1`；一次有界运行，检查实际进程避免重复启动。旧完整工程已结束。源图通过冻结Ω深度/GT相机warp；来源须通过贴地对象包络遮挡排除、20cm LiDAR支持、两时刻深度/RGB一致性，不得复制旧生成背景。没有证据覆盖的像素留给生成模型，不伪称真实观测。

先验证背景；检测或视觉发现再生车则阻断进入Ω，不用新几何渲染掩盖。guard须跑原图正控制和已知旧幻觉控制，零检出不等于通过。原GLB保持，Ω自身的碎裂/错误遮挡结论不因入口修复自动撤销。人工verdict保持null。

## 下一步

完成两组推理、逐帧guard与可复核视频HTML，更新同一V77-F02后提交推送。有明确反例则停止该结果的后续Ω，不无限换seed/阈值。仅对有证据的准入结果推进下游。

`failure_ledger_refs: [V77-F02]`；`human_verdict: null`。
