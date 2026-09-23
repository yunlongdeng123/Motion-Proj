# OmniDreams 24-case 待审提案：第二轮反馈最终候选

Task `WS-V75-OMNI-REVIEW-02`，当前 run `20260923-proposal-r6-human-feedback`。完整的9例截图反馈修订及每例参数见 [r5记录](human-feedback-r5.md)。r6只替换其中一例：`CFB-REMOVE-05-R4-10S` 的日间摩托车在10秒片段中几何入画不足一半，改成 `CFB-REMOVE-05-R5-10S`，scene-0998 前视路口巴士 #1，源帧40–139，第0.5秒起移除。巴士约8秒在画内；自车在它经过期间基本停车，离开后开始前进，这是时间关联，不当作因果证明。夜间画质比日间差，需用户结合原片判断取舍。

```text
nuScenes / DriveStudio 10 Hz 图像、位姿、实例轨迹
                         ↓
           DriverQ 场景/运动/投影查询
                         ↓
       目标重选 + 车身占地/投影/碰撞检查
                         ↓
      24段原片 + 反事实说明 + 黄/绿计划框
                         ↓
                   用户人工确认
                         ↓
       OmniDreams factual / counterfactual（待运行）
```

本版24例均为100帧10Hz、10秒原始参考。自车横移采用俯视绿色车身框（前视相机拍不到自车）；其它目标用校准相机几何框。地图检查不涵盖锥桶、遮挡、车道方向或规划语义，不能替代人工审核。所有 `approval.status=pending`、`inference_allowed=false`；模型调用、生成视频及新评分均为0/空。ReSim保持停止。服务器报告：`/root/autodl-tmp/runs/worldsim_v75/WS-V75-OMNI-REVIEW-02/20260923-proposal-r6-human-feedback/index.html`。本地报告：`outputs/cfbench-20260923/omnidreams-case-review/index.html`。
