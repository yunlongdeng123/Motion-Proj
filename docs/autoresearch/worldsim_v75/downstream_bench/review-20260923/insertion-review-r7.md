# OmniDreams 24-case 待审提案：插入例第三轮修订

当前 run：`WS-V75-OMNI-REVIEW-02 / 20260923-proposal-r7-insertion-review`。依据用户三张截图，只替换 `CFB-INSERT-02`、`CFB-INSERT-04`、`CFB-INSERT-05` 的整个场景和供体；其它21例保持r6的参数、原片和审批状态。r6历史及其[完整说明](human-feedback-r6.md)保留。

```text
nuScenes / DriveStudio 10 Hz RGB + 位姿 + 实例轨迹
                           ↓
              DriverQ 场景/运动/相机候选
                           ↓
        供体轨迹 + 偏移 → 道路足迹/碰撞/双框可见性门控
                           ↓
           10秒原片 + 黄色供体框 + 绿色插入框
                           ↓
                   用户人工确认
                           ↓
        OmniDreams factual / counterfactual（待运行）
```

| case | 新场景/目标 | 反事实（供体原车保留） | 参考帧 |
| --- | --- | --- | --- |
| INSERT-02 R3→R4 | scene-0535、CAM_BACK、汽车 #1 | 第0.5秒起新增同尺寸汽车；相对供体沿行驶方向后方4米、左侧3.5米（局部X −4、Y +3.5米） | 原片第2秒，黄绿框均清楚 |
| INSERT-04 R3→R4 | scene-0436、CAM_FRONT、汽车 #0 | 第0.5秒起新增汽车；相对供体前方8米、左侧3.5米（局部X +8、Y +3.5米） | 原片第5秒，黄绿框均清楚 |
| INSERT-05 R3→R4 | scene-0998、CAM_FRONT、黑色汽车 #5 | 第0.5秒起新增汽车；相对供体沿轨迹前方8米（局部X +8、Y 0米） | 原片第2秒；夜间画质需人工审核 |

三个新案例的10秒片段中自车与供体均在运动。参考帧的黄色供体框与绿色计划框均在画内、各自面积至少15,000像素²，且两框重叠率低于0.15；2Hz采样的19个编辑后时刻，计划车身均在nuScenes可行驶区域内、与现有标注对象及自车车身相交0次。此门控不证明语义合理、无遮挡或真正影响规划；参考图只是几何提案，不是生成结果。scene-0998夜间片段与另一速度编辑使用同一实例但不同时间窗，便于比较两类干预，也减少了场景独立性，供用户取舍。

24例全部 `approval.status=pending`、`inference_allowed=false`。原始参考视频每段100帧、10Hz、10秒；新factual/counterfactual视频和评分栏为空；本轮模型调用与AI视频评分均为0。ReSim继续停止。远端报告：`/root/autodl-tmp/runs/worldsim_v75/WS-V75-OMNI-REVIEW-02/20260923-proposal-r7-insertion-review/index.html`；本地：`outputs/cfbench-20260923/omnidreams-case-review/index.html`。
