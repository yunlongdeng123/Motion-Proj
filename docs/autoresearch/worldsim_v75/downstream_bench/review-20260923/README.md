# OmniDreams：先审case，再推理

Task `WS-V75-OMNI-REVIEW-02`，run `20260923-proposal-r1`。开发性候选，不是新模型结果。`failure_ledger_refs:[]`，`failure_ledger_delta:none`。

```text
nuScenes原始RGB / 标定 / 轨迹
             ↓
4个6秒候选 + 对象/相机 + 单变量干预提案
             ↓
        用户人工确认（待完成）
             ↓
同seed factual / counterfactual（尚未运行）
             ↓
         用户人工review（留空）
```

四例依次为：scene-0230前视ego 0.5×减速；同一运动片段ego 1.5×加速；scene-0255前视ego 0.5×减速；scene-0230后视汽车#8 0.5×减速。ID均追加`-R2-6S`，保留父case。第2例从静止的scene-0242换到scene-0230，明确与第1例共源，不作为独立场景。其余不变更目标/事件/倍率。全部从0.5秒开始干预，窗口0–6秒；计划生成181帧30Hz，无尾部填充。参考61帧10Hz，不减速、不循环、不合成未来。

仅CPU检查轨迹长度、完整性及目标几何可见性：四例计划终点差约11.98/11.77/12.44/19.30米；第4例事实与计划CF的目标投影均61/61帧在画内，不保证无遮挡，不声称已完成全道路/碰撞审核。相机映射来自DriveStudio预处理定义，camera5为CAM_BACK；ego是搭载相机的采集车，不是目标汽车#8。

用户对旧版四例的定性意见保存在[human-review-original-four.json](human-review-original-four.json)，不转换成数值评分；第2例口述“cycle”所指分支未确认。原AI静帧分保留为历史，不新增或冒充人工判断。只校验视频文件/帧数/时间戳，不由AI复核生成质量。

远端HTML与视频：`/root/autodl-tmp/runs/worldsim_v75/WS-V75-OMNI-REVIEW-02/20260923-proposal-r1/index.html`。本地交付：`outputs/cfbench-20260923/omnidreams-case-review/index.html`。原始/factual/CF三列，后两列与新版评价占位，待用户批准。工具脚本`prepare_omnidreams_review_candidates.py`只生成提案和原始参考，不调用模型；用户未批准不得转换为推理队列。

用户要求停止ReSim：已停止队列与子进程及等候导出进程，保留4个完整pair、共9个完成分支及中断日志，未删除产物。停止记录在上一run的`resim/user-stop.json`和`resim/queue-result.json`。当前执行状态只见[RESEARCH_STATUS](../../../../RESEARCH_STATUS.md)。
