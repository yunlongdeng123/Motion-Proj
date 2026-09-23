# OmniDreams：24例10秒提案，先审再推理

Task `WS-V75-OMNI-REVIEW-02`，run `20260923-proposal-r2-10s`。仅原始视频与干预计划，不是新模型结果。`failure_ledger_refs:[]`，`failure_ledger_delta:none`。

```text
nuScenes原始RGB / 标定 / 轨迹
             ↓
24个10秒候选 + 相机/目标 + 单变量干预
             ↓
       用户确认新版（待完成）
             ↓
同seed factual / counterfactual（留空）
             ↓
         用户人工review（留空）
```

用户认为前四例总体无大问题，要求延长10秒并追加余下20例。24例由速度、横移、移除、插入各6例组成；保留父case、相机、事件和目标，不按生成质量重选。第2例沿用上一提案：从静止scene-0242换至运动scene-0230，与第1例共源；本次为不外推1.5倍速轨迹，前缀改为1.4秒，源事件65不动。其余23例在0.5秒干预。横移仍用1.8秒平滑过渡，之后保持局部偏移，不把编辑动作拉长到整段10秒。

原始视频每段100帧10Hz，容器时长严格10秒，帧时刻0–9.9秒；不减速、不循环、不插帧。计划模型原生输出301帧30Hz，只展示前300帧（10秒），末帧裁去，不填充。生成接口扩展和推理尚未执行，用户未批准的新配置不能入队。

HTML包含24行case导航、逐case一句反事实、相机和目标身份、原始/factual/CF三列；后两列及评价留空。非ego目标有原始参考标框：黄色是被编辑目标/供体，绿色是计划位置几何投影，不是生成图。ego为搭载相机的采集车，CAM_BACK是后视，不能凭画面直接称倒车。

仅CPU检查：24例均有完整所需目标轨迹，不做外推。REMOVE-04/05事实目标仅21/20帧投影在画内，10秒视频不等于10秒有效目标观察；静止ego横向重定位和旧短窗已有可行驶区域问题在对应case展示，不自动替换偏移，不声称全道路/碰撞审核通过。旧版人工意见保存在[human-review-original-four.json](human-review-original-four.json)，不转换成数值评分，也不对新版做AI视频/静帧评分。

新版候选见[candidates-10s.json](candidates-10s.json)。旧四例6秒[candidates.json](candidates.json)及原run保留；没有覆盖旧实验结果。

远端报告：`/root/autodl-tmp/runs/worldsim_v75/WS-V75-OMNI-REVIEW-02/20260923-proposal-r2-10s/index.html`。本地交付：`outputs/cfbench-20260923/omnidreams-case-review/index.html`；旧6秒本地报告保留到`omnidreams-case-review-6s/`。用户确认前不跑推理，ReSim保持停止。当前执行状态只见[RESEARCH_STATUS](../../../../RESEARCH_STATUS.md)。

验证：`CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=3 /root/autodl-tmp/envs/motionproj/bin/python -m pytest -q tests/test_cfbench_full.py`，5项通过，包括100帧10秒参考、时间重参数越界不外推、横移1.8秒后保持、移除/插入事件边界。全部导出视频在导出时CPU全帧解码；仅工程校验，不是AI质量核对。
