# 当前研究状态

更新：2026-09-26（Asia/Singapore）。远端分支 **v77**，工作区 `/root/autodl-tmp/motion_proj_v77`。阶段：**显式资产 POC 的目标/指令/放置审计已完成；撤回前轮“GLB 失败、关闭此路线”的过早归因。** 当前两条旧 MOVE 均未准入，新合法 MOVE 尚未执行，不进入训练。详见 [ACTOR_COMMAND_AUDIT](v77/ACTOR_COMMAND_AUDIT.md) 与 [EXPERIMENTS](EXPERIMENTS.md)。

## 当前证据与范围

`WS-V77-ACTOR-COMMAND-AUDIT-20260926/r1` 复用原两份 GLB，零训练、零新增模型前向。scene_0230 的目标为棕色 actor 22：旧 MOVE 在 50/50 帧与后方 actor 2 的 GT 框相交；纯前移/后移 2 m 分别与 actor 14/2 相交。该资产原位合成还把车头放反 180°。scene_0255 的目标为银白 actor 25：100 帧没有 GT 框相交，但最小扫掠间距仅 8.4 cm；参考相机实现偏移约 6 px，逐轴放置相对保比例缩放额外拉高 15.6%。

上述工程与任务定义问题污染了旧资产评价，不能据旧合成否定 GLB。用户在本地 Blender 认为 GLB 看起来没有太大问题；该反馈作为形状复核意见保留，`human_verdict: null`。现有无 GLB 的 ProPainter DELETE 暗斑/残影仍是独立观测；遮挡、阴影、材质与照度匹配仍未完备。没有反向宣称高保真通过。更新同一 [V77-F02](research_failures/entries/V77-F02.md)，不新增失败 ID。

## 下一步与产物

审核页增加目标轮廓和原车裁剪、GT track/尺寸、旧指令俯视扫掠图、同 GLB 朝向/缩放控制及两场景各 10 时刻三栏视频（原始/修正原位/DELETE）。scene_0230 的相机切换明确标注。旧 MOVE 从主审核对比撤下，历史页与完整源运行保留；渲染器默认只运行修正原位，旧命令必须显式请求诊断且命名为未验证。

后续先选清晰目标与可用空间，再查邻车、静态环境、可行驶区和必要的运动学约束；只有指令准入后才做 MOVE 质量验收。几何扫描中的“无框碰撞候选”不代表已合法。当前不扩大分母、不训练、不增加第二套生成/补景网络。

新审计运行：`/root/autodl-tmp/runs/worldsim_v77/WS-V77-ACTOR-COMMAND-AUDIT-20260926/r1`。原 POC：`/root/autodl-tmp/runs/worldsim_v77/WS-V77-EXPLICIT-POC-20260926/r1`。本地交付：`C:\Users\dengyunlong\Documents\Codex\2026-09-26\xia\outputs\v77-actor-audit\index.html`。本轮只有 CPU 几何/Blender 渲染与视频编码，无新 GPU 工作。

VGGT 系列仍为 v77 重建基座；V7.6 HUGSIM/VAD-GS 主线按 [V76-F03](research_failures/entries/V76-F03.md) 保持关闭。`failure_ledger_refs: [V77-F02]`；`failure_ledger_delta: updated V77-F02`；`human_verdict: null`。
