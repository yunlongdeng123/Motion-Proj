# 当前研究状态

更新：2026-09-26（Asia/Singapore）。远端分支 **v77**，工作区 `/root/autodl-tmp/motion_proj_v77`。阶段：**按用户最新授权持续迭代两场景 background + 显式actor pipeline；R2静态障碍与196帧补景控制已完成，下一项查背景观测覆盖与参考图遮挡。** 仍不训练Ω，不增加多套生成/补景模型。完整结果见 [PIPELINE_R2](v77/PIPELINE_R2.md)，按run查 [EXPERIMENTS](EXPERIMENTS.md)。

## 已确认结果与边界

`WS-V77-PIPELINE-R2-20260926/r1` 继续 actor22/25。scene_0255 的旧2m命令和车头前移2m候选虽无其他GT框相交，但100/100帧的新增扫掠区含重复静态LiDAR占据；参考视图投影位于原车前方围栏/草地边界附近，均未准入。scene_0230 前移1m未检测到高于车底15cm的重复静态点，仍不能将稀疏空白当自由空间；旧2m与邻车碰撞结论保留。

只对0230/CAM2、0255/CAM3做一次长上下文强控制：原评价窗口50/100帧RGB/mask冻结，输入补到196帧，原ProPainter权重、subvideo_length由40变200。两路均完成，主要暗斑和车形残影仍在，不再机械扫同一参数。新增0255尾段mask，零训练、无新Ω/Hunyuan推理；六段10Hz对照视频在删除区域外均保持源RGB不变。详见报告的分母与官方100帧图像传播分块边界。

前轮 [ACTOR_COMMAND_AUDIT](v77/ACTOR_COMMAND_AUDIT.md) 已纠正车头/相机与缩放归因。原“GLB失败、关闭路线”结论仍撤回；用户认为本地Blender资产形状可用，该反馈保留，`human_verdict: null`。R2只补充 [V77-F02](research_failures/entries/V77-F02.md)，没有新失败ID或模型家族否定。

## 持续执行与下一步

当前任务heartbeat `v77-pipeline` 已启用，每30分钟继续自主工作；状态不变不通知，只报告实质结果、失败、完成或需要用户行动。先读本状态和实际进程，避免重复运行；每轮是有界对照，按规则备份、验证、提交push，用户人工verdict保持空白。

下一项优先做“背景哪些面真实可见”的覆盖盘点，区分可传播证据与未观测区域；同时把已发现的0255参考crop围栏污染列入输入准入检查。冻结现有GLB几何、校正朝向及模型权重，检查遮挡/贴图/照度原因。只有获得足够空间证据后才渲染新的MOVE，不把另一个相机方向或无邻车框碰撞当合法命令。若当前目标无法给出合理MOVE，记录该目标/动作的限制，再在用户两scene范围内选择有依据的控制，明确登记目标变更。

## 资源与交付

R2控制器PID16361已完成，SAM2和两路ProPainter阶段返回0；本轮GPU步骤已结束，无训练。控制器有互斥锁、GPU串行、单阶段1800秒上限；原结果不覆盖。

完整run：`/root/autodl-tmp/runs/worldsim_v77/WS-V77-PIPELINE-R2-20260926/r1`。最新本地页：`C:\Users\dengyunlong\Documents\Codex\2026-09-26\xia\outputs\v77-pipeline-r2\index.html`；前轮目标/放置审核仍在 `outputs/v77-actor-audit/index.html`，旧POC入口已加最新链接。

VGGT系列仍为重建基座；V7.6 HUGSIM/VAD-GS 主线按 [V76-F03](research_failures/entries/V76-F03.md) 关闭。`failure_ledger_refs: [V77-F02]`；`failure_ledger_delta: updated V77-F02`；`human_verdict: null`。
