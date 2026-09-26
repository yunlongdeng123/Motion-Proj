# 当前研究状态

更新：2026-09-26（Asia/Singapore）。远端分支 **v77**，工作区 `/root/autodl-tmp/motion_proj_v77`。阶段：**两场景 background + 显式actor 持续迭代；用户要求评估DriveEditor补景；源码/资源/两scene mask预检已完成，下一项用已训练权重做零训练删除对照。** 保持frozen VGGT-Ω + SAM2 + 原Hunyuan GLB；ProPainter作为当前背景基线，DriveEditor仅作为用户明确提出的一项替换候选，串行有界比较；未替换已验收模块、不训练Ω。完整新证据见 [PIPELINE_R3](v77/PIPELINE_R3.md)，按run查 [EXPERIMENTS](EXPERIMENTS.md)。

## 当前结论与边界

`WS-V77-DRIVEEDITOR-ASSESS-20260926/r1` 仅CPU预检20帧，无生成前向。官方deletion使用原框确定扩大mask，禁用有效物体外观/位置条件；scale_bbox含随机中心偏移与大框倍率限制。0230/CAM2 f0–9删除mask占28.5%–73.3%画面，0255/CAM3 f15–24占15.0%–15.9%，两例10/10帧均与其他GT框投影相交。是潜在邻车/背景改动范围，不是生成后质量失败。评估与数据方案见 [DRIVEEDITOR_ASSESSMENT](v77/DRIVEEDITOR_ASSESSMENT.md)。

用户最新人工观察（R3审核页）：“目标选中没问题，原位 factual 问题不大，就是删除了以后背景补全就是糊的了”。记录为局部视觉反馈，不扩大为全流程/高保真通过；整体 `human_verdict: null` 保持。

`WS-V77-PIPELINE-R3-20260926/r1` 固定0230/actor22与0255/actor25，GT已知131/181帧×6相机。发现0230原GT框底高度漂移会放过车底射线；将框仅向下闭合到局部地面后，核心168点原本全部“可见”变为0。0255剩9/207视线候选，但无15cm内地面LiDAR支持。两例联合候选为0，背景参考状态为 `no_jointly_supported_core_reference`，停止据此复制所谓已观测背景。

这只是保守地面检查，不是全场景不可见证明或残影唯一因果解释。阴影、mask、照度尚未分离；没有新模型前向、训练、重建或MOVE，画面未宣称改善。原GLB形状失败/关闭路线的过早归因仍撤回，用户本地Blender形状反馈保留，人工 `human_verdict: null`。

前轮放置修正保持：0230车头180°校正、0255方向不变，相机投影已修复。R2空间拒绝保持：0230旧2m碰邻车；0255旧2m/车头前移2m的100/100帧均有静态LiDAR占据。0230前移1m暂无正占据，也不能当已证实自由空间。R2的196帧ProPainter控制仍留主要残影，不再扫长度参数。证据见 [ACTOR_COMMAND_AUDIT](v77/ACTOR_COMMAND_AUDIT.md)、[PIPELINE_R2](v77/PIPELINE_R2.md)。

## 下一次有界工作

先恢复DriveEditor已训练checkpoint，检查旧单3090串行CFG/分块解码适配；官方GUI要求更大显存，不直接声称原版可装入24GB。固定同两个actor、各10帧、1024×576，串行运行A=ProPainter+原mask，B=ProPainter+保存的官方扩大mask，C=DriveEditor+与B相同mask。显式物化mask避免默认二次膨胀，并保存整帧原始生成与局部合成。先验收残留、邻车保持、道路/围栏结构与时序，视频成功后再接冻结Ω检查跨视角几何。

大mask若造成邻车改动，停止该原样输入实验，另列保护其他可见actor的适配，不冒充官方结果。此处框投影相交不等于实际可见mask重叠。当前仅完成评估，尚无DriveEditor对这两个scene的生成成绩。

暂不造训练集、不启动训练。只有零训练对照显示明确训练分布缺口才构造真实干净视频→时序车辆形状mask→原RGB监督的200–500条小样，scene级独立验证后再扩展；生成补景不能冒充真实GT。跨视角不一致须先区分任务条件不足与数据数量。资产参考围栏问题保持次要。

当前未批准任何新的MOVE。只有取得足够空间证据再渲染；道路规则/ego/转向时序仍须区别于无框碰撞。若后续需在这两个scene内更换actor，先记录原目标限制与目标选择依据，不悄悄替换分母。

任务heartbeat `v77-pipeline` 已启用，每30分钟检查并继续；先读状态和实际进程，避免重复启动。状态不变不通知，只报实质结果/失败/完成/所需用户行动。用户授权持续自主迭代，人工verdict只由用户填写。

## 资源与交付

最新DriveEditor预检在envs/driveeditor上成功运行；源码和环境仍在，约12.06GB model.safetensors及demo data.pkl已清理、当前symlink目标不存在。尚未下载或加载模型，RTX3090 24GiB空闲；无新GPU作业。R3的12项几何测试结论保留。R2控制器已退出。原模型、资产、全部前后控制保留；R3原框输出另存 `raw_box_control/`。

本轮预检run：`/root/autodl-tmp/runs/worldsim_v77/WS-V77-DRIVEEDITOR-ASSESS-20260926/r1`；本地评估报告 `outputs/v77-driveeditor-assessment/report.md`。R3完整run：`/root/autodl-tmp/runs/worldsim_v77/WS-V77-PIPELINE-R3-20260926/r1`。最新本地页：`C:\Users\dengyunlong\Documents\Codex\2026-09-26\xia\outputs\v77-pipeline-r3\index.html`，含新证据图与明确标注来源的已有三栏视频；R2完整10Hz视频另有链接。未作浏览器交互验收。旧POC入口已加最新链接。

VGGT系列仍为重建基座；V7.6按 [V76-F03](research_failures/entries/V76-F03.md) 关闭。更新同一 [V77-F02](research_failures/entries/V77-F02.md)，不新增失败ID。`failure_ledger_refs: [V77-F02]`；`failure_ledger_delta: updated V77-F02`；`human_verdict: null`。
