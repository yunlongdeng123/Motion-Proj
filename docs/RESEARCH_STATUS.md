# 当前研究状态

更新：2026-09-26（Asia/Singapore）。远端分支 **`v77`**，工作区 `/root/autodl-tmp/motion_proj_v77`。阶段：**两场景显式资产 POC 已执行并交人眼复核；代理观察未达高保真停止规则，关闭这条单图 Hunyuan3D-2.1 + ProPainter 拼接入口，不进入训练。** 完整输入、架构、结果及工程修正见 [EXPLICIT_POC](v77/EXPLICIT_POC.md)，按run查 [EXPERIMENTS](EXPERIMENTS.md)。人工 `human_verdict` 保持 `null`。

## 本轮结果与边界

`WS-V77-EXPLICIT-POC-20260926/r1` 固定 `scene_0230/22` 与 `scene_0255/25`。SAM2.1 large 以 GT 框提示并传播完整活跃相机序列；ProPainter 生成去车视频；对两场景各10个时间点六视图重跑冻结Ω，按GT相机及框外LiDAR尺度转成逐时刻背景点云；官方 Hunyuan3D-2.1 单张crop生成两个shape/PBR资产，轴向规范后导出两个有效GLB；GT位姿/尺寸放置和固定侧向2m MOVE。零训练。最终六相机原视频、视频DELETE、原位GLB、MOVE、Ω背景DELETE/MOVE分别编码，离线页与GLB已保存。

无资产 DELETE 已有深色涂抹或车形残影，GLB 与原车车型/外观不一致；因此解析变换和有效GLB不足以让画面高保真通过。Ω远景深灰点空洞有点渲染影响，不单独归咎于模型。选8张crop，但2.1实际只消费1张；未验证2mv。SAM2和编辑依赖GT，这轮不是自动端到端系统。新 [V77-F02](research_failures/entries/V77-F02.md) 记录这条拼接入口的限定失败，不对VGGT系列或多视图生成路线作普遍否定。

前轮 `WS-V77-P0-24ACTOR-20260926/r1` 的24对象直接Ω框内点适配与三个完整连续视频仍保留，见 [P0_RESULTS](v77/P0_RESULTS.md)、[VIDEO_REVIEW](v77/VIDEO_REVIEW.md)、[V77-F01](research_failures/entries/V77-F01.md)。两批资产失败来源不同，不能互相替代证据。

## 下一步与资源

优先请用户在离线页核对 `scene_0230/CAM2/f005–010` 和 `scene_0255/CAM3/f020–040`，尤其 DELETE 原位置、GLB 多视角同车程度、MOVE 新位置和六相机时间变化；人工字段由用户或指定评审填写。当前不扩展分母、不训练Ω、不做language/RL，也不自行设计3D生成网络。若要验证2mv，应另建单独协议，先明确4–8张多视图是否真的进入模型。v77基座仍为VGGT系列；V7.6 HUGSIM/VAD-GS 主线按 [V76-F03](research_failures/entries/V76-F03.md) 保持关闭。

远端GPU RTX3090 24GiB，当前有限推理已完成；PBR顺序运行峰值约13.5GiB，冻结Ω六视图约5.26GiB。本轮完整运行留在 `/root/autodl-tmp/runs/worldsim_v77/WS-V77-EXPLICIT-POC-20260926/r1/`，本地交付 `C:\Users\dengyunlong\Documents\Codex\2026-09-26\xia\outputs\v77-explicit-poc\index.html`。`failure_ledger_refs: [V76-F03, V77-F01, V77-F02]`；`failure_ledger_delta: V77-F02`；`human_verdict: null`。
