# 当前研究状态

更新：2026-09-27（Asia/Singapore），分支 **v77**，工作区 `/root/autodl-tmp/motion_proj_v77`。用户授权的完整DELETE工程已跑完；仅使用DriveEditor deletion、冻结Ω及原GLB，不追加ProPainter、训练、新生成器、MOVE或定时任务。

## 当前结果

`WS-V77-DELETE-FULL-20260927/r1`完成：0230/actor22的50帧5秒、0255/actor25的100帧10秒、六相机10Hz；24个DriveEditor窗口、150次六相机Ω、181个原位GLB RGBA/Z层、两条显式DELETE状态。BUILD保存逐时刻B_t，QUERY只关闭目标可见性。900视图mask覆盖与mask外像素检查通过，4项指令/遮挡测试通过，22视频共1650帧实解码通过。

工程完成不等于高保真通过。0230侧面去车较好，但CAM5后向补景重新生成另一辆车；0255前段去掉SUV，后段补景发白、围栏和车列变形。Ω点融合另引入碎裂/重影和远景缺口。0255原位GLB在深度测试后出现明显不合理裁切；取消深度测试的同GLB控制保留，不能据此否定资产，也不能把不遮挡当正确答案。人工verdict保持null。

## 审核与证据

本地`outputs/v77-delete-full/index.html`：原视频（黄标目标）/原位factual/DELETE三栏、六相机总览、直接补景、mask、纯几何、同GLB无深度测试控制、10时刻静帧和18处同时间接缝。[完整报告与architecture](v77/DELETE_FULL.md)、[证据](autoresearch/worldsim_v77/delete_full_20260927/)、[V77-F02](research_failures/entries/V77-F02.md)。远端完整资产根`/root/autodl-tmp/runs/worldsim_v77/WS-V77-DELETE-FULL-20260927/r1`。全部本轮模型/渲染任务已结束，没有后台循环或关机动作。

B_t为逐时刻GT/LiDAR适配点云，非持久静态/4D世界。沿用GT提示SAM门控，小目标/近裁面无mask可能留下目标观测；其他actor仍烘焙在背景。两个已曝光开发场景、隐藏背景无GT、官方训练重叠未核对。旧两条MOVE仍拒绝；不新增自动合法性模块。

## 下一步

优先针对DriveEditor长时序背景漂移做一个固定后段窗口、有/无上一窗条件的强控制，区分累积条件与单窗模型缺口；本轮未执行。Ω错误遮挡独立保留，不扫容差或取消遮挡掩盖。没有证据要求现在自造数据集或训练网络；先由用户审核当前完整视频。

V7.6仍按[V76-F03](research_failures/entries/V76-F03.md)关闭。`failure_ledger_refs: [V77-F02]`；`failure_ledger_delta: updated V77-F02`；`human_verdict: null`。
