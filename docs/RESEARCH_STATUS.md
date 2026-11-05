# 当前研究状态

2026-11-05，Scene008 转入双实例独立恢复与局部表示替换。固定 a6/a37 实测轨迹和 R3 道路，在原生高斯合成中联合处理遮挡。两车分别使用可见实例监督和实测锚点，颜色只由各自的观测更新。原生静态遮挡归因在上一轮收口。[本轮组件图、执行合同和输入说明](autoresearch/scene008_instances_20261105/README.md)

CPU 建库、固定评价域、局部替换候选与接线检查完成。用户开启 GPU 后，先在 raw105、113、130 做真实环境 smoke，覆盖强基线重放、前后向、B/C 完整状态恢复和 D 的固定 PVD 接入。通过后进入完整 49 帧归因、正式更新和配对验收。用户已明确本轮不授权自动关机。

强编辑基线仍为单 a37 原生 10K 加最终 R3 道路 1K。历史对照见 [C2 全帧报告](autoresearch/scene008_a37_C2_vehicle_20261104/index.html)、[静态残留归因](autoresearch/scene008_static_origin_audit_20261104/index.html)和[双动态并集监督对照](autoresearch/scene008_a6_a37_dynamic_20261104/index.html)。正式质量待评价，`human_verdict=null`。
