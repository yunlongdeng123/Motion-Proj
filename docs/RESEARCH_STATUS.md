# 当前研究状态

2026-10-02，wm-3090-1001，v77。Temporal reveal＋合法projected actor-state；surfel后置。数据盘余约44–45GB，用户允许必要时清可恢复旧物，当前未删除。

r37新增固定15scene/450连续曝光；修复有序抽样、多地图分派/显式null道路及空LiDAR处理，Boston旧几何不变。r38的21轨迹完整实例检查仅3条同scene普通背景技术候选，冻结代表S016独立QA0（安全岛落点疑点）；其余未独立QA，不准入。r39唯一过去2秒真实位姿对照两条可行，但均缺几何或显露证据，关闭该尝试。

r40只在固定23例上比较SAM提示帧。6条复用标签的实际提示帧/JPEG溯源已修复，180旧mask帧逐像素不变。原分母：{"nonvehicle_or_incomplete_protection": 8, "instance_identity_or_visibility_uncertain": 7, "insufficient_actual_occlusion": 1, "static_foreground": 1, "ordinary_background": 3, "actual_size": 2, "insufficient_other_frame_evidence": 1}；新分母：{"nonvehicle_or_incomplete_protection": 8, "instance_identity_or_visibility_uncertain": 6, "insufficient_actual_occlusion": 1, "insufficient_other_frame_evidence": 2, "static_foreground": 1, "ordinary_background": 3, "actual_size": 2}。身份疑点7→6，但该例仍缺显露依据；无新增技术候选或训练准入，关闭提示帧对照，不推广为全局策略。

r41固定来源过程筛选完成。在旧440候选池中排除旧scene后203个actor窗口，固定静止B、清楚尺寸、至少16m深度等要求筛至9条，7条有合法30曝光，2条抽样失败。7条相机三秒累计行程仅0.002–0.375m，均不满足预定1–8m运动要求，最终0来源。没有提新RGB、运行SAM或降低阈值；这是既有候选池的缺口，不能推断整个nuScenes不存在该过程。

旧Q060输入/条件QA2保留，Q046条件仍1；r36可见一致性过滤未推广。新数据准入0，50条配方未齐，正式A/B/C新增训练0步。不能将来源增加、地图通过或条件指标当成真实DELETE收益；下一步检查来源池每scene只保留3条旧方案是否提前丢掉了所需过程，保留全部质量要求做一次无截断元数据对照。

模型入口仍r21 sam_full_v2，r26裁洞失败不推广。GT相机/轨迹/LiDAR明示POC辅助，Y只监督/质检，human verdict空。没有新增定时任务；真实DELETE保护车和无车背景两侧跨scene收益未达到，关机条件未满足。

报告及组件图：[r37–r41](v77/TARGET_PROTECTED_FRESH_SOURCES_R37_R40.md)；本地`outputs/v77-target-protected-r38/index.html`。继续更新同一[V77-F02](research_failures/entries/V77-F02.md)。
