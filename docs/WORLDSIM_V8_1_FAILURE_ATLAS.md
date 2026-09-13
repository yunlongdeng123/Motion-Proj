# V8.1 GPU Failure / Goodcase Atlas

![Architecture components](figures/worldsim_v81_gpu/architecture.png)

已完成两模型各220项真实推理。31张图包括24张逐案例对比、2张探索筛选复核图和5张机制/统计/架构图。完整gallery在run交付目录；所有案例为DISCOVERY，人工verdict=null。

| 案例 | 复核 | 观察 |
|---|---|---|
| scene-0071_cd3039e0_CAM_BACK_LEFT_01 | CONFOUND_EXCLUDE_SCIENTIFIC_BADCASE | 全图显示白色区域为前景停放卡车，参考主要来自后方建筑；不是干净低纹理墙面。GT点集中在卡车上沿与背景交界。 |
| scene-0535_05835231_CAM_FRONT_RIGHT_01 | CONFOUND_EXCLUDE_SCIENTIFIC_BADCASE | 前景铁丝网覆盖背景墙；RGB纹理与LiDAR支撑平面不一一对应。 |
| scene-0919_7cd0fb9a_CAM_BACK_04 | CONFOUND_EXCLUDE_SCIENTIFIC_BADCASE | 植被/绿篱，不属于硬质静态平面。 |
| scene-0966_6e497a35_CAM_BACK_LEFT_03 | CONFOUND_EXCLUDE_SCIENTIFIC_BADCASE | 墙面右侧有树枝叶片和前景遮挡，整ROI平面绑定未建立。 |
| scene-0862_9b1a2d20_CAM_BACK_RIGHT_01 | PLANAR_INTERIOR_CANDIDATE | 灰色墙板与接缝；左边有窄前景，需要核对参考分布。 |
| scene-0626_9a9c05fe_CAM_BACK_RIGHT_21 | CONFOUND_EXCLUDE_SCIENTIFIC_GOODCASE | 橙色板下方有网格和混凝土，多深度边界；不能用低平均误差认定干净goodcase。 |
| scene-0450_023039d6_CAM_FRONT_RIGHT_01 | PLANAR_INTERIOR_CANDIDATE | 远处平面外墙，顶部天空边缘；距离不同不能与近墙直接做因果对照。 |
| scene-0139_7e27d5c0_CAM_BACK_LEFT_12 | REVIEWED_GOODCASE_CANDIDATE | 全图与297个参考点支持同一灰色墙板，DVGT MAE0.103m、法向2.73度；仍为探索集。 |
| scene-0800_a4354e58_CAM_BACK_LEFT_23 | EXPOSURE_CONFOUND_CANDIDATE | 深色风化板下方带草木/地面边缘，阴影曝光混杂。 |
| scene-0911_bdb59582_CAM_BACK_RIGHT_21 | CONFOUND_EXCLUDE_SCIENTIFIC_BADCASE | 墙与地面大面积相交，整ROI不属于单一非地面平面。 |
| scene-0919_a6711744_CAM_BACK_LEFT_12 | PLANAR_INTERIOR_CANDIDATE | 墙与门、标牌和空调边缘，需要限制独立参考支撑区域。 |
| scene-0632_1b3e964e_CAM_FRONT_RIGHT_13 | REVIEWED_RAW_VGGT_BADCASE_CANDIDATE | 全图为木板墙，92个held-out支撑；VGGT MAE3.02m/法向29.8度，DVGT0.50m/8.3度；只代表raw VGGT当前尺度合同的局部误差。 |
| scene-0139_7e27d5c0_CAM_FRONT_LEFT_02 | PLANAR_INTERIOR_CANDIDATE | 灰色墙板；属于scene-0139同一日志，不能冒充独立复现。 |
| scene-0632_fd5b6a5c_CAM_BACK_RIGHT_01 | PLANAR_INTERIOR_CANDIDATE | 木板墙内部；与同scene前视图属于同一日志。 |
| scene-0919_7cd0fb9a_CAM_BACK_24 | CONFOUND_EXCLUDE_SCIENTIFIC_GOODCASE | 植被，不能用小误差当平面goodcase。 |

8个CPU冻结卡、3个冻结视角干预目标以及posthoc选择规则均保留来源；不因结果不好删图。框选错绑参考的卡车/栅栏不能作为科学badcase，植被小误差不能作为goodcase。详见[科学报告](WORLDSIM_V8_1_SCIENTIFIC_REPORT.md)和[V8.2决策](WORLDSIM_V8_1_TO_V8_2_DECISION.md)。

![paired_view_effects.png](figures/worldsim_v81_gpu/paired_view_effects.png)

![same_scene_views.png](figures/worldsim_v81_gpu/same_scene_views.png)

![texture_diagnostic.png](figures/worldsim_v81_gpu/texture_diagnostic.png)

![outside_roi_scale_control.png](figures/worldsim_v81_gpu/outside_roi_scale_control.png)

![case_scene-0071_cd3039e0_CAM_BACK_LEFT_01.png](figures/worldsim_v81_gpu/case_scene-0071_cd3039e0_CAM_BACK_LEFT_01.png)

![case_scene-0139_7e27d5c0_CAM_BACK_LEFT_12.png](figures/worldsim_v81_gpu/case_scene-0139_7e27d5c0_CAM_BACK_LEFT_12.png)

![case_scene-0632_1b3e964e_CAM_FRONT_RIGHT_13.png](figures/worldsim_v81_gpu/case_scene-0632_1b3e964e_CAM_FRONT_RIGHT_13.png)
