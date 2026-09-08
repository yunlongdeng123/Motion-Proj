# V7.3：零LiDAR输入与视觉支持路径

2026-09-08。主联合训练的旧cohort定义把零build LiDAR对象统一标为unavailable_input，即使已有图像/相机位姿；缺失被保留在评价分母，但不能由此推断没有可用视觉信息。

已查[VGGT官方原生几何头](https://raw.githubusercontent.com/facebookresearch/vggt/main/vggt/models/vggt.py)与[SparseNeuS官方实现](https://github.com/xxlong0/SparseNeuS)，两者体现图像几何预测不必以Actor LiDAR点为前提。前者CVPR2025、后者ECCV2022。这些工作不保证当前动态Actor的米制对齐、遮挡归属或生成表面正确；本轮优先使用已有DPT与空间查询，不增加另一种世界表示。

## 完整输入范围核查

任务`WS-V73-M2-EMPTY-INPUTS-01/20260908T062000Z__population-build-availability-r1`，code b9690161，CPU1.249847s/RSS0.362228GiB、零新网络推理。全部51个原metadata零LiDAR对象，读取现有标定、框、原build近框束和已完成M1r3 native fusion；仅fit侧统计既定full_track标签数量。

| 项目 | fit | development |
|---|---:|---:|
| 零LiDAR Actor | 43 | 8 |
| 所属独立日志 | 9 | 2 |
| 有相机时刻已知位姿 | 41 | 5 |
| 有保守框—视锥重叠 | 41 | 5 |
| 原native fusion有框内候选/表面 | 33 | 5 |
| 有build近框原束 | 33 | 7 |
| build近框原束总数 | 43894 | 70 |
| 有窗口heldout归属返回的Actor | 14 | 1 |
| 窗口heldout归属返回总数 | 24 | 1 |
| 有fit full_track正目标 | 36 | 不读取 |
| fit full_track目标点总数 | 58710 | 不读取 |

41个有相机的fit对象中，34个有full_track正目标，8个在旧native fusion中没有原生候选，9个无build近框束。其余2个fit/3个开发对象没有相机位姿，仍不能按visual-only处理。相机视锥重叠是保守几何可能性，不是实际可见性；native框内候选数是已有预测支持，不是正确性标签。

开发零LiDAR组只有一条归属回波，无法用该组提出可信的跨日志几何精度结论。其输入覆盖比例和自由空间约束可记录；任何有效质量结论仍需更广的既定新日志确认。

## 显式入口及当前边界

`evaluate_worldsim_v73_fixed_actors.py`新增显式`--include-visual-only`，允许零LiDAR但有相机位姿的对象读取已有原生深度/特征；`--input-subset zero_lidar`按原build计数覆盖全部51个对象，不按预测好坏筛选。默认关闭该入口，R10/R11以及既定R12对照保持原cohort与目标。

已有`native_surface_seeds`先用原生表面，空时使用LiDAR；两者均空时返回None，交给模型原有带三维位置的coarse查询，不伪造LiDAR点。原生融合/原生头模式没有任何支持时仍返回空表面。两种输入都缺失的对象保留缺失。没有新增opacity、存在概率或目标驱动支持筛选。

需要明确：r5 checkpoint没有在这些零LiDAR对象上训练；它的coarse参数主要在“原生seed+小残差”下学习，直接使用Actor尺度coarse查询是输入条件迁移，不能假装这个退路已经充分训练。这里先验证完整固定推理入口并保留硬读出，之后是否将新增训练对象并入须作为独立cohort变化，不能与R12的free目标修改混在一起。

固定推理`WS-V73-M2-EMPTY-INPUTS-01/20260908T063000Z__fixed-joint-r5-visual-only-r2`已完成，code6ebf1289、r5最终checkpoint、全部51零LiDAR对象、0更新，不读外部20日志。wall75.879384s、allocated GPU峰值2.686777GiB、CPU RSS峰值13.917145GiB，PID58903已退出。它与R10/R11同时运行时合计GPU约20282MiB，没有引发资源中断。

## 固定入口结果：支持生成不等于几何改善

43个fit对象中41个生成512片表面，32个来自当前r5原生种子、9个使用coarse退路；2个无相机者保留空。开发8对象中5个生成512片，均有当前原生种子，3个无相机者保留空。先前M1r3 native fusion在fit的33有原生支持/8无原生支持与此不同，因为r5的原生头已适配，不能把差异全部归因于新增查询。

| 固定方法/输入角色 | 有表面 | build近框free(m) | 窗口heldout近框free(m) |
|---|---:|---:|---:|
| M1r3 native fusion / fit | 33/43 | 0.073563 | 0.024463 |
| r5 visual-only入口 / fit | 41/43 | 0.344801 | 1.201042 |
| M1r3 native fusion / development | 5/8 | 0.179105 | 0 |
| r5 visual-only入口 / development | 5/8 | 0 | 0 |

表中先按Actor原近框束加权帧，再日志内Actor及日志等权；fit有9日志、build/heldout分别43894/22932个Actor–ray实例，development有2日志、70/59实例。近框束可能跨Actor重复，不是独立样本量。窗口heldout时间的24条fit归属回波分布14个Actor/6日志；r5日志等权hit5.556%、miss57.222%，native fusion为0.556%/86.667%。这些来自已用于主体适配的fit日志，不承载新日志泛化主张；减少缺失同时free明显增大。

开发唯一归属回波来自scene-0919/94ffa1429ec745bdb5505eb340d76e09，两方法都缺失。尽管可以生成表面、开发近框free为0，也不能推出开发几何更好；无命中和极小评价分母同时保留。没有为一条回波生成bootstrap区间。

结论是显式推理入口可运行，直接把旧r5 checkpoint用于这个输入条件会在fit窗口产生更多自由空间违规。后续需要独立加入真实visual-only训练对象及其fit侧标签/原束约束，不能把coarse输出的存在当作修复。训练器入口目前尚未增加；R10/R11/R12仍保持旧cohort，新的cohort变化必须单独比较，不与free目标修改混在一起。

输入核查证据`docs/autoresearch/worldsim_v73/m2/global/empty_inputs_r1_summary.json`；固定推理原始摘要`empty_fixed_r2_summary.json`、分组分析`empty_fixed_r2_analysis.json`，同属该目录。原run保留全部51表面和逐帧硬读出。F02/F03/F05持续，V7.3未完成，shutdown=false。
