# V7.3：同输入Actor TSDF融合与覆盖边界

2026-09-08。该经典融合对照已完成，但当前稀疏输入下的原生TSDF网格大量缺失，不能把它的低free当作重建成功，也不将它列为主模型胜出的强证据。

## 实现、输入和统计边界

任务`WS-V73-M2-TSDF-FUSION-01/20260908T072000Z__population-build-vdb-r1`，code6f753dab。CPU9.081212s、RSS0.649452GiB、0 GPU/optimizer更新，全部489对象处理完成。原有4个build扫描、25日志、414 fit/75 development；51个零LiDAR对象与无heldout owned返回对象保留。只融合归属Actor的build端点和同采集时刻的规范坐标传感器原点，不使用图像、全轨迹训练标签、heldout几何或位姿校正。

[Curless–Levoy（SIGGRAPH1996）](https://lightfield.stanford.edu/papers/volrange/)提供范围观测的加权距离体积融合依据；实际积分与Marching Cubes来自已安装的[VDBFusion官方实现](https://github.com/PRBonn/vdbfusion)。官方VDBFusion论文为Sensors2022，不能标成SIGGRAPH或ICRA。本轮没有改动其原生积分/网格源码，也不是旧V7.2带射线上限的zero-crossing点集/anchors并集复现。

固定voxel0.1m、trunc0.3m、统一权重、space_carving=true；fill_holes=false、min_weight=0使网格单元8角均须观测权重。不在unknown边界闭合表面，也不在无TSDF支持时退回PCA。保留原网格，并对另一份表面使用全部原build近框束，在原首回波前0.2m执行一次整三角面雕刻，包括非当前Actor归属的束；首回波后方仍未知。两份表面均先构建，再评价heldout。

采用既有CPU BVH的同一字面三角面首交点/点到曲面距离，无opacity/凸包。TSDF原生三角数和面积不等同于固定大小query patches；兼容字段surface_patches仅数TSDF三角元素，不能当成相同输出预算。CPU与旧GPU读出存在浮点/共面细节边界。射线数是Actor–ray实例，可能跨Actor重复，不是独立日志数。

## 表面生成覆盖

| 输入角色 | Actor | 积分build返回 | 非空TSDF Actor | 原三角数 | build雕刻删面 | 雕刻后总面积(m²) |
|---|---:|---:|---:|---:|---:|---:|
| fit | 414 | 194680 | 144 | 131713 | 2677 | 461.760 |
| development | 75 | 18593 | 18 | 18982 | 398 | 66.885 |

雕刻没有使额外Actor整面消失。fit仍有227个、development仍有49个已有LiDAR输入的Actor未形成TSDF网格。两组build近框束分别728977/88116实例，原网格2624/352条build矛盾束，雕刻后的build侵入距离和均为0；这只是已观测构建时段一致性。

| 原build点数 | Actor总数 | 非空TSDF |
|---|---:|---:|
| 0 | 51 | 0 |
| 1–15 | 179 | 0 |
| 16–63 | 88 | 16 |
| 64–255 | 81 | 56 |
| ≥256 | 90 | 90 |

分组仅按原输入计数，未按效果筛选。该配置在所有≥256点对象上都能形成网格，但这本身并不证明它们的覆盖或几何正确。对1–15点对象，要求观测完整单元角使输出为零；该限制说明本配置对极稀疏Actor输入适用性不足，不能外推为所有TSDF、其他分辨率或融合表示的负结论。

## 完整开发cohort的硬读出

下表在Actor内按原束/点数汇总，再按日志内Actor和独立日志等权。全75对象/5日志保留；23对象无owned heldout返回，不为其编造返回率。表面距离仅在有目标且有表面时定义；TSDF为18对象/4日志，其他组分母见原始分析，不能直接用未配对距离均值宣称差异。

| 方法 | 无表面Actor | hit(%) | early(%) | miss(%) | free(m) | 目标→表面距离(m) | 0.2m recall(%) |
|---|---:|---:|---:|---:|---:|---:|---:|
| LiDAR PCA | 8 | 21.245 | 4.353 | 73.410 | .017708 | .304790 | 65.418 |
| TSDF原生mesh | 57 | 3.320 | .266 | 96.283 | .000856 | .469934 | 21.512 |
| TSDF＋build雕刻 | 57 | 3.279 | .258 | 96.330 | .000773 | .469990 | 21.508 |
| M1r3 native＋LiDAR融合 | 3 | 22.777 | 9.155 | 57.354 | .150304 | .175425 | 78.129 |
| LiDAR-only R9 | 8 | 32.469 | 6.006 | 54.569 | .038543 | .227363 | 72.467 |
| Joint R5 | 8 | 22.277 | 20.981 | 37.709 | .351717 | .145289 | 79.713 |

TSDF＋雕刻相对LiDAR PCA，5日志配对hit−17.966pp、95%区间[−32.317,−5.663]；miss+22.919pp [7.264,41.174]；recall−43.910pp [−58.409,−29.412]。free−.016935m [−.035062,−.004568]同时发生，主要收益伴随大量表面缺失，不能单列free宣布优越。可配对的4日志表面距离反而增加.366072m [.280899,.487521]，不是未配对均值相减。

雕刻相对原TSDF的开发hit变化−.0414pp、miss+.0465pp、free−.0000833m，各自区间触及0，远小于缺少表面支持的问题。移动组9 Actor/2日志中6个无表面、3个无heldout owned返回；hit7.657%、early.430%、miss91.237%、free.001030m；表面距离仅1日志，不做其跨日志不确定性主张。

## 研究决策

该结果仅说明0.1m、不补未知洞、原生Marching Cubes的TSDF在当前极稀疏Actor输入上覆盖不足。它保留为经典融合适用性与表面支持诊断，不成为主方法“超过弱基线”的核心证据；LiDAR PCA、native/LoRA适配、同容量LiDAR解码器及AdaPoinTr等仍保留。暂不围绕此弱支持配置开展参数网格搜索或增加主表示。背景TSDF结果基于更密的静态返回，两者不能互相外推。

继续原R10全轨迹联合训练、R11强原生头及已登记R12几何free对照。零LiDAR视觉训练入口另行验证；外部20日志模型质量仍未读取，F02/F03/F04/F05持续，V7.3未完成。

构建与完整配对证据归档`docs/autoresearch/worldsim_v73/m2/global/actor_tsdf_r1_construction.json`和`actor_tsdf_r1_analysis.json`；原run的uncarved/carved目录保留全部表面和逐帧结果。图`V73_ACTOR_TSDF.png/pdf`直接读取这些结果，无新推理。
