# V7.3：AV2逐返回TSDF背景与旧日志组合读出

2026-09-08。旧AV2开发日志的逐返回TSDF构建及固定Actor表面比较已完成；新20日志的同参数背景候选也已构建完成，尚未开始模型质量确认。低背景侵入伴随覆盖损失，不能据此宣布完整场景重建成功。

## 正确处理逐返回传感器原点

[VDBFusion官方接口](https://github.com/PRBonn/vdbfusion)每次积分接收一组点和一个原点。AV2已补偿到reference ego的端点经reference位姿转world一次；真实射线原点仍随每次返回时刻、双LiDAR标定和ego运动变化，不能将整个扫描替换为统一原点，也不能再次对已补偿端点施加逐点ego变换。数据定义沿用项目已有`AV2MetricLog`及其已披露的双LiDAR编号约定，未在新日志重新选择标定。

`vdb_integration.integrate_per_origin`只合并实际相同的float64原点，先排序分组再依次积分，不取整时刻/位置、不平均原点、不裁减返回数。这样保留正确原点，同时避免逐原点反复扫描整点集。组内顺序保持；不同原点组按确定排序处理，未宣称浮点加权融合对任意处理顺序逐比特相同。

固定voxel.1m/trunc.3m、uniform weight、space_carving=true、fill_holes=false/min_weight0。背景来自原父数据同4个build扫描、有效逐返回sensor pose、全部已知有效时刻框+.1m之外的端点，无near-sensor移除。原生MC只在8角都有观测权重的单元形成表面。另存一份经所有原build首回波前0.2m的单次整三角面雕刻表面；首回波后方仍未知。

## 旧开发日志实际构建

`WS-V73-M4-AV2-SCENE-DATA-01/20260908T083000Z__old-development-vdb-per-return-r2`，code7dadd661，CPU38.345745s、RSS1.715073GiB、0 GPU/更新。旧日志02678d04、21 Actor的原始heldout束和逐返回只读轨迹沿用父数据，不改变Actor表面或归属。

| 项目 | 数值 |
|---|---:|
| build原返回 | 375000 |
| 实际积分背景返回 | 364570 |
| 父PCA去重后的背景中心 | 364560 |
| 真实原点分组 | 218766 |
| 原生三角面 | 393002 |
| build雕刻删除 | 11499 (2.926%) |
| 保留三角面 | 381503 |
| build矛盾束 | 9523 → 0 |
| 雕刻后build侵入距离和 | 0m |

四扫描保留返回数91095/89913/90813/92749均与父数据一致。TSDF按原测量逐返回加权，而父PCA先去重float32端点，10个重复返回的差异是权重组织差异，不是改变输入选取规则。只保留可用于读出的两份显式表面和完整构建参数；体积可重建，没有额外保存大VDB文件。

## 同一旧日志的场景读出

`WS-V73-M4-AV2-SCENE-01/20260908T083000Z__old-development-vdb-r4`，codee93c8b94，CPU13.407617s/RSS.704468GiB，0新神经推理/训练。固定已存R5、R9、native fusion，与共同PCA和仅背景在同一TSDF+build雕刻背景上全局排序。使用187494条原返回，其中5257条cohort返回、88条moving cohort返回。该日志已用于实现选择，不能称独立新域确认；没有单日志bootstrap区间。

仅背景相对原PCA+build雕刻：全束hit48.520%→39.175%、early14.150%→4.299%、miss32.133%→45.543%、free.513045→.107526m。差值为hit−9.346pp、early−9.850pp、miss+13.409pp、free−.405518m。相交误差降低伴随覆盖显著下降，不以条件返回MAE下降掩盖缺失。

下表为同一TSDF背景下的cohort返回，按原束汇总，与单体Actor等权表不同：

| 方法 | hit(%) | early(%) | miss(%) | free(m) | 已返回MAE(m) |
|---|---:|---:|---:|---:|---:|
| 仅背景 | .019 | .190 | 86.380 | .006269 | 12.654873 |
| LiDAR PCA | 62.907 | 9.397 | 22.656 | .044827 | .433620 |
| native fusion | 42.534 | 33.137 | 18.889 | .343803 | .796117 |
| LiDAR R9 | 54.194 | 9.017 | 18.528 | .058941 | .683920 |
| joint R5 | 53.472 | 30.455 | 10.101 | .299177 | .568519 |

同背景R5−R9的cohort early增加21.438pp、free增加.240236m，miss减少8.427pp、hit减少.7228pp。R5和R9的FIT标签范围及适配通路不同，该比较是固定候选整体差异，不单独归因于空间attention。原生融合亦不能替代仍在训练的full_track原生强控制R11。

同一R5仅换背景时cohort hit保持53.472%，early30.588%→30.455%，miss9.892%→10.101%，free.314317→.299177m。背景改动不能消除已有Actor前表面错误。moving的88条返回下R5 hit22.727%、early6.818%、miss45.455%、free.855627m，与原背景相同；这里背景没有预测到这些移动对象的首表面，不能把Actor问题解释成背景拼接误差。

## 新20日志准备及研究边界

`WS-V73-M4-AV2-SCENE-DATA-01/20260908T084000Z__external20-vdb-per-return-r2`已完成，code75ac0563。使用父数据`20260908T031000Z__external20-per-return-build-background-r1`的全部20身份/936 Actor/80 build扫描，构建相同参数的TSDF候选。单CPU进程wall1249.572386s（20.83分钟）、RSS4.307499GiB、0 GPU/optimizer更新。保留原PCA背景供敏感性比较；候选准备并不宣布其优于PCA或选为最终背景。

| 新20日志构建量 | 数值 |
|---|---:|
| 原build返回 | 7815152 |
| 实际积分背景返回 | 6839969 |
| 父PCA去重中心 | 6839807 |
| 逐返回真实原点分组 | 4149271 |
| 原生三角面 | 6962322 |
| build雕刻删除 | 146991 (2.111%) |
| 保留三角面 | 6815331 |
| 剩余build矛盾束 / 侵入距离和 | 0 / 0m |
| 链接原heldout束 | 3908250 |

积分数与父PCA中心相差162个原重复返回；TSDF保留原返回单位权重，不是增加新的观测或更改选点。全部20日志的固定build雕刻完成、进程PID64413已退出；该零build侵入仅是构建约束的读数，不能推断heldout完整性或泛化。原生与雕刻两份mesh均保留，未保存大VDB体积。归档`docs/autoresearch/worldsim_v73/m4/av2_external20_vdb_construction_r2.json`包含全部日志的参数、数量和构建记录，未将76MB Actor轨迹index复制进git。

此阶段只读build sweep文件；heldout原束文件直接链接，Actor轨迹/owner/sensor-known定义沿用父数据，没有按新域质量改变背景参数或选择性替换日志。新20日志尚未运行模型质量评估。输入准备就绪后仍需先固定最终方法及背景选择，再进行已登记的独立日志确认。

R10/R11继续，R12与visual-only新训练按既定顺序推进。F02/F04/F05持续，整个V7.3未完成。原始证据：`docs/autoresearch/worldsim_v73/m4/av2_old_vdb_construction_r2.json`、`av2_old_vdb_scene_r4_summary.json`、`av2_old_vdb_scene_r4_paired.json`；原run保留两份背景与逐帧场景读出。
