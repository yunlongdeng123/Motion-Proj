# V7.3 独立确认数据准备

2026-09-08，仅元数据/文件存在性调查，未读取候选图像、点云、标注或模型质量。

nuScenes本地原始载荷调查耗时10.05s：当前31窗口属于25日志，磁盘完整七传感器关键帧足够形成四build时刻的共35场景/27日志；当前日志外仅scene-0139和scene-0379。二者分别出现在V6.4 fit和V5诊断/开发配置，不能因不在当前25日志中就重用为全新确认。逐场景可用性见 `docs/autoresearch/worldsim_v73/coverage/log_payload_inventory_r1.json`。

更完整的身份调查发现：nuScenes trainval全部850场景只属于68日志，而旧V7.2角色表已覆盖全部68条（train54、legacy2、exposure_unknown2、dev4、route_select3、source_test3）。因此按旧记录保守排除后，没有无历史角色的新trainval日志。公开归档确实挂载于 `/root/autodl-pub/nuScenes/Fulldatasetv1.0/Trainval`，十个blob约294GiB，可解决旧日志缺相机载荷问题，但不能创造日志独立性。尚未为此扫描/解压整套归档，也不把更多相邻场景计作独立样本。

对应F05，继续当前开发训练，同时迁移到官方公开AV2 Sensor train中尚未使用的日志作为外部确认候选。官方train/val名称不会决定本研究是否训练：最终确认所选train日志在V7.3中仅供固定方法构建与评价，不能进入共享参数或超参数选择。跨数据集结果需明确是新日志与传感器域同时改变，不能伪称同分布确认，也不能与现有nuScenes开发集混合统计。

依据[AV2官方说明](https://argoverse.org/av2.html)、[官方下载说明](https://argoverse.github.io/user-guide/getting_started.html)与[官方传感器格式](https://github.com/argoverse/user-guide/blob/main/guide/src/datasets/sensor.md)：使用已公开的标定、时间戳、刚体轨迹与周视相机；保留每台相机自己的采集时刻。AV2点云已做ego运动补偿且在egovehicle坐标，双LiDAR来源和束原点需按官方字段重建，不能直接套nuScenes单传感器扫描模型。实现/对应问题在已经使用的AV2开发日志上处理；最终候选先依身份和时间元数据选取，待方法定型后评价，不按预测结果筛日志。

官方S3 train目录已只读列举成功，现有s5cmd可匿名访问。下一步排除历史配置/角色中已出现身份，登记20条日志及四build/两留出时刻的原始载荷清单，按需下载七周视相机、相关LiDAR和标定/轨迹，保留缺输入对象和缺返回。当前尚未选定候选名单、未下载新传感器载荷、未开展外部评价。磁盘约128GiB可用，该卡点是数据独立性与格式迁移，不是要求立即关机的资源不足。
