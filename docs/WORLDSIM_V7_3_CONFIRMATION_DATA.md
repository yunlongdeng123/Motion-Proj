# V7.3 独立确认数据准备

2026-09-08，仅元数据/文件存在性调查，未读取候选图像、点云、标注或模型质量。

nuScenes本地原始载荷调查耗时10.05s：当前31窗口属于25日志，磁盘完整七传感器关键帧足够形成四build时刻的共35场景/27日志；当前日志外仅scene-0139和scene-0379。二者分别出现在V6.4 fit和V5诊断/开发配置，不能因不在当前25日志中就重用为全新确认。逐场景可用性见 `docs/autoresearch/worldsim_v73/coverage/log_payload_inventory_r1.json`。

更完整的身份调查发现：nuScenes trainval全部850场景只属于68日志，而旧V7.2角色表已覆盖全部68条（train54、legacy2、exposure_unknown2、dev4、route_select3、source_test3）。因此按旧记录保守排除后，没有无历史角色的新trainval日志。公开归档确实挂载于 `/root/autodl-pub/nuScenes/Fulldatasetv1.0/Trainval`，十个blob约294GiB，可解决旧日志缺相机载荷问题，但不能创造日志独立性。尚未为此扫描/解压整套归档，也不把更多相邻场景计作独立样本。

对应F05，继续当前开发训练，同时迁移到官方公开AV2 Sensor train中尚未使用的日志作为外部确认候选。官方train/val名称不会决定本研究是否训练：最终确认所选train日志在V7.3中仅供固定方法构建与评价，不能进入共享参数或超参数选择。跨数据集结果需明确是新日志与传感器域同时改变，不能伪称同分布确认，也不能与现有nuScenes开发集混合统计。

依据[AV2官方说明](https://argoverse.org/av2.html)、[官方下载说明](https://argoverse.github.io/user-guide/getting_started.html)与[官方传感器格式](https://github.com/argoverse/user-guide/blob/main/guide/src/datasets/sensor.md)：使用已公开的标定、时间戳、刚体轨迹与周视相机；保留每台相机自己的采集时刻。AV2点云已做ego运动补偿且在egovehicle坐标，双LiDAR来源和束原点需按官方字段重建，不能直接套nuScenes单传感器扫描模型。实现/对应问题在已经使用的AV2开发日志上处理；最终候选先依身份和时间元数据选取，待方法定型后评价，不按预测结果筛日志。

官方S3 train目录包含700日志，已有仓库身份引用排除1条，699条候选中按日志名字典序选择前20条；名单在列举所选日志的对象元数据之前写入。名单及精确载荷见 `configs/worldsim_v73/av2_external_confirmation_r1.json`，脚本 `scripts/prepare_worldsim_v73_av2_confirmation.py`；后续运行复用既有名单，不因下载或质量失败换日志。元数据准备98.49s，无原始图像/点云/标注值读取。

使用LiDAR原始序号[5,15,20,30]作build、[10,25]作留出，对应约0.5/1.5/2.0/3.0s与1.0/2.5s；七周视相机按每个build时刻最近实际采集时间读取，所有时间戳保留。20日志所需760文件共343260344字节（327.36MiB），目录元数据无缺项；相机与LiDAR时间差范围−23.10至+18.11ms，不能将它们硬置为同一时刻。只复制28相机帧、6个LiDAR扫描和4个标定/轨迹/标注文件每日志，不为选帧下载整个1TB数据集，不使用额外三维目标选择输入。

下载后也不立即开展外部模型选择。格式迁移必须先处理AV2双LiDAR来源/ego补偿、纳秒时间与只读Actor姿态插值、前相机纵向画幅和内参变换。当前query模块的6项nuScenes相机嵌入也不能直接索引为7路AV2：保留全部7路观测，在旧AV2开发日志上确定基于已知标定/方向的接口映射或无数据集ID表示，不能临时丢相机或给第7台加入未经训练随机向量后声称可靠零样本结论。该项尚未实现，不是最终确认已经完成。

当前20条日志身份已登记，精确载荷下载已启动（code c00020fb，PID34904，760文件），已于977.74s后全部复制成功、返回0且下载进程退出；日志在 /root/autodl-tmp/controller_logs/v73_av2_confirmation/，尚未解码新图像/点云或开展外部评价。磁盘约128GiB可用，该卡点是数据独立性与格式迁移，不是要求立即关机的资源不足。
