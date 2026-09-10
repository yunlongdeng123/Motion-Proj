# V74 资产保留与清理

日期：2026-09-10；以用户本轮清理授权为准。旧 V4/V5 永久驻留与 SHA 验收要求已归档，不适用于 V74。

保留 nuScenes/AV2 原始 LiDAR、标定、轨迹和角色记录，V73 关键训练 checkpoint、固定表面、逐实验指标与失败证据，V74 BUILD/QUERY 数据以及外部基线必需模型/环境。
可清理冻结视觉前缀特征、旧传感器 worker 的可重建输入/输出副本、退役路线的非关键中间产物、包下载缓存。每项记录绝对路径、大小、用途、恢复来源与实际结果；不计算新校验和或指纹。
不得删除整个版本 run 来省事；不把文件释放的目录大小简单相加冒充实际空闲空间增量。清理使部分旧实验需要重建缓存，这一代价必须明确。

实际清理清单：`autoresearch/worldsim_v74/p0/storage_result.json`（执行后生成）。
[原保留政策历史快照](archive/2026-09/pre-v74/V73_ARTIFACT_RETENTION.md)。


## 已执行清理

WS-V74-P0-STORAGE-01 done，提交 463ef199；4105 个目标，实际释放 131.802 GiB。5 处 frozen_prefix 与退役 V6/V6.3–V6.7 逐帧缓存；保留 340 代表文件。
完整路径/大小/原因/恢复方式见 storage_plan.json，实际空间差见 storage_result.json。P0 后可用 194.126 GiB。V74 已导出的 BUILD/QUERY/geometry 和新 FINAL raw 纳入保留范围。
