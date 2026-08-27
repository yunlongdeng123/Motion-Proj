# P2C Continuous Actor-Time Cost Preregistration

P2R 的 1.5m binary outcome 在 train/eval 都是零正例，`V65-F04` 保留且不通过扩大半径恢复。P2C 把监督迁移为
连续 cost：`target_cost = exp(-target_swept_actor_to_ego_route_min_distance / 6m)`；target actor 缺失固定为 60m。
它对应 joint dynamics/cost map、Occupancy Flow、DTPP/DiffStack 的连续时空 cost 思路。

数据 split、method/target frame separation、A0/A1 features、seed、MLP 与 epochs 全部保持 P2R 不变。新 cache path
保留旧 0-positive cache。Primary gates：Spearman gain `>=0.05`、MSE reduction `>=10%`、matched-40% selected
mean cost reduction `>=10%`、eval scene support=2/2、真实 temporal features 优于 scene shuffle。仍只是 legacy
train-only diagnostic；失败关闭 actor-time family，成功才允许新 fresh cohort。
