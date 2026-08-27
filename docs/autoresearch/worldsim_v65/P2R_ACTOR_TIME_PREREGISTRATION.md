# P2R Actor-Time Action-Outcome Train-Only Preregistration

`WS-V65-H-P2R-001` 不是 P2 的 seed/capacity rescue。P2 的 trajectory-only ranking 负结论保持冻结；本任务把预测
单位从 static voxel 改成 Actor token，把 target 从 hidden-FREE 改成近未来 Actor swept envelope 是否与 ego future
route 相交。

输入只用 legacy V6.3 evidence：method-visible offsets=`[-6,-4,-2,0]`；监督 offsets=`[-5,-3,-1,+1]`，两者
无帧重叠。训练 scenes=`0071/0317/0862/1012`，nested legacy evaluation=`0450/1089`；全部只属于 train-only
机制诊断，不产生新的 V6.5 selection exposure。

唯一两臂：

- `A0_snapshot_actor_pooling`：current Actor extent/centroid、route proximity、lateral/along 与 actor count；
- `A1_actor_time_history`：A0 加 method-visible swept extent、centroid motion、route-distance change、motion-route
  alignment、observed-hit fraction 与 swept/current volume ratio。

两臂同 seed=0、同 `32→16` MLP、120 epochs、matched safe coverage=40%。Positive train-only gate：actor-token
AUPRC gain `>=0.03`、selected outcome risk reduction `>=10%`、两个 eval scenes 都严格降低、真实 temporal features
优于 scene 内 shuffle。失败即关闭 actor-time family；成功也只允许另选未消费 fresh cohort，不得复用 P2 scenes。

迁移依据：PRECOG、M2I、GameFormer、VAD 与 Implicit Occupancy Flow；链接见 `FAILURE_ANALYSIS.md#v65-f03--fresh-task-risk-ranking-invariance`。
