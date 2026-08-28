# P3L Instance-Evidence Local Geometry Head 迁移冻结

日期：2026-08-28

Task：`WS-V66-P3L-ACTOR-LOCAL-GEOMETRY-HEAD-01`

## 卡点

P2N显示Actor existence support充足时，natural actor-owned boundary仍可与target observed-FREE冲突；coarse
certificate recall=0，q0只有弱ranking signal。不能靠删Actor或在P10X扫阈值解决。

## 外部检索与最小迁移

- [Symphonies（CVPR 2024）](https://openaccess.thecvf.com/content/CVPR2024/html/Jiang_Symphonize_3D_Semantic_Scene_Completion_with_Contextual_Instance_Queries_CVPR_2024_paper.html)
  用instance queries与scene context缓解voxel-only几何歧义。迁移为Actor内primitive统计聚合，不搬其Transformer/backbone。
- [GaussianFormer](https://arxiv.org/abs/2405.17429)使用object-centric sparse Gaussians表达不同尺度区域。迁移为
  Actor-local summary，不把所有匿名voxel混池。
- [Cam4DOcc（CVPR 2024 official code）](https://github.com/haomo-ai/Cam4DOcc)强调4D occupancy与instance
  prediction。当前先用current/swept support ratio；完整temporal head留到有fresh sequence truth后。
- [Accurate Training Data for Occupancy Map Prediction Using Evidence Theory（CVPR 2024）](https://cvpr.thecvf.com/virtual/2024/poster/29493)
  指出常见LiDAR occupancy label本身存在质量问题并显式建模不确定性。迁移为local geometry conflict/Actor existence分层，
  UNKNOWN/contradiction不静默压成Actor artifact。

## 冻结实现

两级输出：

1. Actor existence certificate：继续由hit/track/provenance保护，不由P3L删除。
2. Local geometry head：只输出`p_local_conflict`，供REPAIR/ABSTAIN排序。

输入固定为8维：`q0_mean, q0_p90, log1p(boundary/hit/current/swept), hit/current, current/swept`。
模型固定2x32 ReLU MLP，seed0，full-batch weighted BCE；不使用scene ID、artifact family、label元数据或hazard。

数据角色：P10V 6 scenes训练；P10X 6 scenes单次selection。P10X已由P2N读过，只作为mechanism selection，不产生
fresh claim；通过后模型与feature冻结，在另一个scene-disjoint V65 cohort一次confirmation。

Selection只判ranking增量：相对constant deterministic baseline AUROC `+0.03`、AUPRC `+0.05`，至少5个有双类scene的
AUROC `>0.5`。不在selection扫threshold；Actor existence/ID不受模型输出影响。
