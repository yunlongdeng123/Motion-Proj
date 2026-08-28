# P6 Actor-preserving HARP Bake Freeze

Task：`WS-V66-P6-HARP-BAKE-01`

P6复用P3C canonical continuous local-geometry scores与V65 P2V observable method evidence，输出plan约定的八个
runtime文件。bake只读取`base_id/scene/q0_mean/p_local_conflict`，不读取P3C target label；输出也不包含
`local_geometry_conflict/hidden_free_count/hidden_free_rate/target_label`。

由于P3L/P3C未选择binary threshold，P6不暗中补阈值。每个supported Actor均保留，local action固定为
`RANK_REPAIR_OR_ABSTAIN`；本阶段不实际修改geometry。static state与Actor current-envelope primitives分别存储；Actor
元数据包含class、track/lifecycle和采样trajectory。hazard attributes只记录同步轨迹的proximity/closing-speed候选，明确
`controls_actor_existence=false`。

Runtime manifest固定：不加载learned model、不加载hidden target、hazard不控制Actor existence、Actor/static与
physical/appearance分层。首轮只裁决package capability与上述权限边界，不做重复replay、smoke/regression matrix、
hash/checksum/fingerprint或fresh quality claim。
