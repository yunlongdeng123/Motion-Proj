# WorldSim V5 M1 附录索引

本目录保存可直接用于技术报告附录的轻量索引。重型 checkpoint、mask、NPZ 与 source snapshot 不复制到 Git；它们保留在文中给出的不可变 formal run 目录，并由 SHA-256 绑定。

## 当前收录

- `docs/WS_V5_M1_FORMAL_BASE_UNARY_DIAGNOSTIC.md`：r027–r034 30k base 表、r035 audit、r036 frozen SAM、r037 unary、r038 physical graph 指标与解释边界。
- `M1_R035_R037_METADATA.json`：机器可读的 run 路径、分母、指标、哈希和裁决。
- `M1_R038_GRAPH_METADATA.json`：r038 graph 候选、bandwidth、arm delta、topology leakage 与完整哈希。
- `docs/WS_V5_M1_DEVELOPMENT_REPLICATION.md`：r039–r046 三场景 SAM/unary/graph 复制、六单元门槛、基础设施阻塞与解释边界。
- `M1_R039_R046_REPLICATION_METADATA.json`：机器可读的复制选择、run/hash、六单元 delta、topology 与 gate 裁决。
- `docs/WS_V5_M1B_BOUNDARY_RESIDUAL_FORENSICS.md`：3px boundary residual 分解、六单元 primary gate 与 M1B 不解锁结论。
- `M1B_R001_BOUNDARY_RESIDUAL_METADATA.json`：r001 的机器可读 cell 指标、gate、run 与完整哈希。
- `docs/RESEARCH_FAILURES.md`：V5-F20–F33，包含稀疏分母、复制失败、boundary enrichment 误读与 M1B 不解锁限制。

## 当前裁决

`m1_rejected_graph_replication_failed_boundary_ambiguity_not_primary`

M1/M1B 已停止；不得覆盖 r035–r046 或 M1B r001，不得改写指标或恢复 semantic split。validation/test/KITTI quality 仍未读；V5 后续转入 M2。
