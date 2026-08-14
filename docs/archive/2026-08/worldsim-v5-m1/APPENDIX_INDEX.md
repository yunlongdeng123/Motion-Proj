# WorldSim V5 M1 附录索引

本目录保存可直接用于技术报告附录的轻量索引。重型 checkpoint、mask、NPZ 与 source snapshot 不复制到 Git；它们保留在文中给出的不可变 formal run 目录，并由 SHA-256 绑定。

## 当前收录

- `docs/WS_V5_M1_FORMAL_BASE_UNARY_DIAGNOSTIC.md`：r027–r034 30k base 表、r035 audit、r036 frozen SAM、r037 unary、r038 physical graph 指标与解释边界。
- `M1_R035_R037_METADATA.json`：机器可读的 run 路径、分母、指标、哈希和裁决。
- `M1_R038_GRAPH_METADATA.json`：r038 graph 候选、bandwidth、arm delta、topology leakage 与完整哈希。
- `docs/WS_V5_M1_DEVELOPMENT_REPLICATION.md`：r039–r046 三场景 SAM/unary/graph 复制、六单元门槛、基础设施阻塞与解释边界。
- `M1_R039_R046_REPLICATION_METADATA.json`：机器可读的复制选择、run/hash、六单元 delta、topology 与 gate 裁决。
- `docs/RESEARCH_FAILURES.md`：V5-F20–F32，包含 intersection inflation、arm collapse、opacity、timeline、FN tradeoff、单场外推、稀疏分母、SSH 断管、硬编码 denominator 与复制失败限制。

## 当前裁决

`physical_graph_development_replication_rejected_3of6_boundary_support`

下一次追加只能使用新 run ID；不得覆盖 r035–r046 或改写其指标。validation/test/KITTI quality 仍未读；semantic split 只在独立 boundary-residual forensic 证明条件成立后另行冻结。
