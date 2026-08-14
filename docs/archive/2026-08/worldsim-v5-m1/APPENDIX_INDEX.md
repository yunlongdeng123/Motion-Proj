# WorldSim V5 M1 附录索引

本目录保存可直接用于技术报告附录的轻量索引。重型 checkpoint、mask、NPZ 与 source snapshot 不复制到 Git；它们保留在文中给出的不可变 formal run 目录，并由 SHA-256 绑定。

## 当前收录

- `docs/WS_V5_M1_FORMAL_BASE_UNARY_DIAGNOSTIC.md`：r027–r034 30k base 表、r035 audit、r036 frozen SAM、r037 unary、r038 physical graph 指标与解释边界。
- `M1_R035_R037_METADATA.json`：机器可读的 run 路径、分母、指标、哈希和裁决。
- `M1_R038_GRAPH_METADATA.json`：r038 graph 候选、bandwidth、arm delta、topology leakage 与完整哈希。
- `docs/RESEARCH_FAILURES.md`：V5-F20–F28，包含 intersection inflation、arm collapse、opacity、actor representation、timeline、FN tradeoff、单场外推与 proxy leakage 限制。

## 当前裁决

`physical_graph_direction_supported_single_scene_small_effect_replication_required`

下一次追加只能使用新 run ID；不得覆盖 r035/r036/r037/r038 或改写其指标。validation/test/KITTI quality 仍未读。
