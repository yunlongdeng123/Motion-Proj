# DrivingEditor 车辆关联阴影：2026-10-16 源码归档

本目录保存已执行的单序列实验实现、CPU/GPU诊断和本地对比页生成工具。
[实验记录与组件图](../../../docs/v82/DRIVINGEDITOR_SHADOW_SEQUENCE_20261016.md)；当前执行范围统一见 `docs/RESEARCH_STATUS.md`。

## 源码分工

| 目录 | 内容 |
|---|---|
| `diagnosis/` | 原高斯、车影、速度/显隐和局部贡献诊断；`audited_gaussian_model.py` 是审计时的上游参考快照 |
| `bc/` | 真实跨时刻参考、独立阴影、局部道路微调和历史消融；包含 sequence 所需的 `bc_model`、`run_experiments` 等依赖 |
| `sequence/` | 广域道路、真实 αT 归属、未选前景保护、分阶段微调、49帧评估、固定场景编辑、保存恢复验证 |
| `infra/` | 已有刚体/渲染基础设施和 2026-10-15 保存的附近车辆版本；影子分支没有改写原 checkpoint |

`sequence/comparison_template.html`、`build_comparison_entry.py` 和两个检查工具生成本地49帧对比页并追加旧统一入口。它们依赖保存图片和视频，不能直接作为无数据的 GitHub Pages 站点发布。

## 恢复到已有 AutoDL 数据目录

这些是实际运行的快照，保留原绝对路径与导出目录约定；不是新安装包。没有为本次 Git 归档重新训练或改写数值算法。
实际 campaign 根为 `/root/autodl-tmp/drivingeditor_repro/campaign/nearby_actors_20261015_v1`，Python 为 `/root/autodl-tmp/envs/drivingeditor/bin/python`。

| 本目录 | 原运行目录（相对于 campaign 根） |
|---|---|
| `infra/*.py` | `infra/` |
| `bc/*.py` | `shadow_BC_20261016_v1/implementation_v1/` 和 `implementation_deferred_v1/` |
| `sequence/*.py` | `shadow_sequence_20261016_v2/implementation/` |

恢复源码时先检查并备份目标文件；上面的映射不意味着可以覆盖正在使用的基础设施。`infra` 继承部分仅提供依赖参考，实际数据、环境和原先部署仍是复现实验的条件。

必要资产：原 `rigid_runs/scene0008_089_137/full30k/checkpoint_030000.pt`，真实49帧图像/位姿/标注，`rigid_specs`、`rigid_prepared`、`rigid_post30k/.../evaluation`；旧 BC 的 `training_data_v2`、`selection_v1`、`selection_plane_v1`、`runs/deferred_1000_v1/R4/addon_001000.pt`；新 `surface_guarded_data/surface.npz`、`guard_refinement_v2` 及 `unselected_foreground_audit/selection.pt`。

在 campaign 的 `shadow_sequence_20261016_v2/` 目录中，用全新的输出目录复评：

```bash
/root/autodl-tmp/envs/drivingeditor/bin/python implementation/evaluate_sequence.py \
  --output /path/to/new-unused-evaluation-directory \
  --restore unselected_clean_v1/final.pt
```

该命令需要已启动且空闲的 GPU；本次推送没有执行它。检查程序 `sequence/validate_checkpoints.py` 使用 CPU读取保存权重；`verify_shadow_math.py` 也需要已有观测与训练产物。

## 原始训练约定与边界

此独立道路/阴影 pilot 冻结原车身几何与SH外观，路面用显式 clean/observed 路由替代该局部分支的旧全局/静态 RGB 损失。完整刚体训练的 mask 内外监督默认规则仍适用于原训练器；本快照不是新的正式 DrivingEditor 损失默认。

原位40帧阴影伪标注和删车21帧子集都不等于配对编辑真值。参考已使用全部49帧，46帧梯度采样不构成未见帧验证。现有大角度资产破面和恢复范围外黑片保留为失败证据；人工质量 verdict 为 null。
