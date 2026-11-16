# v44 代码更新（2026-11-16）

本次将已验证工作区的算法源码同步到 `research/worldsim-v8.2-drivingeditor-v44`。提交包含共享场景编辑、道路材质与训练、自动补全输入、对象关联阴影、独立输入遮罩/写回范围，以及持久道路与局部静态不透明度交接；补齐结构条件、颜色残差场、适配器及多街段数据准备源码。

```mermaid
flowchart LR
  I[图像、标定与实例观测] --> M[输入遮罩与材质资格]
  M --> G[固定道路补全模型]
  G --> W[世界坐标道路材质]
  W --> H[局部不透明度交接]
  H --> R[统一三维渲染]
  K[保留对象与背景约束] --> H
```

## 入口

- [三维编辑与持久场景](../../scripts/worldsim_v82/drivingeditor_dynamic_edit/README.md)
- [道路材质与训练](../../scripts/worldsim_v82/road_material_field/README.md)
- [自动输入构造](../../scripts/worldsim_v82/road_material_field/input_construction/README.md)
- [最新闭环方法、实验边界与复现脚本](../autoresearch/scene065_closedloop_20261116/README.md)

持久场景由场景清单、道路资产、合并后的不透明度检查点和来源凭据加载：

```python
from scripts.worldsim_v82.drivingeditor_dynamic_edit.persistent_road_handoff import load_persistent_handoff

scene, manifest, receipt = load_persistent_handoff(scene_json, proof_json)
output = scene.render(frame_key, delete_actor_ids=(target_actor_id,))
```

原始高斯引擎、检查点、Waymo 数据、分割/扩散预训练权重由清单或配置指向外部位置。代码更新不包含这些资产、视频、逐帧渲染或特征缓存。多街段数据准备器的 `--decoder-root` 可指向仓库中既有的 `scripts/worldsim_v82/v43_ownership_20261028/v43_decision_r4/actor_diagnostics_20261101`；不需要复制外部解码脚本。

`shared_road_material.py` 保留旧固定父子拓扑的兼容用途，不作为任意道路的通用表示接口。新持久场景加载器不改变默认 factory，也不会因输入遮罩扩大而直接删除三维高斯。

## 验证与边界

CPU 回归：214 项测试及 192 项子测试通过；两个源码包全部 Python 文件通过语法解析。最近一次真实 GPU 实验已验证持久保存/重载与全199帧渲染，原模型和旧现场保留。

065 删除结果仍有明显车形残留与道路支撑缺口。本次是代码发布，不代表编辑质量通过，也不代表已证明跨场景自动泛化。人工验收结论未填写。
