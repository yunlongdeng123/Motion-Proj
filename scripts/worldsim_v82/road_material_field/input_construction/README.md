# 道路补全输入构造 API

- `case_constructor.construct_case_inputs`：像素提议、区域角色、连续视图及选图公共入口。
- `input_builder`：原分辨率信息隔离、概率保留与统一坐标采样。
- `shadow_probability.ShadowField`：对象关联阴影软分数、未校准不确定性、图像/目标/模型身份。
- `shadow_regions.apply_shadow_regions`：物理未知、外观污染、可信材质与许可写回分开；保留合法结构与测距。
- `shadow_bundle`：无损保存概率、观察图、模型条件和独立写回范围；不能用旧孔洞隐式代替写回许可。
- `shadow_composite`：从真实观察复制，许可范围内写预测；未决阴影默认拒绝标为最终结果。
- `FastSAM2Provider` 与 `make_proposal_reducer`：同一分割资格规则的 RAM 路径；框只作提示。
- `depth_adapter`、`lane_structure`：真实测距、几何外推和标线条件，来源分开。
- `input_utility`：规则选择与显式启用的学习选择器。
- `dual_feature_branches`：可选材质／对齐几何特征头；本轮未更改已有网络输入通道。

## 显式概率路径

```python
inputs = construct_case_inputs(
    actual_RGB_before_target_erasure, semantic, target_mask, foreground_provider,
    shadow_mode='object_probability', shadow_field=actual_model_prediction,
)
```

先用擦除前的当前图像和目标实例得到阴影预测，再构造补全主干的灰色未知区。缺失、歧义或目标不匹配的预测会中止该路径，不退回旧亮度规则，也不等价于无阴影。默认 `shadow_mode='legacy'` 保留旧实验供对照；不能把软概率塞进旧 `effect` 二值参数。

概率模式不使用固定车框比例裁断阴影范围。材质条件排除可能阴影与边界不确定性；可见标线和合法深度不因阴影而自动变成物理未知。`W_edit` 与 `target_unknown` 显式分开且 W⊆F；涉及保护对象的冲突或未决范围持续记账。低/高阈值只是冻结的初始工程政策，未经实域概率校准，不能宣称漏检率为零。

预测器与补全器的输入不同：前者需要可见车身；后者不能读车底隐藏道路。冻结阴影预测后才适用补全主干的核心RGB隔离检查。合成阴影alpha为光照混合强度，不等于阴影归属概率；仅在监督目录提供支持/强度/边界标签。

公开 SSISv2 的目标 mask 通过实例关联后处理使用，不是 MetaShadow 直接对象条件网络的复现。`from_ssis_instance_packet` 保留官方软实例图，按真实关联编号和目标软IoU匹配，不假设配对行顺序。官方 CLI 已从固定源码的训练增强代码核验 object=0、shadow=1，并显式传入类别角色；其他来源应先核验语义，再传 `object_class_id` 和 `shadow_class_id`，不要把重叠较大的阴影猜成车身。

实例关联不确定性只作为自动阴影写回的独立门控，不再广播为整幅图的像素不确定性下界。像素级 `4P(1-P)` 仅表示未校准的局部歧义。这个修复保持原 `W_edit` 不变，恢复与目标无关的可见道路材质资格，并不改善模型的阴影概率。`layers['rgb']`、观察 RGB 和 `ShadowField` 必须绑定同一原始观察图。

2026-11-16 GPU 对照与固定 200 步小样本细化探针见 `docs/autoresearch/object_shadow_gpu_20261116/`。细化模型的 7 通道输入为 RGB、目标实例、冻结预训练概率、可见道路与保留对象；仅用于本轮方法可行性验证，不替换默认路径，也不代表真实阴影漏检已经解决。

## 保守输入与独立输出许可

输入过滤可以容忍适度误报；精确阴影归属不作为所有输入遮罩候选的前置门槛。使用显式的 `independent_regions` 路径分别提供 `H_input`、`W_output` 与 `pending_output`，可选增加 `M_material` 和未承诺清除的 `remaining_effect`。

H 只扩大既有隐藏范围，材质资格只缩不增；W 独立给定，不能因 H 扩大而自动扩张。H\W 是只隐藏的上下文，不自动标成未决输出。独立合成严格执行 W 内使用原始解码、W 外复制真实原图、K 不可写；显式未决编辑仍会阻止最终输出。其 `completion_scope='authorized_output_only'` 不等于全部目标阴影已经清除。

`prepare_native_layers`、`apply_shadow_regions`、`construct_case_inputs` 支持可选字典 `independent_regions=None`；默认严格语义不变。已有 layers 可调用 `independent_regions.apply_independent_regions`，直接合成须显式 `region_mode='independent'`，保存后的独立 bundle 自动按其 schema 路由。

固定同 W 的 14 次道路补全对照与 50+41 项合同检查见 [输入／写回独立实验](../../../../docs/autoresearch/road_mask_decoupling_20261116/README.md)。详见 [早期 CPU 准备](../../../../docs/autoresearch/object_shadow_cpu_20261115/README.md)。代码中不包含权重和完整数据。
