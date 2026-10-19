# 请独立审查，不要把“工程检查通过”等同于效果通过

## 优先阅读

1. README.md / experiments.json：确认4组＋1基线，不把第二阶段或20车混进比较。
2. code/diffs/baseline_to_full.diff；code/overlays/full/utils/v2_static_partition.py、v2_runtime_context.py、v2_road_objective.py。
3. code/review_tools/world_road.py、prepare_real_inputs.py、audit_donor_provenance.py；evidence/input/bundle.json、targets/、actual_road_query_evidence.npz。
4. 各组effective_config与source_map/deployed_manifest；对照full_to_real/prior/ownership的diff，核对消融是否只改变预期道路开关。
5. compare.html静态→动态→完整重建。raw89/129/137的原始完整PNG在frames/samples，不要仅凭压缩视频或单帧下结论。

## 请回答

- 静态位置、时间opacity、时间宽度在训练/增密/剪枝/恢复/评估的各路径是否真的冻结？有没有通过尺度、SH、错误深度、遮挡顺序等继续解释车辆的通路？
- 车身RGB/depth/正则的监督路由和动态alpha完整mask是否一致？独立环境颜色detach有什么影响？哪些变化不是“仅道路”消融能隔离的？
- 3D框体积内静态opacity约束是否会漏掉框外/错误深度Gaussian，或误伤道路薄层？道路clearance/软边界/归一化单位是否匹配？不要建议将整个body mask的static alpha硬清零。
- Query是否准确反投影到同一个世界位置并以正确坐标投影到后续帧？遮挡、全部车辆、道路支撑、视角、深度/颜色一致性筛选有没有盲点？
- 融合来源为OBSERVED/PRIOR/UNKNOWN：真实曝光区域是否出现错误确认？供体是否把未观测部分变成固定颜色斑块或错误几何？49个冻结targets与query evidence是否一致？
- 道路loss是否仅回传静态，并与归属loss分开定义、仅在一次原生optimizer更新前累积？真实/先验关闭时其梯度是否实际为0？哪些UNKNOWN仍能经其他loss收到梯度？
- 2×2开关允许推断什么？覆盖区域不同、权重不同、单seed、额外3D标注和未知背景GT如何限制因果解释？
- baseline与ownership_only比较的是归属修复组合；不能把组合效果归因到内部某一个约束。原生基线有一次授权20K恢复，是否影响公平性应明确分析。
- 结合视频和完整PNG，分别指出可观测道路重建、先验纹理、未知区域、静态重复解释、动态主体保留、阴影的收益/回退。不要只用loss/alpha/节点完整性判干净。

## 输出期望

先按严重性列出有代码位置和证据的具体问题；区分确认事实、推断、缺失证据。给出最小修复建议和后续验证设计，不擅自增加训练组/预算。请保留两组独立渲染尚未完成这一事实；不存在的delete/alpha不能评成已通过。

## 输入说明

- bundle.json包含每帧world_to_camera、五身份3D框和道路target hash；targets/*.npz是实际冻结训练目标。保留原远端路径作为provenance，包内相对targets路径可读取。
- actual_road_query_evidence.npz为CPU几何/融合/来源证据，没有模型tensor。query_masks_a6_a37只有两车查询mask，不是五车总监督mask。
- frames/gt是原生评估导出的8-bit观测PNG；与独立导出GT的量化方法最大有1/255差异，不能假设跨导出文件hash相同。
- 原始源码/配置均保留。代码含绝对远端路径和外部依赖引用，不应直接运行训练入口。
