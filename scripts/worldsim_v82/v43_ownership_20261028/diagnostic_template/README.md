# V43 通用GPU诊断模板

每个case共用这些模板；必须另导出scene/cloud/identity/edit_geometry/foreign_clouds/semantic_observations。GPU语义放gpu_semantics/index.js及frame_*.js；深度索引可选。v43_groups.js负责已实际完成新节点，未完成时保持空文件。新旧行ID不同命名空间，不能用原30K 0/1/U覆盖新Actor局部种子；二维预测与原owner分别保存。GPU map为Cityscapes19+unknown，旧CPU map是独立版本。含目标橙框与其他原始车辆蓝框；没有RGB三维背景底图。每次build后做路径、图像和实际Canvas检查。
