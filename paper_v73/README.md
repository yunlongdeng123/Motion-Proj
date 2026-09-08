# WorldSim V7.3 阶段技术报告

这是研究进行中的英文LaTeX草稿，未对外发布。r3同步已完成R10/R12/R14三角、R11锚点与Q-v2共享顶点候选；Q-v2正式训练已启动，结果pending，20新日志质量未读。主表14个完成方法，新增三角配对表与Q-v2实际组件图，保留R5恢复及移动子集/输入覆盖/密度边界。r1/r2 PDF作为历史保留；中文当前状态以docs/RESEARCH_STATUS.md为准。

在仓库根目录生成真实归档表格：

```bash
python scripts/export_worldsim_v73_interim_tables.py \
  --evidence-root docs/autoresearch/worldsim_v73 \
  --output paper_v73/tables
python scripts/plot_worldsim_v73_architecture.py \
  --output paper_v73/figures/architecture_components
```

在`paper_v73/`编译：

```bash
tectonic main.tex
```

或使用现有LaTeX环境的`latexmk -pdf main.tex`。不需要BibTeX、shell escape或新环境；图使用现有Matplotlib导出PDF/SVG/PNG。组件图以模块/箭头/少量标签表达实际结构，不把用户示例中的BEV/occupancy模型当作本项目方法。表格来自已保存日志统计，生成过程不重跑模型、不改变分母。阶段表保留所有已完成对照，不把Q-v2等正在运行的实验填为零。新结果完成时更新正文、表格源和三本研究台账，再重新编译。

核心证据路径：

- `docs/autoresearch/worldsim_v73/m2/global/population_joint_r5_analysis.json`：全489分母、开发75对象/5独立日志与配对差值。
- `docs/autoresearch/worldsim_v73/m2/global/actor_tsdf_r1_analysis.json`：TSDF支持可用性与同输入基线。
- `docs/autoresearch/worldsim_v73/m2/global/population_native_r11_analysis.json`：完整原生DPT控制与同分母日志配对。
- `docs/autoresearch/worldsim_v73/m2/global/visual_only_r13_training_counts.json`：41次新输入反向、5次envelope-only与唯一开发归属回波边界。
- `docs/autoresearch/worldsim_v73/m4/observed_points_r1_summary.json`：可评价返回点P/R/F及条件Chamfer边界。
- `docs/autoresearch/worldsim_v73/m4/vdb_scene_r9_vs_pca_background.json`：nuScenes背景切换的侵入与覆盖代价。
- `docs/autoresearch/worldsim_v73/m4/av2_old_vdb_scene_r4_paired.json`：已曝光旧AV2开发日志的背景比较。
- `docs/WORLDSIM_V7_3_COMPARISON_PROTOCOLS.md`：标签、预算、适配范围、密度与跨数据集比较边界。

原始checkpoint、JSONL与完整表面在AutoDL的`/root/autodl-tmp/runs/worldsim_v73/`，代码与紧凑证据在v73远端分支。普通Git提交引用用于回溯；不添加额外哈希/校验和/指纹。

新增证据：`m2/global/population_joint_r12_analysis.json`（包含R10/R14/R8），`population_native_r14_analysis.json`，三角完整中文报告与`WORLDSIM_V7_3_QV2_SHARED_MESH.md`。Q-v2图由`scripts/plot_worldsim_v73_qv2_architecture.py`生成并复制到本目录`figures/qv2_components.pdf`；固定拓扑可微检查不作为重建成功。
