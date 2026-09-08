# WorldSim V7.3 阶段技术报告

这是研究进行中的英文LaTeX草稿，便于后续arXiv写作；没有对外发布，不代表V7.3完成。r2同步R11完整原生控制负结果、R13真实一轮输入训练及简明组件图。R10/R14仍运行，R12、完整新cohort训练、20新日志确认未完成；中文当前状态仍以`docs/RESEARCH_STATUS.md`为准。作者信息与最终结论尚未定稿。r1 PDF保留旧时间快照，不静默覆盖历史。

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

或使用现有LaTeX环境的`latexmk -pdf main.tex`。不需要BibTeX、shell escape或新环境；图使用现有Matplotlib导出PDF/SVG/PNG。组件图以模块/箭头/少量标签表达实际结构，不把用户示例中的BEV/occupancy模型当作本项目方法。表格来自已保存日志统计，生成过程不重跑模型、不改变分母。阶段表保留已完成R5/R11与固定强对照，不把正在运行的实验填为零。新结果完成时更新正文、表格源和三本研究台账，再重新编译。

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
