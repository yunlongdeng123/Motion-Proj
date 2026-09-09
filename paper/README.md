# 可查询规范刚体表面：V7.3 研究稿

当前入口 main.pdf / main.tex 已于 2026-09-10 更新为 **Queryable Canonical Surfaces from Sparse Driving Observations**。这是证据与方法开发稿，使用 CVPR 模板作本地阅读；不宣称已满足投稿页数或完成全部方法验证。

## 本次内容

- 完成后的 joint r7：75 个 DEV Actor、5 个日志的配对结论与算力账。正确首交点相对 r6 提升 10.536 pp，free-space 保持仍未建立。
- 七种原始保存表面的真实 Blender 全景与责任面片隔离图、精确三角面切片、首面/任意正确交点/邻近表面 Oracle 阶梯，以及共享网格剪枝干预。
- 任务重定义为稀疏观测到可查询规范刚体表面；方法中给出显式 chart 支持域、正观测保留约束、首面 Jacobian、入射归一化正交更新和稀疏自定义伴随。
- 严格区分已训练的 r3–r7、已通过数值验证的算子原型、尚未实现训练的支持域边界。完整主张与反例见 V73_REVISION_NOTES.md。

## 构建

    cd paper
    latexmk -pdf -interaction=nonstopmode -halt-on-error main.tex

本次以 Windows TinyTeX 2026 构建，并将同一 PDF 同步回 AutoDL。验收包含逐页渲染、所有图表与引用核对；无未定义引用、溢出框或过大浮动体警告。已生成的图表位于 figures/v73/ 与 tables/v73/，编译不需要训练环境。

scripts/v73_build_figures.py、scripts/v73_compose_blender_atlas.py 是派生图表源码。复现包提供 paper/、evidence/、forensics/、forensics_r7/ 的 staging 布局；设 WORLDSIM_PAPER_STAGE 指向包根目录后运行这两个脚本。NumPy、Matplotlib 与 Pillow 即可重绘图表，不需要模型推理。

原始几何取证与渲染脚本位于项目根目录 scripts/forensic_worldsim_v73_paper_surfaces.py、scripts/forensic_worldsim_v73_paper_r7.py、scripts/render_worldsim_v73_paper_blender.py。渲染使用官方 Blender 3.6.23、Cycles CPU；Debian 精简构建缺少 OpenColorIO，未用于最终图。每例提供全景和隔离责任面的 .blend。可直接用 Blender 打开查看。隔离只影响展示，指标与切片始终查询原始完整网格。

证据目录：

    docs/autoresearch/worldsim_v73/paper_forensics/
      20260909T184200Z__saved-surface-oracles-r1/
      20260909T190600Z__saved-r7-surface-oracles-r1/

首面算子数值原型为 scripts/worldsim_v73_witness_operator_prototype.py，检查结果在第一目录的 operator_check.json。该原型不接入 r7，不把 DEV 局部试步解释为学习收益。新方案与既有负结论 V73-F02/F03/F04/F09 的关系由统一台账维护，本目录不建立平行 failure ledger。

## 历史与边界

旧 V7.1 的 sections、supplement 与文档保留作为历史材料；旧 scripts/verify_paper.py 检验的是旧稿，不能验证本稿。paper_v73/ 是较早 r4 阶段入口。本次主入口统一为 paper/main.pdf。

参考 DVGT 的任务与输出共同重构、LiDAR-RT 的 representation/renderer 联合设计，以及 FoundationGeo 的 Oracle 诊断写法。本文不声称复现 LiDAR-RT 数值，不把 AdaPoinTr 的 PCA 曲面转换误当成其原生输出。

证据截止时，固定 20 AV2 日志的独立确认由原研究任务执行中；本文未读取部分外部质量来调参，也未将其标成完成。外部确认、完整场景组合、新支持域训练及正交更新端到端收益仍需独立报告。
