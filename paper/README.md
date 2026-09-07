# EAS 论文与本地研究档案

> **当前研究状态（2026-09-07）**：本目录是 V7.1 EAS 的既有证据稿；下文“当前主稿”指这次历史构建。新方向为 [EAS-VGGT](../docs/WORLDSIM_V7_2_EAS_VGGT_RECOVERY_PLAN.md)，实现/实验 E1–E5 尚未完成。本轮不改写论文结果、不重建 PDF；当前执行入口见 [RESEARCH_STATUS](../docs/RESEARCH_STATUS.md)。

当前主稿：**Evidential Actor Surfaces for Physically Consistent Driving Reconstruction**。
科学证据快照为 `research/worldsim-v7.1-learned-evidential-surface@1913ab0e`，本次写作整理覆盖 V6--V7.1，未执行新实验。

## 阅读入口

- [中文精翻 main 与阅读包](../docs/paper/README.md)：统一存放于 `docs/paper/`，对应 `ec9c2e08` 的英文主稿。
- `main.pdf`：当前 EAS 主线。保留 M7/M8 几何、M39 同几何证据组合、有限衰减理论、刚体/视觉所有权，以及必要的分层与 AV2 负结果。
- `supplement.pdf`：仅供本地查看的完整研究档案，不提交 OpenReview。含复现细节、证明、V6--V6.7 演进、V7.1 实验图谱、历史 V7 结果与完整相机证据。
- `CONTRIBUTION_MAP.md`：当前贡献与证据对应；其后保留的 HARP-3D 映射仅为历史记录。
- `sections/supp_guide.tex`：阅读导航与历史口径更正。
- `results/eas_evidence.json`：从 canonical M8/M39/M43/M49/M51 summary 摘录的精确数值及来源。`results_macros.tex` 保留旧研究数值，不覆盖既有 run。

当前使用官方 CVPR 模板的带页码本地阅读模式，无伪造投稿编号。模板来源见 `TEMPLATE_PROVENANCE.md`。
写作参考用户提供的 Gau-Occ 与 DynamicVGGT：聚焦问题、方法、证据，旧探索集中归档。未复制参考文献中的图或实验结果。

## 构建与核对

在有 TeX Live/TinyTeX 的机器上串行执行：

```bash
latexmk -pdf -interaction=nonstopmode -halt-on-error main.tex
latexmk -pdf -interaction=nonstopmode -halt-on-error supplement.tex
python scripts/verify_paper.py
```

核对脚本仅检查本地源文件、39 个表格数值、分层计数与编译日志，不加载数据集或模型。
PDF 另需逐页渲染检查；不要仅用编译成功代替排版验收。
本次在 Windows 本地编译，AutoDL 无卡 2 GB 内存环境只承担串行读取与文件同步。

## 证据解释与投稿边界

M43 正式 categorical 比较及拒绝结论保持不变。只读代码审计发现描述性
`m8_point_surface` 的 early/hit 基线字段被 categorical 计数覆盖；不得将该混算子差值解释为匹配的几何迁移实验。
Chamfer 配对未受影响，canonical run 不修改、不重读 target。详见统一失败账本 `V71-F52`。

`supplement_v7_legacy.tex` 保存旧入口；`sections/supp_v7_restored.tex` 将其完整正文以独立标签空间纳入当前档案。
`arxiv.tex` 仍与 main 共享正文，但真实作者信息、正式投稿模板与元数据需要单独配置；本次未构建或发布 arXiv。
`SUBMISSION_CHECKLIST.md` 的旧 V7 页数、分支及 review 模式仅为历史记录，不能直接作为当前投稿清单。
