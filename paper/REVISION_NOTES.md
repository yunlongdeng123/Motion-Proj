# V7.1 论文整理记录（2026-09-05）

科学证据基线：`research/worldsim-v7.1-learned-evidential-surface@1913ab0e`。
此次只整理已有研究，不执行模型、数据集评测或 target 重读。

## 稿件分工

主稿以 Evidential Actor Surfaces（EAS）组织方法、实验与理论：canonical 几何、producer F/O/U evidence、
categorical ray return、有限衰减与物理/位姿/视觉所有权。保留 source 各分层表、M8 clear 回退及完整 M43
外测权衡，删除反复防御性说明；将旧 selective/reliability 主线与探索性正负结果移入本地研究档案。

档案含阅读目录、训练/评测复现、完整证明、V6--V6.7 研究演进、V7.1 milestone 图谱与详细结果、
原 V7 证书/区间/相机证据正文。当前绘制的总览图不引入新实验图像，其他图来自既有 frozen 结果。
`supplement.pdf` 不提交 OpenReview。未配置真实作者信息、投稿编号或公开发布。

## 本次更正

1. M8 的训练算子是 alpha-composited expected depth；literal beam-tube minimum 是评测算子。
   expected-depth pre-hit hinge 与 M38 的 integrated pre-hit optical thickness 分开描述。
2. M39 冻结经 transmittance 训练的 M35/M38 heads 后改变 composition；categorical joint fine-tuning 是被拒绝的 M40。
3. M43 从剔除 60 个已消费 logs 后的 90-log 补集取位置 `0,4,...,76`，不是直接从 150 logs 每五个取一个。
4. M43 描述性 M8 early/hit baseline 字段发生算子计数覆盖，不能用于独立几何迁移归因（统一账本 `V71-F52`）。
   正式 M39 categorical 比较、冻结判定与 Chamfer 配对不变；canonical run 不改写，runner 本次不修改。
5. 历史 P346 held-out horizon 修正为 artifact 对应的 2.5 秒；P199 保留模型为 full Gaussian copula。
   选择性修复的边际校准统计量不再误称被选择集合的条件风险保证。
6. 有限衰减证明补足正质量、严格衰减、归一化分母与 family-uniform 前提；CDF/median 等价写明插值与端点约定。

## 数值来源

`results/eas_evidence.json` 保存 canonical summaries 的精确摘录及来源路径。主要证据：

- M8：`WS-V71-M8-TEMPORAL-FRAME-COVERAGE-01/20260904T202000Z__m8-temporal-frame-s71110-r2`。
- M39：`WS-V71-M39-CATEGORICAL-AUTHORITY-COMPOSITION-01/20260905T070000Z__m39-categorical-authority-r1`。
- M43：`WS-V71-M43-M39-AV2-ZERO-SHOT-01/20260905T091500Z__m43-m39-av2-zero-shot-r1`。
- M49：`WS-V71-M49-VISIBILITY-SIGN-BOUNDARY-01/20260905T121500Z__m49-visibility-sign-boundary-r1`。
- M51：`WS-V71-M51-SOFT-HARD-FIRST-RETURN-DIAGNOSIS-01/20260905T063000Z__m51-soft-hard-first-return-r1`。

相对路径根为 `/root/autodl-tmp/runs/worldsim_v71/`；具体配置和旧版本凭证由 archive 与 `docs/EXPERIMENTS.md` 导航。

## 验收

- `main.pdf`：7 页，354,994 bytes；`supplement.pdf`：32 页，8,159,955 bytes。
- Windows 本地 TinyTeX：分别运行 `latexmk -pdf -interaction=nonstopmode -halt-on-error main.tex` 与 `supplement.tex`。
- `python scripts/verify_paper.py`：15 个几何表值、24 个 categorical 表值、两组分层计数核对通过。
- main：19 个标签、25 条被引文献；supplement：41 个标签、12 条被引文献。
  两份稿件缺失引用、重复标签、未解析引用及 overfull 均为 0。
- 最终 PDF 全页栅格化检查；总览图连线、长公式、表格、目录和十个相机案例未见截断或重叠。
- AutoDL 无卡 2 GB 环境只串行读取、备份和同步；未新开训练、评测、下载或 GPU 进程。

原始稿件可从 `1913ab0e` 恢复；未 push、未提交 OpenReview、未发布 arXiv。
