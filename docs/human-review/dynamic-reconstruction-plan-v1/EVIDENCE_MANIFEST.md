# 证据清单

## 1. 当前事实源

| 文件 | 用途 |
|---|---|
| `docs/RESEARCH_STATUS.md` | 当前路线、授权、已完成和下一门禁 |
| `docs/RESEARCH_FAILURES.md` | 全历史失败与 `PIVOT-F01..F05` |
| `docs/EXPERIMENTS.md` | 当前 run 注册表 |
| `docs/DYNAMIC_DRIVING_RECONSTRUCTION_PLAN_V1.md` | 环境、数据、实验、统计、停止和人审完整方案 |
| `docs/ARTIFACT_RETENTION.md` | 当前保护范围与删除前置条件 |

## 2. cut-in 历史

| 证据 | 路径 |
|---|---|
| 路线封存索引 | `docs/archive/2026-07/cutin-mining-closed/README.md` |
| 精确清理清单 | `docs/archive/2026-07/cutin-mining-closed/CLEANUP_MANIFEST.md` |
| cut-in 结束状态快照 | `docs/archive/2026-07/cutin-mining-closed/RESEARCH_STATUS_CUTIN_FINAL_SNAPSHOT.md` |
| cut-in 实验台账快照 | `docs/archive/2026-07/cutin-mining-closed/EXPERIMENTS_CUTIN_FINAL_SNAPSHOT.md` |
| 最终报告 | `docs/archive/2026-07/cutin-mining-closed/N1_CUTIN_FINAL_REPORT.md` |
| 最终稀疏审核说明 | `docs/archive/2026-07/cutin-mining-closed/N1_CUTIN_FINAL_SPARSE_HUMAN_REVIEW.md` |
| 最终人工材料 | `docs/archive/2026-07/cutin-mining-closed/n1-cutin-final-resource-rejection-human-review/` |

归档前完整 docs 快照：

```text
/root/autodl-tmp/motion_proj_backups/docs-before-direction-pivot-2026-07-26/
```

## 3. 清理结果

```text
删除前三类精确大小合计：5,537,423,925 B
另删除临时 worktree 与 41 个冗余 .codexbak.*（1,366,426 B）
V7.1 审计目录内 20 个已索引 .codexbak.*（282,326 B）已恢复并继续保留
磁盘：约 67G used / 62G avail → 62G used / 67G avail
memory.events：oom 0 / oom_kill 0
```

明确保留：

```text
/root/autodl-tmp/data/nuscenes/
/root/autodl-tmp/data/occgs/processed_10Hz/mini
/root/autodl-tmp/data/occgs/processed_10Hz/trainval_annots
/root/autodl-tmp/runs/occgs_resim/V7_EVIDENCE_INDEX.json
/root/autodl-tmp/runs/occgs_resim/**/checkpoint_final.pth
/root/autodl-tmp/runs/occgs_resim/v71/V7-H1-11C/render_labels
```

## 4. AD-GS 本地资产审计

完整 scene sensor chain 的 metadata/实际文件数：

| scene | FRONT | FRONT_LEFT | FRONT_RIGHT | LIDAR |
|---|---:|---:|---:|---:|
| 0230 | 234/40 | 230/0 | 230/0 | 387/40 |
| 0242 | 233/40 | 235/0 | 233/0 | 391/40 |
| 0255 | 233/40 | 233/0 | 233/0 | 390/40 |
| 0295 | 234/40 | 235/0 | 231/0 | 391/40 |
| 0518 | 235/41 | 237/0 | 235/0 | 395/41 |
| 0749 | 240/41 | 239/0 | 240/0 | 399/41 |

审计结论：当前 raw 不满足 AD-GS；下一阶段必须从本机只读官方 shards 选择性提取 side cameras 与 sweeps。

## 5. 官方来源

- https://www.nuscenes.org/nuscenes
- https://github.com/nutonomy/nuscenes-devkit
- https://github.com/JiaweiXu8/AD-GS
- https://www.openaccess.thecvf.com/content/ICCV2025/papers/Xu_AD-GS_Object-Aware_B-Spline_Gaussian_Splatting_for_Self-Supervised_Autonomous_Driving_ICCV_2025_paper.pdf
- https://github.com/xiaomi-research/dggt
- https://huggingface.co/xiaomi-research/dggt
- https://github.com/WangXu-xxx/DrivingEditor
- https://github.com/YikangZhang1641/VAD-GS
- https://ojs.aaai.org/index.php/AAAI/article/view/37640
- https://openreview.net/forum?id=PmQlMTBmpa
- https://github.com/TuojingAI/ReconDrive
- https://arxiv.org/abs/2605.13591
- https://arxiv.org/abs/2604.04331

## 6. 本轮未执行

```text
no conda solve
no new environment
no repository clone
no checkpoint download
no tar scan/extraction
no preprocessing
no training
no inference
no GPU run
```

最终文件 SHA-256 见同目录 `SHA256SUMS`；交付 commit 以远端 `git rev-parse HEAD` 为准。
