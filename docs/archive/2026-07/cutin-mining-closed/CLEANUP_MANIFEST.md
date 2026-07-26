# cut-in 路线清理清单

- 执行日期：2026-07-26
- 执行范围：仅删除可再生的缓存、中间 checkpoint、失败渲染副本、临时 worktree 和工具备份。
- 恢复原则：正式报告、失败记录、配置、指标、人工结论、最终 checkpoint、原始 nuScenes 数据全部保留。
- 归档前文档备份：`/root/autodl-tmp/motion_proj_backups/docs-before-direction-pivot-2026-07-26/`
- 执行时资源约束：cgroup `memory.max=2147483648`；不运行 Python 扫描、conda 求解、预处理、训练或评测。

## 删除项

| 类别 | 绝对路径 | 删除前大小 | 可恢复性 |
|---|---|---:|---|
| 可再生 10 Hz 缓存 | `/root/autodl-tmp/data/occgs/processed_10Hz/n1_kinematic_v1` | 747,218,808 B | 可由原始 nuScenes 与已保留脚本/配置重建 |
| 可再生 10 Hz 缓存 | `/root/autodl-tmp/data/occgs/processed_10Hz/n1_receiver_cutin_v1` | 766,197,929 B | 同上 |
| 可再生 10 Hz 缓存 | `/root/autodl-tmp/data/occgs/processed_10Hz/n1_receiver_cutin_final_v1` | 724,281,188 B | 同上 |
| B0 中间 checkpoint | `/root/autodl-tmp/runs/occgs_resim/b0_recon/occgs_b0/b0_2_s0_3cam8s/checkpoint_15000.pth` | 661,944,310 B | 保留同 run 的 `checkpoint_final.pth`；哈希见下 |
| B0 中间 checkpoint | `/root/autodl-tmp/runs/occgs_resim/b0_recon/occgs_b0/b0_3_s1_3cam8s/checkpoint_15000.pth` | 894,981,174 B | 同上 |
| B0 中间 checkpoint | `/root/autodl-tmp/runs/occgs_resim/b0_recon/occgs_b0/b0_4_s2_3cam8s/checkpoint_15000.pth` | 836,441,718 B | 同上 |
| H1C 失败渲染副本 | `/root/autodl-tmp/runs/occgs_resim/v71/V7-H1-11C/render_labels_failed_depth_order` | 477,682,592 B | 保留最终 `render_labels`、失败审计 JSON 和日志 |
| H1C 修复前渲染副本 | `/root/autodl-tmp/runs/occgs_resim/v71/V7-H1-11C/render_labels_pre_state_evidence_fix` | 428,676,206 B | 同上 |
| 临时 worktree | `/root/autodl-tmp/motion_proj_final_clean_a74c55a` | 约 6 MiB | Git commit 可恢复 |
| 临时 worktree | `/root/autodl-tmp/motion_proj_final_clean_d88d5e2` | 约 6 MiB | Git commit 可恢复 |
| 临时 worktree | `/root/autodl-tmp/motion_proj_final_clean_7104f5c` | 约 6 MiB | Git commit 可恢复 |
| 临时 worktree | `/root/autodl-tmp/motion_proj_final_clean_3a548c2` | 约 6 MiB | Git commit 可恢复 |
| 临时 worktree | `/root/autodl-tmp/motion_proj_final_clean_beee1de` | 约 6 MiB | Git commit 可恢复 |
| 冗余文档工具备份 | `/root/autodl-tmp/motion_proj/docs/**/*.codexbak.*`，排除 `archive/2026-07/v7.1-h1-reject/codex-backups/` | 41 files / 1,366,426 B | 完整归档前快照和 Git 历史可恢复 |

前三类已知精确大小合计为 `5,537,423,925 B`（约 `5.16 GiB`），另加临时 worktree 与文档工具备份。

## 已删除中间 checkpoint 的既有 SHA-256

以下哈希在删除前已经记录于：
`/root/autodl-tmp/runs/occgs_resim/V7_EVIDENCE_INDEX.json`

```text
f97395a94643ea6bde029de801cfee43576ed825097283313ce4594d16636b64  b0_2_s0_3cam8s/checkpoint_15000.pth
667b7c1384b4331885c33569f6823ad39a1f51b94634fa18d0906bf336d4eaf3  b0_3_s1_3cam8s/checkpoint_15000.pth
fff864d18c3ae7dd831d7cd182eedfeb4e2083ed0a4b0a3079441ba5b64f3517  b0_4_s2_3cam8s/checkpoint_15000.pth
```

## 明确保留

- `/root/autodl-tmp/data/nuscenes/` 原始数据；
- `/root/autodl-tmp/data/occgs/processed_10Hz/mini`；
- `/root/autodl-tmp/data/occgs/processed_10Hz/trainval_annots`；
- 所有 `checkpoint_final.pth`；
- `/root/autodl-tmp/runs/occgs_resim/V7_EVIDENCE_INDEX.json`；
- V7/V7.1 正式指标、配置、日志、人工审核证据和失败审计；
- 最终 H1C `render_labels`；
- `docs/RESEARCH_FAILURES.md` 及全部历史归档。

## 执行后验证

执行结果：

- [x] 所有清单内删除目标均不存在；
- [x] 所有清单内保留目标仍存在；
- [x] Git 只剩主工作树 `/root/autodl-tmp/motion_proj`；
- [x] `RESEARCH_FAILURES.md` 和归档文档可读；
- [x] `df -h /root/autodl-tmp` 从约 `67G used / 62G avail` 变为 `62G used / 67G avail`，回收约 5 GiB；
- [x] cgroup `memory.events` 在执行后为 `oom 0 / oom_kill 0`，未发生 OOM/RC137；
- [x] 删除 41 个冗余 `.codexbak.*`，合计 1,366,426 B；恢复并继续保留 V7.1 审计目录内 20 个已索引备份（282,326 B）。

执行后文件系统快照：

```text
Filesystem      Size  Used Avail Use% Mounted on
/dev/md0        128G   62G   67G  49% /root/autodl-tmp
```
