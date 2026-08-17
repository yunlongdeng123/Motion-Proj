# WorldSim V4 文档与临时产物清理清单

- 日期：2026-08-17
- 分支：`research/worldsim-v5-structdelta`
- 清理前 HEAD：`f7566beb4d37115700a1d702f524d99cbab24b4e`
- 状态：`done`
- 范围：只整理 V4 文档、移除编辑恢复副本并清理明确的 scratch/staging；不重跑实验、不修改研究结论。

## V4 归档确认

- 权威终局包：`docs/archive/2026-08/worldsim-v4-final/`；清理前后执行 `sha256sum -c SHA256SUMS`，均为
  `78/78 OK`。
- 终态保持：`M1 rejected / M2 done with geometry caveat / M3 confirmed`。
- 技术报告入口保持为 `worldsim-v4-final/TECHNICAL_REPORT_APPENDIX_INDEX.md`；M1 失败、M2 geometry caveat、
  M3 exact-once denominator 和 KITTI 历史 blocker 均未改写。
- `/root/autodl-tmp/runs/worldsim_v4/`、`V4_TEST_FREEZE.json`、`configs/worldsim_v4/`、V4 相关源码和测试未删除。

## 已删除范围

| 绝对目标 | 清理前规模 | 判定与恢复边界 |
|---|---:|---|
| `/root/autodl-tmp/motion_proj/tmp/*` | 4 files / 8,985,579 bytes | DPT/SAM smoke scratch；无 Git 跟踪，目录保留为空，可由 smoke 脚本重建 |
| `/root/autodl-tmp/tmp/*` | 58 files / 402,855 bytes | Python/W&B/ROMA 临时目录；路径保留为空，可由运行时重建 |
| `/root/autodl-tmp/mnt/` | 154 files / 126,234,111 bytes | 旧 V7.1 staging、nuScenes 小样本和 audit ZIP；仓库无路径引用，删除后不由 Git 恢复 |
| `docs/**/codex-backups/` | 6 dirs / 233 files / 13,423,361 bytes | 编辑过程副本；已跟踪部分可由 Git 历史恢复，研究事实由 canonical 快照/run 提供 |
| `docs/**/*.codexbak.*`（不含上一行） | 50 files / 3,459,065 bytes | 散落编辑副本；未跟踪内容永久删除，现行文档和 Git 版本为权威来源 |
| 项目根目录两个 `*.codexbak.*` | 2 files / 8,935 bytes | 旧 `AGENTS.md`/`README.md` 编辑副本；当前文件受 Git 跟踪 |
| 项目根目录误建目录 ` `、`;`、`cp` | 仅空目录树 | 历史 shell 转义产物；无文件、无引用 |

`/root/autodl-tmp/mnt/n1_positive_audit_evidence.zip` 清理前 SHA-256 为
`fbd65c645db00e3c2d7ba3b58b337c1e1b1b686dd1537153b8101ed6d85e9a3b`；整个 `mnt` 文件清单的
排序 SHA-256 串摘要为 `a2bc55cf1f8facf8c195e06c2e7c8a18e2c95a670e82d3c6ecaa4bcca3c95caa`。

## 明确保留

- `/root/autodl-tmp/motion_proj/work/codex-backups/`：1,647 files / 979,529,557 bytes。V4 失败账本明确引用
  `2026-08-12-adgs-r37-partial-scene0230`，本次不把它误判为普通 scratch；后续若要清理，需单独做 canonical
  manifest 引用审计。
- `/root/autodl-tmp/runs/worldsim_v4/` 与 `/root/autodl-tmp/runs/worldsim_v5/` 全部正式 run。
- KITTI 原始 ZIP、选择性抽取数据、nuScenes/processed scene、环境、checkpoint 和第三方依赖。
- `docs/archive/2026-08/worldsim-v4-final/` 全部 78 个受 SHA 清单约束的文件。

## 清理后验证

- `find docs` 对备份目录、`*.bak`、`*.codexbak.*`、`*.orig`、`*.rej`、编辑器 `~` 文件扫描为 `0`。
- 两个 `tmp` 目录存在且为空；`/root/autodl-tmp/mnt` 不存在。
- V4 final `SHA256SUMS`：`78/78 OK`；V7.1 H1 reject 清单按保留文件重建并全部通过。
- `git diff --check` 和文档链接/导航引用检查通过；没有训练、推理或高并发任务。
