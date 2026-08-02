# 人工审核检查表

评审者：`________________`

日期：`________________`

## A. 旧路线封存

- [ ] 最终结论准确写成“当前 strict-v2 可验证事件池过稀”，没有写成“nuScenes 没有 cut-in”。
- [ ] 675-scene、`1 PASS / 1 scene`、`n2_authorized=false` 等事实被完整保留。
- [ ] cut-in 被降为可选 demo，不再承担数据入口或论文前置条件。
- [ ] 旧报告、预注册、人工包、状态快照和失败教训仍可访问。

意见：

```text

```

## B. 清理边界

- [ ] 删除目标均在清单中以绝对路径列出。
- [ ] 三个 10 Hz cache 可由 raw data 与脚本重建。
- [ ] 三个 `checkpoint_15000.pth` 有既有 SHA-256，且相应 `checkpoint_final.pth` 保留。
- [ ] 两个 H1C dense render 副本是失败/修复前产物，最终 render、审计 JSON 和日志保留。
- [ ] raw nuScenes、正式指标、人工 verdict、失败总账、final checkpoints 未删除。
- [ ] 归档前完整 docs 快照与恢复路径存在。
- [ ] V7.1 审计目录内 20 个已索引 `.codexbak.*` 仍保留，删除的只是其余 41 个冗余工具备份。
- [ ] 没有发生 OOM/RC137。

意见：

```text

```

## C. Baseline 与数据

- [ ] AD-GS 作为主基线的理由是“官方协议完整且对象级可编辑潜力”，不是泛称最新 SOTA。
- [ ] 官方六 scenes、`10..69`、三相机、900×1600、60k 均已冻结。
- [ ] scene-0230 只做 pipeline gate，不用单场景数字伪称论文复现。
- [ ] 六场景三项复现带宽和失败停止规则明确。
- [ ] DGGT 被定位为 inference-only characterization，没有和 60-frame optimization 做虚假 matched 排名。
- [ ] 当前 side cameras/sweeps 缺失被记录，计划只做选择性提取，不全量解压 294 GB。

意见：

```text

```

## D. 环境与资源

- [ ] AD-GS、DPT、SAM、DGGT 使用隔离环境，不污染 `motionproj`。
- [ ] upstream commit、license、checkpoint hash、conda/pip/CUDA/gcc 都要求留档。
- [ ] compatibility patch 只处理安装/ABI，不能改方法或指标。
- [ ] 下一轮最低资源和磁盘安全余量合理。
- [ ] 内存/显存不足时立即 `blocked` 并等待，不杀用户服务、不静默缩协议。

意见：

```text

```

## E. 实验与 novelty

- [ ] 编辑不依赖 cut-in 语义，幅度和对象选择规则在看结果前冻结。
- [ ] 去遮挡真值分为 held-out observed / geometric support / unsupported。
- [ ] 2D inpaint 只作诊断，不作为 3D 世界状态。
- [ ] 同时评估对象区、时序、深度、身份、非目标保持和感知一致性。
- [ ] VAD-GS 是补密/可见性 claim 的强制对照。
- [ ] DenoiseGS、Perception-aware 3DGS、DrivingEditor、Real2Sim、GA-GS 的已有贡献没有被重命名。
- [ ] 只有 ≥3 scenes 重复失败才允许创建方法主张。
- [ ] method comparison 要求 matched setting、3 seeds、CI、worst-case、coverage 与 ABSTAIN。

意见：

```text

```

## F. 最终审核

- [ ] 文档内部链接可用。
- [ ] 当前状态、实验表和权威计划一致。
- [ ] 没有把计划中的环境/权重写成已安装。
- [ ] 没有未披露的训练、下载或 GPU 实验。
- [ ] 我已查看 `git diff` 或最终 commit。

建议结论：

```text
APPROVE | APPROVE_WITH_CHANGES | REJECT | UNCERTAIN
```
