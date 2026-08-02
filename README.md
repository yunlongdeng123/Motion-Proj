# Motion-Proj

Motion-Proj 当前研究主线是**动态驾驶场景可编辑重建与失败诊断 V2**：先补齐前馈重建对照和对象级可编辑
基线，再用真实轨迹修改、遮挡重算、去遮挡与非目标区域保持实验定位跨场景稳定失败。项目不会预设必须产生
新方法；只有真实失败、可验证 endpoint 与独立 novelty delta 同时成立，才进入方法实现。

## 当前状态

- 权威状态：[`docs/RESEARCH_STATUS.md`](docs/RESEARCH_STATUS.md)
- 权威计划：
  [`docs/DYNAMIC_DRIVING_EDITING_DIAGNOSTIC_PLAN_V2.md`](docs/DYNAMIC_DRIVING_EDITING_DIAGNOSTIC_PLAN_V2.md)
- 实验台账：[`docs/EXPERIMENTS.md`](docs/EXPERIMENTS.md)
- 失败与防重复账本：[`docs/RESEARCH_FAILURES.md`](docs/RESEARCH_FAILURES.md)

V1 已作为历史终态冻结：AD-GS 六场景 exact reproduction 完成，官方 test mean 为
`PSNR 31.174515 / SSIM 0.927661 / LPIPS(VGG) 0.163489 / coverage 6/6`；DGGT 当时阻塞于
pointops2 的 PEP 517 build isolation；V1 没有执行真实对象编辑。V2 不覆盖这些事实，也不恢复已拒绝的
cut-in 挖掘或“补身份 + 基础轨迹编辑”贡献。

## V2 执行顺序

```text
M0 事实源与环境镜像基线
→ M1 DGGT 修复与推理对照
→ M2 nuScenes 真值 actor 评测适配器
→ M3 对象级可编辑基线
→ M4 scene-0230 真实编辑闭环
→ M5 三场景编辑与去遮挡压力测试
→ M6 基于真实失败的创新性门禁
→ M7 条件式方法与 matched ablation（仅 M6 done 后）
→ M8 人工盲审与最终裁决（仅有真实方法结果后）
```

每个里程碑使用独立 task ID、run 目录和 Git commit；`pending/running/blocked/done/rejected` 是唯一允许的
阶段状态。M7/M8 是条件阶段，不能用占位结果越过门禁。

## 环境

基础开发环境位于数据盘：

```bash
source /root/miniconda3/etc/profile.d/conda.sh
conda activate /root/autodl-tmp/envs/motionproj
cd /root/autodl-tmp/motion_proj
```

V2 的缓存、镜像与网络 smoke 统一由项目脚本配置：

```bash
source scripts/bootstrap_autodl_v2.sh
```

脚本使用项目级 `configs/env/autodl_condarc_v2.yaml`，不会执行 `conda init`，也不会改写
`~/.condarc` 或全局 pip 配置。大型环境、checkpoint 和缓存一律放在 `/root/autodl-tmp`。

## 代码与证据边界

- `motion_proj/dynamic_editing_v2/`：V2 actor 真值适配、投影、选择与评测模块；
- `motion_proj/resim/`：可复用的 WorldState、actor registry、DriveStudio adapter 与 typed render 基础设施；
- `motion_proj/dynamic_recon/`：V1 动态重建审计代码，只作历史证据或窄接口复用；
- `/root/autodl-tmp/runs/dynamic_editing_v2/`：V2 正式 run、日志、指标、资源记录与终态；
- `docs/archive/`：历史计划和结论，只作证据，不构成当前执行授权。

更完整的第三方环境、资产与保留策略见：

- [`docs/ENVIRONMENT.md`](docs/ENVIRONMENT.md)
- [`docs/THIRD_PARTY.md`](docs/THIRD_PARTY.md)
- [`docs/MACHINE_MIGRATION.md`](docs/MACHINE_MIGRATION.md)
- [`docs/ARTIFACT_RETENTION.md`](docs/ARTIFACT_RETENTION.md)
