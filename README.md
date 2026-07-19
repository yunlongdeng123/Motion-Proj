# Motion-Proj：驾驶视频物理对齐研究基础设施

Motion-Proj 是一个面向驾驶视频扩散模型的研究代码库，覆盖动力学审计、几何投影、
Stable Video Diffusion（SVD）适配、实验运行时、独立评测，以及物理偏好关系的构造与校准。

仓库保留了两条已经完成诊断的研究路线：早期 dynamics projection distillation，以及后续
common-prefix sibling physics preference。它们提供了可复用的工程与评测基础设施，但当前均未形成
可以继续扩量训练或用于论文主张的配方。

## 当前研究状态

截至 2026-07-19：

- V1 synthetic projection distillation 已拒绝：LPIPS 改善伴随静态漂移和轨迹动力学恶化；
- 当前 RGB/VAE counterfactual target 与 shared temporal LoRA endpoint 路线未通过合法性和局部性门禁；
- 旧 P-UNC forced-binary preference recipe 已被 48-case 人工复核拒绝；
- common-support selective partial-order oracle 通过了 false-strict 与 shortcut 校准，但旧候选池只有
  `2/96` strict，唯一 earlier-fork fallback 也未通过首帧/质量门禁；
- V5 route-pivot autoresearch 已完成：A0 真实 target machine gate 通过，但 A1 证明冻结 SVD 只能读取 ego
  motion，actor residual 低于 zero baseline 且违反 stationary safeguard，故 Route A 拒绝；
- Route B 的 128 个 natural rollouts 在完整 anti-collapse 下只有 `1/16` conditions 有安全 diversity，P-UNC
  对 random/Base 的 CoTracker win-credit 均为 `41.67%`，故不进入 AWR/SFT；
- Route C 已完成只读审计，选择 `C1 = ReSim exp0_no_carla feasibility`：它显式条件化 future ego trajectory，
  但不要求 future actor boxes；当前未下载、未推理、未训练；
- V5 已完成并关闭；C1-BOOT 仍需下一份预注册计划明确授权，存储维护不会自动触发 ReSim 下载、推理或训练；
- R1 的 32 个 pair 与 A0 v3 的 12 个 panel 仍待人工 review，正式 48 条人工 verdict 和这些 review package
  都是受保护证据；
- 已登记 2026-07-19 历史产物清理批次，以提高后续 C1-BOOT 的磁盘余量。逐路径范围、SVD 固定 revision、
  保护哈希和执行结果见 [`docs/ARTIFACT_RETENTION.md`](docs/ARTIFACT_RETENTION.md)。

这不是“驾驶视频物理对齐不可行”的结论。准确的证据边界、禁止重复项和重新开启条件见：

1. [`docs/RESEARCH_STATUS.md`](docs/RESEARCH_STATUS.md)：唯一当前状态与执行授权入口；
2. [`docs/RESEARCH_FAILURES.md`](docs/RESEARCH_FAILURES.md)：从 V1 开始的 research 负结论与未决风险；
3. [`docs/MOTION_ROUTE_PIVOT_AUTORESEARCH_PLAN_V5.md`](docs/MOTION_ROUTE_PIVOT_AUTORESEARCH_PLAN_V5.md)：
   已完成的 R0→R1→A0→A1/B0→C→D0 执行协议与门禁记录；
4. [`docs/EXPERIMENTS.md`](docs/EXPERIMENTS.md)：正式实验事实源；
5. [`docs/ROUTE_PIVOT_FINAL_REPORT.md`](docs/ROUTE_PIVOT_FINAL_REPORT.md)：V5 最终结论、停止项和下一阶段最多
   三个实验；
6. [`docs/BACKBONE_MIGRATION_AUDIT.md`](docs/BACKBONE_MIGRATION_AUDIT.md)：ReSim/VISTA/OpenDWM/
   MagicDrive-V2 等候选的一手迁移审计；
7. [`docs/ARTIFACT_RETENTION.md`](docs/ARTIFACT_RETENTION.md)：非 Git 产物的驻留、清理与重建账本；
8. [`docs/archive/2026-07/README.md`](docs/archive/2026-07/README.md)：旧计划、报告和评审协议索引。

## 可复用能力

```text
motion_proj/
  data/          nuScenes 数据、scene split 与 preference schema
  backbones/     SVD backbone、官方 conditioning parity 与 LoRA scope
  auditor/       RAFT、ego/static flow、深度与 generated tracks
  projector/     动力学能量、support、smoothing 与 warping
  cache/         可追溯 projection/replay cache
  losses/        projection、flow、tube、anchor 与 V2 loss
  train/         单卡训练、checkpoint 与 pilot 基础设施
  eval/          driving metrics、独立 tracker 与几何诊断
  preference/    paired query、common support、校准与 selective order
  diagnostics/   条件、target、evaluator、branch、pair 与 reaudit 门禁
  runtime/       manifest、fingerprint、原子写入与 stage 管理
configs/         数据、模型、训练、评估和历史诊断配置
tests/           单元、回归和 fail-closed 测试
```

其中“可复用”只表示实现和局部门禁已有证据，不代表对应研究路线已经晋级。开始任何新研究前，必须先读取
`RESEARCH_STATUS.md` 和 `RESEARCH_FAILURES.md`，并以新的预注册假设明确哪些历史门禁需要重新验证。

## 环境与自检

项目环境位于数据盘。每个新 shell 先执行：

```bash
source /root/miniconda3/etc/profile.d/conda.sh
conda activate /root/autodl-tmp/envs/motionproj
cd /root/autodl-tmp/motion_proj
pytest -q
```

环境版本、数据路径和网络说明见 [`docs/ENVIRONMENT.md`](docs/ENVIRONMENT.md)，换机接续见
[`docs/MACHINE_MIGRATION.md`](docs/MACHINE_MIGRATION.md)，第三方依赖见
[`docs/THIRD_PARTY.md`](docs/THIRD_PARTY.md)。SVD 权重说明位于
[`scripts/download_weights.md`](scripts/download_weights.md)。

## 工程入口

安装 editable package 后，以下入口仍可用于复现历史工程链或构建新诊断：

```bash
motionproj-build-cache --help
motionproj-train --help
motionproj-eval --help
motionproj-mine --help
motionproj-inspect --help
motionproj-split-manifest --help
```

这些命令是工程接口，不是当前实验排程。不得仅凭入口存在就重启已拒绝的 cache、训练或搜索流程。

## 证据与产物

- 正式实验登记：`docs/EXPERIMENTS.md`
- 轻量 Git 内证据：`docs/run_manifests/`
- 完整运行产物：`/root/autodl-tmp/runs/`
- cache、checkpoint 与权重：`/root/autodl-tmp/cache/`、各 run 的 `ckpts/`、`/root/autodl-tmp/weights/`

正式 run 目录不可复用或覆盖。历史计划已集中归档，归档中的“当前任务”“下一步”只描述当时状态，
不能授权新的执行。正式 run 可按保留策略删除已关闭路线的大型二进制载荷；这不会改变 run ID、轻量事实源、
人工 verdict 或负结论的重开条件。
