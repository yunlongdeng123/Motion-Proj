# WorldSim V4 P0：Paper-First Scope Freeze

- Task：`WS-V4-P0-SCOPE-PAPER-FREEZE-01`
- 状态：`done`
- 日期：2026-08-11
- 起始 HEAD：`main@21084309480895f5541196a06191a5dffb4e30c1`
- 分支：`research/worldsim-v4-evidelta`
- V3.3 收口提交：`e6663e1`，已确认在 `main` 历史中
- 事实配置：[`../configs/worldsim_v4/p0_scope_v1.yaml`](../configs/worldsim_v4/p0_scope_v1.yaml)
- 计划：[`WORLDSIM_V4_EVIDELTA_GS_PLAN.md`](WORLDSIM_V4_EVIDELTA_GS_PLAN.md)
- canonical：`20260811T080636Z__p0-scope-formal-s0-r2`
- config/summary/manifest/status SHA-256：
  `248bde621343597196c1a608ce8674a0c4a1f974d38abc70710c7783d8ecaaa8` /
  `aba1fbcffbe89e7b992bb1d0c691f398423c143319628b52cff7f7f3d0b51283` /
  `ec32e983ad48e6ed415906562c90338844bfc47ec305afa950bb4a99f1543970` /
  `b39416015b1d6275dd3b8bfefa74c7aa45d4ceee790fdeab4d72b5e3baca272a`
- 训练 / 推理 / 权重下载：`0 / 0 / 0`
- 下一唯一授权：`WS-V4-D0-NUSCENES-COHORT-01`

## 1. P0 结论

P0 将 V4 冻结为一个 paper-first、scene-level 统计的 EviDelta-GS 路线。V3.3 canonical 资产继续只读；V4 不重新发明
base reconstruction，而是在不可变 StreetGS/V3.3 base 上建立三项中心能力：

```text
Evidence-Calibrated Gaussian Field
→ Bayes-Risk Repair Router
→ SE(3) B-Spline Temporal Reversible Delta
```

计划草案中的基线事实已按仓库现状纠正：草案记录的 `main@144ed19` 是历史点，P0 实际起点为 `main@2108430`。
`e6663e1` 和 `144ed19` 均是实际起点的祖先；因此 V4 不从旧分支回退，也不重写共享历史。

## 2. 方法公式到实现证据的冻结映射

| Method 定义 | 配置入口 | 代码入口 | 主消融 | 可计算指标 |
|---|---|---|---|---|
| Beta Gaussian evidence `E=(alpha,beta)` | `method_schema.evidence_state` | `beta_fusion.py` / `evidence_state.py` | E0–E4 | ECE、Brier、NLL、IoU、Boundary F1 |
| 多视图加权证据 | `method_schema.multi_view_update` | `beta_fusion.py` | mask/depth/LiDAR evidence | FP/FN mass、uncertainty-error correlation |
| development-only calibration | `method_schema.calibration` | `evidence_calibration.py` | raw/temperature/beta | ECE、Brier、reliability diagram |
| Bayes repair risk | `method_schema.repair_risk` | `repair_risk.py` / `repair_router.py` | R0–R5 matched | coverage-error、selective risk、PSNR/SSIM/LPIPS |
| `SE(3)` cubic B-spline | `method_schema.temporal_transform` | `se3_bspline.py` | T0–T4 | tLPIPS、warp error、trajectory RMSE、identity switch |
| 可逆 asset delta | `method_schema.reversible_delta` | `temporal_delta.py` / `delta_compiler.py` | frame/linear/B-spline/full | rollback render SHA exact、semantic reintroduction |

这里冻结的是可实现合同，不是结果声明。任何公式只有在对应代码、ablation 和指标均落地后才能进入论文 Method。

## 3. 数据与泄漏合同

nuScenes cohort 在任何新方法结果前冻结为 30 个 scene-disjoint 场景：

| Role | Scene 数 | 使用规则 |
|---|---:|---|
| development | 6 | 允许拟合 calibration、risk 权重、阈值、B-spline 超参数 |
| validation | 6 | 只读确认，失败后不得在同一批场景重调 |
| test | 18 | `V4_TEST_FREEZE.json` 提交后只读取一次 |

选场只能使用时间、天气、道路几何、actor/遮挡/donor 支持、速度、类别和距离等结果前 metadata。每个场景必须保留
high actor、difficult actor/`ABSTAIN`、remove/lateral/insert、2–4 秒连续 clip 以及 train/development/heldout frame
合同；不存在合格 actor 时保留 `ABSTAIN_NO_ACTOR` 到 denominator。

KITTI 仅允许读取 `/root/autodl-pub/KITTI`。P0 实查该路径不存在，当前状态是
`blocked_local_dataset_missing`；这不阻塞 D0 nuScenes cohort，但 D1 不能启动 adapter quality run，也不得通过网络下载
绕过。详见 [`WS_V4_KITTI_AUDIT.md`](WS_V4_KITTI_AUDIT.md)。

## 4. Baseline 冻结

- Tier A：V3.3 frozen、Native StreetGS/DriveStudio、AD-GS；全部要求 same scene/split/resolution/heldout/resource recording。
- Tier B：SplatAD、IDSplat、Inpaint360GS；只有官方实现、adapter 与单 RTX 3090 preflight 通过才进入数值表。
- Paper-only：HorizonForge、RecEdit-Drive、GOR-IS、3D-GIMP 等不可执行或协议不匹配路线只做定性边界，不填伪造数值。

P0 不声称 V4 已优于任何 baseline。B0 阶段必须先让 6 个 development scene 上的 matched evaluator 稳定复现，才解锁
M1。

## 5. 指标与统计冻结

正式图像主指标固定为 `PSNR / SSIM / LPIPS-Alex`，按 global/static/actor/boundary/edit ROI 报告；没有 GT 的 edit
ROI 标记 `undefined`，不得把生成图当 GT。同步保留 editing、temporal、geometry/LiDAR、evidence/calibration、engineering
和 downstream 指标。

统计单位固定为 scene，报告 mean/median/std/IQR/95% scene-bootstrap CI，并使用 paired scene-level test。失败、阻塞和
ABSTAIN 均留在完整 denominator。随机模块只对最终候选执行至少 3 seeds；确定性模块要求 byte-exact replay。

## 6. 单卡与资源门

P0 实测 GPU=`RTX 3090 24,576 MiB`，GPU compute process=`0`，cgroup max=`96,636,764,160 bytes`，
`oom/oom_kill=0/0`，数据盘可用约 `193 GiB`。在 6 dev + 6 val + KITTI 2-sequence adapter smoke 完整闭环前，禁止
DDP/FSDP/tensor/model parallel；之后即使扩卡也只允许 scene-level 横向并行。

## 7. P0 门与下一步

已冻结：论文命题、数学 schema、baseline matrix、nuScenes/KITTI 数据协议、图像/时序/几何/工程指标、统计单位、
test-freeze 协议和一手来源矩阵。P0 没有训练、推理、安装依赖或下载数据/权重。

formal r2 共冻结 5 份 source snapshot，config/plan SHA exact，15 个一手来源、14 个任务状态和全部 P0 gates 通过；
run 总量 `101,624 bytes`，repository audit 明确记录 dirty staging，而不是伪称 clean commit。r1 已全部通过，但其后
提交前 whitespace gate 规范了计划参考文献行尾，导致 plan/config SHA 合法改变；故 r1 只保留为 noncanonical done，
r2 对最终字节重新审计，没有覆盖 r1。

下一步只执行 `WS-V4-D0-NUSCENES-COHORT-01`：枚举结果前 metadata，冻结 6/6/18 split，输出 deterministic cohort
manifest 和 SHA，然后只对两个 development scene 做数据/preprocess smoke。
