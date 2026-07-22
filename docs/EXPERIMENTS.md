# Motion-Proj V7 实验事实源

> **范围**：本文件只登记 OccGS-Resim V7。V1–V6 及整理前全量账本已归档至
> [`archive/2026-07/v7-feasibility/EXPERIMENTS_V1_V7_SNAPSHOT.md`](archive/2026-07/v7-feasibility/EXPERIMENTS_V1_V7_SNAPSHOT.md)。
> **证据基线**：`9722fa2`。
> **当前状态**：见 [`RESEARCH_STATUS.md`](RESEARCH_STATUS.md)；本文件不授权执行。

## 1. 证据完整性说明

V7 feasibility 产物可定位且核心 JSON/ckpt 存在，但 `runs/occgs_resim/` 下未保存本仓库正式协议要求的
`manifest.json`、`resolved.yaml` 与唯一终态标记。因此下表统一标记为 `retrospective`：数值可由现存产物核对，
但不能事后补写未知 seed、运行开始 commit 或 data fingerprint。

后续新 run 必须通过 `V7-EV-10` 的正式 run contract；不得覆盖下列目录或复用 run ID。

## 2. 数据与环境

| Task / artifact | 状态 | 配置与结果 | 证据 | 结论边界 |
|---|---|---|---|---|
| `E0-ENV-01` | completed / retrospective | DriveStudio env；Python 3.9.25；torch 2.1.2+cu118；gsplat 1.3.0；CUDA extension smoke 通过 | [`archive/2026-07/v7-feasibility/OCCGS_E0_ENV_MANIFEST.md`](archive/2026-07/v7-feasibility/OCCGS_E0_ENV_MANIFEST.md) | 环境可用，不是方法结果 |
| `G0-THIRDPARTY-00` | completed / retrospective | DriveStudio `e59bda4...` MIT；SplatAD/Occ3D 只审计；本机完整前向 sweep 限于 mini 10 scenes | [`archive/2026-07/v7-feasibility/OCCGS_THIRD_PARTY_AUDIT.md`](archive/2026-07/v7-feasibility/OCCGS_THIRD_PARTY_AUDIT.md) | 约束后续数据规模 |
| `D0-DATA-02` | completed / retrospective | S0=003/scene-0655，S1=005/scene-0796，S2=004/scene-0757；3 前向相机；8 秒训练窗；10 Hz 处理；数据完整性通过 | [`archive/2026-07/v7-feasibility/OCCGS_DATA_PREPARATION.md`](archive/2026-07/v7-feasibility/OCCGS_DATA_PREPARATION.md)；`data/occgs/scene_specs/d0_frozen_picks_v2.json` | 三场景 feasibility，不支持规模结论 |

## 3. B0 reconstruction

所有全量 run 使用 DriveStudio StreetGS、3 cameras、`t=0..79`、30k iterations、`load_smpl=False`；
B0-1 为 1 camera、4 秒、3k iterations smoke。

| Run ID | 状态 | full PSNR / SSIM / LPIPS | test PSNR / SSIM / LPIPS | vehicle full/test PSNR | 证据 |
|---|---|---|---|---|---|
| `b0_1_s0_1cam4s` | completed / retrospective | 37.59 / 0.968 / 0.059 | 32.91 / 0.928 / 0.072 | 33.26 / 27.90 | `runs/occgs_resim/b0_recon/occgs_b0/b0_1_s0_1cam4s/` |
| `b0_2_s0_3cam8s` | completed / retrospective | 32.87 / 0.937 / 0.103 | 25.60 / 0.799 / 0.142 | 30.98 / 23.48 | `runs/occgs_resim/b0_recon/occgs_b0/b0_2_s0_3cam8s/` |
| `b0_3_s1_3cam8s` | completed / retrospective | 27.07 / 0.815 / 0.220 | 20.18 / 0.472 / 0.325 | 25.12 / 18.26 | `runs/occgs_resim/b0_recon/occgs_b0/b0_3_s1_3cam8s/` |
| `b0_4_s2_3cam8s` | completed / retrospective | 33.41 / 0.923 / 0.094 | 25.37 / 0.697 / 0.142 | 27.24 / 22.11 | `runs/occgs_resim/b0_recon/occgs_b0/b0_4_s2_3cam8s/` |

现存证据包括 `config.yaml`、training `metrics.json`、full/test eval JSON、`checkpoint_final.pth` 与渲染产物。
未发现 NaN/OOM；文档记录训练峰值约 5 GB。B0 证明单卡重建可行，但没有 formal user review，也没有与
Static GS/OmniRe 做方法比较。

## 4. O0 / S0 / C0

| Task / run | 状态 | 关键结果 | 证据 | 准确结论 |
|---|---|---|---|---|
| `O0-OCC-04` | completed artifact / retrospective | 003/004/005 各 80 帧，200×200×16；unknown/free/static/dynamic 分离；unknown 比例约 0.96 | `data/occgs/occupancy/{003,004,005}/{frame_*.npz,meta.json,summary.json}` | occupancy artifact 存在；尚未进入 editor/render/completion 主链 |
| `S0-EDIT-05` | completed prototype / retrospective | actor 003/35、005/34、004/8；0.8 m variants 多数通过，1.6 m 在 S1 拒绝，极端 V4 全拒绝 | `data/occgs/scene_specs/s0_edits/scene_*_actor_*_edits.json` | 当前 validator 使用相对运动学、横向范围和 actor/ego 距离；未查询 occupancy |
| `occgs_c0/{s0,s1,s2}` | completed machine screen / retrospective | 3 scenes；可见编辑 case 46/62 机器合法；按 mean edit effect 选出的 top-24 为 24/24；003/004 明确贡献多数合法 case | `runs/occgs_resim/c0_cf/{s0,s1,s2}/`；`data/occgs/reviews/c0_legality/c0_legality_screen.json` | machine-only；没有用户 human verdict；top-k 不能用于估计全分布合法率 |

补充 provenance 缺口：`s0_edit_summary.json` 会被单次脚本执行覆盖，当前只汇总 scene 004；三场景事实应读取各自
`scene_*_edits.json`，后续由 `V7-EV-10` 修复聚合方式，但不得改写原文件。

## 5. L0 completion feasibility

| Run ID | 状态 | 方法 | 结果 | 证据 |
|---|---|---|---|---|
| `l0_comp/s0_v3` | completed / retrospective | RGB-diff mask + Telea + hard composition；6 帧 | outside-mask L1=0；inside-mask L1≈22.48；mask≈1.27% | `runs/occgs_resim/l0_comp/s0_v3/l0_feasibility.json` |
| `l0_comp/s2_v3` | completed / retrospective | 同上；6 帧 | outside-mask L1=0；inside-mask L1≈25.97；mask≈1.99% | `runs/occgs_resim/l0_comp/s2_v3/l0_feasibility.json` |

outside-mask 为 0 由 `I=(1-M)I_GS+MI_gen` 构造保证。当前 mask 来自 V0/edited RGB 差分，不是 occupancy 或
ray visibility；没有 pseudo-hole 真值、时序质量或用户人工 verdict。因此只登记 locality implementation
feasibility，不登记 H2 pass。

## 6. U0 utility proxy

| Run ID | 状态 | 关键结果 | 证据 | 准确结论 |
|---|---|---|---|---|
| `u0_screen/u0_proxy_v1` | partial / retrospective | V1/V2 accept rate=1.0，V3=0.667，极端 V4=0；V1/V2/V3 mean max RGB diff≈0.454/0.475/0.395；`u0_full_map_pass=false` | `runs/occgs_resim/u0_screen/u0_proxy_v1.json` | 只说明当前规则能拒绝故意无效 V4，且 accepted edit 有像素信号；没有 downstream utility |

该 proxy 的 `naive_V4` 是横向位移约 39–50 m 的强制负例，不是 matched naive GS baseline。它不能支持
“OccGS 优于 naive GS”的研究结论。

## 7. D1 决策

| Task | 状态 | 决策 | 证据 |
|---|---|---|---|
| `D1-DECIDE-09` | done | `modify_method_then_scale` | [`OCCGS_FINAL_REPORT.md`](OCCGS_FINAL_REPORT.md) |

含义是保留路线、优先执行 `V7-EV-10 → V7-H1-11`；不表示 H1/H2/H3 已通过，也不解锁扩场景或双卡。

## 8. 登记规则

- 本文件只追加后续 V7 正式实验；历史全量事实不再回填到当前表。
- 正式 run 不得复用目录或 ID；engineering failure、research rejection 和 completed 都保留。
- 每条新结论必须给出 commit、resolved config、data/code fingerprint、split、seed、证据路径和终态标记。
- 人工 verdict 只能由用户或指定评审者填写；机器 screen 与 agent 目视检查单独登记。
- 任何汇总必须能从 run 产物重新生成；不得手工修饰原始指标或用 top-k 替代全分布结果。
