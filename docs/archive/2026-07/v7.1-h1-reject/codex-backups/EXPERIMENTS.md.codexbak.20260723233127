# Motion-Proj V7 / V7.1 实验事实源

> **范围**：本文件登记 OccGS-Resim V7 retrospective evidence 与 V7.1 正式实验。V1–V6 及整理前全量账本已归档至
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

## 8. V7.1 EV-10 证据合同

| Task / run | 状态 | 配置与结果 | 证据 | 准确结论 |
|---|---|---|---|---|
| `V7-EV-10` evidence index | completed | 对 B0/O0/S0/C0/L0/U0 共 1,610 个文件、4,121,645,920 bytes 逐文件计算 SHA256；index SHA256=`fbd8c65774edef6ad253f458ac01da29a95694c8f9448179b42601e89fbdb613` | `/root/autodl-tmp/runs/occgs_resim/V7_EVIDENCE_INDEX.json` | 旧证据可定位；未知 seed、run-start commit、fingerprint 和 terminal marker 保持显式 missing/unknown，未事后补造 |
| `v71_v7-ev-10__smoke__s0__20260723T141019751134Z__7d97212f` | completed / `COMPLETE` | seed=0；config fingerprint=`7d97212fbb38f85ae9cc6a7e348b424a6229e430b2c8fd4089e3bbe9eae5eb19`；data fingerprint=`fbd8c657...b613`；world/render/artifact-set hash 均通过；CPU only | `/root/autodl-tmp/runs/occgs_resim/v71/V7-EV-10/v71_v7-ev-10__smoke__s0__20260723T141019751134Z__7d97212f/` | 新 run 缺 summary、三层 hash、artifact bytes 或唯一 terminal marker 时不能 COMPLETE；这是工程合同 smoke，不是研究假设结果 |

实现 commit 为 `3590558cd1ef3644f10c1b981366c3ccce9cd580`。验证命令：

```bash
PYTHONPATH=. pytest -q \
  tests/test_v71_run_contract.py \
  tests/test_config_runtime.py \
  tests/test_fingerprint.py
# 25 passed
python resim/v71_run_contract.py validate \
  /root/autodl-tmp/runs/occgs_resim/v71/V7-EV-10/v71_v7-ev-10__smoke__s0__20260723T141019751134Z__7d97212f
```

S0 editor 已改为原子增量合并 `s0_edit_summary.json`，后续单 scene 执行不再覆盖其他 scene；EV-10 未运行
editor，也未修改现有 S0 summary 或任何旧 metrics。

## 9. V7.1 H1-11A 状态底座

| Run ID | 状态 | 配置与结果 | 证据 | 准确结论 |
|---|---|---|---|---|
| `v71_v7-h1-11a__pilot-3__s0__20260723T144155452295Z__0ff143d9` | completed / `COMPLETE` | PILOT-3；seed=0；S0/S1/S2 RigidNodes actor=9/18/5；跨进程 registry hash 稳定；1,679 个 pose 的 coordinate round-trip gate PASS；world/render/artifact-set hashes 完整 | `/root/autodl-tmp/runs/occgs_resim/v71/V7-H1-11A/v71_v7-h1-11a__pilot-3__s0__20260723T144155452295Z__0ff143d9/` | WorldState schema、显式坐标与 actor registry 工程底座通过；未运行 certificate/projection，不构成 H1-CERT/H1-PROJ 结论 |

实现与运行代码 commit 为 `766f2287e79b3cdfc877eb175776482c79c3f98c`，config fingerprint 为
`0ff143d9d060d52001e27b5947de24406f38a487c74e1694a79a40ad522dc724`。逐场景 registry hash：

- 003：`c9359fc3a6adeb135db09eab9a10e5a1ebcf452aa57ba779cd1564e2cb7b1ed0`
- 005：`5cac16f5879df8afb3c0827b4f1ef64a40c4f3abd797f502eef5d0a518ad91ee`
- 004：`d43d16f2682efcb5576570b0bef8e4e4f1fe59a50a0451d8f47605b38faac470`

坐标审计明确冻结：

- annotation：world frame；
- DriveStudio model：起始 `CAM_FRONT` sensor frame；
- O0 grid：per-frame LiDAR sensor frame；
- extrinsics：`T_world_camera`；LiDAR pose：`T_world_lidar`。

三场景最大 world→model→world translation error 为 `6.83e-13 m`，最大 rotation error 为
`7.89e-08 rad`；最大 world→grid→world box error 为 `9.10e-13 m`。checkpoint pose refinement 相对原
annotation 的最大 translation delta 为 `0.8763 m`、最大 rotation delta 为 `0.0532 rad`，作为训练后 pose
差异单独报告，不混入 round-trip gate。

相关验证为 42 passed；正式 run 经 `v71_run_contract.py validate` 独立复验为唯一 `COMPLETE`。

## 10. V7.1 H1-11B 分层证据与 certificate calibration

| Run ID | 状态 | 配置与结果 | 证据 | 准确结论 |
|---|---|---|---|---|
| `v71_v7-h1-11b__pilot-3-calibration__s0__20260723T145956893820Z__b8349bc0` | completed / `COMPLETE` | 240 帧 observation-evidence-v2；32 条真实 source controls，30 条可测且 30/30 PASS；collision/teleport 可检测负例 2/2；off-road 在 map-expansion 缺失时 UNKNOWN；32/32 full certificate 为 UNKNOWN；三类 overlay 分离 | `/root/autodl-tmp/runs/occgs_resim/v71/V7-H1-11B/v71_v7-h1-11b__pilot-3-calibration__s0__20260723T145956893820Z__b8349bc0/` | safety/observation/render-support 职责、三态语义和校准 gate 通过；尚未运行 C/D1/D2 matched H1 实验，不构成 H1-CERT/H1-PROJ 结论 |

实现 commit 为 `002bbb499e2bf967a0b16e19c09088cef2e60ef5`。核心冻结 hash：

- evidence set：`a9b02f08474fdcbfb58ef54e6788900cc1cd9dddf15db488ae6c8f837f807930`
- render support set：`2e4f02979db4aea850ab0c73325e8e5270d4d7fb554ce47fc365659bfa49c671`
- world state：`6ac9b6b0a8ea0b4acf2efffb1e5722709e79201d353b532489f0b8db0ae272e5`
- render request：`02451378ac4cd84bc965d8942eac3da25c23ef0e31d1b4b646b6052cf50cb922`
- artifact set：`c4800f6bc73ba1fce38ee8d8f408dd3921116b53df5fe1d71b41f9a67f8382bd`

旧 rotated-corner AABB 相对 oriented-box center-inclusion 的体素量比分别为 003 `1.721×`、005 `2.249×`、
004 `2.833×`，确认旧表示会系统性膨胀动态体积。分离 actor layer 后，三场景 base unknown 比例仍为
`97.10% / 96.04% / 97.57%`；因此 source actor removal 恢复 UNKNOWN 而不是 FREE，不能靠拒绝困难样本产生
表面 precision。

地图审计只发现 nuScenes 栅格底图 PNG，没有可查询的 map-expansion polygons，故 road-support 和对应 off-road
control 保持 UNKNOWN。真实 control 的可测 retention 只聚合 kinematic、continuous dynamic OBB 与有直接 LiDAR
点支持的 source-observation；完整 certificate 不把 road UNKNOWN 偷记为 PASS。scene 005 的一个 RigidNodes
model index 在 checkpoint 中有 0 个 Gaussian primitive，已作为 render-support 缺失事实保留，不影响 safety
geometry，但后续 visibility/label gate 必须 fail closed。

40 项 V7.1 相关测试通过；正式 run 经 `v71_run_contract.py validate` 独立复验为唯一 `COMPLETE`。机器 overlay
没有 agent 填写的人工 verdict。

## 11. 登记规则

- 本文件只追加后续 V7 正式实验；历史全量事实不再回填到当前表。
- 正式 run 不得复用目录或 ID；engineering failure、research rejection 和 completed 都保留。
- 每条新结论必须给出 commit、resolved config、data/code fingerprint、split、seed、证据路径和终态标记。
- 人工 verdict 只能由用户或指定评审者填写；机器 screen 与 agent 目视检查单独登记。
- 任何汇总必须能从 run 产物重新生成；不得手工修饰原始指标或用 top-k 替代全分布结果。
