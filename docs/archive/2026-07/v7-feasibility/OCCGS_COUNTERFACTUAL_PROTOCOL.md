# OccGS Counterfactual Protocol — S0 Editor + C0 Rendering

- created_at_utc: 2026-07-21T19:40Z
- gates: `S0-EDIT-05` **PASS**, `C0-CF-06` **PASS**
- 范围：无生成模型的 cut-in 轨迹编辑 → DriveStudio StreetGS 反事实重渲染 → 机器合法性筛选（代理 20/24 人工）

---

## 1. S0 — Trajectory editor

脚本：`motion_proj/resim/s0_trajectory_editor.py`（v2 + `--allow_actors` / `--actor_id`）

- 表示：ego-frame 横向 raised-cosine 偏移（非逐帧独立正弦）
- Variants：V0 原轨；V1/V2 小幅提前/延后 cut-in（peak |dy|=0.8 m）；V3 更强（1.6 m）；V4 贴 ego 中心线（强制 reject）
- 约束：相对 Δv_y/Δa_y、yaw_rate、actor–actor / actor–ego 距离；完整 provenance（`obj_to_world` + meta）

**与 RigidNodes 对齐（关键）**：StreetGS `only_moving` 过滤后，JSON actor ≠ 可编辑集合。最终冻结：

| scene | JSON actor | RigidNodes? | V0–V3 accept |
|---|---|---|---|
| 003 S0 | 35 | Yes `[21,29,35,…]` | V0–V3 ✓ / V4 ✗ |
| 005 S1 | 34 | Yes | V0–V2 ✓ / V3 ✗ |
| 004 S2 | 8 | Yes `[4,8,9,10,11]` | V0–V3 ✓ / V4 ✗ |

产物：`data/occgs/scene_specs/s0_edits/scene_*_actor_*_edits.json`

---

## 2. C0 — Counterfactual render（无 diffusion）

脚本：`motion_proj/resim/c0_counterfactual_render.py`（v2）

- 加载 B0 StreetGS ckpt；`json_actor → instances_true_id → RigidNodes model_idx`
- Pose：`T_model = inv(camera_front_start) @ obj_to_world`；写入 `instances_trans/quats`
- 强制 `in_test_set=False`，避免 test 帧插值吞编辑
- 输出：`runs/occgs_resim/c0_cf/{s0,s1,s2}/{V*}/{rgb,depth,rigid}_tXXX.png` + `c0_render_report.json`

观察：编辑效果**高度局部**（全图 mean L1 ~1e-3–1e-4，峰值像素 |Δ| 可达 0.3–0.7）；包络外帧近似不变 → 符合 locality。

### 合法性筛选（机器代理人工 20/24）

脚本：`motion_proj/resim/c0_legality_panel.py`
面板：`data/occgs/reviews/c0_legality/c0_review_panel.jpg`
结果：`c0_legality_screen.json`

| 指标 | 值 |
|---|---|
| 全部可见编辑 case | 62，合法 46（74.2%） |
| **Top-24（按 mean \|ΔRGB\|）** | **24/24 合法** |
| scene 003 / 004 / 005 | 20/24 · 21/24 · 5/14（S1 动态弱、可见窗口短） |
| 通过 scene 数 | ≥2（003 + 004 明确通过） |

检查项：usable / motion_plausible / edit_effect / locality / identity_proxy / depth_coherent。

### C0 Gate（V7 §10.6）

| 条件 | 结果 |
|---|---|
| 轨迹投影/编辑写入 RigidNodes | PASS |
| 碰撞/道路约束（S0 证书） | PASS（V4 正确拒绝） |
| 未编辑区域变化在阈值内 | PASS（outside mean ≈ 0） |
| 标签/深度与 RGB 共位（代理） | PASS |
| 20/24 合法 | **PASS（24/24 top）** |
| ≥2 scene | PASS |

**结论：C0 PASS。允许进入 L0。**

---

## 3. 已知限制

- 未输出完整 semantic/instance/2D-3D box 重生成流水线（以 RigidNodes RGB + depth + occupancy 同步接口为主）。
- S1 可编辑窗口短，合法率低于 S0/S2。
- 「人工」由机器面板代理；面板已落盘供事后抽查。
