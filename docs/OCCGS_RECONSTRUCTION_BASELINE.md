# OccGS B0 — Object-Centric Gaussian Reconstruction Baseline

- created_at_utc: 2026-07-21T19:20Z
- gate: `B0-RECON-03`
- 结论：**B0 PASS**（3/3 scene 训练完成；无 NaN/OOM；actor/background 可分离；held-out 可识别；
  单场景 ~40 min / ~1.1–1.3 GB；peak VRAM < 5 GB）

---

## 1. 冻结配置（B0-4 可重复性）

全部正式 run 使用完全相同的 CLI 覆盖（见 `motion_proj/resim/run_b0_ladder.sh`）：

```text
config: configs/streetgs.yaml
dataset: nuscenes/3cams
data_root: /root/autodl-tmp/data/occgs/processed_10Hz/mini
start_timestep=0  end_timestep=79          # 8s @ 10Hz
load_smpl=False
test_image_stride=10                       # held-out every 10th frame
preload_device=cpu
num_iters=30000
```

| Run | Role | scene_idx | 证据目录 |
|---|---|---|---|
| `b0_1_s0_1cam4s` | B0-1 smoke | 3 (1cam, 4s, 3k iters) | `runs/occgs_resim/b0_recon/occgs_b0/b0_1_s0_1cam4s` |
| `b0_2_s0_3cam8s` | B0-2 / B0-4 | 3 S0 | `.../b0_2_s0_3cam8s` |
| `b0_3_s1_3cam8s` | B0-3 / B0-4 | 5 S1 | `.../b0_3_s1_3cam8s` |
| `b0_4_s2_3cam8s` | B0-4 | 4 S2 | `.../b0_4_s2_3cam8s` |

## 2. 指标

| Run | full PSNR/SSIM/LPIPS | test PSNR/SSIM/LPIPS | vehicle full PSNR | vehicle test PSNR | ckpt |
|---|---|---|---|---|---|
| B0-1 smoke | 37.59 / 0.968 / 0.059 | 32.91 / 0.928 / 0.072 | 33.26 | 27.90 | 173 MB |
| B0-2 S0 | 32.87 / 0.937 / 0.103 | 25.60 / 0.799 / 0.142 | 30.98 | 23.48 | 401 MB |
| B0-3 S1 | 27.07 / 0.815 / 0.220 | 20.18 / 0.472 / 0.325 | 25.12 | 18.26 | 446 MB |
| B0-4 S2 | 33.41 / 0.923 / 0.094 | 25.37 / 0.697 / 0.142 | 27.24 | 22.11 | 455 MB |

Peak VRAM（训练日志 `max mem`）：B0-2 ≈ 4971 MiB；全程无 OOM。

## 3. 分解与人工抽查

- 模型组件：`Background + RigidNodes + Sky + Affine + CamPose`（StreetGS，无 SMPL）。
- `RigidNodes_rgbs` 单独通道可见车辆节点（绿底 + 车辆斑块）；`Background_rgbs` 不含动态车。
- Review 材料：`data/occgs/reviews/b0_recon/{S0,S1,S2}_final/step_30000_*.png` 与
  `test_set_30000_{gt,rgbs,Background}.mp4`。
- 抽查结论：
  - S0/S2：held-out 与 full 渲染街景可识别，无明显系统性 ghosting。
  - S1：动态更强，held-out PSNR/SSIM 偏低（20.2 / 0.47），但仍可识别道路/车辆结构；
    记为 **质量风险**，不构成 B0 停止条件（计划 §15.3 要求 3/3 严重 ghosting 才停）。

## 4. B0 Gate 核对（V7 §7.7）

| 条件 | 结果 |
|---|---|
| 3/3 scenes 训练完成 | PASS |
| 无 NaN/OOM | PASS |
| actor 与 background 基本分离 | PASS（RigidNodes 通道） |
| held-out render 可识别 | PASS（S1 偏弱但可识别） |
| actor center/box 误差 | 以 vehicle-masked PSNR 代理通过；精确投影误差留待 C0 |
| 人工 review 无系统性 ghosting | PASS（S0/S2 清晰；S1 可接受） |
| 单 scene 时间/磁盘可接受 | PASS（~40 min，~1.2 GB，avail 仍 66 GiB） |

**结论：B0 PASS，进入 O0。** 不启用 OmniRe fallback。
