# OccGS-Resim V7 — Final Report

- created_at_utc: 2026-07-21T19:45Z
- plan: `docs/OCCGS_RESIM_AUTORESEARCH_PLAN_V7.md`
- hardware: 1× RTX 4090 24GB；`/root/autodl-tmp` avail ≈ 66 GiB at close
- env: `/root/autodl-tmp/envs/drivestudio`

---

## 1. Executive decision（D1）

**决策：`modify_method_then_scale`（条件性继续，不停止）**

| 维度 | 判定 |
|---|---|
| 显式可编辑 3D 场景（B0） | **成立** — StreetGS RigidNodes 单卡可重建 3/3 mini scenes |
| Occupancy anchor（O0） | **成立** — per-frame ego LiDAR+box occupancy，unknown 保留 |
| 约束轨迹编辑（S0） | **成立** — cut-in variants + V4 负例拒绝 + RigidNodes 对齐 |
| 无生成反事实渲染（C0） | **成立** — top-24 机器合法 24/24；≥2 scene |
| 局部补全协议（L0） | **协议成立 / 视觉弱** — hard composition outside-mask L1=0；Telea 仅可行性 |
| 下游效用（U0） | **未完整验证** — proxy PASS（约束过滤掉 naive V4；OccGS 有可测 RGB 信号）；**未跑 camera 3D det mAP** |

因此：**不 reject 整条 OccGS-Resim 路线**；也不宣称 U0 已打赢 real-only / naive GS。  
下一步应：**改方法强度（更强局部 completion + 标签重生）并扩到检测下游**，而非回到 2D 视频扩散（RF-01–18 仍关闭）。

---

## 2. Gate 汇总

| Gate | 状态 | 证据 |
|---|---|---|
| E0 | done | `docs/OCCGS_E0_ENV_MANIFEST.md` |
| G0 | PASS | `docs/OCCGS_THIRD_PARTY_AUDIT.md` |
| D0 | PASS | `docs/OCCGS_DATA_PREPARATION.md`；frozen mini 003/004/005 |
| B0 | PASS | `docs/OCCGS_RECONSTRUCTION_BASELINE.md`；`runs/.../b0_{2,3,4}_*` |
| O0 | PASS | `docs/OCCGS_OCCUPANCY_STATE.md` |
| S0 | PASS | `docs/OCCGS_COUNTERFACTUAL_PROTOCOL.md` §1 |
| C0 | PASS | 同上 §2；`data/occgs/reviews/c0_legality/` |
| L0 | PASS* | `runs/occgs_resim/l0_comp/{s0,s2}_v3/l0_feasibility.json`（*协议/ locality，非 SOTA 视觉） |
| U0 | partial | `runs/occgs_resim/u0_screen/u0_proxy_v1.json`（`u0_full_map_pass=false`） |
| D1 | done | 本文件 |

\* L0 使用 OpenCV Telea + hard composition，证明「只改 mask 内」可行；不声称超过纯 GS 的感知质量。

---

## 3. 关键实证数字

**B0 test PSNR/SSIM（冻结 StreetGS 30k，3 cams，t=0..79）**

| scene | test PSNR | test SSIM |
|---|---:|---:|
| 003 S0 | ~25.6 | ~0.80 |
| 005 S1 | ~20.2 | ~0.47 |
| 004 S2 | ~25.4 | ~0.70 |

**C0**：top-24 legal **24/24**；全可见 case 46/62。  
**L0**：outside-mask L1 **0.0**（hard compose）；mask 占比 ~1–2%。  
**U0 proxy**：OccGS V1/V2 accept_rate=1.0，naive V4 accept_rate=0.0；OccGS mean max \|ΔRGB\| ≈ 0.45。

---

## 4. 失败与限制（非 RF 重开）

1. JSON actor ≠ RigidNodes true_id — 必须 allowlist。  
2. 官方 mmseg sky mask 不可用 — 改 SegFormer-B5。  
3. 全图 L1 很小但局部峰值大 — 合法性指标须用 in-support / max。  
4. 完整 U0 mAP 未跑 — 磁盘/时间；记为 partial，不把 proxy 写成 mAP 胜利。  
5. L0 无视频扩散 — 遵守「不回整视频扩散」；弱 inpaint 只测 locality 协议。

---

## 5. 建议的下一步（人类决策用）

1. **扩 U0**：在冻结 cut-in strata 上训/评轻量 camera 3D detector 或 event classifier，正式对比 R / R+naiveGS / R+OccGS。  
2. **加强 L0**：geometry-conditioned local video inpaint（仍强制 hard composition）。  
3. **标签重生**：编辑后同步 2D/3D box、instance、occupancy。  
4. 场景池仍限 **nuScenes mini 可用前向完整 scene**，扩 train 前先解决 raw sweep 覆盖。

---

## 6. 收尾

- 文档已回填 V7 §A0.5 / §13、`RESEARCH_STATUS.md`、`EXPERIMENTS.md`。  
- `git commit`（不 push）后执行 `sync && /usr/bin/shutdown -h now`。
