# Motion-Proj ReSim C1 V6 终报（单卡 screening）

> **日期**：2026-07-19（2026-07-20：与活跃 docs / archive 链接对齐）  
> **计划**：[`MOTION_RESIM_C1_AUTORESEARCH_PLAN_V6.md`](MOTION_RESIM_C1_AUTORESEARCH_PLAN_V6.md)  
> **结论**：`H1 rejected`（action response 不足）；不进入 C1P/C1S  
> **硬件**：单张 RTX 4090 24 GB  
> **上游归档**：V5 / Route Pivot 审计见 [`archive/2026-07/`](archive/2026-07/README.md)

## 1. 一句话

公开 ReSim `exp0_no_carla` 在冻结的 10-context E-vs-F screen 上**未能**通过预注册 action 门禁（E 仅 3/8 优于 F），因此 v6 idea screening 以 H1 失败结束。

## 2. 任务结果

| Task | 状态 | 关键数字 / 证据 |
|---|---|---|
| C1A | done | 独立 `resim` 环境、EMA/T5/VAE、磁盘就绪 |
| C1B-00 | done | L0 OOM→L1 256×448；双重复像素一致；峰值 ~23.6 GiB |
| C1B-01 | done | kinematic-lateral proxy：BA `0.778`、turn `0.750`、ρ `0.953` |
| C1B-02 | **rejected** | E wins `3/8`，median improvement `-0.107`；future/quality/stationary 通过 |
| C1B-03 / C1P / C1S | not run | 依赖 H1 pass |

正式 reject run：`/root/autodl-tmp/runs/resim_c1_v6/C1B-02/resim-c1b02-screen-s20260719-v2`

## 3. 为何不是 inconclusive

- 工程有效：20/20 采样成功，输出 shape/质量检查通过；
- proxy 已在真实 nuScenes 上过门槛（非未校准乱评）；
- 失败集中在 **action error 方向**（E 相对 F），不是磁盘/OOM/确定性；
- 按 V6 §9：`C1B 有效但不过 gate → H1 rejected，停止`。

## 4. Reviewer 质疑简答

1. **无 CARLA**：全部结论限于 expert/action-mask 支持内；未声称非专家物理。  
2. **是否 seed/history 噪声**：E/F 同 seed；history 差大多低于 null band；future 像素差大但 action 不对。  
3. **actor physics**：未作绝对 actor 准确率主张；actor 仅观察项。  
4. **偏好支持**：未进入；H1 失败即停止。  
5. **scorer/evaluator 隔离**：action 用 RAFT+affine proxy；CoTracker 只作 safeguard。  
6. **低运动捷径**：stationary 控制未超冻结 p95。  
7–10. **adapter/单对/held-out/多卡**：未训练；全程单卡；失败与排除已记账。

## 5. 负结论 ID

- `RF-17`：旧四类 ridge proxy 不可辨识（已由 kinematic-lateral 重开并通过）  
- `RF-18`：E-vs-F action response 不足（本终报主结论）

## 6. 下一步（需新计划授权）

v6 不再自动续跑。若继续，必须新预注册协议，并改变问题结构（例如完整 checkpoint、官方 IDM、更长 horizon），不得靠降 7/8 门槛或扩 seed 救场。
