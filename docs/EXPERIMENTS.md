# Motion-Proj 实验事实源

这里只记录进入比较表的实验、失败结论和最终参数选择。原始 trial 指标保存在各 run 的 `metrics.jsonl`、SQLite 与 summary 中，不向 Git 追加流水日志。

## 比较实验

| Run ID | 状态 | commit | 数据/cache fingerprint | 方法 | seed | 关键结果 | 证据路径 | 结论 |
|---|---|---|---|---|---:|---|---|---|
| `legacy-mini-v1-2000` | completed | pre-2026-07-11 | legacy/unknown | SVD LoRA Motion-Proj V1 | unknown | 仅确认 2,000 step 链路和 adapter 保存成功 | `/root/autodl-tmp/runs/motionproj_v1` | 不进入正式主表；需固定 seed 生成比较 |

## 可信度验收

| 日期 | Run ID | 状态 | commit | 协议/数据 | seed | 关键结果 | 证据路径 | 结论 |
|---|---|---|---|---|---:|---|---|---|
| 2026-07-11 | `p0-geometry-synth100-s20260711-8c8afef4-e109eb12` | completed | `8c8afef4` | `synthetic-object-track-v1` | 20260711 | 95/100 投影后目标轨迹能量下降，finite/mask 有效率 100%，最低 eligible fraction 78.44%；`temporal_gap` 为 15/20，其余四类均为 20/20 | `/root/autodl-tmp/runs/p0-geometry-synth100-s20260711-8c8afef4-e109eb12/{resolved.yaml,manifest.json,metrics.jsonl,summary.json,COMPLETE}`；config fingerprint `e109eb12`；cache `not-applicable:synthetic-object-track-v1` | 通过 P0 的 70% 合成错误验收；仅证明 `E_obj + 0.1 * E_prior` 下降，不外推为 RGB/FVD 或驾驶可控性结论 |

## 失败与拒绝结论

| 日期 | Run/Trial | 状态 | 结论 | 证据 | 后续 |
|---|---|---|---|---|---|
| 2026-07-11 | legacy cache | rejected | 旧 cache 无 schema/完成标记/fingerprint，且部分历史 metadata 曾含 NaN，不能作为可信投影证据 | `/root/autodl-tmp/cache/projection/*/metadata.json` | 由新 writer 幂等重建 |
| 2026-07-11 | `p0-geometry-mini5-5ff8e8c0-96306871` | completed | 首轮审计运行完成但验收失败，唯一失败项为 `eligible_gate`；数值已由 clean commit 正式 run 完整复现 | `/root/autodl-tmp/runs/p0-geometry-mini5-5ff8e8c0-96306871/{summary.json,manifest.json,metrics.jsonl}`；manifest commit `5ff8e8c0`，但审计新增文件运行时尚未被 Git 跟踪 | 保留为溯源不完整的历史 run，不覆盖；正式证据改用 `p0-geometry-mini5-8c8afef4-96306871` |
| 2026-07-11 | `p0-geometry-mini5-8c8afef4-96306871` | completed | clean commit 正式审计仍仅 `eligible_gate` 失败：nuScenes mini 前 5 个 CAM_FRONT clip 的有效比例均值 62.3507%、最小值 54.7917%，0/5 达到 70%；深度-LiDAR Pearson 均值/最小值为 0.8780/0.8480，污染检测率、finite rate 和 track energy improvement rate 均为 100%；不调整 70% 门槛 | `/root/autodl-tmp/runs/p0-geometry-mini5-8c8afef4-96306871/{resolved.yaml,manifest.json,metrics.jsonl,summary.json,COMPLETE}`；commit `8c8afef4`；config fingerprint `96306871`；split `v1.0-mini:CAM_FRONT:first-5`；seed 1234；cache `not-applicable:online-audit` | 接受 62.35% 为当前真实有效投影覆盖率；将覆盖率作为适用范围披露，不通过调阈值改写事实 |

## 参数选择

| 日期 | 范围 | 已锁定选择 | 依据 |
|---|---|---|---|
| 2026-07-11 | 开发期容量 | LoRA rank 16 | 避免把容量变化混入方法比较 |
| 2026-07-11 | 调参范围 | 见 `docs/CVPR2027_PLAN.md` | 固定搜索空间，防止事后扩域 |

## 登记规则

- 状态只使用 `queued/running/retrying/completed/pruned/failed`。
- promoted、rejected、failed trial 均保留，不复用 run 目录、不覆盖历史结果。
- 每条正式结论必须同时给出 commit、配置、数据/cache fingerprint、seed 和证据路径。
- 文档中的汇总必须能从运行目录或注册表重新生成；不手工修饰原始指标。
