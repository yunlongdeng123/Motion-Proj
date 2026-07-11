# Motion-Proj 实验事实源

这里只记录进入比较表的实验、失败结论和最终参数选择。原始 trial 指标保存在各 run 的 `metrics.jsonl`、SQLite 与 summary 中，不向 Git 追加流水日志。

## 比较实验

| Run ID | 状态 | commit | 数据/cache fingerprint | 方法 | seed | 关键结果 | 证据路径 | 结论 |
|---|---|---|---|---|---:|---|---|---|
| `legacy-mini-v1-2000` | completed | pre-2026-07-11 | legacy/unknown | SVD LoRA Motion-Proj V1 | unknown | 仅确认 2,000 step 链路和 adapter 保存成功 | `/root/autodl-tmp/runs/motionproj_v1` | 不进入正式主表；需固定 seed 生成比较 |

## 失败与拒绝结论

| 日期 | Run/Trial | 状态 | 结论 | 证据 | 后续 |
|---|---|---|---|---|---|
| 2026-07-11 | legacy cache | rejected | 旧 cache 无 schema/完成标记/fingerprint，且部分历史 metadata 曾含 NaN，不能作为可信投影证据 | `/root/autodl-tmp/cache/projection/*/metadata.json` | 由新 writer 幂等重建 |
| 2026-07-11 | `p0-geometry-mini5-5ff8e8c0-96306871` | completed | 审计运行完成但验收失败，唯一失败项为 `eligible_gate`：nuScenes mini 前 5 个 CAM_FRONT clip 的有效比例均值 62.3507%、最小值 54.7917%，0/5 达到 70%；深度-LiDAR Pearson 均值/最小值为 0.8780/0.8480，污染检测率、finite rate 和 track energy improvement rate 均为 100%；不调整 70% 门槛 | `/root/autodl-tmp/runs/p0-geometry-mini5-5ff8e8c0-96306871/{summary.json,manifest.json,metrics.jsonl}`；manifest commit `5ff8e8c0`，但本次审计新增文件运行时尚未被 Git 跟踪，代码溯源待补；config `configs/eval/nuscenes_mini_geometry_p0.yaml`（fingerprint `96306871`）；split `v1.0-mini:CAM_FRONT:first-5`；seed 1234；cache `not-applicable:online-audit` | 接受真实有效率事实；提交审计代码后重跑以闭环溯源，再建立 100 个合成错误集验证投影后能量下降 |

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
