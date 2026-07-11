# P1 投影 Target 人工检查协议

协议 ID：`projection-target-manual-v1`
任务 ID：`P1-PROJECTION-01`

## 目的

验证 V2 投影 target（`x_dagger`）在 synthetic corruption 场景下，相对 corrupted 输入 `y` 是否“更合理”。该结论独立于 P0 的轨迹能量下降指标，不得用能量代理替代人工 verdict。

## 检查包结构

每个 run 目录包含：

- `panels/<case_id>.mp4`：`[corrupted y | x_dagger | reliability mask]`
- `cases/<case_id>.json`：corruption、能量、eligible fraction 等元数据
- `reviews.template.jsonl`：待填写模板
- `reviews.jsonl`：正式人工 verdict（由检查者创建）
- `summary.json`：聚合结果

## Verdict 定义

| Verdict | 含义 | 是否计入 70% 门槛 |
|---|---|---|
| `reasonable` | 相对 corrupted y，x_dagger 静态漂移/目标轨迹/闪烁更少，且无显著新伪影 | 是（分子） |
| `not_reasonable` | 未改善或引入明显新伪影 | 是（分母） |
| `borderline` | 改善与退化并存，无法明确归类 | 否 |

验收条件：全部 case 已 review，且 `reasonable / (reasonable + not_reasonable) >= 0.70`。

## Synthetic case 构造

1. 从 nuScenes mini 真实 clip 运行 auditor。
2. 选取可见帧最多的轨迹，注入 P0 同类 corruption（`center_impulse` 等五类）。
3. 左侧 `y` 为 corruption 直接渲染结果；右侧 `x_dagger` 为 `DynamicsProjector` 输出。

## 运行命令

```bash
# 导出检查包
python -m motion_proj.eval.projection_inspection \
  --config configs/eval/projection_manual_p1.yaml

# 填写 reviews.jsonl 后重新聚合
python -m motion_proj.eval.projection_inspection \
  --config configs/eval/projection_manual_p1.yaml \
  --run-id <run_id> \
  --aggregate-only
```

## 边界说明

- P0 已证明合成 100-case 能量下降；P1 不重复该结论。
- 真实 mini 几何 eligible fraction 均值约 62.35%，仅作适用范围披露，不用于改写 P1 verdict 门槛。
- `temporal_gap` 在 P0 中为 15/20 能量改善，人工检查时应单独记录 borderline 比例。
