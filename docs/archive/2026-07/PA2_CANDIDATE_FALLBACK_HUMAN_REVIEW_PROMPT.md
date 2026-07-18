# PA2-CAND-03D earlier-fork 结构盲审提示词

## 1. 评测目的与非目标

本轮只判断 fork=0.4 的 8 个新 condition 是否仍保持同一驾驶场景、主体身份和布局，决定 SVD
common-prefix fallback 是否具有结构合法性。它**不评物理 winner**、不比较哪列运动更平滑、不推断机器
oracle 对错，也不选择训练样本。

## 2. 盲法与禁止读取的信息

- 只观看 run 内 `panels/*.mp4`；每个 case 有 A–E 五列，包含同一 condition 的 Base 与四条 sibling，列顺序已固定随机化。
- 禁止读取 `review_cases.private.json`、`candidate_manifest.jsonl`、`oracle_graphs.jsonl`、`machine_summary.json`、`resolved.yaml`、trace、seed、metric 或 candidate ID。
- 不要根据列位置猜 Base；不要把“运动幅度更小”当作结构合法。

## 3. 素材范围与观看方式

- 共 8 个 case，每个 14 帧、7 FPS；先完整播放，再逐帧查看首帧、中段和末帧。
- 重点观察道路拓扑、车道线/护栏、主要车辆与行人身份、相机视角、物体相对布局。
- 色彩或轻微纹理差异本身不等于构图变化；若造成主体消失、身份切换或几何破裂，应判 `invalid`。

## 4. Verdict 定义与优先级

按以下优先级每 case 只填一个 verdict：

1. `invalid`：任一列不可解码、黑屏/严重过曝、灾难性闪烁、主体身份崩溃、几何撕裂到无法判断同场景。
2. `different_composition`：素材可看，但至少一列发生道路布局、视角、主体身份/数量或关键相对位置的明显改变，已不是同一场景的合理未来分支。
3. `same_scene`：五列保持同一首帧条件、道路布局、视角和主要主体身份；后续轨迹/纹理可不同，但仍是同一场景的可接受未来。
4. `uncertain`：看完并逐帧检查后，仍无法在 `same_scene` 与前两种失败间稳定判断。

边界例：车辆轻微尺度漂移但身份/位置连续可判 `same_scene`；车辆突然变成另一车型、道路分叉/车道数量变化判
`different_composition`；严重融化或多帧不可辨认判 `invalid`。不要因某列运动较慢、加速度较低或更模糊就选它为 winner。

## 5. JSONL 填写格式

复制 `reviews.template.jsonl` 为 `reviews.jsonl`，保留 8 行与 case_id。每行：

```json
{"case_id":"pa2-cand-review-00","verdict":"same_scene","failure_reasons":[],"reviewer":"你的名字","notes":"简短可审计说明"}
```

`failure_reasons` 可选值：`layout_change`、`identity_change`、`camera_change`、`geometry_break`、
`flicker_or_exposure`、`unreadable`；`same_scene` 时应为空。不得填写 `pending`，不得新增/删除 case。

## 6. 聚合阈值

- 必须完成 8/8；
- `same_scene >= 7/8`；
- `different_composition + invalid = 0`；
- 最多允许 1 个 `uncertain`。

任何结构失败都拒绝 SVD sibling route；通过也只解锁冻结 oracle 的后续 yield/strict-precision 审计，不会直接训练或自动转 preference label。

## 7. 完成后的精确命令

```bash
cd /root/autodl-tmp/motion_proj
source /root/miniconda3/etc/profile.d/conda.sh
conda activate /root/autodl-tmp/envs/motionproj
export PYTHONPATH=.
python -m motion_proj.diagnostics.physics_preference_candidate_fallback \
  --config configs/diagnostics/physics_preference_candidate_fallback.yaml \
  --aggregate-only
```

聚合只读取人工 JSONL 与已冻结 machine artifacts，不加载模型、不重算候选、不代填 verdict。
