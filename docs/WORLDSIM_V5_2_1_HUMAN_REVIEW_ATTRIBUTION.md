# WorldSim V5.2.1 人工复核归因与回测登记

> Task：`WS-V521-P11-HUMAN-ATTRIBUTION-01`
>
> Review denominator：18/18
>
> 归因层状态：`frozen`
>
> 机器事实源：`docs/run_manifests/worldsim-v5.2.1-human-review-attribution-v1/`

## 1. 结论

18 个代表性 case 的人工复核确认：原始 census 将两类性质不同的失败同时收入了 badcase registry。

```text
BASE_FAILURE
  基座全局重建已经崩溃，actor/boundary 指标被污染

M123_ELIGIBLE
  背景基本可用，动态 actor / boundary / ghost 是局部主问题

ATTRIBUTION_UNRESOLVED
  global 与 dynamic 同时失败，现有 panel 不能分解贡献
```

冻结计数：`BASE_FAILURE=9`、`M123_ELIGIBLE=8`、`ATTRIBUTION_UNRESOLVED=1`。

人工判断是视觉诊断假设，不是因果证明。只有完成 pixel→Gaussian、observability/ownership、flow/pose warp、visibility/correspondence bridge 后，才能把某个 case 写成 M1 或 M3 的 causal evidence。

## 2. 18 个 case 的正式映射

| # | Case ID | Split | 数据集定位 | Review axis | Research gate | 人工看到的问题 | 模块定位 |
|---:|---|---|---|---|---|---|---|
| 01 | `BC-ADGS-027fa166e7f1` | Discovery | nuScenes `0139/f92/c2` | Global | `BASE_FAILURE` | AD-GS 几乎纯白，StreetGS/GT 正常 | 基座哨兵 |
| 02 | `BC-ADGS-aa2e7ae9273c` | Discovery | nuScenes `0048/f47/c1` | Global | `BASE_FAILURE` | AD-GS 白粉色大面积崩溃 | 基座哨兵 |
| 03 | `BC-ADGS-5f105be6ff6c` | Confirmation | nuScenes `0230/f12/c2` | Global | `BASE_FAILURE` | AD-GS 均匀暗褐色，StreetGS 接近 GT | 基座哨兵，不是 M123 confirmation |
| 04 | `BC-STREETGS-12f3d4fb3c21` | Discovery | nuScenes `0048/f42/c1` | Global | `BASE_FAILURE` | 静态树木/叶片/纹理模糊 | 基座 static geometry/appearance |
| 05 | `BC-STREETGS-945caf2fc082` | Discovery | nuScenes `0242/f127/c1` | Global | `M123_ELIGIBLE` | 卡车拖影、重影、形态错误，背景正常 | M3 主、M1 次、M2 safety |
| 06 | `BC-STREETGS-62640d591ebc` | Confirmation | nuScenes `0242/f132/c0` | Global | `M123_ELIGIBLE` | 卡车错位、局部消失/重影，背景正常 | M3/M1 confirmation only |
| 07 | `BC-ADGS-f1cd1ab0eca6` | Discovery | nuScenes `0048/f92/c2` | Actor | `BASE_FAILURE` | 整图云雾状崩溃污染 actor metric | 禁止评价 M1 |
| 08 | `BC-ADGS-ec59140d6d30` | Discovery | nuScenes `0139/f77/c1` | Actor | `BASE_FAILURE` | AD-GS 几乎纯色，actor 只是局部 | 禁止评价 M1 |
| 09 | `BC-ADGS-d0ddc9f392e1` | Confirmation | nuScenes `0048/f192/c2` | Actor | `ATTRIBUTION_UNRESOLVED` | 全局 smear 与动态失败并存 | 先做 Base Validity/局部残差分解 |
| 10 | `BC-STREETGS-6132ad736366` | Discovery | nuScenes `0255/f27/c2` | Actor | `M123_ELIGIBLE` | 背景正常，远距小 actor 弱或缺失 | M1 observation scarcity 主靶点 |
| 11 | `BC-STREETGS-84bf82336ee0` | Discovery | nuScenes `0048/f167/c1` | Actor | `M123_ELIGIBLE` | 自行车透明、模糊、ghost | M1×M3 |
| 12 | `BC-STREETGS-68c77ab5bc76` | Confirmation | nuScenes `0255/f37/c2` | Actor | `M123_ELIGIBLE` | 多个远距小 actor，背景正常 | M1 confirmation only |
| 13 | `BC-ADGS-27f91d7e980e` | Discovery | nuScenes `0048/f142/c2` | Boundary | `BASE_FAILURE` | 整图暗且严重糊，boundary 指标被污染 | 基座哨兵 |
| 14 | `BC-ADGS-bbf3276525d0` | Discovery | nuScenes `0139/f87/c1` | Boundary | `BASE_FAILURE` | 全图蓝紫单色 | 完全排除出 M1 evidence |
| 15 | `BC-ADGS-f288cd88450c` | Confirmation | nuScenes `0048/f182/c1` | Boundary | `BASE_FAILURE` | 全局 smear 主导 | 基座哨兵，不是 M123 confirmation |
| 16 | `BC-STREETGS-b363a27e6231` | Discovery | nuScenes `0048/f47/c2` | Boundary | `M123_ELIGIBLE` | 施工人员/设备 ghost、边界扩散 | M1×M3，M2 safety |
| 17 | `BC-STREETGS-4305955afdfd` | Discovery | nuScenes `0139/f47/c1` | Boundary | `M123_ELIGIBLE` | 行人透明拖影/重复，背景稳定 | M3 主、M1 次 |
| 18 | `BC-STREETGS-7e9c9ecf93da` | Confirmation | nuScenes `0048/f17/c0` | Boundary | `M123_ELIGIBLE` | 施工动态区域模糊和局部 ghost | M1×M3 confirmation only |

## 3. 数据来源

全部 case 的逻辑数据集为 `nuScenes trainval`、DriveStudio 10 Hz、三前向相机。存储快照分为：

- `scene-0048 / scene-0139`：`/root/autodl-tmp/data/worldsim_v4/drivestudio_processed_10Hz/trainval`；
- `scene-0230 / scene-0242 / scene-0255`：`/root/autodl-tmp/data/dynamic_editing_v2/drivestudio_processed_10Hz/trainval`。

`cases.jsonl` 对每条记录冻结：

```text
dataset / dataset_split / dataset_storage_snapshot
scene / scene_index / frame / camera / unit_key
split_role / split_hash / evidence_tier
target_path / target_sha256
prediction_sha256 / dynamic_mask_sha256
canonical_panel_path / canonical_panel_sha256
metric_row_sha256
```

因此后续回测必须由 `case_id` 解析 exact 输入，不允许按文件名模糊匹配。

## 4. M123 seed denominator

### Discovery design（5）

```text
BC-STREETGS-945caf2fc082
BC-STREETGS-6132ad736366
BC-STREETGS-84bf82336ee0
BC-STREETGS-b363a27e6231
BC-STREETGS-4305955afdfd
```

### One-shot Confirmation（3）

```text
BC-STREETGS-62640d591ebc
BC-STREETGS-68c77ab5bc76
BC-STREETGS-7e9c9ecf93da
```

Confirmation case 禁止参与 arm/threshold/metric/predicate 设计。算法 candidate 冻结后只允许读取一次；失败也消耗 attempt。

## 5. 后续回测合同

任何算法 run 必须为每个输入 `case_id` 生成一条 `CASE_DELTAS.jsonl`：

```json
{
  "case_id": "BC-STREETGS-6132ad736366",
  "comparator": {"name": "U2_B3", "asset_sha256": "..."},
  "candidate": {"name": "TrackBayes_A5", "asset_sha256": "..."},
  "metrics": {
    "OWNERSHIP_BOUNDARY_F1": {"before": 0.0, "after": 0.0, "delta": 0.0, "status": "defined"},
    "ACTOR_LPIPS": {"before": 0.0, "after": 0.0, "delta": 0.0, "status": "defined"},
    "STATIC_PSNR_DROP": {"value": 0.0, "status": "defined"}
  },
  "bridge_verdict": "pass|fail|undefined",
  "split_role": "discovery"
}
```

报告必须先展示逐 case before/after，再做 scene-balanced aggregate；不得把 base sentinel、Discovery 和 Confirmation 混成一个均值。

## 6. M1/M2/M3 当前结论

- M1：`DIRECTION_SUPPORTED_CAUSAL_BRIDGE_PENDING`。#10/#12 与 observation scarcity 症状高度相容，但未证明失败 pixel 正好来自 low-observation/uncertain Gaussians；
- M3：`SYMPTOM_OVERLAP_STRONG_EXACT_TEMPORAL_BRIDGE_PENDING`。#05/#06/#17 的 ghost 与 pose/trajectory 高度相容，但当前 census 无合法 correspondence；
- M2：`SAFETY_LAYER_PENDING`。没有 case 支持 M2 是 repair 根因；geometry 仍不可比，M2 只能先做 abstention 合同。

正式执行计划见 `WORLDSIM_V5_2_M123_AUTORESEARCH_PLAN.md`。
