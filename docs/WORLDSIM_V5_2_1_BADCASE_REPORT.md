# WorldSim V5.2.1 Base Badcase Report

## 结论

两套 exact 基座均完成同帧 census，coverage terminal=`complete_full`。Discovery 每基座 `576` views，
Confirmation 每基座 `126` views；没有读取 fresh validation/test/KITTI，Stage H/BKI 未执行，也没有算法 candidate。

可合法冻结的 failure 轴为 `GLOBAL_RGB / ACTOR_RGB / BOUNDARY`；geometry 因两基座 depth 语义不可比而 undefined，
occlusion 因缺 visibility transition annotation 而 undefined，temporal 只有 unwarped proxy，identity/observability 缺 exact denominator。
这些 undefined 是 census 结论，不以 proxy 填补。

## 分母与分类

- AD-GS Discovery：`{'cases': 169, 'class_counts': {'B-ACTOR': 35, 'B-BOUNDARY': 29, 'B-RGB-GLOBAL': 150}, 'scenes': ['scene-0048', 'scene-0139', 'scene-0230', 'scene-0242', 'scene-0255']}`
- StreetGS Discovery：`{'cases': 145, 'class_counts': {'B-ACTOR': 46, 'B-BOUNDARY': 16, 'B-RGB-GLOBAL': 112}, 'scenes': ['scene-0048', 'scene-0139', 'scene-0230', 'scene-0242', 'scene-0255', 'scene-0994']}`
- AD-GS Confirmation：`{'cases': 31, 'class_counts': {'B-ACTOR': 9, 'B-BOUNDARY': 5, 'B-RGB-GLOBAL': 28}, 'scenes': ['scene-0048', 'scene-0139', 'scene-0230', 'scene-0242', 'scene-0255']}`
- StreetGS Confirmation：`{'cases': 32, 'class_counts': {'B-ACTOR': 11, 'B-BOUNDARY': 5, 'B-RGB-GLOBAL': 25}, 'scenes': ['scene-0048', 'scene-0230', 'scene-0242', 'scene-0255']}`

P3 以每场 q10 后跨场等权聚合冻结阈值；各轴独立排名，没有综合 scalar。Confirmation 未重拟合阈值、未改 K、未改判词。
class-level verdict 见 `P9_CONFIRMATION_VERDICT.json`。

## Localization 边界

P5 使用完整 census denominator；actor area、actor/static 与 boundary/actor residual ratio 可解释，distance、speed、LiDAR support、
visibility、occlusion transition 全部如实缺测。Spearman 只作相关性，不写因果。

## 18-case 人工复核归因层

用户指定评审者已完成代表性 review package 的 `18/18` 逐图复核。该复核不回写原始
`BADCASE_REGISTRY.jsonl`，而是在其上冻结独立 attribution layer：

- `BASE_FAILURE=9`：global reconstruction 已失败，actor/boundary 指标不具备 M123 因果归因资格；
- `M123_ELIGIBLE=8`：静态背景基本可用，动态 actor、boundary 或 ghost 是局部主问题；
- `ATTRIBUTION_UNRESOLVED=1`：global 与 dynamic 同时失败，需先做 Base Validity/局部残差分解。

8 个 eligible case 均来自 StreetGS，其中 Discovery design=`5`、one-shot Confirmation=`3`。当前最强 seed 为：

- M1 observation scarcity：`BC-STREETGS-6132ad736366`、`BC-STREETGS-68c77ab5bc76`；
- M3 actor motion：`BC-STREETGS-945caf2fc082`、`BC-STREETGS-62640d591ebc`、`BC-STREETGS-4305955afdfd`；
- M1×M3 boundary coupling：`BC-STREETGS-b363a27e6231`、`BC-STREETGS-7e9c9ecf93da`，另有
  `BC-STREETGS-84bf82336ee0` 的 dynamic decomposition/temporal ghost。

这些是视觉诊断假设，不是 causal proof。完整 case→dataset/split/path/hash→问题→模块→回测指标映射见
[`WORLDSIM_V5_2_1_HUMAN_REVIEW_ATTRIBUTION.md`](WORLDSIM_V5_2_1_HUMAN_REVIEW_ATTRIBUTION.md) 与
`docs/run_manifests/worldsim-v5.2.1-human-review-attribution-v1/cases.jsonl`。
