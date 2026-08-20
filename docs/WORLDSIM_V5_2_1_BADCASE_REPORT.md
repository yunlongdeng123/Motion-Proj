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
