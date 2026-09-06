# WorldSim V7.1 M18 — Fresh AV2 Zero-Shot Protocol

状态：`FROZEN / WAITING_DOWNLOAD`（2026-09-04）  
任务：`WS-V71-M18-FRESH-AV2-ZERO-SHOT-01`

## Frozen objects

- source model: `20260905T023000Z__m18-categorical-return-s71120-r1`；
- point comparator: `20260904T202000Z__m8-temporal-frame-s71110-r2`；
- cohort: `configs/worldsim_v71/av2_zero_shot_cohort_v1.json` exactly 20 fresh AV2 Sensor val logs；
- dataset state: `/root/autodl-tmp/data/av2/v71_download_state`；
- compiler: frozen V7 P2 compiler plus V7.1 vehicle categories and frame-count overrides；
- inference: M8 child points and M18 categorical CDF median on identical actors/rays。

## Evaluation contract

1. Wait for each cohort log's `.complete` marker; never inspect an incomplete directory.
2. Compile every frozen log and keep every evaluable actor; no failed-log deletion.
3. Write per-log partial rows for crash recovery, but do not read any partial physical metric.
4. After 20/20 only, aggregate M8 point surface and M18 categorical first-return metrics.
5. Require M8 external point gates: hazard early reduction `>=5%`, Chamfer delta `<=1mm`, actor/hazard retention=`100%`.
6. Require M18 categorical hazard early reduction relative to M8 `>=5%` and all hit delta `>=-1pp`.
7. No fine-tuning, calibration, categorical temperature/median threshold selection, log deletion or retry based on quality.

## Claim boundary

Pass supports zero-shot transfer of the frozen geometry/first-return mechanism from nuScenes to AV2. It does not prove Waymo transfer,
no-return modeling, clear-subgroup safety, image fusion, or dynamic/static separation. Failure closes external transfer for this M18 checkpoint and
cannot be repaired on the same cohort.
