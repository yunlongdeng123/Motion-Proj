# V7 P3-B AV2 Camera Evidence Freeze

## Objective

把 P3-A 的 Actor-local point/surface/ray 证据放回真实 AV2 RGB 与相机深度坐标系中，同时保持案例、相机和 crop
不受最终画面或方法质量影响。该阶段只生成证据，不训练、不校准、不回扫 P3-A。

## Frozen source

- source run：`run://worldsim_v7/WS-V7-P3-AV2-HARD-EVIDENCE-01/20260902T143000Z__ray-certified-s0-r3`；
- source cases：`VISUAL_CASES.jsonl` 全部 30 行；
- main：原 8 行；supplement：原 30 行；
- source Actor evidence：`ACTOR_HARD_EVIDENCE.jsonl`，只用于 panel annotation，不参与 selection。

## Camera and crop policy

1. 按 config 中固定的七个 AV2 ring cameras 顺序遍历；
2. 只投影 query Actor 的真实 LiDAR returns，统计 in-frame point count；
3. 取 count 最大相机，平局取固定顺序更早者；
4. 选择结束后才 decode RGB；
5. crop 只由 query projection bbox、固定 padding 与 minimum size 决定。

该设计迁移 AV2 官方 `AV2SensorDataLoader.project_ego_to_img_motion_compensated` 的时间同步与 calibration boundary。
它不使用 RGB appearance、hazard、P3 metric 或 action success 选相机/裁剪。

## Outputs and claim boundary

- RGB + observed AV2 query returns；
- paired ghost/duplicate/flicker overlay；
- KEEP/PROJECT/COMPLETE/UNKNOWN overlay；
- official motion-compensated sparse LiDAR camera depth；
- before/after MP4，全部 30 cases 写出。

paired artifacts 与视频 flicker 明确是 synthetic compiler-contract probes。输出是 calibrated evidence overlay，不是
photorealistic reconstruction；真实 generalization/geometry claim 仍来自冻结 AV2 target-only depth/ray/surface result。
