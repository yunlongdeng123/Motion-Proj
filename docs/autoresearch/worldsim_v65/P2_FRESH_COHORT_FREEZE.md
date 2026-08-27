# P2 Fresh Representation-Selection Cohort Freeze

冻结于任何 P2 model-score/target quality read 之前。六个 scene 均未出现在 V6.1–V6.4 method configs；选择只使用
nuScenes description、frame count、processed availability 等 metadata。

| scene | index | metadata context | preparation |
| --- | ---: | --- | --- |
| scene-0520 | 410 | pedestrian, parked vehicles | reuse processed |
| scene-0781 | 604 | bus, low traffic, pedestrians/crosswalk | reuse processed |
| scene-0800 | 620 | pedestrians, cones, bus, intersection | reuse processed |
| scene-0996 | 754 | night, bus stop, cyclist, pedestrians | extract/preprocess |
| scene-0443 | 357 | rain, crane, overtake, pedestrian | extract/preprocess |
| scene-0106 | 88 | construction, intersection, cones/barriers | extract/preprocess |

每 scene 固定 12 targets：`17,32,...,182`，总 denominator=72 cases。唯一正式 arms：frozen q0 与 frozen
P1R monotone task risk。Primary gate：fixed-route risk `>=10%` relative reduction、strict lower scene support
`>=5/6`、任一 scene relative regression `<=5%`；同时 non-route `<=5%`、coverage exact matched、monotone
semantics violation=0。失败后不读取第二组 P2 selection，不扩 attention。
