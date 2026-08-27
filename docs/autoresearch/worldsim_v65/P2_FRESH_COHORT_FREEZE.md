# P2 Fresh Representation-Selection Cohort Freeze

冻结于任何 P2 model-score/target quality read 之前。六个 scene 均未出现在 V6.1–V6.4 method configs；选择只使用
nuScenes description、frame count 与冻结 IR-WM temporal-info key availability 等 capability metadata。

初始 cohort `0520/0781/0800/0996/0443/0106` 在预处理期间被标记为
`superseded_pre_read_capability_ineligible`：`0520/0781/0800/0106` 不在冻结 IR-WM 的 700 个 temporal-info key
内。前三个 native worker 在 sidecar/model-score/quality 生成前以 `KeyError` 退出；P2 formal read 仍为 false。
为避免重新生成 temporal infos 造成官方数据管线/schema 漂移，只在同一 metadata-only 规则内换成后端可直接支持的
fresh scenes。失败 run 与首版 freeze 均保留为 `V65-F02` 证据。

| scene | index | metadata context | preparation |
| --- | ---: | --- | --- |
| scene-0996 | 754 | night, bus stop, cyclist, pedestrians | extract/preprocess |
| scene-0443 | 357 | rain, crane, overtake, pedestrian | extract/preprocess |
| scene-0002 | 1 | intersection, pedestrians, waiting vehicle, parked motorcycle | extract/preprocess |
| scene-0043 | 40 | busy intersection, oncoming traffic, construction | extract/preprocess |
| scene-0023 | 22 | overtake parked car, turning bus, motorcycle | extract/preprocess |
| scene-0072 | 69 | wait at intersection, running pedestrian | extract/preprocess |

每 scene 固定 12 targets：`17,32,...,182`，总 denominator=72 cases。唯一正式 arms：frozen q0 与 frozen
P1R monotone task risk。Primary gate：fixed-route risk `>=10%` relative reduction、strict lower scene support
`>=5/6`、任一 scene relative regression `<=5%`；同时 non-route `<=5%`、coverage exact matched、monotone
semantics violation=0。失败后不读取第二组 P2 selection，不扩 attention。
