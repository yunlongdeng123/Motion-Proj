# WorldSim V6.4 Compact Fresh Native Cohort

- Task: `WS-V64-P2-FRESH-NATIVE-SIDECAR-01`
- Hypothesis: `WS-V64-H-P2-001`
- Status: `recovery preregistered / r3 not run`
- Seed: `0`

## Fresh boundary

本轮 fresh 仅排除 V6.1–V6.3 已读取 quality 或作为正式 train/selection/legacy 的 scene。扫描三版当前计划、
autoresearch 文档与配置得到 21 个排除 scene。候选还必须存在于冻结的
`nuscenes_temporal_infos_train.pkl`；这是 IR-WM current-state extractor 的输入能力条件，不是质量筛选。

候选选择只使用 scene description、帧数、传感器文件完整性和 temporal metadata membership，不使用
Occupancy/UQ/false-safe/model quality。

## Pre-quality erratum

r2 暴露最初六场景中的 val-split scene 不在冻结 train temporal metadata 中。r2 为 blocked partial：只有
`scene-0230`完成 12 个 native units（528 MiB run leaf，scene worker peak `4.1305 GiB`），`scene-0100`与
`scene-0632`在 scene lookup 处 `KeyError`，其余场景未形成完整 denominator；没有 target evidence 或质量读取，
不得作为 canonical，也不得把其部分 sidecar 混入 r3。

恢复在任何 fresh quality read 前完成：改用同样未进入 V6.1–V6.3 quality ledger、且属于 train temporal metadata 的
六个场景。两个 evaluation scene 仅从本机已有 raw nuScenes 通过官方 DriveStudio preprocessing 物化；不下载额外数据，
不改变模型、target frame、seed、分母或 UQ 方法。

## Frozen cohort

Fit：

- `scene-0139`：白天、行人/摩托车；
- `scene-0230`：施工与停放车辆；
- `scene-0255`：停车区域；
- `scene-0994`：夜间、低交通、暗光行人。

Evaluation：

- `scene-0359`：白天公交站、行人与停车；
- `scene-0998`：夜间环岛、公交与行人横穿。

每 scene 固定 12 个 target frame：
`17,32,47,62,77,92,107,122,137,152,167,182`，共 72 units。evaluation target 在 UQ decision 生成前不读取。

## Extraction

复用已经通过 V6.3 76-unit formal 的 IR-WM native extractor，输出真实 `17D logits + 256D BEV latent`及 entropy/margin；
不使用 prototype，不训练 backbone。预计约 3.4 GiB，双 worker 峰值上界约 8.3 GiB，当前单卡 3090 和约 60 GiB
磁盘余量足够，不需要多卡。

本阶段不增加 smoke；先用官方 DriveStudio 流程预处理 evaluation 两场景，再用全新 r3 leaf 直接运行 72-unit formal。
r2 保留为失败证据，不删除、不覆盖、不复用。
