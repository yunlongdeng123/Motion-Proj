# WorldSim V6.4 Compact Fresh Native Cohort

- Task: `WS-V64-P2-FRESH-NATIVE-SIDECAR-01`
- Hypothesis: `WS-V64-H-P2-001`
- Status: `preregistered / not run`
- Seed: `0`

## Fresh boundary

本轮 fresh 仅排除 V6.1–V6.3 已读取 quality 或作为正式 train/selection/legacy 的 scene。扫描三版当前计划、
autoresearch 文档与配置得到 21 个排除 scene；本机已有 DriveStudio processed、且不在该排除集的候选共有 9 个：

`scene-0100, scene-0139, scene-0230, scene-0255, scene-0520, scene-0632, scene-0781, scene-0800, scene-0994`。

候选选择只使用 scene description、帧数和传感器文件完整性，不使用 Occupancy/UQ/false-safe/model quality。

## Frozen cohort

Fit：

- `scene-0100`：白天路口、骑行者/行人；
- `scene-0230`：施工与停放车辆；
- `scene-0632`：雨天工业区；
- `scene-0781`：公交/行人/低交通。

Evaluation：

- `scene-0800`：行人、公交、路口；
- `scene-0994`：夜间、低交通、暗光行人。

每 scene 固定 12 个 target frame：
`17,32,47,62,77,92,107,122,137,152,167,182`，共 72 units。evaluation target 在 UQ decision 生成前不读取。

## Extraction

复用已经通过 V6.3 76-unit formal 的 IR-WM native extractor，输出真实 `17D logits + 256D BEV latent`及 entropy/margin；
不使用 prototype，不训练 backbone。预计约 3.4 GiB，双 worker 峰值上界约 8.3 GiB，当前单卡 3090 和约 60 GiB
磁盘余量足够，不需要多卡。

本阶段不增加 smoke；prereg commit/push 后直接运行 72-unit formal。
