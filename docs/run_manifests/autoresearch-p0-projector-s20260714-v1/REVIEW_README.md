# P0 point-track tube 人工复核

每张 `panels/*.png` 显示 Base rollout 的 frame-0/末帧、observed/projected 轨迹、confidence 推导的 uncertainty 及 before/after velocity、acceleration、jerk。

1. 复制 `reviews.template.jsonl` 为 `reviews.jsonl`。
2. 对每个 case 将 `verdict` 填为 `valid`、`invalid` 或 `uncertain`，并补充 notes。
3. 重新执行同一命令并增加 `--aggregate-only`。

`valid` 仅表示该 point-track tube 的运动修正物理上可辨识；它不表示 dataset object-instance监督，也不允许根据人工复核改写缓存。
