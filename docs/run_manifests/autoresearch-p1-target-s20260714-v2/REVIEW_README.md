# P1 P-UNC RGB/VAE target 人工复核

每个 `panels/*.mp4` 依次显示 Base、P-UNC compositor `X_dagger`、差分、mask、decode(hybrid)、decode(dilated hybrid)。重点检查 source duplication、ghosting、无深度occlusion order 的覆盖、纹理拉伸和运动方向。

复制 `reviews.template.jsonl` 为 `reviews.jsonl`，逐项填 `valid`/`invalid`/`uncertain`，然后以 `--aggregate-only` 重跑。未完成 8 个 review 不得将 P1 标记 pass。
