# E0 CoTracker3 独立 evaluator 人工复核

仅查看 `track_overlay/*.mp4`。所有点来自 CoTracker3 first-frame grid，颜色为 evaluator-onlycamera-compensated strata，不是 RAFT/P0/P1 track。`valid`：大多数点贴合纹理且遮挡/低纹理失效被 visibility 正确标记；`invalid`：系统性漂移/伪连续；`uncertain`：无法判定。

复制模板为 `reviews.jsonl` 后以 `--aggregate-only` 更新。
