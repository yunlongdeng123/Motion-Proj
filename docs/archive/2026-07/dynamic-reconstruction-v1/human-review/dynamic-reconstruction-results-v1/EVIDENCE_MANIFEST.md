# 机器终止证据清单

| 阶段 | 状态 | 证据 |
|---|---|---|
| M4 | done / all gates passed | `/root/autodl-tmp/runs/dynamic_recon/DR-M4-ADGS-6SCENE-01/20260728T141204__aggregate6-s0-wm3090/` |
| M5 | blocked / pointops2 packaging | `/root/autodl-tmp/runs/dynamic_recon/DR-M5-DGGT-NUSC-01/20260729T133346__native-nusc-s0-wm3090/` |
| M5 common-observation | not run / upstream blocked | 216-target mapping only passed read-only pre-audit |
| M6 | done / 6-scene stable identity failure | `/root/autodl-tmp/runs/dynamic_recon/DR-M6-STRESS-01/20260729T145645__identity-audit-s0-wm3090/` |
| M7 | execution done / research rejected | `/root/autodl-tmp/runs/dynamic_recon/DR-M7-HYPOTHESIS-01/20260729T145748__novelty-audit-s0-wm3090/` |
| M8 | rejected / not authorized | M7 `downstream_stop.json` |
| M9 | rejected / not triggered | M7 `downstream_stop.json` 与本目录 README |

## 关键文件

```text
M4: summary.json, metrics.jsonl, terminal.json
M5: manifest.json, resolved.yaml, stages/, metrics.jsonl, summary.json, terminal.json
M6: manifest.json, resolved.yaml, track_audit.json, failure_matrix.json,
    edit_coverage.json, pseudo_hole_coverage.json, noise_coverage.json,
    metrics.jsonl, summary.json, terminal.json
M7: manifest.json, resolved.yaml, novelty_matrix.json, downstream_stop.json,
    metrics.jsonl, summary.json, terminal.json
```

本包没有 `reviews.template.jsonl`：没有 M8 method samples 时，M9 人工任务未触发，空模板会暗示存在待评输入。
