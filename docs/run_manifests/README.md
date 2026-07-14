# Run 轻量归档

从 `/root/autodl-tmp/runs/` 摘取的 `resolved.yaml`、`manifest.json`、`summary.json`、
`machine_summary.json` 与人工 review 说明，便于换机后对照 commit 与实验 fingerprint。

**不是**完整实验证据：metrics.jsonl、panel mp4、checkpoint 等仍在 run 目录，需 rsync 或重跑。

| 目录 | 说明 |
|---|---|
| `autoresearch-c0-conditioning-s20260714-v2` | C0 SVD parity 通过 |
| `autoresearch-p0-projector-s20260714-v1` | P0 machine pass，待人工 review |
| `autoresearch-p1-target-s20260714-v2` | P1 target legality fail |
| `autoresearch-e0-evaluator-s20260714-v2` | E0 v2 实现缺陷（保留） |
| `autoresearch-e0-evaluator-s20260714-v3` | E0 v3 machine pass，待人工 review |
| `p2-v2-gen04-track8-s20260713-59c3f05-net1` | GEN-04 object gate 通过 |
| `p2-v2-replay05-review20-s20260713-8d750f3` | REPLAY-05 20-case review 通过 |

完整登记表见 `docs/EXPERIMENTS.md`。
