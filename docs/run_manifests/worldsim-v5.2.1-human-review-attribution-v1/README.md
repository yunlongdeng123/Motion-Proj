# WorldSim V5.2.1 Human Review Attribution v1

Canonical run：

```text
/root/autodl-tmp/runs/worldsim_v521/20260820T130000Z__p11-human-review-attribution-s0-r001
```

文件：

- `cases.jsonl`：18 个 case 的 census identity、nuScenes path/split/hash、人工问题归因、模块映射与回测指标；
- `backtest_contract.json`：5 个 Discovery design、3 个 one-shot Confirmation、base sentinels 与 M1/M3/M2-safety cohorts；
- `manifest.json`：输入/输出 SHA256、review source hash、source commit 与计数；
- `summary.json` / `status.json`：冻结结果与终态。

该目录是 P10 原始 `BADCASE_REGISTRY.jsonl` 之上的只读 attribution layer。视觉判断的 claim status 固定为
`visual_diagnostic_hypothesis_not_causal_proof`；后续只有通过正式 causal bridge 才能升级因果措辞。
