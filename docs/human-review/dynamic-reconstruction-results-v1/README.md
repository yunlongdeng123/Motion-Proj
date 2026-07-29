# 动态重建结果审核包 — 未触发说明

- 生成日期：2026-07-29
- M5 证据：`/root/autodl-tmp/runs/dynamic_recon/DR-M5-DGGT-NUSC-01/20260729T133346__native-nusc-s0-wm3090/`
- M5 common-observation：未运行；216-target mapping 只完成只读预审
- M6 证据：`/root/autodl-tmp/runs/dynamic_recon/DR-M6-STRESS-01/20260729T145645__identity-audit-s0-wm3090/`
- M7 证据：`/root/autodl-tmp/runs/dynamic_recon/DR-M7-HYPOTHESIS-01/20260729T145748__novelty-audit-s0-wm3090/`
- M8 状态：`rejected / not authorized`
- M9 状态：`rejected / not triggered`
- 人工结论：`null`，Codex 未代填

## 为什么没有 blind pairwise clips

预注册顺序要求 M7 novelty 通过、M8 三 seeds primary 与 guardrails 通过后，M9 才能生成六场景全量 blind pairwise
包。本轮在 M7 已因直接 novelty 重合而 `rejected`，没有 proposed method、matched ablation 或可供盲审的 method
clips。创建空标签视频、复用 baseline 视频或要求评审评价不存在的方法，都会伪造 M9 的输入条件。

因此本目录是机器终止证据包，不是人工结果审核任务。用户/评审者无需填写 verdict。

## 可审核的机器边界

1. M4 是否确为官方六场景 exact reproduction 且三项门禁通过；
2. M5 是否保留完整 DGGT upstream 终态与失败/成功证据；
3. M6 是否把 0/12 eligible object slots 和全部 ABSTAIN 纳入 coverage；
4. M7 是否只考察决策表 A，并因 InstDrive、Director、OmniRe、HorizonForge、G²Editor 的直接覆盖而停止；
5. 是否没有生成 M8 数字、M9 clips 或人工 verdict。

详细说明见 [`../../DR_M7_NOVELTY_AUDIT.md`](../../DR_M7_NOVELTY_AUDIT.md) 和
[`EVIDENCE_MANIFEST.md`](EVIDENCE_MANIFEST.md)。
