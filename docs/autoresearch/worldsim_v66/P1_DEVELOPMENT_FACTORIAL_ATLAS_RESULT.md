# P1-D Actor Validity×Hazard Development Atlas 结果

Task：`WS-V66-P1-VALIDITY-HAZARD-SEPARATION-ATLAS-DEV-01`

Canonical：`run://worldsim_v66/WS-V66-P1-VALIDITY-HAZARD-SEPARATION-ATLAS-DEV-01/20260828T084915Z__factorial-atlas-dev-s0-r1`

Verdict：`supported_development_factorial_separation_proceed_to_p2`

## 分母与构造

- 输入：P10V已消费的6 scenes / 72 units，只作Tier-L development mechanism。
- eligible base：409 actor-unit；每个base必须具有sensor hit、current/swept envelope和至少4个grounded boundary points。
- artifact family：unsupported ghost、duplicate shell、lifecycle flicker、teleport、shape jump。
- 每个base/family生成V0-H0、V0-H1、V1-H0、V1-H1；2,045 clusters、8,180 rows，四象限各2,045。
- 推理排除artifact label/family、hazard label/score和variant ID；首轮hazard intervention只改变task attribute。

## 结果

| 指标 | 结果 |
|---|---:|
| q0 artifact AUROC / AUPRC | 0.5000 / 0.5000 |
| q0 hazard AUROC / AUPRC | 0.5000 / 0.5000 |
| certificate artifact AUROC / AUPRC | 1.0000 / 1.0000 |
| 五类artifact recall | 全部1.0000 |
| clean-hazard false artifact | 0.0000 |
| legitimate hazardous Actor retention | 1.0000 |
| artifact+hazard detection | 1.0000 |
| q0 / certificate hazard-pair mean abs delta | 0 / 0 |

四个development gates全部通过。wall=`8.0297s`，peak GPU=`0.02359GiB`，peak RSS=`0.9098GiB`。

## 正确解释

q0在同一base的representation-level corruption pair中保持原分数，所以AUROC 0.5是预注册的actor-blind结构baseline，
不能写成“q0对重新渲染自然artifact只有随机性能”。certificate的满分来自可审计确定性注入，说明reason-coded factor接口
能够表达这五种mechanism；它不能证明自然reconstruction artifacts可被满分识别。

本结果只解锁独立P2 certificate接口与P4 repair capability。P3 learned superiority没有headroom，除非fresh/natural
benchmark暴露deterministic ceiling；fresh selection、真实hazard edit、reactive simulation与RL仍锁定。
