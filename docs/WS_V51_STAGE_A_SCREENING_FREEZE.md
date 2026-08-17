# WorldSim V5.1 Stage A Screening Freeze

## Freeze decision

Stage A historical H 已收口：A1 visibility 与 A2 UNKNOWN 通过 H gate；A3 Kish effective count 和 A4 CIF-lite
在任何 quality arm 前因机制不可识别/no-op rejected。进入 S 的候选严格限定为 `A1/A2`，不再修改候选 family。

S scene 按 `development_roles_v1.yaml` 固定为 `scene-0998/scene-0359`。两场沿用 V5 已冻结 30k checkpoint、
split=`modulus 5`、evidence remainder=`0/1/3`、development evaluation remainder=`2`、heldout remainder=`4`，
以及与 H 完全相同的 SAM operator、A1 visibility threshold、A2 UNKNOWN thresholds 和 seed=`20260814`。

## One-shot screening gate

- `2/2` scene Boundary F1 delta vs B3 non-negative；
- 至少 `1/2` scene Boundary F1 delta `>=0.001`；
- scene-balanced mean Boundary F1 delta `>0`；
- scene-balanced mean FN semantic mass delta `<=+0.02`；
- mean Brier/ECE delta 各自不超过 `+0.005`；
- A2 额外要求 scene-balanced coverage `>=0.60` 且 abstained subset error 大于 accepted subset error。

若 A1/A2 的 conditional gate 都通过且 A2 selective gate 通过，唯一 survivor 为 A2；若 A2 selective gate 失败而
A1 conditional gate 通过，则 survivor 为 A1；否则回退 U2/B3。S 后最多保留一个 unary candidate。

## Read and stop policy

本文件与 `stage_a_screening_freeze_v1.yaml` 提交前，不得读取 S mask/evaluation quality。提交后只允许按预登记的
r047/r048 生成两场 SAM sidecar，并由一个新 V5.1 screening run 一次性读取 development evaluation。任何 blocked
工程尝试保留 terminal，以新 run 修复；不得因质量不理想重跑、改 threshold、换候选或打开 heldout/C/validation/test。

Failure refs=`V5-F20–F26/F29–F32 + V51-F01–F07`；freeze 时 `failure_ledger_delta=none`。

## Frozen materialization binding

- SAM r047/r048 均为 `done`；0998/0359 accepted views=`24/18`，checkpoint 与输入 SHA exact。
- B3/evidence materialization r049/r050 均为 `done`；evidence views=`15/15`，accepted evaluation views=`12/9`，
  abstained views=`3/6`，checkpoint 前后 SHA exact，validation/heldout read=`false`。
- r049/r050 的唯一用途是给 V5.1 screening 提供冻结、只读的 B3 observation/evaluation 输入；候选 A1/A2 的 S 质量
  在本节写入时仍未读取。正式绑定见 `configs/worldsim_v51/stage_a_screening_v1.yaml`，runner 见
  `scripts/run_worldsim_v51_stage_a_screening.py`。
- 并行启动 wrapper 的 PID 转义问题登记为 `V51-F08`；它没有生成重复 run。根据单进程后段约 `20–22 GiB` 的实测
  显存保留，r049/r050 改为串行执行，候选筛选同样保持单实例。

## Execution result

r007=`done`，source=`dc24f28`，结论=`stage_a_screening_selected_u2_b3`。A1 的 BF1 nonnegative/clearly-positive=
`1/2, 0/2`，mean delta=`-0.0000165293`；A2 mean coverage=`0.557435<0.60`。因此 A1/A2 均 rejected，冻结
U2/B3；不得重跑或改门。完整结果见 `configs/worldsim_v51/stage_a_closeout_v1.yaml`、`docs/EXPERIMENTS.md` 和
`V51-F09/F10`。
