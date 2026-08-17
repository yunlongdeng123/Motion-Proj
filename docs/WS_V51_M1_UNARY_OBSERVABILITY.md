# WorldSim V5.1 M1 Unary Observability

## 当前结论

`WS-V51-M1-A-UNARY-OBSERVABILITY-01` 正在执行。A0 已证明当前分支能从 V5 冻结 observations 对
r037/r042/r043 的 B0/B1/B3 Bayesian posterior 与 Gaussian metrics 做 exact replay；因此下一步可以只改变
visibility/missingness 机制，避免基线漂移干扰归因。

## A0：V5 Bayesian Unary Exact Replay

Canonical run：

```text
/root/autodl-tmp/runs/worldsim_v51/WS-V51-M1-A-UNARY-OBSERVABILITY-01/
20260817T102000Z__m1-a0-v5-unary-replay-s20260814-r001
```

| Scene | Evidence NPZ | Arms | Array groups | Bit mismatch | Metric values | Max abs delta |
|---|---:|---:|---:|---:|---:|---:|
| 0471 | 15 | 3 | 18 | 0 | 18 | 0.0 |
| 1087 | 15 | 3 | 18 | 0 | 18 | 0.0 |
| 0379 | 15 | 3 | 18 | 0 | 18 | 0.0 |

每个 arm 重算并比较：

```text
unary_posterior
unary_uncertainty
effective_evidence_count
multi_view_disagreement
boundary_ambiguity
depth_support
```

每个 arm 的 Gaussian Brier、ECE、IoU、NLL、FP semantic mass 与 FN semantic mass 均按 frozen implementation
重算，canonical delta 严格为 `0.0`，没有用 tolerance、舍入或格式化结果替代 exact gate。

三个 V5 run 各自的 12 个核心生成源码与当前分支逐 SHA exact；canonical 159-file inventory 与所有 evaluation
artifact 也逐 SHA exact。边界是本轮没有重新执行 GPU renderer，故 2D 证据写作
`canonical_artifact_bytes_and_generation_source_sha_exact`，不能写作新的 2D quality run。

summary/status/fingerprint/manifest SHA：

```text
b9b33bbd8304bb184e9388b4c102a49236a30ded1ae7d10c071cfc9859914878
5d695add5efb535da91da619f520f6a3f7c9b78b717e612e03422279e435e432
6c7466b3a2f8dfb4fe905841fcfe91bce230200862fba56e843811a237d700ae
c4a5e4fdb7189cbfca0c756365f21ac6d981a6e9c30616548870c204c3103d3d
```

## A1：Visibility-Masked Bayesian Update

状态：`pending/unlocked`。

下一步只允许比较 U2/B3 与 A1/B3，唯一变量是 observation 是否具备 semantic update 资格：

```text
visible + semantic available  -> 按冻结 B3 累积正/负 fractional count
不可见或 semantic unavailable -> delta alpha = delta beta = 0
```

A1 不引入 UNKNOWN threshold、effective-count correction、CIF state、DINO、Graph、anchor、temporal 或 LiDAR
kernel。先在 H=`0471/1087/0379` 做 matched diagnostic，之后才按 frozen gate 判断是否进入 S。

## Failure ledger

- refs：`V5-F20–F26`、`V5-F29–F31`、`V51-F01/F02`。
- A0 `failure_ledger_delta=none`。
- `V51-F02` 只修正人工 float32 常数单测的错误 oracle；formal canonical metric 仍要求 delta 严格等于 0。
