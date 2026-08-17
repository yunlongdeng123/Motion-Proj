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

状态：`H gate passed / candidate / S unread`。

下一步只允许比较 U2/B3 与 A1/B3，唯一变量是 observation 是否具备 semantic update 资格：

```text
visible + semantic available  -> 按冻结 B3 累积正/负 fractional count
不可见或 semantic unavailable -> delta alpha = delta beta = 0
```

A1 不引入 UNKNOWN threshold、effective-count correction、CIF state、DINO、Graph、anchor、temporal 或 LiDAR
kernel。canonical r002：

```text
/root/autodl-tmp/runs/worldsim_v51/WS-V51-M1-A-UNARY-OBSERVABILITY-01/
20260817T104000Z__m1-a1-visibility-h-s20260814-r002
```

| Scene | Eval views | Semantic-valid ratio | ΔBF1 | ΔIoU | ΔFN | ΔBrier | ΔECE |
|---|---:|---:|---:|---:|---:|---:|---:|
| 0471 | 8 | 0.857484 | +0.000020836 | +0.000124291 | +0.000484780 | -0.000044421 | -0.000436149 |
| 1087 | 1 | 0.967938 | 0 | 0 | +0.000000529 | -0.000000729 | +0.000010298 |
| 0379 | 3 | 0.826925 | +0.003446302 | +0.001256639 | +0.002831752 | +0.000003233 | -0.000008710 |

scene-balanced mean ΔBF1/IoU/FN/Brier/ECE=
`+0.001155713/+0.000460310/+0.001105687/-0.000013972/-0.000144854`；五项 H gate 全通过。

B3 在本轮被当前 GPU 重新渲染，12/12 evaluation NPZ 与 canonical byte exact，aggregate metrics delta=`0`；三个
checkpoint 前后 exact。A1 仍只是 H candidate：1087 只有一个 evaluation view，0471 的 BF1 增益只有约 `2.08e-5`，
不能声明 scene-disjoint 稳定或直接进入 S。

## A2：Semantic UNKNOWN / ABSTAIN

状态：`H gate passed / candidate / S unread`。

下一步只在 A1 posterior 上增加 UNKNOWN state 与 selective metrics；阈值只能来自 H evidence/training statistics，
不得读取 evaluation quality 选择。A3 effective-count、A4 CIF、S scenes 与后续 Stage 继续锁定。

冻结配置：`configs/worldsim_v51/m1_unary_unknown_v1.yaml`。校准总体只包含三个 H scene 中
`A1_effective_evidence_count > 0` 的 Gaussian，固定下/上四分位阈值为：

```text
effective observation count <= 0.19274792820215225
posterior entropy >= 0.005402358970383594
cross-view disagreement >= 8.494543610182426e-12
UNKNOWN = high entropy AND (low count OR high disagreement)
image abstain threshold = 0.5
```

未观测 Gaussian 在 0379/1087 占 `67.39%/97.20%`；若在全量 Gaussian 上做 inclusive 分位数 OR，count 与
disagreement 阈值会退化为 0 并造成 0 coverage。该失败假设已登记 `V51-F04`，不能在 A2 quality 读取后重新解释或调参。

canonical r003：

```text
/root/autodl-tmp/runs/worldsim_v51/WS-V51-M1-A-UNARY-OBSERVABILITY-01/
20260817T113000Z__m1-a2-unknown-h-s20260814-r003
```

| Scene | Eval views | Gaussian UNKNOWN | Coverage | Accepted abs error | Abstained abs error | UNKNOWN recall/errors |
|---|---:|---:|---:|---:|---:|---:|
| 0471 | 8 | 0.204240 | 0.464865 | 0.040882 | 0.251780 | 0.868498 |
| 1087 | 1 | 0.001749 | 0.736836 | 0.003579 | 0.169013 | 0.959160 |
| 0379 | 3 | 0.002478 | 0.958186 | 0.000213 | 0.071956 | 0.875421 |

scene-balanced coverage=`0.7199625`；accepted/abstained error=`0.0148914/0.164250`，UNKNOWN error separation=
`+0.149358`，全部 selective checks 通过。A1 conditional posterior 独立重渲染 `12/12 byte exact`，A2−A1 七项
conditional metric delta 全为 0；因此 A2 的证据是错误集中/选择性风险改进，而不是额外的 BF1/IoU 增益。

限制：0471 coverage 只有 `46.49%`，冻结 gate 只要求 scene-balanced mean>=60%，所以 A2 不能写成逐场稳定；1087
仍只有单个 evaluation view。A2 仅进入 H candidate 集合，不能提前读 S。下一门只解锁 A3 correlation-aware
effective count；A4/CIF、S 与后续 Stage 仍锁定。

## Failure ledger

- refs：`V5-F20–F26`、`V5-F29–F32`、`V51-F01–F04`。
- A0 `failure_ledger_delta=none`。
- A1 `failure_ledger_delta=none`。
- A2 `failure_ledger_delta=none`；formal refs 包含 `V51-F04`。
- `V51-F02` 只修正人工 float32 常数单测的错误 oracle；formal canonical metric 仍要求 delta 严格等于 0。
- `V51-F03` 固化 configured/applied visibility threshold，避免 float32 边界静默改变 eligibility 分母。
