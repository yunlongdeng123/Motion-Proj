# P6R Evidence-Dropout Recovery Closeout

## Terminal outcome

- Task: `WS-V62-P6R-EVIDENCE-DROPOUT-RECOVERY-01`
- Training canonical: `run://worldsim_v62/WS-V62-P6R-EVIDENCE-DROPOUT-RECOVERY-01/20260824T101705Z__feature-dropout-train-s0-r2`
- Legacy canonical: `run://worldsim_v62/WS-V62-P6R-EVIDENCE-DROPOUT-RECOVERY-01/20260824T102709Z__feature-dropout-legacy28-s0-r1`
- Training source: `fb0744b`
- Legacy source: `d0e5950`
- Decision: `rejected / CPSC-Lite family closed`
- Failure ledger: `V62-F06 active, recovery exhausted`; `V62-F07 resolved`

P6R 是 P6 失败后预注册且唯一允许的机制级 recovery。它完成后仍未通过未改动的 legacy28 gate，因此按计划不得进入
P7/P8、不得改选 projection/set-valued recovery，也不得做 threshold、grid、window、backend、model-size、bridge 或 seed
sweep。

## Training result

从冻结 P5 best 同时初始化 student 与 frozen teacher。每个 train query 以 `p=0.5` 独立切换 full/prototype features，
student优化原 P5 task loss 加 `0.25 × KL(teacher full-view base probability || student corrupted-view base probability)`。
pure-prototype development selection按预注册复合objective选点。

| 指标 | baseline | best epoch 2 |
|---|---:|---:|
| composite objective | 2.448369 | 2.274951 |
| hidden-FREE false-OCC | 0.399349 | 0.414406 |
| safe-OCC retention | 0.872897 | 0.887356 |
| target accuracy | 0.452581 | 0.462246 |
| UNKNOWN fraction | 0.221945 | 0.228375 |
| hard violations | 0 / 1,286,134 | 0 / 1,286,134 |

训练共 5 epochs / 840 optimizer steps，wall=`383.489s`，peak=`0.377805GiB`。虽然复合objective改善，主风险单项没有
改善；不能事后改用主风险更低但复合objective更差的epoch 0/3/4。

## Unchanged legacy28 result

| Arm | ACCEPT | false-safe | mask yield | FREE conflict mean / worst |
|---|---:|---:|---:|---:|
| B0 IR-WM argmax | 10 / 28 | 10 | 0.398300 | 0.267482 / 0.570571 |
| B1 hard clip | 10 / 28 | 10 | 0.398300 | 0.050578 / 0.117225 |
| B3 recovered evidential | 4 / 28 | 4 | 0.094024 | 0.043202 / 0.079602 |
| B5 recovered projection | 4 / 28 | 4 | 0.094024 | 0.049166 / 0.087379 |

B5 接受集合与 P6 完全相同，均为 scene-0242 的四个 missing-route-support cases；R10只保留 `2/3`，new Actor=`0`，
new static/disocclusion=`2`。source-valid UNKNOWN 从 P6 `0.827351` 降到 `0.638518`（absolute `-0.188833`，relative
`-22.82%`），说明缺失特征暴露有效缓解了高 UNKNOWN，但没有改变 unsafe decision set。safe-OCC retention=`1.0`，
hard projection=`0/939,206 violations`，所以 rejection 不是all-UNKNOWN、已知OCC丢失或projection实现错误。

失败 gates：ACCEPT、false-safe、mask yield、worst FREE conflict、R10、Actor gain 与 UNKNOWN上限；通过的只有mean FREE
conflict、static/disocclusion gain、hard projection和safe-OCC retention。wall=`48.109s`、peak=`0.531876GiB`、
pre-closeout output=`2,293,068 bytes`。

## Post-failure primary-source audit

- RELIOcc 用混合不确定性学习与离线校准提升 occupancy reliability；这需要重新训练/校准原生 occupancy head，不是对
  argmax-only legacy artifact 的无训练修补：`https://www.ijcai.org/proceedings/2025/0220.pdf`。
- OCCUQ 在 dense occupancy head加入 uncertainty module，并另拟合 feature-level GMM；其官方实现同样要求原生feature和
  uncertainty training：`https://github.com/ika-rwth-aachen/OCCUQ`。
- α-OCC 的 hierarchical conformal prediction依赖独立 calibration并输出prediction sets；这是下一版本可能的Tier C
  路线，不允许在已失败P6之后绕过计划直接开启P8：`https://arxiv.org/abs/2406.11021`。
- NeurIPS selective classification把风险与coverage作为显式权衡；当前结果已说明多输出UNKNOWN本身不会认证剩余四个
  accepted surfaces：`https://proceedings.neurips.cc/paper/7073-selective-classification-for-deep-neural-networks.pdf`。
- COLT 2024证明更细粒度weighted coverage的估计受校准样本复杂度限制；没有独立calibration data时，不能把当前
  development/O_eval事后包装成条件安全保证：`https://proceedings.mlr.press/v247/areces24a.html`。

这些方案都需要被本次scope明确锁住的新数据、原生feature、独立calibration或第二个head recovery。不存在同时满足
“不第二recovery、不读O_eval调参、不重跑backbone”的合法迁移，因此本版本按stop rule关闭，而不是继续堆机制。

## Reopen boundary for a future version

合法复开至少需要新的scope freeze，并同时提供：原生per-voxel logits/features、独立于development与legacy O_eval的
calibration cohort、直接覆盖hidden-surface false-safe的监督/风险定义，以及在任何legacy评分前冻结的set-valued或
risk-control policy。以上只是下一版本研究建议，不解锁 V6.2 P7/P8。
