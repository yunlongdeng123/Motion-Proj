# WorldSim V6.1 Minimum Experiment Closeout

## 最终结论

WorldSim V6.1 已按预注册 stop rule 负结论收口。研究证明 oracle Occupancy 能显著抬高 proposal 上界，但两种独立的
learned predicted Occupancy 都无法在隐藏 observed-FREE 证据下保持 zero false-safe。因此，本轮不能把无训练、无校准的
argmax occupancy 表面提升为安全 authority；ME-4 未执行，后续只允许从冻结 artifact 合成 arXiv 技术报告。

## 决策链

| 阶段 | 结果 | 保留结论 |
| --- | --- | --- |
| P0 / ME-0 | PASS | 28-case、O_method/O_eval 分离与三态 SceneIR-O authority 建立。 |
| ME-1 oracle | PASS | O2=`10/28`、false-safe=`0`、mask yield=`39.83%`，证明几何来源是可提升因子。 |
| P4 Hunyuan smoke | PASS | 官方 voxel-conditioned generator 可在 3090 执行，不代表场景一致性。 |
| ME-2 Hunyuan | REJECTED | 四臂均`0/6`；生成闭合表面持续占据 observed FREE，路线停止。 |
| P6 GaussianWorld | PASS | 官方 streaming current occupancy capability 可执行。 |
| ME-3 GaussianWorld | REJECTED | `10/28`且 yield=oracle，但`10/10 false-safe`。 |
| P7/P7R IR-WM | PASS | 官方 vision-centric current occupancy 在 3090 有效；形式合同恢复没有重复 forward。 |
| ME-3R IR-WM | REJECTED | 再次`10/28`且 yield=oracle，但`10/10 false-safe`；唯一 recovery 消费。 |

## 最终运行

```text
run://worldsim_v61/WS-V61-ME3R-IRWM-PREDICTED-OCC-01/20260822T145543Z__irwm-predicted-occ-s1-r1
```

- source commit：`6de27f5704914711e38090c7416d7145f2a610be`
- primary：`10 ACCEPT / 0 ABSTAIN / 18 REJECT`
- false-safe：`10`
- accepted mask-area yield：`0.39830013613398907`
- oracle yield fraction：`1.0`
- hidden FREE conflict：route-support `0.344..0.571`；actor/disocclusion `0.106..0.173`
- wall：`124.29696408100426s`
- conservative peak GPU upper bound：`8.251458168029785GiB`
- gate / summary / manifest / terminal：`e990ee68...920f / abf0c711...18cf / 1134b0db...29d6 / 67e21afb...ca6`

## 报告边界

可写入技术报告的主张：oracle Occupancy upper bound、生成表面与 observed FREE 的冲突、两个 predicted occupancy
backend 的 capability/unsafe-authority 分离、truth-tier 隔离与 zero-false-safe stop rule。

不得主张：现实驾驶安全、confirmation/test 泛化、所有 Occupancy 模型均失败、经过 calibration 的 uncertainty 无效，
或 ME-4 已被实验拒绝。ME-4 的准确状态是 `not executed / not authorized after ME-3R failure`。

## 冻结与清理

保留 `runs/worldsim_v61`、数据、模型、隔离环境、官方 source archives、cleanup manifests 与仓库提交。可删除 setup/
download helper、外层 tmux log、preflight/staging、JIT/build cache 和本地上传副本；这些不是论文证据。任何后续报告图表
必须从 canonical run artifacts 生成，不重跑 threshold、hash campaign 或模型实验。
