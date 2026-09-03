# WorldSim V7.1 Research Closeout

日期：2026-09-03

分支：`research/worldsim-v7.1-learned-evidential-surface`

终态：`v71_research_complete_negative_boundary_paper_ready`

## 结论

V7.1证明了连续Actor表面移动在train-only oracle层面存在联合改善点，但两个固定learned实现都无法在Source
Selection保持该Pareto性质。M0通过把多数completion candidate转成UNKNOWN降低literal early；M1没有生成可抽取的
learned zero-crossing。非学习B4 evidential TSDF恢复了surface completeness，却使hazard early约翻倍。由此得到可发表的
安全边界：减少first-return冲突与恢复稠密碰撞表面在当前证据条件下是两个独立目标，单一目标下降不能作为三维物理自洽性。

## 冻结证据

| 阶段 | 数据角色 | hazard early相对下降 | Chamfer变化 | hit recall变化 | 判定 |
| --- | --- | ---: | ---: | ---: | --- |
| S1 displacement oracle | train-only，64 Actors | 5.802% | -3.778 mm | +2.269 pp | action space feasible |
| M0 learned displacement | Source Selection，88 Actors | 19.435% | +48.745 mm | -10.819 pp | rejected，74.51% UNKNOWN |
| M1 evidential implicit field | Source Selection，88 Actors | 28.485% | +120.674 mm | -14.366 pp | rejected，0 learned points |
| B4 evidential TSDF | consumed Selection，121 Actors | -106.530% | -83.961 mm | +4.786 pp | opposite frontier |

B4覆盖多于M0/M1，因为它不要求Actor存在可学习candidate；这不是跨行同样本显著性比较，而是计划内强制的描述性端点。

## Failure ledger与停止规则

- `V71-F01`：入口工程问题，已恢复；
- `V71-F02`：consumer误读临时NPZ，已恢复；
- `V71-F03`：raw corpus不足1000，使用role-disjoint processed train恢复至1004；
- `V71-F04`：M0 source Selection rejected，机制为UNKNOWN collapse；
- `V71-F05`：M1 source Selection rejected，机制为无learned field extraction；
- B4正常完成，不新增failure；下一可用ID=`V71-F06`。

冻结规则要求source Selection失败后不得用Source Final或AV2 target救模型。因此不执行PCGrad、LoRA、第二seed、
band/top-k/evidence-threshold/loss sweep，也不混合B4与M1调权；不回到selector/delete研究线。

## Claim边界

支持：

- Actor-local连续位移动作空间在train oracle上可同时改善literal early、Chamfer与hit recall；
- 固定learned displacement和implicit field没有将该oracle收益迁移到Source Selection；
- 当前三态证据下，ray-safety与surface-completeness呈可解释的双端失败边界。

不支持：

- V7.1 learned模型的Source Final或AV2零样本泛化；
- domain-invariant、collision-free、closed-loop或real-road safety保证；
- 以loss下降、UNKNOWN删除或更密TSDF表面替代联合物理合同。

## Canonical artifacts

- S1：`run://worldsim_v71/WS-V71-S1-DISPLACEMENT-ORACLE-01/20260903T101000Z__s1-displacement-oracle-r1`
- S2 raw：`run://worldsim_v71/WS-V71-S2-ACTOR-CORPUS-01/20260903T102000Z__actor-corpus-r1`
- S2 recovery：`run://worldsim_v71/WS-V71-S2-PROCESSED-CORPUS-RECOVERY-01/20260903T110000Z__processed-corpus-recovery-r1`
- M0：`run://worldsim_v71/WS-V71-M0-RAY-SURFACE-DISPLACEMENT-01/20260903T111000Z__m0-ray-displacement-s71101-r3`
- M1：`run://worldsim_v71/WS-V71-M1-EVIDENTIAL-SURFACE-FIELD-01/20260903T113000Z__m1-evidential-field-s71102-r1`
- B4：`run://worldsim_v71/WS-V71-B4-EVIDENTIAL-TSDF-01/20260903T120000Z__b4-evidential-tsdf-r1`

所有V7.1训练进程已结束；没有生成hash、checksum或fingerprint。CVPR主稿使用macro-driven结果并保留终测未读边界；
最终定向编译=`11 pages / 1,936,223 bytes`，无undefined citation/reference，仅保留既有Table 1 `6.03pt` overfull。
