# V7.1 S1 Candidate Displacement Oracle

## 结论

`WS-V71-S1-DISPLACEMENT-ORACLE-01`的唯一正式run支持D1：固定candidate set已经包含可形成hazard
literal-first-return / surface-fidelity Pareto的连续位置解，因此V7.1选择M0 ray/surface displacement，不执行S1-B。

## Canonical evidence

```text
run://worldsim_v71/WS-V71-S1-DISPLACEMENT-ORACLE-01/
20260903T101000Z__s1-displacement-oracle-r1
```

| quantity | baseline | oracle | delta |
|---|---:|---:|---:|
| hazard literal early | 18.635% | 17.553% | -5.802% relative |
| all literal early | 17.578% | 17.047% | -3.026% relative |
| clear literal early | 15.371% | 15.988% | +4.009% relative |
| symmetric Chamfer | 269.673 mm | 265.895 mm | -3.778 mm |
| target hit recall | 41.916% | 44.184% | +2.269 pp |
| Actor/hazard retention | 100% | 100% | 0 |

数据仅来自12个train scenes中的64个Actor；36,101条oracle-check rays与oracle-fit rays互斥。Selection、source final和
AV2未读。Oracle保留全部COMPLETE candidate，只优化bounded ray/normal displacement；observed KEEP与matched PROJECT
从未进入optimizer。

## 边界与下一步

Hazard D1通过不代表clear stratum通过。M0必须用输入证据学习何时、向何处位移，并在source Selection同时报告all/hazard/
clear、Chamfer与hit recall。如果M0不能逼近oracle，只有在candidate support ceiling证据成立时才解锁M1；不得回到
keep/delete、UNKNOWN阈值或Actor router。
