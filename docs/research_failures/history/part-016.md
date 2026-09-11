# 历史原始记录 016

> 冻结历史，不是当前执行规则。当前规则见 [scaling law](../../../auto-research_scaling_law.md) 与 [AGENTS](../../../AGENTS.md)。
> 原文件：`docs/RESEARCH_FAILURES.md`；迁移前提交 `abbd431c`。原有相对链接按原目录解释，证据路径原样保留。
> 按索引读取相关小节；旧哈希、过度门控和资源指令均不重新生效。

## V7.1 M45 pre-registration note — oriented probability is not hard collision support（2026-09-05）

M10--M11拒绝的是learned oblate ellipsoid的earliest hard intersection；M45不复开该范式，而把既有GT normal/
thickness作为M39 categorical child kernel的metric。所有中心、tangent、authority和readout冻结，不调thickness/
normal/scale。若失败，关闭oriented categorical重组，不以M11旧结果或M44 attribution选择case；M43 external
candidate不变。M43失败保留`V71-F43`，M45失败使用`V71-F44`。

## V7.1 M44 outcome note — completion risk is now isolated, not filtered（2026-09-05）

M44无新failure。99,208 rays上median early与boundary CDF条件100%等价，family分解残差=`2.98e-7`。
M39整体pre-boundary CDF下降`0.331pp`由anchor `-0.346pp`贡献，child反而`+0.016pp`；clear中child增量
`+0.163pp`抵消anchor `-0.265pp`。因此后续若external失败，应研究child-specific supervision/representation
的跨域稳定性，不得把CDF margin变成threshold、UNKNOWN mask、删child或case rejection，也不得继续调M42全局
loss。下一failure ID仍=`V71-F43`。

## V7.1 M44 interpretation note — safety accounting is not post-hoc consistency（2026-09-05）

M44只解释由M35/M38 GT F/O/U supervision得到的M39 surface measure：early boundary由pre-GT CDF是否超过0.5
精确定义，并解析归因到anchor/child family。不得把低/高risk margin变成删点、拒绝case或新阈值，也不得以该
exposed audit回选checkpoint。物理改善仍须来自supervision/representation，M44只补足可解释性与显式安全边界；
下一failure ID仍=`V71-F43`。

## V7.1 M43 pre-registration note — AV2 build evidence is input, AV2 target is evaluation only（2026-09-05）

M43冻结M39全部checkpoint、metric evidence radii、31/39维feature contract、64-bin CDF median及三项判定。跨传感器
adapter只把AV2编译器已有query/build/canonical/provenance转换为source-domain同构输入；不得用target sweep拟合
normalization、authority、阈值或log选择。DGLSS（CVPR 2023）证明LiDAR配置/稀疏度本身构成显著domain gap，故
该差异必须由冻结zero-shot结果诚实暴露，不能在目标域校准。任一判定失败登记`V71-F43`且关闭AV2 tuning；下一
failure ID=`V71-F43`。

## V71-F42: M42 global interval events do not resolve stratum-dependent CDF safety（2026-09-05）

- run=`run://worldsim_v71/WS-V71-M42-INTERVAL-EVENT-SUPERVISION-01/20260905T083000Z__m42-interval-event-s71142-r1`；
- evidence=all/hazard early相对baseline为`-0.052/-0.108pp`且all hit=`+2.538pp`，但clear early=
  `+0.221pp`；相对M39 all early=`+0.469pp`，decisions=`4/5`；
- optimization=interval NLL下降，anchor/child correlation=`0.4818/0.5700`，family-total residual=
  `1.53e-5`，无NaN/OOM；失败不是训练、可辨识性或守恒数值问题；
- root cause=hit-band项持续改善，但not-early NLL不降且部署early probability微升；同一全局event loss在固定
  geometry/measure上仍以增加GT附近概率为主，不能同时保证hazard和clear两种ray分布的CDF边界；
- anti-repeat=不在同一66 Actors上调event权重、0.20m tolerance、epoch、seed、bin或median；不以clear标签
  做部署门控。M40--M42微调支线关闭，冻结M39 categorical surface measure作一次cross-domain确认；
- claim impact=M42不进external；结果支持“物理event必须从GT监督”但否定“全局interval objective本身足够”。
  下一failure ID=`V71-F43`。

## V7.1 M42 pre-registration note — safety event is defined by GT, not a deployment gate（2026-09-05）

M42的0.20m直接沿用literal GT evaluator tolerance，训练前冻结；它定义not-early/hit supervision，不在推理时删点
或拒绝case。若五门任一失败登记`V71-F42`，不调event weight/tolerance/bin/median/epoch/seed。下一failure ID=
`V71-F42`。

## V71-F41: M41 conserved family measure still optimizes the wrong CDF event（2026-09-05）

- run=`run://worldsim_v71/WS-V71-M41-CONSERVED-SURFACE-MEASURE-01/20260905T080000Z__m41-conserved-measure-s71141-r1`；
- evidence=family-total residual=`1.53e-5`且anchor/child correlation=`0.5236/0.6128`，但all/hazard/clear
  early=`+0.763/+0.773/+0.716pp`；all hit=`+3.239pp`，decisions=`3/5`；
- optimization=NLL与两项CE持续下降，无NaN/OOM；不是measure守恒失效或可辨识性不足；
- root cause=单GT-bin CE只提高该bin相对概率，不约束`CDF(d_gt-0.20)<0.5`这一部署median safety event；族内
  redistribution可同时增加hit和early；
- literature response=LidaRF（CVPR 2024）用LiDAR sight loss约束GT depth附近的ray-weight distribution并通过
  CDF计算；ordinal depth文献明确指出nominal CE忽略顺序。M42直接训练not-early和hit-band区间概率；
- anti-repeat=不调M41 conservation/epoch/seed/bin/median或evidential weight；保留分族measure，替换错误的
  point-bin/depth代理为验收同构interval likelihood；
- claim impact=M41不进external；sampling-density total shortcut已排除，剩余是loss-to-decision mismatch。
  下一failure ID=`V71-F42`。

## V7.1 M41 pre-registration note — conserved measure is not an opacity threshold（2026-09-05）

M41冻结的是由M39 GT-supervised heads给出的每Actor两族总measure，不是按M40错误ray或holdout结果设置的阈值；
所有primitive保留，F/O/U probability连续且受GT CE。若五门任一失败登记`V71-F41`，不调family total、归一化、
loss或seed。下一failure ID=`V71-F41`。

## V71-F40: M40 categorical fine-tuning shifts total authority into the dense child family（2026-09-05）

- run=`run://worldsim_v71/WS-V71-M40-JOINT-CATEGORICAL-EVIDENTIAL-AUTHORITY-01/20260905T073000Z__m40-joint-categorical-s71140-r1`；
- evidence=anchor/child occupied correlation=`0.4937/0.6163`、all hit=`+2.864pp`，但all/hazard/clear early=
  `+0.609/+0.614/+0.585pp`，decisions=`3/5`；
- optimization=categorical NLL及两项evidential CE均下降，无NaN/OOM；失败不是监督缺失或数值不稳定；
- root cause=child数量固定为parent的4倍；joint ray loss将child mean occupied从M38 `0.383`推到`0.663`，
  以改变anchor/child family总surface mass换取hit，重现sampling-density-dependent amplitude shortcut；
- literature response=Vol3DGS强调normalized Gaussian measure，RT-Splatting（CVPR 2026）也显式分离geometric
  occupancy与optical opacity。M41迁移为anchor/child分族守恒measure，只让ray loss在族内分配；
- anti-repeat=不调M40 loss/epoch/seed/bin/median，不缩scale或删除children；冻结M39每Actor两族总mass作为已监督
  measure，训练/部署共用同一归一化；
- claim impact=M40 checkpoint不进external；train--deploy同构本身成立，但unconstrained family amplitude拒绝。
  下一failure ID=`V71-F41`。

## V7.1 M40 pre-registration note — direct return loss cannot erase evidential semantics（2026-09-05）

M40联合更新anchor/child head但冻结全部几何；categorical first-return与部署同构，同时持续保留两套GT F/O/U soft
CE。若ray改善而任一occupied correlation低于0.25，仍视为不可解释global reweighting并登记`V71-F40`；失败后
不调loss/epoch/seed/bin/median。下一failure ID=`V71-F40`。

## V7.1 M39 outcome note — authority is surface-return evidence, not additive volume density（2026-09-05）

M39无新failure。事前固定M38主candidate在categorical composition下all/hazard/clear early均不增且all hit保留，
3/3通过；相对baseline分别`-0.521/-0.621/-0.030pp` early与`+2.172pp` all hit。不得因M37 descriptive
early略低而结果后回选M37。后续只确认冻结M38 categorical表示，不重开transmittance opacity/scale/margin/bin/
median sweep；下一failure ID仍=`V71-F40`。

## V7.1 M39 pre-registration note — composition audit is not model selection（2026-09-05）

M39主candidate在quality read前固定为M38 categorical composition；M37只作配对描述，不能看到结果后择优。若M38
三门任一失败，登记`V71-F40`并关闭scalar authority + isotropic categorical energy，不调bin/median/scale或回选
M37。下一failure ID=`V71-F40`。

## V71-F39: M38 pre-hit survival cannot decouple endpoint evidence from Gaussian front tail（2026-09-05）

- run=`run://worldsim_v71/WS-V71-M38-PREHIT-FREE-SPACE-SURVIVAL-01/20260905T063000Z__m38-prehit-survival-s71138-r1`；
- evidence=holdout pre-hit child optical mass `0.1870→0.1272`；相对M37 early=`-2.409pp`、hit=`+1.330pp`，
  但相对baseline all/hazard/clear early仍`+3.319/+3.676/+1.563pp`，decisions=`2/4`；
- supervision health=child occupied correlation=`0.3990`，训练全程无NaN/OOM；因此不是输入不可辨识或free loss无梯度；
- root cause=一个非负scalar occupied mass同时缩放endpoint surface evidence和isotropic Gaussian在endpoint前的体密度
  尾部；free survival继续下降时，GT F/O/U CE从`0.9336`恶化至`1.0583`，显示目标角色耦合；
- literature response=NeuS与VolSDF（NeurIPS 2021）都指出generic volume density会产生surface geometry bias，并把
  rendering density绑定到显式surface/SDF；M39先用已有M18式categorical surface-return分布做冻结归因，不直接重开
  已失败的M10--M18 field/primitive sweep；
- anti-repeat=不调M38 weight/margin/epoch/seed，不缩Gaussian scale、不删除child；下一步只比较同一authority在
  additive transmittance与sampling-density-invariant categorical return composition中的行为；
- claim impact=M38不进external；GT pre-hit supervision的方向性成立，当前scalar optical parameterization拒绝。
  下一failure ID=`V71-F40`。

## V7.1 M38 pre-registration note — observed FREE interval is training supervision（2026-09-05）

M38只在loss中加入native LiDAR pre-hit survival，不根据M37 early ray删除或重标任何primitive。固定margin=0.20m、
weight=1.0、4 epochs/seed71138；若失败登记`V71-F39`并关闭当前additive child optical mass，不扫margin/weight/
scale/seed。下一failure ID=`V71-F39`。

## V71-F38: M37 child authority improves unit-opacity completion but leaves pre-hit optical mass（2026-09-05）

- run=`run://worldsim_v71/WS-V71-M37-SUPERVISED-CHILD-TRANSMITTANCE-01/20260905T060000Z__m37-child-transmittance-s71137-r1`；
- evidence=child occupied correlation=`0.4306`，learned-vs-unit-child early=`-11.351pp`且hit=`+8.880pp`；
  但learned-vs-original baseline all/hazard/clear early=`+5.728/+6.346/+2.690pp`，decisions=`2/4`；
- optimization=loss与evidential CE正常下降，无NaN/OOM；categorical NLL只从`3.7546`到`3.7516`，不是
  数值或输入可辨识性失败；
- root cause=当前categorical first-return目标对命中bin概率做整体竞争，却未把GT endpoint之前的已观测FREE
  interval单独约束为survival；模型可同时提升hit并在前方保留过多累计hazard；
- literature response=ALSO（CVPR 2023）直接从LiDAR sensor origin到return构造occupancy supervision；Neural
  LiDAR Fields（ICCV 2023）将return/drop作为物理射线过程。迁移为训练期`-log T_pre`，不是部署过滤；
- anti-repeat=不删child、不用UNKNOWN mask、不调Gaussian/opacity scale、epoch、seed或margin sweep；冻结M37
  checkpoint并只增加GT pre-hit free-space survival项；
- claim impact=M37不进入external；child evidence supervision与ordered composition的方向性成立，但当前目标函数拒绝。
  下一failure ID=`V71-F39`。

## V7.1 M37 pre-registration note — child authority is a separate supervision problem（2026-09-05）

M37冻结已定位的anchor head，只给completion child独立输入/GT/head；不得让共享context用anchor标签代替child
可辨识性，也不得让geometry/scale吸收ray loss。若任一decision失败不调seed/epochs/loss/opacity，登记
`V71-F38`并关闭此参数化。下一failure ID=`V71-F38`。

## V7.1 M36 outcome note — unit completion opacity is the dominant transmittance error（2026-09-05）

M36无工程失败。learned anchors-only early=`18.386%`接近unit-energy baseline=`18.187%`，children-only unit
却为`37.465%`；all learned=`35.266%`。因此不得继续调anchor head或全局scale；下一监督对象是每个M8
completion child的opacity/F/O/U，输入只能来自parent candidate build evidence与child geometry，且保持与anchor
head解耦。下一failure ID=`V71-F38`。

## V7.1 M36 pre-registration note — optical attribution before calibration（2026-09-05）

M36不拟合任何opacity，只冻结分解anchor/child/all optical contribution。它防止看到M35高early后直接手调
全局scale；若adapter/checkpoint/decomposition失败登记`V71-F38`，否则结果只定位下一监督对象。
下一failure ID=`V71-F38`。

## V71-F37: M35 unit Gaussian optical thickness over-terminates near surfaces（2026-09-05）

- run=`run://worldsim_v71/WS-V71-M35-TRANSMITTANCE-ANCHOR-AUTHORITY-01/20260905T050000Z__m35-transmittance-anchor-s71135-r1`；
- evidence=unit-transmittance early/hit=`42.294/50.939%`，learned=`35.266/56.707%`，原baseline=
  `18.187/62.283%`；authority相对同算子方向正确，但绝对early仍`+17.079pp`；
- root cause=每个密集/重叠primitive用unit line-integrated optical thickness，表示不具采样密度不变性；条件
  hit分布还掩盖unit/learned `0.320/0.399` no-return mass；
- anti-repeat=不调segments、Gaussian scale、seed或手工opacity；先归因anchor/child贡献，再从GT监督表面
  measure或opacity calibration；
- claim impact=M35不进入external；有序transmittance结构保留为候选，但当前密度参数化拒绝。
  下一failure ID=`V71-F38`。

## V7.1 M35 pre-registration note — replace normalization, not supervision（2026-09-05）

M35保留M33 evidence/GT与M34 anchor-only隔离，只替换ray composition为解析Gaussian积分+prefix transmittance。
unit-transmittance单独报告；条件return分布不冒充no-return建模。若失败，不调segment count、opacity scale、
Gaussian scale、seed或阈值，登记`V71-F37`。下一failure ID=`V71-F37`。

