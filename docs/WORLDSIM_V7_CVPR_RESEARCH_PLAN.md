# WorldSim V7 研究计划

## 2026-09-02 P22 AV2 literal first-return correction completed

P22 canonical在523 Actors/1,435,391 rays上把all/hazard/clear new-early从proxy
`1.3483/1.9115/.9797%`纠正为literal `9.8943/13.9261/7.2556%`，放大约`7.3x`。全部142,022
literal new-early由COMPLETE首占用；literal new-hit/new-early仅`1.194`，因此P15的proxy `14.51`不能继续承担
first-return物理效用论证。

该结果与P20 source约6倍低估方向一致，形成跨传感器metric-failure硬证据，但P22 cohort已消费，不能升级为新的
zero-shot/fresh confirmation。后续只修正论文claim、宏与图，不开启deletion或selector方法；第三批10 logs保持unread。

## 2026-09-02 P22 consumed-AV2 literal first-return metric correction frozen

P22是停止方法扩展后的合法metric纠错：复用P20 minimum-positive-depth operator，在已消费P15的20-log AV2上
重测always-COMPLETE，不训练或选择新policy。它回答source约6倍proxy低估是否也出现在外部传感器，并同步纠正
KEEP/PROJECT/COMPLETE首回波来源；结果只能标记`consumed_external_diagnostic`。

配置固定20 logs、P2 observed-hit compiler、P15 legacy rows、`.20m` lateral/depth tolerance和唯一run ID；失败log不删，
结果后不扫operator/tolerance/policy。第三批未读10 logs不参与，P22也不授权其quality read。

## 2026-09-02 P21 supported and CVPR safety-boundary integration completed

P21 canonical=`run://worldsim_v7/WS-V7-P21-MONOTONE-SAFETY-BOUNDARY-01/
20260903T134500Z__monotone-safety-boundary-r1`。集合包含关系精确给出
`S' subset S => d_{S'}(r)>=d_S(r)`：删除点的literal first-return early风险单调非增；matched hits、Chamfer、
collision与policy不在保证内。P17/P17R/P19经验交换率只作冻结source frontier，P19不因效率最高而被事后选择。

P20/P21已进入CVPR主文和supplement；冻结双面板图把proxy低估与deletion utility frontier直接可视化。
Final main=8 content pages + 1 references page，supplement=9 pages；undefined citation/reference=0，关键页视觉QA通过。
V7三条补强方向现有可写证据为：

1. 三维硬证据：literal first return把hazard exposure从proxy `1.4362%`纠正到`8.7421%`；
2. 跨域：nuScenes-only P4在冻结AV2上保留经验zero-shot结论，但明确sensor-opportunity与rank-reversal边界；
3. 理论/工程：set-deletion单调方向可解释，hit/Chamfer/closed-loop安全边界明确。

按第11节停止条件，不再扩展deletion/router/head方法族。允许的后续工作限定为论文写作、图表、代码收敛、
supplement与合法metric纠错；第三批10-log AV2继续下载但保持未读，除非已有冻结source-passing候选按原合同授权。

## 2026-09-02 P20 hard-evidence correction supported, P21 safety boundary frozen

P20 true first-return揭示always-COMPLETE total/hazard early=`5.9561/8.7421%`，约为legacy proxy六倍。P17/P17R/P19
分别减少hazard events `1570/833/131`，但三者Chamfer都高于`.1945869m`，登记`V7-F29`。因此hard evidence修正
成立，但所有deletion policy仍非Pareto，fresh AV2未读。

P21形式化`S' subset S => d_{S'}(r)>=d_S(r)`，从而true first-return early对删除单调非增；hit/Chamfer无该保证。
只从P20 summary计算events removed per hit/mm，不重编译、不训练、不选policy；用于可解释性与安全边界章节，并关闭
deletion-only sweep。

实现输出`summary.json + BOUNDARY.md`，同时固化premise、corollary与non-guarantees；zero denominator保留null。当前
`p21_ready`，执行后直接进入论文安全边界整合。

## 2026-09-02 P19 rejected, P20 true first-return correction frozen

P19只veto 35/3325 candidates就把hazard early降34 events，但Chamfer退`.0000919m`且少136 new hits，登记
`V7-F28`并关闭veto sweep。代码审查同时发现legacy visible-failure是target-nearest point proxy，不是literal
first-return，这正是视觉/几何“硬证据”必须修正的评估边界。

按CVPR 2024 evidential occupancy raw-ray depth evaluation，P20固定每ray lateral tolerance内minimum positive depth，query/
compiled同算子。一次审计冻结P17/P17R/P19；checkpoint、policy、`.5`、tolerance、Chamfer、scene全不变，fresh AV2
不读。若三者均无true-ray Pareto，登记`V7-F29`并把安全--表面utility冲突写成明确边界。

实现使用chunked GPU `minimum positive depth`并在三policy间共享Actor baseline attribution；输出逐Actor审计rows与三组
独立Pareto decision。当前`audit_ready`，无optimizer、无target access。

## 2026-09-02 P18 rejected, P19 sparse hazard veto frozen

P18的P17R-dominant support只有fit `3/85`、consumed test `2/228`；router在test全选baseline，hazard early与Chamfer
均不变。登记`V7-F27`、fresh AV2未读，关闭Actor routing。post-verdict oracle只显示2 candidates/1 hazard event的
极窄上限，不能支撑继续调router。

按CVPR 2024 SparseOcc“稀疏避免empty-space hallucination”和CVPR 2025 EvOcc UNKNOWN/first-return迁移P19：hazard Actor
若有冻结P17R score `<.5`，仅minimum-score一个candidate标UNKNOWN；clear Actor恒always-COMPLETE。无训练、无新阈值，
source双Pareto失败即`V7-F28`，通过才允许fresh AV2 exact-once。

实现固定复用P17R checkpoint；source只编译consumed test，external在10/10 marker和source pass前拒绝创建run。当前
`source_ready`，直接执行一次source development，不做policy sweep。

## 2026-09-02 P17R rejected, P18 fixed-expert router frozen

P17R把coverage恢复到`88.96%`并保留hazard early下降，但Chamfer仍`.1945868→.1957160m`，登记`V7-F26`，
不放宽Pareto、不读fresh AV2。关闭继续调ray/Chamfer objective。

P18按ICML 2025 two-stage expert routing迁移系统动作：冻结always-COMPLETE与P17R，`23-32-2/seed71801/120
epochs/batch32/AdamW .001,.0001/inverse-sqrt CE`只路由完整expert。P17R label=逐Actor Chamfer与new-early双不退且
一项严格改善，否则baseline；argmax无threshold。Source gate通过才读取第三批AV2，下一failure=`V7-F27`。

P18实现已固定：router checkpoint只引用冻结P17R；always-COMPLETE action恢复完整baseline actor结果；AV2 external按log
串行编译以适应2GB无卡下载期内存。当前`fit_ready/target unread`，下一步直接执行单卡source fit。

## 2026-09-02 P17 partial result and single hybrid recovery

P17 canonical=`run://worldsim_v7/WS-V7-P17-RAY-SET-COMPLETION-FIT-01/
20260903T084500Z__ray-set-fit-s71701-r1`。Joint ray loss把hazard new-early从`1.4362%`降到`1.3760%`，但
Chamfer从`.1945868m`退到`.1994111m`、new hits少5,013，登记`V7-F25`并保持fresh AV2 unread。

按OccFlowNet“rendering + 3D supervision”迁移唯一P17R：架构/feature/seed/epoch/ray renderer全不变，只加入expected
bidirectional Chamfer；ray与Chamfer分别除以always-COMPLETE source reference后固定1:1相加，不扫mix weight。

## 2026-09-02 P17 joint ray-set completion frozen

针对`V7-F24`，P17只保留11维pre-target feature作控制变量，把prediction object改为整条ray的first-return set。
依据CVPR 2024 SelfOcc、official OccFlowNet与OpenOcc ray metric，用`T_i=product(1-alpha_j)`、`w_i=T_i alpha_i`
联合渲染candidate depth；固定KEEP/PROJECT为fallback，禁止通过删除core获益。

Exact=`64-64-1 / seed71701 / 160 epochs / Actor batch8 / max1024 rays / hard .5 forward / straight-through sigmoid /
Smooth-L1 depth / AdamW .001,.0001`。Source双Pareto通过才允许第三批AV2 exact-once；否则登记`V7-F25`并保持
external unread，不扫loss/threshold/temperature/ray budget。

## 2026-09-02 P16 source fit rejected, external preserved

P16 r2 canonical=`run://worldsim_v7/WS-V7-P16-EVIDENTIAL-COMPLETION-FIT-01/
20260903T073500Z__completion-fit-s71601-r2`。Source test FREE recall=`0`、UNKNOWN F1=`.0619`；更重要的是hazard
new-early=`1.4362→1.4811%`且Chamfer=`.1945868→.1950206m`，源域两个primary方向均退化，登记`V7-F24`。

根因不是继续调class weight即可解决：first-return由整条ray上的point set共同决定，移除独立candidate会暴露另一个更早
return。关闭P16 independent-point family，不运行external、不读取fresh AV2、不扫feature/threshold/seed；下一阶段只允许
joint ray/set transmittance或differentiable first-return depth模型。第三批10 logs继续下载但保持unread。

## 2026-09-02 P16 fit r1 implementation correction

`20260903T071500Z__completion-fit-s71601-r1`在首个source scene后因`package`解包NameError终止，发生于训练/checkpoint/
fresh AV2 read之前。只修复`row, package`局部变量并以r2重启；11维features、网络、seed、loss、cohort与decision全部不变。

## 2026-09-02 P16 evidential completion responsibility frozen

P16直接处理P15暴露的candidate-level COMPLETE机制，不再迭代Actor selector。按CVPR 2024 OccupancyM3D/evidential
occupancy、CVPR 2025 EvOcc与NeurIPS 2024 object-centric temporal completion迁移三态物理接口：nuScenes held-out ray
把candidate标为FREE/OCCUPIED/UNKNOWN；one fixed small network只在source train学习，argmax OCCUPIED才发
COMPLETE，其余UNKNOWN。KEEP/PROJECT、Actor状态、surface geometry与tolerance全部冻结。

Exact implementation=`11 features / 64-64-3 ReLU / seed71601 / 120 epochs / batch512 / AdamW .001,.0001 / source
inverse-sqrt class weights`；fit合并既有nuScenes train+calibration，test保持disjoint。唯一fresh decision为hazard new-early
严格降低且population Chamfer不差于frozen always-COMPLETE，不以query-only trivial fallback冒充支持。

新的10-log AV2 cohort在任何P16输出/quality read前按metadata冻结：150 UUID排序移除已消费50 logs，对剩余100 logs
取positions `0,10,...,90`。串行下载预计约11GiB，当前108GiB足够，75GiB free-space floor；IO期间并行推进单3090
source corpus/training。Fresh read后禁止threshold/feature/loss/seed/tolerance sweep或删失败样本；结论限定经验三维
自洽与nuScenes→AV2 transfer，不升级为calibration/causal/road-safety guarantee。

## 2026-09-02 P15 paper integration completed

P15 fresh hazard-by-action mechanism已进入CVPR main/supplement、结果宏、bibliography与contribution map。Main将
`COMPLETE=94.70% new early / 99.94% new hits`与`KEEP=95.68% contradictions`解释为两个并存的物理失败通道，
并报告P4/P6-C hazard rate相对always=`1.0055x/1.0010x`，不再把aggregate修复包装为hazard-risk改善。

Limitations固定nearest-output provenance非action counterfactual、PROJECT被KEEP-first voxel dedup折叠的边界。
Final main=`10 pages/1,771,543 bytes`（8 content + 2 references），supplement=`8 pages/7,228,331 bytes`；main 7与
supp 2视觉QA通过，唯一warning仍为既有Table 1 `6.03pt` overfull。P15全部完成，下一研究只接受能改变三维物理
机制、独立跨域证据或安全边界的新信息，不继续同cohort无穷描述审计。

## 2026-09-02 P15 fresh hazard-by-action mechanism result

Raw/audit canonical分别为`run://worldsim_v7/WS-V7-P15-FRESH-HAZARD-ACTION-ATTRIBUTION-01/
20260903T043000Z__fresh-hazard-action-raw-s0-r1`与`run://worldsim_v7/WS-V7-P15-FRESH-HAZARD-ACTION-AUDIT-01/
20260903T044500Z__fresh-hazard-action-audit-s0-r1`。20 logs/523 Actors/1,435,391 target rays；raw单3090 wall=
`46.70s`、peak GPU/RSS=`.0996/1.186GiB`，均status=`done`。

Aggregate new-early=`19,354/1.348%`，new hit=`280,889`，ratio=`14.51`。COMPLETE解释`94.70%` new early与
`99.94%` new hits；KEEP却解释`95.68%` surface contradictions，证明early termination与surface contradiction是两个
不同物理失败通道。Hazard/clear new-early=`1.912/.980%`（`1.951x`）；P4 selected=`1.922/.968%`（`1.985x`），
hazard rate为always的`1.0055x`；P6-C为`1.0010x`。

结论：P4/P6-C没有过滤hazard中的completion-driven early-return机制；hazard COMPLETE hit/early也低于clear
`13.67 vs 17.62`。PROJECT=0来自KEEP-first voxel dedup，不是0 causal harm。该结果与既有claim不冲突，不登记
`V7-F24`；0 training/fit/threshold/action/cohort change。完整result=`P15_FRESH_HAZARD_ACTION_ATTRIBUTION_RESULT.md`。

## 2026-09-02 P15 fresh hazard-by-action mechanism audit frozen

P14暴露hazard-stratum risk未降后，P15冻结为机制追踪而非新head：在已消费20-log fresh AV2上以原P2 compiler、
`.20m` ray/depth tolerance和observed-hit PROJECT复跑一次CUDA P3-D attribution，再exact join既有P4/P6-C scores。
固定报告always/P4/P6-C selected与abstained在hazard/clear的new-early、new-hit、resolved-early、surface contradiction
及KEEP/PROJECT/COMPLETE来源，不改模型/threshold/action/cohort。

迁移CVPR 2016 ray visibility constraint、CVPR 2025 EvOcc first-occupied ray evaluation与ICCV 2021 early/late occupancy
asymmetry；不迁移planning/safety claim。PROJECT可能因KEEP-first voxel dedup归并为KEEP，0计数不能解释为0 causal harm。
一次单3090 raw run + CPU join，无训练/fit/gate；freeze=`P15_FRESH_HAZARD_ACTION_ATTRIBUTION_FREEZE.md`，下一failure=
`V7-F24`。

## 2026-09-02 P14 hazard-stratified defer result

Canonical=`run://worldsim_v7/WS-V7-P14-HAZARD-STRATIFIED-DEFER-01/
20260903T031500Z__hazard-stratified-s0-r1`；20 logs/523 Actors，其中hazard=`142/27.15%`。两条finite-sample
accounting identity最大残差=`5.55e-17`。P4 hazard/clear coverage=`93.66/71.13%`，selected visible risk=
`47.37/31.00%`；相对always的`46.48/33.86%`，hazard反而`+0.89pp`、clear=`-2.86pp`。

P4总体少48个introduced failure，但hazard只少3个、clear少45个；hazard仅占Actors `27.15%`，却占P4 failures
`42.86%`（burden amplification=`1.578x`）并贡献`56.42%` Chamfer gain。P6-C 34个failure reduction全部来自clear；
P4∧visibility的0 hazard failure建立在仅`3.52%` hazard coverage与`.00122m`总gain上。

结论：Actor/hazard字段100%保留不等于hazard-stratum visibility安全；aggregate改善主要由clear stratum驱动，低risk
方案则几乎不修hazard。该结果未与既有“hazard-preserving只指状态不变”的claim冲突，不登记`V7-F24`。0 dataset/
training/fit/calibration/threshold/gate change；完整result=`P14_HAZARD_STRATIFIED_DEFER_RESULT.md`。
Paper integration完成：新增selective/defer相关工作、Method exact decomposition及main/supp hazard-stratum evidence。
TeX Live main=`10 pages/1,770,484 bytes`（content 1--8、references 9--10），supplement=`8 pages/7,227,451 bytes`；
main 2--3/7--8与supp 2视觉QA无裁切、重叠或断式，唯一既有warning为Table 1 `6.03pt` overfull。

## 2026-09-02 P14 hazard-stratified defer contract frozen

P14冻结为P13完整fallback world的hazard/clear精确分层审计。对每个既有policy只计算两条有限样本恒等式：
`composite gain=coverage*selected gain`，`introduced-visible mass=coverage*selected risk`，总体量再按stratum Actor
占比加权。额外报告hazard failure burden与gain contribution，直接解释aggregate trade-off是否掩盖危险Actor代价。

迁移NeurIPS 2018 learning-to-defer的完整系统视角、ICML 2019 selective risk--coverage与ICML 2022 subgroup
selective-regression警告；不套用ICLR 2024 conformal risk control，因为不选阈值且不声明跨域exchangeability。
只读P13 canonical JSONL，0 dataset/checkpoint/model/fit/calibration/threshold/policy search，0新增gate；下一failure保持
`V7-F24`。冻结合同=`docs/autoresearch/worldsim_v7/P14_HAZARD_STRATIFIED_DEFER_FREEZE.md`。

## 2026-09-02 P13 defer-to-query composite world result

Canonical=`run://worldsim_v7/WS-V7-P13-DEFER-TO-QUERY-COMPOSITE-01/
20260903T023000Z__defer-to-query-s0-r1`。完整fallback policy下，P4/P6-C/P4∧visibility的population introduced-visible /
composite Chamfer gain分别为`28.11%/.08311m`、`30.78%/.08895m`、`.76%/.00122m`，形成repair-policy Pareto front。
P11 provenance=`4.78%/-.00088m`，被query-only=`0/0`严格支配；visibility-only也被P4∧visibility支配，登记`V7-F23`。

结论：selected conditional risk必须与fallback composite utility共同报告；P6-C只保留AV2 frontier地位，P12 dual只保留
低risk/近零gain边界，均不推翻既有跨域/coverage rejection。0 dataset/model/threshold/gate change。Waymo official GCS
仍403且不绕license；下一failure=`V7-F24`。
Paper integration完成：main=`9 pages/1,762,749 bytes`、supplement=`8 pages/7,226,250 bytes`；main 7--8、supp 4--5
视觉QA通过。Failure ledger已补齐V7-F20--F23，P13 result与claim boundary可回溯。

## 2026-09-02 P10/P11 predicate-semantics correction completed

P12独立统计暴露P10/P11 r1把safe predicate `nonnew_visible_violation=true`反向计成failure。`d8bf0df`已最小修复为
failure=`not nonnew`；r1保留审计但不再canonical。P10/P11 r2只读现有JSONL并均正常完成，不重读数据、不训练、
不改selection/gate/cohort。Canonical分别为`20260903T010000Z__physical-authority-s0-r2`与
`20260903T011500Z__provenance-authority-s0-r2`；V7-F20/F21与paper全部按正确方向重述。
Final TeX Live main=`9 pages/1,762,273 bytes`、supplement=`8 pages/7,225,029 bytes`；main 6--8与supp 2--3视觉QA
无裁切/重叠。此纠错里程碑完成后才允许进入下一项research；下一failure id=`V7-F23`。

## 2026-09-02 P12 nuScenes-only visibility authority result

Canonical=`run://worldsim_v7/WS-V7-P12-NUSCENES-VISIBILITY-AUTHORITY-01/
20260903T004500Z__visibility-authority-s71201-r1`。Source safe AUROC=`.753`，AV2 safe AUROC=`.625`，证明target-specific
visibility ranking可nuScenes→AV2 zero-shot迁移。Visibility-only AV2 coverage/risk/upper/Chamfer-worse=
`8.22/11.63/22.02/37.21%`；P4 dual=`7.46/10.26/20.98/35.90%`，hazard coverage=`3.52%`。

七门前三类visibility目标通过，但Chamfer、minimum coverage与hazard coverage失败；登记`V7-F22`，关闭head family。
不扫seed/architecture/feature/source coverage/AV2 threshold；下一可用failure=`V7-F23`。

## 2026-09-02 P11 provenance-conditioned authority corrected result

Canonical=`run://worldsim_v7/WS-V7-P11-PROVENANCE-AUTHORITY-AUDIT-01/
20260903T011500Z__provenance-authority-s0-r2`。P4∧no-COMPLETE在523 Actors上coverage=`23.71%`，visible failure=
`20.16%`、upper=`26.70%`，但Chamfer-worse=`43.55%`、hazard coverage=`3.52%`；五门=
`pass/pass/fail/pass/fail`，保留`V7-F21`。正确结论是observed provenance改善visibility，却不能形成joint
geometry/hazard/future-view authority；关闭completion-count/provenance gate family，不删completion、不扫threshold。

## 2026-09-02 P11 provenance-conditioned authority contract

在检索CVPR ray visibility、SelectiveNet selective risk与DGLSS source consistency后，P11冻结一个更硬且可解释的
迁移：只在`COMPLETE==0`时认定compiled output全部由KEEP或matched-hit PROJECT观测见证，并与P4 stored authority
取交集。固定五门为visible risk、Wilson upper、Chamfer tail、10% coverage、50% hazard coverage；只读已消费fresh
AV2 join，不训练/调threshold/改compiler。Observed-ray witness不等于future-view completeness，失败即`V7-F21`。

## 2026-09-02 P10 frozen physical-authority corrected result

Canonical=`run://worldsim_v7/WS-V7-P10-FROZEN-PHYSICAL-AUTHORITY-AUDIT-01/
20260903T010000Z__physical-authority-s0-r2`。523 fresh AV2 Actors上，P4 coverage=`77.25%`并把Chamfer-worsening从
`19.50%`降到`14.60%`；selected visible failure也从`37.28%`轻微降到`36.39%`，但95% upper=`40.40%`仍高于
always point risk，safe-visible AUROC=`.533`、capture=`24.62%`。三门=`pass/fail/pass`，保留`V7-F20`并禁止
本cohort recovery。论文主张收窄为：P4只授予frozen Chamfer authority，不能提供visibility confidence separation。

## 2026-09-02 P10 frozen physical-authority audit contract

为把三维硬证据与可解释安全边界直接连接，P10只join冻结P3-C fresh visibility与P6-C/P4 fresh scores，不训练、
不重读AV2、不调threshold。P4 stored selection为primary，固定衡量selected/abstained visible violation、Chamfer tail、
abstention capture、AUROC/AURC、hazard coverage与selected risk一侧95% Wilson upper。Strong empirical containment三门
在aggregate前冻结；失败即形成`V7-F20`边界，不能在同cohort恢复。P6-C只作context，不能替换P4论文主模型。

## 2026-09-02 user-directed V7 focus lock

后续auto-research聚焦三条主线并做深：（1）视觉/几何硬证据必须落到可观测三维ray、surface、visibility与
Actor-level反例，不把feature filtering或overlay写成物理证明；（2）以nuScenes-only训练、AV2/Waymo direct test为
核心跨域合同，不用target refit/recalibration掩盖sensor shift；（3）把可解释性绑定到确定性物理对象、Actor-level
decision interval、risk--coverage与明确UNKNOWN/abstention安全边界。优先复用单卡GPU并与IO串行/错峰；只有任务确实
需要多卡时才按用户约束shutdown等待。减少P0/smoke/regression，不增加hash/checksum/fingerprint或过度gate。

## 2026-09-02 metadata-locked four-quadrant teaser milestone

Figure 1已用冻结P3-B metadata组成`valid-safe / artifact-safe / valid-hazard / artifact-hazard`四象限：first
non-hazardous main=`q01-a0`、first hazardous main=`q00-a0`。每个clean/artifact pair保持Actor、trajectory、extent、
camera、hazard完全一致，只改变canonical panel内的synthetic-artifact overlay；不按视觉质量或metric选case，不新增
render/scientific read。图下保留HARP-3D五步pipeline，并明确overlay不是photorealistic reconstruction或complete
unseen-world proof。

Final main compile=`9 pages/1,760,570 bytes`，pages 1--8 content、page 9 references-only；Figure 1 page 3 at 160 dpi
visual QA通过，无裁切/重叠，唯一warning为既有Table 1 `6.03pt` non-clipping overfull。资产索引、contribution map、
status/experiments/failures同步完成；无新failure，下一可用=`V7-F20`。

## 2026-09-02 P5/C3 direct reliability paper-evidence milestone

按原计划Table 3要求补齐C3 direct evidence，但不违反final-test后冻结：复用V6.7 canonical而不重训/重读target。
P182/P183在独立10-log fresh scene-level cohort相对P173的mean integrated-Brier/calibration reduction=
`28.48%/69.38%`；P199 source-heldout 3,742 trajectories=`16.97%/71.85%`，P201独立10-log/1,846-trajectory
fresh confirmation=`17.52%/53.37%`。P346仅作边界：reused P201 q90 coverage/unsafe=`32.84%/9.09%`，source
held-out-H=`29.42%/26.71%`。

新增macro-driven C3主表、Experiments段、limitations与contribution registry；明确P183/P201是scene-level fresh，P346
是reused development。不得称V7 physical repair effect、AV2 reliability transfer、cross-horizon stability或formal
calibration/safety guarantee。无新run/failure，下一可用=`V7-F20`。

Final compile main=`9 pages/1,171,711 bytes`：pages 1--8为全部content，page 9仅cited references；CVPR 2026 official
guideline明确references-only additional pages不计8-page limit。Supplement=`7 pages/7,222,572 bytes`。Main pages 6--9
visual QA通过，无裁切/重叠/断表，唯一warning仍为既有Table 1 `6.03pt` non-clipping overfull。

## 2026-09-02 fresh exact-once paper-integration milestone

P3-C fresh visibility与P6-C fresh AV2/fresh-nuScenes reversal已完整进入CVPR main、supplement、结果宏、主表和
contribution map；P4保持primary，`V7-F18/V7-F19`不被外域pass覆盖。Final compile main=
`8 pages/1,169,034 bytes`、supplement=`7 pages/7,222,572 bytes`；main pages 4--8及supplement pages 2--4
逐页QA通过，无裁切/重叠/断表，唯一既有Table 1 `6.03pt` overfull视觉未裁切。此里程碑完成，不触发后验
refit/recalibration/threshold/cohort/claim修改；无新failure，下一可用=`V7-F20`。

## 2026-09-02 P3-C fresh visibility exact-once result

Canonical=`run://worldsim_v7/WS-V7-P3C-AV2-VISIBILITY-CERTIFICATE-FRESH-01/
20260902T231500Z__fresh-visibility-s0-r1`。P6-C正常退出后，同一预冻结runner在20 fresh logs/523 Actors上串行
exact-once执行。query→compiled hit recall=`.497233→.689919`、early=`.034478→.029314`、visible precision=
`.995141→.996813`、F-score=`.663127→.815447`、surface contradictions=`912→509`；20/20 partitions与Actor/hazard
retention通过。

Actor-level nonnew-visible=`.627151`、exact-zero=`.022945`、Chamfer-worsened=`.195029`，worsened stratum的F-score
noninferior仅`.166667`。故fresh cohort独立确认aggregate ray/depth/surface physics与`V7-F18`边界，不升级为universal
certificate，不调`.20m` tolerance/operator，不删除completion或失败Actor。下一failure id保持`V7-F20`。

## 2026-09-02 P6-C fresh AV2 exact-once result / cross-fresh boundary

Canonical=`run://worldsim_v7/WS-V7-P6C-SPARSITY-CONSISTENT-SELECTOR-01/
20260902T173000Z__sparsity-consistent-s70602-r1`。20/20 recovery logs与`ALL_COMPLETE`后，frozen P6-C/P4在523 Actors
上exact-once读取：candidate/P4 repair AUROC=`.676168/.654837`、coverage=`.839388/.772467`、false-repair=
`.124283/.112811`、selective Chamfer=`.178291/.184133m`、score-shift Wasserstein=`.203971/.207216`；外域四门通过。

但 P8-A fresh nuScenes 已显示 candidate/P4 AUROC=`.747253/.782280`、AURC=`.126429/.105633`，方向相反；且fresh AV2
candidate false-repair比P4高`1.15pp`。故`V7-F19`固定结论为“外域支持、跨fresh域不稳定”，P6-C不晋升为全局或paper
primary selector，P4保持主模型。禁止以AV2 pass回写模型、threshold、gate或P8-A结论。P3-C fresh继续按原合同串行执行，
不读取该verdict作方法分支。

## 2026-09-02 fresh P3-C serial auto-launch contract

P3-C fresh canonical run预定为 `20260902T231500Z__fresh-visibility-s0-r1`；单实例 watcher等待20/20下载、
`ALL_COMPLETE`、P6-C canonical summary/status与 evaluator退出后才串行exec，不并发GPU、不启动第二下载器。
P6-C verdict不作为分支条件，P3-C仍使用已冻结20 logs、`.20m` ray/depth tolerance和 observed-hit PROJECT。
existing run path直接停止而非第二次读取。启动前为15/20，P6-C/P3-C summary均未产生。
Watcher已由commit `b775807` 启动为PID `33968`；首次日志确认16/20、P6-C/P3-C summary均未产生，保持等待。
最终20/20与`ALL_COMPLETE`后，P6-C和P3-C按合同串行完成、均status=`done`且进程正常退出；无重复read。

## 2026-09-02 P3-C/P3-D paper integration milestone

CVPR main abstract/method/experiments已加入 visibility-conditioned双向状态、pooled hard evidence、per-Actor `V7-F18`
boundary与 completion `14.96×` hit/early gain；supplement加入ray/surface partition、三行表、全Actor provenance与failure
ledger。Contribution map同步 P3-C/P3-D canonical ownership。TeX Live final compile main=`8 pages/1,167,975 bytes`、supplement=
`7 pages/7,221,609 bytes`；PDF visual QA无裁切/重叠/断表，main只保留既有 Table 1 `6.03pt` non-clipping warning。
本里程碑不改变任何科学run/claim boundary/fresh exact-once合同。

## 2026-09-02 official-template provenance recheck

官方 `cvpr-org/author-kit` main HEAD仍为已固定 `2917585`，latest release仍为 CVPR2026，尚无 CVPR2027 kit。
因此保持现有官方 `cvpr.sty`/BibTeX style，不采纳第三方模板；`TEMPLATE_PROVENANCE.md` 已记录复核。正式 2027 kit
发布后才允许迁移并重新检查 page limit。

## 2026-09-02 paper source-convergence milestone

最终 source package已移除从未被 `\input` 的 legacy `main_results.tex`、18个 placeholder macros及 `\todoresult`；
所有真实 P1--P9/P7-C result macros与五张当前表保留。TeX Live compile保持 main 8页、supplement 6页，warning不增加。
该里程碑清除误导性 TBD source，不改变 scientific result；该时点fresh AV2仍由单实例watcher等待，后续已20/20
完成并按顶部里程碑集成。

## 2026-09-02 corrected V6.7 branch-base audit

当前 `research/worldsim-v7-harp3d-cvpr` 的 exact merge-base 为用户指定的
`research/worldsim-v6.7-anisotropic-surface` terminal `d97c3f2`；ancestor=true，审计时 ahead/behind=`50/0`。
因此本计划严格从 V6.7 继续，不使用 main、V6.4 或 V6.5 base。该项只读，不改变 Git graph 或 scientific state。

## 2026-09-02 contribution/evidence-map synchronization

`paper/CONTRIBUTION_MAP.md` 已同步当前4项贡献、Method 4.1--4.4/Experiments位置、实际 V7 code interface、P3至P9
canonical registry、V7-F09/F11--F17 与 allowed/prohibited claims。该时点ownership固定为 P4 primary selector、P8-A
fresh nuScenes rejection、P6-C仅等待fresh AV2；后续external结果已完成并登记`V7-F19`，仍未覆盖既有模型、阈值、
cohort、gate或verdict。该里程碑修正文档漂移，不新增 scientific read。

## 2026-09-02 project-page/video asset milestone

`paper/PROJECT_PAGE_ASSET_INDEX.md` 已索引 P3-B canonical 的 10 logs/30 Actors、30 panels 与30 MP4，包含完整
log/Actor identity、hazard、camera、query visibility、crop depth及 main/supplement role。8 main、10 compact supplement、
30 full-package cases均保持预注册身份，弱可见性和 Chamfer-worsening failures不删；46MiB bundle只保留 canonical run
一份，不在Git重复存储。计划交付物“项目页与视频素材清单”完成；该时点fresh AV2 quality未读，后续已exact-once完成。

## 2026-09-02 cross-sensor LiDAR literature boundary

CVPR main Related Work 已补充 DGLSS（CVPR 2023）与 3DLabelProp（ICCV 2023）：source subsampling consistency和
sequence/geometry common representation可缓解 LiDAR sensor shift，但不能替代 actor-level repair 的 exact-once
source/external evidence。官方模板仍为 `8 pages/1,165,317 bytes`，pages 2/7/8 QA通过；fresh AV2 verdict保持未读。

## 2026-09-02 P6-C fresh AV2 auto-launch contract

已加入单锁低资源 watcher：只在冻结 recovery cohort 的 `20/20 .complete` 与 `ALL_COMPLETE` 同时成立后，原样执行
P6-C canonical run 的一次 CUDA external read；等待期间不 compile/score、不读取 target quality。重复 watcher、非 waiting
status 或下载器提前退出均停止，不启动第二下载器、不换 log、不修改 model/standardizer/threshold/gate。启动时下载
`11/20`、剩余磁盘 `117G`、无 error/retry；formal external result仍未读。

## 2026-09-02 supplement evidence milestone

`paper/supplement.tex/.pdf` 已从模板骨架扩展为 6 页 evidence-rich supplement：SceneIR/四动作合同、matched-ray
physical certificate、冻结 cohort/exact-once、P7-C actor-level interval、V7-F09/F11/F13--F17 failure ledger、
10 个预注册顺序的 AV2 camera-evidence panels 与 resource/reproducibility boundary。官方模板编译为
`6 pages/7,219,027 bytes`，逐页 QA 无空白页、float 乱序、裁切或重叠。该里程碑只收敛论文资产，不改变任何科学
run、cohort、threshold 或 gate；完整 30-panel/30-video 证据包继续由 P3-B canonical artifact 提供。

## HARP-3D：危险保真、伪影修复的三维物理世界编译器

> **Hazard-Preserving, Artifact-Repairing Physical 3D World Compiler**
>
> 面向 CVPR / ICCV 主会投稿；先发布 arXiv，后提交匿名主会版本。

---

## 0. 计划定位

本文档是 **V7 的 Markdown 研究执行计划**，不是论文正文。

V7 的核心研究成果之一，是在研究推进过程中同步产出并持续更新一套 **CVPR/ICCV 风格的 LaTeX 论文初稿与可编译 PDF**：

```text
paper/
├── main.tex
├── supplement.tex
├── sections/
├── figures/
├── tables/
├── bibliography.bib
└── results/results_macros.tex
```

计划本身继续使用 Markdown，方便 Codex / AutoResearch 直接读取、修改和执行；论文初稿是 V7 的正式交付物，而不是计划格式。

---

# 1. 项目北极星

V7 不再继续扩展 V6.7 后期的多条件 authority head，也不以强化学习（Reinforcement Learning，RL）为主论文贡献。

主问题收敛为：

> **高质量驾驶世界不应通过删除危险 Actor 获得“安全”，而应修复违反传感器证据、三维几何与时空连续性的伪影，同时保留合法但危险的交通参与者。**

论文中心命题：

\[
\boxed{\text{World validity} \neq \text{World difficulty}}
\]

进一步分解为：

\[
\boxed{\text{Artifactness} \neq \text{Hazardness}}
\]

\[
\boxed{\text{Appearance} \neq \text{Physical collision state}}
\]

\[
\boxed{\text{Physical validity} \neq \text{Task-conditioned reliability}}
\]

V7 的目标不是继续做一个更强的 Occupancy 小头，而是形成以下显式世界表示：

\[
\mathcal W_t =
\left(
\mathcal G_t^{app},
\mathcal S_t^{phys},
\mathcal A_t,
\mathcal P_t,
\mathbf q_t^{val}
\right)
\]

其中：

- \(\mathcal G^{app}\)：外观高斯或渲染表征；
- \(\mathcal S^{phys}\)：显式可碰撞物理表面、符号距离场或占据表面；
- \(\mathcal A\)：带身份、轨迹、生命周期和危险属性的动态 Actor；
- \(\mathcal P\)：来源、射线、多帧支持和传感器证据；
- \(\mathbf q^{val}\)：局部有效性、未知性和物理证书。

强制保留：

```text
Gaussian opacity != occupancy probability
Appearance Gaussian != collision volume
Unknown != free
Hazardous Actor != artifact
```

---

# 2. 面向论文的三项主贡献

V7 将 70% 左右的主文篇幅、图表和实验预算集中在 C1–C3。

## C1：危险保真的三维物理几何编译

### 核心问题

V6.7 已证明运动补偿后的 Actor 内向射线修复可以降低局部冲突并保留 Actor 身份、轨迹和危险语义，但当前操作仍主要是：

```text
KEEP / UNKNOWN
```

容易被视觉审稿人视为几何过滤或后处理规则。

### V7 升级目标

将 Actor-local 物理编译升级为：

```text
KEEP
PROJECT
COMPLETE
UNKNOWN
```

并建立显式 Actor canonical physical surface：

\[
S_i^{phys}
=
\operatorname{Fuse}
\left(
\{T_{actor,t}^{-1}T_{ego,t}T_{sensor}x_t\}_{t}
\right)
\]

目标是证明：

1. 被删除或修复的是证据不一致的伪影；
2. 合法危险 Actor 的存在、身份、轨迹、尺寸、TTC 和危险行为保持；
3. 修复后的表面在三维空间、射线可见性和时间维度上更自洽；
4. 外观层和物理层明确分离。

---

## C2：有效性与危险性的条件解耦

### 核心问题

不能将重建困难、风险较高或与 Ego 接近的 Actor 自动判为无效。

V7 不追求全局统计独立，而追求 **配对条件不变性（paired conditional invariance）**。

定义：

\[
z_i^{val}
=
f_{val}(E_i,S_i^{phys},P_i)
\]

\[
z_i^{haz}
=
f_{haz}(X_{i,0:H},\tau,\mathcal M)
\]

对应：

\[
a_i=P(\text{artifact}\mid z_i^{val})
\]

\[
h_i=P(\text{hazard}\mid z_i^{haz})
\]

配对约束：

- 仅把同一 Actor 从普通驾驶改成合法加塞、急刹或近碰撞时：

\[
a_i(A_i^{safe})\approx a_i(A_i^{hazard})
\]

- 仅在同一 Actor 上注入 duplicate shell、ghost、flicker 或 teleport 时：

\[
h_i(A_i^{clean})\approx h_i(A_i^{artifact})
\]

V7 需要通过四象限数据、配对反事实、交叉探针和表征可视化证明：

```text
合法且普通
合法且危险
伪影且普通
伪影且危险
```

可以被正确区分，而不是把“危险”清洗掉。

---

## C3：连续任务代价密度与多时域可靠性

V6.7 已获得较强证据：直接建模连续任务代价密度，比为多个阈值分别训练二元分类器更稳定。

定义候选 Ego 轨迹 \(\tau\)、Actor 残差 \(\epsilon_{i,t}\)、轨迹边界法向 \(n_{\tau,t}\) 和净空 \(d_{i,t}(\tau)\)：

\[
C(\tau,H)
=
\max_{i,t\le H}
\frac{
|n_{\tau,t}^{\top}\epsilon_{i,t}|
}{
\max(|d_{i,t}(\tau)|,\varepsilon)
}
\]

学习连续条件密度：

\[
p_\theta\left(\log(1+C)\mid z^{val},z^{haz},\tau,H\right)
\]

查询可靠性：

\[
R(\tau,H,b)
=
P_\theta(C(\tau,H)\le b)
\]

必须满足：

\[
\frac{\partial R}{\partial b}\ge0
\]

\[
\frac{\partial R}{\partial H}\le0
\]

V7 不重新搜索大量 density family，而是整合并简化 V6.7 已成立的链路：

```text
Actor residual distribution
→ analytic trajectory boundary query
→ continuous cost density
→ input-conditioned multi-horizon dependence
→ monotone runtime reliability surface
```

重点转为：

- 跨数据集零样本迁移；
- 与物理几何质量的因果联系；
- 解释性；
- 统一 paper implementation；
- 最终一次 exact-once test。

---

# 3. 论文定位与 Claim Boundary

## 3.1 推荐论文标题

首选：

> **Validity Is Not Difficulty: Hazard-Preserving Physical 3D World Compilation for Autonomous Driving**

备选：

> **HARP-3D: Hazard-Preserving and Artifact-Repairing Physical World Compilation for Autonomous Driving**

## 3.2 一句话摘要

> A valid driving-world compiler should repair evidence-inconsistent 3D artifacts while preserving legitimate hazardous actors, and should expose task-conditioned reliability through a continuous multi-horizon cost distribution rather than a single confidence score.

## 3.3 可以主张

- Actor-preserving、ray-consistent 三维物理编译；
- validity 与 hazard 的配对条件解耦；
- learned Actor uncertainty 与 analytic task geometry 的因子化；
- 连续代价密度和 input-conditioned multi-horizon reliability；
- nuScenes 训练、Argoverse 2 零样本测试；
- 显式、可查询、确定性的 runtime SceneIR；
- 下游候选轨迹选择可以消费该可靠性接口。

## 3.4 不能主张

- 形式化现实道路安全保证；
- 所有物理伪影都能被修复；
- Gaussian opacity 是 collision occupancy；
- 所有 Surface/CVaR/UQ 方法均失败或均成立；
- 完整 RL-ready simulator 已经成立；
- 仅凭 proxy metric 即证明 closed-loop policy 提升；
- Argoverse 2 上做过 fine-tuning 时仍称零样本。

---

# 4. V7 总体架构

```text
Real Logs / Reconstructed World
              │
              ▼
┌──────────────────────────────────────┐
│ Actor-Preserving Physical Compiler   │
│                                      │
│ multi-frame Actor alignment          │
│ ray FREE/OCC evidence                │
│ canonical physical surface / SDF     │
│ KEEP / PROJECT / COMPLETE / UNKNOWN  │
└────────────────┬─────────────────────┘
                 │
                 ▼
┌──────────────────────────────────────┐
│ Validity–Hazard Factorization        │
│                                      │
│ validity: evidence / geometry        │
│ hazard: dynamics / TTC / route       │
│ paired invariance / leakage probes   │
└────────────────┬─────────────────────┘
                 │
                 ▼
┌──────────────────────────────────────┐
│ Task-Conditioned Reliability         │
│                                      │
│ Actor residual distribution          │
│ analytic trajectory-boundary query   │
│ continuous task-cost density         │
│ joint multi-horizon dependence       │
└────────────────┬─────────────────────┘
                 │
                 ▼
┌──────────────────────────────────────┐
│ Monotone Runtime Reliability Surface │
│ R(trajectory, horizon, budget)       │
└────────────────┬─────────────────────┘
                 │
                 ▼
       Planner / Action Selector / RL demo
```

---

# 5. 执行阶段

## V7-P0：Paper-first 分支与最小代码收敛

### P0 执行记录（2026-09-01）

- 基线：V6.7 `d97c3f2`；分支：`research/worldsim-v7-harp3d-cvpr`。
- 论文：固定官方 `cvpr-org/author-kit@2917585`（当前官方 CVPR 2026 kit；2027 kit 尚未发布），建立匿名
  `main.tex`、`supplement.tex`、章节、结果宏、架构图和 contribution map；主稿与补充材料均已编译。
- 代码：已从历史链抽出 paper-facing `worldsim_v7` 六模块，固定
  `KEEP/PROJECT/COMPLETE/UNKNOWN`、validity--hazard 输入隔离、Actor residual、连续 cost density、单调 runtime
  surface 与 AV2 SceneIR 适配器；不回改历史 Pxxx。
- 数据：AV2 Sensor `val` 的 150 个 UUID 仅按 metadata 排序，每隔 5 个冻结 30 logs；前 20 个 quantitative、后
  10 个 qualitative。冻结发生在任何方法输出或质量 read 之前，且禁止 AV2 fine-tune、校准、阈值选择和失败场景删除。
- 传输：远端小文件下载实测约 `0.58 MiB/s`，本地 D 盘约 `2.86 MiB/s`，故按预案改为本地串行下载后手工上传；
  远端只保留 `s5cmd`，测速残片删除。
- 验证：P0 定向单元测试 `5 passed`；完整首个 AV2 log 的 metadata-only SceneIR smoke 得到
  `2671` ego poses、`11` sensors、`59` Actors、`5337` states；本轮不训练、不做 repo-wide regression。
- 失败账本继承：`V66-F01`（Actor existence 与 local geometry 分权）、`V66-F02`（triage 不等于 physical
  repair）、`V67-F232`--`V67-F236`（后期可靠性目标错配）；本轮资源/协议事件登记为 `V7-F01`。

### 目标

从实际远端 V6.7 最新 HEAD 新建：

```text
research/worldsim-v7-harp3d-cvpr
```

建立 paper-facing 目录：

```text
paper/
├── main.tex
├── supplement.tex
├── sections/
│   ├── 01_intro.tex
│   ├── 02_related.tex
│   ├── 03_problem.tex
│   ├── 04_method.tex
│   ├── 05_experiments.tex
│   ├── 06_limitations.tex
│   └── 07_conclusion.tex
├── figures/
├── tables/
├── results/results_macros.tex
└── bibliography.bib
```

重构 paper-facing 实现：

```text
motion_proj/worldsim_v7/
├── physical_compiler.py
├── validity_hazard.py
├── actor_reliability.py
├── boundary_cost_density.py
├── runtime_surface.py
└── sceneir_adapter.py
```

### 原则

- 不做 repo-wide regression；
- 不加哈希、指纹、checksum；
- 不重构所有历史 Pxxx；
- 仅将论文最终依赖链从历史实验脚本中抽出来；
- 每个 V7 里程碑同步更新 LaTeX 结果宏、表格和图占位。

### 交付物

- `WORLDSIM_V7_CVPR_RESEARCH_PLAN.md`；
- 可编译的 CVPR/ICCV 风格匿名论文骨架；
- 第一版 `main.pdf`，允许结果为 TBD；
- 一张总架构图草稿；
- 一页 paper contribution map。

---

## V7-P1：四象限物理—危险证据图谱

### P1 执行冻结（2026-09-02）

- Formal task：`WS-V7-P1-AV2-FACTORIAL-ATLAS-01`；冻结 AV2 quantitative 20 logs，zero-shot only。
- 使用官方 AV2 ego-frame LiDAR/cuboid 合同，在 GPU 上完成 Actor-local 入盒、坐标变换、`0.12m` surfel fusion
  和 held-out surface evaluation；固定 2/3 build、1/3 held-out，不按质量换帧或 Actor。
- 迁移 NeuRAD/SplatAD 的 Actor-local point seeding 边界，但禁止其 mirrored seed augmentation 进入物理层。
- 每个真实 surfel 产生 clean 与 paired artifact probes，Actor ID、轨迹、尺寸和 hazard 不变；Validity/Hazard
  输入严格隔离。配置与 gates 在 formal quality read 前提交并推送。
- 首次 verdict 后按冻结停止条件决定：surface 指标支持则进入 P2 四动作真实几何编译，否则直接进入 P2 canonical
  surface 表示修复，不在 P1 扫 voxel/threshold/cohort。

### P1 执行结果（2026-09-02）

Canonical=`run://worldsim_v7/WS-V7-P1-AV2-FACTORIAL-ATLAS-01/20260902T104500Z__av2-factorial-s0-r1`；10/10 gates，
1,046 Actors（241 hazard）、1.075M surfels、134,914 paired probes。真实 held-out LiDAR target distance
`0.864→0.131m`，recall `26.16%→85.72%`；Actor/hazard retention=100%。paired synthetic corruption 的 action/
artifact score=100%、clean-hazard false-artifact=0，只登记为接口合同证据。P1 参数族关闭，进入 P2。

### 数据构造

建立四象限：

| 物理有效性 | 普通行为 | 危险行为 |
|---|---|---|
| 合法 | valid-safe | valid-hazard |
| 含伪影 | artifact-safe | artifact-hazard |

危险行为至少包括：

- cut-in；
- hard braking；
- close merge；
- pedestrian/cyclist crossing；
- near miss；
- physically feasible collision。

伪影至少包括：

- observed-FREE ghost；
- duplicate shell；
- floating surface；
- temporal pop/flicker；
- teleport；
- identity/surface mismatch；
- ray-inconsistent geometry。

### 评测

不先训练复杂网络，只回答：

1. V6.7 inward-ray repair 能消除哪些伪影？
2. clean-hazard 是否被误判为 artifact？
3. 修复是否改变 Actor ID、trajectory、size、TTC？
4. artifact-hazard 是否能够修伪影而保留 hazard？

### 主要指标

- artifact recall / precision；
- clean-hazard false-artifact rate；
- Actor identity retention；
- trajectory/TTC shift；
- free-space violation；
- temporal flicker；
- primitive/surface retention。

### 停止条件

如果现有 inward-ray 只能改善内部 conflict proxy，却无法改善任何三维或时序物理指标，则 C1 不允许直接包装成 geometry repair，必须进入 P2 canonical surface。

---

## V7-P2：Actor canonical physical surface

### P2 formal freeze（2026-09-02）

- 分支从 `research/worldsim-v6.7-anisotropic-surface@d97c3f2` 直接拉出 V7。
- Formal task=`WS-V7-P2-AV2-FOUR-ACTION-COMPILE-01`；冻结 30 AV2 logs 一次完成 quantitative 与 qualitative
  confirmation，不在两阶段之间改 candidate。
- build/query/target 按固定 frame index 分离；所有 action 仅读取 build canonical 与 query evidence，target-only 几何
  只用于最终 before/after 评价。
- `KEEP/PROJECT/COMPLETE/UNKNOWN` 均产生真实坐标 surface；completion 必须有 build-side `>=3` temporal 与 `>=2`
  view support。13-gate AND rule 在 formal read 前提交。
- r1 仅在首 log metadata 后因漏配复用的冻结 hazard 段退出，未进入 point/surface/action/metric；r2 只恢复 P1
  同一 hazard 段并保持上述合同不变，不把工程入口失败计作科学 trial。

### P2 执行结果（2026-09-02）

Canonical=`run://worldsim_v7/WS-V7-P2-AV2-FOUR-ACTION-COMPILE-01/20260902T125000Z__four-action-s0-r2`；
30/30 logs，quantitative/qualitative 各 13/13 gates。Quantitative 433 Actors（147 hazard），真实 target-only
recall=`.5336→.6260`、precision=`.7959→.9767`、Chamfer=`.2544→.1681m`；Actor/hazard retention=`1/1`。
paired action rates 只作 deterministic contract 证据。P2 参数族关闭，不回扫，进入 P3 硬证据与预冻结视觉结果。

### P2-A：多帧 Actor 对齐

将每个 Actor 的多帧 LiDAR / point / primitive 转换到 canonical Actor frame：

\[
x_{i,t}^{actor}
=
T_{actor,t}^{-1}
T_{ego,t}
T_{sensor}
 x_t^{sensor}
\]

构造：

- point/surfel surface；
- FREE/OCC ray evidence；
- temporal support count；
- local normal；
- Actor box/shell prior；
- provenance。

### P2-B：物理表面表示

第一版优先比较：

1. motion-compensated surfel fusion；
2. truncated signed distance field（TSDF）或 signed distance field（SDF）。

不同时上大型神经隐式场。

### P2-C：四动作编译

- `KEEP`：已有稳定表面支持；
- `PROJECT`：靠近可信零水平集但违反局部射线一致性；
- `COMPLETE`：多帧、多方向支持下的局部孔洞补全；
- `UNKNOWN`：无足够证据。

### P2-D：外观—物理解耦

外观高斯可以通过：

- center-to-surface distance；
- covariance normal alignment；
- depth consistency；
- visibility consistency；

附着到物理表面。

但下游碰撞和占据查询只读取 `S_phys`。

### 学习模块解锁条件

只有确定性表面在以下方面出现系统性缺口时，才允许学习 residual completion：

- 远侧不可见面；
- 小型 Actor；
- 长遮挡；
- 稀疏 LiDAR；
- 局部多帧支持不足。

第一学习候选必须是低容量 Actor-local residual SDF；最多一个主方案和一个结构 fallback。

---

## V7 reviewer-facing 补强主线（2026-09-02）

1. **视觉与几何硬证据**：V7 必须从 feature filtering 推进到 ray/depth/surface/collision 的真实三维自洽；失败时优先
   修物理 provenance 或表示，而不是增加后处理规则。
2. **外部泛化降维打击**：模型训练/选择限定 nuScenes，AV2 为冻结 zero-shot 主外域；资源允许时再加入 Waymo 第二外域，
   但不以扩大数据面替代 AV2 深证据。
3. **理论边界与工程极致**：将 sensor opportunity、`UNKNOWN`、immutable Actor/hazard state、可修复/不可修复条件和
   runtime/resource 明确暴露，提供可解释边界；禁止包装成 road-safety formal guarantee。

后续 auto-research 聚焦这三条，不横向堆叠与论文主张无关的模块。

---

## V7-P3：三维硬证据与视觉结果

### P3-A formal freeze（2026-09-02）

- Formal task=`WS-V7-P3-AV2-HARD-EVIDENCE-01`，冻结 30 logs 与 P2 canonical config；新 overlay 只定义硬指标、
  8-gate non-regression rule 和 renderer，不复制 Actor/hazard 参数。
- target-ray depth/termination 使用各 target LiDAR point 原始 Actor-local sensor origin，target 仍不参与 action；
  temporal jitter 是 build-frame-to-canonical residual 的跨帧标准差。paired free-space/ghost component 明确为合成合同。
- qualitative 10 logs 不删；每 log 词法序前三 eligible Actor。main 固定前 8 logs 首 Actor，supplement 固定全部 30；
  首阶段产出 point/surface/ray panel，P3-B 再接 frozen camera RGB/depth/video，不按结果换案例。
- r1 30/30 完成但两 role 均 6/8；任意 compiled-surface Euclidean residual 破坏逐 primitive ray provenance，
  free-space/ghost 两门无效且总 verdict rejected（`V7-F07`）。r2 仅迁移 NeuRAD/SplatAD/LiDAR-RT 的最小
  `ray_o/ray_d + hit + aligned output` 边界；不改案例、动作、阈值、gates 或其余六项指标。
- r2 的正确 ray metric 仍为 6/8：nearest-canonical 输出沿 matched beam 平均早于 hit 约 9cm，确认 PROJECT
  operator 的真实物理失败（`V7-F08`）。r3 冻结为 ray-certified PROJECT：有 direct observed hit provenance 才投回
  hit，无 matched hit 则 UNKNOWN；不调门/案例/target isolation，不把 zero-level 最近点冒充 line-of-sight 证据。

### P3-A 执行结果（2026-09-02）

Canonical=`run://worldsim_v7/WS-V7-P3-AV2-HARD-EVIDENCE-01/20260902T143000Z__ray-certified-s0-r3`；
两 role 各 8/8、视觉 3/3。Quantitative free-space=`1→0`、ghost components=`10983→0`、target depth=
`.2067→.1541m`、ray termination=`.5694→.6438`、zero-level=`.1400→.0583m`、Chamfer=`.2544→.1753m`，
Actor/hazard state shift=0。相较 rejected nearest-surface r2 的 `.1681m` Chamfer，r3 明确选择约 7.2mm 的拟合代价
换 matched-ray free-space 安全边界。P3-A 关闭，进入 P3-B RGB/camera-depth/video，不扫 tolerance。

### P3-B formal freeze（2026-09-02）

- Formal task=`WS-V7-P3B-AV2-CAMERA-EVIDENCE-01`；逐项复用 P3-A 8 main + 30 supplement cases，case identity 与
  main/supplement 身份不可变。compiler 只新增 query timestamp 与 Actor ego rigid transform diagnostics，不改 action/metric。
- 相机候选固定为 AV2 官方 7 ring cameras。选择只计算 query Actor LiDAR points 经官方 motion-compensated projection
  后的 in-frame count；取最大者，平局按 config 顺序，完成选择后才 decode RGB。crop 只由 query projection 决定。
- 每 case 固定输出 RGB+observed returns、paired artifact overlay、four-action output、sparse camera-depth 四联图与动态
  before/after MP4。全部 30 cases 写出，不按可见结果换图，也不扫 camera/crop/palette。
- paired ghost/duplicate/flicker 与视频闪烁是 synthetic contract evidence；RGB panel 是 calibrated evidence overlay，
  不是 photorealistic reconstruction。真实几何结论继续由 P3-A target-only LiDAR depth/ray/surface 承担。

### P3-B 执行结果（2026-09-02）

Canonical=`run://worldsim_v7/WS-V7-P3B-AV2-CAMERA-EVIDENCE-01/20260902T150000Z__camera-evidence-s0-r1`；
10/10 logs 与 30/30 frozen cases 完成，8 main + 30 supplement panels、30 MP4。query visibility points min/median=
`14/82`，minimum visibility fraction=`.7278`，zero-visible=0；8 main 人工检查均可读且未换图。wall=`43.38s`、
GPU=`.0663GiB`、RSS=`1.339GiB`、run=`46MiB`。

不挑图同时暴露 `V7-F09`：全 634 Actors 的 aggregate Chamfer 改善下仍有 `109/634=17.19%` individual worsening，
depth/ray 方向变差分别为 `271/253` Actors。故 P3 只支持 aggregate zero-shot geometry，不支持 per-Actor universal
repair certificate。P4 必须把 validity head 改成 nuScenes-only selective repairability/risk--coverage，并保持 hazard 输入
独立；AV2 未知 shift 下只做 zero-shot evaluation，不声称 conformal exchangeability guarantee。

### P3-C visibility-conditioned ray certificate freeze（2026-09-02）

- 为避免 Chamfer 把不可见背面补全与真实自由空间冲突混为一谈，冻结双向 observed-ray partition：target ray=
  `early/hit/late/unmatched`；surface primitive=`contradicted/supported/occluded/UNKNOWN`。UNKNOWN 绝不并入 free/occupied。
- 直接复用 P3 的 `.20m` ray lateral/depth tolerance 与 ray-certified observed-hit PROJECT，不扫 tolerance、metric weight、
  operator 或 Actor subset。target仍只在 action完成后进入 evaluator。
- 逐 Actor 报告相对 clean query 的新增 early count、新增 visible contradiction count、exact-zero contradiction、target-hit 与
  visibility-F-score non-inferiority；`V7-F09` 的 Chamfer-worsened stratum必须完整单列，不能用 aggregate mean覆盖。
- consumed 30 logs只作 descriptive implementation audit；fresh 20 recovery logs在任何新 visibility metric read 前已由
  metadata冻结，下载完成后 exact-once confirmation。不得用 fresh 结果训练/校准/选 threshold/改 operator/换 scene。
- 物理解释借鉴 NeuRAD 的显式 LiDAR ray/sensor model；DGLSS 的 source-only sparsity consistency与 3DLabelProp 的时序
  几何只限定跨传感器边界。证书只覆盖 observed ray set，不证明不可见背面、collision/planning/closed-loop/road safety。

### P3-C consumed-cohort result / per-Actor boundary（2026-09-02）

Canonical=`run://worldsim_v7/WS-V7-P3C-AV2-VISIBILITY-CERTIFICATE-DEV-01/20260902T223000Z__visibility-audit-s0-r1`。
30 logs/634 Actors 上，query→compiled pooled target-hit recall=`.49691→.69831`、early termination=`.03395→.02868`、
visible precision=`.99606→.99765`、visibility F-score=`.66304→.82156`、surface contradiction=`856→415`。这把 C1 的
aggregate geometry improvement从 Chamfer推进到可证伪 observed-ray physics。

逐 Actor only `406/634` nonnew visible violation、`10/634` exact-zero；214 Actors early count增加，34 Actors surface
contradiction增加。query-relative Chamfer-worse 105 Actors中只有24个 F-score non-inferior。故 `V7-F18` 明确关闭
per-Actor universal certificate；fresh 20-log合同不变。下一步只允许一次全 Actor output-provenance attribution，区分
nearest-ray matching/KEEP coverage与 PROJECT/COMPLETE 实体新增；不扫描容差、不按 failure挑 Actor。只有 source-only
provenance明确指向 completion free-space conflict时，才允许冻结 source-ray space-carving operator。

### P3-D all-Actor visible-failure attribution freeze（2026-09-02）

- consumed 30 logs 的全部 Actors/target rays参与，不按 `V7-F18` 大小、hazard、category或 Chamfer选择案例。
- compiled/new early return、compiled/new hit、surface contradiction逐项归因到 `KEEP/PROJECT/COMPLETE`；point provenance
  精确复现 compiler concat顺序和 `.06m` voxel-first output。
- `.20m` lateral/depth tolerance、observed-hit PROJECT、Actor policy和 fresh 20-log合同保持不变；这是 post-result
  mechanism diagnostic，不是 independent confirmation或 recovery gate。
- 只有 COMPLETE 同时主导 new early与surface contradiction，才允许冻结 target-independent source-ray carving；若
  KEEP/PROJECT主导，则将 failure定位为 coverage下的 nearest-ray assignment边界，不通过删除表面优化指标。

### P3-D result / source-ray carving closed（2026-09-02）

Canonical=`run://worldsim_v7/WS-V7-P3D-AV2-VISIBLE-FAILURE-ATTRIBUTION-01/20260902T224500Z__visible-failure-attribution-s0-r1`。
COMPLETE占 new early `24,445/25,698=95.12%`，同时贡献 `365,634` new hits（`14.96×` hit/early）；全局 early
`60,779→51,340`，净减少 `9,439`。surface contradictions中 KEEP/COMPLETE=`359/56`，COMPLETE只占`13.49%`；
observed-hit PROJECT与 clean query在`.06m` voxel内重合，因此按原 concat-first语义无独立 output point。

预冻结的 carving条件要求 COMPLETE 同时主导 new early与surface contradiction，当前不成立。关闭 source-ray carving、
completion删除与任何 tolerance/operator scan。C1 结论收敛为 aggregate observed-ray physics improvement加明确
completion gain--tail trade-off；`V7-F18` per-Actor边界保留，逐 Actor authority交给 P4 select/abstain。下一步只做
paper integration与已冻结 fresh 20-log confirmation。

### 必须补齐的几何指标

- free-space violation rate；
- ray termination consistency；
- surface precision / recall；
- Chamfer distance 或 point-to-surface error；
- SDF consistency；
- LiDAR depth consistency；
- temporal surface jitter；
- ghost connected components；
- Actor surface completeness；
- collision-shell consistency。

### 必须补齐的危险保真指标

- Actor retention；
- ID/lifecycle retention；
- trajectory displacement；
- speed/acceleration shift；
- TTC shift；
- cut-in / hard-brake / near-miss label retention；
- hazard event count change。

### 视觉证据

至少制作：

- 6–8 个主文 qualitative cases；
- 20+ supplement cases；
- RGB / depth / point / surface / ray evidence 并列图；
- 动态视频：修复前后 ghost、flicker、危险 Actor 保留。

禁止按最终质量挑图。Qualitative scene 在看方法结果前按 metadata 和预注册 failure strata 冻结。

---

## V7-P4：有效性—危险性条件解耦

### P4 执行冻结（2026-09-02）

- Formal task=`WS-V7-P4-NUSCENES-SELECTIVE-FACTORIZE-01`；nuScenes `11/14/38` scenes 分别用于
  train/calibration/test，角色完全 scene-disjoint；30 个 AV2 frozen logs 仅作外域 zero-shot。
- P3 `before` 带 synthetic corruption，P4 不再以此定义“修复成功”。逐 Actor label 固定为
  `Chamfer(compiled,target) <= Chamfer(clean query,target)`；abstain 返回 clean query，target 不进入 action/features。
- validity 仅用 13 个 runtime surface/ray/provenance/support 特征，hazard 仅用 TTC/clearance/closing/brake/crossing；
  比较唯一 shared-input two-head baseline 与 structurally factorized two-head candidate。
- 单 seed、固定 32 hidden/80 epochs，无 architecture/seed/threshold sweep。标准化=train only；false-repair threshold
  只在 nuScenes calibration 以 `.05` monotone adjusted risk 冻结，随后 AV2 不调任何参数。
- formal guarantee 只限 exchangeable nuScenes calibration 边界；未知 nuScenes→AV2 shift 只报告 risk--coverage/geometry，
  不声称 conformal、collision、planning、closed-loop 或 road-safety guarantee。Waymo 在本地无冻结 sensor corpus 时暂缓。

### P4 执行结果（2026-09-02）

Canonical=`run://worldsim_v7/WS-V7-P4-NUSCENES-SELECTIVE-FACTORIZE-01/20260902T161000Z__selective-factor-s70401-r2`；
7/7 gates，nuScenes Actors=`29/56/228`，AV2=30 logs/634 Actors。factorized nuScenes test repair/hazard AUROC=
`.6491/.9811`，shared=`.6426/.9166`；factorized paired swap shift=`0/0`，shared=`.2014/.2886`。

相同 nuScenes-only threshold 在 AV2 覆盖 `75.55%`，false repair `16.56%→8.99%`（always→selective）；
clean-query/always/selective Chamfer=`.2577/.1770/.1821m`，hazard coverage=`92.17%`。结果支持 empirical
nuScenes→AV2 selective transfer 与结构解耦，但 calibration→AV2 coverage=`8.93%→75.55%` 明确否定跨域 formal
exchangeability guarantee；selective 也不在平均 Chamfer 上支配 always repair。train 仅 29 Actors，冻结为限制。

### Baselines

- shared encoder；
- two-head shared trunk；
- independent encoders；
- paired-invariance factorized encoders（V7）。

### 输入

Validity encoder：

```text
ray support
FREE/OCC contradiction
surface residual
multi-frame consistency
source provenance
visibility
local geometry
```

Hazard encoder：

```text
Actor trajectory
relative pose/velocity
TTC
route relation
map context
interaction state
```

### 损失

\[
\mathcal L
=
\mathcal L_{artifact}
+
\mathcal L_{hazard}
+
\lambda_{pair}\mathcal L_{pair}
+
\lambda_{leak}\mathcal L_{leak}
\]

其中重点是 paired invariance，而非追求全局零相关。

### 评测

- artifact AUROC/AUPRC；
- hazard AUROC/AUPRC；
- clean-hazard false artifact；
- artifact-hazard hazard retention；
- hazard-from-validity probe；
- artifact-from-hazard probe；
- paired latent distance；
- paired prediction consistency。

### 成功标准

V7 模型必须相对 shared encoder：

- artifact 与 hazard 主任务均不退化；
- clean-hazard 误删率明显降低；
- 配对不变性改善；
- cross-probe leakage 下降；
- 不能通过 all-UNKNOWN 或降低危险样本分母获得改善。

---

## V7-P5：C3 论文整合与轻量重训

### P5-A alignment audit freeze（2026-09-02）

P4 `scene_name/instance_token` 与 V6.7 `scene_index/numeric_actor_id` 只允许通过 nuScenes official scene ordering 和
DriveStudio `instances_info[id]` exact join。Direct joint fit 需 P4 train 对齐 `>=3 scenes/20 Actors`，否则无法建立非平凡
scene-heldout split，禁止用 calibration/test Actors 回流补样本；仅执行 frozen descriptive interface audit。

Alignment canonical=`run://worldsim_v7/WS-V7-P5-PHYSICAL-RELIABILITY-ALIGNMENT-AUDIT-01/20260902T180000Z__physical-reliability-alignment-s0-r1`。
P4 train/cal/test 对齐=`2/5/14 scenes`、`5/25/88 Actors`、`1,320/7,645/39,285 rows`；train 未达
`3/20`，因此 `V7-F13` 拒绝 direct joint fit。P5 后续只允许冻结 P4/P346 的 descriptive multi-horizon interface audit。

P5-B 固定使用 calibration/test `25/88` exact-match Actors，比较 frozen P4 selected/abstained、geometric
helpful/harmful、selected-and-harmful 在 retained V6.7 `.8/1.5/2.5/3.0s` cost/state-error/decision-flip/
false-safe 的分层。无 fit、threshold、gate 或 P346 execution；只给共现安全边界，不做因果改善声明。

P5-B canonical=`run://worldsim_v7/WS-V7-P5B-FROZEN-PHYSICAL-RELIABILITY-INTERFACE-01/20260902T183000Z__frozen-physical-reliability-interface-s0-r1`。
test `88 Actors/39,285 rows` 中 selected=`5/1,781`，false-safe/flip=`0/0`，23 个 geometric-harm Actors 全部
abstained；但 selected q90 state error=`1.577m` vs abstained `.693m`，3.0s=`4.246m`。因此 paper graph 固定为
physical repair gate 与 downstream multi-horizon reliability authority 分离；零事件小样本不升级为 safety guarantee。

CVPR draft 已把该边界写入 composed-authority equation、P5 table、abstract/limitations/conclusion；official template full
compile=`6 pages`，pages 2--6 visual check 通过。后续已补入冻结P182/P183/P199/P201 direct C3 evidence table，同时保持
P346 held-out-H limitation；未改现有P5/P7 claim或执行joint fit。

### 保留模块

- V6.7 已支持的 Actor residual distribution；
- analytic trajectory boundary query；
- conditional continuous cost density；
- input-conditioned multi-horizon dependence；
- monotone runtime reliability surface。

### 关闭模块

不复活：

- fixed Student-t 重尾分布；
- mixture component sweep；
- CRPS/Energy-loss family；
- DCT temporal family；
- end-to-end occupancy-flip classifier；
- learned selective authority head；
- P333 后继续扩展的条件校准树。

### V7 新增重点

1. 让 C3 输入显式使用 V7 validity representation；
2. 做 `with/without physical repair` 对照；
3. 检查 C1 是否改善 density 的 NLL/Brier/calibration；
4. 检查危险 Actor 保留是否维持或增加困难样本覆盖；
5. 简化 checkpoint 依赖为一套 paper-facing end-to-end inference graph。

---

## V7-P6：nuScenes → Argoverse 2 零样本迁移

### P6-B sensor-opportunity recovery freeze（2026-09-02）

P7 的 `V7-F11` 将失败层级定位为 `sensor opportunity shift`。唯一恢复候选删除 raw observation-frame count，并将
canonical surfels、temporal/view support 变为 per-observation dimensionless ratios；只用 nuScenes train/calibration 训练与
定阈值，hazard head 和 structural factorization 不变。模型固定为 hidden32/seed70601/80 epochs，无 feature/model sweep。

External recovery cohort 不复用已消费 30 logs：对 150 个 AV2 val UUID 排序，排除 v1 indices `0,5,...,145`，再从
120-log complement 每隔 6 个取一个，共 20 fresh logs。元数据选择在任何 recovery score/quality 前冻结。通过条件只包括
nuScenes non-inferiority、机会变换不变性、fresh coverage/false-repair/Chamfer，以及相对 frozen P4 的 fresh score-shift
改善；不得在 fresh read 后调 threshold。通过也不产生 formal cross-domain risk guarantee。

### P6-B ratio result / P6-C sparsity-consistency direction（2026-09-02）

Ratio-only canonical=`run://worldsim_v7/WS-V7-P6-OPPORTUNITY-INVARIANT-SELECTOR-01/20260902T170000Z__opportunity-invariant-s70601-r1`。
fixed opportunity shift=`0`，但 nuScenes-test repair AUROC=`.60728`，低于 P4 `.64908` 与 floor `.62908`；因此
`V7-F12` 在 external read 前关闭 ratio-normalization family，fresh AV2 quality 仍未读。

顶会迁移方向改为 CVPR 2023 DGLSS 的 source-only sparsity augmentation + invariant consistency：保留 raw evidence amount，
训练时构造固定下采样机会视图并约束 repair score 一致。P6-C 必须在读取 fresh AV2 前独立冻结 model/seed/loss/gates；
不得根据 ratio candidate 或未来 fresh 结果扫描 consistency weight。

P6-C 已冻结为原始 13D validity inputs + `.5x/.75x` joint opportunity subsampling，BCE(original+augmented)+weight-1
probability consistency，hidden32/seed70602/80 epochs。fresh 前两门固定为 nuScenes AUROC non-inferiority `.02` 与
intervention-score shift 相对 P4 `<=.70x`；通过后外域仍沿用 coverage/false-repair/Chamfer/Wasserstein 四门，不调阈值。

P6-C fit canonical=`run://worldsim_v7/WS-V7-P6C-SPARSITY-CONSISTENT-SELECTOR-01/20260902T173000Z__sparsity-consistent-s70602-r1`；
nuScenes AUROC=`.63239>=.62908`，candidate/P4 intervention shift=`.019526/.183026`、ratio=`.10668`，2/2 通过。
model/standardizer/threshold 已冻结。fresh AV2 exact-once 523 Actors的 candidate/P4 AUROC=`.676168/.654837`、
coverage=`.839388/.772467`、false-repair=`.124283/.112811`、selective Chamfer=`.178291/.184133m`、score-shift=
`.203971/.207216`，external四门通过；但与P8-A fresh nuScenes rejection方向相反，按`V7-F19`不晋升，P4保持primary。

### 主域

- 训练、模型选择和校准：仅 nuScenes；
- 测试：Argoverse 2 Sensor Dataset；
- Waymo Open Dataset 为资源允许时的第二外测，不阻断 V7。

### AV2 adapter 允许做

- 坐标系转换；
- 单位统一；
- 类别映射；
- timestamp 对齐与 pose 插值；
- Actor-local frame；
- sensor opportunity normalization；
- SceneIR 字段适配。

### 禁止

- AV2 fine-tuning；
- AV2 post-hoc calibration；
- 看 AV2 quality 后改 threshold；
- 删除失败场景；
- 用 AV2 validation 选择模型；
- 因类别差异重新定义 target。

### 推荐规模

```text
AV2 main quantitative: 20 logs
AV2 qualitative:       10 logs
Total:                  30 logs
```

场景在结果读取前按 metadata 冻结。

### 零样本主表

报告：

- C1 physical metrics；
- C2 artifact/hazard disentanglement；
- C3 density、Brier、calibration、multi-horizon；
- runtime；
- relative degradation from nuScenes to AV2。

### 如果零样本失败

只允许判断失败层级：

```text
sensor opportunity shift
coordinate/interface mismatch
representation shift
density calibration shift
Actor dynamics shift
```

V7 主实验不允许通过 AV2 fine-tuning 救结果；可以在 supplement 增加少量-shot adaptation，但必须与零样本结果分开。

---

## V7-P7：理论边界与可解释性

### P7-A 执行冻结（2026-09-02）

- 先解释 P4 frozen selector：21-point risk--coverage/geometry/hazard envelope、固定 threshold 标记、跨域 score shift，
  以及 factorized validity head 的 Integrated Gradients；不重训、不重校准、不从 AV2 曲线重选 operating point。
- exact proposition：factorized repair/hazard heads 的输入计算图不连通，故 cross-input derivative=`0`；IG 仅解释各自
  合法输入内的模型敏感度，不主张 causality。
- P5 C3 integration 暂缓到 physical Actor 与 V6.7 trajectory-density row 建立充分、scene-disjoint 对齐之后；当前少量
  overlap 不足以把轻量重训解释成可靠性改善。

### P7-A 结果与边界（2026-09-02）

Canonical=`run://worldsim_v7/WS-V7-P7-INTERPRETABLE-SAFETY-ENVELOPE-01/20260902T163000Z__safety-envelope-s0-r1`。
在不训练/重校准/适配的条件下，P7 保留 P4 empirical zero-shot operating point 和 factorized exact-zero cross-input
derivative，但暴露 validity head 的 sensor-opportunity shortcut：calibration→AV2 score Wasserstein=`.2170`、KS=`.7041`，
AV2 median score=`.999992`；`observation_frame_count` 的 IG attribution 从 nuScenes test `18.25%` 增到 AV2 `49.53%`。

因此 `V7-F11` 把主张边界收窄为“nuScenes-trained empirical AV2 repair-or-abstain transfer”，不得称 domain-invariant
selection 或外域 formal risk guarantee。恢复只允许在 nuScenes development 中把 raw observation opportunity 替换为
dimensionless density/support 或施加 sparsity invariance；已消费 30 AV2 logs 不允许参与后验 feature/threshold 选择。
任何恢复后的 external claim 必须使用 metadata-frozen 的新 AV2 cohort 或 Waymo。

### 命题 1：危险保真

若 physical compiler 不改变 Actor identity、trajectory 和尺寸：

\[
\mathcal R(A_i)
=
(ID_i,X_{i,0:H},D_i,\widetilde S_i)
\]

则任何只依赖 \(ID_i,X_{i,0:H},D_i,\tau\) 的危险函数保持：

\[
h(\mathcal R(A_i),\tau)=h(A_i,\tau)
\]

### 命题 2：确定性编译幂等性

\[
\mathcal R(\mathcal R(W))=\mathcal R(W)
\]

允许固定数值容差。

### 命题 3：可靠性单调性

\[
b_1\le b_2
\Rightarrow
R(\tau,H,b_1)\le R(\tau,H,b_2)
\]

\[
H_1\le H_2
\Rightarrow
R(\tau,H_1,b)\ge R(\tau,H_2,b)
\]

### 命题 4：几何误差与任务代价边界

若任务代价关于 surface 和 residual distribution 为 Lipschitz：

\[
|\mathbb E_{\hat P}C-\mathbb E_{P^*}C|
\le
L_eW_1(\hat P,P^*)
+
L_sd(\hat S,S^*)
\]

理论部分只解释接口和误差传播，不声称现实道路 safety guarantee。

### 可解释性图

- ray evidence → surface action；
- validity latent vs hazard latent；
- trajectory normal projection；
- conditional cost density；
- reliability surface；
-跨数据集失效分解。

### P7-B geometry-to-cost sensitivity freeze（2026-09-02）

对 `C(d)=max_j |n_j^Te_j|/max(d_j,epsilon)` 使用 max 1-Lipschitz 与 clipped reciprocal 推导
`|Delta C| <= max_j a_j|delta|/(m_jm'_j) <= a_max|delta|/epsilon^2`。固定在 575,596 retained P109 rows 与
P5 test strata 上评估 `±.05/±.10/±.20m`、`epsilon=.05m` 的 shift/bound/tightness/clearance crossing；无训练、
校准、阈值或新传感器读取。只解释误差传播，不能称扰动概率或 safety certificate。

P7-B r1 的 FP32 maximum `shift-bound=3.8147e-6` 超过 frozen `1e-6`，登记 `V7-F14`；P5 strata 为 0 violation。
只把 algebra compute 改 FP64，deltas/floor/tolerance/groups/rows 均不变后 r2，不放宽判据。

P7-B r2 canonical=`run://worldsim_v7/WS-V7-P7B-GEOMETRY-COST-SENSITIVITY-01/20260902T191500Z__geometry-cost-sensitivity-s0-r2`：
575,596 rows×6 shifts 为 0 violation，max overage=`1.42e-14`。`-.20m` full-source mean/q99 cost shift=
`.02922/.14501`，仅 `.8409%` sign crossing；P5 selected mean shift=`.000847` vs abstained `.004560` 且 0 crossing。
结合 P5-B selected motion error 更大，结论固定为 geometry sensitivity、repairability、motion uncertainty 三轴分离。

CVPR draft 已同步 P7-B theorem/result/limitation，official template=`7 pages`；page 7 仅 references continuation，无 float/
clipping。P6-C fresh external 前不根据该解析 audit 改 selector 或 threshold。

### P7-C explainable interval certificate freeze（2026-09-02）

为补齐“可解释性与安全边界”，固定对 P4 factorized validity MLP 做一次 descriptive FP64 interval-bound audit。模型、
standardizer、threshold不变；在 nuScenes-train standardized feature space 固定 `.05/.10/.20` stress boxes，`.10` 为逐 Actor
解释半径。输出 robust-select/unresolved/robust-abstain，并分解 `sensor_opportunity` 与 `physical_surface` 两组以及 top-3
单特征区间宽度。Target 不参与证书 state，只在结果后描述错误且稳定的决策。

本项不读 fresh AV2、不训练/refit/recalibrate、不扫描 radius/group，也不增加 scientific pass/fail gate。证书只回答 frozen
network threshold decision 是否对指定 feature box 稳定，不等于几何正确、跨域 exchangeability、collision/planning/
closed-loop 或道路安全保证。

P7-C canonical=`run://worldsim_v7/WS-V7-P7C-VALIDITY-INTERVAL-CERTIFICATE-01/20260902T220000Z__validity-interval-s0-r1`。
`.10` all-feature box 下，nuScenes test nominal/robust-select=`9/3`，AV2=`479/437`；但 AV2 robust-select 仍含
`47/437=10.76%` target false repair，登记 `V7-F17`。Sensor-opportunity group 的 mean logit width 在 test/AV2=
`1.5496/1.2850`，高于 physical-surface `1.1329/.9836`；`log_query_points` 为 test `190/228`、AV2 `603/634`
Actors 的 top-1 sensitivity。该结果只支持 deterministic network-decision interval与可解释 failure localization，明确否定
“stable decision即correct/safe repair”。

CVPR main 已同步 P7-C actor-level interval、stable-error boundary 与 LiDAR visibility/robust-OOD references。Official-template=
`8 pages/1,164,284 bytes`；pages 5--8 visually valid，page 8 仅 references continuation，无新增 layout/scientific failure。

---

## V7-P8：Final exact-once evaluation

### P8-A fresh nuScenes freeze（2026-09-02）

- 排除 P4 train/calibration/test 后剩余 106 个本地可用 trainval scenes；只按 official scene index 与文件可用性排序，
  用 `round(linspace(0,105,20))` 冻结 20 scenes，冻结前 quality unread。
- frozen candidate=P6-C source sparsity-consistent selector；baseline/hazard=P4 factorized model；standardizer、threshold、
  compiler 与 Actor policy 均不变。
- one formal read；同 rows 报告 repair/hazard、coverage、false-repair、selective Chamfer 与 score shift；不换 scene、
  不 refit/recalibrate/调阈值。
- core gates：candidate repair AUROC 不低于 P4 `-.02`、coverage `>=.10`、false-repair 低于 always-repair failure、
  selective Chamfer 不差于 clean query。失败即保留 negative result。
- 该冻结时点AV2 fresh 20-log由P6-C frozen external runner等待单实例下载；后续已20/20 exact-once完成，且P8-A结果
  未改变AV2 candidate/protocol。

### P8-A exact-once result（2026-09-02）

Canonical=`run://worldsim_v7/WS-V7-P8A-FRESH-NUSCENES-EXACT-ONCE-01/20260902T200000Z__fresh-nuscenes-final-s0-r1`。
20/20 scenes、123 Actors、one read、0 replacement/update。P6-C/P4 repair AUROC=`.747253/.782280`，退化
`.035027 > .02`，因此 3/4 gates、verdict rejected（`V7-F15`）。P6-C coverage=`.26829`、false-repair=`.02439`、
selective/query Chamfer=`.196891/.234408m` 的 operating-point support 保留，但 descriptive AURC=`.126429` vs P4
`.105633` 也确认 global ranking 较弱。P6-C 不晋升；P4 保持 paper primary selector。禁止 recovery fit/threshold/scene
replacement。随后独立完成的 fresh AV2 read 得到 P6-C/P4 AUROC=`.676168/.654837`，但false-repair=
`.124283/.112811`；external gate pass与fresh nuScenes rejection共同形成`V7-F19` cross-fresh reversal，不推翻P8-A或
P4 primary decision。

CVPR main 已同步 P8-A two-row table、negative recovery、AURC boundary 与 P4 retention；official-template=`7 pages`，
pages 4/6/7 visually valid。该 paper update 不新增 metric gate/model/cohort 或第二次 test read。

所有 architecture、threshold、surface operation、density model、calibration 和 adapter 冻结后，建立：

```text
nuScenes final test: >=20 fresh scenes
AV2 zero-shot test: >=20 frozen logs
```

要求：

- distinct logs/session；
- metadata-only selection；
- quality unread；
- exact-once；
- 不换场景救结果；
- resource failure 与 scientific failure 分离。

Final test 前不再设计新模型。

---

## V7-P9：最小下游应用

### P9 composed-authority freeze（2026-09-02）

为避免把物理 repair 冒充 planner benefit，四臂改写为可识别的 `2×2` factorial：

```text
B0 query surface + no task authority
B1 HARP-3D surface + no task authority
B2 query surface + frozen P346 task authority
B3 HARP-3D surface + frozen P346 task authority
```

固定 P5 test exact identities 的 retained P109 rows、6-query/4-horizon lattice、P346 artifact heldout 2.5s、authority set size 2、
middle frozen cost ceiling、requested reliability .90 与 heldout task conditions。无训练/calibration/threshold/budget sweep、
无新 sensor read、无 critic/RL/closed-loop。核心只检验 physical CD、authorized visited-cost/risk、coverage 与 denominator/
Actor/hazard retention；physical branches 的 action result 相同是预期 non-interference，不是无效实验。

Canonical P9=`run://worldsim_v7/WS-V7-P9-COMPOSED-AUTHORITY-FIXED-LATTICE-01/20260902T213000Z__composed-authority-s0-r1`。
1,228 sets 中 authority coverage=`.49430`；mean cost `.149711→.027887`，unsafe `.164495→.011532`；HARP-3D
surface CD `.229314→.223253m`，Actor/hazard retention=`1/1`，5/5 gates。B0/B1 与 B2/B3 action metrics 分别完全相同，
证明分权 non-interference。旧 freeze 文字将 artifact heldout 误标为 3.0s，`V7-F16` 更正为实际 2.5s；executable
contract 与 canonical summary 一直正确，故不改代码、不重跑。

CVPR 主稿已同步 P9 2×2 定义、Table 5、retained-source utility 与 open-loop/causal safety 边界。Official-template=
`8 pages/1,162,514 bytes`；pages 6/7/8 visually valid，page 8 仅 references continuation，无新增 layout failure。

RL 不再是主线，也不阻断论文。

下游只回答：

> V7 编译后的物理世界和 reliability surface，是否能帮助固定候选轨迹选择器避开不可靠访问状态，同时保留危险但合法 Actor？

### 四臂

```text
B0 Naive reconstruction
B1 q0-filtered world
B2 V6.7 inward-ray world
B3 V7 HARP-3D + task reliability
```

### 固定条件

- 同一 Actor IDs；
- 同一 Actor trajectories；
- 同一 hazard distribution；
- 同一 action lattice；
- 同一 planner / scoring budget；
- 同一 case denominator。

### 指标

- actual visited-state cost；
- collision/TTC proxy；
- progress；
- stuck；
- hazard Actor retention；
- abstention；
- artifact exploitation proxy。

### 篇幅

主文仅 0.4–0.6 页和一张小表。

如果可低成本完成小型 RL demo，可放 supplement；不为 RL 新开大规模模型搜索。

---

# 6. 论文交付计划

## 6.1 V7 必须持续产出 CVPR/ICCV 格式论文初稿

从 P0 开始保持 `paper/main.tex` 可编译。

每个里程碑完成后：

1. 将数值写入 `results/results_macros.tex`；
2. 更新对应主表；
3. 更新一张图或 qualitative panel；
4. 更新 Method/Experiments 对应段落；
5. 编译 anonymous PDF；
6. 不要求每次润色全文。

## 6.2 论文阶段性交付

### Draft 0：结构骨架

- 完整章节；
- 标题、摘要、贡献；
- Figure/Table 占位；
- V7 数值允许 TBD。

### Draft 1：C1 完整

- C1 method；
- 物理指标；
- qualitative；
- Figure 1/3；
- Table 1。

### Draft 2：C2+C3 完整

- paired disentanglement；
- density / joint-H；
- Table 2/3；
- zero-shot 初步结果。

### Draft 3：实验完整

- final exact-once；
- AV2；
- minimal downstream；
- limitations；
- supplement failure ledger。

### arXiv v1

- 非匿名；
- 完整 8 页风格主文；
- supplement；
- 视频/项目页可后补。

### Conference submission

- 匿名；
- 使用届时正式 CVPR/ICCV author kit；
- 保持与 arXiv 的方法和主结果一致；
- 不在审稿期继续通过新 test 调主方法。

---

# 7. 主图与主表

## Figure 1：Teaser

四象限并列：

```text
valid-safe
valid-hazard
artifact-safe
artifact-hazard
```

展示：

- RGB；
- 3D surface；
- ray evidence；
- 修复动作；
- 修复后结果；
- 危险 Actor 保留。

## Figure 2：总架构

```text
Physical Compiler
→ Validity/Hazard Factorization
→ Actor Reliability
→ Analytic Task Query
→ Cost Density
→ Reliability Surface
```

## Figure 3：物理几何机制

展示：

- Actor canonical frame；
- motion compensation；
- FREE ray；
- hit endpoint；
- inward support；
- zero-level surface；
- KEEP / PROJECT / COMPLETE / UNKNOWN。

## Figure 4：特征解耦

同一 Actor 的两组配对：

- safe → cut-in；
- clean → ghost。

展示：

- artifact score；
- hazard score；
- latent distance；
- cross-probe。

## Figure 5：连续密度和跨数据集

展示：

- conditional cost density；
- CDF；
- joint-H reliability；
- nuScenes→AV2 zero-shot degradation。

## Table 1：物理与危险保真

- free-space violation；
- surface error；
- depth consistency；
- temporal jitter；
- ghost count；
- Actor retention；
- TTC / trajectory shift。

## Table 2：Validity–Hazard 解耦

- artifact AUROC/AUPRC；
- hazard AUROC/AUPRC；
- paired invariance；
- clean-hazard false artifact；
- cross-probe leakage。

## Table 3：连续可靠性与零样本

- Spearman；
- NLL/Brier；
- calibration；
- joint-H Brier；
- runtime；
- nuScenes；
- AV2 zero-shot。

## Table 4：最小下游

- actual visited-state cost；
- collision/TTC proxy；
- progress；
- hazard retention；
- abstention。

---

# 8. CVPR/ICCV 主文篇幅分配

8 页正文建议：

| 模块 | 比例 |
|---|---:|
| C1 物理几何编译 | 32% |
| C2 有效性—危险性解耦 | 18% |
| C3 连续代价密度与多时域可靠性 | 20% |
| 跨数据集实验 | 22% |
| 下游、限制、结论 | 8% |

\[
\boxed{C1+C2+C3\approx70\%}
\]

---

# 9. 研究纪律与资源合同

## 9.1 简化原则

每个阶段最多：

- 一个主方法；
- 一个结构性 fallback；
- 2–4 个真正决定结论的指标。

## 9.2 禁止

```text
hash / checksum / fingerprint
repo-wide regression matrix
大量 smoke tests
seed sweep
threshold sweep
mixture-count sweep
hidden-size sweep
为了过门更换 final cohort
复活已关闭的方法家族
用 hazard score 删除 Actor
把 opacity 当 occupancy
无限追加 P 编号
```

## 9.3 允许直接修复

- 路径和运行入口；
- archive locator；
- scene-ready pipeline；
- CPU/GPU overlap；
- chunked surface fusion；
- normal numerical stabilization；
- LaTeX 编译和表格宏更新。

## 9.4 资源

默认：

```text
1 × RTX 3090 24GB
```

优先：

- Actor/scene chunking；
- mixed precision；
- gradient accumulation；
- activation checkpointing。

禁止为了显存：

- 降空间分辨率；
- 缩 ROI；
- 缩时域；
- 丢危险 Actor；
- 只保留容易场景。

只有 faithful C1 配置经 chunking 后仍超过 24GB，才判定 `blocked_resource` 并申请多卡。

---

# 10. 六周执行节奏

| 周次 | Research | Paper |
|---|---|---|
| Week 1 | P0 + 四象限图谱 | Draft 0、Introduction、teaser storyboard |
| Week 2 | Actor canonical surface、ray-SDF | C1 Method、Figure 3、Table 1 初版 |
| Week 3 | PROJECT/COMPLETE、C2 paired disentanglement | Figure 1/4、Table 2 |
| Week 4 | C3 整合、AV2 adapter、zero-shot | Figure 5、Table 3 |
| Week 5 | 理论、final exact-once、最小下游 | Table 4、Limitations、Supplement |
| Week 6 | 收敛实验、8 页压缩、arXiv | Draft 3、arXiv PDF、匿名版本 |

---

# 11. 版本停止条件

V7 不以所有实验都成功为结束条件。

当以下四项完成后，停止方法扩展：

1. C1 有明确三维物理硬证据和 qualitative；
2. C2 有 paired counterfactual 解耦证据；
3. C3 有 nuScenes→AV2 零样本结果；
4. final exact-once 和最小下游完成。

之后仅允许：

- 论文写作；
- 图表重绘；
- 代码收敛；
- supplement；
- 合法的 bug 修复。

禁止因某个表格不够漂亮继续增加方法分支。

---

# 12. V7 最终交付物

## 研究资产

- HARP-3D physical SceneIR；
- Actor canonical physical surface；
- validity/hazard factorized representation；
- continuous task-cost density；
- joint multi-horizon reliability surface；
- nuScenes→AV2 zero-shot benchmark；
- exact-once final cohort；
- minimal downstream demo。

## 论文资产

- `paper/main.tex`；
- `paper/main.pdf`；
- `paper/supplement.tex`；
- `paper/supplement.pdf`；
- 主图 1–5；
- 主表 1–4；
- `results_macros.tex`；
- arXiv 非匿名版本；
- CVPR/ICCV 匿名版本；
- 失败账本精简版 appendix；
- 项目页与视频素材清单。

---

# 13. 最终决策

V7 的重点不再是：

```text
继续扩展 authority 的 H / q / k / task 条件轴
```

而是：

```text
把 V6.7 已经验证的可靠性链，建立在真正自洽、危险保真的 3D 物理世界上；
通过 paired disentanglement 和跨数据集零样本，形成视觉顶会可以直接理解的硬证据。
```

最终论文因果链应当是：

\[
\boxed{
\text{Physical Artifact Repair}
\rightarrow
\text{Hazard Preservation}
\rightarrow
\text{Task-Conditioned Reliability}
\rightarrow
\text{Minimal Downstream Utility}
}
\]

而不是一个由数百个实验编号组成的 autoresearch diary。
