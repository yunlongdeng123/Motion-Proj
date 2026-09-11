# 历史原始记录 002

> 冻结历史，不是当前执行规则。当前规则见 [scaling law](../../../auto-research_scaling_law.md) 与 [AGENTS](../../../AGENTS.md)。
> 原文件：`docs/RESEARCH_FAILURES.md`；迁移前提交 `abbd431c`。原有相对链接按原目录解释，证据路径原样保留。
> 按索引读取相关小节；旧哈希、过度门控和资源指令均不重新生效。

## r6收口并固定最终视觉对照（2026-09-09 13:10 UTC）

r6相对r4的early下降7.8152pp（95%日志配对区间[−14.2862,−2.8201]pp），五日志均改善；但distance增加.019181m（[+.002470,+.043629]m），五日志均变差。hit+2.9980pp、miss+.1699pp、free−.028730m、recall−1.0169pp的区间均跨0。相对r3，hit+7.9915pp、miss−7.8169pp、recall+2.8675pp改善，同时early+6.7616pp、free+.072641m恶化，这五项区间均不跨0；distance区间跨0。r6不是共同覆盖/物理胜出方法，不能把miss不显著变化当保持等效。

WS-V73-Q-V2-01/20260909T123000Z__open-charts-lidar-first-surface-s7304-r6，训练code d1c22ebe，30轮11130更新/0跳步/0恢复，489 initial/final complete；主汇总只执行一次。DEV75/5日志，23无owned/8空保留；hit/early/miss/free/distance/recall=.390677/.229815/.205579/.237807/.173315/.724428。wall2139.057728s（35.65min）、GPU.210189GiB、RSS1.933098GiB，无DPT/图像前缀；20新日志未读。

固定r6开放chart和首面/miss目标用于最终Joint重接：此选择依据其相对r4明确early改善、相对r3更多hit/更少miss，同时保留tradeoff，并非宣称其全面最优。最终run登记WS-V73-Q-V2-01/20260909T131000Z__open-charts-joint-first-surface-s7304-r7（pending），mode joint、native_surface种子、native-data-weight=1（同原Q-v2联合路径）；64chart/.15/4×4，原coverage+.5beam(.03/res32)+.05box+1first_surface，ray1024/ratio20÷3/RNG7305，event0；相同FIT/cohort/full_track/seed7304/30轮/AdamW1e−5/clip1。fresh M1 DPT初始化，不从r6续训，不新增upper或更换基座。

最终主对照r7−r6：视觉数据、原生种子、DPT多尺度特征与原生build测量辅助项为整条联合通路差异，不能称纯feature因果试验或严格相同总loss。表面/物理损失、Actor数据和更新预算相同，分别报告计算成本；视觉辅助只用相同build侧LiDAR，不引入新heldout监督。一次真实FIT检查单独确认几何损失→DPT与各层特征梯度，辅助深度梯度不冒充几何梯度；通过后正式启动，检查权重不用。

不根据r7中途DEV调配置或挑epoch；完整30轮后收口，固定最终Joint/LiDAR到20保留新日志，无再适配或按确认结果调参。报告所有六项硬表面/射线指标与日志不确定性，保留缺输入对象。若无可信联合收益，明确V7.3假设失败/未获支持并区分原因与推测；如有收益，主张限于实际几何和物理证据，未解决场景边界仍公开。最终报告含组件图、三本台账push成功后停止调度、确认所有任务退出，再shutdown。

证据ray_support/first_surface_r6_{summary,final_manifest,analysis,training}.json与配对/训练图。failure_ledger_delta=update V73-F02; no new failure ID；F02/F03/F04/F09仍active，下一V73-F10。

---

## r6首面梯度检查完成并启动（2026-09-09 12:33 UTC）

代码d1c22ebe的一次检查done：早面距1m、后面正好位于测量2m时，首面项为1m，梯度只作用早面（.612372），后面梯度0；一次局部步后.999625m，有限差分误差9.46e−12。miss项与r4相同且梯度4.714045；近切面梯度612.372742，轮廓切换loss从1到0，证明这些不连续性/放大量仍在，不能声称已解决。真实首个ready FIT五条owned束均有首面，新项.416071m，normal/height/position梯度2.429358/.036763/40.152752；联合一次优化后顶点最大变化.008665m。检查2.259955s、GPU.087645GiB、RSS1.426708GiB，不使用DEV或新日志，也不复用检查权重。

WS-V73-Q-V2-01/20260909T123000Z__open-charts-lidar-first-surface-s7304-r6已于12:31:39 UTC fresh启动，PID154801，训练code d1c22ebe。12:33:26快照：489 initial完整，第1/30轮，81次实际更新/0跳步，峰值GPU.205304GiB。首正式步包含302条首面、722条miss吸引；原coverage/free随机采样保持独立于ray RNG。不把训练loss作为最终质量。完整final后仅运行一次scripts/summarize_worldsim_v73_first_surface_r6.sh，r6−r4/r3/R8六项日志配对指标；r6当前质量pending。

接下来的有限收尾：依据完整r6与既有开放曲面对照选定监督；在相同Actor/cohort、显式曲面、原几何目标、30轮/seed/优化器设置下，重新接现有可训练DPT多尺度几何通路。优先复用Q-v2原联合路径（native种子与native测量辅助项），明确这是整条联合系统增量而非纯视觉特征因果分离；不新增upper/DINOv3/full FT。只缓存完全冻结aggregator，DPT每步重算并接受表面梯度。最终配置选择后在20个保留日志确认，不继续按确认结果调参。Joint无可靠增益就报告核心假设失败/未获支持，完成报告与台账push、停止调度并确认无任务后shutdown。

证据ray_support/first_surface_check_r1.json、r6_started{,_manifest}.json。failure_ledger_refs=[V73-F02,V73-F03,V73-F04]，failure_ledger_delta=none，检查不解除风险；无资源不足。每30分钟跟进，当前不关机。

---

## 首面监督 r6 登记（2026-09-09 12:30 UTC）

在r4同开放曲面/数据/优化预算下，仅将ray-support-kind切为first_surface：owned原始返回有字面交点时使用实际最前面距离的绝对误差，miss时使用r4各向异性最近面吸引，按全部采样owned束平均。前面由几何顺序选择，目标不选择后面；无opacity/删面/UNKNOWN标签。离散轮廓切换与近切面大梯度仍存在，沿用原全局clip1。first_surface统计区分首面数与miss数，lateral/parallel统计仅针对miss，不能与r4全束统计直接混比。

登记WS-V73-Q-V2-01/20260909T123000Z__open-charts-lidar-first-surface-s7304-r6，pending。64chart/1024顶点/1152面，LiDAR-only、full_track、seed7304 fresh、独立ray RNG7305、30轮11130预期更新；coverage+.5finite-beam(.03m/res32)+.05box+1ray，event/native0，最多1024owned。完整489 initial/final及75 DEV/5日志，20新日志未读。本次先一次必要梯度检查，再正式启动；不复用检查权重。r5是固定支持诊断，不是训练编号缺失。

主比较r6−r4及r6−r3；完成后选定表面/监督并重新接可训练视觉几何通路做最终Joint/LiDAR对照，不无限扩展。用户最终收尾安排保持：最终证据与报告/三本台账push成功、暂停调度、确认无任务后shutdown。现阶段不关机。failure_ledger_refs=[V73-F02,V73-F03,V73-F04]，failure_ledger_delta=none at registration，旧风险未解除。

---

## 最新用户终点：完成视觉重接对照后结束 V7.3（2026-09-09）

本条覆盖此前“完成不关机/继续参考方案”的安排。先补完当前首个可见表面监督候选，再将可训练 visual foundation 接回选定显式表面，在相同数据、监督与可比训练预算下完成 Joint / LiDAR 对照。视觉侧须有真实可训练几何通路，不能使用冻结最终特征冒充适配；不无限扩展候选，不默认增加 DINOv3/full FT 或 loss 网格。当前首面候选与最终视觉对照仍 pending，不把本次终点登记当实验完成。

结论规则：若 Joint 在表面与物理查询上有可信增量，论文可围绕 foundation geometry + sparse LiDAR → physically queryable dynamic Actor surface，但主张限定于实际证据；仅覆盖提高而 early/free 恶化不算该主张成功。若 Joint 无明确增量或更差，明确记录本轮 V7.3 核心假设实验失败/未获支持，总结实证原因与仍属推测的机制，不外推为所有视觉基座无效。区间跨零不等于证明统计等效，报告实际差值和不确定性。结构选择用原 DEV，最终方法选定后使用保留新日志确认，不按新日志结果继续调参。

完成后同步远端 RESEARCH_STATUS.md、RESEARCH_FAILURES.md、EXPERIMENTS.md、最终报告和简单 architecture components 图，保存结果与必要 checkpoint，提交并成功 push GitHub 当前 v73 分支。然后暂停本任务30分钟自动跟进和会启动作业的控制器，确认训练、评价、数据任务全部退出，再执行 AutoDL shutdown 并报告实际结果。研究期间仍每30分钟跟进；现在不关机。此终点不要求额外用户确认。

---

## 首面支持诊断收口与阶段paper更新（2026-09-09 11:14 UTC）

固定r3/r4支持分解done，code20b47943；task WS-V73-M2-SURFACE-SUPPORT-01/run20260909T110600Z__open-charts-r3-ray-r4-support-r5。75原DEV/5日志、11886条owned heldout束，23无owned与8空保留；只读保存三角面，0次神经推理/优化，.478369s、RSS.628590GiB。按Actor归一再日志等权，any正确沿束支持39.0399%→58.3205%，其中early且后方有正确面7.9637%→22.2508%，增加14.2872pp；early且无后方正确面8.2562%→8.5458%，仅增加.2896pp。两类增加合计为总early增加14.5768pp。

当前额外early的聚合增量主要落在“早面挡住后方正确支持”类别，因此下一候选优先绑定字面首面：有交点时监督实际首面深度，无交点时保留距离吸引。该候选尚未实现/训练；离散可见性切换、近切面位置梯度及对其他束的冲突需一次必要检查，不预先声称连续性或成功。不是删除早面、把它透明化或将后交点替换为物理输出；不做loss权重网格、不默认解冻upper。这里是类别聚合变化，不是逐射线转移表，更不是归因因果证明。

阶段paper已更新为interim r4：11页、18个完成方法的主表，新增r4−r3完整配对表与开放曲面/射线吸引组件图，修正Q-v2“质量pending”旧叙述，纳入联合对照、局部形变与此次支持分解。LaTeX编译成功，无overfull/编译warning；已检查关键页面和图表。表格读取远端完整证据生成，无新推理：本地首次导出缺少旧R11归档，因此转在远端生成表格再编译，没有重跑R11实验。不是arXiv投稿或研究完成，20新日志质量未读。

证据autoresearch/worldsim_v73/ray_support/support_r5/{summary,manifest}.json、WORLDSIM_V7_3_RAY_SUPPORT.md和paper_v73/；PDF归档autoresearch/worldsim_v73/paper/WORLDSIM_V7_3_INTERIM_r4.pdf。failure_ledger_refs=[V73-F02,V73-F03,V73-F04]；failure_ledger_delta=update V73-F02 evidence; no new failure ID，下一V73-F10。三本台账/计划/报告同步push，upper/DINOv3/full FT与near-boundary free后置，30分钟ACTIVE、完成不关机。

---

## r4收口：吸引改善覆盖但加重物理冲突（2026-09-09 11:06 UTC）

r4射线条件吸引改善了测量附近覆盖，但加重物理冲突。相对同表示r3，miss−7.9868pp、distance−.025294m、recall+3.8844pp，三项95%日志配对区间均不跨0；同时early+14.5768pp、free+.101371m，区间也不跨0。hit+4.9934pp区间[−2.3967,+12.6030]pp跨0，不能宣称正确首交点已可靠改善。r4不作为共同覆盖/物理胜出方法，不把这个结果外推为所有constructive监督无效。

任务WS-V73-Q-V2-01，run20260909T063700Z__open-charts-lidar-ray-support-s7304-r4，训练code28ac577a，30轮11130实际更新/0跳步/0恢复，完整489 initial/final done；汇总仅执行一次。75 DEV/5日志，23无owned/8空保留；hit/early/miss/free/distance/recall=.360696/.307967/.203879/.266536/.154133/.734597。wall2331.936524s（38.87min）、GPU allocated .210189GiB、RSS1.930431GiB、checkpoint12049071字节；无DPT/图像前缀。20新日志质量未读。

下一步登记固定表面支持分解WS-V73-M2-SURFACE-SUPPORT-01/20260909T110600Z__open-charts-r3-ray-r4-support-r5（pending）：只读r3/r4保存三角面和原75 DEV束，按已有八类区分early且后方有正确支持、early且只有邻近支持、early且无正确/邻近支持；不重新神经推理、不改曲面或将后交点替代首返回。此前Q-v2诊断不能代替新参数化的这次判别。结果用于决定应加强首表面归属还是支持位置/方向/范围，不进行alpha/weight网格，也不直接把最近面吸引接回视觉基座。

Point2Mesh BeamGap官方源码已核实，其离散目标/面中心吸引不能等同原始首返回一致性，迁移边界见WORLDSIM_V7_3_RAY_SUPPORT.md。证据ray_support/ray_support_r4_{summary,final_manifest,analysis,training}.json与两图；failure_ledger_refs=[V73-F01,V73-F02,V73-F03,V73-F04,V73-F05,V73-F06,V73-F09]；failure_ledger_delta=update V73-F02 evidence; no new failure ID，下一V73-F10。三本台账/计划/报告同步push；upper/DINOv3/full FT/near-boundary free后置，30分钟ACTIVE、完成不关机。

---

## 射线吸引检查通过，r4正式训练启动（2026-09-09 06:44 UTC）

代码28ac577a的一次检查done：离轴未命中三角的原event NLL=28、位置梯度0；新ray项3.333333→3.311111m，横向.500000→.496667m，梯度范数4.714045。alpha=1欧氏最近点误差0，非切换坐标有限差分误差1.56e−10。真实首个ready FIT（scene-0626/055e6afb9b1143a68f69a181b7f266e4）5条owned返回，新项.010925m，ray-only法向/高度/位置梯度1.181130/.568250/25.316832；联合一次更新最大顶点变化.007960m。2.273089s、GPU.087677GiB、RSS1.410324GiB。未读取DEV/新日志，检查不等于质量提升、不能解除F03；不复用检查权重。

WS-V73-Q-V2-01/20260909T063700Z__open-charts-lidar-ray-support-s7304-r4已于06:43:05 UTC fresh启动，code28ac577a/PID123769。06:44:44快照完整489对象initial done、第1/30轮、10实际更新/10呈现/0跳步，已记录有效ray监督；GPU峰值当时.195063GiB，盘余67GiB，无资源短缺。30轮预期11130更新，未完成；不把initial或训练采样loss当最终收益。

同r3开放表示、原coverage/finite-beam/box和数据/优化器不变，新项权重1、横向比例20/3、最多1024 owned束，独立CUDA RNG7305另存/恢复。首个正式step的原coverage1.536066m/free.656263m与r3首步一致；新项7.765499m参与真实反传。此单步一致不是全训练等价性证明。无DPT或视觉前缀，20新日志质量未读；upper/DINOv3/full FT和near-boundary free继续后置。

证据autoresearch/worldsim_v73/ray_support/{path_check_r1,r4_started,r4_started_manifest}.json，定义与组件图见WORLDSIM_V7_3_RAY_SUPPORT.md。完整final后仅运行一次summarize_worldsim_v73_ray_support_r4.sh，对照r3为主并附r2/R8；本次尚未收口r4。failure_ledger_refs=[V73-F01,V73-F02,V73-F03,V73-F04,V73-F05,V73-F06,V73-F09]；failure_ledger_delta=none，旧风险仍active、下一V73-F10。三本台账/计划/报告同步push，30分钟ACTIVE、完成不关机。

---

## 射线条件曲面吸引实现与r4登记（2026-09-09 06:37 UTC）

基于r3收口8b012af1，保持64开放chart/1024顶点/1152面，只新增真实返回条件的各向异性最近面吸引。delta按只读射线方向分解，min_surface ||(20/3)*delta_perp+delta_parallel||，单位米、权重1；原coverage+.5beam(.03m/res32)+.05box保持。64ray×512face分块，完整度量最优重心停止梯度后按包络定理反传实际顶点；未命中仍可拉动非空表面，但不保证首交点/正面积或空拓扑出生。不是新的event概率模型或未知FREE。

登记WS-V73-Q-V2-01/20260909T063700Z__open-charts-lidar-ray-support-s7304-r4（pending）：同r3 LiDAR-only/full_track、seed7304 fresh、30轮预期11130更新、AdamW1e-5/clip1，完整489 initial/final，75 DEV/5日志/空预测保留。新项最多1024 owned原始首返回，从全部FIT测量束独立抽样，CUDA Generator seed7305另存/恢复，不消耗旧coverage/free随机流；相对去重point coverage也增加返回观测权重，不能将收益只归因方向度量。无DPT/视觉前缀，20新日志质量未读。

一次必要检查pending：解析miss时原event零梯度、新项梯度和下降方向、各向同性一致性/一个非切换坐标有限差分；元数据首个ready FIT的新项位置/法向/高度梯度与一次联合更新。通过后fresh正式训练，不复用检查权重、不重复旧factory/regression。实现、SoftRas/DRC一手迁移及简单组件图见WORLDSIM_V7_3_RAY_SUPPORT.md；入口check_worldsim_v73_ray_support.sh/run_worldsim_v73_ray_support_r4.sh，完整final后一次summarize_worldsim_v73_ray_support_r4.sh，对照r3为主、闭合r2/R8附列。不做weight/alpha网格、不启动upper/DINOv3/full FT或near-boundary free。

failure_ledger_refs=[V73-F01,V73-F02,V73-F03,V73-F04,V73-F05,V73-F06,V73-F09]；failure_ledger_delta=none at registration，旧风险未解除、下一V73-F10。三本台账/计划/实现同步push；30分钟ACTIVE，完成不关机。

---

## 开放chart r3收口：支持仍未兑现物理收益（2026-09-09 06:29 UTC）

相对闭合LiDAR r2，开放chart r3六项区间均跨0，没有建立优势或等效。相对旧窄片R8，miss−28.4589pp（区间[−40.4369,−16.4809]pp），free+.130846m（[+.003899,+.294775]m）；hit+.0734pp、early+10.6289pp、单向distance−.050126m、recall−2.3699pp均跨0。开放支持仍未同时改善覆盖和物理，不能将此前冲突唯一归因闭合，也不能宣称所有开放表示失败。

WS-V73-Q-V2-01/20260909T054000Z__open-charts-lidar-full-track-beam-s7304-r3，训练code72607d0c，30轮11130实际更新/0跳步/0恢复、完整489初始/最终评价done。DEV75/5日志含23无owned/8空；hit/early/miss/free/distance/recall=.310762/.162199/.283747/.165166/.179427/.695753。r3−r2 hit+.5273pp、early−3.2702pp、miss+2.8359pp、free+.024642m、distance+.017062m、recall+4.7504pp，全部区间跨0。1752.516051s、GPU.210188GiB、RSS1.940502GiB、checkpoint12048622字节。参数1706413，未调用视觉模块无梯度，无DPT/图像前缀，不冒充联合结果。

下一步保持r3表示，按已查[SoftRas官方实现](https://raw.githubusercontent.com/ShichenLiu/SoftRas/master/soft_renderer/functional/soft_rasterize.py)的距离梯度与[DRC作者页](https://shubhtuls.github.io/drc/)的射线一致性思路，独立实现射线条件最近面吸引，做一次未命中梯度检查后登记r4。现有coverage已有真实吸引；新候选检验方向信息，不能写成first-hit保证或新物理概率模型，不导入opacity、未知FREE、正占据厚度或删面。upper/DINOv3/full FT与near-boundary free后置；20新日志质量未读，30分钟ACTIVE、完成不关机。

证据WORLDSIM_V7_3_OPEN_CHARTS.md与autoresearch/worldsim_v73/open_charts/open_charts_lidar_r3_*；failure_ledger_refs=[V73-F01,V73-F02,V73-F03,V73-F04,V73-F05,V73-F06,V73-F09]；failure_ledger_delta=update V73-F02 evidence; no new failure ID，下一V73-F10。三本台账/计划/报告同步push。

---

## 开放曲面r3正式训练进行中（2026-09-09 05:42 UTC）

`WS-V73-Q-V2-01/20260909T054000Z__open-charts-lidar-full-track-beam-s7304-r3`已从72607d0c启动，PID118560。05:42:05 UTC快照：完整489对象initial评价done，进入第2/30轮，596次实际更新/596呈现，0跳步；GPU allocated峰值0.210188GiB，运行RSS约1.90GiB，盘余67GiB。状态正常，保持原配置，不把采样训练loss或初始评价当最终收益。检查脚本的一次优化不计入r3，正式作业fresh开始。

64个开放chart、1024顶点/1152面、参数1706413，原full_track/seed7304/coverage+beam.5/.03/res32/env.05、event0、无DPT/图像前缀，30轮预期11130更新。它先隔离表面支持分配；原生可训练几何/局部视觉接口保留，但本LiDAR对照没有训练视觉。20新日志质量未读，upper PEFT/DINOv3/full FT和near-boundary free未启动。

实际证据`docs/autoresearch/worldsim_v73/open_charts/lidar_r3_started{,_manifest}.json`；组件图及定义见OPEN_CHARTS报告。完整final done后仅运行一次`scripts/summarize_worldsim_v73_open_charts_r3.sh`，比较同LiDAR闭合r2与R8窄片，保留75 DEV/5日志、空预测和6项指标；该收口尚未执行。不把支持/初始化/尺度同时改变的结果只归因闭合。随后独立实现/比较constructive ray监督，不重跑已完成诊断。

failure_ledger_refs=[V73-F01,V73-F02,V73-F03,V73-F04,V73-F05,V73-F06,V73-F09]；failure_ledger_delta=none，质量pending。三本台账/计划/报告同步push，30分钟ACTIVE、完成不关机。

---

## 开放曲面真实通路通过，进入正式r3（2026-09-09 05:39 UTC）

基于019735d2的一次真实FIT检查done：按原元数据顺序首个ready车辆，3个build点/515个上下文查询，64chart/1024顶点/1152面，半宽.330649m；实际5个full_track目标点、194条原始束参与coverage/beam/box。一次AdamW更新，normal/height/displacement梯度范数=.666979/.573098/15.371632，表面最大变化.007932m；固定推理factory按保存配置恢复误差0。检查1.828437s、GPU.080078GiB、RSS1.421513GiB、参数1706413；不代表完整数据资源或DEV质量，更不代表视觉通路已训练。

首次直接调用缺少runtime bin的PATH，PyTorch扩展找不到现有Ninja，在反传/更新前退出；先查官方cpp_extension的ninja --version调用，补齐与正式入口一致的PATH/CUDA环境后同一检查通过，未安装环境/包。首错和通过记录保留`docs/autoresearch/worldsim_v73/open_charts/path_check_attempt1.txt`与`path_check_r1.json`；复用这次检查，不再重复。新增shell入口显式设置现有工具路径。

任务WS-V73-Q-V2-01/r3仍按既定30轮/seed7304/full_track、原coverage和finite-beam参数，代码提交后启动；正式实际PID/初始评价与更新数另记，不把通路检查的1次更新算入r3。训练manifest明确LiDAR未加载DPT/前缀，关联F09也保留；没有新科学失败，failure_ledger_delta=none，环境调用问题已恢复。计划、三本台账与OPEN_CHARTS报告同步，20新日志未读、30分钟ACTIVE、完成不关机。

---

## 开放局部曲面实现与LiDAR对照登记（2026-09-09 05:40 UTC）

基于d11e6788实现ActorOpenChartQueryDecoder并接入训练/固定推理：原≤1536观测/补全查询进行三层空间更新，在初始支持中FPS分配64输出chart，4×4共享网格共1024顶点/1152面。只读局部PCA坐标架+可训练法向残差/UV高度，尺寸确定半宽.15*(LWH)^(1/3)，无opacity/radius/删面/全局闭合。旧查询更新提取复用，原patch计算顺序/权重名保持；heightfield旧小片已存在，新增支持分配/尺度不是新拓扑保证。chart间相交、PCA污染和固定尺度仍可能失败。

任务WS-V73-Q-V2-01，登记`20260909T054000Z__open-charts-lidar-full-track-beam-s7304-r3`（pending）：先LiDAR-only同原cohort/full_track/seed7304/30轮、coverage/beam.5/.03/res32/env.05/event0、AdamW1e-5clip1，与闭合r2/窄片R8比较。完整489对象初始与最终评价，51不可用保留；这是表面支持、初始化和拓扑组合改变，不能只归因闭合。原生DPT/局部图像接口保留，但本r3不加载/训练DPT，不冒充视觉研究结果。

先一次真实FIT对象梯度/保存恢复检查（pending），通过后正式执行；不重复旧smoke/回归。实际实现、参数化边界、一手AtlasNet/SoftRas/DRC引用及简单组件图见`docs/WORLDSIM_V7_3_OPEN_CHARTS.md`。实现/训练入口和此登记同提交，随后真实过程补记。constructive ray监督下一独立因素，upper/DINOv3/full FT和near-boundary free均不启动；20新日志未读。

failure_ledger_refs=[V73-F01,V73-F02,V73-F03,V73-F04,V73-F05,V73-F06,V73-F09]；failure_ledger_delta=none at registration，旧风险保持、下一V73-F10。三本台账/计划同步，30分钟ACTIVE、完成不关机。

---

## Q-v2诊断收口，进入开放局部曲面候选（2026-09-09 04:55 UTC）

固定网格诊断支持优先构造正确沿束支持，也修正了局部塌缩的归因：joint early26.9437%中仅1.4956个百分点有后方正确支持，其余25.4481个百分点没有；LiDAR r2该项为6.5867个百分点。joint 67非空网格未检出非邻接三角自交，局部形变温和；LiDAR r2有48/67检出相交且有局部高拉伸。因此不能把joint较差简单归因于collapse/stretch或自交。两支都未同时兑现物理与覆盖，闭合拓扑是否为原因仍需要改变参数化来检验。

两项固定诊断done，code61016ca5：support-r4仅joint，.367797s/RSS.628716GiB；mesh-deformation-r1比较r1/r2，3.196148s/RSS1.008175GiB。75原DEV、5日志、8空保留，仅读已保存表面，r2 support-r3复用。完整八类支持/局部形变量及边界见Q-v2报告文末和`docs/autoresearch/worldsim_v73/qv2/{support_r4,mesh_diagnostic_r1}/`。joint局部形变温和不能说明整体位置正确；Open3D检测跳过相邻面，0检出不是无折叠保证。

下一项进入开放局部结构化chart实现（pending）：重分配更大局部支持、保留空间交互/可训练DPT接口/同一显式三角面，避免固定闭合外壳；chart内局部高度图不是新增发明（旧小片已有），chart间重叠/错位仍须评价。随后单独检验射线方向分解的constructive支持吸引，不用未知FREE或opacity。已核对AtlasNet官方实现、SoftRas论文与DRC作者页，迁移边界见报告；不再追加无尽诊断，upper PEFT/DINOv3/full FT不启动，near-boundary free后置。

failure_ledger_refs=[V73-F02,V73-F03,V73-F04]；failure_ledger_delta=update V73-F02 evidence; no new failure ID，下一V73-F10。三本台账、计划与报告同步；20新日志未读，30分钟ACTIVE、完成不关机。

---

## Q-v2固定表面诊断登记（2026-09-09 04:50 UTC）

joint r1收口已提交3ac15ca2并push，当前联合通路无明确增量，按策略二优先表面/constructive ray监督。登记两项顺序CPU诊断，状态pending：`WS-V73-M2-SURFACE-SUPPORT-01/20260909T045000Z__qv2-joint-r1-support-r4`仅对r1做原八类沿束支持分解，LiDAR r2既有support-r3直接复用；`WS-V73-Q-V2-MESH-DIAGNOSTIC-01/20260909T045000Z__fixed-dev-mesh-deformation-r1`读取r1/r2原75 DEV固定网格与原checkpoint模板，统计局部形变和非邻接三角相交。入口`scripts/run_worldsim_v73_qv2_fixed_diagnostics_r1.sh`；代码与本登记同提交后执行，无神经推理/优化/新数据。

逐面3×2映射奇异值相对尺寸缩放解析模板，整体刚体转动不影响统计；记录smin/smax、面积/边长比/各向异性分位数及描述性比例，不把模板当GT或分位数当门槛。Open3D0.19官方源码明确跳过共享任意顶点的面配对，因此相交数不覆盖相邻折叠，零相交也不保证无问题；正常两个沿束深度层不算自交。8空预测在物理分母保留，局部形变量记不可用；67非空对象先Actor统计、日志内均值、日志等权，不把面数当独立样本。不会改网格、删面、重网格化或取凸包。

已核对PyTorch3D变形教程与Open3D三角相交实现，相关来源见Q-v2报告和run manifest。判别目标是明确错误返回是否有后方正确支持，并测量局部变形风险；这些只定位症状，不能证明闭合/视觉为唯一原因。诊断后选择开放/结构化表面及constructive监督，不继续无限诊断/仅调free。20新日志未读、upper/DINOv3/full FT不启动，30分钟ACTIVE、完成不关机。

failure_ledger_refs=[V73-F02,V73-F03,V73-F04]；failure_ledger_delta=none at registration，现有风险保持，下一V73-F10。三本台账、计划与实现同步。

---

## Q-v2 joint r1收口：优先表面表示与沿束支持（2026-09-09 04:40 UTC）

Q-v2联合r1与同网格LiDAR r2均done。r1−r2的early为+7.4536pp，95%日志配对区间[+1.1217,+17.7188]pp；hit−2.0905pp、miss+4.6790pp、free+.038779m、单向distance+.026002m、recall−5.3473pp均值方向均较差，但这五项区间跨0。当前整条联合通路没有显示明确的有效增量，不能把跨0当等效性证明，也不能外推所有视觉基础模型无用。按第18节策略二/不确定性分支，下一轮优先surface representation与constructive ray supervision；upper PEFT、DINOv3、full FT不启动，near-boundary free后置。

task WS-V73-Q-V2-01/run `20260908T221500Z__shared-mesh-full-track-beam-s7304-r1`，code bfc181b4、分析50222d3e。30轮11130实际更新、零跳步/恢复，完整489对象最终评价done；75 DEV/5日志含23无owned、8空预测。hit/early/miss/free/distance/recall=.284584/.269437/.302179/.179303/.188367/.594776。相对R12 miss−34.9277pp，但early+21.3014pp/free+.118729m/recall−16.7426pp，这四项区间不跨0；不能把减少缺失写成正确表面。耗时22410.642356s、GPU11.591165GiB、RSS34.133816GiB、checkpoint412177558字节；训练/最终评估均已退出。

完整六指标及对照区间、训练过程、FIT/移动DEV限制见`docs/WORLDSIM_V7_3_QV2_SHARED_MESH.md`文末，证据`docs/autoresearch/worldsim_v73/qv2/shared_mesh_joint_r1_{summary,final_manifest,analysis,training}.json`和两图。只执行一次收口分析，两图已目视检查；原始checkpoint和表面保留。

接下来先做固定网格局部变形及r1沿束支持诊断，复用r2 support-r3，之后选open/structured surface和constructive监督；不重复旧bootstrap，不重推理取证，不以模板作GT或两层交点作自交。已查PyTorch3D/Open3D官方实现及AtlasNet一手资料，迁移边界见报告。现有可训练几何路径保留，但不投入upper PEFT/DINOv3/full FT；near-boundary free后置。20新日志质量未读，30分钟ACTIVE，完成不关机。

failure_ledger_refs=[V73-F01,V73-F02,V73-F03,V73-F04,V73-F05,V73-F06,V73-F09]；failure_ledger_delta=update V73-F02 evidence; no new failure ID，下一V73-F10。三本台账、计划、AGENTS与专项报告同里程碑提交/push。

---

