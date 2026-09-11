# 历史原始记录 013

> 冻结历史，不是当前执行规则。当前规则见 [scaling law](../../../auto-research_scaling_law.md) 与 [AGENTS](../../../AGENTS.md)。
> 原文件：`docs/RESEARCH_FAILURES.md`；迁移前提交 `abbd431c`。原有相对链接按原目录解释，证据路径原样保留。
> 按索引读取相关小节；旧哈希、过度门控和资源指令均不重新生效。

### V73-F02 — 自由生成可能后退、消失或收缩来逃避 free

- 最终独立确认补充（2026-09-09 21:30 UTC）：最终固定AV2跨域确认失败：Joint r7相对匹配LiDAR r6的hit−2.9419pp、early+5.3637pp、free+.325997m、target→surface距离+.013259m、recall−2.4204pp，五项95%日志配对区间均排除0且方向更差；free在20/20日志变差。miss−.9625pp的区间[−1.9366,+.0576]pp跨0，不构成可靠改善或等效。开发5日志hit+10.5363pp等正结果仍成立，但未通过20个独立AV2日志的跨数据集确认。本轮V7.3联合方法的可泛化物理表面主张实验失败/未获支持，不外推所有视觉基座无用。 证据docs/autoresearch/worldsim_v73/final_confirmation/{analysis,models}.json及配对图；run WS-V73-FINAL-CONFIRMATION-01/20260909T135000Z__fixed-r7-r6-external20-r1，执行code6499d34d，全部五方法936对象/20日志/0优化更新。878 ready、58缺输入、119无owned heldout对象均保留；含221移动>2m/s、108运动未知、556 build<100对象。模型/尺度/配置固定在读取外部质量之前，未按AV2再训练、调参或挑场景。

- 最终联合r7（2026-09-09）：最终开发对照r7−r6显示联合几何增益：hit+10.5363pp（95%日志配对区间[+6.4624,+14.6212]pp，5/5改善），miss−4.3793pp（[−9.3948,−.2924]pp，4/5），测量→表面distance−.079399m（[−.187384,−.009326]m，4/5）。early−1.5225pp（[−5.7790,+2.7510]pp）、free+.042022m（[−.000489,+.069738]m）、recall+8.4288pp（[−.7072,+23.9799]pp）三项区间跨0。free四日志变差，不能用不显著掩盖其风险，也不能把此结果归类为Joint≈LiDAR或全部失败。当前仅支持联合通路对开发几何与正确首返回有增量；完整物理主张和跨域泛化尚待固定确认。 维持物理冲突active，不能将旧模型的负结果外推本联合通路无效。

- r6首面监督（2026-09-09）：r6相对r4的early下降7.8152pp（95%日志配对区间[−14.2862,−2.8201]pp），五日志均改善；但distance增加.019181m（[+.002470,+.043629]m），五日志均变差。hit+2.9980pp、miss+.1699pp、free−.028730m、recall−1.0169pp的区间均跨0。相对r3，hit+7.9915pp、miss−7.8169pp、recall+2.8675pp改善，同时early+6.7616pp、free+.072641m恶化，这五项区间均不跨0；distance区间跨0。r6不是共同覆盖/物理胜出方法，不能把miss不显著变化当保持等效。 固定该目标做最终视觉重接，不扩展候选；支持/物理冲突仍active。

- r3/r4固定支持分解（2026-09-09 11:14 UTC）：early且后方有正确面7.9637%→22.2508%，该类别增加14.2872pp；无正确后面early只增加.2896pp。聚合增量支持优先首表面监督，但不是逐束因果归因。code20b47943、ray_support/support_r5/与RAY_SUPPORT报告；不以删面/opacity替代修正，F02仍active。

- 射线吸引r4（2026-09-09 11:06 UTC）：r4射线条件吸引改善了测量附近覆盖，但加重物理冲突。相对同表示r3，miss−7.9868pp、distance−.025294m、recall+3.8844pp，三项95%日志配对区间均不跨0；同时early+14.5768pp、free+.101371m，区间也不跨0。hit+4.9934pp区间[−2.3967,+12.6030]pp跨0，不能宣称正确首交点已可靠改善。r4不作为共同覆盖/物理胜出方法，不把这个结果外推为所有constructive监督无效。 code28ac577a，RAY_SUPPORT报告与ray_support_r4_*。下一固定分解r5区分错误首面与后方支持，当前总量不能唯一诊断因果；保持active，无新ID。

- 开放chart r3补充（2026-09-09 06:29 UTC）：相对闭合LiDAR r2，开放chart r3六项区间均跨0，没有建立优势或等效。相对旧窄片R8，miss−28.4589pp（区间[−40.4369,−16.4809]pp），free+.130846m（[+.003899,+.294775]m）；hit+.0734pp、early+10.6289pp、单向distance−.050126m、recall−2.3699pp均跨0。开放支持仍未同时改善覆盖和物理，不能将此前冲突唯一归因闭合，也不能宣称所有开放表示失败。 相对initial更少early/free却更多miss/更低recall，四项区间不跨0。code72607d0c、OPEN_CHARTS报告与open_charts_lidar_r3_*；保持active，无新ID。

- 固定网格诊断（2026-09-09 04:55 UTC，code61016ca5）：joint early26.9437%仅1.4956pp有后方正确支持；joint67非空网格0个检出非邻接自交，LiDAR48/67检出。joint局部形变温和，不能将其负结果直接归因collapse/stretch；LiDAR确有局部强变形风险，两者因果未定。下一步开放局部支持分配与独立constructive监督，证据qv2/support_r4及mesh_diagnostic_r1。F02保持active，无新失败ID。

- Q-v2联合r1收口（2026-09-09 04:40 UTC）：同网格r1−r2 early+7.4536pp，95%日志配对区间[+1.1217,+17.7188]pp，其余五项均值较差但区间跨0；相对R12减少miss却增加early/free并降低recall。当前联合路径未显示明确有效增量，不能推导所有视觉无用或闭合拓扑为唯一原因。按用户策略二优先表面表示/constructive ray监督，局部collapse/stretch待固定网格统计。30轮11130更新、完整489最终评价done；证据qv2/shared_mesh_joint_r1_*及Q-v2报告，code bfc181b4。F02保持active，无新失败ID。

- Q-v2 support-r3（2026-09-08 23:55 UTC）：r2 early19.4901%中仅6.5867个百分点有后方正确交点，另12.9034个百分点无正确沿束支持；late24.4222%。仅剔除早表面或加free不能保证正确返回。固定r2/R8、75 DEV、11886 owned束，见qv2/support_r3；不以两层交点认定自交、不归因背景全束，联合r1继续。

- Q-v2 LiDAR r2补充（2026-09-08 23:50 UTC）：共享网格相对R8减少miss31.295pp，却增加early13.899pp/free.106203m，三项日志区间不跨0；hit/距离/recall均跨0。连通支持未同时满足观测表面与物理要求，且此冲突不依赖视觉输入；不能唯一归因为视觉背景污染或据此拒绝联合基座。r1仍运行，固定表面诊断登记pending。证据Q-v2报告与qv2/shared_mesh_lidar_r2_*，训练code95050522；保持active，无新失败ID。

- R12三角补充（2026-09-08 22:05 UTC）：同Query只改beam free后，DEV early−14.621pp/free−.210453m，但miss+32.528pp、hit−5.553pp、recall−6.385pp，六项日志配对区间不跨0。R12−R14 free改善但miss增加；R12−同beam R8在5日志降低hit/增加miss。支持覆盖与物理兑现仍冲突，触发Q-v2表面参数化；根因不唯一、保持active，不重复loss网格。证据`WORLDSIM_V7_3_TRIANGLE_RESULTS.md`及`m2/global/population_joint_r12_*`，训练code dc6fe427。

观察：用户指出目标漏洞，继承 V71-F20/F22 的支持/前尾边界，当前尚未实测新查询。根因候选：缺乏覆盖责任、可学习置信度/支持半径与惩罚耦合。迁移：AdaPoinTr 集合生成 + observed-target coverage + 原始束几何 free，UNKNOWN 不作负标签；不学任意 opacity。比较同几何监督加free后的召回/侵入/missing，而非只看loss。复开/解决条件：真实表面覆盖与侵入共同改善且没有支持消失；证据=plan13.2，task=M2/M3。

2026-09-08 R10实测补充（上述为初始登记风险）：状态active，已不只是未实测假设。全DPT＋Query/full_track/hard-free完成30轮；同目标R10−R11在5开发日志均提高hit、降低missing，也在5日志均增加early，free配对+.106062m、95%[+.013990,+.209804]m。相对R7同监督LiDAR，free+.199452m、[+.077589,+.317941]m。距离/召回的平均改善区间跨0，不称稳定准确性优势。当前不是靠opacity或半径缩小逃避惩罚，具体错误仍可能涉及片形状、方向、重叠、对应与目标代理，尚未完成因果归因。固定R5/R9/R11表面诊断还显示邻近覆盖不等于正确沿束支持，后方正确片也不能替代首返回。证据：R10 code26a7e509、`m2/global/population_joint_r10_{summary,analysis,training}.json`，支持诊断code866ed28f；R12/R14仍运行。复开路径：先完整三角，冲突若保留优先Query surface parameterization；不退回native-only，不把unknown标FREE，不将局部边界监督或attention一致性未经对照混入当前训练。无新失败ID。

### V73-F03 — 缺支持时首事件 NLL 无法独立创造几何

观察：固定支持概率梯度不等于位置支持生成保证。迁移：DS-NeRF 终止分布作为先例，coverage/三维吸引与 coarse-to-fine 生成保持有效；必要时有限宽度延拓。禁止丢弃无候选射线和在推理中用 target 生点。判断：同架构对照记录缺支持率、硬交点及几何梯度，不能仅以NLL下降关闭风险；证据=plan13.3，task=M3。

### V73-F04 — Actor/背景重复表面与边界伪影

- 最终固定场景确认（2026-09-09 21:42 UTC）：最终场景确认done：20日志40 heldout帧×6方法=240记录，每方法3908250原始束。cohort返回r7−r6 hit−2.211pp（[−3.253,−1.136]pp）、miss+1.347pp（[+.542,+2.432]pp）、returned MAE+.223034m（[+.136486,+.323606]m）；边界带hit−1.578pp、free+.034947m、returned MAE+.191461m的区间也均不跨0且变差。全束hit−.1845pp/miss−.1695pp均不跨0，其余全束三项区间跨0；背景分母较大不能遮盖Actor退化。边界与归属为box proxy、背景不完整、未知区域不当FREE，F04未解决。CPU wall720.392s、RSS1.207802GiB，0优化更新，无重复评价/汇总。 证据final_scene/{analysis,summary,manifest}.json，任务WS-V73-FINAL-SCENE-01，执行code65854c27。执行done，科学风险active；本轮已关闭不再启动修复候选。

观察：背景 ghost、Actor 缺口与外扩可造成假 early/hit 改变，目前未完成 V7.3 scene 应用。迁移：Street Gaussians 分层思想，但物理使用统一硬排序，不用其外观透明度；逐测量时刻排除动态点，遮挡后背景UNKNOWN。最小比较：同一背景/轨迹的 Actor、边界带和全场景分列指标，反事实示例不冒充真值；证据=plan13.4，task=M4。

清理 outcome：删除明确退役依赖/下载缓存；保存包版本与逐项目录，全部 data/run 保留；无需分配新的科学 failure。failure_ledger_delta=`V73-F01:F04_active`；同逻辑提交包含 plan/status/experiments/storage report。

---

## V7.3 当前失败状态（2026-09-07）

- 当前没有新增 V7.3 科学失败；下一编号保持 `V71-F70`。
- 已知风险是单卡 24GB 可能无法在目标视图数和分辨率下反传 VGGT 原生多层 DPT/上层聚合器。先采用混合精度、梯度检查点、梯度累积、冻结早期前缀和可复用前缀缓存；这些只改变执行方式，不把研究路线改回冻结最终特征外挂。
- V7.2 `V71-F68/F69` 仍有效，但边界仅为 raw late evidence 融合的跨日志退化及冻结 source 缺乏视觉支持。它们不否定 V7.3 的可训练几何解码通路。
- 新卡点的处理顺序固定为：确认具体机制症状；检索最接近的顶会/官方开源；迁移到当前数据和评价；记录负结果与下一动作。普通 dtype、I/O、字段适配错误直接修复，不扩张成新的门控流程。

---

## V71-F69 — 冻结 source 视觉 cohort 无可观测 Actor 候选（2026-09-08）

- category=`protocol/evaluation_support`；status=`closed_inconclusive`；task=`WS-V72-E5-FROZEN-CONFIRMATION-01`。
- symptom：12 个锁定 source windows 和 8 个 Actor pose matches 中，`observed_candidate_count=0`、`camera_observation_count=0`。formal evaluator 对空 evidence tensor 求均值产生 NaN，严格 JSON 序列化拒绝写 summary；四份 surface actor rows 已写出，但不能据此生成视觉证据比较。
- boundary：这是冻结 cohort 的模态支持失败，既不是 EAS 性能负数，也不是正结果。source scene/sample、checkpoint、代码 hash 和 route-selected `alpha=.35` 均匹配原 lock；不得换 scene、调 alpha 或把 undefined 指标写成 0。
- research migration：联网核对 CVPR 2022 missing-modality robustness、CVPR 2025 MoME 与 selective prediction 后，迁移为 target-free support audit 和显式 insufficient-support verdict。canonical audit=`run://worldsim_v72/WS-V72-E5-FROZEN-SUPPORT-AUDIT-01/20260908T002500Z__e5-source-support-audit-s7502-r4`，不加载 checkpoint/label、不做选择。
- prevention：未来独立 cohort 在任何 quality read 前冻结 camera/Actor overlap 与 minimum observed-candidate eligibility；无视觉支持时使用 measurement fallback 并单独报告覆盖率。当前 source 不再重复读取以制造确认。
- failed formal=`run://worldsim_v72/WS-V72-E5-FROZEN-CONFIRMATION-01/20260907T235000Z__e5-source-frozen-s7502-r1`；support audit r1–r3 为字段名适配工程失败，r4 canonical。

## V71-F68 — raw visual expert 在 route 上伤害 Brier（2026-09-08）

- category=`scientific/domain_shift_fusion`；status=`resolved_narrowly_on_route`；task=`WS-V72-E5-FROZEN-CONFIRMATION-01`。
- symptom：VGGT raw late visual 相对 no-visual 的 route NLL `.95972→.95848` 略好，但 Brier `.23753→.24477` 退化；Pi3X 在 development 也明显退化。直接把视觉 expert 全权用于新日志没有可靠性。
- migration：先查 CVPR 2025 MoME 的独立专家/质量路由和 CVPR 2021 domain-drift calibration，再在 route 上选择 simplex-preserving `p=p0+alpha(pv-p0)`。`alpha=.35` 得到 Brier `.23434`、NLL `.94380`，2/3 logs Brier 改善；模型参数、几何和输入不变。
- boundary：alpha 在 route 上选择，只能支持 route-select 恢复；source 因 F69 不可计算。不得写通用校准或独立泛化。Pi3X 负结果保留，不能因 VGGT 恢复声称多基座成功。
- evidence=`run://worldsim_v72/WS-V72-E5-FROZEN-CONFIRMATION-01/20260907T223000Z__e5-route-calibrated-s7501-r2`；source lock 在 payload read 前保存 alpha、checkpoint、代码与 cohort hash。

## V72 工程恢复补记（2026-09-08）

- nuScenes RGB mirror 实际为 `800x450`，按冻结清单从三个相机 archive 选择性恢复并记录 SHA；缺 RGB 的 route cache r1 保留失败。
- source LiDAR 首次 subset 参数未传给 worker，误启全部十个 shard；中断并清除 orphan 后修复，只扫描官方 shard 02/10，canonical `677/677`。
- MapAnything r1 将 float64 intrinsics 传入 float32 ray encoder，r2 使用未登记 scale enum；按官方 API 转 float32 并使用 `metric_aligned` 后 r3 完成。均为适配工程错误，不解释为模型质量。

下一可用统一失败编号：`V71-F70`。

---

## 历史补充风险总览（2026-09-07，已由文首 F68/F69 与结果页更新）

| 当前范围 | 状态与解释 | 证据入口 |
|---|---|---|
| EAS-VGGT 方向继承 | F65 的规划纠偏保持；V7/V7.1 正负结果均保留 | 当前 plan 第 2 节 |
| 论文主张与证据缺口 | F66 active；revision 2 已加强设计，尚无新实验解除 | 当前 plan 第 3–7 节、补充调研文档 |
| E1 工程状态 | F67 resolved；双基座与 beam/split 合同已完成，Waymo payload 待授权 | E1 报告、canonical r4 |
| 下一步 | E1 running，E2–E5 pending；旧 A/B 不恢复 | RESEARCH_STATUS 文首 |

## V71-F67 — VGGT BF16 位姿求逆与 token 轴假设破坏公共适配器（2026-09-07）

- category=`engineering/backbone_adapter_contract`；status=`resolved`；task=`WS-V72-E1-VGGT-EVIDENCE-IO-01`；resolved commit=`4c86e621`。
- symptom：正式 r1 在 VGGT 相机 pose 的 `torch.linalg.inv` 处报 `Low precision dtypes not supported. Got BFloat16`，没有产生 summary。改用闭式逆后 r2 完成，但 VGGT feature shape 为 `[3,27,48,0]`；适配器把官方 `[B,S,N,C]` token 的相机轴误当成 token 轴。
- cause/boundary：VGGT 官方在 Ampere 上用 BF16 推理，而 PyTorch 通用矩阵逆只支持 float/double/complex；VGGT aggregator 保留显式 sequence 维。这是薄适配层假设错误，不是基座精度负结果。r1 未读 target quality/source/external test；r2 的几何数字有效但 feature contract 不完整，不能作 canonical。
- resolution：复用 VGGT 官方 `closed_form_inverse_se3`；按 `[B,S,patch_start:,C]` 提取特征，要求 cache 的空间/通道维均非零并增加轴布局回归。r3 验证修复但使用未提交源码，最终在 commit 后生成 r4；r4 feature shape 均为 `[3,27,48,2048]`，manifest code=`4c86e621`。
- prevention：外部模型 adapter 必须验证官方张量轴、dtype 与坐标约定，非空 shape 是接口通过条件；dirty tree run 不登记 canonical。对精确工程错误复用官方实现并重跑一次，不把它扩展成模型 sweep。
- evidence=`run://worldsim_v72/WS-V72-E1-VGGT-EVIDENCE-IO-01/20260907T142000Z__e1-vggt-pi3x-train-observation-s7201-r1`、canonical r4、`docs/WORLDSIM_V7_2_E1_BACKBONE_AND_BEAM_REPORT.md`；failure_ledger_delta=`V71-F67_resolved`。

下一可用统一失败编号：`V71-F68`。

## V71-F66 — 条件回波和接口非干扰不足以支持通用传感器适配（2026-09-07）

- category=`protocol/research_claim_scope`；status=`active`；mitigation=`plan_revised_verification_pending`；task=`WS-V72-E0-CONFERENCE-INTEGRATION-02`；baseline commit=`35ca52da`。
- observation：上一版 EAS-VGGT 已继承正结果，但主终点仍偏 Actor-box 条件 median，no-return/概率遮挡被延后；强调物理与外观隔离，未要求两套状态在可见深度/轮廓/遮挡上的独立对应证据；有效多基座、强深度适配与充分独立数据也未成为必需项。用户补充调研指出额外 LiDAR、转换误差、PSNR 不变和“即插即用”的混杂。
- cause/boundary：接口正确与旧条件任务机制不能直接推出完整传感器过程或通用视觉几何适配。该风险来自计划证据范围不足，不是本轮训练失败，也不否定 M8/M39/M22/M28/M49 已有结果。
- mitigation：revision 2 定义明确主任务，增加两基座分层诊断、同信息校正/融合/CAPA/scalar/LiDAR-only 对照，新增有序表面事件与真实 no-return 监督、物理—外观对应及第三来源冻结迁移；资源不限制研究范围。
- anti-repeat：不能把 F22/M35 的长前尾体密度改名为有序事件；必须辨别 surface support/事件定位，并对照单 hazard 和 M39+no-return。不能将 F/O/U、CDF 单调、质量守恒、PSNR 不变或增加参数单独写成方法有效。跨基座新训 projector/校准须标明，不能写零更新通用插件。
- resolution criteria：实际完成四组证据 A–D，在相同测量信息下确认专门机制增量、表面与传感器语义正确、物理/外观有独立真值对应，以及独立场景/来源的明确泛化范围；仅改计划不解除本风险。若结果只能支持较窄任务，相应收窄论文主张。
- evidence=`docs/WORLDSIM_V7_2_EAS_VGGT_RECOVERY_PLAN.md` revision 2、`docs/WORLDSIM_V7_2_FOUNDATION_ADAPTATION_RESEARCH.md`、历史 `V71-F20/F22/F23/F24/F37–F43/F47` 与 D0 W0–W4；本轮无新科学质量读取、训练或 shutdown。

下一可用统一失败编号：`V71-F68`；下方总览和 next-ID 属于历史记录。

## 当前路线总览（2026-09-07，基线 debe8697）

| 范围 | 最新解释 | 入口 |
|---|---|---|
| V7/V7.1 | canonical compiler、M8/M39、物理/外观/轨迹分工为 EAS-VGGT 起点；保留外域、容量和算子负结果 | EAS-VGGT plan 第 2/8 节，历史 task/条目 |
| V7.2 外部补全 A1 | 该候选未越过 G1；不是 EAS-VGGT 的科学拒绝 | V71-F63 |
| V7.2 过早整体收口 | 缺失 B 实验不能解释成 false；旧修复队列被新方向替代 | V71-F64 |
| V7.2 当前恢复 | EAS-VGGT E1 running；双基座诊断和数据合同已有结果，E2–E5 pending | V71-F65、RESEARCH_STATUS 文首 |

<a id="v71-f65-eas-vggt-direction"></a>
## V71-F65 — recovery 继续外部补全/神经 LiDAR，未继承 EAS 已有机制（2026-09-07）

- category=`governance/research_scope`；status=`resolved`（方向与文档已纠正，效果未验证）；task=`WS-V72-E0-EAS-VGGT-REPLAN-01`；evidence baseline commit=`debe8697`。
- observation：task-first 将对象补全/full neural LiDAR 设为选路中心，先前 recovery 又安排 NKSR、LiDAR-RT、A2/B1；V7/V7.1 的 canonical surface、continuous evidence/categorical return、physical/appearance ownership 和 SE(3) 只作为对照或外围资产。用户明确指出方向偏离，要求 EAS-VGGT。
- cause：把下游任务宽度与替换成外部完整模型混同，把局部失败扩大为原表示应被替换；上一轮主要修正决策完整性，未重新核对研究对象与正证据继承。此为研究规划失误，不是新算法实验失败。
- correction：新计划围绕三机制与 VGGT→canonical EAS 的学习接口；旧 A/B、R1–R7 队列不再执行。M8/M39/M22/M28/M49 等原结果及 M43、W0–W4、A1 负结果保留；外部模型降为可选对照/能力记录，不作为新主线前置。
- prevention：每次新计划先列“已有正结果→原范围→新学习增量→反证对照”；先检索再迁移必须服务当前问题，不能变成外部模型排队；结构正确、学习增益、泛化和完整交付分别判断。README/status/failure/experiment 与旧入口同步，避免旧队列复活。
- reopening：E1 完成 RGB/投影/缓存，E2 用同几何 scalar/F/O/U 和 VGGT feature 控制辨别收益，E3/E4 做真实外观与刚体组合；科学晋级取决于新证据。不是恢复 M40/F41 家族权重扫参，也不修改旧阈值或数据曝光身份。
- evidence=`docs/WORLDSIM_V7_2_EAS_VGGT_RECOVERY_PLAN.md`、`paper/results/eas_evidence.json`、`paper/sections/05_experiments_appendix.tex`、`motion_proj/worldsim_v71/authority_contract.py`；本轮 new training/new target quality/source-test/external-test read=false，shutdown=false。

下一可用统一失败编号：`V71-F66`。历史 next-ID 只反映当时状态。

## V71-F64 — 缺失 B 实验被机械当作失败，导致整体过早收口（2026-09-07）

- category=`research_decision_completeness`；status=`superseded`（解释已纠正；旧 runner 修复任务被 EAS-VGGT 主线替代）；task=`WS-V72-R0-RESEARCH-PLAN-01`；证据基线 commit=`e6bb9a971114e7b60f234f9c2cf4766ee784973a`。
- observation：原总计划要求 A/B 都完成同信息量强基线比较后选路。实际 B 仅完成 LiDAR4D capability；`scripts/decide_worldsim_v72_d1.py::main` 不接收 B 结果，固定 `route_b_pass=False`，且 A dev 失败时直接给出 `close_method_claim`。缺实验被解释成双方科学失败，随后文档整体收口并关机。
- cause：决策实现混合“比较未完成”和“候选已被否定”；停止线适用范围超出已经完成的实验。不是 A1 指标算错，也没有证据支持路线 B 已通过。
- correction：保留 `V71-F63` 的 A1 负结果、旧 canonical gate 与论文；在当前状态和恢复计划中撤回“B 已被证伪／所有可执行工作完成”的解释。用户已要求遇卡点先检索再迁移，并已重新开机。
- prevention：未来决策显式接收各路 `comparison_complete` 与结果证据；缺证据只产生待完成任务，不能默认 false 后触发整体结束。每个实质卡点先查官方论文／代码／issue，再做有依据迁移；不以同一候选失败代替路线审查。
- reopening（2026-09-07 更新）：允许在既有 dev 上继续机制研究并保留曝光身份；原 NKSR/LiDAR-RT/A2/B1 队列及 R1 决策器修复前置由 V71-F65 替代。新路线从 EAS-VGGT E1 开始；旧 D1 不参与决策。缺证据不能当失败的原则继续有效，旧阈值与 run 不追溯修改。
- evidence=`scripts/decide_worldsim_v72_d1.py`、`docs/WORLDSIM_V7_2_RESEARCH_FIRST_RECOVERY_PLAN.md`、原计划第 4/10B/11 节；新训练／新 target quality read=0；旧后续 task=`WS-V72-R1-DECISION-AND-DIAGNOSIS-01`，status=`rejected`（调度范围被替代，非算法拒绝）；当前见 `WS-V72-E1-VGGT-EVIDENCE-IO-01`。

下一可用统一失败编号：`V71-F65`。下方历史段落中的 next-ID 只反映当时状态。

## V71-F63 — A1 观测约束补全未越过干净 dev 的 G1 TSDF 前沿（2026-09-07）

> 2026-09-07 适用范围更新：A1 的负结果保留；关于 B 与整体停止的解释由文首 V71-F64 纠正。用户已要求先检索再迁移，后续可在保留 dev 暴露身份的前提下开发不同机制；本条旧 prevention 不构成永久禁止研究。

- category=`scientific_route_rejection`；status=`closed_by_preregistered_d1_stop_rule`；task=`WS-V72-D1-A-DEV-GATE-01`。
- run=`run://worldsim_v72/WS-V72-P2-A1-OBSERVATION-CONSTRAINED-DEV-01/20260906T230000Z__a1-observation-dev-s7210-r1`；data=4 个全依赖链隔离 dev logs、501 Actors、641,930 held-out rays；冻结 density cap=`512`。
- observation：G1 TSDF 的 CD/F-score/early/hit=`.178599m/.750319/.411462/.549722`，A1 anchored=`.194976m/.714542/.439344/.522150`。候选的 relative CD reduction=`-9.169%`、absolute F gain=`-3.578pp`、early delta=`+2.788pp`、hit delta=`-2.757pp`；主效应与两项副作用均失败。纯 A1 相对 G3 只有 early `-.427pp`、hit `+.184pp` 的小变化，同时 CD/F-score 略退化。
- cause boundary：观测 hit/free loss 在训练目标上收敛，但没有产生优于简单 build-only TSDF 的表面；75% TSDF anchor 能恢复部分几何，却仍同时损失完整性、early 和 hit。结果拒绝当前 A1 候选及其几何贡献，不证明所有对象补全或所有观测约束方法无效。
- route consequence：路线 B 只有 LiDAR4D 公共场景 capability，没有冻结规则要求的 matched-protocol method gain，故 D1 route A/B 均不通过，decision=`close_method_claim`。route-select、source-test、external-test、P3、P4 不解锁。
- prevention：不得在同一 dev 上扫描 anchor fraction、loss weight、密度、阈值或追加第三路线；任何新结构必须作为新版本重新定义问题并建立新的依赖链隔离证据，不能把本次称为“几乎通过”。
- evidence=`.../D1_DEV_GATE.json`、`docs/WORLDSIM_V7_2_D1_NEGATIVE_CLOSEOUT.md`；source/external final read=false。

下一可用统一失败编号：`V71-F64`。

## V71-F62 — clean split 物化假定 metadata-only 日志的 keyframe 已在 raw root（2026-09-07）

- category=`clean_data_io_availability`；status=`resolved_selective_official_archive_extraction`；task=`WS-V72-P2-CLEAN-ACTOR-DATA-01`。
- failed run=`20260906T222000Z__clean-dev-actor-v2-s0-r1`；首个 dev scene 在读取第一个缺失的 `samples/LIDAR_TOP/*.pcd.bin` 时退出，0 Actor bundle、0 target artifact、0 metric。冻结角色与 metadata index 已读取，但没有打开任何点云质量；source/external final read=false。
- root cause：P0 的 850-scene metadata 完整，而本地 raw root 只保留历史已消费 scene 的 LiDAR；metadata 可用不等于 candidate sensor payload 已物化。
- resolution：保持冻结的 4 dev / 3 route-select / 3 unopened source-test candidate 不变，只从 `/root/autodl-pub` 官方十个 nuScenes trainval blob 分卷中选择性提取 dev/route 的 3,882 个 keyframe LiDAR；不提取 source-test candidates。成功 I/O 后以新 immutable run 重启 builder。
- prevention：clean role 物化前先按 metadata 生成所需 member 清单并完成 payload availability gate；数据缺失不得换日志或解释为模型失败。

下一可用统一失败编号：`V71-F63`。

## V71-F61 — LPIPS AlexNet 权重的 Python 首次下载连接停滞（2026-09-07）

- category=`external_metric_checkpoint_transport`；status=`resolved_prefetched_same_upstream`；task=`WS-V72-B0-LIDAR4D-CAPABILITY-01`。
- failed run=`20260906T192156Z__lidar4d-kitti360-f4950-s0-r1`；LiDAR4D 在构造官方 LPIPS meter 时通过 `torchvision` 首次下载 `alexnet-owt-7be5be79.pth`，连接超过 2 分钟仍停在握手且缓存为 0 bytes，故以 SIGTERM 收口。run wall=`142.20s`、峰值 GPU=`488MiB`、平均利用率=`0%`，没有 checkpoint 或 metric。
- exposure：失败发生在 KITTI-360 dataloader、训练、验证和测试构造之前；Motion-Proj source/external final read=false。
- resolution：使用同一官方 `download.pytorch.org` URL 预取相同权重到冻结 `TORCH_HOME`，bytes=`244,408,911`、SHA-256=`7be5be79...dee02`；不修改 LiDAR4D、LPIPS、模型、数据、split、seed 或 30k 协议，r2 继续。
- prevention：含第三方 metric checkpoint 的外部基线在正式 run 前记录 URL、bytes、SHA-256 并完成缓存；传输失败不得解释为模型能力失败。

下一可用统一失败编号（该段记录时）：`V71-F62`；当前编号以文首为准。

