# 历史原始记录 015

> 冻结历史，不是当前执行规则。当前规则见 [scaling law](../../../auto-research_scaling_law.md) 与 [AGENTS](../../../AGENTS.md)。
> 原文件：`docs/RESEARCH_FAILURES.md`；迁移前提交 `abbd431c`。原有相对链接按原目录解释，证据路径原样保留。
> 按索引读取相关小节；旧哈希、过度门控和资源指令均不重新生效。

## V7.1 paper literature boundary — related mechanism is not implemented evidence（2026-09-05）

主稿不得因引用DynamicVGGT/DeGO/SelfOccFlow而声称当前模型具有scene-flow、non-rigid deformation或future
forecasting；不得因Gau-Occ两阶段联合优化而推断未公开的completion梯度路径。相关工作只用于支持状态拆分和
supervision-first设计，当前实证仍限M8/M39/M49/M51；M43在handoff时incomplete/no-verdict。纯写作压缩无新
failure；next ID仍=`V71-F50`。

## V7.1 M51 outcome note — soft error improvement cannot certify hard support（2026-09-05）

M51在相同support上确认：新增hard early rays中`38.03%`的smooth absolute depth error反而改善；加入正式
full-anchor/voxel deployment后仍为`36.54%`。因此禁止再以更低smooth first-depth L1/Huber、更多samples或frame
平均包装为literal safety；任何保证都必须直接约束ordered support/CDF margin并重新训练，而不能用M51的GT边界
事后筛ray。M51精确解释并关闭`V71-F49`，无新failure；next ID仍=`V71-F50`。

## V7.1 M51 pre-registration — explanation is not a hard-event selector（2026-09-05）

M51只量化smooth expected depth与hard minimum的非蕴含，不按soft/hard disagreement删除ray、primitive、Actor或
frame，不改变0.20m beam/depth tolerance、0.06m voxel或renderer参数。即使精确解释`V71-F49`，M50 verdict仍为
rejected，且不得由diagnostic选择M8/M50 mixture。当前next failure ID仍=`V71-F50`。

## V71-F49: frame-balanced smooth first-return worsens hard temporal boundary（2026-09-05）

- run=`run://worldsim_v71/WS-V71-M50-FRAME-BALANCED-FIRST-RETURN-01/20260905T060000Z__m50-frame-balanced-first-s71150-r2`；
- symptom=按target frame等权后，worst-frame early all/hazard/clear恶化`1.135/1.286/0.887pp`，aggregate hazard
  early恶化`0.590pp`，两个冻结decision均失败；
- retained evidence=Chamfer改善`0.871mm`、all hit `+0.023pp`，说明optimizer能移动surface，但不是physical Pareto；
- mechanism=frame reweighting修正采样测度，却没有消除smooth differentiable depth与hard earliest-return event的
  surrogate gap；geometry/physics gradients在`83.2--86.6%` batches冲突，且free loss最终略升；
- literature response=DynamicVGGT/DeGO依靠显式scene-flow或rigid/non-rigid state supervision；当前vehicle corpus无
  non-rigid target，不能用新deformation capacity掩盖该surrogate failure；
- resolution=关闭frame-balance fine-tuning，不调frame cap、weight、margin、seed或epoch；不加motion/hazard/visibility
  input，不替换M39/M43；未来若研究temporal evolution，必须有独立trajectory/scene-flow target和状态；
- claim impact=M8 frame-balanced coverage仍成立，但不升级为per-frame hard-return consistency。

下一可用编号：`V71-F50`。

## V71-F48: M50 loader omitted frozen M5 runtime centers（2026-09-05）

- run=`run://worldsim_v71/WS-V71-M50-FRAME-BALANCED-FIRST-RETURN-01/20260905T054500Z__m50-frame-balanced-first-s71150-r1`；
- symptom=reference-child reconstruction raised `KeyError: m5_centers_t`；
- cause=M8 checkpoint stores the frozen M5 run path, while prepared corpus actors do not persist M5 runtime tensors；M50 loaded
  the child head but omitted the same frozen-base reconstruction used by M8；
- exposure=failed before optimizer/training/holdout/metric; external/M43 quality read=false；
- resolution=load checkpoint-declared M5 model read-only and recreate `m5_centers_t` before M8 child prediction；same config,
  loss, frame groups, seed and decisions；r2 is the sole scientific trial；
- anti-repeat=all future M7/M8-derived loaders must reconstruct their declared parent coordinate base before `_predict`。

下一可用编号：`V71-F49`。

## V7.1 M50 pre-registration — temporal balance is GT supervision, not a motion/visibility head（2026-09-05）

M50只改变target-frame physical loss的采样测度，不增加time、velocity、moving/static、hazard、category、image或
visibility输入；rigid trajectory保持read-only transform。禁止把per-frame结果用于删frame/Actor，禁止在结果后改
frame grouping、loss weight、margin、seed或epoch。若worst-frame primary与aggregate Pareto guard不能同时成立，关闭
该方向；不得用M50替换冻结M39→M43 external candidate。当前next failure ID仍=`V71-F48`。

## V7.1 paper structure boundary — legacy V7 evidence is supplement-only（2026-09-05）

旧validity/hazard/reliability与P1--P22长链不得重新混入V7.1主贡献，避免把post-hoc selection、nonlearned AV2
operator或早期M18/M21 development candidate包装成M39 learned transfer。完整材料保存在可编译supplement，不等于
被删除或否认；主稿只保留与“GT supervision→geometry、producer evidence→termination、typed ownership、M49 boundary”
直接相关的证据。原V7 supplement正文必须保留于`supplement_v7_legacy.tex`，不得因新driver覆盖而遗失。
M43完成前仍禁止补写external方向。纯结构调整无新failure，next ID仍=`V71-F48`。

## V7.1 paper anti-overclaim refresh — M43完成前不预写泛化结论（2026-09-05）

摘要、引言、结论已删除过时的M18/M21 learned-external叙事，只陈述M8/M39 source-domain结果、M22/M28
factorization和M49解析边界。M43尚未完成时禁止引用partial metric、预写提升/退化方向，或将旧nonlearned compiler
AV2结果冒充learned transfer；M43无论通过或失败都禁止用AV2 target adaptation补救。M49仅证明normalized-mixture
attenuation的符号边界，不推出collision-free、domain-invariant或real-road-safe。此次纯写作同步无新failure；next
failure ID仍=`V71-F48`。

## V7.1 synthesis boundary — do not recombine separated claims（2026-09-05）

论文主张必须保持canonical geometry、ray termination、rigid pose、static Background、appearance和hazard authority分离。
M22/M28的结构保证不等于几何正确，M39 source pass不等于opacity/collision probability，M49解释不等于deployment
gate，M7/M8 source Pareto不等于AV2 generalization。M43完成前不得写cross-domain数值；若失败不得target-adapt救结果。
详细矩阵见`WORLDSIM_V71_RESEARCH_SYNTHESIS.md`。next failure ID仍=`V71-F48`。

## V7.1 M49 outcome note — attenuation is not monotone-safe in a normalized mixture（2026-09-05）

M49在99,208条冻结ray上确认`∂C/∂log w_j=r_j(C_j-C)`：uniform child attenuation有害/安全方向分别
`58.85/30.96%`，M48实际adverse pressure在`65.45%`占优，finite CDF在`59.32%`上升；一阶符号预测
exact change达`95.73%`。禁止把`C_j>C`转成primitive/ray选择器，因为该条件使用目标边界且只解释已暴露
development ray；它是M48的anti-repeat定理，不是oracle filter。`V71-F47`关闭，next ID=`V71-F48`。
论文图只展示解析sign与实际finite change的总体比例；不得从图中读取阈值或用于选择case。
有限衰减式证明同一component/family的错误符号不能由opacity/identity/temperature幅度修复，纳入anti-repeat边界。

## V7.1 M49 pre-registration note — an analytic boundary, not a rescue gate（2026-09-05）

M49只核验`∂C/∂log w_j=r_j(C_j-C)`与M48实际attenuation的CDF符号；不得根据结果选择primitive、ray、
Actor或stratum，不得把derivative正负变成部署gate，不得重训M48。即使线性分解精确解释失败，也只能形成
安全边界和anti-repeat依据，不能把M48升级为supported。当前next failure ID仍为`V71-F48`。

## V71-F47: supervised bounded child visibility transfers risk under categorical normalization（2026-09-05）

- run=`run://worldsim_v71/WS-V71-M48-SUPERVISED-CHILD-VISIBILITY-MEASURE-01/20260905T114500Z__m48-child-visibility-s71148-r1`；
- evidence=safe NLL下降但early probability上升、hit probability下降；holdout visibility=`0.598`；相对M39
  all/hazard early=`+0.347/+0.483pp`，仅clear=`-0.322pp`，decisions=`1/3`；相对M45又损失`1.953pp` hit；
- root cause=即便visibility在训练/部署同一joint measure内且只能连续attenuate，归一化CDF的响应符号仍取决于
  被衰减component自身pre-boundary mass相对全局CDF的位置；统一物理descriptor无法保证每条ray同向安全；
- anti-repeat=V7.1 plan明确关闭`visibility head v2`，此后禁止调identity、initial visibility、hidden、lr、epoch、
  seed或加入hazard/motion/category输入；禁止把continuous visibility阈值化为filter；
- next=M49只推导并核验attenuation derivative的解析符号边界，不训练新renderer。下一failure ID=`V71-F48`。

## V7.1 M48 pre-registration note — visibility is supervised renderer state, not a filter（2026-09-05）

M48只允许completion child的ray-conditioned visibility进入训练和部署完全相同的joint measure；M8/M11 geometry与
M35/M38 authority固定。visibility只能连续衰减、不能删除primitive，且不读取GT endpoint/depth、hazard、motion、
category或image。GT只通过not-early/hit区间loss监督。禁止把M47 incidence区间用作rule，禁止loss/seed/hidden/
identity/initial visibility sweep。若三strata不能同时改善，登记`V71-F47`并关闭局部visibility支线。

## V71-F46: local oriented support causes heterogeneous ray-label transport（2026-09-05）

- run=`run://worldsim_v71/WS-V71-M47-MOTION-PROVENANCE-INCIDENCE-DIAGNOSIS-01/20260905T111500Z__m47-physical-factor-diagnosis-r1`；
- evidence=all early净`-0.048pp`掩盖`2.996pp`新增与`3.044pp`消除；hazard moving改善`-0.015pp`，
  quasi-static却`+0.809pp`；near-normal `+0.887pp`但grazing `-0.832pp`；KEEP/projected均改善；五个actor
  correlation绝对值均`<=0.100`；
- concentration=hazard quasi-static的净`+62`条early由1 trailer + 1 truck的`+67`主导，6 cars为`-5`；
  不是可泛化的motion二分或单一incidence规律；
- root cause=M46的局部normal/thickness改变全局categorical ordering，产生跨ray/类别的label transport；canonical
  shape、visibility/termination与temporal deformation仍未在表示和supervision中独立；
- anti-repeat=禁止按moving、hazard、KEEP/PROJECT、incidence或category后处理/删点；禁止继续调normal/thickness、
  event权重、bin、seed。若继续Gaussian表示，可见性必须成为joint return measure中的训练变量；动态分支必须由
  trajectory/scene-flow监督而非hazard标签定义；
- claim impact=M45--M47均不进external，M43仍为唯一冻结candidate。下一failure ID=`V71-F47`。

## V7.1 M47 pre-registration note — separate physical factors before changing the paradigm（2026-09-05）

M45/M46的hazard退化尚不能归因于“动态”或“入射角”。M47冻结两个renderer，复用M31的独立物理编译结果，
只作motion/provenance/incidence描述性分解；禁止将hazard/moving/KEEP-PROJECT标签输入模型，禁止按诊断结果删点或调阈值，
禁止把相关性写成因果。若moving集中，下一步显式分离rigid pose与local shape；若provenance集中，先修GT producer；
若grazing集中，则把visibility/incidence写进supervision；否则升级为completion/scene-flow factorization。当前failure ID仍为
`V71-F46`。

## V71-F45: CDF-supervised orientation improves clear rays but transfers risk to hazard rays（2026-09-05）

- run=`run://worldsim_v71/WS-V71-M46-CDF-SUPERVISED-ORIENTED-SUPPORT-01/20260905T104500Z__m46-cdf-oriented-s71146-r2`；
- evidence=相对M39 all/clear early=`-0.048/-0.591pp`、all hit=`+4.320pp`，但hazard early=`+0.062pp`，
  decisions=`2/3`；且hazard退化超过未训练M45的`+0.034pp`；
- optimization=safe/hit NLL均下降，early probability下降、hit probability上升，无NaN/OOM；不是train--renderer
  mismatch、梯度或资源失败；
- root cause=统一GT interval loss在固定center/tangent/authority、仅normal/thickness可动时仍产生stratum-dependent
  redistribution；local planar support无法单独解释hazard ray的global ordering，可能还含motion/pose/incidence差异；
- anti-repeat=不调event/geometry权重、thickness bound、normal residual、epoch/seed/bin/median，不向surface head加入
  hazard标签；冻结诊断motion/provenance/incidence后再决定是否需要显式dynamic/static层；
- claim impact=M46不进external，M43仍是唯一冻结cross-domain candidate；oriented支线关闭。下一failure ID=
  `V71-F46`。

## V7.1 M46 engineering note — missing fixed branch factor before training（2026-09-05）

M46 r1在任何optimizer step/quality read前因config缺少M11 helper所需`branch_factor`退出；该值由M8/M11架构固定
为4，不是可调参数。补齐后以r2重跑；r1不登记`V71-F45`，不得把它解释为模型失败。

## V7.1 M46 pre-registration note — only support orientation may absorb CDF supervision（2026-09-05）

M46不允许M40式authority family-mass shortcut或M8 geometry/scale compensation：point encoder、slots、hidden head、
center、tangent、两套F/O/U head全部冻结，只有M11 final normal/thickness输出行可更新。GT直接定义not-early与hit
interval，hazard标签不进模型。若任一三门失败登记`V71-F45`，不调loss权重、event tolerance、thickness bound、
epoch/seed/bin/median；不得替换已运行M43 candidate。

## V71-F44: frozen M11 orientation does not preserve hazard early safety under M39 composition（2026-09-05）

- run=`run://worldsim_v71/WS-V71-M45-ORIENTED-CATEGORICAL-SURFACE-MEASURE-01/20260905T100000Z__m45-oriented-categorical-r1`；
- evidence=相对M39 all/clear early=`-0.019/-0.280pp`且all hit=`+3.868pp`，但hazard early=`+0.034pp`，
  decisions=`2/3`；
- representation health=normal thickness=`0.0200m`、anisotropy=`8.10×`，相对unit baseline hit=`+6.040pp`，无
  数值/资源问题；planar support对endpoint集中有效；
- root cause=M11 normal/thickness是按earliest hard intersection训练，迁移到categorical measure后虽改善local
  endpoint likelihood，却未被该部署分布的global multi-primitive CDF ordering直接约束；hazard ray仍有轻微前移；
- literature response=Geometry Field Splatting（CVPR 2025）支持planar Gaussian geometry field，但kernel形状必须
  与实际renderer同构优化。后续只可在train split用GT not-early/hit interval训练normal/thickness，M8 centers/tangent
  与M39 authority冻结；
- anti-repeat=不调M11 thickness、normal residual、scale、bin、median或seed，不因巨大hit增益放宽worst-stratum
  gate；M45不替换已冻结M43；
- claim impact=oriented categorical是有价值但尚未安全通过的next-gen表示；下一failure ID=`V71-F45`。

