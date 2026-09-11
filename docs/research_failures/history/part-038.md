# 历史原始记录 038

> 冻结历史，不是当前执行规则。当前规则见 [scaling law](../../../auto-research_scaling_law.md) 与 [AGENTS](../../../AGENTS.md)。
> 原文件：`docs/RESEARCH_FAILURES.md`；迁移前提交 `abbd431c`。原有相对链接按原目录解释，证据路径原样保留。
> 按索引读取相关小节；旧哈希、过度门控和资源指令均不重新生效。

### V67-F34 — P46 anchored hybrid cross-cohort new-H replication候选

- 分类：`algorithm/cross-cohort-task-condition-replication`；状态：`resolved_by_cross_cohort_horizon_replication`。
- 方法：P44结构/anchor/loss不变；新增P6R-H1.5 development domain；P10R4-H1.5 target首次物化并作budget1/3 read。
- 判定：相对P31 `>=+0.005`、exact、minimum group `.50`、至少6/8 scenes。
- 结果：H1.5 eligible=`1077/1152`；exact=`353/353`、coverage/min group=`.666667/.583333`；
  reduction=`.683908`，相对P31=`+.011489`、相对fixed=`+.431285`，8/8 scenes，4/4 gates。
- 边界：source cohort曾在H2训练，但H1.5 task target未见；不作fresh population claim，不扫参。下一编号=`V67-F35`。

### V67-F35 — P47 frozen cross-cohort anchored nested replication候选

- 分类：`algorithm/cross-cohort-nested-replication`；状态：`closed_negative_high_budget_regression`。
- 方法：P46/P31/P20全冻结；P10R4-H1.5 quarter/half strict nesting，quarter anchor回退P20。
- 判定：两端exact、strict nesting、minimum group `.50`、相对各自P31非退化、至少6 scenes。
- 结果：low delta=`0`，high delta=`-0.032572`；exact/nesting/group/8 scenes通过，高预算非退化失败。
- 结论：low anchor可迁移，但full-residual high endpoint不能跨cohort稳定；单侧anchor high family关闭。
- 边界：第二cohort新H结构复制，无训练/refit/sweep。下一编号=`V67-F36`。

### V67-F36 — P48 double-anchored interior-budget adapter候选

- 分类：`algorithm/endpoint-preserving-residual-adapter`；状态：`closed_negative_below_frozen_gain_gate`。
- 调研迁移：Residual Adapters以共享主干+小域适配器服务多domain；Net2Net强调function-preserving变换。P48保持两端
  冻结函数不变，只在budget内部激活adapter。
- 方法：amplitude在`.25/.50`为0，在`1/3`为1，分段线性；11 domains×3 budgets训练；P31 allocator冻结。
- 数据：新增P10R4-H1.5 development；P10R2-H1.5 target首次物化并作peak read。
- 结果：P48/P31 reduction=`.742759/.740902`，delta=`+.001857`，exact/group/8 scenes通过但未达冻结`+.005`。
- 结论：双端function-preserving合同成立，内部adapter增益不足；不降门、不扫两端/peak/amplitude/model/loss/gate。
  下一编号=`V67-F37`。

### V67-F37 — P49 Fishr-inspired domain-gradient consistency候选

- 分类：`algorithm/domain-gradient-consistency`；状态：`resolved_by_new_horizon_task_condition`。
- 调研迁移：Fishr（ICML 2022）以跨域梯度统计一致性提升domain generalization；P49只迁移到小adapter末层的
  normalized domain update direction dispersion，不声称实现完整per-sample gradient-variance Fishr。
- 方法：P48双端anchor/model/P31 allocator不变；加入固定`.01`末层方向离散惩罚，12 domains×3 budgets训练。
- 数据：新增已消费P10R2-H1.5 development；P3C-H1.5首次物化=`710/864` eligible并作一次budget1/3 read。
- 结果：exact=`236/236`、group=`.666667`；P49/P31 reduction=`.710322/.695815`，delta=`+.014506`；5/5
  scenes，4/4 gates。固定轻量gradient一致性训练得到正式支持。
- 防重复：不扫gradient weight、layer、anchor、peak、model、loss或gate。下一编号=`V67-F38`。

### V67-F38 — P50 frozen second task-condition replication候选

- 分类：`confirmation/frozen-cross-condition-transfer`；状态：`resolved_by_second_new_horizon_condition`。
- 方法：P49/P31/P20冻结；P2V-H1.5首次物化=`774/864` eligible、72 cases，budget1/3一次读取。
- 判定：exact、minimum group `.50`、相对P31 `+.005`、至少5 scenes；无训练/refit/weight/anchor/gate sweep。
- 结果：exact=`252/252`、group=`.541667`；P49/P31 reduction=`.789696/.739907`，delta=`+.049789`；
  6/6 scenes、4/4 gates。冻结方法跨第二新H条件复制。
- 边界：source cohort已消费且H2进入development，但H1.5 target未用于P49训练；不作fresh population claim。
  下一编号=`V67-F39`。

### V67-F39 — P51 large-cohort new-horizon gradient replication候选

- 分类：`algorithm/large-cohort-gradient-consistency`；状态：`resolved_by_large_cohort_new_horizon`。
- 方法：P49 gradient penalty/anchor/model/loss/gates不变，只将已消费P2V-H1.5滚入第13个development domain。
- 数据：P6E 16-scene H1.5 target首次物化=`2049/2304` eligible、192 cases，与GPU训练并行；budget1/3一次read。
- 判定：exact、四scene-group minimum coverage `.50`、相对P31 `+.005`、至少12 scenes。
- 结果：exact=`673/673`、group=`.50`；P51/P31 reduction=`.806000/.796088`，delta=`+.009912`；
  15/15 evaluable scenes，4/4 gates。
- 防重复：不扫gradient weight/anchor/peak/model/loss/group/gate。下一编号=`V67-F40`。

### V67-F40 — P52 frozen below-range horizon extrapolation候选

- 分类：`confirmation/frozen-horizon-extrapolation`；状态：`resolved_by_short_horizon_extrapolation`。
- 方法：P51/P31/P20冻结；P10V-H0.8首次物化=`694/864` eligible，低于所有训练H。
- 结果：exact=`229/229`、group=`.521739`；P51/P31 reduction=`.761914/.680754`，delta=`+.081161`；
  6/6 scenes，4/4 gates。无训练/refit/sweep。
- 边界：source cohort已消费；只称task-condition extrapolation。下一编号=`V67-F41`。

### V67-F41 — P53 jointly unseen budget + horizon gradient hybrid候选

- 分类：`algorithm/joint-budget-horizon-generalization`；状态：`resolved_by_joint_unseen_condition`。
- 方法：P51 gradient/anchor/model/loss不变；training budgets加入固定`.40`，形成14 domains×4 budgets。
- 数据：P10X-H0.8首次物化=`662/864` eligible、72 cases，与GPU训练重叠；formal budget=`.375`未见。
- 判定：exact、minimum group `.50`、相对P31 `+.005`、至少5 scenes。
- 结果：exact=`218/218`、group=`.541667`；P53/P31 reduction=`.733916/.724912`，delta=`+.009004`；
  6/6 scenes，4/4 gates。
- 防重复：不扫budget集合、gradient weight、anchor、model、loss或gate。下一编号=`V67-F42`。

### V67-F42 — P54 frozen second-cohort joint-condition replication候选

- 分类：`confirmation/frozen-joint-condition-transfer`；状态：`resolved_by_second_joint_cohort`。
- 方法：P53/P31/P20冻结；P4C-H0.8=`861/1152` eligible；formal budget `.375`，无训练/refit。
- 结果：exact=`282/282`、group=`.50`；P53/P31 reduction=`.830087/.775806`，delta=`+.054281`；
  8/8 scenes，4/4 gates。联合条件跨第二cohort复制。
- 边界：globally consumed source cohort；不作fresh population claim。下一编号=`V67-F43`。

### V67-F43 — P55 fixed-tail weight-averaged gradient hybrid候选

- 分类：`algorithm/flat-minimum-weight-averaging`；状态：`closed_negative_after_single_trial`。
- 调研迁移：SWAD（NeurIPS 2021）以flat minima缩小domain generalization gap；SWA（UAI 2018）沿训练轨迹平均权重。
- 方法：P53完全不变，只固定平均最后20%=1,200 checkpoints；不用validation选窗口，不改学习率或训练长度。
- 数据：P10R4-H0.8=`984/1152` eligible、96 cases；同一formal read加载冻结P53作method baseline。
- 判定：exact/group/scenes、相对P31 `+.005`、相对P53 `+.002`；不扫averaging start/schedule/gate。
- 结果：P55/P53/P31 reduction=`.688694/.698266/.694007`；相对P53/P31=`-.009572/-.005313`；
  exact/group/8 scenes通过但两decision gates失败，3/5 gates。
- 结论：固定末20% averaging退化；不改window/schedule，不继续weight-averaging family。下一编号=`V67-F44`。

### V67-F44 — P57 fixed-radius SAM gradient hybrid候选

- 分类：`algorithm/sharpness-aware-optimization`；状态：`closed_negative_after_single_recovery`。
- 调研迁移：SAM（ICLR 2021）优化邻域worst loss；ASAM（ICML 2021）提供scale-aware扩展。P57只用标准SAM，
  固定`rho=.05`，不尝试ASAM或radius sweep。
- 方法：P53 data/model/gradient/budgets/anchor/loss/seed/epochs不变；每epoch标准两步SAM。
- 数据：P10R2-H0.8=`1034/1152` eligible、96 cases；同read比较冻结P53。
- 判定：exact/group/scenes、相对P31 `+.005`、相对P53 `+.002`。失败则关闭sharpness优化family。
- 结果：P57/P53/P31 reduction=`.731922/.723709/.727373`；相对P53=`+.008213`，相对P31=`+.004549`，
  后者差`.000451`未过冻结门；4/5 gates。
- 结论：严格拒绝，不降门/扫rho/试ASAM；flat/sharpness optimization family关闭。下一编号=`V67-F45`。

### V67-F45 — P58 differentiable case-selective residual expert失败

- 分类：`algorithm/case-selective-mixture-of-experts`；状态：`closed_negative_after_single_trial`。
- 调研迁移：DSelect-k（NeurIPS 2021）提供连续可微expert selection；sparse MoE（ICLR 2017）按输入路由专家。
- 方法：P20 frozen base expert + P53形式residual expert；新增固定width8 sigmoid case gate；其余P53合同不变。
- 数据：P6R-H0.8=`868/1152` eligible、96 cases；同read比较P53/P31。
- 异步复现输入：P3C-H0.8=`695/864` eligible、72 cases；仅物化，P58通过前selection read=false。
- 判定：exact/group/scenes、相对P31 `+.005`、相对P53 `+.002`；不扫gate width/expert count/temperature。
- 结果：P58/P53/P31 reduction=`.777488/.774840/.797323`；相对P53=`+.002649`通过，但相对P31
  `-.019835`，scene non-increasing=`5/6`；3/5 gates。gate mean=`.904328`，多数case近全开。
- 结论：不扫gate结构/温度/top-k，不读取P59 replication quality；case-selective expert关闭。检索PRECOG、MotionLM与
  uncertainty-aware actor prediction后，下一步改变预测对象为given-`tau` Actor-state reliability。下一编号=`V67-F46`。

### V67-F46 — P60 trajectory-conditioned Actor-state reliability结果

- 分类：`prediction-object/trajectory-conditioned-actor-state-reliability`；状态：`supported_with_ranking_limitation`。
- 调研迁移：PRECOG（ICCV 2019）按受控Ego goal条件化多Actor forecast；MotionLM（ICCV 2023）支持条件rollout；
  CVPR 2018证明trajectory uncertainty可指示实际预测误差。P60迁移为当前单卡可训练的小型reliability estimator。
- 对象：Actor常速度H-step endpoint error乘`exp(-predicted tau separation/6m)`；只使用logged dense tracks监督，
  不声称counterfactual response。
- Protocol：scene `%5!=0`、H `.8/1.5s`训练；scene `%5==0`、H `2.0s`confirmation；query MLP与同容量
  Actor-only baseline同read。三门=`Spearman .55 / MAE gain 10% / AUROC .75`。
- 锁：不扫width、history、exposure radius、error threshold、split或horizon；失败即改变Actor target/forecast family，
  不回到world-state authority补救。
- 结果：unseen-scene/H2 query Spearman=`.756794`、MAE=`.093437`、AUROC=`.960804`；相对Actor-only MAE
  降低`25.45%`，但Spearman低`.014367`。3/3原门通过；准确结论限于校准/事件识别，排序增益未成立。
- 下一步：P61只加固定pairwise term恢复排序；不改预测对象或P60门。下一编号=`V67-F47`。

### V67-F47 — P61 fixed pairwise Actor-reliability recovery失败

- 分类：`algorithm/pairwise-ranked-actor-reliability`；状态：`closed_negative_after_single_trial`。
- 方法：P60 query head增加weight `.10`、temperature `.05`、gap `.02`、shifts `[1,17,257]`的pairwise loss；
  Actor-only head仍只做Huber，其余特征/架构/epoch exact。
- Protocol：development split改为scene `%5==1`、H2；P60三门加Spearman delta over Actor-only `+.01`。
- 结果：Spearman `.755004/.740135`、delta=`+.014869`，AUROC `.945763`；但MAE
  `.156003/.117128`，query退化`33.19%`；3/4 gates。
- 结论：排序恢复但尺度失配；不扫pair配置。调研后只允许一次train-only monotone calibration恢复。下一编号=`V67-F48`。

### V67-F48 — P62 train-only monotone calibration恢复失败

- 分类：`algorithm/order-preserving-regression-calibration`；状态：`closed_negative_after_single_recovery`。
- 调研迁移：ICML 2018 calibrated regression支持model-agnostic recalibration；UAI 2025强调instance-wise monotonic map
  保持ranking。P62使用最小正斜率affine map，不使用非参数binning/isotonic complexity。
- 方法：P61模型/排序配置exact；训练完成后仅用training prediction/target最小二乘拟合`slope>0,bias`，confirmation
  只apply。第三development split为scene `%5==2`、H2，四门保持exact。
- 锁：不扫map/slope/pair配置；失败即关闭pairwise+calibration recovery。下一编号=`V67-F49`。
- 结果：map=`1.016524*x-.002156`；query/Actor-only Spearman=`.742362/.751193`、MAE
  ` .084206/.081360`、AUROC=`.951511/.950829`；rank和MAE两门失败，2/4 gates。
- 结论：校准近恒等，跨split conflict不是scale-only；pairwise+calibration family关闭。

### V67-F49 — P63 Rank-N-Contrast-inspired representation失败

- 分类：`algorithm/continuous-rank-contrastive-representation`；状态：`closed_negative_after_single_trial`。
- 调研迁移：Rank-N-Contrast（NeurIPS 2023）把continuous target order编码进representation，报告更好的regression与shift
  generalization。P63用固定三shift triplet surrogate预训练query encoder，不向scalar output直接加ranking loss。
- 方法：500 contrastive epochs；冻结encoder后1000 Huber head epochs；Actor-only baseline总计1500 Huber epochs。
  第四split scene `%5==3`、H2；四门exact，不扫contrastive配置。
- 结果：query/Actor-only Spearman=`.242097/.799512`、MAE=`.136035/.098264`、AUROC=`.820529/.961380`；
  1/4 gates，冻结representation不能被linear scalar head可靠读出。
- 结论：不解冻encoder/延长head/扫contrastive配置；family关闭。下一编号=`V67-F50`。

### V67-F50 — P64 plain-Huber Actor-reliability replication支持

- 分类：`replication/plain-actor-reliability`；状态：`supported_second_split`。
- 调研迁移：WACV 2020 actor motion uncertainty采用可部署的feed-forward预测并强调state、velocity、acceleration、heading；
  当前资产没有raster/map supervision，因此P64不虚构大backbone，直接复现P60的低容量state-query MLP。
- 方法：P60特征、plain Huber、架构、1500 epochs、三门exact；第五split scene `%5==4`、H2。
- 结果：query/Actor-only Spearman=`.769725/.424100`、MAE=`.092651/.107035`（降低`13.44%`）、
  AUROC=`.957408/.850937`；3/3 gates，27/32 scenes rank noninferior。
- 结论：P60 plain-Huber在第二split复现；不加辅助loss/calibrator。该population不是fresh confirmation。下一编号=`V67-F51`。

