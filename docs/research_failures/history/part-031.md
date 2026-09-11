# 历史原始记录 031

> 冻结历史，不是当前执行规则。当前规则见 [scaling law](../../../auto-research_scaling_law.md) 与 [AGENTS](../../../AGENTS.md)。
> 原文件：`docs/RESEARCH_FAILURES.md`；迁移前提交 `abbd431c`。原有相对链接按原目录解释，证据路径原样保留。
> 按索引读取相关小节；旧哈希、过度门控和资源指令均不重新生效。

### P290 additive-family confirmation freeze note — 无新增 failure

- P277 rows虽然已被旧 beta-LCB family消费，但 additive P284--P289 从未训练/选模/读取其 quality，复用边界已显式写入；
- P290 只在 P289 通过后做一次 frozen chain read，不以 confirmation 反向调 P284/P288/P289；
- 下一可用 failure id 保持 `V67-F198`。

### P289/P290 outcome note — 无新增 failure

- P289 P201 price/attained-fraction MAE=`.0267769/.0143254`、composite regret=`5.28e-6`、violations=0，2/2；
- 冻结后一次读取的 P290 P277-reuse confirmation 为 attained-fraction MAE=`.0122735`、composite regret=
  `1.85e-5`、violations=0，2/2；未据此反向调参或更换 cohort；
- P290 仅是 additive-family untouched reuse；P277 cohort 已被旧 beta-LCB P277 消费，不记为全项目 fresh；
- 下一可用 failure id 保持 `V67-F198`。

### V67-F198 — P291 symmetric L1 student破坏 adaptive teacher coverage

- canonical=`run://worldsim_v67/WS-V67-P291-CONTEXT-ADAPTIVE-ADDITIVE-LCB-SURFACE-01/
  20260831T214500Z__context-adaptive-additive-lcb-s0-r1`；
- symptom：teacher P201 coverage=`.92829/.84311/.73917/.65513` 接近 desired `.925/.85/.75/.65`，但 symmetric L1
  student coverage=`.79963/.73361/.67050/.59656`，max undercoverage=`.125366>.12`；fidelity 与 efficiency 两门通过，
  coverage门失败；
- root cause：MAE=`.0042137` 很小但未限制误差方向，LCB student 小幅系统上偏会被 simultaneous-horizon coverage
  放大；这与 NeurIPS 2023 报告的 distillation confidence exaggeration 一致；
- closed candidate：不再继续 symmetric L1、step/width/lr/threshold 扫描；P292 不得引用 rejected P291；
- structural recovery：P291R 冻结全部 adaptive teacher与三门，仅用 fixed lower-quartile pinball 做单侧蒸馏；
- 下一可用 failure id 为 `V67-F199`。

### V67-F198 resolution / P292 start note — 无新增 failure

- P291R P201 max undercoverage=`.051625`、mean conservatism=`.0158992 < P284 .0172214`，3/3；说明单侧
  distillation 在不放弃 adaptive efficiency 的情况下恢复 empirical coverage；
- P292 只编译冻结 P291R 到 P285 同构 soft-floor allocator，不改变 coverage对象、student loss或 P285两门；
- 下一可用 failure id 保持 `V67-F199`。

### P292 outcome / P293 group-dual start note — 无新增 failure

- P292 P201 budget MAE=`.0215751`、LCB-floor regret=`.0003612`，2/2，price/floor violations=0；
- P293 只编译冻结 P292 的 attainable-budget fixed64 dual，不改变 P291R coverage或 P292 utility定义；
- 下一可用 failure id 保持 `V67-F199`。

### P293 outcome / P294 adaptive tail-CVaR start note — 无新增 failure

- P293 P201 attained-fraction MAE=`.0255409`、group regret=`9.79e-5`，2/2，violations=0；
- P294 将冻结 P291R 作为 coverage layer，并以 P287 同一 empirical tail-CVaR定义训练 fixed64 primal；不修改
  P291R coverage、tail mass、eta/grid或 gates；
- 下一可用 failure id 保持 `V67-F199`。

### P294 outcome / P295 variable-cardinality start note — 无新增 failure

- P294 P201 budget MAE=`.0220461`、composite regret=`.00023215`，2/2，price/floor violations=0；
- P295 仅 warm-start同一架构到 sizes32/64/128并在48/96评估，不改变 P291R coverage或 P294 tail-risk定义；
- 下一可用 failure id 保持 `V67-F199`。

### P295 outcome / P296 variable-set dual start note — 无新增 failure

- P295 P201 budget MAE=`.0204663`、composite regret=`.00021336`，heldout sizes48/96 2/2，violations=0；
- P296 只编译冻结 P295 attainable-budget dual，不改变 P291R coverage、P294 tail-risk或 cardinality protocol；
- 下一可用 failure id 保持 `V67-F199`。

### P297 direct compiler freeze note — 无新增 failure

- P297 不是绕过 empirical constraint：teacher budgets仍由冻结 P295 + 同一 bisection target定义；变化只在推理
  路径从 dual→primal 两调用改为 fraction→budgets 单调用；
- fraction取负后进入 P295 positive price-rate axis，固定结构保证更多 attainable budget不产生更低 budgets；
- 仅在 P296 通过后训练，不以 P296/P297 结果调整 gate；下一可用 failure id 保持 `V67-F199`。

### P296 outcome / P297 direct training start note — 无新增 failure

- P296 P201 attained-fraction MAE=`.0240720`、composite regret=`1.58e-5`，heldout sizes48/96 2/2，violations=0；
- P297 按冻结协议启动，训练 teacher仍为 P295+bisection；P296仅作两阶段 baseline，不作为 P297训练组件；
- 下一可用 failure id 保持 `V67-F199`。

### P297 outcome / P298 interaction-aware start note — 无新增 failure

- P297 P201 attained-fraction MAE=`.0297261`、regret=`6.62e-5`、violations=0，2/2；direct compiler成立，
  但 constraint fidelity弱于 P296 two-stage `.0240720`，准确记录为可行性而非质量提升；
- P298 只加入 zero-gated one-block 4-head set attention并预注册必须严格优于 P297 attained MAE；不进行
  head/depth/width/step/gate sweep；
- 下一可用 failure id 保持 `V67-F199`。

### V67-F199 — P298 self-attention未改善 direct constraint fidelity

- canonical=`run://worldsim_v67/WS-V67-P298-ATTENTIVE-DIRECT-AUTHORITY-COMPILER-01/
  20260901T000000Z__attentive-direct-authority-s0-r1`；
- symptom：P201 attained MAE=`.0334487`，高于冻结 P297 baseline `.0297261`；通用 fidelity/regret门仍通过，
  comparative improvement门失败；size48/96=`.0310884/.0358091`；
- root cause：新增 pairwise capacity未直接约束 group mean budget，且 forward `.3904s` 高于 P297 `.3188s`；
  当前证据不支持 attention 是 direct authority fidelity 的必要结构；
- closed candidate：不扫 attention heads/depth/width/normalization/steps；保留 P297 direct baseline；
- structural pivot：P299 回到 P297 架构，等权加入 group-mean budget distillation loss，直接对齐 attained constraint；
- 下一可用 failure id 为 `V67-F200`。

### V67-F200 — P299 group-mean soft loss未跨 cohort改善 direct constraint

- canonical=`run://worldsim_v67/WS-V67-P299-CONSTRAINT-AWARE-DIRECT-AUTHORITY-COMPILER-01/
  20260901T001500Z__constraint-aware-direct-authority-s0-r1`；
- symptom：source attained MAE=`.0225111 > P297 .0204608`，P201=
  `.0317413 > P297 .0297261`；regret与通用 fidelity仍过门，但 comparative门失败；
- root cause：soft mean matching仍只是 teacher imitation，并未在推理结构中强制 fraction与 group mean的关系；
  source/P201差异表明固定 loss weight不能保证跨 cohort constraint；
- closed candidate：不扫 group-loss weight、Actor/group loss比例、steps/lr；保留 P297 direct baseline；
- structural pivot：P300 采用同模型0/1端点的自一致可微 mean projection；
- 下一可用 failure id 为 `V67-F201`。

### V67-F201 — P300 uniform-shift projection破坏逐 Actor fraction单调性

- canonical=`run://worldsim_v67/WS-V67-P300-PROJECTED-DIRECT-AUTHORITY-COMPILER-01/
  20260901T003000Z__projected-direct-authority-s0-r1`；
- registered gates：P201 attained MAE=`.0120266`、regret=`.00014114`、严格优于 P297，3/3；预注册 verdict不反改；
- latent structural defect：P201 fraction-budget monotonicity violations=`10,754`（size48/96=`4,744/6,010`），
  source=`20,746`。uniform mean shift + box clip可保 aggregate mean，却不能保每个 Actor budget顺序；
- accurate disposition：记录 supported constraint fidelity，但 P300不进入最终 compiler，不写 monotone authority claim；
- structural recovery：P301 同一 model 0/1 endpoints逐元素 convex combination已通过，P201 attained MAE=
  `.0285229`、regret=`7.7359e-5`、violations=`0`，F201关闭；P302固定三锚点与 P303归一化正积分 warp
  分别研究非线性表达力/单调用校准；P302进一步达到 attained MAE=`.0256586`、violations=`0`，不重新打开 F201；
- 下一可用 failure id 为 `V67-F207`。

### V67-F202 — P303 warp积分错误复用 base knot count

- failed run=`run://worldsim_v67/WS-V67-P303-NORMALIZED-MONOTONE-WARP-AUTHORITY-COMPILER-01/
  20260901T011500Z__normalized-monotone-warp-authority-s0-r1`；
- symptom：独立 warp输出8 knots，但调用 base `_integral`时索引宽度使用 base rate knot count，首个 forward触发
  CUDA gather index out-of-bounds；0 completed training steps、0 quality read、无科学 verdict；
- research check：PyTorch gather官方合同要求所有 index落在 source dim范围内；UMNN开源实现也将积分器与自身
  integrand参数绑定。问题是局部张量维度所有权，不是 positive-integral假设失败；
- minimal recovery：r2用 warp rates自己的 `shape[-1]`计算 width/index/cumulative，模型、knots、hidden、steps、
  seed、teacher、gates全部不变；
- concurrent branch：P304只复用已支持 P302 的 base做单调用压缩训练，不依赖 F202失败路径；
- 下一可用 failure id 为 `V67-F206`。

### V67-F203 — P303 context warp未跨规模保持 attained-fraction fidelity

- canonical=`run://worldsim_v67/WS-V67-P303-NORMALIZED-MONOTONE-WARP-AUTHORITY-COMPILER-01/
  20260901T013000Z__normalized-monotone-warp-authority-s0-r2`；F202修复后合同原样完成；
- result：P201 budget/attained MAE=`.0066260/.0288235`、regret=`6.0888e-5`、violations=`0`、forward=`.7006s`；
  attained门及严格优于 P301门失败，2/4；size48/96=`.0269137/.0307334`；
- root cause：source attained=`.0200754`且训练 element loss很低，但 group-conditioned warp依赖 raw pooled moments，
  在 P201/size96 shift下 group mean calibration失配；这是 OOD group calibration问题，不是单调结构或预算拟合失败；
- literature response：NeurIPS 2024 multicalibration/OOD工作要求跨重叠groups显式校准；COLT 2022指出普通 DRO
  不保证 shift下统一低regret。因此不靠增加 knots/hidden补救，也不把 source改善外推为 P201 claim；
- disposition：关闭 normalized-warp结构扫描；保留其 `.7006s`单调用/零违例结果为 Pareto descriptive negative，
  独立 P304从 P302 anchor-trained base做单调用压缩，不复用 context warp；
- 下一可用 failure id 为 `V67-F206`。

### V67-F204 — P304 anchor-trained base单调用压缩未保持P201 fidelity

- canonical=`run://worldsim_v67/WS-V67-P304-ANCHOR-INITIALIZED-SINGLE-CALL-AUTHORITY-COMPILER-01/
  20260901T014500Z__anchor-initialized-single-call-s0-r1`；
- result：P201 attained MAE=`.0290092 > P301 .0285229`，但 regret=`6.5460e-5`、violations=0、forward=`.3070s`；
  3/4拒绝；source attained=`.0192548`，size96=`.0310086`；
- interpretation：NeurIPS 2021指出 student即使容量足够也常无法高保真匹配 teacher；本项目中移除 P302 piecewise
  decision rule后，anchor initialization保留了速度但没有保留跨场景 group calibration；
- disposition：不做 distillation weight/EMA/step sweep；与 F203共同关闭 single-call compression，选择 P302为
  fidelity/monotonicity端的 Pareto compiler；
- 下一可用 failure id 为 `V67-F206`。

### V67-F205 — P305 confirmation rows指向prep run

- failed run=`run://worldsim_v67/WS-V67-P305-FRESH-REUSE-PIECEWISE-AUTHORITY-CONFIRMATION-01/
  20260901T020000Z__fresh-reuse-piecewise-authority-confirmation-s0-r1`；
- symptom：`P277_FRESH_EPISTEMIC_ROWS.npz`实际位于 P277 confirmation run，配置误指 prep run；`np.load`前
  FileNotFound，0 training、0 quality read、无科学 verdict；
- minimal recovery：r2只替换 source/p201 artifact run path；模型、rows文件、conditions、sizes、teacher、gates不变；
- resolution：r2 3/3 supported，attained MAE=`.0167472`、regret=`-.0001084`、violations=0；
- next research：P306保持 P302三锚点规则，只共享重复 context computation；不依赖 P303/P304失败权重；
- 下一可用 failure id 为 `V67-F207`。

