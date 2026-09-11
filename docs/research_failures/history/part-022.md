# 历史原始记录 022

> 冻结历史，不是当前执行规则。当前规则见 [scaling law](../../../auto-research_scaling_law.md) 与 [AGENTS](../../../AGENTS.md)。
> 原文件：`docs/RESEARCH_FAILURES.md`；迁移前提交 `abbd431c`。原有相对链接按原目录解释，证据路径原样保留。
> 按索引读取相关小节；旧哈希、过度门控和资源指令均不重新生效。

## V7.1 M8 outcome note — 逐帧监督形成新Pareto点且不新增failure（2026-09-04）

- canonical=`20260904T202000Z__m8-temporal-frame-s71110-r2`；frame-balanced distance在moving/quasi-static均改善，
  原五项physical contract与新增mechanism gate共6/6通过；
- M8没有motion/time/hazard/image输入，moving分层只用于结果解释，支持shape/trajectory authority显式拆分；
- hazard early改善从M7的`10.220%`降到`5.123%`，故M8不能写成全面替代；其优势是Chamfer `-6.207mm`与
  frame coverage `-5.965mm`，M7仍是更强hazard point；
- 不用事后loss weight、epoch、seed或M7/M8混合恢复trade-off；两者按各自冻结表示进入同一fresh AV2 cohort；
- 下一可用failure ID仍为`V71-F13`。

下一可用编号仍为：`V71-F13`。

## V71-F12 — M8帧target在inference上下文创建导致首步autograd拒绝（2026-09-04）

- 分类/状态：engineering / recovered before optimizer step；run=`20260904T201000Z__m8-temporal-frame-s71110-r1`；
- 观察：首个minibatch的`torch.cdist`报`Inference tensors cannot be saved for backward`；
- root cause：加载冻结M5/M7输出与构造帧target共用了同一个`torch.inference_mode()`块；后者随后参与M8梯度图；
- exposure：0 backward/optimizer update、0 holdout/Selection/Source Final/external quality read，不产生科学trial；
- resolution：只把`_prepare_temporal_actor`移到inference块外；target内容、帧组、loss、模型、seed、split和gate不变；
- prevention：冻结模型推理tensor与训练监督tensor分阶段构造；不为此增加全套回归测试；
- claim impact：无；新run-id恢复，下一可用failure ID=`V71-F13`。

下一可用编号：`V71-F13`。

## V7.1 M8 prevention note — 不让trajectory或appearance替geometry解释误差（2026-09-04）

- M8 geometry head沿用M7输入，明确排除trajectory、velocity、timestamp、hazard与image；不能靠给动态Actor单独容量通过；
- 各帧仅提供actor-canonical endpoint coverage；canonical-to-world始终是只读GT rigid transform，不学习形变补偿标签误差；
- 单帧partial observation不适合symmetric Chamfer，故只用target→surface；完整surface仍由原union target双向set loss约束；
- moving/quasi-static阈值只分层报告，不能作为训练采样、loss weight或failure过滤条件；
- M8若不能降低frame-balanced distance并保留原physical合同，登记`V71-F12`并关闭当前fine-tune，不扫超参恢复。

下一可用编号仍为：`V71-F12`。

## V7.1 M7 paper-evidence note — development机制证据不升级为外域结论（2026-09-04）

- 证据图按baseline危险率中位数事前规则选Actor，不按M7改善量挑最好案例；图中保留anchors与全部children，不隐藏坏点；
- 论文明确M7 development曾被旧预训练暴露，仅证明监督/表示机制可行，不能称独立confirmation或zero-shot；
- 冻结AV2完成前禁止读取partial物理指标、修改abstract数值、改模型或替换log；最终20-log aggregate一次性决定外域claim；
- 本里程碑没有新增scientific/engineering failure；第一次本地复制后编译因文件可见时序失败，原样重试即通过，不占用
  `V71-F12`。

下一可用编号仍为：`V71-F12`。

## V7.1 M7 external execution note — scene-ready入口正常（2026-09-04）

- evaluator已按`.complete`处理3/20 logs与62 Actors，权重/standardizer/compiler均冻结，未读partial physical aggregate；
- 与M5 evaluator并行只会重复只读编译/推理，不启动第二下载器、不修改log；当前无工程或科学failure；
- 下一可用failure ID保持`V71-F12`。

下一可用编号仍为：`V71-F12`。

## V7.1 M7 external prevention note — 同一fresh cohort比较不反馈M5局部结果（2026-09-04）

- M7方法/checkpoint/external合同在读取任何M5 AV2 physical metric前冻结；此前监控只看log/Actor计数和进程状态；
- M7与M5可在同一事前冻结20-log cohort上作paired model comparison，但各自保留完整20-log aggregate，不删困难log；
- M5输出不得用于M7选择；M7 external失败即登记`V71-F12`，不以M5差值、hit或per-log子集改写primary verdict。

下一可用编号仍为：`V71-F12`。

## V7.1 M7 outcome note — set-to-set child surface通过且不新增failure（2026-09-04）

- canonical=`run://worldsim_v71/WS-V71-M7-GT-SUPERVISED-SEED-EXPANSION-01/
  20260904T121000Z__m7-gt-seed-expansion-s71109-r1`；五门全过，hazard early=`10.220%`、Chamfer=`-2.902mm`、
  hit=`+2.173pp`、retention=`100/100%`；
- 与`V71-F11`不同，四children在训练图内接收target set与ray physics supervision；部署不增加mask/filter/threshold，
  因此结果支持candidate support是M6关键瓶颈；
- free-space differentiable proxy最终仍为M5 reference的`1.109`，而literal heldout early显著改善；二者算子差异如实保留，
  不据此追加proxy gate或重训；
- holdout有历史pretraining exposure，故不升级为泛化结论；进入冻结fresh AV2外测，下一可用failure ID仍为`V71-F12`。

下一可用编号仍为：`V71-F12`。

## V71-F11 — 一候选一最近GT点可学但不能形成集合级危险表面覆盖

- 分类/状态：representation + target assignment / terminal for one-to-one anchor correction；canonical=`run://worldsim_v71/
  WS-V71-M6-GT-SUPERVISED-GAUSSIAN-RELOCATION-01/20260904T120000Z__m6-gt-supervised-gaussian-s71108-r1`；
- 观察：direct center/plane/scale与free-space均下降，但训练Chamfer比停在`~1.002`；development hazard early只降
  `3.670%<5%`，其余Chamfer/hit/retention四门通过；相对M5，clear退化减轻但hazard收益被削弱；
- root cause：659个nonempty Actors的candidate/target中位数仅`17/966`（`1.64%`）；nearest endpoint标签只优化
  candidate→target精度且每个seed仍只能产生一个中心，无法优化target→generated-surface coverage；
- literature response：PoinTr (ICCV 2021)明确把completion建模为set-to-set translation；SnowflakeNet (ICCV 2021)
  通过parent-child splitting生成局部结构。迁移为固定4-child seed expansion和双向set supervision，而非换nearest metric；
- resolution：关闭M6 one-to-one correction，不调loss/seed/residual bound；M7仍使用相同actor-canonical GT、hard anchors与
  first/free-space，只改变输出支撑数及set supervision；
- claim impact：M6证明GT supervision可优化局部几何标签，但不支持完整表面或hazard physical consistency claim；下一可用
  failure ID=`V71-F12`。

## V7.1 M7 prevention note — seed expansion必须由GT set训练而非部署增密

- children在训练图内生成并直接接收完整target set、plane、scale、first/free-space loss；禁止训练后插值/复制点冒充completion；
- observed anchors仍不可学习，四个child slot固定且初始重合于M5 center；不扫branch factor或slot count；
- 若M7失败，登记`V71-F12`并关闭当前candidate-seed family，再考虑implicit/voxel生成；不回调M6最近点标签或追加mask。

下一可用编号仍为：`V71-F12`。

## V7.1 M6 prevention note — target-first不等于Gau-Occ原样迁移（2026-09-04）

- Gau-Occ的20-sweep target只声明ego-motion alignment，直接用于动态Actor会把物体运动写成completion geometry；V7.1必须
  逐帧用GT box/pose进入actor-canonical frame，现有compiler保持该合同；
- completed cloud在Gau-Occ中是Gaussian initialization而非硬anchor，后续image fusion可改center/scale/rotation；因此不得
  用该论文替post-hoc StreetGS sidecar升级claim；
- M6首轮完全隔离image/semantic/motion，只让build evidence预测geometry；target endpoints只构造center/scale/plane标签，
  GT geometry与literal first-return loss在Stage P仍持续回传；
- observed anchors结构上不可学习，UNKNOWN不作部署硬删除；失败不得通过mask、threshold、第二seed或loss sweep恢复；
- 若M6失败，登记`V71-F11`并判断是直接GT correspondence/candidate support问题，再决定生成式completion或新表示；不得
  回到appearance bridge或PCGrad解释错误target。

下一可用编号仍为：`V71-F11`。

## V71-F10 — StreetGS sidecar结构成立但不能证明训练内生物理一致

- 分类/状态：claim-boundary + scientific decomposition / terminal for post-hoc proof；canonical=`run://worldsim_v71/
  WS-V71-M5-STREETGS-APPEARANCE-BRIDGE-01/20260904T111000Z__m5-streetgs-bridge-r1`；
- 观察：12 Actors/106,807 appearance Gaussians可按identity无损连接，Chamfer/hit为`-2.113mm/+0.525pp`，但hazard early
  只降`0.798%`且all/clear恶化`1.309/12.877%`；
- root cause：bridge在冻结M5输出之后连接两种表示，只能证明ownership/immutability，不能把post-hoc physical sidecar升级为
  由GT和supervision定义的geometry consistency；原strict-positive hazard gate过弱；
- correction：不修改canonical run或事后加filter；把结论降为representation interface，并将GT state construction、
  completion learning、appearance/dynamics coupling拆开；
- literature/action：专项审计Gau-Occ completion diffuser target、Gaussian anchor约束、image conditioning与occupancy联合训练后，
  才冻结下一supervision-first候选；
- claim impact：无V7.1 GS geometry/RGB improvement claim；下一可用failure ID=`V71-F11`。

## V7.1 appearance-bridge prevention note — 不修改GS checkpoint冒充外观提升（2026-09-04）

- Street Gaussians/DrivingGaussian支持Actor-composed appearance ownership，但不保证其Gaussian means是安全collision
  surface；因此M5 physical surface作为同identity sidecar，而非覆盖appearance geometry；
- checkpoint所有外观/轨迹tensor只读，不重新render后声称RGB指标提升，不训练GS或颜色；
- bridge只要求已有rendered asset、Actor identity、物理surface改变与hazard early方向，不把train scene称为泛化；
- 若同场hazard early不降，登记`V71-F10`并保留结构接口但不作physical bridge positive claim。

## V7.1 AV2 execution prevention note — scene-ready不改变exact-once合同（2026-09-04）

- evaluator只在下载器写出对应`.complete`后读取该log，I/O重叠不允许看部分log结果后改变后续模型或日志；
- checkpoint、standardizer、类别、modulo、tolerance和verdict已在首个fresh log前冻结；
- 所有20 logs按cohort顺序执行，0-Actor/困难log不替换；partial JSONL只用于进度恢复，不作为模型选择信号；
- external失败即`V71-F10`，不进行target adaptation或第二cohort恢复。

## V71-F09 — V7.1 AV2 downloader解释器路径不存在并误写0-log完成标记

- 分类/状态：engineering / recovered before external read；入口=`scripts/download_worldsim_v71_av2.sh`；
- 观察：写死的`/root/miniconda3/envs/motionproj/bin/python`不存在；process substitution失败未使`mapfile`退出，数组为空，
  循环未执行却写出`ALL_COMPLETE 0 logs`；
- exposure audit：s5cmd未启动、0 AV2 files/annotations/LiDAR/model output/quality read，故不消耗冻结external cohort；
- root cause：motionproj环境实际位于高速盘`/root/autodl-tmp/envs/motionproj`，与conda base前缀不同；
- resolution：只修Python绝对路径并断言冻结项数为20；cohort/顺序/workers/retry/model/gates不变；
- claim impact：无科学结论；下一可用failure ID=`V71-F10`。

## V7.1 fresh AV2 prevention note — 不复用V7已消费的60 logs（2026-09-04）

- 原30-log、20-log recovery与10-log EviComp cohorts均已被V7研究读取，不可作为M5 fresh confirmation；
- V7.1新20-log cohort在M5 external output前从其90-log补集metadata-only冻结，下载失败只能原log重试，不替换；
- external读取后禁止fine-tune、calibration、threshold/tolerance/field scale调整或第二cohort救援；
- 若冻结三项primary合同失败，登记`V71-F09`并把结论限制为source/train-domain，跨传感器迁移关闭。

## V7.1 M5 prevention note — PCGrad只改冲突梯度，不改变部署表面（2026-09-04）

- Diagnostic C直接观测全模型`72.7%`、encoder`81.8%` minibatches负内积，因此M5不是事后loss-weight sweep；
- 对两个物理任务使用对称projection；无冲突batch精确保留原task梯度，三个auxiliary只按原权重相加；
- UNKNOWN action/mask不恢复，训练、development与部署都使用anchors加全部moved candidates；
- 66-Actor development曾进入M0预训练，只可筛选方法，不得包装为独立confirmation；失败即`V71-F09`，不改seed/lr/
  epoch/teacher weight恢复。

## V7.1 diagnostic C prevention note — 不把PCGrad当作默认补丁（2026-09-04）

- PCGrad只在任务梯度内积为负时投影；因此先在冻结M2的原始minibatch合同上测量，而不是因两个指标不同就假定冲突；
- 诊断只读train corpus与M2权重，不训练、不读取已消费Source Final明细、不访问AV2；
- 若all/encoder/head梯度主要同向，后续不得使用PCGrad、GradNorm或loss-weight sweep掩盖表示可辨识性问题；
- 本阶段不产生failure编号，下一可用仍为`V71-F09`。

