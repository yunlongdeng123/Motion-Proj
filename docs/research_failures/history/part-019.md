# 历史原始记录 019

> 冻结历史，不是当前执行规则。当前规则见 [scaling law](../../../auto-research_scaling_law.md) 与 [AGENTS](../../../AGENTS.md)。
> 原文件：`docs/RESEARCH_FAILURES.md`；迁移前提交 `abbd431c`。原有相对链接按原目录解释，证据路径原样保留。
> 按索引读取相关小节；旧哈希、过度门控和资源指令均不重新生效。

## V7.1 M24 implementation note — 只在内存替换目标Actor（2026-09-05）

- 每个variant先从同一checkpoint恢复；geometry-locked只替换rigid index12的Gaussian parameter对象，hidden只置低其opacity；
- Background、其他Actors和instances trajectory不赋值，runner没有checkpoint save路径；
- 数据集preload置CPU避免与M21 waiting evaluator争用显存；三个camera串行；
- 只做入口`py_compile`，不做额外render smoke或质量阈值回归。

下一可用编号仍为：`V71-F26`。

## V7.1 M24 prevention note — render footprint不是physical supervision（2026-09-05）

- original-vs-hidden mask只定位目标Actor像素，不能用来删physical Gaussian、调scale/opacity或证明first-return；
- Actor/frame/camera在新render quality读取前固定；全部3 camera保留，零footprint也不删除；
- PSNR没有pass阈值，下降即作为geometry-locked carrier的真实视觉限制写入；
- runner不得新增hash/checksum/fingerprint，不写checkpoint；若rasterizer入口失败才登记`V71-F26`。

下一可用编号仍为：`V71-F26`。

## V7.1 M23 paper prevention note — attribute carrier不是render或geometry正结果（2026-09-05）

- related work只迁移visual attribute field与geometry/texture分层，不声称复现Feature 3DGS或Neural Shell训练；
- results显式保留all/hazard early恶化`1.60/0.60%`，不得仅报告Chamfer/hit改善；
- limitations写明没有对新sidecar做novel-view rendering，不能获得photorealism或semantic fidelity；
- M23仍不影响M21物理external判定，SH/opacity不进入ray energy。

下一可用编号仍为：`V71-F26`。

## V7.1 M23 outcome note — 安全附着不隐去单场景physics反向（2026-09-05）

- 3/3 interface decisions通过：5791/5791 physical Gaussians均获同Actor SH/opacity，geometry与checkpoint只读；
- scene-0230描述性physical结果为Chamfer `-7.160mm`、hit `+3.186pp`，但all/hazard early相对baseline
  `-1.604/-0.599%`（负号表示恶化）；
- M23协议未把物理改善设为attribute-carrier gate，因此不新增failure ID，但该场景禁止进入“appearance bridge同时减少
  ghost”的正例；
- assignment max `1.006m`仅报告，不以阈值删点/Actor，也不允许借视觉属性反向移动geometry修复early。

下一可用编号仍为：`V71-F26`。

## V7.1 M23 implementation note — association不是geometry correction（2026-09-04）

- 最近邻只选择同Actor的visual attribute row；其距离不用于移动、缩放、删除或拒绝physical Gaussian；
- sidecar分别存M8 centers/scales与StreetGS SH/opacity，显式不包含appearance-derived geometry；
- 不增加assignment radius/quality gate，不因远距离匹配删Actor，避免appearance coverage变成隐式后处理；
- 仅`py_compile`和一次canonical run，不扩展回归矩阵。

下一可用编号仍为：`V71-F26`。

## V7.1 M23 prevention note — 视觉属性不得反向改写物理carrier（2026-09-04）

- Gau-Occ式joint FFN不能作为geometry preservation证据；M23只复制同Actor SH/opacity，不复制appearance几何；
- physical center/scale由冻结M8/GT supervision唯一决定，appearance assignment没有梯度、loss或checkpoint写回；
- Background、trajectory、image pixels与semantic/hazard不进入M21 physics；
- assignment距离只描述visual support稀疏性，不作为删除、拒绝或几何移动阈值；所有12 Actors保留；
- 若identity/attribute覆盖失败则登记`V71-F26`，不通过跨Actor匹配或扩大搜索半径补偿。

下一可用编号仍为：`V71-F26`。

## V7.1 M22 paper prevention note — SE(3)恒等式不得扩权（2026-09-04）

- method将motion写成inverse-query composition，不称per-frame learned geometry或post-hoc correction；
- results同时报告r1的3.125cm float32假残差与r2 FP64 inverse-query结果，不能隐去实现失败；
- limitations明确不获得trajectory accuracy、dynamic discovery、occlusion、collision或surface-improvement claim；
- M21 fresh AV2仍是几何/物理跨域结论的独立未决条件。

下一可用编号仍为：`V71-F26`。

## V7.1 M22 outcome note — 等变通过不等于几何性能通过（2026-09-04）

- r2在原12 Actors/36 Actor-frames上以inverse-query得到energy residual=`3.397e-14`、distance residual=
  `1.243e-14m`，4/4 numerical/dataflow decisions通过；
- checkpoint、trajectory与Background均未写入，SH/opacity/image未进入physics，未生成新模型；
- 该结果只关闭M8/M21 canonical physics到world frame的部署组合缺口；几何质量仍由M8 native-3D/逐帧GT loss及
  M21 fresh AV2独立判定；
- 不允许把SE(3)恒等式写成新的Chamfer/occupancy/safety增益，也不据此提前读取M21 partial quality。

下一可用编号仍为：`V71-F26`。

## V71-F25 — world-frame float32 cdist消减伪造刚体不一致（2026-09-04）

- 分类/状态：numeric coordinate implementation / resolved before scientific interpretation；run=
  `20260904T160000Z__m22-se3-composition-r1`；
- symptom：12/12 identity和11个moving Actors通过，但最大energy residual=`0.0617554`、pairwise residual=
  `0.03125m`，故verdict=`se3_composition_implementation_rejected`；
- root cause：最大Actor translation约`126m`，默认float32 `torch.cdist`在点数大于25时使用Euclidean MM路径，
  大平方范数相减对厘米级局部距离产生catastrophic cancellation；这不是canonical geometry或trajectory authority失败；
- evidence：PyTorch官方`torch.cdist`文档明确给出默认MM切换条件与`donot_use_mm_for_euclid_dist`模式；
- resolution：部署定义直接执行`E_world(x,t)=E_actor(T_t^{-1}x)`，避免在大world坐标上构造距离；数值审计使用
  float64 direct Euclidean kernel。M8/M21参数、12 Actors、36 audited frames、query数和冻结容差均不变；
- claim impact：r1只作为失败实现保留，禁止把其残差写成动静解耦科学反例。

下一可用编号：`V71-F26`。

## V7.1 M22 implementation note — 不以快照比对替代只读数据流（2026-09-04）

- runner代码路径不包含checkpoint写回、Background更新或trajectory tensor赋值；只生成run目录summary/rows/status；
- 不新增hash、checksum、fingerprint，也不复制百万级Background tensor做形式化前后比对；
- energy与pairwise数值残差只验证坐标实现，科学依据仍是M8训练内逐帧GT supervision；
- 若数值容差失败才登记`V71-F25`，不扩大为smoke/regression矩阵。

下一可用编号仍为：`V71-F25`。

## V7.1 M22 prevention note — motion只能搬运canonical physics（2026-09-04）

- M8逐帧GT coverage已经进入训练目标；M22只审计部署组合，不以post-hoc trajectory变换冒充新的物理监督；
- Actor energy固定为M8 anchors/children/scales，world energy仅通过只读SE(3) inverse query定义；
- Background完全不参与Actor query，SH/opacity/image不进入physics；trajectory也不回流geometry head；
- 固定12个现有identity match和首/中/末最多3帧，不筛Actor、不扫frame/query/容差；
- 若SE(3)不变量失败则登记`V71-F25`并修正坐标/变换约定，不以learned deformation或appearance补偿。

下一可用编号仍为：`V71-F25`。

## V7.1 paper prevention note — 相对field增益必须绑定绝对anchor reference（2026-09-04）

- M19必须报告point surface反转与field绝对early，不能只报告field-vs-degraded-point相对改善；
- M20必须报告M8-initial energy反事实与scale膨胀，不能只与M18比较后称joint training有效；
- M21必须标注development-selected、AV2 pending，且Gaussian scale不解释为校准occupancy概率；
- 论文已同时保留native-3D监督、ray监督、轨迹只读与image隔离边界；13页PDF编译成功。

下一可用编号仍为：`V71-F25`。

## V7.1 M21 launch note — 只读进度、不读partial quality（2026-09-04）

- evaluator PID=`25098`，当前`5/20 logs / 129 Actors`，等待第6个log；
- 唯一downloader PID=`21975`，没有并发第二下载器；剩余磁盘约`93GiB`；
- 已读取字段仅为status、log id、Actor row count和进程状态；partial metric/row未读；
- M21完成前不得用M5/M7/M8/M18/M21任一partial quality修改checkpoint、scale、bin、threshold或cohort。

下一可用编号仍为：`V71-F25`。

## V7.1 M21 prevention note — development-selected energy只能由fresh AV2确认（2026-09-04）

- M21固定canonical M8 checkpoint、M8 children scale、anchor scale `0.08m`、64 bins和CDF median；无训练；
- 20-log cohort在任何M21 external quality读取前冻结；允许逐log写partial，但20/20前禁止读取metric/row；
- external gate直接比较energy与原始baseline的绝对early/hit，不使用可能恶化的current point作为相对分母；
- M8 point Chamfer与Actor/hazard retention同时绑定；禁止filter、阈值、scale、bin、checkpoint或log替换；
- 若失败登记`V71-F25`并关闭，不用M20 fine-tune或M18 decoder恢复。

下一可用编号仍为：`V71-F25`。

## V71-F24 — decoder-free ray loss以尺度膨胀损害anchors与自身energy（2026-09-04）

- 分类/状态：isotropic support-scale shortcut / terminal for joint energy fine-tuning；canonical=
  `20260904T152000Z__m20-decoder-free-energy-s71122-r1`；
- formal result：5/6；Chamfer vs M8=`-2.160mm`、energy vs M18 hazard early=`-4.477pp`、all hit=
  `+2.325pp`，但point hazard early相对baseline=`-3.247%`，直接物理门失败；
- scale evidence：mean/median `0.162/0.150→0.313/0.313m`，q90=`0.376m`逼近`0.40m`上限，normalized
  scale loss `0.583→2.395`；NLL下降但depth L1、entropy上升且GT-bin probability下降；
- counterfactual：冻结M8使用同一energy已达all/hazard early=`18.187/17.422%`、hit=`62.283/62.068%`；
  M20训练使early恶化`2.102/2.462pp`、hit下降`4.493/4.737pp`；
- root cause：isotropic scale同时控制3D support与ray energy宽度，categorical objective可通过扩大support降低NLL，
  PCGrad只能处理梯度方向，不能恢复参数角色可辨识性；
- resolution：关闭M20 fine-tune，不扫scale/loss/PCGrad；保留冻结M8 + analytic energy为新的representation candidate，
  必须在未读fresh AV2上事前冻结后确认；
- claim impact：M20不能证明ray fine-tuning改善Gaussian geometry；诊断只支持冻结supervision-native geometry的
  analytic energy具有开发集潜力。

下一可用编号：`V71-F25`。

## V7.1 M20 prevention note — physics gradient不得绕开Gaussian geometry（2026-09-04）

- 删除可学习field decoder；ray logit只能由immutable anchors与可学习children/scales的metric Gaussian energy产生；
- native-3D set/plane/scale/frame监督持续存在，categorical first-return与geometry对同一surface head做PCGrad；
- 评价point hazard early相对原始baseline，并把energy绝对hazard early/hit与冻结M18比较，不再只看field-vs-current；
- anchor scale固定`0.08m`，32 bins/4 epochs/seed固定；禁止temperature/scale/loss/bin/epoch sweep；
- 若point physical或M18绝对边界失败，登记`V71-F24`并关闭isotropic energy，不恢复learned compensator。

下一可用编号仍为：`V71-F24`。

