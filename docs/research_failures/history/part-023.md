# 历史原始记录 023

> 冻结历史，不是当前执行规则。当前规则见 [scaling law](../../../auto-research_scaling_law.md) 与 [AGENTS](../../../AGENTS.md)。
> 原文件：`docs/RESEARCH_FAILURES.md`；迁移前提交 `abbd431c`。原有相对链接按原目录解释，证据路径原样保留。
> 按索引读取相关小节；旧哈希、过度门控和资源指令均不重新生效。

## V71-F08 — M4平衡占据边界可抽取但定位到错误表面

- 分类/状态：scientific + identifiability / terminal for global-latent implicit field；canonical=`run://worldsim_v71/
  WS-V71-M4-BALANCED-OCCUPANCY-BOUNDARY-01/20260904T103259Z__m4-balanced-occupancy-boundary-s71105-r1`；
- 观察：66/66 holdout Actors形成zero-logit surface，排除M3 single-sign与M1 post-hoc AND extraction问题；但hazard/all
  early均约翻倍，Chamfer恶化`+53.482mm`、hit recall下降`2.073pp`；
- mechanism：共享Actor latent能学到front/back分类并形成闭合边界，却没有足够的ray-local build evidence确定真实首表面位置；
  loss下降和完整抽取因此不能代表三维物理自洽；
- literature response：Occupancy Networks支持单一decision boundary，但它不消除partial observation下的条件歧义；下一步
  依据PCGrad的负梯度冲突定义先直接测量surface/first-return梯度，而不是事后套用多任务优化器；
- decision：关闭M4/global-latent implicit，不扫loss weight/offset/seed，不读AV2；下一可用failure ID=`V71-F09`。

## V7.1 M4 prevention note — 平衡decision boundary不是M3超参恢复（2026-09-04）

- M4改变监督类型：从有量纲SDF regression转为显式平衡occupancy BCE；这是一种新表示，不回写M3 offset或初始化；
- 只有一个logit，target/anchor零边界、ray render和edge extraction共享它；禁止追加evidence gate/UNKNOWN mask；
- development holdout已被M3使用，只能筛除无surface模型；真正确认必须是冻结后的未读AV2；
- M4失败即`V71-F08`并关闭global-latent implicit field，不扫temperature/weight/seed。

## V71-F07 — M3 signed regression收敛到无zero crossing的单符号场

- 分类/状态：scientific + implicit initialization / terminal for M3；canonical=`run://worldsim_v71/
  WS-V71-M3-RAY-SIGNED-LEVEL-SET-01/20260904T102101Z__m3-ray-signed-level-set-s71104-r1`；
- 观察：训练四项loss均稳定下降，Eikonal降至`0.0561`，但66 holdout Actors的grid SDF全部为正，global range
  `[0.11886,0.32123]m`，zero crossing=`0/66`；
- consequence：输出只有hard anchors，hazard early以表面缺失下降28.609%，Chamfer/hit恶化`+136.343mm/-17.760pp`；
- mechanism：有量纲smooth-L1在surface-zero、ray正负和abs-SDF rendering共同作用下选择了坏的单符号局部解；Eikonal
  约束梯度范数但不保证符号两侧或zero-set存在；
- literature response：IGR/SAL强调几何初始化以避免坏level-set解；当前不补做M3初始化/offset sweep，而采用Occupancy
  Networks式平衡二分类decision boundary，显式保证front/back两类竞争；
- decision：M3关闭、external unread；下一可用failure ID=`V71-F08`。

## V7.1 M3 prevention note — signed scalar同时承担监督、render与抽取（2026-09-04）

- 禁止恢复M1的unsigned SCF + evidence class双head；M3只有一个signed scalar，zero crossing就是部署surface；
- front/back符号只由train target ray方向定义，heldout target不进入evidence encoder；UNKNOWN不产生监督标签或删点；
- train-role holdout按固定索引冻结，不因结果更换；任一stage gate失败即`V71-F07`，不扫voxel/offset/Eikonal/seed；
- 实现阶段无failure，Selection/Source Final/AV2均不读。

## V71-F06 — M2部署一致位移在Source Final仅Chamfer不达标

- 分类/状态：scientific / terminal for M2；canonical=`run://worldsim_v71/
  WS-V71-M2-DEPLOY-CONSISTENT-RELOCATION-01/20260904T100928Z__m2-deploy-consistent-relocation-s71103-r1`；
- 观察：69 Source Final Actors（44 hazard）上hazard early下降`5.262%`、hit recall增加`1.205pp`、retention 100%，
  但Chamfer恶化`+1.581mm`，4/5；
- mechanism：移除UNKNOWN action后F04的`+48.745mm/-10.819pp`大退化消失，说明训练—部署错位已修；剩余约1.1mm
  gate gap来自低容量统一位移在early和双向surface距离之间仍有不可忽略折衷，不是删除collapse；
- decision：M2拒绝且AV2不读；不得在已消费Source Final上调teacher weight、训练轮数、lr、seed或loss ratio；
- response：采用Occupancy Networks/NeuS/Neural-Pull启发的单一signed decision variable，让训练ray surface、物理render与
  extraction共享同一level-set；下一可用failure ID=`V71-F07`。

## V7.1 M2 prevention note — 一个surface贯穿训练与部署（2026-09-04）

- M2不允许UNKNOWN action/output/threshold；UNKNOWN只保留在输入evidence mass中，不能删点；
- checkpoint迁移只复制共享encoder和ray/normal两行，第三行显式丢弃；训练loss与部署都使用全部moved candidates；
- 不因Source Final结果调teacher weight、epoch、lr或seed；五项任一失败即`V71-F06`并关闭M2；
- 只有Source Final五项全过才读取AV2；实现阶段无failure。

## V7.1 diagnostic closure — F04/F05根因完成分解，不新增failure（2026-09-04）

- A证明`V71-F04`主要来自部署UNKNOWN mask：相同冻结位移取消mask后，hazard early仍下降5.924%，且Chamfer/hit变为
  `+0.445mm/+0.723pp`；禁止将post-hoc replay改写为旧M0通过，禁止继续扫unknown threshold；
- B证明`V71-F05`不是grid/band空：885,574 grid上band-only 57,323、occupied-only 107,830但joint=0，oracle band
  33,280；known surface probes同样joint=0；
- prevention：下一模型只能用单一部署surface对象训练；UNKNOWN可作为输入证据但不能作为未进入physical loss的硬删除
  action。若研究implicit field，surface必须是一个signed/occupancy decision boundary，不能再把unsigned distance与
  evidence argmax作后验AND；
- 两项诊断正常完成，不占用`V71-F06`。M2 frozen前Source Final/AV2保持未读。

## V7.1 diagnostic prevention note — 冻结重放不升级为调参（2026-09-04）

- A不得改变位移、weight、feature或compiler，只比较UNKNOWN mask on/off；结果不能用于回写旧M0 threshold；
- B不得扫band/grid/evidence threshold，只分解固定输出；oracle-grid只判断extraction几何是否具备可抽取点，不是模型结果；
- 两项不读Source Final/AV2、不生成新selector或gate；下一训练假设必须在结果后单独冻结；
- runner实现阶段无failure，不占用`V71-F06`。

## V7.1 repository prevention note — terminal wrapper归档，不动复现资产（2026-09-04）

- 归档对象仅为6个已关闭阶段的一次性shell wrapper；实际runner/config/corpus/canonical artifact仍是active evidence；
- Git repack只压缩可达对象，`.git`从约498MiB降至39MiB；不以删除历史、数据或权重换空间；
- 后续不得从归档launcher误复开M0/M1或重读终测；诊断与新假设必须使用新run ID和独立runner；
- 本阶段无scientific/engineering failure，不占用`V71-F06`。

## V7.1 terminal note — B4揭示相反端点，不新增failure（2026-09-03）

- canonical=`run://worldsim_v71/WS-V71-B4-EVIDENTIAL-TSDF-01/
  20260903T120000Z__b4-evidential-tsdf-r1`；121 Selection Actors、41 hazard、137,940 rays；
- B4向120 Actors加入103,240个TSDF surface points，Chamfer改善`-83.961mm`、hit recall增加`4.786pp`，但hazard
  literal early相对恶化`106.530%`；它证明surface completeness可恢复，却不是同时满足ray safety的Pareto解；
- 与`V71-F04`（M0 UNKNOWN collapse）和`V71-F05`（M1 zero learned points）合并后，learned C1已穷尽计划内固定分支；
  该baseline正常完成，不登记`V71-F06`；
- prevention：不把B4当作候选、不在已消费Selection上混合/调权，不读Source Final/AV2救失败，不复开delete/selector；
  下一可用failure ID仍为`V71-F06`，当前无active hypothesis。

## V71-F05 — M1未生成任何可抽取field surface，learned C1终止

- 分类/状态：scientific + representation / terminal for learned C1；canonical=`run://worldsim_v71/
  WS-V71-M1-EVIDENTIAL-SURFACE-FIELD-01/20260903T113000Z__m1-evidential-field-s71102-r1`；
- 观察：88 Selection Actors上hazard/all/clear literal early相对下降`28.485/26.740/19.514%`，但Chamfer
  `0.245436→0.366110m`（`+120.674mm`），hit recall `43.935→29.569%`（`-14.366pp`）；3/5 gates；
- mechanism：88/88 Selection Actors的implicit field新增点为0，输出全部退化成hard anchors；冻结checkpoint在前64个
  train Actors的只读抽取同样`0/64`非零，说明不是单纯domain generalization，而是当前field/extraction未学成；
- training：659 Actors，geometry/evidence loss=`1.37663→0.89748`，physical=`1.15025→0.92208`，数值/GPU正常；
  loss下降不能替代可抽取surface合同；
- literature/open-source response：SelfOcc以SDF-induced连续ray weights监督几何，Object-Centric Occupancy Completion用
  implicit decoder生成动态尺寸shape；当前固定M1虽迁移二者机制，仍未把三态证据转成zero-crossing surface；
- decision：不调SCF band/top-k/evidence threshold，不做第二M1；不触发LoRA，因为失败首先位于field parameterization/
  extraction而非已证实的encoder容量瓶颈。Source final与AV2不读，避免用终测救source失败；
- closeout：learned C1停止，不回到selector/delete；只完成预注册B4非学习baseline并把负结果写入论文。下一可用failure
  ID=`V71-F06`。

## V7.1 M1 prevention note — 条件field不复用Selection调参（2026-09-03）

- `V71-F04`只触发计划中预注册的M1分支；17D输入、128D latent、4x128 decoder、seed71102、20+10 epochs、
  SCF band与三态定义在M1 Selection读取前一次冻结；
- PCGrad不执行：M0失败由74.51% UNKNOWN collapse主导，并无预训练gradient-conflict证据；Selection后补做PCGrad会成为
  outcome-guided M0 recovery；
- LoRA保持锁定：只有M1 train也无法逼近oracle且field/optimizer无异常才允许一次；不得用Selection失败直接触发；
- B4 TSDF完全非学习、只读build rays，不选择M1参数；M1失败则关闭learned C1，不追加field/grid/threshold sweep；
- source-final/AV2在M1冻结前保持未读，下一可用failure ID仍为`V71-F05`。

## V71-F04 — M0以UNKNOWN collapse换取首返回下降，surface Pareto失败

- 分类/状态：scientific / terminal for M0；canonical=`run://worldsim_v71/
  WS-V71-M0-RAY-SURFACE-DISPLACEMENT-01/20260903T111000Z__m0-ray-displacement-s71101-r3`；
- 观察：88 Selection Actors（36 hazard）、126,088 rays上，hazard/all/clear literal early相对下降
  `19.435/17.338/8.651%`，但Chamfer `0.245436→0.294181m`（`+48.745mm`），hit recall
  `43.935→33.117%`（`-10.819pp`）；retention两项100%，仅3/5 gates通过；
- mechanism：2,295个COMPLETE candidates中1,710个（74.51%）被判UNKNOWN，hazard/clear分别79.39/65.33%，
  9个Actor全候选消失；mean displacement仅`0.0673m`。M0主要重现delete/abstain，不是S1 oracle的surface relocation；
- literature/open-source response：NeurIPS 2020 PCGrad只在任务梯度负内积时投影冲突梯度；CVPR 2024 SelfOcc通过
  SDF-induced ray weights联合几何与深度；NeurIPS 2024 object-centric occupancy completion用长轨迹与implicit shape
  decoder生成动态尺寸占据。这里主故障是UNKNOWN collapse，且Selection已读，不能据结果追加PCGrad或调threshold/loss；
- decision：关闭M0，不复用其Selection调参或二次M0。S1连续几何oracle已可行而M0无法逼近，按预注册条件只解锁一次
  M1 Actor-local evidential implicit field；B4 TSDF baseline仍必须实现；source-final/AV2保持未读；
- artifact boundary：MODEL.pt保留为被拒candidate，不能称frozen主模型；canonical summary旧F02 placeholder由本条覆盖；
  下一可用failure ID=`V71-F05`。

## V71-F03 — 已在盘raw train/reserve语料不足1000条冻结门槛

- 分类/状态：data capability / resolved by role-disjoint processed recovery；canonical S2=`run://worldsim_v71/
  WS-V71-S2-ACTOR-CORPUS-01/20260903T102000Z__actor-corpus-r1`；
- 观察：120个主train加13个实际可索引reserve共133 scenes只产出721条合格刚体Actor（319 hazard、64 oracle target），
  低于冻结`>=1000`；wall=`72.40s`、cache=`20,022,368 bytes`；
- exposure：仅train读取；Selection/source-final/AV2均未读。M0 r2完成40轮oracle distill与24轮physical train后在同一
  manifest门槛处停止，未保存冻结模型、未读Selection；
- 根因：本机`drivestudio_raw_trainval`只保留174个左右有keyframe LiDAR的场景，当前split已经耗尽其中可用train，
  不是模型、显存或下载器失败；S2旧summary中的`V71-F02-required-at-closeout`占位由本条实际编号取代；
- literature/open-source response：nuScenes官方devkit要求full trainval经条款页面取得全部数据包；本机同时已有
  DriveStudio标准`instances_info/lidar/lidar_pose`处理资产，因此先复用同数据集现存资产，避免下载阻塞GPU研究；
- recovery：保持1000门槛和20/20 Selection/Final不动；只从96个与所有冻结角色不重叠的processed scenes按stride 5
  取2 Hz关键帧，以同一三模帧角色、Actor编译器和evidence物化合同补语料。metadata审计得到2,794条候选刚体轨迹；
- resolution：canonical recovery=`run://worldsim_v71/WS-V71-S2-PROCESSED-CORPUS-RECOVERY-01/
  20260903T110000Z__processed-corpus-recovery-r1`；42 scenes新增283 Actors，总数1004，Selection/Final/AV2 read=false；
  wall=`18.04s`、peak GPU/RSS=`0.0478/1.091GiB`；
- 防重复：不得把Selection/Final搬入train、降低minimum states/tracklet target，或把M0 r2 train-only曲线包装为模型结果；
  processed recovery达标后从头运行同一M0配置。下一可用failure ID=`V71-F04`。

## V71-F02 — consumer把producer临时NPZ误当成完整Actor cache

- 分类/状态：engineering / resolved；canonical failed run=`run://worldsim_v71/
  WS-V71-M0-RAY-SURFACE-DISPLACEMENT-01/20260903T102200Z__m0-ray-displacement-s71101-r1`；
- 观察：`glob("*/*.npz")`同时返回`<track>.tmp.npz`，producer原子rename后consumer打开旧路径而触发
  `FileNotFoundError`；
- exposure：发生于initial payload enumeration；0 standardizer/model/optimizer/training，Selection/source-final/AV2 read=false；
- 修复：只枚举不以`.tmp.npz`结尾的完整Actor文件；producer的临时写+atomic rename合同不变；
- 防重复：所有后续动态corpus consumer必须显式忽略临时后缀，不得通过取消原子写或重试损坏文件规避；
- recovery：相同M0科学配置/seed/run contract启动r2；next failure ID=`V71-F03`。

## V7.1 S2/M0 prevention note — I/O并行不放宽语料与Selection边界（2026-09-03）

- S2可以在M0训练期间继续物化，但每个Actor NPZ先写临时文件再原子rename；consumer只读取完整`.npz`；
- 120-scene主train若不足1000 tracklets，只允许读取metadata冻结的14个`train_reserve`；不得移动selection/final、删除难Actor
  或降低1000门槛；所有available train仍不足时登记下一可用failure ID并在Selection前停止；
- M0固定25维非hazard输入、128D residual、seed71101、oracle-distill→physical fine-tune两阶段；不得用Selection选择
  hidden/loss/seed/unknown threshold；
- physical training可动态吸收producer新增Actor，但checkpoint必须在corpus complete后冻结；Selection只读一次，source-final/
  AV2保持未读；
- 本里程碑冻结时只有实现与定向测试；后续r1工程失败已由`V71-F02`记录。

