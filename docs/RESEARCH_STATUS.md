# Research Status

## WorldSim V6.3 P5D objective-collapse diagnosis passed / P5R preregistered（2026-08-25）

状态：`v63_p5d_complete_objective_collapse_confirmed_p5r_preregistered`；active task=
`WS-V63-P5R-CONSTRAINED-SURFNCC-TRAIN-01`；active hypothesis=`WS-V63-H-P5R-001`；P6=`locked`。

P5D canonical=`run://worldsim_v63/WS-V63-P5D-AUTHORITY-COLLAPSE-DIAGNOSTIC-01/20260825T084844Z__authority-diagnostic-s0-r2`
正式`passed=true`：完整读取`48 train units`，safe-OCC/hidden-FREE/UNKNOWN点数分别为
`62454/495817/6036885`；固定target17四单元gradient probe=`79 batches/609288 points`；optimizer steps=`0`、training=
`false`、hard violations=`0`、wall=`796.702s`、peak=`0.323995 GiB`。selection/P6/calibration/confirmation/test均未读。
H-P5D-002 supported，H-P5D-001入口失败已由本run闭合，`V63-F20 resolved`。

机制结论分三层：

- `risk/authority composition failure`被排除为主因：safe-OCC的raw/post-projection/post-authority decision counts完全相同，
  正确class index order=`FREE/OCCUPIED/UNKNOWN`下均为`153/0/62301`，authority veto=`0`；hard projection没有抹掉任何
  usable OCC。
- representation保留弱排序但authority supervision区分不足：safe-OCC vs hidden-FREE的raw `P(OCC)` binned AUC=
  `0.722684`，但绝对均值仅`0.006459 vs 0.004181`；`q_AUTH` AUC仅`0.578070`，中位数=`0.0205 vs 0.0145`。
  authority target prevalence在safe-OCC/hidden-FREE/UNKNOWN仅=`10.31%/8.26%/9.24%`，说明证据authority标签与安全OCC
  语义只有弱对齐。
- `objective optimization collapse`被支持为主根因：safe-OCC retention component loss mean=`0.968547`、P50=`0.996666`，
  已近饱和；冻结权重后的tail training-term全模型gradient mean=`1.555512`，是retention `0.281250`的`5.531x`；仅看
  direct tail仍为`1.715x`，state-head为`1.732x`。77个同时非零batch的tail-retention gradient cosine mean/P50=
  `-0.411568/-0.370905`，显示系统性方向冲突。raw模型因而把safe OCC概率整体压扁，而不是最后policy拒绝。

P5D artifact的`DECISION_STAGE_COUNTS.json`唯一描述性错误是`class_order`文字写成UNKNOWN/FREE/OCCUPIED；实际
`torch.bincount`数组索引由冻结常量`FREE=0/OCCUPIED=1/UNKNOWN=2`决定，underlying counts、groups、distributions、
gradients与机制结论均正确。canonical run保持不可变，runner已修正未来label，登记`V63-F21 resolved`，不为metadata文字
重跑13分钟正确诊断。

下一唯一训练hypothesis=`WS-V63-H-P5R-001`已预注册为proxy primal-dual constrained recovery：从P5 epoch3模型权重
warm-start、fresh AdamW，保留模型/数据/dropout/FP16/CVaR/12 epochs/seed0/hard projection；把safe-OCC retention>=0.60、
emitted OCC coverage>=0.10和non-UNKNOWN coverage>=0.40作为约束，原离散rate更新dual、可微`P(OCC)*q_AUTH`更新model。
dual step固定0.01且不sweep。只有四个原始离散gate全过的checkpoint可叫candidate；best progress永不冒充candidate。

## WorldSim V6.3 P5 training capability passed / candidate rejected / P5D ready（2026-08-25）

状态：`v63_p5d_h002_entrance_recovery_ready`；active task=
`WS-V63-P5D-AUTHORITY-COLLAPSE-DIAGNOSTIC-01`；active hypothesis=`WS-V63-H-P5D-002`；P6=`locked`。

P5 canonical=`run://worldsim_v63/WS-V63-P5-SURFNCC-TRAIN-01/20260825T051530Z__surfncc-train-s0-r1`完成
`7 epochs/1792 optimizer steps`，wall=`12111.626s (3.364h)`、peak=`0.403084 GiB`、finite training=`true`、累计
hard violations=`0`，AMP initial/final均=`1024`、math SDPA与deterministic cuBLAS合同保持不变；train/selection仍严格为
`4/2 scene-disjoint scenes`，calibration/confirmation/test均未读。runner的`passed=true`只证明完整denominator训练、数值、
资源、硬投影和checkpoint产物能力，不是SurfNCC晋级结论。

冻结lexicographic objective选择epoch 3为**best training-objective checkpoint**：hidden-FREE tail=`0.0145069`、matched
rank surrogate=`0.0815163`、primary=`0.0960231`、hard violations=`0`。但同一checkpoint的safe-OCC retention=`0`、
emitted-OCC coverage=`0.0371977 < 0.10`、source-valid UNKNOWN=`0.861807 > 0.60`，因此不满足P1/P5防all-UNKNOWN与晋级
合同，`p5_candidate_promotable=false`。它不得称为best SurfNCC candidate，也不得解锁P6。epoch 6虽出现
retention=`0.0002227`，其primary=`0.1285593`且仍远未过门，patience按冻结规则终止；不据此追选checkpoint。

连续7个epoch与best checkpoint的hard violations均为0，支持observed FREE/OCC、contradiction、lifecycle与hard projection
继续保持冻结；当前失败位于无直接硬证据曲面的learned risk/authority路径。`SafeOCCRetention=0`比低coverage更直接指向
positive-authority collapse症状：危险/缺证据曲面与有正向OCC支持的安全曲面都被拒绝。根因尚不能在
representation/supervision、risk-authority composition与weighted-objective optimization之间武断选择，登记
`V63-F19 active_diagnostic_ready`。

P5D已预注册为仅训练集、零更新的机制诊断：全部48个train units分别统计safe-OCC/hidden-FREE/UNKNOWN三组的
`q_AUTH`、raw/post-projection `P(OCC)`、point/patch/proposal tail分布和三阶段decision转移；固定四个train target-17
units测量tail/retention/authority直接梯度幅值与tail-retention cosine。它不重采structural dropout、不读selection/P6/
calibration/H/T、不改threshold/gate/hard solver，也不增加seed/epoch/model/CVaR sweep。若证据支持objective collapse，
下一训练hypothesis只允许另行预注册proxy/primal-dual constrained optimization；简单把`lambda_ret`调大不授权。

P5D implementation已staged：配置=`configs/worldsim_v63/p5d_authority_collapse_diagnostic_v1.yaml`，runner=
`scripts/run_worldsim_v63_p5d_authority_diagnostic.py`。分布使用固定1000-bin streaming histogram并输出一张六面板图；
决策计数显式区分raw argmax、hard projection和authority veto；gradient probe同时报告raw/frozen-weighted全模型及分head
L2 norm。summary只验证finite/完整train denominator/zero hard violation/resource，不用新的质量阈值自动判根因。

H-P5D-001第一次formal入口在run directory、checkpoint/data read与GPU context前因新task namespace尚不存在而失败：runner
把`shutil.disk_usage`直接调用在不存在的`run_dir.parent`，触发`FileNotFoundError`；没有run leaf或科学结果。Python官方合同
要求`disk_usage(path)`接收已有filesystem path。登记`V63-F20 resolved_recovery_ready`；H-P5D-002只向上找到最近已存在父目录
做同一20 GiB disk检查，仍由runner随后创建唯一leaf。diagnostic groups、checkpoint、48+4 units、FP16、阈值、梯度、资源与
全部data locks不变，不新增smoke/regression矩阵。

## WorldSim V6.3 P4 capacity passed / P5 training unlocked（2026-08-25）

状态：`v63_p5_training_ready`；active task=`WS-V63-P5-SURFNCC-TRAIN-01`；active hypothesis=`WS-V63-H-P5-001`。

P3 canonical formal=`run://worldsim_v63/WS-V63-P3-SURFACE-CORPUS-01/20260824T154059Z__surface-dl-s20260824-r1`
正式通过：`6 scenes/72 targets`、`86,360 surfaces/111,282 patches/86,360 proposals/11,583,001 points`；
surface type=`3,042 route-support / 82,499 static-disocclusion / 790 actor / 29 actor-swept`。minimum normal-valid=`1.0`、
maximum patch=`940<=2048`、maximum surface=`181,752`、8/8 negative contracts、missing fields=`[]`、source overlap=`0`。
output=`333,197,992 bytes`，wall=`47,568.466s`（`13.213h`），maximum unit wall=`3,334.282s`；source在启动和终态均clean，
prototype/calibration/confirmation/test read均false。P3 hypothesis supported，P4 H-P4-002 execution正式解锁。

P3终态前统计语义审计发现，run内`hidden_free_count`实际保存的是全部`target==FREE`，缺少
`method==UNKNOWN && !method_contradiction`条件；point payload中的method/target/contradiction字段正确，P4/P5 loader及
P5 loss/selection均从点字段重算，故语料和模型路径不受影响。正式run保持不可变；72个原始NPZ一次重算得到target
FREE/OCC/UNKNOWN=`1,545,584/335,050/9,702,367`、correct hidden-FREE=`688,837`，旧summary的`1,545,584`
不得按hidden-FREE引用。未来materializer以additive v2同时区分这些字段，登记`V63-F16 resolved`，不重跑正确语料。

P4 H-P4-001已在任何真实P4运行或quality read前撤回：对P3最先完成40 units的method-only结构统计显示，40/40均有
proposal超过8192 points，最大=`173488`，其完整patch set最大=`417`。只看largest proposal首个chunk会系统性丢失
proposal interaction，不能证明冻结合同。按Set Transformer/Perceiver的分层set encoding迁移为H-P4-002：点图仍以
8192-point chunks有界执行，但先汇总完整proposal的全部patch tokens，再运行2层patch attention与唯一proposal token；
训练chunk以可微token替换no-graph cache中的对应位置。模型、units、2 optimizer steps、accum4与22 GiB ceiling均不变。
登记`V63-F10 resolved_preexecution`；P4实现/配置/预注册已staged，P3 formal pass后执行已解锁。
P4原`cvar_gradient_nonzero`曾用会同时收到BCE梯度的hidden-free head总梯度做代理，可能假阳性；现用
`autograd.grad(proposal_cvar.mean(), state/hidden-free/authority heads)`直接检查CVaR图，聚焦synthetic三条head路径均
finite/nonzero，登记`V63-F15 resolved_preexecution`。既有gate含义被校正，未增加新gate或实验分母。

P4 H002 r1=`run://worldsim_v63/WS-V63-P4-CAPACITY-01/20260825T045854Z__capacity-h002-s0-r1`在11.181s终态
`passed=false`：完整train/selection proposals执行，peak=`0.1961 GiB`、loss finite、direct CVaR三head gradient nonzero、
proposal-token gradient nonzero、hard violations=`0`、checkpoint reload成功；但FP16总gradient出现nonfinite，且CUDA
attention相同/重载forward max abs diff均=`9.0599e-6`，未过冻结的finite/exact-0 gate。PyTorch官方说明GradScaler初始
scale可导致FP16 overflow，CUDA SDPA后端也有不同确定性；唯一有界恢复固定AMP initial scale=`1024`并禁用flash/
memory-efficient SDPA、只用deterministic math backend。模型、FP16、units、steps、loss、gate与22GiB ceiling不变；
当时登记`V63-F17 active_recovery_ready`；现已由r3闭合为resolved。r1保持不可变，不写成算法失败。

P4 H002 r2=`run://worldsim_v63/WS-V63-P4-CAPACITY-01/20260825T050400Z__capacity-h002-s0-r2`在第一次CUDA math
attention forward、任何optimizer step或summary前被deterministic runtime拒绝：cuBLAS矩阵运算要求进程启动前设置
`CUBLAS_WORKSPACE_CONFIG`。r2叶目录为空，quality/calibration/confirmation/test均未读，不能写成F17恢复或capacity失败。
按NVIDIA cuBLAS与PyTorch官方确定性合同，r3在launcher和pre-torch-import runner双层固定`:4096:8`；约24 MiB workspace
开销仍远低于22 GiB ceiling。除该运行时前置条件外，r1已冻结的AMP scale=`1024`、math SDPA、deterministic algorithms
及所有模型/数据/FP16/dropout/loss/optimizer/steps/accum/gates均不变。当时登记`V63-F18 active_recovery_ready`；现已由
r3闭合为resolved。r3仍是F17唯一有界恢复的第一次实际执行。

P4 canonical r3=`run://worldsim_v63/WS-V63-P4-CAPACITY-01/20260825T051200Z__capacity-h002-s0-r3`在
`11.863s`正式`passed=true`：train/selection各`2 complete proposals / 16 chunks`，maximum full proposal=
`117,663 points / 263 patches`；peak=`0.256589 GiB`，AMP scale initial/final均=`1024`，loss与unscaled gradient finite，
direct CVaR三head及proposal-token gradient nonzero，hard violations=`0`，checkpoint reload成功，repeat/reload max diff均
`0.0`。quality/calibration/confirmation/test read均false。H-P4-002 supported，`V63-F17/F18 resolved`，P4收口并只解锁
已预注册的P5完整denominator训练。

P5完整denominator实现已staged并由P4 pass解锁、尚未执行：每个complete proposal在每epoch只生成一个semantic dropout selector，所有chunks
继承全局actor/static、安全标签与point count；masked evidence同步移除temporal/observed-actor与证据派生authority通道，
temporal-window则由保留sweeps重算。selection把全部chunks的hidden-FREE probability连接后计算exact proposal CVaR，
完整patch context驱动proposal attention；训练仅声明memory-bounded stochastic CVaR surrogate。最终决策先保留硬投影，
只把method-UNKNOWN且低authority的learned OCC转UNKNOWN，coverage/retention/accuracy共享同一decision。逐loss审计又发现
ranking若只按chunk共现配对会漏掉完整unit的nearest-size pairs；现从unit metadata一次生成同stratum一对一匹配，并让当前
完整patch-token cache通过可微proposal attention/risk head每unit计算一次，未采用Cross-Batch Memory的stale queue。
selection端又发现曾把24个selection units合并后跨scene/frame配对；现与训练一致，严格在完整scene/frame unit内匹配并对
有pair的unit等权平均，不允许跨案例规模巧合改变checkpoint排序，登记`V63-F14 resolved_preexecution`。
另发现surface-wide edges会随相邻patch是否同chunk而漂移，现把两层6-neighbor local aggregation绑定完整冻结patch（从不切分、
max2048），跨patch交互只走完整proposal attention；登记`V63-F11/F12/F13 resolved_preexecution`。聚焦审计中，
modular-forward等价检查覆盖12 outputs且max abs difference=`0.0`；另以两个完整patch跨packing验证有向边数均为`4`，
分属不同chunk的safe/unsafe proposals仍生成冻结pair=`[(0,1)]`，跨unit safe/unsafe则为`0 pair`。没有真实P5
data/training、threshold搜索或新增
smoke/regression矩阵。

P3 canonical probe=`run://worldsim_v63/WS-V63-P3-SURFACE-CORPUS-01/20260824T153526Z__surface-probe-s20260824-r6`
已通过：`1 unit / 191 surfaces / 498 patches / 191 proposals / 152,226 points`，output=`3,055,106 bytes`，
wall=`201.356s`。minimum normal-valid=`1.0`、maximum patch=`635<=2048`、8/8 negative contracts、
`missing_point_feature_fields=[]`；逐sweep state/contradiction、exact signed distances、patch-local coordinate、normalized
ray order与全部native/evidence/actor/authority字段均存在。按F16正确重算的target FREE/OCC/UNKNOWN=
`19,609/3,891/128,726`、hidden-FREE=`8,311`；旧registry的`19,609`不得按hidden-FREE引用。
prototype/calibration/confirmation/test read均false。

P3 probe gate与随后72-unit formal均正式通过（历史`V63-F03–F07`保留且resolved）；probe外推的2.0h低估了大量
large surfaces，实际formal为13.213h，但仍低于24h资源线。下一步只执行预注册P4 H-P4-002，不再运行P3 probe/formal。

P3 r5=`run://worldsim_v63/WS-V63-P3-SURFACE-CORPUS-01/20260824T152843Z__surface-probe-s20260824-r5`
完成`191 surfaces/498 patches/152,226 points/3,029,206 bytes`，wall=`188.725s`、runner `passed=true`；新增
signed distances、patch-local xyz、behind-hit、四类temporal counts、normalized ray order与actor observed-hit均可读取。
但P4 structural-dropout loader设计审计确认：聚合temporal counts不能忠实执行冻结的整段`temporal_window` dropout，
必须保留每个method sweep的state/contradiction。该发现仍属于同一`V63-F06` frozen-schema completeness根因；r5只记
aggregate schema capability，不放行formal。

r6增加`[point,sweep]` temporal state/contradiction矩阵，并把P1必需字段清单写入P3配置，runner只做一次直接缺字段检查。
VideoMAE/Masked Spatio-Temporal Structure Prediction支持连续时空mask必须保留时间结构这一迁移，但mask比例仍使用P1冻结
的25%，不迁移其预训练目标或高mask ratio。r6通过后不再加probe，直接72-unit formal。

r6窄接口检查首次把scene名误当processed index并访问`trainval/000`，在文件打开前失败；立即读取冻结cohort得到
`scene-0071 -> processed_index 68`后，同一检查通过，per-sweep shapes=`[3,300,300,40]`。登记`V63-F07 resolved`，
未创建run、未改代码/科学合同，也不增加测试矩阵。

P3 r4=`run://worldsim_v63/WS-V63-P3-SURFACE-CORPUS-01/20260824T152300Z__surface-probe-s20260824-r4`
通过几何/资源门：`191 surfaces / 498 patches / 152,226 points`，minimum normal-valid=`1.0`、patch max=`635`、
8/8 negative contracts、runner `passed=true`，wall=`194.306s`、output=`2,429,675 bytes`。但formal放行前与P1 frozen
point schema逐字段对照发现：payload尚缺signed FREE/OCC distance、patch-local xyz、behind-hit与第四个temporal count，
且`ray_hit_order`误存metric distance。r4只证明geometry capability，不能升级为完整P3 pass。

按SciPy官方exact EDT补method-visible FREE/OCC signed distance；按Point Transformer的relative-position原则补patch-local
coordinate；显式保存behind-hit、temporal UNKNOWN、ray distance和bundle内normalized hit order，并补actor observed-hit。
这些是预冻结输入的实现补全，无新超参、无quality选择，不改变任何proposal/topology/label/gate；登记`V63-F06 resolved`，
r5为最后一个schema-complete probe，通过后直接72-unit formal。

P3 r3=`run://worldsim_v63/WS-V63-P3-SURFACE-CORPUS-01/20260824T151618Z__surface-probe-s20260824-r3`
首次完整构建出`191 surfaces / 498 patches / 152,226 points`，wall=`194.540s`、output=`2,429,273 bytes`；
native/evidence/registry/patch bounds/8项负向合同均完成，但`101`个微小static components（85个singleton，其余3–11
voxels）存在离散对称法向量抵消，minimum normal-valid=`0`，所以probe诚实未过。Gradient-SDF一手论文说明SDF梯度
在medial axis最近面不唯一处存在奇异性；Open3D官方接口也要求显式viewpoint orientation。r4仅对“外露面和+centroid
方向都为零”的退化点使用target-sensor viewpoint确定方向，不删proposal、不改topology/patch/cohort/gate，登记
`V63-F05 resolved`。

P3 r2=`run://worldsim_v63/WS-V63-P3-SURFACE-CORPUS-01/20260824T151429Z__surface-probe-s20260824-r2`
因外层launcher提前创建了本应由runner原子创建的新run directory而在入口触发`FileExistsError`；0 unit、0 surface、
0 quality read。Python官方`pathlib.Path.mkdir(exist_ok=False)`合同确认目标已存在必须失败。恢复仅移除launcher的叶目录
预创建，父目录已存在；登记`V63-F04 resolved`，冻结配置和source commit均不变，revision 3已就绪。

P3 r1=`run://worldsim_v63/WS-V63-P3-SURFACE-CORPUS-01/20260824T150842Z__surface-probe-s20260824-r1`
在surface extraction和任何quality判据之前失败：native-to-target helper错误地对长度分别为`300/300/40`的三条轴执行
`numpy.stack`，触发same-shape合同错误。按NumPy官方接口将未被消费的轴元组原样返回，并在重跑前收紧route-support
局部surface type更新、法向量有效统计与显式native-valid报告；均不改变冻结proposal/topology/science合同。r1保持不可变，
登记`V63-F03 resolved`，同配置revision 2已就绪。

P3实现/输入合同已冻结：static proposal=`native occupied + observed OCC - actor envelopes`，Actor proposal按method-visible
current/swept actor ID分开；先声明volume再取6-connected boundary，拓扑不改几何。每点保存native mapping、normal、
method/target/contradiction、逐method sweep时序support、ray bundle/order、actor identity/lifecycle与authority bits；patch按
lexicographic BFS冻结为64/512/2048。仅运行一个`scene-0071/f017` probe，通过即72-unit formal。

P2D native-to-pointwise interface与唯一 formal 已预注册：冻结 V6.2 P5 best，不训练、不调阈值，把 P2 完整 native
logits/BEV 按真实网格坐标映射到 legacy 0.2m grid，并保持原 legacy28/P6 gate、method-before-O_eval 顺序。该诊断只有
一次formal，不做capacity probe、seed或threshold sweep；结果无论正负均只裁决 prototype vs pointwise 根因。

P2D canonical=`run://worldsim_v63/WS-V63-P2D-NATIVE-POINTWISE-DIAGNOSTIC-01/20260824T145924Z__native-pointwise-s0-r1`
已正式rejected：Native B2=`4/28 ACCEPT,4/4 false-safe`，接受集合仍是四个scene-0242 missing-route-support cases；
R10=`2/3`、Actor/static gain=`0/2`、mask-area=`0.094024`、FREE conflict mean/worst=`0.045783/0.092105`、
UNKNOWN=`0.639211`、safe-OCC retention=`1.0`、hard violations=`0/939206`。恢复native feature没有改变决策集合，
因此prototype不是主因，pointwise/mean-query结构根因成立，登记`V63-F02 active`。

负结果后补查CVPR 2024 Point Transformer V3官方实现与visibility-aware surface reconstruction：可迁移点是高效确定性
point neighborhood/serialized patch以及将FREE visibility显式置于surface边界，而不是扩大网络或重新调阈值。P1冻结
的6-neighbor surface topology + patch CVaR方案保持不变；P2D不做recovery，直接进入P3 corpus。

P2 原生接口已实现并冻结：复用已验证的 official IR-WM current forward，每 target 直接保存完整
`200x200x16x17` logits、`200x200x256` BEV latent、argmax/entropy/margin/source-valid 为 memory-mappable arrays；
不再依赖 V6.2 query-deduplicated sidecar，也不存在 prototype。首个 formal denominator 固定为 Tier D 72 targets +
Tier L 4 targets；C/H/T按阶段解锁后生成。只允许一个 `scene-0071/f017` capability probe，通过即运行76-target formal。

唯一 P2 probe=`run://worldsim_v63/WS-V63-P2-NATIVE-SIDECAR-01/20260824T144921Z__native-probe-s1-r1` 已通过：
1 scene/1 target，完整原生数组=`46,081,727 bytes`，峰值 GPU=`4.0496 GiB`、wall=`25.19s`；fresh memory-map reload、
shape/finite均成立，prototype/target/calibration/confirmation/test read均为false。下一步直接formal，不追加probe。

P2 formal canonical=`run://worldsim_v63/WS-V63-P2-NATIVE-SIDECAR-01/20260824T145110Z__native-dl-s1-r1` 已通过：
8 scenes/76 targets（D=72,L=4），完整 sidecar=`3,502,211,483 bytes`，wall=`200.763s`；maximum worker peak=
`4.1314 GiB`、two-worker peak-sum upper bound=`8.2623 GiB`。76/76 native tensors完整、finite并可fresh mmap reload；
prototype/target/calibration/confirmation/test read均为false。P2 hypothesis成立，failure ledger delta=`none`；P2D已解锁。

P1 已完成一手文献/官方仓库审计。RELIOcc/OCCUQ/alpha-OCC/EvOcc 已覆盖 Occupancy reliability、uncertainty、evidence
与 conformal set 的单项；QueryOcc 覆盖连续4D query；Point/Set Transformer覆盖结构编码；CRC/NCRC/structured
segmentation覆盖独立校准；CVaR与visibility-aware reconstruction覆盖尾部损失与FREE约束。未发现把原生 Occupancy
feature、proposal surface、exact hard evidence、surface CVaR、positive OCC authority 与 case-level admission risk
统一成驾驶 world compiler 的直接重合。novelty gate 只对该组合通过，任何单组件均不主张贡献。

在任何 V6.3 quality read 前已冻结：完整 IR-WM `17D logits + 256D BEV` 原生 sidecar；6-connected proposal boundary；
patch `64/512/2048`；256D two-block/two-transformer surface encoder；CVaR alpha=`0.90`；全部 loss/训练超参；Tier
D/C/H/T/L scene-disjoint cohort；case score、`0..1 step .025` fixed-sequence exact-binomial calibration；risk target=
`0.05@95%`；anti-trivial、P6–P10 gates 与单卡资源合同。C/H/T分别为6/3/4 fresh scenes、72/36/48 target cases；H/T
保持 sealed。详见 `docs/autoresearch/worldsim_v63/P1_NOVELTY_PROTOCOL_FREEZE.md`。

V6.2 的 `v62_cpsc_lite_family_closed_negative`、`V62-F06 recovery exhausted` 与 P7/P8 未解锁结论保持不变。
按 V6.3 计划，V6.2 已经由临时 integration branch 以 fast-forward 合入 `main`，定向 projection test 在正确的
`PYTHONPATH=.` 合同下为 `1 passed`，随后从同步后的 `main` 新建并推送独立分支
`research/worldsim-v6.3-surface-tail`。首次定向测试因入口路径写错、第二次因未设置 repo-local import path 而在 collection
阶段失败，均未读取数据或产生科学结果；已登记 `V63-F01 resolved`。

V6.3 北极星冻结为：使用原生 17D Occupancy logits、256D BEV latent 与真实硬证据，对完整 proposal surface 做联合
编码与 patch/proposal 尾部风险控制，经 scene-disjoint case-level 独立校准后才允许 singleton OCC 写入 Physical State。
禁止 prototype bridge、legacy O_eval 调参、voxel-level 伪独立校准、mean query risk 替代 surface tail、用 all-UNKNOWN
冒充安全，以及新建哈希/校验和/指纹机制。默认资源为单卡 RTX 3090 24GB；只有冻结最小配置在一次合法资源恢复后仍
失败，才进入 `blocked_resource` 并向用户申请升级资源。

P0/P1/P2/P2D/P3/P4 当前完成；P5 H-P5-001 training ready。
calibration/confirmation/test保持sealed。

## WorldSim V6.2 CPSC-Lite family closed negative（2026-08-24）

状态：`v62_cpsc_lite_family_closed_negative`；active task=`none`；P7/P8=`not unlocked`。

V6.2 已从 V6.1 最小实验负结论的 `main@c8e9dee` 新建分支 `research/worldsim-v6.2-cpsc`。V6.1 终态
`v61_minimum_experiment_closed_negative` 保持不可变；V6.2 不再遍历第三个 Occupancy backend，而是研究 CPSC：把真实
FREE/OCC 作为前向硬约束、learned Occupancy 作为可推翻软先验，并对证据不足或矛盾区域输出 UNKNOWN。

P0 冻结了 legacy28 机制门槛、fresh development/calibration/confirmation/test 的数据纪律、IR-WM frozen 边界和
单卡 3090 资源上限。按用户约束，V6.2 新产物不加入哈希、校验和或指纹，也不复制 V6.1 的重审计/重门控体系；身份以
逻辑路径、语义版本、task/run ID 和 Git 提交记录为准，只保留与科学结论直接相关的精简验证。

P1 只读一手论文/官方仓库后未发现同时覆盖“硬观测 FREE/OCC + 可推翻 learned prior + selective UNKNOWN + proposal
bake/collision asset + world-simulation false-safe”的直接重合，novelty gate 通过。但单组件均有强先例：ReliOcc/OCCUQ
覆盖可靠性与 uncertainty，EvOcc 覆盖冲突/未知证据，alpha-OCC 覆盖分层保形集合，QueryOcc/DIO 覆盖 4D query 与
留出补全，HardNet/可微投影覆盖硬约束，MultiSafe 已把 conformal 用于 false-safe 控制。因此 CPSC 的可主张贡献被收窄为
`hard-evidence-constrained physical-state compilation` 的完整任务/接口/评测组合，不能把 uncertainty、三态、query、
projection、conformal 或 evidence dropout 单独写成新贡献。

P3 已实现独立于 V6.1 重审计 runner 的小型 PyTorch closed-form projection，约束优先级固定为
`contradiction > observed FREE/OCC > lifecycle > soft prior`。单个 synthetic contract test=`1 passed`；真实
scene-0048/f052 `O_method` fixture 抽样 48 query，hard FREE/OCC、contradiction/lifecycle→UNKNOWN 与 simplex 最大误差
均为 `0`，梯度 finite，未约束 query 梯度非零；第二个 fresh process 结果一致。canonical=
`run://worldsim_v62/WS-V62-P3-FEASIBILITY-PROJECTION-01/20260824T080731Z__projection-s0-r1`。

P2 已冻结 6 个 scene-disjoint development scenes：`scene-0071/0317/0450/0862/1012/1089`。该集合完整复用 V4 在
V6.2 结果出现前、仅按 metadata 冻结的 validation 六场景，不从已有质量结果里选子集；6/6 均属于 nuScenes 官方
train，覆盖 Boston/Singapore、day/dusk/night、dry/rain。每场固定 12 个 target=`17..182`、步长15，共72 units；
method candidate offsets=`[-6,-4,-2,0]`，每个 target 轮换留出一个 dropout sweep，其余三个作为 method input，独立
target offsets=`[-5,-3,-1,1]`。所有 processed scene 均有 6-camera、LiDAR、pose，最短 scene 191 帧，覆盖最大 offset。

P2 materializer 已在 scene-0071/f017 做单 unit、无质量读取的资源/类别探针。r2 canonical probe=
`run://worldsim_v62/WS-V62-P2-EVIDENCE-QUERY-DATASET-01/20260824T082318Z__query-probe-s20260824-r2`：100k
queries，六类 candidate pool 全部非空，source role overlap=`0`，disk=`2,036,102 bytes`，wall=`2.96s`。method 包含
168,487 FREE、11,936 OCC、4,923 contradictions、6,854 motion-compensated actor hits；target supervised query=
38,088/100,000。

r1 probe 在科学执行前暴露 V6.1 evidence state=`U/F/O 0/1/2` 与 P3 model class index=`F/O/U 0/1/2` 的潜在歧义，
没有被用于训练或结果；r2 已显式同时保存 `*_evidence_state` 与 `*_class_index`，登记 `V62-F02 resolved`。其后启动的
formal 仍不读取 occupancy quality、proposal outcome、O_eval、confirmation/test。

首次72-unit formal r1=`20260824T082601Z__query-dataset-s20260824-r1` 在 `scene-1012/f152` 暴露
instantaneous actor envelope 空池并终止，未形成最终 manifest、未进入训练或质量裁决。元数据定位显示该帧仍有4个 actor，
只是全部位于冻结 ROI 外；其中一个 actor 在可见 method sweep f146 穿过 ROI。按 QueryOcc 相邻时刻查询与动态稀疏
query 时序传播的思路，actor support 已改为 current target envelope 与 visible method-sweep envelopes 的并集；它只
影响 actor query 坐标，不把 box 变成 hard OCC，也不读取 dropout/target evidence或改任何配额。

定点 r5=`20260824T083403Z__actor-sweep-repro-s20260824-r5` exit=`0`：current actor envelope=`0`、visible
swept envelope=`450` voxels、actor-type queries=`15000/15000`、total queries=`100000`。`V62-F03 resolved`；据此
从恢复提交直接重跑 formal r2，没有追加更多 smoke。

formal r2 canonical=
`run://worldsim_v62/WS-V62-P2-EVIDENCE-QUERY-DATASET-01/20260824T083654Z__query-dataset-s20260824-r2` 已通过：
`6 scenes / 72 units / 7,200,000 queries`，每场12 units，method/dropout/target source roles=`216/72/288` 且
交集=`0`；六类 query 总数依冻结比例为 `1.8M/1.08M/1.8M/1.08M/1.08M/0.36M`。六类最小候选池=
`156406/6860/6533/382175/167/2446`，72/72 combined actor pools 非空；唯一 current-envelope 空 unit 已由
visible sweep support覆盖。target supervised rows=`2,639,153`，磁盘=`155,249,746 bytes`，wall=`151.47s`，
confirmation/test read=`false/false`。failure ledger delta=`none`（正式成功不灌入 failure ledger）。

P2 已收口，P4 预注册为复用 V6.1 frozen IR-WM environment/weights，在新 development scenes 上 batch1、scene worker
串行或最多2个，抽取与同一 query coordinates 对齐的 prior logits/selected features；不训练 IR-WM、不读取 target
evidence。用户约束覆盖原计划里的内容寻址/model hash 项：P4 只记录逻辑路径、语义版本、backend identity、task/run ID
与 Git 提交，不新增哈希、校验和或指纹。

P4 最薄接口已实现、尚未执行 GPU probe：官方 current occupancy head 提供 `200×200×16×17` logits，current
`ref_bev` 提供 `200×200×256` latent。sidecar 不按100k query重复拷贝 latent，而保存唯一3D prior cells、唯一2D BEV
cells和两组 query→cell 索引；source extent 外 query 显式标为 prior-invalid，留给 CPSC 输出 UNKNOWN/依赖硬证据。probe
固定 `scene-0071/f017`、history=`[7,12,17]`，成功后直接2-worker全量72 units。failure ledger delta=`none`。

P4 probe r1=`20260824T085711Z__prior-sidecar-probe-s1-r1` 在 plugin import、GPU forward和sidecar前被
`Ninja is required` 阻塞。env 内的 `bin/ninja` 已存在，根因是controller使用隔离 Python但未把同一 env bin prepend
到 PATH。按 PyTorch cpp-extension 官方查找机制和 V6.1 成功 worker合同，恢复只补齐 PATH/PYTHONNOUSERSITE/
OMP/MKL/CUDA arch 环境，不安装依赖、不改科学输入；`V62-F04 resolved`，随后重跑同输入 probe r2。

同输入 probe r2 canonical=
`run://worldsim_v62/WS-V62-P4-IRWM-PRIOR-SIDECAR-01/20260824T085956Z__prior-sidecar-probe-s1-r2` 已通过：
100,000 queries 中97,434个 source-valid，去重为27,467个3D prior cells与5,633个2D BEV cells；输出=
`4,002,647 bytes`，worker peak=`4.0496GiB`，official forward=`1.066s`，controller wall=`98.29s`（含首次native
extension启动）。missing keys仅V6.1已知的两项官方删除 `reference_points`，unexpected=`0`；target evidence、
confirmation、exact-once test均未读。P4 进入 formal 6-scene/72-target/max2-worker，不再追加 smoke。

P4 formal canonical=
`run://worldsim_v62/WS-V62-P4-IRWM-PRIOR-SIDECAR-01/20260824T090444Z__prior-sidecars-s1-r1` 已完成：
6 scenes、72/72 targets、7.2M query mappings；source-valid=`6,811,702`（94.607%）、invalid=`388,298`，每unit
valid最小=`91,305`。unique prior cells/unit=`23,129..38,500`，unique BEV cells/unit=`4,973..10,364`；sidecars=
`368,162,079 bytes`。72次official inference合计=`119.41s`，formal wall=`176.27s`，single-worker peak=
`4.1265GiB`、two-worker peak sum upper bound=`8.2523GiB`。6/6 workers unexpected keys=`0`，仅保留相同的两项
官方删除 key记录；target evidence/confirmation/test read=`false/false/false`。failure ledger delta=`none`。

P5 已预注册为只训练 prior adapter、query decoder、evidential head与projection-compatible residual；IR-WM 进程已退出且
权重保持 frozen。输入仅为P2 query/evidence与P4 sidecars，development内部划分和目标函数在启动训练前冻结；不读取
legacy28 O_eval、confirmation或exact-once test，也不新增哈希/校验和/指纹。先审计最薄 loader/model/loss与单卡batch
预算，再直接进入 bounded training，不铺设多轮 smoke/regression 矩阵。

P5 design 已冻结：train=`scene-0071/0317/0862/1012`（48 units），scene-disjoint selection=
`scene-0450/1089`（24 units）；后者只按预先冻结的 Boston rain 与 Holland Village night metadata选择。模型输入为17维
prior logits、entropy/tri-state/source-valid、256维BEV latent、method evidence、normalized coordinates与actor support；
query type、dropout evidence和target evidence明确不进模型。loss固定为query/evidential/hidden-FREE/safe-OCC/
actor-temporal/prior-preserve，hard-conflict target不反向要求模型违反method硬证据。训练配置=`FP16, batch16384,
accum2, AdamW 3e-4, max12 epochs, min4/patience3`，仅运行seed0。先做一次8 optimizer-step capacity probe，
通过后直接全量训练；failure ledger delta=`none`。

P5 capacity canonical=
`run://worldsim_v62/WS-V62-P5-CPSC-LITE-TRAIN-01/20260824T092410Z__cpsc-lite-capacity-s0-r1` 已通过：
3个train units、1个selection unit、8 optimizer steps，608,366 parameters，prior/query dims=`278/13`，FP16 peak=
`0.3724GiB`、wall=`4.91s`，finite best objective=`2.13624`，hard violation=`0`。8步 learned 与projection-only
只作非退化诊断：target accuracy=`0.4233 vs 0.3713`、safe-OCC retention=`0.9569 vs 0.9502`、UNKNOWN fraction=
`0.1773 vs 0.0767`，但hidden-FREE false-OCC=`0.2680 vs 0.2616` 尚未改善；因此不宣称质量pass/fail，只说明
loader/forward/backward/projection/resource合同成立。下一步直接formal 48/24-unit bounded training，不调loss/threshold。

P5 formal canonical=
`run://worldsim_v62/WS-V62-P5-CPSC-LITE-TRAIN-01/20260824T092636Z__cpsc-lite-train-s0-r1` 已通过：
48 train units、24 scene-disjoint selection units，608,366 parameters；9 epochs/1,512 optimizer steps 后按冻结 patience
提前停止，best epoch=`5`，best selection objective=`2.099165`。FP16 peak=`0.3724GiB`、wall=`341.66s`；BEST/FINAL
模型各约2.45MB，hard projection violations=`0/1,286,134`。

最佳 learned 相比同一 selection 的 projection-only：hidden-FREE false-OCC=`0.38457 vs 0.45371`，绝对下降
`0.06914`、相对下降`15.24%`；safe-OCC retention=`0.90106 vs 0.90068`，没有用UNKNOWN换取安全OCC丢失；target
accuracy=`0.48376 vs 0.35677`。learned UNKNOWN=`0.24758`、unconstrained UNKNOWN=`0.46960`，并非all-UNKNOWN。
target evidence仍只作监督，query type/dropout/target均未进入model features；IR-WM不驻留，legacy O_eval、confirmation、
exact-once test均未读。P5 hypothesis在冻结配置上成立，failure ledger delta=`none`；不追加seed/smoke矩阵。

P6 现按计划只做一次 frozen legacy28 matched mechanism benchmark：读取 V6.1 frozen IR-WM sidecar、ME0 O_method、ME1
O_eval和R10 comparator，不重跑IR-WM。主门槛固定为`ACCEPT>=5/28, false-safe=0, R10=3/3 retained, >=1 Actor
新增, >=1 static/disocclusion新增, accepted mask-area>=12%, accepted FREE conflict mean/worst<=0.05`；同时报告
UNKNOWN/ABSTAIN和oracle accepted surface safe-OCC retention，防止all-UNKNOWN。legacy28只裁决机制，不宣称fresh
scene generalization；失败时只允许先查一手来源后从projection architecture、evidence dropout、set-valued head三者中选
一个机制级恢复，不做threshold/grid/window/backend/model-size sweep或删case。

P6 接口审计发现计划文字与 canonical artifact 不一致：V6.1 ME3R 只保存`200×200×16 argmax class`，没有P5所需的
17 logits/256D BEV；同时B2需要尚未允许的Tier-C threshold calibration，B4没有no-evidence-dropout checkpoint，full
M0的grouped conformal按阶段计划要到P8才产生。`V62-F05 resolved_for_artifact_bounded_P6`：参考ProtoSeg的训练特征
类原型，只用P5四场景train split按17个argmax class求query-weighted logits/BEV均值，legacy查表；明确承认它不能恢复
逐cell uncertainty。不得重跑IR-WM、用O_eval拟合bridge或伪造B2/B4/M0。

只读失真审计覆盖24个selection units/2.4M queries，bridge fit未读selection target：full/bridge预测一致=
`0.896898`；bridge hidden-FREE false-OCC=`0.399349`，仍优于projection-only=`0.453707`；safe-OCC retention=
`0.872897`、target accuracy=`0.452581`、UNKNOWN=`0.221945`、hard violations=`0`。P6 formal固定执行B0 replay、B1 hard
clip、B3 evidential-no-projection与B5 pre-conformal；B5为primary，M0明确defer到P8。anti-trivial固定safe-OCC retention
`>=0.50`和source-valid UNKNOWN`<=0.50`。接口实现完成后只做一次formal，不增加bridge/model/threshold sweep。

P6 canonical=`run://worldsim_v62/WS-V62-P6-LEGACY28-ME-01/20260824T095529Z__legacy28-s0-r1`，source=
`d14827d`，正式 rejected。B0=`10/28,10 false-safe`；B1=`10/28,10 false-safe`，虽把accepted mean/worst FREE
conflict降到`0.05058/0.11722`，仍未触发projection-only Stop 1。B3与B5均=`4/28,4 false-safe`、mask-area=
`0.09402`；B5只保留R10 `2/3`、Actor新增=`0`、static新增=`2`。hard projection=`0/939,206 violations`，oracle
surface safe-OCC retention=`1.0`，但source-valid UNKNOWN=`0.82735`，说明主要失败是缺失logits/BEV的feature-shift与
hidden surface authority，不是硬约束或已知OCC丢失。wall=`47.20s`、peak=`0.5319GiB`，IR-WM未重跑。

`V62-F06 active`。按P6 stop rule与一手missing-modality文献，只授权一次evidence-dropout recovery：student从P5 best
继续训练，train query以`p=0.5`替换为train-only class prototype，frozen full-feature teacher提供`0.25×KL`一致性；其余
P5 task loss不变。固定`AdamW1e-4, FP16, batch16384, accum2, max6/min3/patience2, seed0`，pure prototype selection
一次选点；不读legacy O_eval、不加capacity smoke。checkpoint冻结后只运行一次相同P6 gate的P6R；失败则关闭
CPSC-Lite，不再选择第二种机制恢复。

P6R 已按预注册实现：同一608,366-parameter CPSC-Lite student/teacher均从P5 best初始化，teacher冻结；每个
train query独立以`p=0.5`切换到train-only class prototype logits/BEV，student损失为原P5 task loss加`0.25×teacher→
student base-probability KL`。selection固定pure-prototype view，并同时报告full view；P5 evidential anneal从best epoch继续，
不重置。配置=`configs/worldsim_v62/p6r_evidence_dropout_v1.yaml`，入口=
`scripts/run_worldsim_v62_p6r_evidence_dropout.py`。

首次formal入口=`20260824T101047Z__feature-dropout-train-s0-r1` 在baseline selection、任何optimizer step之前因batch缺少
`prior_tristate`触发`KeyError`；未形成checkpoint、未读取legacy O_eval，也没有科学质量结果。损失函数的完整batch读取
已一次性核对；按PyTorch官方mapping batch合同，恢复让`prior_tristate`与输入证据视图同步：pure-prototype selection
使用prototype三态，训练中的full/prototype逐query混合使用同一个`corrupt_prior[:,18:21]`。`V62-F07 resolved`；r1
保持不可变。同次静态接口核对还把尚未执行的legacy `_query_features`返回语句归位，避免唯一复评路径返回`None`；没有
新增失败run。恢复提交=`fb0744b`。

P6R formal r2 canonical=
`run://worldsim_v62/WS-V62-P6R-EVIDENCE-DROPOUT-RECOVERY-01/20260824T101705Z__feature-dropout-train-s0-r2`
已完成：5 epochs、840 optimizer steps，按冻结min3/patience2选择best epoch=`2`；wall=`383.489s`、FP16 peak=
`0.377805GiB`、output=`2,475,348 bytes`、hard violations=`0/1,286,134`。pure-prototype composite objective从
baseline `2.448369`降到`2.274951`，accuracy=`0.452581→0.462246`、safe-OCC retention=`0.872897→0.887356`，
但hidden-FREE false-OCC=`0.399349→0.414406`；full-view也为`0.384568→0.401991`。不据单项风险事后改选epoch 0/3/4，
best epoch 2按预注册复合目标冻结。训练未读legacy O_eval、confirmation/test，IR-WM未运行；下一步只执行一次完全相同
legacy28 arms/gates的P6R recovery。

P6R legacy canonical=
`run://worldsim_v62/WS-V62-P6R-EVIDENCE-DROPOUT-RECOVERY-01/20260824T102709Z__feature-dropout-legacy28-s0-r1`，
source=`d0e5950`，terminal=`rejected`。B0/B1仍=`10/28,10 false-safe`；recovered B3/B5仍=`4/28,4 false-safe`，
B5 mask-area=`0.094024`、accepted FREE conflict mean/worst=`0.049166/0.087379`、R10=`2/3`、new Actor/static=
`0/2`。B5接受集合与P6相同，均为四个scene-0242 missing-route-support cases。UNKNOWN从`0.827351`降至
`0.638518`，relative下降`22.82%`，但仍超过0.50且没有移除任何false-safe；safe-OCC retention=`1.0`，hard projection=
`0/939,206 violations`。resource=`48.109s / 0.531876GiB / 2,293,068 pre-closeout bytes / 64.104GiB free`。

失败后的一手来源复核显示，RELIOcc/OCCUQ需要原生head/features的重训或离线校准，α-OCC与conformal risk control需要
独立calibration，selective classification也只把risk/coverage权衡显式化；它们都不能在“不第二recovery、不用O_eval
调参、不重跑backbone”的V6.2边界内合法迁移。`V62-F06 active, recovery exhausted`；CPSC-Lite family按计划关闭，
P7/P8/confirmation/test均不解锁。完整证据见`docs/autoresearch/worldsim_v62/P6R_EVIDENCE_DROPOUT_CLOSEOUT.md`。

范围冻结见 `configs/worldsim_v62/p0_scope_freeze_v1.yaml` 与
`docs/autoresearch/worldsim_v62/SCOPE_FREEZE.md`；P1 failure ledger delta=`none`，继承边界=
`V62-F01,V61-F11,V61-F13`。

## WorldSim V6.1 minimum experiment 已负结论收口（2026-08-22）

状态：`v61_minimum_experiment_closed_negative`；当前无 active hypothesis，ME-4 未执行且不再授权。

最终 canonical：

```text
run://worldsim_v61/WS-V61-ME3R-IRWM-PREDICTED-OCC-01/20260822T145543Z__irwm-predicted-occ-s1-r1
```

source=`6de27f5704914711e38090c7416d7145f2a610be`。两个 IR-WM scene workers 在一张 RTX3090 并行，
每个只载模一次，完成 target52/57 的4个固定 current occupancy。primary 与 oracle O2 均为 `10/28 ACCEPT`，
accepted mask-area yield 也相同（`0.3983001361`，oracle fraction=`1.0`），但10个接受项全部 false-safe；唯一
顶层失败 gate 是 `predicted_zero_false_safe`。route-support hidden FREE conflict=`0.344..0.571`，actor/disocclusion=
`0.106..0.173`，全部超过冻结上限0.05。wall=`124.30s`，两个 worker peak sum upper bound=`8.25GiB`。

这消耗了 GaussianWorld 失败后由一手文献审计授权的唯一 recovery。GaussianWorld 与 IR-WM 两个不同机制都复现
oracle 的接受集合与 yield，却分别得到 `10/10 false-safe`，因此本协议下不能把 learned argmax occupancy 当作安全
authority。该结论不否定模型 perception capability，也不构成现实驾驶安全声明；它拒绝的是本轮无需训练/calibration
就把预测表面提升为 task-verifiable 几何真值的机制。

按预注册 stop rule：不进入 ME-4，不再换 backend、选 confidence threshold、改 grid/history/checkpoint、放宽 verifier，
也不运行会把10例全部变成 abstain 的 observed-FREE 事后 veto。V6.1 实验实现正式停止，后续只从冻结 artifact 合成
arXiv 技术报告。完整收口见 `docs/autoresearch/worldsim_v61/V61_MINIMUM_EXPERIMENT_CLOSEOUT.md`，失败登记为 `V61-F13`。

## WorldSim V6.1 P7R 已通过；唯一 ME-3 IR-WM recovery 已预注册（2026-08-22）

状态：`p7r_irwm_capability_passed / me3r_irwm_only_recovery_pre_registered`

P7R canonical：

```text
run://worldsim_v61/WS-V61-P7R-IRWM-CONTRACT-RECOVERY-01/20260822T144446Z__irwm-contract-recovery-s1-r1
```

source=`c42bf50809a8a6813d49c841be76f524edbb8bb7`。analysis-only recovery 对 H001 的完整 immutable
artifact、官方删除源码、CUDA wheel build string 和 exact missing-key 集合完成8项 gate，全部通过；wall=`0.023s`，
没有 GPU、model reload、训练或 confirmation。H001 rejected terminal 保持不变，`V61-F12` 只在新 task 中恢复，
P7R 结论严格限于 IR-WM current occupancy 的 3090 capability，不包含安全性声明。

唯一科学恢复 task=`WS-V61-ME3R-IRWM-PREDICTED-OCC-01`，hypothesis=`WS-V61-H-ME3-IRWM-001`。
两个 development scene 各启一个 worker 并在同一 RTX3090 并行；每个 worker 只载模一次，固定 target52 的历史窗口
`42/47/52` 和 target57 的 `47/52/57`。输出映射=`class0→FREE / 1..16→OCCUPIED / extent外→UNKNOWN`；
UNKNOWN 封 ray，predicted FREE 不作为 observed truth，native OBB 只在模型已预测 occupied cell 上绑定 identity。

科学 denominator/gate 与 GaussianWorld ME-3 原样一致：28 matched cases，primary 至少 `8/28`、false-safe=`0`、
严格超过 R10 的3例、accepted mask-area yield 至少为 oracle O2 的80%。method decisions 在读取 O_eval 前冻结。
本次失败即关闭 learned occupancy 和 V6.1 minimum experiment negative；不再换 backend，也不调 confidence、checkpoint、
grid、history window 或 verifier threshold。

## WorldSim V6.1 P7 有效 forward；H002 形式合同恢复已预注册（2026-08-22）

状态：`p7_irwm_forward_valid / h001_contract_rejected / h002_analysis_recovery_pre_registered`

H001 canonical：

```text
run://worldsim_v61/WS-V61-P7-IRWM-3090-SMOKE-01/20260822T143153Z__irwm-current-smoke-s1-r1
```

source=`c5728207ce5ac9b0649afb61c9eedbe418b8d1c9`。官方 IR-WM fully-decoupled checkpoint 已在 RTX3090
完成一次 truth-free current-state forward：raw logits=`1×3×1×40000×16×17`，最终 grid=`200×200×16`，
occupied/free=`40778/599222`，finite，inference=`1.066s`、worker wall=`15.45s`、peak=`4.050GiB`。
两历史帧+当前帧、六相机和 ego motion 完整；没有读取 occupancy GT、O_method、O_eval 或 confirmation，且没有启动
future decoder、planning、training 或 calibration。

H001 的17项 gate 只有 `environment_versions_exact` 与 `model_state_exact` 为 false，因此该 immutable run 继续保留
`rejected` terminal，并登记 `V61-F12`，但不能把有效 forward 误写成 capability 科学拒绝。窄源码审计确认：官方
Detectron2 0.6 CUDA11.1 wheel 的安装版本字符串为 `0.6+cu111`；checkpoint 唯一 missing keys 是
`pts_bbox_head.transformer.reference_points.{weight,bias}`，而冻结官方 `WorldBEVFormerHead.init_weights()` 会主动
`del self.transformer.reference_points`，current-BEV 路径也不调用检测 decoder。

H002=`WS-V61-P7R-IRWM-CONTRACT-RECOVERY-01` 只复用 H001 的 immutable output/report，不重复 GPU 推理。它要求
H001 除上述两项外其余 gate 全通过、所有 artifact hash 精确、Detectron2 完整 build string 精确、missing keys 恰好是
源码证明的两项、unexpected keys 为空。通过才允许预注册唯一一次 ME-3 IR-WM recovery；失败则停止 learned occupancy。
不改 checkpoint、config、input、class mapping、threshold 或 verifier，也不做第二次 capability forward。

## WorldSim V6.1 ME-3 GaussianWorld 已科学拒绝；IR-WM capability 已预注册（2026-08-22）

状态：`me3_gaussianworld_rejected / irwm_capability_pre_registered`

ME-3 canonical：

```text
run://worldsim_v61/WS-V61-ME3-PREDICTED-OCC-01/20260822T134559Z__predicted-occ-s1-r1
```

source=`4c048ecd2db834ae494deb998947136f9918d9bb`。两个官方 batch1 scene workers 在同一 RTX3090 并行完成
24 次 streaming inference，4 个 target occupancy 与 28 个 method decisions 全部落盘；wall=`28.36s`、
per-process peak sum upper bound=`4.47GiB`。预测臂得到 `10/28 ACCEPT`，mask-area yield=`0.3983001361`，
与 oracle O2 的接受集合和 yield 完全一致；但 10 个接受项全部在隐藏 O_eval 上 false-safe，因此唯一失败 gate 是
`predicted_zero_false_safe`。route-support 的 hidden observed-FREE conflict ratio=`0.766..0.958`，actor/disocclusion=
`0.159..0.328`。该结果登记为 `V61-F11`，停止 GaussianWorld argmax Occupancy 作为安全 authority。

源码审计排除了低级适配错误：GaussianWorld 官方 head 使用 `[x,y,z]` 网格、class1..16=occupied、class17=empty；
DriveStudio nuScenes preprocessing 原样保存 camera/lidar world transform，直接 `lidar2img` 与官方 temporal metadata 的
后相机矩阵在机器精度内一致，前相机小差异符合异步 sensor timestamp。因而不授权轴交换、投影修补、confidence/grid/
schedule sweep。把 observed O_method FREE 作为 veto 会令这10例全部 abstain，产出率为0，结果可由已有 artifact 直接推出，
不再为它创建形式化回测。

文献审计显示 ReliOcc、α-OCC 与 OCCUQ 的可靠 uncertainty 都需要训练或 calibration；朴素 max-softmax/entropy 也没有
足够 OoD 可靠性，不能在本轮事后选阈值。OccWorld 依赖过去 Occupancy 输入，会把 oracle 引回 predictor；
Drive-OccWorld 主分支没有发布任务权重。IR-WM 官方分支发布了 vision-centric fully-decoupled checkpoint，并显式从
历史相机建立 current BEV state，因此只预注册一次 truth-free current-state capability smoke。smoke 通过后才允许唯一
一次 ME-3 recovery；失败则终止 learned occupancy，不建安装/调参支线。

gate/arm-summary/summary/resource/manifest/terminal=`508b3551...d74 / 23efb5e5...18c / f6391f49...721 /
7c2c6104...6f4 / 0bb0618f...2fc / 25c01504...4bd`。完整审计见
`docs/autoresearch/worldsim_v61/ME3_GAUSSIANWORLD_FAILURE_AND_BACKEND_AUDIT.md`。

## WorldSim V6.1 P6 已通过；ME-3 GaussianWorld development 已预注册（2026-08-22）

状态：`p6_passed / me3_gaussianworld_predicted_pre_registered`

P6 canonical：

```text
run://worldsim_v61/WS-V61-P6-GAUSSIANWORLD-3090-SMOKE-01/20260822T132526Z__gaussianworld-smoke-s1-r1
```

source=`95c842a883652f679cb1bee93bf1db0e3092c5b2`。官方 streaming checkpoint 完整载入，missing/unexpected
keys=`0/0`，输出=`1×18×200×200×16`、occupied=`29608`、empty=`610392`；inference=`0.8524s`、
worker wall=`3.0384s`、peak=`2.1499GiB`。17 项 gate 全部通过，未读取 SurroundOcc label、O_method/O_eval/
confirmation，未训练或选阈值。gate/summary/resource/manifest/terminal=`dd59fd9e...133 / da079429...b21 /
b6dc3b48...9ac / 24b19cbb...0d9 / 8f886211...ab7`。

ME-3 固定两个 scene-level 官方 batch1 worker 在同一 RTX3090 并行，时序帧=`2,7,...,52,57`，只输出52/57。
类别映射固定为 `0→UNKNOWN / 1..16→OCCUPIED / 17→FREE`；UNKNOWN 封住射线并触发 abstain，predicted FREE
不作为观测 FREE。native OBB 只给模型已预测 OCCUPIED 的 cell 绑定 actor identity，绝不生成几何。method decisions
在读取 O_eval 前固化；主门槛为 `>=8/28`（ME-1 oracle 10例的80%）、false-safe=`0`、mask-area yield 保留
oracle 的 `>=80%` 且严格超过 V6 的3例。不训练、不 calibration、不 threshold sweep；若失败只允许先按具体失败因子
查文献，再预注册一次不降低阈值的保守 recovery。

H-ME3-GW-001 第一次正式入口在 run directory/GPU 前因 tmux 非登录环境缺少 repository root `PYTHONPATH` 而
失败，登记 `V61-F10`，不存在模型或方法结论。H-ME3-GW-002 只让 wrapper 从自身路径自举 repo root；所有科学合同
与预算不变，并在无 run/GPU 的 `--help` smoke 后从新干净提交重跑。

## WorldSim V6.1 ME-2 已完成并拒绝 Hunyuan 路线；ME-3 backend 审计中（2026-08-22）

状态：`me2_rejected / hy3d_route_stopped / me3_backend_audit_in_progress`

ME-2 canonical：

```text
run://worldsim_v61/WS-V61-ME2-HY3D-OCC-ACTOR-01/20260822T121848Z__hy3d-actor-s1234-r1
```

source=`98cec20ae808600309afd2066f7826b2d94ed0b9`。H-ME2-003 完成全部冻结工作：4 个唯一 actor unit、
16 个生成资产、四臂各 6 例、共 24 个 case-arm evaluation；昂贵 Omni diffusion 保持 batch2，只有官方明确
batch1 的 VAE/marching-cubes decode 串行。H002 的 4 个 A0 资产仅在 plan/input/report/asset hash 全部精确后
复用。正式 run 完全离线，无训练和 confirmation read；wall=`675.64s`、peak=`9.45GiB`。

结果为 A0/A1/A2/A3 均 `0/6 ACCEPT`，主臂 A3 false-safe=`0`，但没有任何可接受 case。全部四臂在 method 与
hidden eval 都出现观测 FREE-space conflict；A3 每例 method conflict=`6..246`、eval conflict=`8..273`。与此同时
A3 的 native actor coverage=`0.4949..0.8461`、hole coverage=`0.4738..0.8641`、silhouette IoU=`0.4044..0.8431`，
说明主要问题不是提示词或轮廓质量，而是通用闭合生成表面不能满足场景已观测 FREE 约束。这个结论登记为
`V61-F09`；按预注册 stop rule 停止 Hunyuan actor proposal，不改 prompt、seed、texture、steps、octree、
compiler 或 verifier threshold，也不做事后 clipping。

gate/arm-summary/summary/resource/manifest/terminal=`1eab2226...d86 / dc2222df...505 / 85e20dd9...e73 /
e438e93e...dde / f7fae41a...118 / 9b90d9eb...dc9`。下一任务严格按计划转入
`WS-V61-ME3-PREDICTED-OCC-01`：只审计一个有官方权重、与本机 nuScenes 六相机数据兼容、能在 24GB 单卡执行的
学习式 occupancy backend；优先 GaussianWorld，其次 OccWorld。ME-2 rejection 不被错误扩展为 learned occupancy
路线 rejection。

P6 已选择并预注册 GaussianWorld pretrained：官方 commit=`b43629e...4fc`，stream checkpoint/backbone/temporal
metadata 分别为 `298029831 / 177818375 / 530760430` bytes，SHA-256=`54770811...be3 / 1ee46d1c...ccf /
302fcb86...b54`。官方 metadata 同时包含 scene-0048/0242 各40 keyframes；本机已有两个 development scene 的
六相机 DriveStudio 图像与标定。smoke 固定 scene-0048/frame52、官方 camera order、官方 200×200×16/0.5m 输出，
只验证单卡权重载入、finite/nonempty 输出和 `<22GiB`；不读 SurroundOcc label、O_method/O_eval 或 confirmation，
不做 calibration/threshold selection。通过后直接进入一次 ME-3 development；失败时只审计一次 OccWorld source/
resource，不调 GaussianWorld 输入尺寸、camera order、权重或参数。详见
`docs/autoresearch/worldsim_v61/P6_GAUSSIANWORLD_SOURCE_AUDIT.md`。

## WorldSim V6.1 P4 与 ME-2 预注册/恢复历史（已由 H003 正式结果取代，2026-08-22）

历史状态：`p4_done / me2_h002_batch_decode_failure / h003_formal_retry_ready`

当时 active hypothesis=`WS-V61-H-ME2-003`，task=`WS-V61-ME2-HY3D-OCC-ACTOR-01`。V6 selector 研究族继续冻结，
V6.1 转向 Occupancy-authoritative、Gaussian-rendered、task-verifiable 的四维世界编译器，不再继续阈值、selector、
2D inpainting 或 per-case generator 混选。

P4 canonical：

```text
run://worldsim_v61/WS-V61-P4-HY3D-OMNI-3090-SMOKE-01/20260822T112707Z__voxel-smoke-s1234-r1
```

source=`a97b2743935e3a7143d5b75da9e7bc5bac95e317`。正式 worker 完全离线，用固定官方 voxel demo、seed1234、
50 steps、512 octree、guidance4.5 生成 `1,238,856` vertices / `2,477,728` faces 的 finite mesh 与非空
sampled points；wall=`235.16s`、peak=`7.90GiB`。gate/summary/manifest/terminal=`23451b2d...5cf /
8133a65b...ab7 / 7c4783cb...9a2f2 / 177ce781...8a3`，全部 capability/resource/license gate PASS。
`V61-F04/F05/F06` 保留为不可变失败证据，byte-exact DINO ref 修复后已关闭，不再继续 cache/安装探测。

ME-2 冻结四臂=`A0-image / A1-bbox / A2-point / A3-voxel`。A0 使用同系列官方 Hunyuan3D-2.1 image-only，
A1–A3 使用固定 Omni；4 个唯一 scene/frame/actor 输入按字节复用到 6 个冻结 actor cases，避免为重复 frontend
浪费生成算力，同时保留完整 case denominator。point/voxel 只读 raw LiDAR 与 `O_method`；method decisions 落盘并
冻结后才允许读取 `O_eval`。生成 mesh 只做轴置换与一个 uniform scale，不做 anisotropic warp、clipping 或 case 特判。

单次结构预检没有读取 `O_eval`、没有载入生成模型：4/4 controls finite，raw actor points 非空，target O_method
voxels=`10878..23088`，6 个 case 的最小 actor-hole coverage=`0.6322`。native LWH 已按官方 Omni 合同转换为
LHW；最大 actor `15.454m / 256 = 0.0604m`，低于冻结 `0.2m` occupancy cell，故固定 octree256 而不做分辨率 sweep。

主臂 A3 gate=`>=2/6`、false-safe=`0`、accepted FREE conflict=`0`、unfiltered swept collision=`0`。
scene-0242 只过滤 actor4 truck 与 actor15 trailer 的精确铰接 contact：141 连续帧相交，最大相对平移步长
`0.09814m`、最大相对 yaw 步长 `0.07619°`；不放宽全局碰撞阈值。失败即停止 Hunyuan 路线，不做 prompt、
texture、seed、steps、resolution 或 verifier threshold 调参。

H-ME2-001 已创建 failed run `20260822T120008Z__hy3d-actor-s1234-r1`：所有 source gate 和4个输入构造完成，
但 A0 worker 在载模/GPU推理前导入官方 Hunyuan3D-2.1 package 时缺少其 requirements 固定的
`pymeshlab==2022.2.post3`（`V61-F07`）。没有生成 asset、method decision 或科学结论。H-ME2-002 只在隔离
环境补齐该官方依赖并增加 exact version gate；一次离线 base pipeline import smoke 已通过。全部科学合同不变，
从新干净提交重试。

H-ME2-002 failed run `20260822T120519Z__hy3d-actor-s1234-r1` 已完成4个有效 A0 mesh；Omni 也完成首个
2-sample A1 diffusion/decode，但官方 vanilla extractor 把两份 SDF reshape 后只对 `grid_logits[0]` 做 marching
cubes，因此只返回1个 mesh。runner 对 `1 != 2` fail-closed，没有静默丢弃第二例（`V61-F08`）。H-ME2-003
保持 diffusion batch2，改为返回2份 latent 后逐份调用同一官方 VAE decode；只串行官方明确 batch1 的 mesh
extraction。H002 A0 只在旧 plan/input/report/assets 全部精确后复用，不重复4次 GPU 生成；科学参数和 gate 不变。

P0 精确绑定：

- V6.1 plan SHA-256=`8ac58801...38be`；
- R10 28-case baseline=`3 ACCEPT / 7 ABSTAIN / 18 REJECT`、false-safe=`0`、accepted mask pixels=`107807`；
- scene mapping=`scene-0048 -> processed 045`、`scene-0242 -> processed 191`；
- `O_method` 与 `O_eval` 使用不重叠的 raw LiDAR sweep 路径，confirmation 保持锁定；
- failure refs=`V6-F25/V6-F26/V6-F65/V6-F71/V6-F78/V6-F79`。

H-P0-001 在创建 run 或读取任何科学输入前因新 namespace 不存在而触发 `FileNotFoundError`；GPU/训练/生成器均未启动，
没有方法结论，登记为 `V61-F01`。H-P0-002 只创建精确 run namespace 后正式通过，`V61-F01` 已 resolved。

P0 canonical：

```text
run://worldsim_v61/WS-V61-P0-SCOPE-FREEZE-01/20260822T100812Z__scope-freeze-s20260822-r1
```

source=`6247fd89068615f791b428c3296faf945e713c75`；gate/summary/manifest=`fb2a416a...ae40 / e53a86f2...907c /
2ed96578...7593`。全部 gate PASS；R10=`3/28`、false-safe=`0`、case identity 与 scene mapping exact，
method/eval source paths disjoint。

ME-0 canonical：

```text
run://worldsim_v61/WS-V61-ME0-OCCIR-01/20260822T101817Z__occir-s20260822-r1
```

source=`5a3bc42eb68cfcda673df3c32d81479373b1bff3`；4 scene/frame units、8 truth tiers、28 case bindings 全部
通过。`O_method/O_eval` 的 raw LiDAR path 与 payload hash 全局互斥；每格 UNKNOWN/FREE/OCCUPIED 非零；
oriented actor volume、identity/lifecycle、source-removal→UNKNOWN、fresh-process content exact 与
`<=2.14e-14m` round-trip 均通过。gate/summary/manifest=`1e818074...8bb7 / 6e50644b...b14f /
386d99ab...59ec`；wall=`10.57s`，4 CPU workers，无训练/生成器/confirmation read。

ME-1 预注册固定五臂：冻结 Big-LaMa 的 `B0-2D`、冻结 R10 的 `B1-R10`、不增 coverage 的 `O1-GATE`、
主臂 `O2-OCC-GEOMETRY` 与带 native trajectory/lifecycle/swept OBB collision 的 `O3-OCC-4D`。编译只读
`O_method`，先固化 method decisions，再让 `O_eval` 只计算 hidden truth/false-safe。阈值来自既有合同：
0.2m voxel、0.1m ray step、R9 的 50% coverage 与 20% depth consistency；没有 case 特判或 threshold sweep。
一次结构审计显示 10 个 P1-ACCEPT case 的 method mask coverage=`73.65%..94.78%`，故直接进入正式 run。
若 O2 不能达到 `>=5/28`、false-safe=`0`、保留原3例并新增 actor+static/disocclusion，则停止模型接入。

H-ME1-001 在创建 run directory 或启动 GPU 前读取 ME-0 gate 时误把 authority 从 `checks.passed` 当成顶层
`passed`，触发 `KeyError`；无 run、无方法结果，登记为 `V61-F02`。H-ME1-002 只修正该 schema 路径并增加回归测试，
所有科学输入、arms、thresholds、预算与 stop rule 不变。

ME-1 canonical：

```text
run://worldsim_v61/WS-V61-ME1-ORACLE-OCC-PROPOSAL-01/20260822T104207Z__oracle-occ-s20260822-r1
```

source=`e422f0528c2c98e80d3cfbd8052ccb106734d043`。B0=`0/28`；B1/O1 均为 `3/28`；primary O2=
`10/28`、false-safe=`0`、accepted mask pixels=`450865`、yield=`39.83%`，保留原3例并新增3 actor+4
static/disocclusion。O3=`6/28`、false-safe=`0`；actor 例被真实 native OBB overlap（主要 actor4/15）拒绝，
不通过阈值豁免。后续控制准备另发现 actor ID0 与 empty sentinel 冲突（`V61-F03`）：不影响 O2 主结论，但 O3 的
scene-0048 identity 诊断降格；ME-2/ME-4 使用 `-1` sentinel 修复。wall=`3.60s`、peak=`0.51GiB`。
gate/summary/metrics=`6aca5f2f...246d / 61713df4...afb9 / dbb1d0a3...ffb6`，ME-2 已解锁。

P4 绑定 Hunyuan3D-Omni 官方 git commit=`4d47c0cc...bfa8`、HF model revision=`70e803bf...d485` 与
DINOv2-large=`47b73eef...2d6c`。官方一手实现声明约10GB VRAM且支持 bbox/point/voxel；正式 smoke 固定官方 voxel
demo、seed1234、50 steps、512 octree、无EMA/fast decode/sweep，离线运行并要求 mesh/points 有效、peak<22GiB。
模型使用受 Tencent community license 的地域与用途限制；本轮只在中国 AutoDL 主机科研执行，不分发模型/输出，
也不用于训练其他模型。P4 通过后直接跑固定6例 ME-2；失败则停止 Hunyuan 路线，不反复调安装/推理参数。

P4 首次入口在 run/GPU 前发现 VAE digest 被手工多录一个尾字符（`V61-F04`）；实际文件 SHA 与固定 revision
HTTP `X-Linked-ETag` 完全一致。只修正 65→64 字符的 provenance transcription，并新增 digest 结构回归；模型、
权重、demo、seed、steps、octree、gate 与 stop rule 均不变。推理环境已按官方版本收窄为 shape-inference closure，
`pip check`、CUDA、DINO cache 与官方 pipeline import 均通过，训练/UI/texture 后处理依赖不进入 P4。

第二次入口已创建 failed run `20260822T111747Z__voxel-smoke-s1234-r1`：DiT/VAE 精确载入，DINO repo-id
因 exact-commit cache 缺少默认 `refs/main` 而在离线解析处失败（`V61-F05`），尚未生成 mesh/points 或 capability
结论。修复只建立标准 cache ref 并把它精确绑定冻结 DINO commit；runner 在载模前验证 ref、snapshot、config 与
model SHA，正式入口继续完全离线，不修改官方源码、backbone 或任何推理参数。

第三次入口 `20260822T112159Z__voxel-smoke-s1234-r1` 暴露了更精确的根因（`V61-F06`）：运行时 cache
root 正确，但安装版本以原样 `f.read()` 解析 ref；staging 文件尾换行使 ref 为41 bytes，无法匹配40字符 snapshot。
外部 cache ref 已规范化为 byte-exact token；孤立离线 repo-id smoke 成功载入 `Dinov2Model` 的
`304368640` 个参数。只有该最小解析测试通过后才重新授权完整 P4，避免了继续重复载入12GB Omni 权重。

## WorldSim V6 收口：selector 研究族已冻结（2026-08-22）

状态：`selector_research_family_frozen_closeout_complete`

当前没有 active hypothesis。R141 未执行。按照最终研究决策，本研究族不再继续 threshold 13/45、新 actor、新编辑方向，也不引入新的 selector 机制。

### R140 recovery

R140 H001 与 H002 已完成科学计算，但由于 Python 源码使用小写 JSON boolean，在正式 closeout 阶段失败；它们继续作为 V6-F97 与 V6-F98 不可变保留。H003 只把剩余的 `false` 改为 `False`，所有科学输入、公式与 gate 均保持不变，并从干净且已推送的源提交 `a13759ba8db03e1f740ad93e246ca24f0ff2d7fa` 完成。

Canonical run：

```text
run://worldsim_v6/WS-V6-R140-CROSS-FRONTEND-END-TO-END-UTILITY-01/20260822T063937Z__end-to-end-utility-s20260821-r1
```

| 条件 | End-to-end reduction | Reconstruction errors |
| --- | ---: | ---: |
| StreetGS | 0.13533665047667254 | 0 |
| AD-GS development | 0.11143415340582441 | 0 |
| AD-GS exact-once confirmation | 0.016636471392706964 | 0 |
| Macro | 0.08780242509173464 | 0 |
| Worst | 0.016636471392706964 | 0 |

Full 与 selective 路径以相同方式计入 sensor time。这些数值是单次已观测 artifact cost，不是 replicated performance estimate。

Artifacts：

- certificate `913833af47e4171e27707f71418b6625ed358b538d1c8a5a18bca5ac7f585363`
- gate `ac3c79c0e93f2932a076da8323b89a210ff2cbaac27ffa13079ce89ae9d07b51`
- summary `50900ff99736055a10c32f4362176b7fc87862ae84667591077d6c17024e635b`
- manifest `1cc753b3c0a9489ced2a58b23035466ee26cba963d2abc56c84ebd4d057e5a62`
- resource audit `06c110236591529d5fef5f4178bfed696b6c0ad0cfbce94c497896dc92230265`
- terminal `be263ba010cdb936fbd01dbfa0fe294b8022101aae348c95030d2a42d45fdb77`

### Selector 最终证据

| 实验 | 状态 | 保留结论 |
| --- | --- | --- |
| R134 | rejected / V6-F94 | threshold 13 漏检 AD-GS frame 13（RGB 1、label 1）。 |
| R136 | rejected / V6-F95 | 冻结 threshold 1 在 heldout frame 14 出现 1 个 FP；精确分类声明失败。 |
| R137 | accepted development | 157 个 AD-GS 帧，调用减少 16.56%，0 false reuse，628 个 hash 全部精确。 |
| R138 | failed consumed / V6-F96 | 负数 CLI 参数在 sensor 输出前失败；不存在方法结论。 |
| R139 | accepted exact-once | 39 个 AD-GS 帧，调用减少 17.95%，0 false reuse，156 个 hash 全部精确。 |
| R140 | V6-F97/F98 recovery 后 accepted analysis | Macro 端到端 reduction 8.78%，worst 1.66%，0 reconstruction errors。 |

### 治理状态

- Failure ledger 的当前权威边界是 V6-F98；recovery 注记不删除或重分类失败 attempt。
- Selector 研究族在 R140 后冻结。R141 明确为未执行，不是 rejected，也不是 accepted。
- Confirmation 与 test 分区继续锁定。
- Claim boundary 只覆盖 operational equivalence 与已观测 wall-time accounting；不声明 semantic、physics、planning 或 safety correctness。
- 仓库收敛目标为唯一远端分支 `main`，指向本次 closeout。

详见 [selector 研究族收口](autoresearch/worldsim_v6/SELECTOR_RESEARCH_FAMILY_CLOSEOUT.md)、[failure ledger](RESEARCH_FAILURES.md) 与 [V6 plan](WORLDSIM_V6_VERIFIABLE_WORLD_COMPILER_AUTORESEARCH_PLAN.md)。
