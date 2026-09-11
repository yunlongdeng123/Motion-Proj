# 历史原始记录 021

> 冻结历史，不是当前执行规则。当前规则见 [scaling law](../../../auto-research_scaling_law.md) 与 [AGENTS](../../../AGENTS.md)。
> 原文件：`docs/RESEARCH_FAILURES.md`；迁移前提交 `abbd431c`。原有相对链接按原目录解释，证据路径原样保留。
> 按索引读取相关小节；旧哈希、过度门控和资源指令均不重新生效。

## V7.1 M14 prevention note — compact support是表示，不是事后gate（2026-09-04）

- local patch值内生定义为plane half-space与M8-radius ball的CSG intersection，patch外FREE参与所有训练/部署query；
- support radius固定为M8 GT-supervised tangent scale，不按M13/M14结果乘系数或裁剪primitive；
- patch union使用occupancy `min`，禁止soft average异号planes；zero threshold仍为GT hit定义的0；
- 前/hit/窄后方标签与AABB zero crossing合同不变，M8 point/temporal只读；
- 若field合同失败登记`V71-F19`并关闭compact-patch field，不扫radius/neighbors/loss/seed/epoch。

下一可用编号仍为：`V71-F19`。

## V71-F18 — unbounded local-plane blending制造大量前方zero crossings（2026-09-04）

- 分类/状态：implicit local-support parameterisation / terminal for soft-blended unbounded planes；canonical=
  `20260905T000000Z__m13-local-field-s71115-r1`；
- 观察：field early all/hazard=`46.408/46.593%`，相对M8恶化`80.577/76.676%`；hit下降`11.650/10.574pp`；
  observable all/hazard=`94.28/95.90%`，说明错误是spurious front surface而不是missing surface；
- 优化审计：hit=`0.509→0.480m`、physics=`10.163→9.685`，仍远离GT；冲突从0升到`69.1%`，无NaN/OOM；
- root cause：每个oriented plane在AABB全域有效，nearest-4 soft averaging使异号planes在两者之间产生与任何M8 anchor
  无关的zero set；bounded residual无法移除基础场的拓扑伪面；
- literature/open-source response：LDIF以local spatial decomposition限制局部implicit function作用域，ARO-Net以anchor
  visibility使occupancy query-specific；迁移为compact radial support与CSG occupancy union，不做阈值裁剪；
- resolution：关闭M13 unbounded blend；M14用`max(plane, radial support)`定义每patch、用`min`聚合，radius固定M8 scale，
  原GT窄带和zero-crossing部署不变；不扩大residual/loss/epoch；
- claim impact：local implicit方向尚未成立；M8 point/temporal结论不变。下一可用ID=`V71-F19`。

下一可用编号：`V71-F19`。

## V7.1 M13 prevention note — local implicit不是M3/M4 global latent复跑（2026-09-04）

- 每个query必须读取nearest M8 child feature、relative coordinate、scale与oriented build normal；禁止单个Actor pooled latent；
- field以M8 local plane的signed distance初始化，避免M3单符号场；learned residual初始为0；
- GT符号只来自ray front/hit/窄`5–10cm` back band，禁止把远端behind-surface空间全标occupied；
- zero level同时用于训练hit与AABB ray deployment，不加UNKNOWN/opacity/occupancy threshold或输出filter；
- M8 point/temporal representation只读，不因field loss移动；若field合同失败登记`V71-F18`并关闭当前local-field形式。

下一可用编号仍为：`V71-F18`。

## V71-F17 — finite chart局部几何改善但earliest visibility不改善（2026-09-04）

- 分类/状态：representation aggregation / terminal for independent finite planar chart union；canonical=
  `20260904T234000Z__m12-finite-chart-s71114-r2`；
- 观察：M8 point五门保持；chart hit all/hazard=`+0.128/+0.166pp`，但early all/hazard相对变化=
  `-0.428/-0.598%`，hazard门失败，最终6/7；
- 优化审计：normal=`0.995→0.990`、point-to-plane=`1.000→0.997`、in-radius=`0.685→0.652`，而exact
  first=`0.9991→0.9993`、free=`0.7000→0.7118`；无NaN/OOM；
- root cause：zero-thickness消除了normal volume，仍未解决多个独立charts之间的global earliest visibility；局部normal调整
  改善几何标签但无法稳定改变ray上的first valid primitive；
- literature/open-source response：CVPR 2023 ARO-Net与CVPR 2020 Local Implicit Grid/IF-Net/LDIF使用query-specific
  local features而非单一global code，提高局部可辨识性；Iso-Points以显式点引导隐式面。迁移为M8-guided local field；
- resolution：关闭sphere/ellipsoid/disc primitive-union物理族；M13用nearest M8 child局部特征初始化signed field，沿GT
  ray窄带监督同一zero level；不调M12 radius/loss/seed/epoch；
- claim impact：M8 point/temporal保留，explicit primitive collision/surface-support claim关闭。下一可用ID=`V71-F18`。

下一可用编号：`V71-F18`。

## V71-F16 — M12冻结scale仍携带inference-tensor属性（2026-09-04）

- 分类/状态：engineering pre-step / recovered；run=`20260904T233000Z__m12-finite-chart-s71114-r1`；
- 症状：首个batch的`in_radius=tangent/scales`在backward前抛`Inference tensors cannot be saved for backward`；
- exposure：0 epoch、0 optimizer step、0 holdout metric、0 scientific result；AV2/protected external未读；
- root cause：M8 children/scales在`torch.inference_mode()`中物化，虽冻结但参与含可学习normal的表达式，autograd需要保存
  scale用于反向；
- resolution：离开inference mode后clone children/scales/anchor normals为普通tensor；不改模型、loss、seed、cohort、
  gate或训练长度；新run-id原样重跑；
- prevention：任何冻结reference只要进入含trainable operand的计算图，都必须在inference block外物化；不增加全局检查门。

下一可用编号：`V71-F17`。

## V7.1 M12 prevention note — Gaussian appearance体不等于物理surface（2026-09-04）

- M12部署primitive是zero-thickness finite disc chart，不能再以任意`σ` ellipsoid解释碰撞边界；
- center/tangent radius逐值冻结M8，只有normal head可学习，point/temporal surface不得被chart physics改写；
- GT normal、point-to-plane、in-radius coverage与exact disc first/free均在训练内，不能只在输出后把Gaussian压扁；
- 所有charts保留，无opacity threshold、filter、mask或按hazard删除；结论仅限可见surface first-return；
- 若hazard early或hit失败，登记`V71-F17`并关闭finite-chart family，不扫radius倍数/loss/seed/epoch。

下一可用编号仍为：`V71-F17`。

## V71-F15 — exact ellipsoid监督保住hit但无法降低earliest collision（2026-09-04）

- 分类/状态：representation support identifiability / terminal for learned normal-thickness ellipsoid；canonical=
  `20260904T230000Z__m11-exact-support-s71113-r1`；
- 观察：M8 point surface被逐值冻结并通过原五门；exact support hit all/hazard=`+0.026/+0.097pp`，但early all/hazard
  相对变化=`-1.012/-1.213%`，hazard `>=5%`门失败，最终6/7；
- 优化审计：exact first=`0.9995→0.9993`基本不动，free=`0.7639→0.7634`，boundary=`0.9845→0.9470`，normal
  `0.9967→1.0565`；梯度冲突约`46%–62%`，无NaN/OOM；
- root cause：当center/tangent coverage冻结时，有限normal/thickness自由度只能改变有厚度ellipsoid体积，不能同时重排多
  primitive earliest intersection并保持所有target intersections；解析forward同构排除了M10的采样代理解释；
- literature/open-source response：CVPR 2025 Geometry Field Splatting明确以surface geometry field而非普通volume建模
  opaque boundary；MAtCha将2D Gaussian surfel绑定到surface charts。迁移为finite zero-thickness chart，不复刻RGB渲染；
- resolution：关闭M11 normal/thickness ellipsoid head，不调thickness/loss/seed/epoch；M12冻结M8 center/radius，学习GT
  normal并直接训练/评价analytic ray--disc intersection；
- claim impact：M8 point/temporal成立，Gaussian ellipsoid collision-support claim关闭。下一可用ID=`V71-F16`。

下一可用编号：`V71-F16`。

## V7.1 M11 prevention note — 物理forward model与部署边界必须同构（2026-09-04）

- M11训练和验收都使用同一analytic ray--oblate-ellipsoid entrance depth，禁止sampled alpha或sphere proxy；
- M8 center/tangent scale与point surface冻结，support任务只学习normal/thickness，禁止用物理损失破坏已通过的几何中心；
- no-hit梯度只由GT endpoint boundary residual补充，FREE侧仍由GT ray在首返回前的解析入口约束；
- reference固定为M8 center/scale + parent normal + `0.02m` thickness，不按M11结果改阈值或support level；
- 若hazard early或hit合同失败，登记`V71-F15`并关闭本解析support head，不调loss/thickness/seed/epoch恢复。

下一可用编号仍为：`V71-F15`。

## V71-F14 — oriented support减少early但sampled训练代理损失解析hit（2026-09-04）

- 分类/状态：training-deployment forward-model mismatch / terminal for sampled-density M10；canonical=
  `20260904T220000Z__m10-oriented-planar-s71112-r1`；
- 观察：exact hazardous support early相对M8下降`18.451%`，但support hit `30.837→28.444%`（`-2.393pp`）；
  point hazardous early仅改善`2.618%<5%`，M8同一reference为`5.123%`；最终5/7；
- 正证据：all/hazard/clear support early均改善，Chamfer=`-7.413mm`、point hit=`+2.838pp`、retention 100%；
  oriented primitive本身有价值，失败不能归为表示容量不足；
- 优化审计：sampled support free相对损失`0.520→0.423`、support first`1.002→0.990`，无NaN/OOM；约
  `75%–83%` batches仍有geometry/physics conflict；
- root cause：训练优化sampled Gaussian alpha-compositing depth，部署/门槛读取hard analytic `1σ` ellipsoid first hit；
  前者可降低密度型FREE惩罚而不保证后者的target intersection，仍违反训练—部署同构；
- literature/open-source response：WACV 2025 RayGauss与ICCV 2025 EVER均将Gaussian/ellipsoid直接置于ray-casting
  forward model；RayGaussX进一步以scale regularization抑制false-positive intersections。迁移原则是训练物理算子必须与
  primitive的部署相交语义一致，而不是复用其外观渲染目标；
- resolution：关闭M10 sampled-density恢复路线；M11冻结M8 center/scale，只学习normal/thickness并直接反传解析首交，
  加GT endpoint boundary residual处理no-hit；不调M10 sigma/loss/thickness/seed/epoch；
- claim impact：尚不能声称Gaussian support物理自洽；M8 point/temporal结论保留。下一可用ID=`V71-F15`。

下一可用编号：`V71-F15`。

## V7.1 M10 prevention note — tangent radius与normal thickness必须分权（2026-09-04）

- M10不能复用单个isotropic scale；tangent radius只负责沿GT plane覆盖，normal thickness只负责表面带宽/free-space；
- normal由build PCA初始化但必须接受GT local-plane loss；推理不读取GT normal，所有输出来自build-only feature；
- exact support evaluator同时报告early与hit，禁止只展示M9式early改善而隐藏ray hit损失；
- M8 oriented initializer事前固定，不按M10输出换reference；所有children无条件保留；
- 若7项任一失败，登记`V71-F14`并关闭当前oriented head，不调thickness、sigma、loss、seed或epoch恢复。

下一可用编号仍为：`V71-F14`。

## V71-F13 — isotropic Gaussian support靠缩scale降early但损失surface hit（2026-09-04）

- 分类/状态：representation parameterisation / terminal for isotropic collision sphere；canonical=
  `20260904T205000Z__m9-gaussian-support-s71111-r1`；
- 观察：exact `1σ` hazard support early相对M8下降`64.854%`，但support hit从`31.766%`降至`25.218%`；mean
  scale由`0.16196m`缩至`0.11227m`，local-scale监督误差升至`0.835`；point hazard early仅降`2.653%<5%`；
- 非优化失败：support first/free持续下降，Chamfer改善`7.626mm`、point hit增加`2.874pp`，无NaN/OOM；
- root cause：单个isotropic radius既决定切平面覆盖范围又决定法向碰撞厚度；缩小它可清除free-space support，却必然漏掉
  target rays，无法同时表达宽表面与薄法向；
- resolution：关闭sphere-support family，不改0.5σ/2σ、loss/seed/epoch；下一表示拆为normal、tangent radius和normal
  thickness，并用GT local plane与ray physics分别监督；
- claim impact：M7/M8仍只支持center/point几何，尚无Gaussian covariance物理自洽claim；下一可用ID=`V71-F14`。

下一可用编号：`V71-F14`。

## V7.1 M9 prevention note — center通过不能替Gaussian support通过（2026-09-04）

- M7/M8现有point renderer/evaluator不读取predicted scale，因此不得把其结果表述为3D Gaussian体的FREE一致；
- M9固定报告`1σ` collision iso-surface并用exact intersection评价；不能在结果后改为0.5σ或按scale删点；
- local-scale标签、scale-aware ray physics和部署scale是同一模型输出，避免训练一个对象、评价另一个对象；
- 本轮不引入normal/orientation，以便support失败可明确归因isotropic volume；失败后才转向2D/oriented Gaussian；
- 若原point合同或hazard support gate失败，登记`V71-F13`，不调support cutoff、loss weight、seed或epoch恢复。

下一可用编号仍为：`V71-F13`。

## V7.1 M8 external execution note — scene-ready入口正常（2026-09-04）

- M8 evaluator已按`.complete`处理4/20 logs与78 Actors，M5/M7/M8均等待相同第5项；
- 三个evaluator只重复只读编译/推理，不修改下载log、不启动第二下载器，当前资源充足；
- 未读任何partial physical aggregate；当前无新增engineering/scientific failure，下一可用ID保持`V71-F13`。

下一可用编号仍为：`V71-F13`。

## V7.1 M8 external prevention note — paired cohort不是事后模型选择（2026-09-04）

- M8 external checkpoint/gates在任何M5/M7 partial physical aggregate读取前冻结；只知道下载/Actor进度；
- M7与M8是事前定义的两个Pareto点，各自对完整20 logs作独立verdict，不用跨模型差值删除log或修改claim门槛；
- M8 external不计算moving/quasi-static标签，也不读取trajectory作为shape输入；外域只复用actor-canonical编译合同；
- M8失败则登记`V71-F13`并限制形状—轨迹因子化的跨传感器claim，不回调fine-tune epoch/weight/seed。

下一可用编号仍为：`V71-F13`。

