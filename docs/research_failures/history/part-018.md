# 历史原始记录 018

> 冻结历史，不是当前执行规则。当前规则见 [scaling law](../../../auto-research_scaling_law.md) 与 [AGENTS](../../../AGENTS.md)。
> 原文件：`docs/RESEARCH_FAILURES.md`；迁移前提交 `abbd431c`。原有相对链接按原目录解释，证据路径原样保留。
> 按索引读取相关小节；旧哈希、过度门控和资源指令均不重新生效。

## V7.1 M28 mechanism boundary — non-interference不替代GT几何正确性（2026-09-05）

M28选择typed physical/pose/visual sibling states和无visual参数的纯physical query，不新增
post-hoc filter、selector或阈值。该结构可证`dQ_phy/dV=0`与刚体组合的SE(3) equivariance，
但不能证明GT完整、learned geometry/trajectory正确、background collision完备或closed-loop safety。
几何仍必须由GT set/plane/scale/frame/first-return losses在训练期直接约束，M21 fresh AV2 20/20仍是
必须等待的外部边界。这是claim边界而非feature failure；下一failure ID仍为`V71-F30`。

## 2026-09-05: V7.1 storage cleanup completed without research-state loss

精确删除仅包含旧实验产物的`/root/autodl-tmp/runs/worldsim_v3`、`worldsim_v32`与
`worldsim_v33`（删除前合计约4.9GiB）。删除前已确认三个目标的resolved path严格位于
`/root/autodl-tmp/runs/`，无运行进程占用，且V7/V7.1 configs、scripts与paper无路径引用。
当前仍被M22--M27使用的V4 StreetGS checkpoint、config、render与Actor registry全部保留。
磁盘可用空间从约92GiB回升至约96GiB；删除不可恢复。未触发research failure，下一failure ID仍为`V71-F30`。

## V7.1 M27 outcome note — 分层visual residual恢复development视角一致性（2026-09-05）

- canonical r2完成320/320 steps，4/4 decisions通过；相对冻结M25，6/6 held-out views全部为正，delta中位数
  `+0.4731dB`、最小`+0.2383dB`、最大`+1.0730dB`；
- pooled held-out=`17.7305dB`，相对M25 `+0.7871dB`、相对M26 `+0.1580dB`；相对原StreetGS仍差
  `7.6222dB`；
- interpretation：coarse parent与fine surface-derived surfels共同渲染，能修复M26只偏向最大footprint视图的问题；
  但3090 vs 32522 primitives和单Actor开发设置仍有显著容量缺口；
- exposure boundary：六个views在M27设计前已暴露，结果只支持representation mechanism diagnosis，不支持新held-out、
  cross-scene、photorealism或generalization claim；
- closure：单Actor appearance分支停在M27，不继续调opacity split、steps、geometry或view；后续回到physical external与
  authority/safety boundary；
- no new failure：`V71-F29`仅为已解决入口路径错误。

下一可用编号仍为：`V71-F30`。

## V71-F29 — M27 StreetGS配置路径转录遗漏（2026-09-05）

- 分类/状态：configuration path / resolved before data or metric exposure；run=
  `20260904T193000Z__m27-hierarchical-visual-residual-r1`；
- symptom：`OmegaConf.load`报`FileNotFoundError`；0 dataset load、0 render、0 optimizer step、0 quality exposure；
- root cause：从M25配置转录路径时漏掉run ID的`-formal30k-s0-r17`后缀及
  `work_dirs/worldsim_v4_streetgs`层；文件与原生runtime均存在；
- resolution：r2逐值复用M25 canonical `streetgs_config`绝对路径；不改表示、分支opacity、view、seed、steps、loss或lr；
- prevention：后续派生真实renderer配置从canonical config逐值复用路径，不手写缩短；
- claim impact：纯入口失败，不计hierarchy科学结果。

下一可用编号：`V71-F30`。

## V7.1 M27 prevention note — hierarchy诊断不得冒充未暴露泛化（2026-09-05）

- M27显式标记六个development views已在M25/M26暴露，只能解释机制，不形成新held-out/generalization claim；
- 唯一representation判定为per-view delta中位数大于0，不以pooled像素权重决定；positive count/min/max完整报告；
- coarse/fine两层geometry均冻结且均不进入physical query；RGB只更新attributes/opacity；
- 两分支用固定half optical-depth初始化，不扫融合权重、steps、loss或seed；失败则关闭该hierarchy；
- 迁移Octree-GS/LOD-GS的多尺度组织，不迁移其可学习geometry、selector或densification。

下一可用编号仍为：`V71-F30`。

## V7.1 M26 outcome note — pooled容量改善但view-uniform不成立（2026-09-05）

- canonical r1完成320/320 steps，4/4预注册decisions通过；held-out pooled Actor PSNR
  `16.9699→17.5725dB`，相对冻结M25 final=`16.9434dB`提高`0.6290dB`；
- 相对原StreetGS `25.3527dB`仍差`7.7802dB`，2781 surfels只关闭M25原`8.4093dB`缺口的7.5%；
- aggregate-to-view boundary：相对M25 final，f77/camera2提高约`1.07dB`，其余5/6 held-out views退化约
  `0.10--0.31dB`；pooled gain由最大footprint视图主导，不能写成view-uniform改善；
- interpretation：training-supervision-derived tangent surfels确实增加pixel-weighted visual capacity，且无需污染physical
  carrier；但固定PCA frame/3x3 grid没有跨视角一致优势，仍不是photorealistic appearance solution；
- prevention：下一步不得以只报pooled PSNR隐藏per-view方向，也不得事后换view/权重；若研究visual objective，只能在
  事前定义的view-uniform风险上训练/评价，geometry继续冻结；
- no new failure：预注册capacity gate通过，结果作为“aggregate supported / uniform rejected”保留。

下一可用编号仍为：`V71-F29`。

## V7.1 M26 prevention note — visual扩容不得成为第二套无监督物理几何（2026-09-05）

- 309个M8 carriers继续是唯一physical query；2781个visual surfels不进入M21 energy、collision或surface评价；
- visual center/scale/rotation只由冻结M8 center/scale的8NN PCA与固定3x3 tangent grid生成，并在RGB训练中冻结；
- 只训练目标Actor SH/opacity；trajectory、Background、其他Actor和所有geometry均无梯度；
- view split与M25逐值相同，M25 final `16.943422dB`在M26 quality前冻结为唯一capacity reference；
- 若不超过M25，关闭该分支，不开放image-to-geometry gradient、不扫描grid/thickness/seed；
- 来源边界：迁移Scaffold-GS anchor/local-Gaussian与2DGS surface-aligned primitive思想，不迁移其可学习几何。

下一可用编号仍为：`V71-F29`。

## V7.1 M25 outcome note — 外观梯度有效但309-carrier容量不足（2026-09-05）

- canonical r2在显式model-tree冻结后完成320/320 steps，6/6 held-out views均有非零Actor footprint，三项decision通过；
- pooled held-out Actor PSNR由M23初始化`15.8597`升至`16.9434dB`（`+1.0837dB`）；6个单视图全部改善，范围
  `+0.4856`至`+2.4501dB`；
- 同一held-out集合的原StreetGS Actor为`25.3527dB`，故优化后仍有`8.4093dB`缺口；train pooled仅
  `15.3677→17.0601dB`，不是只发生在held-out的统计偶然；
- interpretation：GT图像loss能穿过真实rasterizer训练冻结geometry上的attributes，排除“没有学习信号”；剩余主因是
  309个isotropic one-carrier primitives无法承载原32522个visual primitives的遮挡、轮廓与局部外观容量；
- boundary：不得为补画质移动M8 centers/scales、开放trajectory/Background，或把visual residual加入physical query；
  下一步若扩容，只允许由训练内3D surface support生成的独立visual children/surfels，图像仍只训练appearance；
- no new failure：M25预注册唯一学习判定通过，`V71-F28`保持为已解决实现失败。

下一可用编号仍为：`V71-F29`。

## V71-F28 — M25外层冻结未覆盖DriveStudio普通dict子模型（2026-09-05）

- 分类/状态：autograd ownership / resolved before post-training quality exposure；run=
  `20260904T181000Z__m25-geometry-locked-attribute-opt-r1`；
- symptom：第1步`total.backward()`触发PyTorch version-counter错误，涉及`[24,4]`的`IndexPutBackward0`；0 optimizer
  step、0 final/held-out quality exposure；
- root cause：DriveStudio `BaseTrainer.__init__`以普通`self.models = {}`持有RigidNodes/Background/CamPose/Affine，
  `trainer.parameters()`不会递归这些模型；因此旧冻结逻辑没有关闭RigidNodes trajectory quaternion梯度，而其
  `interpolate_quats`包含原地index assignment，进入颜色反传图后触发版本冲突；
- literature/action：PyTorch autograd官方文档说明原地写会维护version counter并在反传所需张量被修改时抛错；
  `requires_grad_(False)`应在forward前排除冻结子图。r2显式遍历`trainer.models.values()`冻结完整模型树，再只开放
  目标Actor的SH DC/rest与opacity；
- protocol isolation：不改carrier、view split、GT ROI、seed、320 steps、loss权重或学习率；不以`detach`掩盖未知图，
  不修改DriveStudio依赖源码；
- prevention：任何依赖外部trainer的属性优化必须核对模型容器是否为注册的`ModuleDict`；冻结声明以实际owner参数树为准；
- claim impact：r1是纯实现失败，不计appearance capacity结果，也不影响M8/M21 physical claim。

下一可用编号：`V71-F29`。

## V7.1 M25 prevention note — 图像监督只更新appearance attributes（2026-09-05）

- 309个M8/M23 centers/scales与identity rotations不加入optimizer；trajectory、Background和其他Actor参数全部冻结；
- 目标Actor SH/opacity虽与其他RigidNodes共用tensor，gradient hook仅允许该Actor 309行更新，Adam无weight decay；
- original-vs-hidden footprint只定义图像loss/评价ROI，不删除、移动、接受physical primitive，也不进入M21 physics；
- 8 train / 6 held-out frame-camera pairs由冻结3D视锥分层预选，M24已暴露frame42只放train；不按M25画质换view；
- 固定320 steps/seed71123/lr/loss，不扫参数；只写轻量appearance sidecar，不写StreetGS checkpoint；
- 唯一学习判定为pooled held-out footprint PSNR是否高于M23-nearest初始化；相对original剩余gap无论方向都报告。

下一可用编号仍为：`V71-F29`。

## V7.1 M24 paper prevention note — 视觉负例不得反向污染physical claim（2026-09-05）

- 主文同时给出GT/original/carrier同裁剪图和`-11.37dB` footprint PSNR，不以“renderer compatible”省略模糊/ghost；
- limitations把M23旧“未渲染”边界替换为真实native-rasterizer负结果，不再暗示attribute attachment足以获得photorealism；
- visual failure只约束appearance interface，不否定M8训练内GT surface supervision、M22 SE(3)组合或未决M21 external；
- 论文不提出尚未实现的visual residual shell结果，只把“隔离且受训练内3D support约束”写成后续必要条件。

下一可用编号仍为：`V71-F28`。

## V7.1 M24 outcome note — 可渲染不等于视觉保真（2026-09-05）

- r3以3D视锥规则冻结frame42后，3/3 camera与3/3 variants完成，visibility decisions 2/2通过；camera0形成
  `18971 px` Actor footprint，故不新增implementation failure；
- 309个GT-supervision-native physical Gaussians替换32522个StreetGS visual Gaussians后，Actor-footprint PSNR
  `28.592→17.223dB`（`-11.370dB`），camera0 full-image PSNR下降`3.993dB`，视觉上出现强烈模糊/ghost；
- 该负结果表明最近邻复制SH/opacity无法在大幅压缩视觉primitive后恢复方向性外观与细节；不得通过放宽footprint、
  调scale/opacity或把visual centers复制回physical carrier修复；
- representation结论：physical surface与appearance capacity必须分层。M8/M21 centers/scales继续由GT几何/first-return定义；
  后续若建visual shell，只能作为不进入physical query的有界残差层，并需用训练内3D surface约束限制其支持；
- claim impact：M23仅保留attribute-carrier接口结论，删除任何隐含photorealism暗示；M24不是geometry/physics失败，
  但否定naive one-carrier appearance替换。

下一可用编号仍为：`V71-F28`。

## V71-F27 — M24冻结中间帧不在目标Actor相机视锥内（2026-09-05）

- 分类/状态：evaluation support / resolved before any nonzero-footprint quality exposure；run=
  `20260904T171000Z__m24-geometry-locked-render-r2`；
- symptom：真实rasterizer完成3 variants x 3 cameras，但original、geometry-locked与actor-hidden逐像素相同，
  `visible_camera_count=0`、footprint/carrier changed pixels均为0，故verdict=`geometry_locked_render_interface_rejected`；
- root cause：Actor 12在checkpoint的196帧`instances_fv`均有效，但冻结frame98时已离开camera0/1/2视锥；使用冻结
  dataset extrinsics/intrinsics与GT Actor pose投影中心，前三视相机可见帧并集为`[0,84]`；
- literature/action：Street Gaussians/DriveStudio按逐帧刚体pose组合动态实例，实例时域有效不等于进入当前相机视锥；
  r3在读取任何非零footprint/PSNR前固定可见帧并集的中位数frame42；
- resolution：Actor、rigid index、全部3 cameras、checkpoint和M23 carrier不变；frame修正仅依赖3D相机几何，
  不按render quality挑帧，也不调scale/opacity；
- claim impact：r2没有可解释视觉质量，不能写成carrier正/负结果；保留9次成功render作为入口事实，但接口仍未判定。

下一可用编号：`V71-F28`。

## V71-F26 — M24误用缺少PyTorch3D的motionproj环境（2026-09-05）

- 分类/状态：environment entry / resolved before data or metric exposure；run=
  `20260904T170000Z__m24-geometry-locked-render-r1`；
- symptom：DriveStudio import在`pytorch3d.transforms`报`ModuleNotFoundError`；0 renders、0 quality rows；
- root cause：M24沿用V7.1几何脚本的motionproj Python3.10/torch2.4.1环境，但真实StreetGS checkpoint运行时属于
  独立drivestudio Python3.9/torch2.1.2+cu118/PyTorch3D0.7.5/gsplat1.3.0环境；
- literature/action：PyTorch3D官方要求与PyTorch/CUDA匹配安装或源码构建；服务器已有原生兼容环境，故不重复安装；
- resolution：r2使用`/root/autodl-tmp/envs/drivestudio/bin/python`，并移除对M22 runner的跨环境import；固定
  Actor/frame/camera/sidecar不变；
- claim impact：r1没有科学信息，不计render或quality failure。

下一可用编号：`V71-F27`。

