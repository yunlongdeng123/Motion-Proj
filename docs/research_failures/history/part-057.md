# 历史原始记录 057

> 冻结历史，不是当前执行规则。当前规则见 [scaling law](../../../auto-research_scaling_law.md) 与 [AGENTS](../../../AGENTS.md)。
> 原文件：`docs/RESEARCH_FAILURES.md`；迁移前提交 `abbd431c`。原有相对链接按原目录解释，证据路径原样保留。
> 按索引读取相关小节；旧哈希、过度门控和资源指令均不重新生效。

## V6.3 SurfNCC 防重复结论（2026-08-24）

- `V63-F01`（`engineering`, `resolved`）：P0 integration branch 的首次定向验证错误指向不存在的
  `tests/test_worldsim_v62_projection.py`，第二次改到真实文件后又因 pytest 进程未包含 repo-local `PYTHONPATH` 而在
  import collection 阶段触发 `ModuleNotFoundError: motion_proj`。两次均未读取数据/quality、未运行 GPU、未改变源码，
  不是 V6.2 projection 回归或 V6.3 算法失败。唯一恢复是在同一 conda 环境用
  `PYTHONPATH=. pytest -q tests/worldsim_v62/test_projection.py`，结果 `1 passed`。防重复：远端 repo 的定向 pytest
  必须使用真实 `tests/<version>/...` 路径并显式提供 repo-local import path；不为这一入口错误增加 smoke/regression
  矩阵。证据=`WS-V63-P0-SCOPE-GIT-01` 与 P0 shell terminal。

- `V63-F02`（`algorithm/evaluation`, `active`）：P2D canonical=
  `20260824T145924Z__native-pointwise-s0-r1` 用冻结 P5 best 与真实 per-cell IR-WM logits/BEV 执行 unchanged legacy28
  gate，Native B2仍为`4/28 ACCEPT,4/4 false-safe`，接受集合仍是scene-0242四个missing-route-support cases；R10=
  `2/3`、Actor gain=`0`、static/disocclusion gain=`2`、mask-area=`0.094024`、accepted FREE conflict mean/worst=
  `0.045783/0.092105`。source-valid UNKNOWN=`0.639211`，safe-OCC retention=`1.0`，hard violations=`0/939206`。
  与prototype P6/P6R相同的接受集合和false-safe说明V62-F05的feature bridge是加重因素而非主因；已推翻“只要恢复
  native feature，逐voxel CPSC即可获得hidden-surface authority”。防重复：不得对P2D重训、调threshold/seed/grid、
  用legacy O_eval选epoch或把mean conflict过门包装成安全。按预注册迁移到P3：Point Transformer V3官方实现支持
  efficient serialized point neighborhoods，visibility-aware reconstruction明确把FREE visibility作为surface约束，CVaR
  直接优化局部尾部；V6.3只迁移已在P1冻结的deterministic surface topology + patch CVaR，不改变alpha/cohort/gates。
  证据=`docs/autoresearch/worldsim_v63/P2D_NATIVE_POINTWISE_PREREG.md`、
  `https://openaccess.thecvf.com/content/CVPR2024/papers/Wu_Point_Transformer_V3_Simpler_Faster_Stronger_CVPR_2024_paper.pdf`、
  `https://pmc.ncbi.nlm.nih.gov/articles/PMC4897344/`。

- `V63-F03`（`engineering`, `resolved`）：P3 probe r1=`20260824T150842Z__surface-probe-s20260824-r1`
  在`_native_occupied_target_grid`对长度`300/300/40`的三个target-grid axis arrays调用`numpy.stack`时触发
  `ValueError: all input arrays must have the same shape`。该返回值未被调用方消费，失败发生在surface extraction、
  target supervision与任何quality gate之前，run仅4 KB，不能写成surface方法失败。NumPy官方`stack`合同要求每个输入
  shape相同；恢复为返回三个独立axis arrays，并在同轮pre-run audit中把route-support类型更新限定到对应local surface、
  法向量统计限定为finite unit vectors、target grid超出native z范围的点显式标为invalid而非用`100% valid`作错误门禁。
  这些都是接口/统计修复，不改变proposal volume、6-connected topology、patch参数、cohort或科研门槛。r1不可覆盖；
  revision 2复用冻结配置。防重复：不同长度的坐标轴不得stack；native coverage必须作为显式映射事实交给后续模型处理，
  不得偷偷clip或删除proposal。证据=`docs/autoresearch/worldsim_v63/P3_SURFACE_CORPUS_PREREG.md`、
  `https://numpy.org/doc/2.0/reference/generated/numpy.stack.html`。

- `V63-F04`（`engineering`, `resolved`）：P3 probe r2=`20260824T151429Z__surface-probe-s20260824-r2`
  在runner入口触发`FileExistsError`：外层launcher先`mkdir`了叶run directory，runner为保护不可变run又显式拒绝已存在
  路径。0 unit、0 surface、0 quality read；不能解释为F03恢复失败或科研结果。Python官方`Path.mkdir`说明默认
  `exist_ok=False`时目标存在即抛`FileExistsError`。恢复只让launcher确保task父目录存在、把叶目录留给runner原子创建；
  不修改源码、配置或科研合同。r2目录和console保留，revision 3使用新路径。防重复：带immutable-run自建语义的runner
  不得由外层预建叶目录。证据=`docs/autoresearch/worldsim_v63/P3_SURFACE_CORPUS_PREREG.md`、
  `https://docs.python.org/3/library/pathlib.html#pathlib.Path.mkdir`。

- `V63-F05`（`engineering/data-representation`, `resolved`）：P3 probe r3=
  `20260824T151618Z__surface-probe-s20260824-r3`首次完整产出`191 surfaces/498 patches/152226 points`，但101个
  微小static components的至少一点法向量无效（85个singleton，其余component size 3–11），令minimum normal-valid=
  `0`、probe未过。根因不是surface缺失：对称孤立voxel的六个外露面法向量相消，centroid fallback在离散medial-axis点
  也为零。Gradient-SDF说明SDF梯度给出normal但medial axis因最近surface不唯一而奇异；Open3D法向量接口要求需要时
  显式按camera location定向。恢复只在face-sum和centroid fallback都为零时，用target sensor viewpoint给出确定性单位
  方向，最后仅为sensor恰与点重合保留固定轴退路；不删除tiny proposal、不改变volume/topology/patch/cohort/gate。
  r3及其完整诊断保持不可变，r4使用新run。防重复：不得把tiny components静默过滤来换取normal-valid=1，也不得使用
  随机法向量。证据=`docs/autoresearch/worldsim_v63/P3_SURFACE_CORPUS_PREREG.md`、
  `https://openaccess.thecvf.com/content/CVPR2022/papers/Sommer_Gradient-SDF_A_Semi-Implicit_Surface_Representation_for_3D_Reconstruction_CVPR_2022_paper.pdf`、
  `https://www.open3d.org/docs/release/python_api/open3d.geometry.PointCloud.html`。

- `V63-F06`（`engineering/protocol`, `resolved`）：P3 r4=`20260824T152300Z__surface-probe-s20260824-r4`虽以
  `minimum normal-valid=1.0`和8/8 negative contracts得到runner `passed=true`，但formal前对照P1冻结point encoder
  schema发现payload缺signed FREE/OCC distance、patch-local coordinate、method/target behind-hit与第四个temporal support，
  且`ray_hit_order`字段实际保存raw metric distance。该问题不会改变r4的geometry capability，却使其不足以喂给冻结
  SurfNCC，故不得把r4写成完整P3 pass。恢复使用SciPy exact Euclidean distance transform按0.2m sampling生成仅依赖
  method-visible evidence的signed distances；patch coordinate减冻结patch centroid；hit order在每个surface ray bundle内按
  distance+lexicographic tie-break归一化，并另存raw distance；同时显式补behind-hit、temporal UNKNOWN与actor observed-hit。
  r5=`20260824T152843Z__surface-probe-s20260824-r5`验证上述aggregate字段后，P4 loader审计继续发现aggregate counts无法
  执行冻结的整段temporal-window dropout；同一恢复因此再补每个method sweep的state/contradiction `[point,sweep]`
  矩阵，并用配置中的单一required-field清单防止再次静默漏项。VideoMAE与MaST-Pre支持结构化时间mask应保留时间维，
  但V6.3不迁移其高mask ratio/预训练目标。无target信息进入proposal/feature decision、无新超参或quality选择。
  防重复：capacity前必须逐字段对齐P1 schema；不能用字段名掩盖语义错位、用aggregate冒充per-sweep，或事后删掉冻结输入
  以让loader先跑。证据=
  `docs/autoresearch/worldsim_v63/P3_SURFACE_CORPUS_PREREG.md`、
  `https://docs.scipy.org/doc/scipy/reference/generated/scipy.ndimage.distance_transform_edt.html`、
  `https://openaccess.thecvf.com/content/ICCV2021/papers/Zhao_Point_Transformer_ICCV_2021_paper.pdf`、
  `https://proceedings.neurips.cc/paper_files/paper/2022/file/416f9cb3276121c42eebb86352a4354a-Paper-Conference.pdf`、
  `https://openaccess.thecvf.com/content/ICCV2023/papers/Shen_Masked_Spatio-Temporal_Structure_Prediction_for_Self-supervised_Learning_on_Point_Cloud_ICCV_2023_paper.pdf`。

- `V63-F07`（`engineering`, `resolved`）：r6 pre-run per-sweep窄检查首次凭scene名猜测processed path为
  `trainval/000`，在读取`instances_info.json`时触发`FileNotFoundError`；没有创建run或读取quality。冻结cohort是该映射的
  唯一事实源，实际`scene-0071 processed_index=68`；改用`trainval/068`后检查通过，state与contradiction均为
  `[3,300,300,40]`且逐voxel FREE+OCC+UNKNOWN count恒等于3。防重复：raw processed目录只按cohort metadata中的
  `processed_index`解析，不从scene display name猜目录；这一入口错误不扩展smoke/regression。证据=P3 r6 pre-run shell与
  `configs/worldsim_v62/p2_development_cohort_v1.yaml`。

- `V63-F08`（`engineering`, `resolved`）：P4尚未解锁执行时的temporary synthetic AMP interface r1在
  `binary_cross_entropy(sigmoid(hidden_free/authority))`触发PyTorch RuntimeError；128个随机点、无真实surface/quality、
  未创建正式run。PyTorch官方AMP文档明确说明BCELoss backward梯度可能无法用FP16表示，autocast因此主动拒绝，并要求
  使用`binary_cross_entropy_with_logits`。恢复同时输出hidden-FREE/authority logits供loss使用，推理概率仍为sigmoid；
  r2合成forward/backward finite，proposal-token gradient存在。防重复：FP16训练的二元head必须保留logits并在autocast
  下用BCE-with-logits，不通过禁用AMP或转FP32绕开冻结precision。证据=`docs/autoresearch/worldsim_v63/P4_CAPACITY_PREREG.md`、
  `https://docs.pytorch.org/docs/stable/amp.html#prefer-binary-cross-entropy-with-logits-over-binary-cross-entropy`。

- `V63-F09`（`engineering`, `resolved`）：未来P5的packed-proposal synthetic r1把NumPy API名迁到PyTorch，调用不存在的
  `torch.flatnonzero`而在token selection前失败；128随机点、无真实surface/quality、无正式P5 run。PyTorch官方提供
  `torch.nonzero(input, as_tuple=False)`返回二维索引；对一维mask用`.squeeze(1)`得到所需索引。替换后r2完成2个proposal、
  4个patch的FP16 forward/backward，proposal CVaR shape=`[2]`且Transformer/proposal-token gradient非零。防重复：
  NumPy的`flatnonzero`不得假设存在于torch namespace；CUDA mask索引统一使用官方`torch.nonzero`/`torch.where`接口。
  证据=temporary packed-interface terminal、`https://docs.pytorch.org/docs/stable/generated/torch.nonzero.html`。

- `V63-F10`（`engineering/protocol`, `resolved_preexecution`）：P4/P5正式执行前对首40个P3完成单元做method-only结构读取，
  发现每个单元恰有一个surface超过冻结的8192-point microbatch（`40/40`，最大`173488` points）；这些大surface的完整
  patch set平均`297.45`、最大`417` tokens。H-P4-001只取最大proposal的首chunk，既未覆盖全部点，也只让每个chunk
  独立生成proposal token；这只能证明局部point memory，不能证明P1冻结的complete-proposal interaction。没有创建P4 run、
  启动GPU或读取target quality。H-P4-001因此在执行前withdrawn；H-P4-002保持同一两个unit、模型宽度、两步AdamW、
  accumulation4、CVaR/gate/resource不变，改为先按8192点编码完整patch，再把当前proposal的全部patch token汇合后运行
  两层attention与唯一proposal token。Set Transformer与Perceiver支持用小型set/latent bottleneck承接大输入；本迁移
  不增加learned token、删点、改分辨率或改denominator。未切块路径的12项输出模块化等价审计max abs diff=`0.0`。
  防重复：capacity不得用首chunk冒充完整proposal；point microbatch只能切point graph，proposal identity与patch context
  必须在上层重组。证据=`docs/autoresearch/worldsim_v63/P4_CAPACITY_PREREG.md`、
  `https://proceedings.mlr.press/v97/lee19d.html`、`https://proceedings.mlr.press/v139/jaegle21a.html`。

- `V63-F11`（`engineering/protocol`, `resolved_preexecution`）：同一次P5执行前审计发现packed chunk曾各自计算actor/safe/
  unsafe标签、各自抽structural dropout，并在移除hard/temporal/actor-observed evidence后仍保留原`authority_bits`输入与
  原authority标签；selection还把各chunk hidden-FREE CVaR的最大值当完整proposal CVaR。这会让同一proposal跨chunk
  标签/selector漂移、从辅助authority通道看见已mask支持，并改变tail统计。没有真实P5 run、checkpoint或quality read。
  恢复把actor/safe/unsafe/full point count绑定完整proposal；每epoch/proposal只抽一个semantic selector并由所有chunk
  消费；遮蔽后从剩余method/temporal支持重算authority输入和监督，保留合法Actor current/swept与closure支持；selection
  汇合全部hidden-FREE点后精确计算alpha0.90 proposal CVaR，并统一为hard projection优先、仅learned low-authority OCC
  转UNKNOWN的最终decision后再统计coverage/UNKNOWN/accuracy。训练端明确保留内存受限的packed stochastic CVaR surrogate，
  不冒充exact full-batch optimizer。MAE支持“encoder不可见mask输入、原semantic target仍监督”；minibatch risk文献提示
  tail functional在小batch上可能有偏。防重复：任何evidence-derived辅助通道必须与mask同步；chunk-local统计不得冒充
  proposal统计。证据=`docs/autoresearch/worldsim_v63/P5_TRAIN_PREREG.md`、
  `https://openaccess.thecvf.com/content/CVPR2022/html/He_Masked_Autoencoders_Are_Scalable_Vision_Learners_CVPR_2022_paper.html`、
  `https://arxiv.org/abs/2301.11724`、
  `https://openaccess.thecvf.com/content/CVPR2025/papers/Kalble_EvOcc_Accurate_Semantic_Occupancy_for_Automated_Driving_Using_Evidence_Theory_CVPR_2025_paper.pdf`。

- `V63-F12`（`engineering/protocol`, `resolved_preexecution`）：完整proposal context恢复后继续逐loss审计，发现ranking仍只在
  当前packed chunk恰好共现的safe/unsafe proposals间配对；跨chunk的大proposal会多次占用同一safe对照，而不共现的
  nearest-size pair永远没有loss。这违反P1冻结的complete-proposal、同actor/static stratum、nearest full-point-count
  一对一匹配。没有真实P5 run、checkpoint或quality read。Cross-Batch Memory证明小batch pair mining可由历史embedding
  扩展，但其stale queue与额外memory state在本项目无必要：每个unit的完整patch token set本来就小。恢复先从完整unit
  metadata一次性生成一对一pair，再用当前权重的detached完整patch-token cache运行可微proposal attention/risk head，
  每unit只施加一次全proposal ranking；point losses仍按8192 chunks有界。margin=`0.10`、weight=`0.25`、labels、cohort、
  optimizer与denominator均不变，也没有新增queue/momentum/hyperparameter。防重复：proposal-level pair loss不得由chunk
  共现关系定义；batching只能限制point graph，不能改变匹配集合。聚焦语义审计把safe/unsafe proposals置于两个不同chunk，
  仍得到冻结unit pair=`[(0,1)]`。证据=
  `docs/autoresearch/worldsim_v63/P5_TRAIN_PREREG.md`、
  `https://openaccess.thecvf.com/content_CVPR_2020/html/Wang_Cross-Batch_Memory_for_Embedding_Learning_CVPR_2020_paper.html`。

- `V63-F13`（`engineering/protocol`, `resolved_preexecution`）：batch invariance审计发现6-neighbor edge曾按surface identity
  构建，但一个大surface按patch边界进入多个point chunks时，跨chunk patch边会被静默删除；同一对patch偶尔共处一个chunk
  时该边又存在，因此输出依赖packing而非冻结几何。没有真实P4/P5 run、GPU结果或quality read。GraphSAINT明确指出induced
  subgraph minibatch丢失外部边会产生sampling bias；Point-BERT则提供local patch先编码为token、再由Transformer组合的成熟
  分层结构。V6.3不引入随机subgraph sampling、归一化估计或halo超参，而是把两层deterministic 6-neighbor aggregation的
  local neighborhood明确绑定到已冻结的完整patch；patch最大2048且从不切分，所以边集合与8192 packing无关，跨patch交互
  由完整proposal patch attention承担。proposal surface/6-connectivity、patch membership、模型层数、point features、cohort
  与denominator不变。防重复：任何point microbatch必须保持local encoder的计算单元完整；不能让edge存在性取决于邻patch
  是否碰巧同batch。两完整patch的聚焦语义审计得到full/split有向边数=`4/4`。证据=
  `docs/autoresearch/worldsim_v63/P4_CAPACITY_PREREG.md`、
  `https://openreview.net/pdf?id=BJe8pkHFwS`、
  `https://openaccess.thecvf.com/content/CVPR2022/html/Yu_Point-BERT_Pre-Training_3D_Point_Cloud_Transformers_With_Masked_Point_Modeling_CVPR_2022_paper.html`。

- `V63-F14`（`engineering/protocol`, `resolved_preexecution`）：训练端已把matched ranking限定为完整scene/frame unit，
  但selection汇总曾把24个selection units的proposal rows一次交给全局nearest-size matcher，因此safe/unsafe可跨scene/frame
  配对；完整proposal risk虽正确，checkpoint objective仍会受跨案例规模巧合影响。没有真实P5 run、checkpoint或quality
  read。CVPR 2016 lifted structured embedding与CVPR 2022 graph sampling都说明pair mining的候选关系/采样边界是目标的一部分，
  不能把扩大候选池当作中性的batch实现。恢复仅按`(scene,target_frame)`分组执行原有actor/static、nearest full-point-count、
  one-to-one matcher，再对有pair的unit loss等权平均；margin=`0.10`、weight=`0.25`、proposal risk、cohort、threshold与gate
  均不变。聚焦synthetic把safe/unsafe分别置于两个unit得到`0 pair`。防重复：train/selection的proposal matching边界必须
  同为完整unit；不得跨case挖pair或引入memory queue。证据=`docs/autoresearch/worldsim_v63/P5_TRAIN_PREREG.md`、
  `https://openaccess.thecvf.com/content_cvpr_2016/html/Song_Deep_Metric_Learning_CVPR_2016_paper.html`、
  `https://openaccess.thecvf.com/content/CVPR2022/html/Liao_Graph_Sampling_Based_Deep_Metric_Learning_for_Generalizable_Person_Re-Identification_CVPR_2022_paper.html`。

- `V63-F15`（`engineering/evaluation`, `resolved_preexecution`）：P4已有`cvar_gradient_nonzero` gate曾在total loss backward后
  检查`hidden_free_head.weight.grad`，但该head同时受BCE-with-logits监督；即使proposal CVaR图断开，BCE也足以让flag
  为true，造成capacity假阳性。没有真实P4 run、quality read或scientific denominator。CVaR优化的一手工作明确把risk
  objective的梯度作为优化对象，PyTorch官方`autograd.grad`提供指定outputs到inputs的直接VJP。恢复在原forward图上对
  `proposal_cvar.mean()`分别向state/hidden-free/authority heads求梯度并只接受finite nonzero direct path；聚焦synthetic
  三条head路径均通过。原gate名称、阈值、units、steps、模型与资源合同不变，也未新增回归矩阵。防重复：多项loss共享
  parameter时，不得以总梯度证明某个特定loss已连通。证据=`docs/autoresearch/worldsim_v63/P4_CAPACITY_PREREG.md`、
  `https://docs.pytorch.org/docs/stable/generated/torch.autograd.grad.html`、
  `https://proceedings.mlr.press/v235/kim24x.html`。

- `V63-F16`（`data/evaluation`, `resolved`）：P3 formal终态前语义审计发现surface registry与summary字段
  `hidden_free_count/hidden_free_point_count`实际只计算`target_state==FREE`，没有同时要求
  `method_state==UNKNOWN && !method_contradiction`，所以该描述统计不能按hidden-FREE引用。point NPZ中的method/target/
  contradiction、proposal/patch/native features均正确；P4不消费此计数，P5 training/selection从point arrays使用正确布尔
  条件，因此不是标签污染或语料重建失败。NeurIPS dataset documentation实践强调保留旧版本并显式记录metadata限制。
  恢复边界：不改写formal artifact；terminal后从72个原始NPZ一次重算得到target FREE/OCC/UNKNOWN=
  `1545584/335050/9702367`、correct hidden-FREE=`688837`，已在三本账与P3 prereg登记勘误；未来materializer
  以additive v2把`target_free/occupied/unknown`与正确`hidden_free`分字段输出。
  canonical r6 probe一次重算得到target FREE/OCC/UNKNOWN=`19609/3891/128726`、correct hidden-FREE=`8311`，确认旧值
  `19609`只是target-FREE。
  禁止因描述字段误名重跑13.213小时正确point corpus，也禁止继续引用旧summary的hidden-FREE数字。证据=
  `docs/autoresearch/worldsim_v63/P3_SURFACE_CORPUS_PREREG.md`、
  `https://arxiv.org/abs/1803.09010`、
  `https://papers.neurips.cc/paper_files/paper/2024/file/605bbd006beee7e0589a51d6a50dcae1-Supplemental-Datasets_and_Benchmarks_Track.pdf`。

- `V63-F17`（`engineering/numerics`, `resolved`）：H-P4-002 r1 canonical=
  `20260825T045854Z__capacity-h002-s0-r1`在11.181s完成全部2 train/2 selection complete proposals，peak仅
  `0.196070 GiB`，finite loss、direct CVaR三head gradient、proposal-token gradient、hard violations=`0`、checkpoint
  reload与selection finite均成立；但unscale后的total gradients含nonfinite，且same-model/reloaded FP16 forward max abs
  difference均为`9.059906e-6`，所以冻结finite与exact-zero determinism gate诚实未过。没有quality conclusion、calibration、
  confirmation或test read，不是资源/算法失败。PyTorch官方AMP文档说明默认initial scale可能使FP16 gradient overflow；
  reproducibility文档说明CUDA SDPA不同backend/backward确定性不同，math backend配合deterministic algorithms可确定执行。
  唯一有界r2恢复保留FP16但固定GradScaler initial scale=`1024`，禁用flash/memory-efficient SDPA、只启用math SDPA并开启
  deterministic algorithms。模型、units、dropout、loss、optimizer LR/WD、2 steps、accum4、gate与22GiB ceiling均不变。
  r3=`20260825T051200Z__capacity-h002-s0-r3`在同一合同下以finite gradient与exact-zero repeat/reload正式通过。
  防重复：不得放宽exact-zero阈值、忽略nonfinite flag、增加steps或把r1写成quality negative。
  证据=`docs/autoresearch/worldsim_v63/P4_CAPACITY_PREREG.md`、
  `https://docs.pytorch.org/docs/stable/amp.html`、
  `https://docs.pytorch.org/docs/stable/notes/randomness.html`。

- `V63-F18`（`engineering/runtime`, `resolved`）：H-P4-002 r2 canonical=
  `20260825T050400Z__capacity-h002-s0-r2`在第一个CUDA math attention forward处被PyTorch deterministic-algorithm runtime
  拒绝：CUDA>=10.2的cuBLAS矩阵运算只有在进程启动前设置`CUBLAS_WORKSPACE_CONFIG=:4096:8`或`:16:8`后才允许确定执行。
  r2在任何optimizer step、capacity summary、quality/calibration/confirmation/test read之前终止，run leaf为空；因此它没有检验
  F17的AMP-scale或exact-zero恢复，不是第二个科研/数值尝试。NVIDIA cuBLAS官方结果可重复性说明`:4096:8`会固定workspace
  配置且增加约24 MiB，PyTorch deterministic文档对CUDA matmul给出同一前置条件。恢复把`:4096:8`同时绑定到launcher和
  runner的pre-torch-import环境，并由P4/P5配置显式记录；新增开销远低于22 GiB ceiling。r3继续F17的同一次有界恢复，
  model/data/FP16/AMP scale/SDPA backend/dropout/loss/optimizer/steps/accum/gates均不变。防重复：不得关闭determinism、放宽
  exact-zero门或把入口异常写成capacity/quality失败。r3实际peak=`0.256589 GiB`、wall=`11.863s`并正式passed，说明环境
  恢复闭合而无需新增资源或协议变化。证据=`docs/autoresearch/worldsim_v63/P4_CAPACITY_PREREG.md`、
  `https://docs.nvidia.com/cuda/cublas/index.html#results-reproducibility`、
  `https://docs.pytorch.org/docs/stable/generated/torch.use_deterministic_algorithms.html`。

- `V63-F19`（`algorithm/evaluation`, `resolved_by_constrained_recovery_p6_unlocked`）：P5 canonical=
  `run://worldsim_v63/WS-V63-P5-SURFNCC-TRAIN-01/20260825T051530Z__surfncc-train-s0-r1`完成全部48个train与24个
  scene-disjoint selection units，`7 epochs/1792 steps`、finite training、peak=`0.403084 GiB`且累计hard violations=`0`；
  runner据此正确报告capacity/training `passed=true`。但冻结lexicographic objective选择的epoch 3仅是best
  training-objective checkpoint：safe-OCC retention=`0`、emitted-OCC coverage=`0.0371977<0.10`、source-valid UNKNOWN=
  `0.861807>0.60`。该checkpoint把有正向OCC支持的安全曲面与危险/缺证据曲面一起拒绝，不能作为SurfNCC candidate，
  不能用低false-safe或低tail掩盖零仿真效用。这是positive-authority collapse症状；现有证据尚不足以区分representation/
  supervision重叠、raw/post-projection decision composition或weighted-objective optimization collapse，故不得提前把根因写成
  ordinary underfit或任一优化结论。

  P5D H002 canonical=`run://worldsim_v63/WS-V63-P5D-AUTHORITY-COLLAPSE-DIAGNOSTIC-01/20260825T084844Z__authority-diagnostic-s0-r2`
  已把根因收敛：safe-OCC raw/projected/post-authority decision均为实际`[FREE,OCCUPIED,UNKNOWN]=[153,0,62301]`且
  authority veto=`0`，排除hard projection与decision composition；raw `P(OCC)`虽以AUC=`0.722684`保留弱排序，绝对
  mean仅`0.006459`，`q_AUTH` AUC也仅`0.578070`。weighted tail/retention gradient mean比=`5.531x`，direct-tail与
  state-head比分别=`1.715x/1.732x`，tail-retention cosine mean=`-0.411568`；retention loss mean=`0.968547`。
  因此primary root确认为weighted-objective optimization collapse，evidence-authority supervision弱对齐为次级机制，
  而不是solver或authority veto失败。

  P5R canonical=
  `run://worldsim_v63/WS-V63-P5R-CONSTRAINED-SURFNCC-TRAIN-01/20260825T091631Z__constrained-train-s0-r1`
  以同一SurfNCC representation、数据、hard projection、seed0与P5 epoch3 model-only warm-start运行proxy primal-dual；
  retention/emitted-OCC/non-UNKNOWN改为约束，旧weighted retention term置0。formal完成`10 epochs/2560 steps`、finite、
  hard violations=`0`，未读P6/calibration/H/T。best feasible epoch6的retention=`0.721226`、coverage=`0.114148`、
  non-UNKNOWN=`0.686101`，四项exact gate全过，tail+rank=`0.520541`，因此`candidate_promotable=true`并解锁P6。
  epoch 7–9连续三轮没有更优feasible candidate后按patience停止；尤其epoch 8/9虽tail更低但coverage/UNKNOWN失门，未覆盖
  epoch6。由此F19的positive-authority collapse已由约束优化闭合，而不是靠降低gate或回改solver闭合。

  防重复：不得增加epoch、换seed、加大模型、改变CVaR alpha、降低retention/coverage/UNKNOWN gate、提高
  `lambda_ret`，也不得回改已连续零违反的FREE/OCC projection、ray hard constraint、lifecycle或V6.2 solver。P5R不再追加
  recovery/sweep；合法下一步仅为冻结best candidate进入原P6 fresh matched AB。P6必须保留Native B2、surface encoder、
  CVaR与authority消融及原晋级门；P5R的candidate pass不能冒充P6/校准/confirmation/deployment结论。
  证据=`docs/autoresearch/worldsim_v63/P5_TRAIN_PREREG.md`、
  `docs/autoresearch/worldsim_v63/P5D_AUTHORITY_COLLAPSE_DIAGNOSTIC_PREREG.md`、
  `configs/worldsim_v63/p5d_authority_collapse_diagnostic_v1.yaml`、
  `scripts/run_worldsim_v63_p5d_authority_diagnostic.py`、
  `configs/worldsim_v63/p5r_constrained_surfncc_train_v1.yaml`、
  `scripts/run_worldsim_v63_p5r_constrained_train.py`、
  `https://proceedings.mlr.press/v97/geifman19a.html`、`https://proceedings.mlr.press/v98/cotter19a.html`、
  `https://proceedings.mlr.press/v97/cotter19b.html`。

- `V63-F20`（`engineering/runtime`, `resolved`）：H-P5D-001第一次formal入口在创建run leaf、读取P5
  checkpoint/train arrays或建立CUDA context前，将`shutil.disk_usage`直接调用于尚不存在的
  `/root/autodl-tmp/runs/worldsim_v63/WS-V63-P5D-AUTHORITY-COLLAPSE-DIAGNOSTIC-01`，触发`FileNotFoundError`。
  新task namespace按设计尚未由runner创建，所以canonical run=`null`，没有任何分布、梯度或科学结论。Python官方
  `shutil.disk_usage(path)`要求path指向已有filesystem位置；`Path.mkdir(parents=True)`才负责创建缺失父目录。
  H-P5D-002只在disk check前向上寻找最近已存在父目录并执行相同20 GiB资源检查，之后仍由formal runner创建唯一leaf；
  checkpoint、48-unit分布、4-unit gradient probe、模型/FP16、threshold/gate、零optimizer与P6/H/T locks全部不变。
  防重复：新task resource check不得假设namespace已存在，也不得为了通过检查预创建并冒充failed/canonical run；本恢复不
  增加smoke或质量读取。证据=`scripts/run_worldsim_v63_p5d_authority_diagnostic.py`、
  `https://docs.python.org/3/library/shutil.html#shutil.disk_usage`、
  `https://docs.python.org/3/library/pathlib.html#pathlib.Path.mkdir`。

- `V63-F21`（`evaluation/metadata`, `resolved`）：P5D H002 canonical的`DECISION_STAGE_COUNTS.json`把
  `class_order`描述文字写成`UNKNOWN/FREE/OCCUPIED`，但生成counts的`torch.bincount`直接以argmax class index为bin，冻结
  project constants实际为`FREE_INDEX=0/OCCUPIED_INDEX=1/UNKNOWN_INDEX=2`。因此三个counts数组本身、raw/projected/
  post-authority等值关系、authority veto=`0`、全部distribution/gradient/summary均正确；错误只在数组标签文字。正确
  safe-OCC counts为`FREE/OCCUPIED/UNKNOWN=153/0/62301`，不是旧文字顺序的解释。canonical artifact保持不可变，runner
  future label已改成`FREE/OCCUPIED/UNKNOWN`；不为13分钟正确诊断重跑，也不改写artifact。防重复：任何class-count数组
  必须从同模块index constants生成或明确按constants记录order，不凭tri-state自然语言习惯手写顺序。证据=
  `motion_proj/worldsim_v62/projection.py`、`scripts/run_worldsim_v63_p5d_authority_diagnostic.py`、
  `https://docs.pytorch.org/docs/stable/generated/torch.argmax.html`、
  `https://docs.pytorch.org/docs/stable/generated/torch.bincount.html`。

- `V63-F22`（`engineering/operations`, `resolved`）：P5R terminal文档收口的首次SSH备份命令在本地PowerShell双引号中
  使用远端`$b`，变量在发送前被本地展开，导致远端备份在复制前失败并在项目外创建`/docs`重复副本树；随后一次包含
  `$(realpath /docs)`的保护命令也先被本地PowerShell解释并在任何删除动作前失败。两次均未修改
  `/root/autodl-tmp/motion_proj`、canonical run、checkpoint或Git工作树。只读`find`确认`/docs`全部是可由原仓库恢复的
  重复副本后，以显式绝对目标删除该树；随后不用变量或命令替换，以显式
  `/tmp/worldsim_v63_pre_p5rclose_20260825T1440Z`成功备份七个文档。P6 prereg同步后的inline `python -c` YAML检查又因
  同一PowerShell→SSH引号层在读文件前`SyntaxError`；改为将只读Python源码经stdin传给远端解释器后验证通过，项目仍未变。
  防重复：从PowerShell发送SSH文件操作时，不在双引号命令中使用远端`$var`、`$(...)`或嵌套inline Python字符串；备份、
  清理目标使用已解析的显式绝对路径并拆成独立步骤，结构验证统一经stdin发送。

- `V63-F23`（`engineering/runtime`, `resolved_pre_quality_read`）：P6 B0/B1/B2首次formal入口直接执行
  `python scripts/run_worldsim_v63_p6_development_ab.py`，解释器按官方合同只把输入脚本所在`scripts/`目录置于module search
  path首位，因而在首个project import触发`ModuleNotFoundError: motion_proj`。失败发生在run leaf创建、P3/native数据、B2
  checkpoint与CUDA context之前，canonical run=`null`，没有P6 quality或科学结果。恢复不改源码/配置/denominator/gate，
  只从repo root改用`python -m scripts.run_worldsim_v63_p6_development_ab`，使当前目录进入module path；同解释器`--help`
  入口验证通过。防重复：repo-local runner若import project package或兄弟`scripts` module，formal launcher统一用`python -m`
  或已安装console entry point，不把direct-file import失败登记为算法reject，也不为此扩展smoke矩阵。证据=
  `https://docs.python.org/3/library/sys_path_init.html`、
  `https://packaging.python.org/en/latest/guides/creating-and-packaging-command-line-tools/`。

- `V63-F24`（`algorithm/evaluation`, `active route-closed`）：P6 B3 Surface-Mean虽在训练内冻结epoch1 feasible checkpoint
  （hard0、retention=`0.636863`、OCC coverage=`0.285326`、UNKNOWN=`0.550411`），但统一逐scene stage evaluator在两scene
  都不优于冻结Native B2。scene-0450 common surface hidden-FREE CVaR=`0.596685 vs 0.497850`，相对改善=
  `-19.852%`，accepted area ratio=`0.406270`且source-valid UNKNOWN=`0.651678>0.60`；scene-1089 tail=
  `0.655861 vs 0.465122`，改善=`-41.008%`，area ratio=`0.499323`。两scene hard0、retention、case、actor/static过门，
  说明失败不是hard solver回归或all-UNKNOWN，而是surface architecture在保留一定OCC后仍同时放大hidden-FREE tail并显著
  收缩相对Native B2的写入面积。supporting scenes=`0/2`，H-P6-001 rejected。

  主计划Stop2因此关闭surface architecture family：B4 Surface-Max、B5 Surface-CVaR和M0 authority均不执行，H-P6-002/
  H-P6-003关闭未读，P7没有frozen P6 M0输入而保持locked；legacy/calibration/confirmation/test均未读。不得用pooled
  retention、训练内candidate、较高accuracy或hard0掩盖逐scene tail/area失败，也不得换seed、加大模型、改CVaR alpha、
  降低area/UNKNOWN/2%门或先读legacy/H/T复开。未来合法复开必须在新版本预注册feature-level aleatoric/epistemic
  uncertainty与scene/stratum-conditional coverage约束，并使用fresh development denominator；相关候选仅为EvOcc
  （CVPR 2025）、ReliOcc（IJCAI 2025）、OCCUQ（ICRA 2025开源）及UAI 2024 conditional robust optimization，不构成
  V6.3 recovery授权。证据=`docs/autoresearch/worldsim_v63/P6_SURFACE_FAMILY_CLOSEOUT.md`、
  `run://worldsim_v63/WS-V63-P6-DEVELOPMENT-AB-01/20260826T014500Z__b3-eval-s0-r1`、
  `https://openaccess.thecvf.com/content/CVPR2025/html/Kalble_EvOcc_Accurate_Semantic_Occupancy_for_Automated_Driving_Using_Evidence_Theory_CVPR_2025_paper.html`、
  `https://www.ijcai.org/proceedings/2025/220`、`https://github.com/ika-rwth-aachen/OCCUQ`、
  `https://proceedings.mlr.press/v244/chenreddy24a.html`。

<a id="detail-v62"></a>

