# 历史原始记录 056

> 冻结历史，不是当前执行规则。当前规则见 [scaling law](../../../auto-research_scaling_law.md) 与 [AGENTS](../../../AGENTS.md)。
> 原文件：`docs/RESEARCH_FAILURES.md`；迁移前提交 `abbd431c`。原有相对链接按原目录解释，证据路径原样保留。
> 按索引读取相关小节；旧哈希、过度门控和资源指令均不重新生效。

## V6.4 原生不确定性编译器详细账本（2026-08-26）

- `V64-F01`（`engineering/runtime`, `resolved_pre_quality_read`）：P0 定向投影测试首次使用控制台入口
  `pytest -q tests/worldsim_v62/test_projection.py`，测试收集阶段即因仓库根目录未进入 `sys.path`触发
  `ModuleNotFoundError: motion_proj`。失败发生在 formal run、GPU context、数据与 V6.4 quality read 之前；没有科学结果。
  依据 pytest 官方 import-path 合同，仅把入口改为
  `python -m pytest -q tests/worldsim_v62/test_projection.py`，结果 `1 passed in 1.59s`。防重复：仓库内测试统一用
  `python -m pytest`；不为入口差异改包结构、污染环境或扩展回归矩阵。证据=
  `WS-V64-P0-SCOPE-GIT-01`、`docs/autoresearch/worldsim_v64/P0_SCOPE.md`、
  `https://docs.pytest.org/en/stable/explanation/pythonpath.html`。

- `V64-F02`（`engineering/operations`, `resolved_pre_formal_run`）：UQ prereg commit 首次两次普通
  `git push`在 30 秒窗口内无输出，远端 ref 保持 `04b343f`，本地 `f1764de`未丢失，formal run 尚未启动。GitHub
  官方状态页显示 Git Operations operational；本地 `localtun sessions`确认当前 AutoDL session 的 remote proxy 后，
  仅为该 push 显式设置 HTTP/HTTPS/ALL proxy，普通 push 成功推进远端到 `f1764de`。防重复：远端网络慢时先检查
  GitHub 状态和当前 LocalTUN session；端口是会话级，禁止复用旧记忆值，也不 force push。证据=`f1764de`、
  `https://www.githubstatus.com/`。

- `V64-F03`（`engineering/runtime`, `resolved_post_run_read`）：canonical UQ run 已成功结束后，第一次只读 summary
  命令在非登录 shell 直接调用裸 `python`，因环境未激活在读取文件前返回 `command not found`。按仓库环境合同 source
  `conda.sh`并激活 `motionproj`后，同一只读程序完成逐 scene 指标读取；run、summary 与模型均未修改。防重复：任何
  非登录 Python 命令显式激活环境；该错误不登记为算法或 formal run failure。证据=
  `run://worldsim_v64/WS-V64-P3-NATIVE-UQ-01/20260826T080200Z__uq-retrospective-s0-r1`、
  `https://docs.conda.io/projects/conda/en/25.1.x/dev-guide/deep-dives/activation.html`。

- `V64-F04`（`engineering/runtime`, `resolved_pre_data_read`）：fresh sidecar 首次 formal 入口
  `20260826T081300Z__fresh-native-s0-r1`在 wrapper 调用继承 runner 后，因 task parent 尚不存在而对
  `shutil.disk_usage`触发 `FileNotFoundError`。run leaf 未创建，GPU、processed scene、IR-WM 与 quality 均未触达，
  canonical run=`null`。恢复只在 wrapper 中先 `mkdir` task parent，再由未改的 runner 创建 exclusive run leaf；
  cohort、seed、资源门和 denominator 不变。防重复：disk probe 必须绑定已存在的挂载内路径，不把前置目录缺失写成磁盘
  不足或算法失败。证据=`WS-V64-P2-FRESH-NATIVE-SIDECAR-01`、
  `https://docs.python.org/3/library/shutil.html#shutil.disk_usage`。

- `V64-F05`（`data/interface`, `resolved_pre_quality_read`）：fresh sidecar r2=
  `20260826T081500Z__fresh-native-s0-r2`启动冻结 IR-WM worker 后，初始 cohort 中 val-split 的`scene-0100`与
  `scene-0632`在`nuscenes_temporal_infos_train.pkl`查询处触发`KeyError`。`scene-0230`已完成12个native units，
  worker wall=`35.8975 s`、peak GPU=`4.1305 GiB`，blocked run leaf总计`528 MiB`；其余scene未形成完整denominator，
  canonical=`null`，target evidence与任何fresh quality均未读。根因是selector只核对processed/raw可用性，却漏掉冻结
  extractor的train temporal metadata membership。检索IR-WM、BEVFormer与DriveStudio官方数据准备合同后，恢复仅在
  pre-quality 阶段改冻为六个均在train temporal metadata、且未进入V6.1–V6.3 quality ledger的scene；evaluation两scene
  从本机raw数据用官方DriveStudio流程物化。防重复：保留r2，不把12个部分unit混入r3，不生成val temporal metadata救旧
  cohort，不改变seed/targets/model/UQ/门；r3必须使用全新exclusive leaf完成72-unit denominator。证据=
  `docs/autoresearch/worldsim_v64/P2_FRESH_COHORT_FREEZE.md`、
  `https://github.com/ziyc/drivestudio/blob/main/docs/NuScenes.md`、
  `https://github.com/fundamentalvision/bevformer`、`https://github.com/APRIL-ZJU/IR-WM/blob/ir-wm/README.md`。

- `V64-F06`（`engineering/operations`, `resolved_post_run_read`）：r3 已成功输出正式summary后，首次只读收口程序
  假定文件名为`P2_NATIVE_SUMMARY.json`，对不存在路径调用`Path.read_text()`触发`FileNotFoundError`；run、GPU、
  artifacts与quality均未修改。实际继承的V6.3 extractor写出`P2_SUMMARY.json`。按Python pathlib官方合同先枚举run根目录，
  再读取实际文件；canonical r3与全部指标不变。防重复：继承runner的consumer不得根据task或版本猜文件名，先用明确目录
  枚举或读取runner源码中的输出合同；不为只读路径错误重跑formal。证据=
  `run://worldsim_v64/WS-V64-P2-FRESH-NATIVE-SIDECAR-01/20260826T082600Z__fresh-native-s0-r3`、
  `docs/autoresearch/worldsim_v64/P2_FRESH_SIDECAR_CLOSEOUT.md`、
  `https://docs.python.org/3/library/pathlib.html`。

- `V64-F07`（`engineering/operations`, `resolved_pre_run`）：fresh evidence首次launcher在Windows PowerShell传给SSH的
  双引号字符串中使用`$(git status --porcelain)`；PowerShell按官方解析规则在本地提前执行subexpression，因本地工作目录
  不是repo返回fatal。远端evidence run目录仍不存在、数据/quality/GPU均未触达。恢复仅删除嵌入式subexpression，在单独
  只读命令已确认远端branch clean与目标路径不存在后，使用同一config和固定r1启动。防重复：PowerShell到SSH的双引号
  参数不嵌入`$()`或`$var`远端shell表达式；状态检查拆成独立命令，不把本地解析错误写成formal failure。证据=
  `run://worldsim_v64/WS-V64-P2E-FRESH-EVIDENCE-01/20260826T084000Z__fresh-evidence-s0-r1`、
  `https://learn.microsoft.com/en-us/powershell/module/microsoft.powershell.core/about/about_parsing`。

- `V64-F08`（`resource/protocol`, `resolved_by_native_voxel_recovery`）：继承的V6.3 surface compiler历史formal为
  `72 units / 47,568.47 s wall / 3,334.28 s max unit`。fresh surface r1运行约4分钟时仍为`0/72 units`、4 KiB，两个worker
  CPU各约100%且无资源异常；照旧执行预计浪费约13小时，并生成当前UQ不消费的signed-distance、patch、normal、actor与
  proposal registry。检索OCCUQ ICRA 2025的原生`200×200×16` voxel-level feature GMM，以及CuPy/cuCIM exact EDT后，
  选择前者：在UQ score读取前按精确PGID停止并保留partial，预注册唯一native-boundary-voxel r1；不安装CuPy、不继续旧
  full-stack。防重复：不得把partial写成算法失败，不得删除/覆盖r1；native r1后禁止回旧surface、换EDT/denominator或
  sweep救结果。证据=`docs/autoresearch/worldsim_v64/P4N_NATIVE_VOXEL_UQ_RECOVERY_FREEZE.md`、
  `https://github.com/ika-rwth-aachen/OCCUQ`、
  `https://docs.cupy.dev/en/latest/reference/generated/cupyx.scipy.ndimage.distance_transform_edt.html`。

- `V64-F09`（`algorithm/interface`, `resolved_pre_evaluation`）：native-voxel r1在四个fit scene采样后，冻结occupied-boundary
  denominator的预测FREE geometry组仅`43`点，小于4-component GMM固定最低`80`点，`NativeFeatureDensityUQ.fit`抛错；
  r1仅8 KiB resolved/status，无model、evaluation score或gate verdict。根因是把retrospective全surface上的双预测geometry
  条件化机械搬到几乎全occupied的region。OCCUQ官方`gmm_utils.py`按真实voxel类别收集feature，并在推理时对类密度做
  `logsumexp`边缘化。恢复因此固定为region整体一个boundary-global GMM-4；不是降低样本门或扫描组件数。防重复：不得复制
  43点、降80门、把evaluation并入fit或回到双组；v2只执行r2一次，其他输入/gate不变。证据=
  `docs/autoresearch/worldsim_v64/P4N_NATIVE_VOXEL_UQ_RECOVERY_FREEZE.md`、
  `https://raw.githubusercontent.com/ika-rwth-aachen/OCCUQ/main/tools/gmm_utils.py`。

- `V64-F10`（`algorithm/evaluation`, `active`）：native-voxel r2按冻结协议完成并通过相对门：pooled U2 AUROC=
  `0.518545`、较最佳U0增`0.083047`，scene support=`2/2`。但两个scene内U2 AUROC仅`0.498387/0.498295`，
  FPR@95TPR=`0.965465/0.960623`；scene-0359 AP低于prevalence，scene-0998的50% coverage risk高于prevalence。
  pooled改善可能部分来自scene-level prevalence/score shift，不能包装成可靠场景内ranking、authority或calibration。
  顶会迁移依据：OCCUQ将dense UQ supervision与feature GMM分工；ReliOcc采用plug-and-play hybrid voxel uncertainty；EvOcc
  用evidence supervision显式建模unobserved/contradicting evidence。恢复只允许先冻结一个用四fit scenes hidden-FREE标签训练的
  轻量risk head，再在相同两scene分母执行一次；禁止扫描GMM/PCA/seed/denominator/gate或读取更多split救结果。证据=
  `docs/autoresearch/worldsim_v64/P4N_FRESH_UQ_CLOSEOUT.md`、`https://github.com/ika-rwth-aachen/OCCUQ`、
  `https://doi.org/10.24963/ijcai.2025/220`、
  `https://openaccess.thecvf.com/content/CVPR2025/papers/Kalble_EvOcc_Accurate_Semantic_Occupancy_for_Automated_Driving_Using_Evidence_Theory_CVPR_2025_paper.pdf`。

- `V64-F11`（`algorithm/evaluation`, `active`）：P5固定监督risk head按预注册通过pooled和两scene AUROC门，pooled U3
  AUROC/AUPRC=`0.658118/0.148720`，scene AUROC=`0.640682/0.636266`。但FPR@95TPR仍为pooled`0.867738`、
  scene`0.859069/0.907021`；logistic输出也未在独立calibration set上校准。故本run只支持ranking，不支持低误报authority、
  calibrated probability、conditional coverage或safety。检索ICLR 2024 Conformal Risk Control后，合法恢复必须先冻结新的
  scene-disjoint calibration/confirmation cohort，以scene/unit为交换单元选择单调selective set，并在untouched confirmation
  一次验证；不得用已读scene-0359/0998选threshold、扫描risk/coverage或把voxel当独立样本制造虚假样本量。证据=
  `docs/autoresearch/worldsim_v64/P5_SUPERVISED_RISK_CLOSEOUT.md`、
  `https://proceedings.iclr.cc/paper_files/paper/2024/hash/f3549ef9b5ff520a7e41ff3cc306ab2b-Abstract-Conference.html`、
  `https://github.com/aangelopoulos/conformal-risk`。

- `V64-F12`（`resource/operations`, `resolved_by_pipeline`）：P6正式准备入口扫描共享盘官方tar超过一小时后，已完成
  `9/10`个shard、临时raw约`14 GiB`、盘余量约`45 GiB`，但旧入口必须等全部raw和24个processed scene完成才启动
  IR-WM，观测GPU=`0% / 1 MiB`。24-scene native按既有6-scene输出外推约`13.3 GiB`，再叠加processed和临时raw，
  证明此前`~21.6 GiB`整批持久化估算缺少足够余量。该事实是I/O/调度阻塞，不是模型或数据质量结论。检索NVIDIA
  DALI异步pipelined execution、bounded prefetch queue及WebDataset shard streaming后，恢复保留已完成shard工作，并在
  DriveStudio scene达到冻结的`1176 images + 196 lidar`后立即按scene送入IR-WM，最多两个GPU worker；先处理16-scene
  calibration，模型冻结后再处理锁定confirmation。禁止重复扫描已完成shard、启动无关GPU filler、降低quality边界、把
  confirmation提前读入校准或用多卡掩盖共享盘瓶颈。证据=
  `run://worldsim_v64/WS-V64-P6-CALIBRATION-SIDECAR-01/20260826T100000Z__calibration-prep-s0-r1`、
  `https://docs.nvidia.com/deeplearning/dali/archives/dali_190/user-guide/docs/advanced_topics_performance_tuning.html`、
  `https://github.com/webdataset/webdataset`。

- `V64-F13`（`data/interface`, `resolved_pre_quality_read`）：prep r1在全部tar扫描完成、首个scene-1045官方
  DriveStudio转换成功后，以`images=1206, lidar=201`对旧六场景硬编码的`1176/196`做比较并抛错。r1未读
  Occupancy/UQ/hidden-FREE或calibration/confirmation质量；临时raw和完整首场景均保留。根因是nuScenes scene记录有
  `nbr_samples`，而`interpolate_N=4`的官方DriveStudio时间表长度为`(nbr_samples-1)*5+1`；scene-1045的
  `nbr_samples=41`故应为201帧、六相机1206图，不是文件缺失或重复。恢复只从冻结metadata派生每scene期望数，并让新r2
  显式复用现有临时raw和完整scene；不删除额外合法帧、不重扫tar、不改变12 target、cohort、seed或backend。首场景已独立
  完成`12/12` native targets，证明201帧接口可供IR-WM消费，但不构成质量结论。证据=
  `run://worldsim_v64/WS-V64-P6-CALIBRATION-SIDECAR-01/20260826T100000Z__calibration-prep-s0-r1`、
  `run://worldsim_v64/WS-V64-P6-CALIBRATION-SIDECAR-01/20260826T111500Z__calibration-native-scene-1045-s0-r1`、
  `https://github.com/ziyc/drivestudio`、`https://www.nuscenes.org/nuscenes?frame=0&sceneId=scene-0011&view=regular`。

- `V64-F14`（`engineering/operations`, `resolved_pre_quality_read`）：Windows PowerShell中的两个长驻feed lane反复调用
  短`ssh` readiness/publish命令；远端命令已经结束，但client继承PTY stdin后未退出，导致下一scene不推进。一次lane恢复时
  scene-0810的原远端worker仍在运行，新wrapper只在发现run leaf已存在时抛`FileExistsError`，没有覆盖或重复GPU计算；
  原worker随后正常完成`12/12`。OpenSSH官方手册明确后台/编排调用用`-n`禁止读取stdin；所有短检查、publish和wrapper调用
  加`-n`后lane连续推进，双worker达到100% GPU。防重复：不得因client挂起杀未知远端进程或新建重复run；先查remote PID/
  summary，再恢复缺失scene。证据=`https://man.openbsd.org/ssh`及P6逐scene run leaves。

- `V64-F15`（`algorithm/evaluation`, `resolved_by_new_version`）：冻结U3在16个独立calibration scene的192个case上没有任何正coverage
  通过case risk合同。最低5% coverage已有`41/192` failure，empirical risk=`0.213542`、simultaneous UCB=
  `0.292860`；night/vulnerable-transit分别`16/48`与`13/48`，所以不是Bonferroni或Clopper-Pearson过严。10%到50%
  coverage的failure继续增至`54,62,74,80,93`。根因边界是PCA16线性risk ranking不能跨新night/rain/construction/
  vulnerable场景提供case-level hidden-FREE控制；P5两scene AUROC通过不再足以解锁calibration/authority。confirmation target仍
  未读。禁止降低epsilon=0.05、提高conflict threshold=0.05、删stratum、读confirmation选策略或添加<5%事后coverage。
  合法复开必须是新模型版本：16个已消费scene只作development training，当前8个untouched scene作独立calibration，并先
  metadata-only冻结新confirmation。迁移依据=`https://proceedings.mlr.press/v97/geifman19a`、
  `https://proceedings.neurips.cc/paper/2019/hash/0c4b1eeb45c90b52bfb9d07943d855ab-Abstract.html`、
  `https://openaccess.thecvf.com/content_iccv_2017/html/Lin_Focal_Loss_for_ICCV_2017_paper.html`；closeout=
  `docs/autoresearch/worldsim_v64/P6_CASE_CALIBRATION_CLOSEOUT.md`。
  迁移在读取原confirmation前已冻结：16个已读scene仅作development，原8个quality-unread scene转独立calibration；新
  confirmation按剩余metadata-only pool/seed1固定为`1023,1105,0903,0451,0981,0537,0789,0157`。模型固定为完整
  273D的`128/64` focal-loss MLP且不做超参扫描。此冻结没有读取新quality、没有产生新failure ID；详见
  `docs/autoresearch/worldsim_v64/P6R_SELECTIVE_MLP_FREEZE.md`。
  第一阶段正式训练已完成：`786054` points，loss=`0.0337864->0.0251443`，development AUROC=`0.8811503`
  （仅描述），GPU fit=`10.1545 s`。模型现已冻结且原8-scene calibration仍未读；这既不关闭V64-F15，也不产生新
  failure。下一判定只来自预注册的96-case独立校准。
  独立证据现已在模型冻结后一次完成`8 scenes/96 units`，source-role overlap与query均为0；尚未读取模型分数或选择
  coverage，故V64-F15状态不变且没有新增failure ID。
  P6R独立评分随后以0.05--0.40 coverage全部得到`0/96` failure和simultaneous UCB=`0.048647`，选择最大通过40%；
  50%为`3/96`、UCB=`0.103218`而正确拒绝。故失败以“新模型版本解决”关闭；原PCA16线性U3负结论不改写，且新
  confirmation仍未读。证据=`run://worldsim_v64/WS-V64-P6R-CALIBRATION-01/20260826T141500Z__case-calibration-s0-r1`。
  exact-once confirmation只在冻结40%上读分一次，得到`1/96` failure；四strata分别`0/24,1/24,0/24,0/24`，
  总体和分层gate均通过。V64-F15因此以独立校准加新确认的完整新版本证据收口，但不产生现实安全声明。

- `V64-F16`（`resource/operations`, `resolved_by_scene_ready_streaming_and_catalog_finalize`）：exact-once新8 scene在现有raw cache均无payload；需要约24.8k sensor member。
  前一P6整批扫描虽已学习43033个member->shard映射，但`scan_shards`写回时只保留当前batch，故unseen batch仍会触发10个
  `.tgz`全扫，预计重现>1h GPU空转屏障。WIDS明确把index用于稀疏random access，ratarmount为compressed tar持久化SQLite
  index；结合当前代码不新增依赖，迁移为superset member->shard catalog，并以scene raw-ready为边界并发DriveStudio preprocess
  和最多两个IR-WM consumer。本批不可避免的一次scan完成后目录可复用；禁止重新裁catalog、等待整批processed才启GPU或用
  多卡掩盖I/O。证据=`https://github.com/webdataset/wids`、`https://github.com/mxmlnkn/ratarmount`和
  `docs/autoresearch/worldsim_v64/P6R_CONFIRMATION_EXECUTION_FREEZE.md`。
  下游exact-once合同已在target read前固定为40% policy、overall最多4/96且每stratum最多1/24 loss；不以本I/O失败
  改变科学gate。
  scene-ready priority scheduling已完成全部`8 scenes/96 blind targets`：先完成单shard/相关shard组，DriveStudio与IR-WM按
  scene流水，最大worker显存`4.1314 GiB`；没有等待整批processed或启用多卡。故GPU-idle/全批屏障部分已恢复；superset
  catalog的剩余EOF写回和可重建临时raw删除仍由原prep controller收口，完成后再把V64-F16标为resolved。
  exact-once评分后剩余scanner恢复并正常EOF：prep=`20260826T143000Z__confirmation-prep-s0-r1`完成8 scene，superset
  catalog=`57338 entries / 6880063 bytes`，temporary raw由controller删除。至此资源failure正式关闭；总wall `5872.4206 s`
  保留为首次稀疏scan成本证据，不把它误写成GPU wall。

- `V64-F17`（`data/interface`, `resolved_pre_score`）：exact-once evidence r1完成33/96 units后，scene-1105 frame62
  在`load_frame_boxes`用直接dict索引触发`KeyError`。processed审计显示该scene缺0--9、56--64的frame_instances键；这些
  frame在instances_info中逐一为0 annotation，`missing_with_annotations=[]`，不是sensor缺失或hidden target异常。nuScenes
  官方devkit对non-keyframe box使用相邻sample annotation插值，没有annotation时返回空/当前集合；故common loader把缺键解释为
  empty actor list。r2以hardlink复用33个完整NPZ、只算剩63；NPZ未存储的三个summary字段显式null，不伪造。禁止重算33、
  改scene/policy/gate或用target score挑恢复。证据=`https://github.com/nutonomy/nuscenes-devkit/blob/master/python-sdk/nuscenes/nuscenes.py`
  和`docs/autoresearch/worldsim_v64/P6R_CONFIRMATION_EVIDENCE_RECOVERY_FREEZE.md`。
  r2 canonical=`20260826T152500Z__confirmation-evidence-s0-r2`按上述合同完成`96/96`，其中33 hardlink复用、63新算，
  query/role overlap均0，wall=`74.6360 s`；模型分数仍未读，故以pre-score状态关闭。
  随后的exact-once评分成功消费该证据一次且未触发第二次恢复，确认本interface failure没有改变冻结策略或coverage。

- `V64-F18`（`data/interface`, `resolved_pre_quality`）：P4C v1 metadata-only selection只检查nuScenes scene table、
  sample count与used-scene ledger，漏掉IR-WM train temporal pickle membership。7个scene完成blind native；`scene-0276`
  DriveStudio完成后在worker读取`payload["infos"][scene]`处`KeyError`，native output/target/model score均未读。官方BEVFormer/
  IR-WM使用生成的temporal train/val infos，不能用未分割scene table替代membership。恢复保留7个valid leaf并只替换无效scene：
  从commit `4813438`重建seed2 fallback；`scene-0572`是`flipped`误命中substring `ped`，首个token-valid且temporal-member
  的vulnerable候选为`scene-0813(631)`。不得重选其余7 scene、改policy/model/gate或读取quality挑替换。
  因v1 controller持有旧catalog snapshot，replacement写独立JSON并在两者结束后union，避免两个`os.replace` writer互相丢更新；
  证据=`https://github.com/APRIL-ZJU/IR-WM/blob/ir-wm/README.md`、`https://github.com/fundamentalvision/BEVFormer`和
  `docs/autoresearch/worldsim_v64/P4C_TEMPORAL_MEMBERSHIP_RECOVERY_FREEZE.md`。
  冻结replacement随后完成raw准备、DriveStudio与blind IR-WM native；corrected aggregate复用7个valid leaf并加入
  `scene-0813`，得到`8 scenes/96 targets/4423846027 bytes`，maximum worker peak=`4.1314 GiB`。全过程未读取confirmation
  target/quality/model score，未改变C0/M0、模型、gate或96-case denominator，故本interface failure在quality read前关闭。
  canonical=`run://worldsim_v64/WS-V64-P4C-CONDITIONAL-CONFIRMATION-SIDECAR-01/20260826T170000Z__native-aggregate-s0-r1`。
  corrected evidence随后一次完成`96/96 units`、query/source-role overlap=`0/0`且未触发同类membership或actor-frame错误；
  model score仍未读，没有新增failure ID。evidence=`run://worldsim_v64/WS-V64-P4C-CONDITIONAL-CONFIRMATION-EVIDENCE-01/20260826T171500Z__confirmation-evidence-s0-r1`。
  exact-once scorer随后只读冻结C0/M0一次，两臂均为`0/96` failure且M0 coverage uplift=`0.0750164`，三项gate全部通过；
  没有触发第二次恢复、mapping选择或新failure。本条保持`resolved_pre_quality`，并由fresh positive result确认恢复没有改变合同。
  下游P10M state-bake在该结果后冻结为target-free materialization：只读METHOD/native/model并让package-only consumer读取结果；
  未发现新blocker或新增failure ID，不回开本条。
  P10M formal随后一次完成96个package，M0比C0新增`74499`个emitted voxels且两项gate通过；state bake target read=false、
  runtime model/evidence access=false，没有新增failure ID。该结果不把voxel materialization外推为GS/sensor/collision authority。

- `V64-F19`（`integration/resource`, `resolved_by_sparse_gaussian_adapter`）：P10M fresh cohort的8个nuScenes scene均无同场景StreetGS/
  SceneIR checkpoint；旧V6 GS runtime只绑定其他scene，并以manifest SHA256/双bake bit-exact为入口，直接复用既不满足same-scene
  语义，也违反V6.4禁止新增hash/checksum/fingerprint的约束。不得把跨scene checkpoint硬接到fresh package、恢复旧hash治理，或把
  voxel package直接称为photorealistic GS。检索GaussianFormer的sparse semantic Gaussians、GaussianWorld的
  `{position,scale,rotation,semantic,feature}`表示与GaussianOcc的voxel-grid uniform scale/identity rotation后，冻结P10G最小迁移：
  每个M0 emitted voxel一个Gaussian，fixed `scale=0.256m/opacity=0.95/identity rotation`，GPU probabilistic BEV splat；只读P10M
  package，不读target/model/StreetGS。证据=`https://github.com/huang-yh/GaussianFormer`、
  `https://openaccess.thecvf.com/content/CVPR2025/papers/Zuo_GaussianWorld_Gaussian_World_Model_for_Streaming_3D_Occupancy_Prediction_CVPR_2025_paper.pdf`、
  `https://openaccess.thecvf.com/content/ICCV2025/papers/Gan_GaussianOcc_Fully_Self-supervised_and_Efficient_3D_Occupancy_Estimation_with_Gaussian_ICCV_2025_paper.pdf`。
  formal P10G一次完成`96/96` package，生成`534581`个M0 semantic Gaussians并在GPU BEV splat获得相对C0
  `+41016` support cells；target/model/StreetGS access均false，两项gate通过。因此“无法在no-hash/same-scene约束下进入任何
  Gaussian consumer”的integration blocker关闭；photorealistic StreetGS/sensor binding仍不在该恢复声明内。
  后续P10R直接冻结为logged future-lidar route corridor semantic consumer；只读P10G package与pose，不产生新failure ID，也不
  用route overlay冒充photorealistic或collision recovery。
  P10R formal在36/96 cases得到`+375` route support cells并通过冻结门，但C0/M0 binary intercept均为96/96、additional intercept
  cases=0；这是明确的metric saturation边界而非新implementation failure。不得把support gain包装成更多collision case被拦截。

- `V64-F20`（`evaluation/metric`, `resolved_by_route_local_cell_severity`）：P10R binary route intercept对C0/M0均为`96/96`，使case-level
  hit metric完全饱和；虽然M0在36 cases新增375 support cells，但不能回答这些新增state是否把hidden FREE写成OCCUPIED。不得事后缩短
  horizon、缩小corridor或提高density threshold制造未饱和case。Waymo Occupancy Flow以固定current-ego grid做cell-level occupancy
  metric，Implicit Occupancy Flow允许planner在连续时空点query，soft collision optimization使用连续势能而非binary hit；迁移为
  同一冻结2s/1.5m corridor上的route-local target hidden-FREE rate。policy/model/route均不改，只允许一次target audit；pooled M0
  conflict门保持原0.05，case failure只描述。证据=`https://github.com/waymo-research/waymo-open-dataset/blob/master/src/waymo_open_dataset/protos/occupancy_flow_metrics.proto`、
  `https://openaccess.thecvf.com/content/CVPR2023/html/Agro_Implicit_Occupancy_Flow_Fields_for_Perception_and_Prediction_in_Self-Driving_CVPR_2023_paper.html`、
  `https://openaccess.thecvf.com/content/CVPR2023W/E2EAD/papers/Kedia_Integrated_Perception_and_Planning_for_Autonomous_Vehicle_Navigation_An_Optimization-Based_CVPRW_2023_paper.pdf`。
  P10C一次读取冻结target后得到M0 route emitted=`10013`、conflict=`43`、pooled rate=`0.004294`，并相对C0新增563 state；
  cell-level severity因此成功打破binary metric saturation，本条关闭。但5/96局部case仍超0.05，转入V64-F21，不由pooled pass覆盖。

- `V64-F21`（`evaluation/tail-risk`, `closed_negative_tail_authority`）：P10C pooled M0 route hidden-FREE rate虽仅
  `0.004294`，仍有5/96 case局部超过0.05，最高`scene-0895/f152=0.106383`；其余包括`0876/f047=0.076923`、
  `0876/f182=0.069444`、`0454/f122=0.063492`、`0895/f137=0.057692`。这阻止把pooled severity提升为route/collision
  authority。不得在已读target上调policy、改route或挑scene。CVaR对最坏尾部而非均值进行风险汇总，且PAC-Bayesian CVaR工作
  明确区分empirical tail与generalization bound；故只冻结alpha0.10/worst10 empirical audit，M0门仍0.05，不做优化或population
  声明。证据=`https://proceedings.neurips.cc/paper/2015/hash/64223ccf70bbb65a3a4aceac37e21016-Abstract.html`、
  `https://proceedings.neurips.cc/paper/2020/hash/d02e9bdc27a894e882fa0c9055c99722-Abstract.html`。
  P10T rows-only formal得到C0/M0 worst10 empirical CVaR=`0.0504298/0.0517085`，M0比C0高`0.0012787`且超过0.05；
  verdict=`rejected_empirical_route_tail`。本次未重读target、未改policy或tail fraction。故current M0 route/collision tail authority
  正式关闭并锁定P11；不得用P4C pooled/fresh pass、P10G support或P10R exposure覆盖该负结论。合法恢复必须是新版本、独立
  calibration与新confirmation，不能回调本次frozen M0。
  新版本恢复已冻结为P10R2/M1：把已消费P6R confirmation降格为development/calibration cohort，保持每case总selected count与
  M0相同，仅把route corridor名义覆盖限制到独立C0=`0.40`，并按原冻结risk score把释放预算重分配到non-route；模型、M0
  stratum coverage、2s/1.5m route与worst10 tail均不变。M1只可在该consumed cohort形成candidate，不能关闭本条；若两项冻结门
  通过，仍需metadata-only冻结全新temporal-member confirmation并exact-once确认。不得扫route cap、改尾部比例、重训模型，或用
  M1 calibration结果改写current M0 negative closeout。
  P10R2 formal一次完成：总coverage delta=`0`，route selected `5912->3826`、hidden-FREE conflict `23->9`，route worst10
  empirical CVaR `0.0220499->0.0114783`，M1最大case rate=`0.0454545`，两项candidate gate通过。该结果来自已消费cohort，
  因此只支持进入fresh confirmation；V64-F21仍保持`closed_negative_tail_authority`且P11仍锁定。下一步不得复用该cohort作确认，
  也不得因margin较大再扩route cap；只允许按metadata冻结未读质量的新temporal-member cohort并exact-once检验固定M1。
  fresh confirmation现已在任何target/model-score read前按seed3冻结为`1020,1016,0596,0590,0006,0472,0070,0371`，
  8/8均为IR-WM train temporal member且>=40 samples。选择只用description/name/count/index与当前124-scene排除集合；固定
  M1、2s/1.5m route、worst10和两项gate均未变。该prereg不关闭V64-F21；只有新96-case exact-once结果才能决定M1是否获得
  bounded fresh empirical route-tail authority，且无论结果如何都不改写历史M0负结论。

- `V64-F22`（`resource/operations`, `resolved_by_io_reassignment`）：P4C科学链已完成并推送后，两套cleanup controller仍为
  不同required-member集合重复顺序扫描同一10个official tar，持续占用NVMe且不产生新科学证据；这会把P10R2 fresh
  confirmation的scene-ready GPU feed推迟到可选catalog union之后。再次确认P4C native aggregate、evidence与exact-once summary
  均完整后，终止两个scanner tree；不把未完成union写成成功，而是明确放弃optional catalog enrichment。仅删除预注册为official
  tar可恢复的`worldsim_v64_p4c_raw_batch`与`worldsim_v64_p4c_replacement_raw_batch`（约6.8GiB），保留全部processed、native、
  evidence、model、run artifacts及已有`57338-entry/6880063-byte`catalog。I/O随后只服务P10R2一套新扫描与scene-ready feeder；
  不新增hash/checksum/fingerprint，也不改变任何科学policy/gate/result。

- `V64-F23`（`resource/scheduling`, `recovery_frozen_pre_target`）：P10R2 10-shard scan完成并成功流水化`0590/0596/0070`
  三个native leaf后，`scene-1020(778)`已由第二producer写成canonical processed，但对应feeder线程仍排在另一长耗时
  preprocess mutex后，形成head-of-line blocking并让GPU空闲。不得增加无关GPU filler或重算有效leaf。NVIDIA DALI明确以
  asynchronous pipelined execution和分离CPU/GPU prefetch queues隐藏阶段时延；迁移为同一feeder prefix的可恢复调度：启动先复用
  `passed=true,target_count=12` leaf，canonical processed直接绕过preprocess lock进入GPU semaphore。只丢弃当前可从raw重建的
  staging partial；cohort/model/policy/targets/gates/canonical IDs完全不变，target与model score仍未读。证据=
  `https://docs.nvidia.com/deeplearning/dali/user-guide/docs/pipeline.html`。
  ready-first恢复最终复用6个complete leaf，并只对`0006/0371`启动两个GPU worker；8 leaf全部12/12通过，本条关闭为
  `resolved_by_ready_first_resume`。

- `V64-F24`（`resource/operations`, `resolved_by_producer_single_owner`）：全shard scan结束后，prep主循环与feeder各自成为
  DriveStudio producer，先对不同scene并行有利，但随后同时开始`scene-0371(288)`，若均完成会竞争同一canonical目录。
  在任何duplicate canonical write前终止较晚的prep producer/tree，保留feeder较早staging、已落盘catalog和全部complete outputs。
  feeder第一次恢复还遇到`scene-0006`仅有run目录无summary的中断partial；确认无进程占用且仅1个未完成文件后精确删除并从
  complete canonical processed重建。最终8/8 processed、8/8 native、96 targets全部通过。prep以新r2和
  `--reuse-temporary-raw`只读8个complete canonical scene，`0.8171s`写summary并删除raw，不重扫tar/重做preprocess。
  防重复：scene-ready阶段只有feeder拥有producer写权；prep在stream结束后只作reuse finalize。科学cohort/policy/target lock未变。
  后续fresh evidence一次完成96/96 units、0 reuse与0 source-role overlap，未复发temporal membership、producer或partial问题；
  V64-F23/F24保持关闭，不新增failure ID。target现已读，故此后只允许预注册exact-once scorer，不得再改M1或cohort。

- `V64-F25`（`evaluation/generalization`, `resolved_exact_empirical_cohort_relative_confirmation`）：P10R2 prereg的绝对M1 route CVaR门在fresh
  96 cases通过（`0.0403133<=0.05`），总coverage严格保持，故formal verdict按合同为supported。但calibration中的相对改善没有
  确认：fresh M0 CVaR=`0.0391815`，M1-M0=`+0.0011318`；M1 pointwise failures从1增至2、maximum从`0.06818`升至
  `0.08333`。同时M1 route selected/conflicts从`8117/54`降到`4971/20`，说明绝对冲突质量下降但case-rate尾部受更小分母与
  稀疏离散事件支配。不得用absolute gate pass声称相对改善，也不得在已读confirmation上调route cap、tail fraction或挑case。
  合法下一步必须先检索denominator-stable sparse risk / occupancy-flow severity方法，再冻结rows-only诊断或新版本；P11 comparative
  authority保持锁定。current M0 P10T负结论与M1 absolute fresh pass分别保留，互不覆盖。
  检索Waymo Occupancy Flow fixed ego-grid cell metrics、Occupancy Flow Fields与Implicit Occupancy Flow后，冻结P10R3为
  `conflict count / route-eligible voxel count`的fixed-opportunity rows-only诊断，在consumed calibration与fresh confirmation分别
  使用同一worst10。该post-hoc诊断无confirmatory gate，不可关闭本条；只用于判断selected-only可变分母是否解释方向反转，且
  不得借结果回调M1或解锁P11。证据=`https://github.com/waymo-research/waymo-open-dataset/blob/master/src/waymo_open_dataset/protos/occupancy_flow_metrics.proto`、
  `https://waymo.com/research/occupancy-flow-fields-for-motion-forecasting-in-autonomous-driving/`、
  `https://openaccess.thecvf.com/content/CVPR2023/papers/Agro_Implicit_Occupancy_Flow_Fields_for_Perception_and_Prediction_in_Self-Driving_CVPR_2023_paper.pdf`。
  P10R3 canonical rows-only结果在consumed calibration与fresh confirmation的固定分母worst10均为M1更低：
  `0.0132351->0.00455240`与`0.0216470->0.0149832`；pooled density也分别下降`0.00143870/0.00265563`。
  这使“selected-only可变分母导致方向反转”成为一致的描述性诊断，但P10R3是在读过fresh confirmation后冻结，不能作为独立
  confirmation；本条继续active，禁止据此回写P10R2 formal verdict或解锁P11。下一合法动作是先检索paired sparse-event
  confirmatory设计，再决定是否在从未读quality的test cohort冻结一次固定分母exact-once，不能复用已读cohort做显著性包装。
  检索NeurIPS 2021 `rliable`与ICLR 2024 Conformal Risk Control后，P10R4冻结一次untouched 96-case test：保留个体成对
  方向作描述，不做bootstrap/significance；CRC因目标是单调expected loss而不迁移为tail gate。三项confirmatory gate只包含
  coverage保持、fixed-denominator worst10不劣、pooled fixed density不劣。test未读前不设最小effect、不改M1；若失败则本条
  terminal rejected并保持P11锁定，若通过也只关闭exact empirical cohort层面的relative问题。输入I/O改为单遍metadata与
  raw-only producer/单feeder，避免GPU因重复8次`sample_data.json`扫描或duplicate preprocess owner空等。
  P10R4唯一untouched exact-once最终三门全过：coverage delta=`0`，fixed-opportunity worst10 M0/M1=
  `0.020725740/0.010821074`（delta=`-0.009904666`），pooled fixed density=`0.004944667/0.002001413`
  （delta=`-0.002943254`）；paired M1 lower/equal/higher=`18/78/0`。因此本条只在独立96-case exact empirical cohort的相对
  fixed-opportunity层面关闭，并解除由本条造成的P11 bounded-design锁。不得把它写成P10R2 selected-denominator formal重判，
  也不覆盖V64-F21对current M0的负结论；population、physical collision、planning、closed-loop与safety仍无authority。

- `V64-F26`（`io/execution`, `resolved_by_restricted_shards_and_dual_queue`）：P10R4首个raw-only入口发现`14437` required members均不在持久catalog，
  10个`.tgz`并发扫描约4分钟仅到`4--10%`，workers主要处于page wait且GPU尚无完整scene。CPython tarfile对gzip selected
  members仍需顺序流；ratarmount/rapidgzip可建seek-point index，但为一次性cohort新建10份index仍先消耗全量扫描。
  现有`71555`条semantic member→shard catalog显示七scene的capture prefix唯一落在05/06/08/10，`scene-0668`由相邻
  temporal range与已经原子落盘的exact-prefix files冻结到07。恢复只扫描`05,06,07,08,10`，保留all-shard尝试已完成文件，
  原workers停后只删除其`.partial.<pid>`；继续使用已运行的唯一feeder，不启动第二preprocess producer。若任一member找不到，
  restricted scan必须失败并回到未猜测的全量扫描，不得换scene或读quality。科学合同与test unread状态不变。
  restricted r1在进入scan前因既有resume目录仍执行`mkdir(exist_ok=false)`退出；该run不含新增archive读、preprocess、GPU或
  target read。恢复只把显式`--resume-raw-scan`的mkdir改为`exist_ok=true`，默认新run防覆盖不变；以新r2继续，不新增failure ID。
  r2使scene-0598 native以`45.4004s/4.1314GiB`完成，但单preprocess mutex下一scene转换超过2分钟，GPU再次出现供给缺口。
  按既有DALI分离CPU/GPU queue依据，停止feeder parent但让唯一in-flight scene-0462预处理完成，保留0598 native；同prefix
  feeder恢复为两个独立per-scene staging与`2 preprocess / 2 native` slots。不得对同scene启动第二owner；完整canonical/native
  必须reuse。科学合同与test unread不变，本恢复仍归V64-F26。
  canonical r2最终在`1807.8114s`扫描05/06/07/08/10并找齐`14437/14437`；per-shard命中
  `5401/1824/1818/1783/3611`也揭示capture prefix会跨archive boundary，但冻结五分片union完整。catalog增至
  `85992 entries`。raw完成时双队列已完成0598/0462 native且GPU峰值均`4.1314GiB`，故本条关闭；若后续native/evidence
  出现科学或独立工程故障应另记，不得重开全量tar scan。

- `V64-F27`（`io/execution`, `resolved_by_exact_stage_path_and_reuse`）：双preprocess独立target已完成scene-1084/1081，但DriveStudio实际将
  `..._processed_824`重写到`..._processed_10Hz_824/trainval/824`；feeder按常规append查找`..._824_10Hz`，因此在
  canonical install/native前抛出。824/821 stage分别有完整`1206/201`与`1176/196` images/lidar且无native partial。
  parent已停，唯一in-flight 424/522不终止、不重复；修复只镜像DriveStudio既有字符串重写。进程退出后四scene原子安装，
  night两scene用冻结underlying native command与原计划run dirs直接供GPU，patched feeder随后同prefix复用。若任一stage计数
  不完整则只重建该scene；不得删除完整stage、换scene或读test quality。
  恢复最终复用4个complete native leaf，并对其余4 scene完成同prefix native；最后两scene从stage ready到native启动仅等待
  `0.0646/0.0625s`。aggregate为`8 scenes / 96 targets / 4423846058 bytes / passed`，峰值worker显存`4.1314GiB`；
  test target/quality/model score仍未读。finalizer只登记8个complete canonical scene并删除可重建raw，故本条关闭；后续evidence或
  exact-once若失败必须按其实际阶段登记，不能重跑native或改cohort。

- `V64-F28`（`algorithm/evaluation`, `closed_negative_after_single_recovery`）：P11 bounded critic formal run按预注册只以selected-policy
  false-safe不劣与progress/stuck作gate，Real-only/naive/verified分别为`13/12/12` false-safe、progress均`1.0`、stuck均`0`，
  因此formal verdict合法为supported。但完整1248-action主指标揭示三臂unsafe recall仅`2.17%/0/1.09%`，false-safe=
  `180/184/182`；verified与naive的policy false-safe和reward完全相同，Brier/ECE反而更差。训练正例为`3/384`、
  `191/1152`、`96/768`，说明固定0.5输出在长尾与跨cohort下没有形成violation critic authority；这不是全刹车作弊，
  而是稀有unsafe识别塌缩。不得只引用三门PASS声称collision improvement，也不得在已读P10R4上调threshold、改lattice、
  重训或另跑同test。参考CVPR 2019 class-balanced loss、ICLR 2021 logit adjustment与Recovery RL后，唯一有界复开是保留
  已冻结模型，用从未生成action label的独立cohort解析选择一次unsafe-recall threshold，再在另一未读action-label cohort
  exact-once；若progress/stuck或recall失败则关闭P11，不训练大型NWM/RL。证据=
  `run://worldsim_v64/WS-V64-P11-BOUNDED-COLLISION-CRITIC-01/20260827T033000Z__bounded-collision-critic-s0-r1`。
  恢复已冻结为P11R：三critic不重训；P10R2 action labels只作独立calibration，每臂用unsafe score的20%分位解析选择
  target recall=0.80的单一threshold；threshold落盘后才允许生成P4C从未读取的action labels并exact-once。P10R4 labels、
  threshold grid、lattice/feature/model修改和第二evaluation均禁止。门只包含recall、policy false-safe不劣与progress/stuck。
  P11R最终threshold=`4.25e-18/0.191678/0.084891`；离散20%分位使naive/verified calibration recall均为`70/88=0.79545`，
  未静默改quantile或重跑。P4C evaluation中verified recall进一步降至`85/137=0.62044`；其policy false-safe/progress/stuck=
  `2/0.87240/0.11458`。Real-only threshold把全部action判unsafe，`96/96` fallback stop带来false-safe0但progress0/stuck1，说明
  recall与anti-trivial progress不能靠单一operating-point同时恢复。四门仅progress/stuck通过，P11R rejected，本条以negative
  terminal关闭P11；大型NWM/RL、再校准、换loss/model/lattice、第二evaluation均不解锁。后续只允许rows-only failure
  characterization和V6.4报告收口，不得创建新的P11科学attempt。
  P11D rows-only诊断进一步显示calibration→evaluation unsafe prior=`0.07051->0.10978`，verified unsafe q20/median score却下移
  `0.05328/0.13745`且safe median近乎不变；AP/AUROC=`0.24710/0.71165 -> 0.13740/0.56274`。这同时存在prior shift与
  unsafe ranking degradation，不支持“只换一个threshold即可恢复”的解释。该诊断无gate、无native/evidence reread，不改P11
  terminal；未来复开必须是新版本、新的可迁移violation representation与独立cohort，而不是本版本校准修补。

<a id="detail-v63"></a>

