# 历史原始记录 061

> 冻结历史，不是当前执行规则。当前规则见 [scaling law](../../../auto-research_scaling_law.md) 与 [AGENTS](../../../AGENTS.md)。
> 原文件：`docs/RESEARCH_FAILURES.md`；迁移前提交 `abbd431c`。原有相对链接按原目录解释，证据路径原样保留。
> 按索引读取相关小节；旧哈希、过度门控和资源指令均不重新生效。

## V4 B0 防重复结论（2026-08-11）

- `V4-F17`：历史 summary/metrics 记录 checkpoint 曾经存在，不等于当前 checkpoint 可执行。B0 首次磁盘审计中，
  scene-0230/0242/0255 的历史 StreetGS 文件均已不在路径；只能保留 bytes/hash provenance，不能把历史
  `exists=true` 或旧质量数值登记为当前 executable。正式 matrix 必须在每次 run 重新检查真实文件。
- `V4-F18`：AD-GS historical aggregate 的 6-scene metrics 只覆盖旧 cohort，且其历史 source/env/checkpoint 路径已缺失。
  与 V4 development 重叠的 0230/0242/0255 也只能记 `historical_metrics_only`；2026-08-12 新恢复的 exact official
  source/env 只解除执行前置，不使历史 metric executable。不得把旧三场景数值拼接新三场景、用 aggregate mean 代替
  scene rows，或把环境 smoke 写成 V4 same-split checkpoint。
- `V4-F19`：baseline inventory 的 `blocked` terminal 是当前资产前置条件未齐，不是 B0 task 的永久 blocked 或方法失败。
  B0 继续 `running`，通过新 run 补齐资产；旧 inventory 不覆盖。只有 V3.3/StreetGS/AD-GS 各 6/6 且统一 evaluator
  完整后才可收口，不能因 M1 实现更有趣而跳过 matched baseline。
- `V4-F20`：DriveStudio preprocess 会把输出根再追加 `_10Hz`，并按零填充 scene index 写目录。r4 的上游命令成功
  不等于 runner 目录合同成功；必须验证真实输出 root、scene dir、`1,176 RGB / 196 LiDAR` 后才能登记 done，不能移动
  或重命名一个未审计路径来掩盖合同错误。
- `V4-F21`：远端网络不可达不能用镜像、floating revision 或未校验模型绕过。sky-model r10 因 `Errno 101` 保留
  blocked；本机只从官方固定 revision URL 获取三文件，传输前后均按 bytes/SHA256 exact 校验，远端恢复经临时目录
  原子发布且 generation 保持 offline。任何 staging 漂移都必须 fail-closed。
- `V4-F22`：preprocess 预建的空 `sky_masks/` 与已有推理产物不是同一状态。r12 因旧 runner 只看目录存在而 blocked；
  修复只允许非 symlink 的空目录，发布前再次 `rmdir`，已有 mask、非目录或 partial 一律拒绝，不能覆盖正式产物。
- `V4-F23`：StreetGS 100-step profile 只证明训练链、checkpoint schema 和单卡资源门可执行。r16 不能计入 6-scene
  30k formal coverage，也不能读取/登记质量或据此宣称 baseline 已完成；每场 formal 必须新建不可变 run。
- `V4-F24`：六场景 StreetGS 的 Gaussian 数、wall 与 peak GPU 差异很大；scene-0255/0048 sampled peak 达
  `24,092/24,000 MiB`，scene-0994 final RigidNodes 仅 `1,029`。不能用一个 scene 的 profile 外推所有资源，不能因
  actor 稀疏补点或删 scene，也不能把无 OOM 的近上限运行倒写成资源失败。主表保留每场分母与工程行。
- `V4-F25`：StreetGS 原生 `test_image_stride=10` 不是冻结的 `sample_index mod 5` 三分区。r17/r20/r22/r24/r26/r28
  即使完成 30k、checkpoint finite 且未主动读 test quality，余数 4 的 heldout 输入仍可能进入训练，因此只能保留为
  protocol-mismatch provenance；r29 的 `StreetGS=6` 被 corrected inventory r33 明确推翻。不得用“训练成功”替代
  matched-contract 合规，也不得覆盖旧 run 来修正历史。
- `V4-F26`：不读取 test quality 还不够，训练进程也必须在 I/O 层隔离 development/heldout。AD-GS adapter 正式训练
  只物化 `train` 的 354 张图；兼容补丁增加 `--disable_test_evaluation`，避免上游在 final iteration 自动将 test
  iterations 加入评测。审计/统一 evaluator 可以显式物化三分区，但训练 runner 不得复用该全量目录。
- `V4-F27`：source checkout、权重和 Python 包必须分别固定 commit/bytes/SHA，环境恢复成功不等于 baseline scene
  executable。r34 从冻结本地环境离线复制，编译 `simple_knn` 与 `diff_gaussian_rasterization` 并通过真实 CUDA
  forward/backward smoke；在 strict preprocess + checkpoint 完成前，AD-GS coverage 仍为 `0/6`。
- `V4-F28`：传输/命令包装失败与模型失败必须分开。CoTracker/plyfile 的首次远端校验受 shell quoting 影响，DPT 下载
  曾出现两个进程指向同一 partial，发布后的附带 `stat` 也曾因 quoting 失败；这些尝试均未被登记为 canonical。
  只有停止冲突进程、临时路径原子发布并对最终 bytes/SHA 做独立复验后才可使用，且不得把 wrapper failure 写成
  权重、CUDA 或算法失败。
- `V4-F29`：preprocess 失败后的目标目录不是可静默复用的 canonical。r35 因启动包装器预建 run 目录而被不可变门拒绝；
  r36 因环境构建留下的未跟踪目录被 source-audit 拒绝；r37 完成 adapter/depth/segment 后因可选诊断依赖
  `flow_vis` 缺失而在 flow 启动时 blocked。三者均未训练或读取 dev/heldout；r37 partial 移入
  `work/codex-backups/2026-08-12-adgs-r37-partial-scene0230`，不覆盖、不伪装 resume。正式 flow 通过显式
  no-visualization 合同移除诊断视频依赖，CUDA extension 后续从 run-local source copy 构建，避免再次污染 official checkout。
- `V4-F30`：修掉 preprocess 可视化依赖不等于训练 import graph 已解除同名依赖。r39 尚未进入 iteration，
  `loss_utils -> flow_utils` 就因全局 `flow_vis` import blocked；正确修复是只在 TensorBoard flow 图真正调用时 lazy import，
  并把 `utils/flow_utils.py` 纳入 exact compatibility patch。不得安装非必要诊断包来掩盖正式无评测训练合同。
- `V4-F31`：Python import 成功不证明 CUDA extension 包含当前 GPU kernel。r41 能加载 PyTorch3D 0.7.5，
  但 inherited `_C.so` 在 RTX3090 KNN 首次执行时报 `no kernel image`；必须从 clean frozen source 在 run-local
  目录以 `TORCH_CUDA_ARCH_LIST=8.6` 重编，并在环境 smoke 中真实调用 `knn_points`。r42 完成该合同后 r43 才进入
  100/100 iteration；r39/r41 继续保留 blocked，不倒写为成功。
- `V4-F32`：进度条达到 60k 或路径上存在 checkpoint 不等于 scene executable。r44 只有在 formal step、run 内
  `point_cloud/deform/env` 三文件 bytes/SHA、fingerprint/manifest、source HEAD、六修改文件与兼容补丁全部精确后，
  才由 r45 从 `AD-GS 0/6` 更新为 `1/6`；StreetGS 同样从“存在即计数”收紧为 runtime+bytes+SHA 精确。
  `AD-GS-2026-07-27.patch` 是 zero-context patch，reverse-check 必须显式传 `--unidiff-zero`，否则会产生审计假阴性。
- `V4-F33`：单个方法达到 6/6 不等于 B0 完成。StreetGS r32/r46/r48/r50/r52/r54 已按 strict mod5、
  checkpoint bytes/SHA 与 clean r55 inventory 收口为 6/6，但 V3.3/AD-GS 仍各为 1/6，统一 evaluator 也尚未生成
  完整 scene rows；不得据此启动 M1、读取 test quality，或把 inventory 的 `matched_baseline_assets_incomplete`
  倒写成 StreetGS 失败。后续只补缺失方法/场景并保留旧 inventory。

## V4 P0 防重复结论（2026-08-11）

- `V4-F01`：计划草案记录的 HEAD 不是执行时事实。草案写 `main@144ed19`，P0 实查为 `main@2108430`，且 V3.3
  收口 `e6663e1` 已进入其历史。V4 必须从真实 `main` 建分支，不得回退旧 HEAD 或把草案 provenance 写成 canonical。
- `V4-F02`：计划写“`/root/autodl-pub/KITTI` 已在公共盘”不等于当前机器已挂载。P0 实查目录不存在，状态固定为
  `blocked_local_dataset_missing`。不得创建空目录、下载 KITTI、借其他 layout 冒充，或把外部缺盘写成 adapter/算法失败。
- `V4-F03`：H0 在计划表中写了 `conditional`，但同一计划规定任务状态只允许
  `pending/running/blocked/done/rejected`。V4 注册表将 H0 规范化为 `pending`，条件授权单独记录；不得引入第六种状态。
- `V4-F04`：一手论文、项目页或官方源码存在不等于 baseline 已在本机 single RTX 3090 + same split 执行。SplatAD、
  IDSplat、HorizonForge、RecEdit-Drive 等必须分开记录 paper/source/executable 状态；没有 matched run 不填数值。
- `V4-F05`：KITTI 缺失不阻塞 D0 nuScenes cohort，但会阻塞 single-card closure 的 KITTI adapter smoke。不得因此提前
  读取 nuScenes test、跳过跨数据集条件，或用多卡/新下载掩盖外部前提。
- `V4-F06`：V4 的公式必须同时对应 config、代码、ablation 和可计算指标。P0 只冻结 schema，不代表 M1/M2/M3
  已实现或有效；后续失败必须按预注册早停，不得继续堆 evidence feature、diffusion 或数学包装。
- `V4-F07`：formal run 通过不代表可跳过提交前 whitespace gate。P0 r1 使用的方法合同正确，但未跟踪计划的参考文献
  含 Markdown 行尾空格，`git diff --check` 拒绝提交；规范引用格式后 plan/config SHA 改变。r1 保留 noncanonical
  done，r2 对最终字节重新审计并成为 canonical；不得倒写或覆盖 r1。

<a id="detail-v3"></a>

## V3.3 R0 防重复结论（2026-08-11）

- `V33-F41`：R0 不能把 JSON 中“语义相近”的类型或枚举视为相同。diagnostic 前三次分别把 S2 的空列表写成
  数值 0、把 S3 `heldout` 写成 `heldout_confirmation`、把 S4 `real_renderer_evaluation` 写成
  `evaluation`，均 fail-closed。以后 verifier 必须比较原始类型/枚举；不得用字符串归一化掩盖 schema 漂移。
- `V33-F42`：正式 instance-field validator 通过不代表 NPZ 必须有未约定的 `schema_version`。r4 在报告层
  冗余读取该字段而 failed；修复只移除报告假设，仍执行完整 validator。以后不能把“自己希望存在的字段”变成
  canonical 资产失败，也不能因此跳过正式 schema 校验。
- `V33-F43`：RoadPatch 成为 V3.3 主方法不等于已在 matched 协议下胜过 V3.2 Telea。两者 base、空间语义与
  评测协议不同；R0 答案固定为 `not_directly_ranked`。不得用 B1 相对 B0 的 heldout gate 写成 Telea head-to-head，
  也不得因缺直接排名否定 RoadPatch 的 3D-native/provenance/heldout 成立结论。
- `V33-F44`：内容寻址 release 不允许为“完整”而复制 579 MB base checkpoint。R0 package 含 O1 sidecar、
  RoadPatch/A4/S4 delta、production renders 与 external reference，forbidden model suffix count=`0`；离线 verifier
  同时锁 file set/bytes/SHA。任何新增 `.pth/.pt/.ckpt/.safetensors` 或未登记文件都必须拒绝。
- `V33-F45`：deterministic archive 不能包含当前 run timestamp、绝对输出路径或可变 ZIP metadata。R0 release
  ledger 只引用固定 canonical 输入，ZIP entry 排序/1980 timestamp/permission/compression 固定；diagnostic、
  formal 和同 run replay SHA 均为 `cffaad16...44a7`。以后新增 release 字段必须先证明跨 run byte-exact。
- `V33-F46`：R0 的 `v33_supported` 只覆盖 scene-0230 主链、冻结确认视图和单 RTX 3090。它不证明
  scene-0242/0255 完整 V3.3 transfer、生成 actor GT、相邻视频时序、闭环安全或传感器真实性。F0 LiDAR-EVS
  仍是 conditional 新任务，不能倒写进 R0 的 4/4 success criteria。

## V3.3 S5 防重复结论（2026-08-11）

- `V33-F35`：unconstrained Harmonizer 不能因“只做视觉润色”进入删除生产链。canonical r4 的 edit target
  delete candidate 让冻结 SAM2 semantic mass/fraction 增加 `+0.126399/+0.133885`；production raw fallback
  两项均为 `0`。以后不得关闭 detector、改用候选图作 delete，或只展示另外四个未触发视图。
- `V33-F36`：跨视图平均改善不能掩盖单视图确认失败。五视图 contact 平均为改善，但 heldout f060/c1 的
  contact L1 delta=`+0.422686`，超过冻结逐视图上限 `+0.25`；因此 G1 必须 rejected、production=G0。
  不得改成 aggregate gate、放宽上限、删除该视图或用 edit target 的强改善抵消它。
- `V33-F37`：heldout confirmation 不是第二个开发集。S5 先只用 f091/c1、f005/c0、f065/c1 选择 G1，随后才
  读取 f020/c0、f060/c1；看到 F36 后不得调 contact/shadow 区域、权重、cap 或阈值并继续在同一 heldout 上
  宣称泛化。合法复开需要新假设、新 task 和未读 confirmation 数据；R0 只登记当前负结果。
- `V33-F38`：冻结推理环境不应为共享模块的无关顶层依赖而污染。r1 的 Harmonizer 已完成，SAM2 启动时因
  `semantic_gate.py` 顶层 SciPy import failed；SAM2 只消费 semantic mass/decision，并不构建 gate。修复是将
  SciPy 限在 builder 内 lazy import，不是往冻结 SAM2 环境临时安装包；r1 terminal 保持 failed。
- `V33-F39`：R3D2 official code、Apache license 与 clean commit 不等于存在作者 pretrained pipeline。
  canonical S5 只登记 `blocked_pretrained_model_unavailable`，`model_loaded=false/training=false`。不得拿
  SD-Turbo/TAESD base、Harmonizer 或自训权重冒充 R3D2-fast。
- `V33-F40`：S5 的五个冻结视图不是相邻视频帧，不能从 deterministic image SHA、跨 run exact 或五视图
  quality 推出 temporal consistency。canonical 明确 `not_evaluated_non_temporal_frozen_five_view_protocol`；
  时序 claim 需要独立连续视频协议、新指标与新 run。

## V3.3 S4 防重复结论（2026-08-11）

- `V33-F30`：S1 `hard_instance_id` 是候选身份集合，不等于所有 Background 候选都应被硬 ERASE。r2 将
  high actor 的 `36,736 Background + 4,525 Rigid` 全部设为零，虽然 target coverage=`0.999741`，但目标外
  L1=`0.821965>0.5`，方法按冻结门 rejected。不得通过放宽 L1 门、扩大 target mask 或只报告 coverage 复活该臂。
  合法修复是使用 S1 已训练 instance opacity 的 MAP 正类 `p>=0.5` 选择 Background，同时保留全部物理 Rigid core；
  r8 以 `1,614+4,525` 行将目标外 L1 降到 `0.225349`，门与视角未变。
- `V33-F31`：S2 canonical `roadpatch_delta.npz` 的 104 行同时服务 high/boundary 两个 actor，不是任一单 edit
  都应加载的整体。S4 high package 必须按冻结 `target_role=high_support` 只取 25 行，并保留 parent delta SHA；
  把 boundary 的 79 行混入 high edit 会产生无关场景变化，不能用“同属 Background repair”掩盖。
- `V33-F32`：恢复同一批 Python/Parameter 对象是必要条件，但不足以证明可回滚。renderer 可能持有缓存或顺序状态；
  每个 stack 卸载后必须重新执行 source render 并比较 tensor schema+bytes SHA。canonical r8 为 5 视角×4 stack=
  `20/20` exact，另有 full deterministic replay 和 replay rollback；以后不能只比较 checkpoint SHA 或 object id。
- `V33-F33`：immutable base + delta package 不能把 579 MB checkpoint 复制到 base 目录后仍称“小型 delta”。
  canonical package 的 base 只含 checkpoint/registry reference descriptor，完整 checkpoint copy=`0`，最大 payload
  `3,942,422 bytes`；任何 materialized deployment checkpoint 必须作为独立 deployment factor，不得改写 authoring state。
- `V33-F34`：S4 r3/r4 的方法和指标有效，但最终核心增加 fail-closed policy validator 后其 source snapshot 不再是
  提交态；r5/r6 与最终方法一致，但 builder 后续补齐异常 terminal。最终只认完整重跑 r7/r8 为 canonical；不得以
  “只是校验/失败处理代码”绕过源码 byte-exact 合同。

## V3.3 S3 防重复结论（2026-08-11）

- `V33-F22`：DriveStudio `instances_info.obj_to_world` 在冻结数据中可直接是 list，也可能是历史 JSON string。
  selector r0 只接受 string，在首个候选 fail-closed，未发布 selection。修复必须对两种 schema 都验证 4×4 shape
  与 finite，不得无校验 `json.loads` 或续写 r0；r1/r2 已以新 run 收口。
- `V33-F23`：view selector 的确定性不能用复制 manifest 冒充。high r1→r2、boundary r7→r8 都重新执行全部
  D2 original/delete 候选渲染，formal 传入 diagnostic 的 expected selection/input SHA 并得到 byte-exact；候选池
  必须继续排除 heldout 与 reserved development，不能让评估帧参与 view selection。
- `V33-F24`：更多输入视图不自动等于更好 actor。high development 中 A4 被选择是因为冻结 metric order 下
  IoU/boundary 改善且 LPIPS/PSNR/背景漂移/横移碎裂 retention 全过；heldout 只允许 A0 与 frozen A4 一次确认。
  不得从 heldout 重选 A1/A2/A4，或把生成背面写成 GT accuracy。
- `V33-F25`：V3.2 manual A0 只绑定 high-support token `af663...c5c29`，不是 boundary actor 基线。boundary
  评估必须使用同 identity 的 immutable D2 native actor，不能复用 high A0、换 actor 或仅凭 class 相同跳过三元核对。
- `V33-F26`：Asset Harvester 对 boundary A4 成功生成 PLY/NPZ 不等于 production override 可接受。r12 中 A4
  相对 native 的 IoU/boundary F1 从 `0.666562/0.555343` 降到 `0.624832/0.492141`，LPIPS/PSNR 也失败；
  决策为 `ABSTAIN_GENERATED_OVERRIDE`，未读 boundary heldout。不得通过放宽门、读取 heldout 或只展示 orbit
  render 复活该资产。
- `V33-F27`：CLI 编排提供错误 inference manifest SHA 时，importer 必须在物化前 fail-closed。boundary r10
  因传错 SHA 保留失败证据；正确 SHA 只用于新 r11。不得删除 r10 后续写、绕过 SHA，或把编排错误写成模型失败。
- `V33-F28`：canonical eval 的 source snapshot 必须等于最终提交源码。high r5/r6 的指标与 r13/r14 完全相同，
  但 evaluator 后来增加 native baseline 支持，旧快照不再等于提交态；因此只把 r13/r14 作为 canonical。不得用
  “逻辑没变”绕过 byte-exact 合同。
- `V33-F29`：scene-0242/0255 有旧 V3 checkpoint 不等于存在本任务冻结的 V3.3 S1/S2 mask/actor 输入链。
  在 boundary transfer 已拒绝的情况下更不能混用旧资产补齐跨场景表。合法复开需要新 task、每场景 exact identity/
  mask/checkpoint 协议、冻结 high actor policy 和新 run；旧 S3 terminal 不续写。

