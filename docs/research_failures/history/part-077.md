# 历史原始记录 077

> 冻结历史，不是当前执行规则。当前规则见 [scaling law](../../../auto-research_scaling_law.md) 与 [AGENTS](../../../AGENTS.md)。
> 原文件：`docs/RESEARCH_FAILURES.md`；迁移前提交 `abbd431c`。原有相对链接按原目录解释，证据路径原样保留。
> 按索引读取相关小节；旧哈希、过度门控和资源指令均不重新生效。

### V6-F84：retry 配置的 hypothesis_id 必须与追加的预注册记录一致

H-R82-002 run `20260821T215606Z__three-actor-package-s20260821-r1` 数值上通过全部 package gates：3个 actor、45个 payload、34,257 primitives、588 trajectory rows、195,658,443 bytes，三棵 actor package tree byte-exact且双 bake 完全一致。但 digest repair commit 只修正 source SHA256，遗漏把 YAML `hypothesis_id` 从已关闭的 `WS-V6-H-R82-001` 更新为 active 的 `WS-V6-H-R82-002`，因此 SUMMARY 错绑旧 hypothesis。该 run 不得作为 canonical acceptance。

不得回写 run、把数值通过覆盖 provenance mismatch 或重新编号旧记录。H-R82-003 只把 YAML hypothesis binding 更新为 `WS-V6-H-R82-003`，保持已验证的 R70/R80/R81 hashes、bake bytes、所有 denominators、资源与 claim boundary 不变；新 commit/push 后重跑。

### V6-F85：稀疏固定时点的 actor lifecycle 有效性不能替代前视相机可见性

H-R95-001 canonical run `20260821T234324Z__scene0048-actor-visibility-s20260821-r1` 在第二个独立 scene0048 matched-formal30k checkpoint 上完整枚举9个 RigidNodes actor、196帧 lifecycle 与15,717个 primitives；source、partition、checkpoint immutability、GPU 和全部 denominator gate 均通过。但在预注册的 frames `[0,49,98,147,195]` 前视相机中，所有9个候选的 actor-only effect pixels 均为0，最大值仍为0，低于非平凡可见门64，因此 H-R95-001 正式 `rejected`。这证明固定五时点 lifecycle-active 不能推出 camera-visible support，并不否定 scene0048 checkpoint 或 actor 表示。

不得删除64像素门、把非零 primitives/opacity 当作屏幕可见、改用事后选中的单帧，或追认 R95 为成功。H-R96-001 保持相同 checkpoint、9个候选、前视相机、opacity阈值0.01和选择规则，改为在单个冻结进程内穷举全部196帧的所有 lifecycle-active actor/frame 对；仅在完整分母上按最大 effect pixels、actor index、frame index确定性选择。若穷举仍为0，则保留第二次 rejection 并转向三相机覆盖实验，而不是继续猜前视帧。

### V6-F86：全时域 sensor conformance 不得要求生命周期外的 actor frame_valid 恒为真

H-R98-001 canonical run `20260822T001113Z__scene0048-selector-transfer-s20260821-r1` 完成196帧 logged/edited RGBD、784次冻结 DeepLab 推理与全部资源/immutability 分母。零校准 threshold45 在 scene0048 得到 TP=30、TN=166、FP=0、FN=0，precision/recall/F1=1、skip=84.69%，优于 fixed256 的 F1=0.9831 与原生36帧 lifecycle 的 F1=0.9091。但预注册 gate 把 `package_actor_frame_valid` 作为196帧均须为真的 sensor-conformance 合取项；actor8 的冻结 lifecycle 正确地仅在 frames160..195 为真、frames0..159 为假，因此唯一方法检查 `all196_compiled_native_sensor_conformant` 与总 `passed` 为假，run 按合同正式 `rejected`。逐项诊断确认196/196帧数值 conformance、repeat 与 native state restoration 全部通过，最大 RGB MAE `1.36e-8`、depth MAE `6.19e-7m`。

不得追认或回写 R98、不得把160个 inactive frame 改成 active、不得删除全196帧 sensor 数值门，也不得重跑昂贵推理来掩盖治理错误。H-R99-001 只把合同修正为“每帧 `package_actor_frame_valid` 必须与冻结 native lifecycle 精确相等”，从 R98 的内容寻址 sensor/perception artifacts 独立重算196帧 conformance、784输出重复性与 selector 指标；R98 保持 rejected，R99 作为新的 governance-repair authority。

### V6-F87：跨场景配置不得凭摘要转抄 checkpoint authority

H-R102-001 首次正式启动在创建 run directory、读取 sensor 或启动 GPU worker 前被冻结输入校验拒绝。配置误用了不存在的 scene0255 matched-baseline 路径 `20260812T132516Z__streetgs-scene0255-matched-formal30k-s0-r50`，并把 R101 摘要中截断/误记的 digest `dba249822f22317d926cc2953d0a433f6a95e6963d35e42750b8f7074dad6acd` 当作 authority；R101 实际已通过的冻结 checkpoint 是 `20260811T214009Z__streetgs-scene0255-matched-formal30k-s0-r48`，SHA256 为 `dba24982a3f25e162b5e293165258a588cf9bd7a49e54e05d0d052de703cb2d2`。本次没有 run、gate、sensor、perception 或方法结果。

不得跳过 `_verify`、制造不存在的 checkpoint、把 launcher 失败解释成 selector rejection，或继续从人工摘要抄 authority。H-R102-002 只从已接受 R101 配置复制确切 checkpoint 路径与 SHA256，并更新 hypothesis binding；R101/R90 artifact、actor34 edit、196/784 分母、threshold45、逐帧 lifecycle 合同、资源与 claim boundary 全部不变。必须在新 commit/push 后重试。

### V6-F88：零 AABB interaction 不意味着 RGB factorial interaction 必须非零

H-R111-001 canonical run `20260822T022113Z__scene0255-two-actor-factorial-s20260821-r1` 精确绑定 R102 actor34-only、R110 actor24-only 与 R109 joint 的 00/10/01/11 冻结 sensor/perception arrays。三个 source 的 196/784 分母、repeat、logged cell 与 hashes 全部一致；actor34/actor24 条件边际分别覆盖 19/161 帧，joint 与 single-target union 的帧级 F1=1，像素 F1=0.999086，single-selector OR 对 joint target 的 F1=1。但预注册错误要求至少 1 个 RGB pixel 的 `rgb11-rgb10-rgb01+rgb00` 绝对残差超过 1/255；实际 196 帧的最大残差、平均残差与超阈像素数全部严格为 0，因此唯一 `sensor_factorial_interaction_detected` gate 失败，run 正式 `rejected`。

不得删除该 gate、追认 R111 或把精确 superposition 描述成非线性 renderer evidence。H-R112-001 以 R111 rejection 为冻结诊断 authority，改测与观测一致的新机制：sensor 层必须逐值 exact affine superposition；下游冻结 DeepLab 允许有界的非线性 residual，但 joint/single-union 像素 F1 必须不低于 0.995、对称差比例不高于 0.005，两个 actor 条件边际与帧级/selector-OR exactness 仍须保留。semantic correctness、local causality、contact、dynamics、physics、planning 与 safety 继续 ABSTAIN。

### V6-F89：正式 runner 的模块归属必须与导入路径一致

H-R116-001 首次正式启动在创建 run directory、读取冻结 artifact 或启动 GPU worker 前，以 `ModuleNotFoundError: No module named 'motion_proj.worldsim_v6.r116_scene0255_fourth_actor_edit_compiler'` 退出。入口脚本正确从项目包 `motion_proj.worldsim_v6` 导入主体，但实现文件被错误提交到 `scripts/worldsim_v6`；因此本次没有 run、gate、sensor、proposal、GPU 结果或方法结论，H-R116-001 按实现合同记为 infrastructure rejection。

不得通过临时修改 `PYTHONPATH`、从未承诺的工作树文件导入、忽略失败启动或把它追认为 actor1 方法结果。H-R117-001 仅修复模块 ownership：主体放入 `motion_proj/worldsim_v6`，runner 仍位于 `scripts/worldsim_v6` 并从项目包导入；actor1、frame195、4,489 effect pixels、838 Gaussians、196-frame lifecycle、80 translations、所有 source hashes、阈值、资源上限和 claim boundary 保持不变。必须在新 commit/push 后从干净工作树重新正式运行。

### V6-F90：RGB-difference 邻域不能假定覆盖冻结感知的全局标签响应

H-R122-001 canonical run `20260822T034305Z__spatial-impact-locality-s20260821-r1` 精确绑定 R118/R121 两个四 actor 饱和方向的392帧 sensor 与 frozen-DeepLab arrays，所有 source hashes、逐文件 hashes、196/196 正帧和资源门均通过。但预注册把实际 `450x800` 图像误写为 `600x1200`，因此 shape gate 正式失败；更关键的是方法门也独立失败：RGB-diff mask 膨胀64px 后聚合标签召回仅 `0.604784`，最差帧仅 `0.027397`，不存在预注册网格内逐帧100%覆盖半径。R122 按合同正式 `rejected`，不得因 shape 书写错误而追认其空间局部性假设。

不得只修正 shape 后删除 exact per-frame coverage、把60.48%聚合召回解释为稀疏验证成功、依赖平均 ROI 掩盖最差帧，或宣称 crop inference 等价。非 canonical 恢复诊断显示392/392帧所需半径均大于128px、391/392帧大于256px，中位所需半径约412px、最大约684px；固定256px平均已覆盖73.56%画面却仍只有92.90%标签召回。H-R123-001 必须用正确450x800分母和扩展半径网格正式复算这些非局部性边界，接受的结论应是拒绝 RGB-diff 膨胀稀疏机制，而不是放宽成近似覆盖。semantic correctness、crop equivalence、speedup、physics、planning 与 safety 继续 ABSTAIN。

### V6-F91：跨实验 source digest 必须直接复制磁盘 SHA256，不能依赖人工转抄

H-R124-001 首次正式启动于 `2026-08-22T03:57:05Z` 前在创建 run directory、聚合任何向量或产生方法结果之前被 source verifier 拒绝。R109 `SELECTOR_TRANSFER.json` 的配置 digest 被人工转抄为 `0b972e5d0ff102c1eda06a2b077f769fe836f4b3d856242f4df58dbecafc6eafd91`，而冻结磁盘文件的实际 SHA256 是 `0b972e5d0ff102c1eda06a2b077f769fe836f4b3d856242f4df75406faeafd91`。本次没有 canonical run、gate、聚合向量、指标或科学结论，按 infrastructure/source-authority rejection 记录。

不得跳过 `_verify`、修改 R109 artifact、追认 H-R124-001，或调整 threshold45、11条件、2156帧、类别支持、分离间隔及资源门来掩盖该错误。H-R125-001 只把 R109 selector digest 改为磁盘实值并更新 task/hypothesis identity；其余 policy/source authorities、condition corpus、门限、预期方向、资源合同和 claim boundary 全部保持不变，必须在新 commit/push 后从干净工作树正式运行。

### V6-F92：语料内精确阈值不能未经新条件检验就提升为前瞻不变量

H-R128-001 canonical run `20260822T042521Z__scene0230-orthogonal-holdout-s20260821-r1` 在预注册后新生成 scene0230 actor12 `[0,0,+0.5]m` 的196帧 sensor 与784个冻结 DeepLab 输出；所有 source、proposal、0新增 overlap、38,541 primitives、196帧 lifecycle、compiled/native sensor、repeat、GPU、wall 与 abstention gate 均通过。但 threshold45 在78个正帧中漏掉 frame77：RGB changed pixels 为26而 frozen-label changed pixels 为5，得到 TP77、FN1、TN118、FP0、recall0.987179、F1=0.993548。run 按 zero-error 合同正式 `rejected`。

不得追认 R128、删除 frame77、把5个标签像素降为无关、放宽 F1/recall，或在同一 holdout 上改 threshold 后宣称前瞻成功。诊断显示全部118个负帧的最大 RGB feature 仍为0、78个正帧的最小值为26，使包含 R128 的开发并集精确阈值区间缩为 `[1,26]`。H-R129-001 必须把 R128 明确降格为 threshold-revision development evidence，与 R126 的2156行和 R127 的196行合并，按预注册 max-min margin 规则选择 threshold13并只声明开发集精确性；随后必须在另一个新条件上做独立前瞻检验。

### V6-F93：AD-GS 全时域实验必须区分 train、development 与锁定的 heldout camera 分区

H-R134-001 首次正式启动 `20260822T053238Z__adgs-cross-frontend-threshold13-s20260821-r1` 在任何 sensor 或 perception 输出产生前，于请求 frame0 时退出。冻结 R3 adapter 的196时间轴只物化了118个 train、39个 development 时间步并刻意排除39个 heldout 时间步；worker 仅从 `getTestCameras()` 建表，因此只看见 development，frame0 不存在。失败目录仅4 KiB、sensor 文件数为0，不构成 threshold13 或 cross-frontend 方法结论。

不得把 heldout 图像补入 adapter、把缺帧静默删除后仍声称196分母、追认 H-R134-001，或把本次启动失败解释为 AD-GS transfer rejection。H-R134-002 只能合并 `getTrainCameras()` 与 `getTestCameras()`，从冻结 `partition.json` 预先导出 camera0 的精确118+39=157帧，并保持39个 heldout 未读；AD-GS checkpoint/edit、threshold13、正负支持、0 FP/FN、skip、资源门与所有 abstention 不变，新 commit/push 后重新正式运行。

### V6-F94：StreetGS 上冻结的单一 RGB 像素阈值不能直接宣称跨 frontend 不变

H-R134-002 canonical run `20260822T053744Z__adgs-cross-frontend-threshold13-s20260821-r2` 在 heldout 未读的前提下完成 AD-GS scene0048 的118 train+39 development 帧、157组 logged/edited sensor 与628个重复精确 DeepLab 输出；checkpoint、adapter、aggregate actor state restoration、分区、GPU、wall 与所有 abstention gate 均通过。冻结 StreetGS threshold13 在131个正帧、26个负帧上得到 TP130、FN1、TN26、FP0、recall0.992366、F1=0.996169，唯一漏检为 train frame13：RGB changed pixels=1、label changed pixels=1；run 按0-error合同正式 `rejected`。

不得追认 R134、删除 frame13、把1个标签像素降为无关、放宽 recall/F1，或把 AD-GS 数据用于回改 StreetGS threshold13 后仍称全局策略。诊断显示26个 AD-GS 负帧的最大 feature 为0、131个正帧的最小 feature 为1，开发区间是脆弱的单点 `[1,1]`。H-R135-001 只能显式声明 frontend-conditioned router：StreetGS 保持13，AD-GS 用 R134 开发集按预注册规则拟合为1；R134 保持 rejected，且 AD-GS 的39个 heldout 时间步必须在 policy freeze 后一次性验证。
### V6-F95: AD-GS threshold-1 exact classification does not survive the sole heldout confirmation

H-R136-001 canonical run `20260822T055538Z__adgs-heldout-confirmation-s20260821-r1` consumed the one allowed confirmation attempt before reading heldout quality. All 39 camera-0 heldout frames, 78 AD-GS renders, and 156 frozen DeepLab outputs completed within contract. Source immutability, adapter partitioning, repeat exactness, actor-state restoration, positive/negative support, skip, GPU, wall, output budget, and all abstention gates passed. The frozen R135 AD-GS threshold 1 nevertheless produced TP=31, TN=7, FP=1, FN=0: frame 14 changed 11 RGB pixels but changed 0 label pixels. Precision was 0.96875, recall 1.0, F1 0.984127, and the run was correctly rejected.

Do not retune a scalar threshold on these heldout rows, rerun the consumed candidate, delete frame 14, or reinterpret conservative over-execution as exact classification. The next hypothesis changes method family and objective: an exact-input identity guard may reuse cached perception only for byte-identical RGB inputs and must execute otherwise. R137 evaluates that one-sided operational contract on R134 development data plus the already frozen R133 StreetGS execution authority; R136 remains rejected.
### V6-F96: A negative comma-separated translation must not be passed as a detached argparse value

H-R138-001 canonical failed run `20260822T061548Z__adgs-antithetic-exact-input-s20260821-r1` created the exact-once attempt and completed train+heldout adapter materialization. The sensor subprocess then exited in argument parsing before frame 0: the detached token `-0.5,0.0,0.0` was treated as an option, so `--translation-world` reported that its argument was missing. No sensor array or perception output was produced, and the exact-input method was not measured. The attempt is nevertheless consumed under the preregistered any-outcome rule.

Do not rerun the same antithetic condition, claim a method rejection, or edit its run into a success. R139 uses a distinct world-z +0.5m condition and binds the vector as `--translation-world=<csv>`, while preserving the exact-once, 39-frame heldout, full/reference selective execution, identity-only reuse, reconstruction, support, resource, and abstention gates.
### V6-F97: Python booleans in formal JSON closeout code must use `False`, not JSON `false`

H-R140-001 failed run `20260822T063253Z__end-to-end-utility-s20260821-r1` verified all immutable inputs and wrote the end-to-end certificate, gate, and summary, but exited before RESOURCE_AUDIT, MANIFEST, and TERMINAL. The resource dictionary used the undefined Python name `false` for `gpu_used`, raising `NameError`. No GPU, training, confirmation read, or source mutation occurred. Although the partial gate was written, it is not a canonical acceptance because terminal closeout is incomplete.

Do not hand-create missing success artifacts or promote the partial gate. H-R140-002 changes only `false` to `False` and updates the YAML hypothesis binding; all sources, formulas, three conditions, zero-error authorities, end-to-end thresholds, resources, and claim boundaries remain unchanged for a new clean-commit run.
### V6-F98: A literal recovery must search the whole Python module, not only the first failing line

H-R140-002 failed run `20260822T063601Z__end-to-end-utility-s20260821-r1` reproduced the same certificate and gate as H001, then failed on the immediately following `training_started: false` field. The first recovery changed only `gpu_used`, leaving two lowercase JSON booleans in Python source. Again, no GPU, training, confirmation read, or source mutation occurred, and the partial gate is not canonical.

Do not promote either partial run or continue one-line-at-a-time repair. H-R140-003 is preregistered after an exhaustive `true|false|null` token search. It changes the exactly two remaining resource-audit values (`training_started`, `confirmation_content_read`) to Python `False` and updates the hypothesis binding; all scientific inputs, formulas, thresholds, denominators, budgets, and claim boundaries remain fixed.

### V61-F02：下游 runner 必须读取上游 gate 的真实嵌套 authority

H-ME1-001 正式入口完成所有冻结文件 hash 校验后，把 `ME0_GATE.json` 的通过位误读为顶层 `passed`，而
`worldsim_v61.me0_gate.v1` 的真实 authority 是 `checks.passed`。因此触发 `KeyError`；异常发生在 run directory
创建、O_method/O_eval tensor 读取、GPU ray compiler、proposal 编译和任何方法计算之前。canonical run=`null`，
不存在 oracle upper-bound 科学结果，不能把本次记成 method rejected。

不得跳过 ME-0 authority、修改 canonical ME-0 artifact、放宽 ME-1 gate，或把 launcher failure 追认为科学 attempt。
H-ME1-002 只把读取路径修正为 `document["checks"]["passed"]` 并增加嵌套 schema 回归；28-case、五臂、source
hashes、0.2m voxel/0.1m ray step、50% coverage、20% depth consistency、false-safe/stop rule 与资源预算全部不变。

### V61-F03：合法 actor ID 0 不得与 raster 的空身份 sentinel 共用

ME-2 actor control 准备审计发现，ME-0 的 scene-0048 sparse identity layer 合法包含 actor ID `0`，但 ME-1
相机 raster 用零初始化 `actor_grid`，导致 actor0 与“该 voxel 无 actor”无法区分。ME-1 primary O2 的10个 ACCEPT
全部来自 scene-0242，scene-0048 两个 actor case 均已由冻结 P1 REJECT，因此 O2=`10/28`、false-safe=`0`、
mask yield 与 primary gate 不受影响；但 canonical ME-1 的 O3 scene-0048 identity/swept 诊断不能提升为完整 actor 结论。

不得把 actor0 改号、删除 scene-0048、追认 O3 actor safety，或为此重跑 ME-1 主臂。后续实现把 empty sentinel 改为
`-1` 并增加 actor0 回归；ME-2/ME-4 必须消费修复后的 identity raster，已落盘 ME-1 run 保持不可变。

### V61-F04：冻结 source digest 必须先满足 SHA-256 的 64 字符结构合同

P4 第一次正式入口在创建 run directory、载入模型或占用 GPU 前，被 VAE source gate 拦截。实际固定 revision
`70e803bfb4e127d534049d8ab8c8cb511780d485` 的 VAE 文件为 `1311145138` bytes，实际 SHA-256 与服务器
`X-Linked-ETag` 均为 `379995ca170d8a899019125f389ba8692b2e35625ff64ddc3fdaa8c9302ac340`；预注册配置在
末尾误多录一个 `2`，形成 65 字符值。模型字节没有漂移，canonical run=`null`，不存在 capability 科学结果。

不得跳过 source gate、改写模型文件或重复下载。修复只删除多录字符，并新增所有 model digest 必须是 64 位小写
十六进制的回归测试；官方 commit、model/DINO revision、demo、seed、50 steps、512 octree、资源门与 stop rule 全部不变。

### V61-F05：离线 Hugging Face repo-id 解析必须有显式 revision ref

P4 第二次正式入口通过全部 source gate 并创建 failed run
`20260822T111747Z__voxel-smoke-s1234-r1`；Omni DiT 与 VAE 均以 0 missing/0 unexpected 成功载入，随后
`Dinov2Model.from_pretrained("facebook/dinov2-large")` 在离线模式失败。固定 snapshot 与三个文件已完整存在，但按
exact commit 下载不会自动创建默认 `refs/main`；官方 encoder 只传 repo-id、未传 revision，因而无法把默认 main
解析到已缓存 snapshot。该 run 没有生成 mesh/points 或 capability gate，不是模型能力 rejection。

不得开启正式 run 网络、重复下载 DINO、改官方 encoder 或更换 backbone。修复只按 Hugging Face 标准 cache schema
创建 `refs/main`，内容精确绑定已冻结 commit `47b73eefe95e8d44ec3623f8890bd894b6ea2d6c`；runner 在模型载入前
验证 ref、snapshot、config 与 model SHA，之后仍保持 `HF_HUB_OFFLINE=1` 和 `TRANSFORMERS_OFFLINE=1`。

### V61-F06：Hugging Face cache ref 是无换行 commit token，不是普通文本行

P4 第三次正式入口创建 failed run `20260822T112159Z__voxel-smoke-s1234-r1`，再次在相同 DINO 离线解析点失败。
运行时常量确认 `huggingface_hub.HF_HUB_CACHE` 与 `transformers.TRANSFORMERS_CACHE` 都精确指向预期 cache root，
排除了环境变量和根路径猜测。直接审计已安装 `huggingface_hub.file_download.try_to_load_from_cache` 发现，它对
`refs/main` 使用原样 `f.read()`，不执行 `strip()`；由普通文本 staging 上传的 ref 是 41 bytes，尾部 `0a` 使
revision token 与 40 字符 snapshot 目录不相等。该 run 仍未生成 mesh/points 或 capability 结果。

不得继续猜 cache roots 或重复完整 P4。修复把精确目标 ref 机械规范化为 40 bytes，并让 runner 也要求 byte-exact
40 字符内容；先单独执行一次 repo-id 的离线 DINO load smoke，只有它通过后才允许下一次正式 P4。模型与参数不变。
修复后孤立 smoke 由 repo-id 离线载入 `Dinov2Model` 的 `304368640` 个参数，故解析卡点已关闭。

### V61-F07：共享 shape 环境必须显式包含 image-only backend 的官方导入依赖

H-ME2-001 failed run `20260822T120008Z__hy3d-actor-s1234-r1` 已通过全部冻结 source gate 并构造4个 actor
inputs，随后 A0 worker 导入官方 Hunyuan3D-2.1 package 时失败。该 package 的 `__init__.py` 无条件导入
`postprocessors.py`，后者依赖官方 `requirements.txt` 精确固定的 `pymeshlab==2022.2.post3`；既有 Omni
shape-inference 环境没有这个 image-only backend 依赖。失败发生在模型载入、GPU inference、asset 生成和 method
decision 之前；canonical run=`null`，不是 A0 或 A3 能力结论。

不得跳过 A0、patch 官方 `__init__.py`、安装无关 texture/UI 全依赖、改变四臂或把本次追认为方法 rejected。
H-ME2-002 只安装官方固定的 `pymeshlab==2022.2.post3`，在配置/runner 增加 exact version gate，并执行一次
离线 base pipeline import smoke；该 smoke 已成功导入 `Hunyuan3DDiTFlowMatchingPipeline`。模型、权重、
4 units/6 cases、seed、batch、steps、octree、compiler、truth separation、thresholds、资源和 stop rule 全部不变。

### V61-F08：Omni diffusion 支持 batch>1，不代表默认 marching-cubes extractor 也支持

H-ME2-002 failed run `20260822T120519Z__hy3d-actor-s1234-r1` 完成4个有效 A0 mesh，并成功载入 Omni、完成首个
A1 的2-sample diffusion 与 VAE implicit query。runner 随后发现输出 mesh 数为1而输入数为2并 fail-closed。
官方源码显示 `extract_geometry_vanilla` 虽把 logits reshape 为 `(batch_size,X,Y,Z)`，但 marching cubes 固定读取
`grid_logits[0]`，wrapper 也固定返回一元素 list；因此第二份 latent 没有 mesh，不是随机空输出或 OOM。
本次没有 A1/A2/A3 asset、method decision 或科学结论；canonical run=`null`。

不得静默丢弃第二样本、把6-case缩为3-case、patch官方源码、把全部 diffusion 降为batch1后冒充并行，或改变生成
参数。H-ME2-003 保持昂贵 diffusion batch2与逐样本generator，令官方 pipeline 返回两份 latent，再逐份调用同一
官方 VAE 的 batch1 decode/export。H002 已完成的4个 A0 只在 plan、input hashes、report 与每个 asset hash 全部
精确后复用，不重复GPU计算。模型、controls、seed、steps、octree、guidance、compiler、truth、threshold与stop rule不变。

#### V6-F97/V6-F98 recovery 收口

H-R140-003 从干净且已推送的源提交 `a13759ba8db03e1f740ad93e246ca24f0ff2d7fa` 完成 canonical run `20260822T063937Z__end-to-end-utility-s20260821-r1`。Scientific certificate SHA256 为 `913833af47e4171e27707f71418b6625ed358b538d1c8a5a18bca5ac7f585363`，与两次 partial computation 完全一致；完整 gate、summary、manifest、resource audit 与 terminal 的 SHA256 依次为 `ac3c79c0e93f2932a076da8323b89a210ff2cbaac27ffa13079ce89ae9d07b51`、`50900ff99736055a10c32f4362176b7fc87862ae84667591077d6c17024e635b`、`1cc753b3c0a9489ced2a58b23035466ee26cba963d2abc56c84ebd4d057e5a62`、`06c110236591529d5fef5f4178bfed696b6c0ad0cfbce94c497896dc92230265` 与 `be263ba010cdb936fbd01dbfa0fe294b8022101aae348c95030d2a42d45fdb77`。

该 recovery 不删除或重分类 V6-F97/V6-F98：两个失败目录继续保持不可变，只有 H-R140-003 是 canonical。完整 account 报告 StreetGS、AD-GS development 与 AD-GS exact-once confirmation 的端到端 reduction 分别为 13.5337%、11.1434% 与 1.66365%（macro 8.78024%、worst 1.66365%），reconstruction error 为 0。Selector 研究族在此次 recovery 后冻结；R141 未执行，本收口不授权继续 threshold、actor 或方向实验。

