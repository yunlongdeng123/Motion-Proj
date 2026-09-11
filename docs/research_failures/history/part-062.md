# 历史原始记录 062

> 冻结历史，不是当前执行规则。当前规则见 [scaling law](../../../auto-research_scaling_law.md) 与 [AGENTS](../../../AGENTS.md)。
> 原文件：`docs/RESEARCH_FAILURES.md`；迁移前提交 `abbd431c`。原有相对链接按原目录解释，证据路径原样保留。
> 按索引读取相关小节；旧哈希、过度门控和资源指令均不重新生效。

## V3.3 S2 防重复结论（2026-08-11）

- `V33-F15`：正式 run 目录不能在 runner 注册前由 `nohup ... > run.log` 预创建。r0 因 shell 先创建
  `run.log` 而触发 non-empty run-directory fail-closed；不得删除日志后续写。正式托管应把 launcher 日志写到
  run 目录外，或由 runner 创建目录后再写；r10/r11 使用新 run 收口。
- `V33-F16`：V3.1 P3 package 的空间网格是 `(x,y)` 且绑定 V3.2 P2 FP16 mixed checkpoint，不是当前 D2 FP32
  原生 Background 的道路索引。DriveStudio 首个 CAM_FRONT 是 OpenCV `x-right/y-down/z-forward`，道路 BEV
  必须使用 `(x,z)`；r1 因错误要求 P3 manifest 绑定 D2 exact SHA 而失败。不得为复用旧索引而放宽 checkpoint hash。
- `V33-F17`：相机内参在当前 DriveStudio 输入中是 9 个值（`fx/fy/cx/cy + 5 distortion`），不是只含 4 个值；
  r2 在正式物化前 fail closed。所有 adapter 必须显式接受已冻结 schema、校验前四项和 distortion 长度，不能静默切片
  后假称已适配其他相机模型。
- `V33-F18`：对整格直接取 `max_scale/max_plane_residual` 会被一个天空、立面或跨层 Gaussian 污染。r3 得到
  `53,541` patches 但 `0` valid；这不证明场景没有道路 donor。修复应先逐行排除 actor/generated/低 support/
  scale outlier，再确定性选择 `<=0.75 m` 的 densest vertical slab，最后做 plane/normal 门。r10 由此得到
  `822` valid patches；不得回到 whole-cell 放宽阈值。
- `V33-F19`：cross-view sidecar 的 `visible_view_count` 与 front-camera frustum observation 是两个合同。
  r4/r5 已有 valid 4 m patch，但把 `minimum_multi_camera_count=2` 当作当前六相机逐相机观测计数，导致两个真实
  target 的 top-5 手工门失败；当前实现明确要求 sidecar `visible_view_count>=5` 且至少一个 front-camera frustum
  observation，不虚构不可得的逐相机 visibility。
- `V33-F20`：donor 几何合格不等于新增 Gaussian 数量可以无限。r8 的 2,150-row dense delta 在 development
  selection 中可见，但 heldout PSNR/SSIM 退化 `-0.8553 dB/-0.00619`，保持 rejected。修复不是从 heldout 选
  top-K，而是在候选资格阶段冻结 `maximum_rows_per_target=512`，让搜索选择最小可见 delta；r11 最终为
  `25+79=104` rows，并通过全部 heldout 门。不得复活 r8 或事后改写其 terminal。
- `V33-F21`：官方 Inpaint360GS source clean、Apache-2.0 不等于当前 StreetGS/3090 上已复现。官方声明
  RTX 4090/CUDA 11.8，并需要主环境、独立 LaMa 环境、CropFormer/Big-LaMa/SAM/DeAOT/GroundingDINO 权重；
  官方代码没有 DriveStudio/StreetGS checkpoint adapter。r12 因这些前置条件 fail-closed 为
  `blocked_single_3090`、`official_execution_attempted=false`。该状态不是 B2 质量负结论，也不得用 Telea、base
  SAM 或自写 RoadPatch 输出冒充官方 Inpaint360GS。

## V3.3 S1 防重复结论（2026-08-11）

- `V33-F07`：磁盘可用空间在 P0 后被外部流程扩大，同时 V3.2 SAM2 checkout/weight/runtime 被删除；不能把
  canonical train masks 仍存在误判为 heldout 推理环境仍可用。普通 clone 又被大型 demo checkout 拖住，已终止本任务
  PID 并保留 `sam2.incomplete-20260811T0138`。恢复只能 sparse checkout exact commit、下载 exact weight，另建
  隔离环境并冻结 package list；不得修改 DriveStudio 环境或复用不完整 checkout。
- `V33-F08`：heldout-target r2 在旧 `/root/autodl-tmp/envs/worldsim-v32-sam/bin/python` 不存在时 exit=`127`；
  prompts 虽生成但 run terminal=`failed`，不得续写。新环境必须复原 V3.2 记录的 Python/torch/torchvision 版本并
  新建 run；r4 已按此收口。
- `V33-F09`：SAM2 singleton predictor 在当前 exact runtime 返回 `[object,1,H,W]`，而旧兼容路径也可能给
  `[object,H,W]`。r3 无条件 `unsqueeze` 产生 5D interpolation 错误；修复必须显式接受 rank 3/4、拒绝其他 rank，
  并新建 r4。r3 未发布正式 mask，不得当质量证据。
- `V33-F10`：更宽的 O3 ambiguous reassignment 并不自动改善边界。100-step development smoke 中 O3 的
  boundary F1/IoU=`0.123499/0.160504`，低于 O1 的 `0.149382/0.181752`，且 FP 更高；O3 已排除。除非提出
  新的几何邻域/身份证据并使用新 task/run，否则不得因“候选更多”重开。
- `V33-F11`：O1 在 heldout 显著改善 boundary/IoU/NBD/FP，但 FN mass 从 `0.061278` 增到 `0.109356`，
  identity presence 仍为 `0.972973`。因此只能声明对象边界与 false-positive 抑制突破，不能声明全面支配或完整
  召回；S2 delete mask 必须继续报告 FN/残留语义，不能用 O1 的高 precision 掩盖漏删。
- `V33-F12`：`np.savez_compressed` 默认把当前时间写入 ZIP entry header；即使 r6/r7 的 O0 全部数组 exact，文件
  SHA 也会漂移。r6 因此保留为 done noncanonical。r7 writer 固定 entry 排序、1980 ZIP timestamp、权限与压缩
  参数，并用同一 field 二次写入 byte-exact 测试锁定容器确定性。该合同不等于宣称 CUDA 训练位级确定性：
  r6→r7 O1 最大 logit/opacity 漂移为 `0.001357 / 8.918e-05`，但 heldout aggregate exact。
- `V33-F13`：正式 run 的 source snapshot 必须与最终提交源码 byte exact，纯 EOF 空白也不能例外。r7 方法与门禁
  均通过，但提交前 `git diff --check` 清理了 4 个新文件的多余 EOF 空行，导致其中 2 个冻结快照不再与待提交源码
  exact；r7 因此降为 done noncanonical，不能仅凭“空白不影响算法”继续引用为 canonical。
- `V33-F14`：长 GPU 任务不能把前台 SSH 生命周期当作任务托管。r8 已完成模型与 finalizer，但 124 秒调用超时关闭
  stdout，外层 `tee` 收到 SIGPIPE，terminal 按预注册 trap 写成 `failed / exit 141`；不得把已有 summary 反推成 done。
  r9 改用 `nohup` 后台托管并以只读 SSH 轮询，最终正常 `done`、GPU 释放、9 个 source snapshots 全 exact。

## V3.3 P0 防重复结论（2026-08-11）

- `V33-F01`：官方 SAM3.1 source 可 checkout 不等于 checkpoint 可执行。当前代码固定为 `96914d2`，但
  `hf auth whoami` 为未登录且 cache 无 SAM3.1；不得绕过 gated access、猜权重 revision/hash，或让该门阻塞
  dual-opacity 主假设。S1 必须 exact fallback 到 V3.2 SAM2.1 canonical masks；未来解锁需新 task/protocol/run。
- `V33-F02`：论文写 code available、GitHub 仓库存在或项目页可访问，不等于存在 runnable implementation。
  GS-RoadPatching `468f812` 只有 HTML/CSS/JS/图片、无算法源码和根 LICENSE；OP2GS、3D-GIMP、FocusGS、
  LiDAR-EVS 也没有可固定官方 runnable source。后续只能称 `*-inspired` 或 `audit_only`，不得写 reproduction。
- `V33-F03`：R3D2 `3fc6e31` 已公开 Apache-2.0 训练/export/eval 代码，但仓库只声明下载 `sd-turbo/taesd`
  base，没有作者训练并导出的 R3D2 pipeline。单卡从零训练不是“补齐 inference”，且被计划禁止；S5 保持
  `weights_blocked`，不能拿 base diffusion 输出冒充 R3D2。
- `V33-F04`：GOR-IS source release 不消除许可与运行合同。根许可证只允许 non-commercial research/evaluation，
  torch/CUDA 未 pin，且要求 nvdiffrast、CUDA rasterizer 和 OptiX gtracer；没有 pretrained manifest。它只能作
  optional audit，不能抢占 RoadPatch 主线或被写成单卡已验证 baseline。
- `V33-F05`：Inpaint360GS 的官方 source/Apache-2.0 只支持进入 adapter/preflight。上游验证环境是 RTX 4090 /
  CUDA 11.8，并依赖外部 CropFormer/LaMa 权重；在 StreetGS split、相机、分辨率和输入 schema 冻结前，不得
  安装/训练。若 24 GiB 下必须静默降正式分辨率、改 heldout 或改相机数，必须 `blocked_single_3090`。
- `V33-F06`：P0 重新 hash 的 D2/S2/S3/mixed/chunk 五资产 exact 与 V3.2 `36 passed` 只证明 immutable baseline
  仍成立，不证明 V3.3 方法有效。P0 全程无训练/模型推理；S1 必须另建 protocol、run 和指标证据。

## V3.2 终局处置与复开门禁（2026-08-11）

V3.2 已以 `WS-V32-R0-INTEGRATION-01=done`、整体 `none_plan_complete` 收口。归档位于
[`archive/2026-08/worldsim-v3.2/`](archive/2026-08/worldsim-v3.2/README.md)。下面的 `V3-F34`–`V3-F46`
继续约束任何后续路线，但不构成继续执行 V3.2 的任务清单。

| 分支 | 终局处置 | 禁止的延续方式 | 合法复开条件 |
|---|---|---|---|
| S1 semantic lift | `done`，canonical r6 | 复用 identity-invalid r5；绕过 ID/token/rigid 三元核对 | 新数据或新语义假设；新 task/protocol/run；继续 fail-closed identity |
| S2 background inpaint | `done`，canonical r3 | 把 Telea unseen RGB 当作 geometry/GT；复活退化的 r2 | 独立深度或多视图证据；预注册未观测 3D 门；新 run |
| S3 actor harvest | `done`，canonical r3 | 把生成背面写成 GT；倒写 CUDA preflight 失败的 r2 | 新 actor/方法假设；固定 source/weight/license；新 task/run |
| S4 harmonizer | task `done`，non-temporal `excluded diagnostic`；temporal `blocked` | 仅凭锐化或全图指标把删除区重生车辆纳入生产链；绕过 gated 权重 | 合法取得 gated base；显式 semantic preservation + temporal gate；新 task/protocol/run |
| S5 multiview upper bound | `blocked`，未授权 | 猜测许可证、移植无根许可证代码或把未执行写成质量结论 | 明确可执行许可证与权重；独立资源审计；用户重新授权的新 task |
| R0 integration | `done`，canonical r4 | 从 exact package 外推 streaming、跨场景、闭环安全或 GT correctness | 为对应 claim 增加独立数据、协议、测量与 run；不得续写 r4 |

统一复开规则：外部门禁解除只改变“是否可提出新任务”，不会把 S4 temporal 或 S5 自动变成当前任务。任何复开都
必须引用相关失败 ID，使用新 task ID、新冻结 protocol 和不可复用 run ID；旧 `blocked/rejected/done` terminal
保持不可改写。当前 `next_action=none_plan_complete`。

## V3.2 防重复结论（2026-08-10）

- `V3-F34`：actor role 必须同时绑定 dataset instance ID、`instances_info.id`/instance token 与 checkpoint
  rigid model index。只分别验证 class、token registry 和 core count 会允许“2D mask 属于 actor A、D2 core 属于 actor B”
  的静默错配。所有 prompt、semantic lift、asset generation 在运行前必须 fail-closed 核对三元 identity；旧 r5
  因 ID `5`/token `af663…` 错配而失效，不得通过后续 adapter 或人工选图补救。

- `V3-F35`：AutoDL 根 `.condarc` 可把 `nvidia` channel 重写到缺包镜像；第三方官方 setup 的 CUDA channel
  不能只凭 channel 名复现。Asset Harvester 必须使用 `--override-channels` 和明确的官方 NVIDIA/defaults URL，
  同时记录 setup 日志；TUNA 对应 404 不能被误判成官方包不存在。
- `V3-F36`：复制第三方 setup 脚本到 `/tmp` 后，基于脚本位置计算的 `REPO_DIR` 会静默变成 `/tmp`。
  transport-only patch 必须冻结真实 checkout 绝对路径，并让恢复 wrapper 显式接收 formal `RUN_DIR`；不得把
  环境完成结果写入已拒绝的旧 run。
- `V3-F37`：`gsplat` 的浅克隆或 transport-only 复制不会自动带上 GLM submodule。Asset Harvester
  环境不能只以 `pip install` 成功为准；必须固定 `gsplat` commit、初始化 GLM 到 exact commit，
  并在当前 GPU 上 import CUDA extension。
- `V3-F38`：第三方 setup 子进程里的 conda activation 不会传回父 wrapper。恢复脚本后续记录、
  校验或 formal 推理必须使用明确的环境 Python 绝对路径，不得依赖裸 `python`
  或父 shell 的隐式 PATH。
- `V3-F39`：PyTorch 2.10 下 `torch.cuda.manual_seed_all` 不会立即建立 CUDA context；在此前调用
  `reset_peak_memory_stats(0)` 可以在模型加载前报 `Invalid device argument`。资源监控 runner 必须先
  `set_device` + `cuda.init`，再清零峰值计数器。S3 r2 因此在 GPU peak=`0 MiB`、无部分输出时
  `rejected`；修复后必须新建 formal run，不得改写 r2。
- `V3-F40`：边界目标的时间邻近帧不等于存在静态世界几何重叠。S2 r1 的固定支持集合跨过 frame `31` 后，
  boundary mask 的有效跨视图覆盖低于 `32` 像素；放宽深度门也不能修复零几何重叠。后续只能在 train-only
  视图上冻结 exhaustive camera/frame geometry audit，再新建 run；不得读取 held-out 来选支持帧，也不得单纯
  放宽深度容差掩盖视锥不重叠。
- `V3-F41`：2D unseen completion 可生成不等于其深度足以作为全时段静态 Background。S2 r2 把全部未观测
  Telea 区域写入 3D checkpoint 后，四路 held-out 平均 PSNR/SSIM 退化 `0.495842 dB / 0.007160`，形成后续帧
  灰色遮挡；候选必须拒绝。S2 r3 保留完整 2D unseen artifact/provenance，但高支持 checkpoint 只持久化
  cross-view observed geometry，并重新通过未放宽的 held-out 门。后续不得把 inpainted RGB 自动升级为
  geometry-grounded world state；若要持久化 unseen 3D，必须增加独立深度/多视图证据和新预注册门。
- `V3-F42`：Harmonizer 导出 JIT 不是脱离官方 NGC runtime 即可原样执行的普通 TorchScript。当前权重包含
  `tex_ts::rmsnorm_fwd_inf_ts`，PyTorch 2.10 还会把两个 einops shape scalar 随 `map_location` 移到 CUDA，
  造成 shape tensor CPU/CUDA 冲突。当前适配只允许使用独立公式验证为 BF16 exact 的 RMSNorm 回退，并将整数
  1/2 shape scalar 放回 CPU；必须记录 runtime deviation 和测试，不能写成 untouched official runtime。
- `V3-F43`：生成式 final-render enhancer 可以恢复外观，同时破坏明确的 counterfactual 语义。S4 r2/r3 在
  G1 remove+inpaint 区域重新生成 actor-like 黑色车辆外观；r3 的 mask 内 L1=`14.217278`、changed fraction
  `0.541750`，失败冻结 `12.0 / 0.40` 门。全图 PSNR、outside drift 或“看起来更锐利”都不能覆盖 actor deletion
  失败；non-temporal Harmonizer 仅保留 optional diagnostic，不得默认进入 remove 输出链。未来复开必须增加
  显式 semantic conditioning/preservation 和连续帧 temporal gate，并取得 gated Cosmos base 的合法授权。
- `V3-F44`：DriveStudio 只读 forward 使用 `torch.inference_mode()` 不等于 trainer 已切换到 eval。训练态
  renderer 会对 `means2d` 调用 `retain_grad()`，与 inference tensor 冲突；R0 r2 因此在首个 forward 明确
  `rejected`。所有只读 replay 必须在每次 state load 后显式 `trainer.set_eval()`；失败 run 不得补写结果，
  只能修复代码后新建唯一 run。
- `V3-F45`：质量门必须冻结指标、区域、单位和范数；`MAE<=1 uint8` 不能被实现成逐像素 L∞/max-error
  `<=1`。R0 r3 的 source→mixed PSNR=`67.24–68.43 dB`、MAE=`0.0093–0.0123`，但两视角存在极少量
  max error=`2`，因错误合同仍保持 `rejected`。修复必须更新 protocol/runner hash 后新建 run，不能在旧结果上
  改 gate 或把 max error 隐藏。
- `V3-F46`：R0 `done` 只证明当前 scene 的 V3.2 资产可追踪集成与固定三视角 storage/package 等价。
  `GENERATED_BACKGROUND` 和 `GENERATED_ACTOR` 仍不是 GT；S4 仍被排除；S5 仍阻塞。432 MB mixed checkpoint、
  444 MB chunk payload 与 8.36 GiB 峰值也不证明 streaming、load、render、跨场景泛化或闭环安全收益；未来相关
  claim 必须有独立 protocol、数据与测量，不能由 R0 exact reassembly 外推。

- `V3-F18`：A3 R1 的工程链可逐位重放，但 heldout 评测越过冻结 GPU ceiling，且资源无效 diagnostic 为
  geometry 改善与 RGB safeguard 退化并存。后续不得提高旧 ceiling、替换旧 renderer 或继续调同一四步配方。
- `V3-F19`：传感器原始尺寸、checkpoint 原生加载尺寸和评测输出尺寸是三个不同合同。A4-P0 v1 的
  `1600×900` 误记不能在后续路线重现；每次 profile 必须同时记录三层尺寸和 downscale 来源。
- `V3-F20`：checkpoint `state_dict` 键不等于加载后的 runtime attribute。任何新资产注册、恢复或 streaming
  代码必须同时审计保存端、加载端赋值和 live object，不能从序列化 schema 猜运行时 API。
- `V3-F21`：A4-P1 最小预注册剪枝臂 b05 已违反 global/non-target 质量门。不得事后增加 b01/b02、放宽
  `0.10 dB` 或只挑 actor/boundary 指标，把同一结果改写成剪枝成功。
- `V3-F22`：A4-P2 只证明选择性 FP16 参数存储在冻结质量门内可把 checkpoint 减少 `25.35%`；它没有证明
  renderer、load、FPS 或 peak VRAM 加速，且 Gaussian means 不能安全降为 FP16。
- `V3-F23`：A4-P3 只证明 159-file static/actor chunk package 可 exact 重组。package 比 source 大
  `2.79%`、load/reassembly 更慢；没有 demand loading、cache policy 和驻留集测量时，不能声称 streaming/LOD 收益。
- `V3-F24`：F0 只完成 Instant NuRec 官方能力与本机前置审计；本机没有执行 inference，standalone CLI 只导出
  static PLY。不得把它写成前馈基线质量失败，也不得把 static PLY 当完整 dynamic WorldSim checkpoint。
- `V3-F25`：R0 的 63 inputs、23 decisions、12 deliverables 与 P3 package exact 只证明 V3.1 证据链闭环。
  它不证明 D2 dominance、R1/P1 有效、P2/P3 加速、完整 world model、跨场景泛化或闭环安全。

