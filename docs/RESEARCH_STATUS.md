# Research Status

- 更新时间：2026-08-03
- 当前路线：动态驾驶场景可编辑重建与失败诊断 V2
- 当前里程碑：`DR-V2-M5-STRESS-3SCENE-01`
- 状态：`pending / user-authorized / M0–M4 done`
- 当前门禁：M4 已通过并独立提交；只执行 M5
- 权威计划：[`DYNAMIC_DRIVING_EDITING_DIAGNOSTIC_PLAN_V2.md`](DYNAMIC_DRIVING_EDITING_DIAGNOSTIC_PLAN_V2.md)
- V2 授权前 Git 基线：`1e83ad5b`（`main` / `origin/main`）
- 当前分支：`research/dynamic-editing-v2`
- M0 正式 run：
  `/root/autodl-tmp/runs/dynamic_editing_v2/DR-V2-M0-BOOTSTRAP-01/20260802T115419Z__bootstrap-s0-r3`
- M1 正式证据：
  `20260802T125138Z__native-nusc-s0-r6` + `20260802T133151Z__common-retry-s0-r8` +
  `20260802T133912Z__regional-s0-r9`
- M2 正式 run：
  `/root/autodl-tmp/runs/dynamic_editing_v2/DR-V2-M2-ACTOR-EVAL-01/20260802T140312Z__actor-eval-s0-r5`
- M3 正式 run：
  `/root/autodl-tmp/runs/dynamic_editing_v2/DR-V2-M3-EDIT-BASELINE-01/20260802T163930Z__formal-checkpoint-recovery-s0-r12`
- M4 正式 run：
  `/root/autodl-tmp/runs/dynamic_editing_v2/DR-V2-M4-EDIT-PILOT-01/20260802T171000Z__scene0230-pilot-s0-r7`

## 当前裁决

`DR-V2-M1-DGGT-REPAIR-01` 已按计划完成。pointops2 upstream CUDA 构建/反传通过，
checkpoint provenance 通过，untouched 失败与单行 patch 分离；18/18 1-view、18/18 3-view、
216/216 common target 和 504 行动态/边界诊断全部完成。

`DR-V2-M2-ACTOR-EVAL-01` 已完成：三场景 eligible actor=`16/20/6`，冻结 cohort=`2/2/2`，
4,356/4,356 camera observations 使用 timestamp+exact sample token 映射，输入哈希、cohort table、
raw 轨迹和三相机 QA 齐全。nuScenes GT 仍只用于评测选择和 oracle 诊断。

`DR-V2-M3-EDIT-BASELINE-01` 已完成：原生 30k checkpoint、token-first registry 与
`original/lateral +1m/remove` 三相机可逆编辑 smoke 均通过。M3 只证明基线与编辑 API 可运行，
不对视觉质量作结论。

`DR-V2-M4-EDIT-PILOT-01` 已完成：scene-0230 全部 196 帧、三相机、original/lateral/delete
共 1,764 张 RGB 和配套深度/掩码/footprint 均落盘，9 个同步视频非空，1,176 行 paired metrics
与 16/16 自动检查通过。人工抽检只确认产物非黑、编辑效应位于目标 footprint；不对视觉质量作通过结论。

## V1 历史终态

V1 结论保持不变，完整快照在
[`archive/2026-07/dynamic-reconstruction-v1/`](archive/2026-07/dynamic-reconstruction-v1/README.md)：

- AD-GS 六场景 exact reproduction `done`：
  `PSNR 31.174515 / SSIM 0.927661 / LPIPS(VGG) 0.163489 / coverage 6/6`；
- DGGT `blocked` 于 pointops2 PEP 517 build isolation，checkpoint/inference 未启动；
- V1 M6 只是持久身份资产审计，没有真实编辑、去遮挡或噪声压力测试；
- 候选 A 的 novelty `rejected`，但该结论不否定 V2 基于真实编辑失败重新立题。

## 现场事实

### 机器与资源

- GPU：NVIDIA GeForce RTX 3090，24,576 MiB；driver `580.105.08`；审计时 GPU 0 MiB；
- cgroup memory：`96,636,764,160` bytes（90 GiB）；`oom=0 / oom_kill=0`；
- 数据盘：`/dev/nvme0n1`，250G；清理后容量见 [`ENVIRONMENT.md`](ENVIRONMENT.md)；
- 无活跃研究 tmux/controller/GPU 进程；Cursor/Jupyter/TensorBoard 属于用户服务，不得终止。

### V2 直接输入

- AD-GS 六场景最终 `model_60000`、official renders、metrics 与 39G processed 输入保留；
- nuScenes 六场景 raw subset 与 manifests 保留；
- DGGT repo 固定 commit `a3276d2bbe4cbb03bcc117830b1836110a27adeb`；
- DGGT 完整预下载候选：
  `/root/autodl-tmp/checkpoints/dggt_preload/model_latest_nuscenes.pt`，
  `5,411,266,466` bytes，SHA-256
  `fd15644b3a878849470cbf5f0f9eae39167cfec1b853092898ae754c4f3acde9`；
- DriveStudio repo commit `e59bda4fa681f829dbb1d65f0de582b0f633c450` 与环境存在；V2 三个 pilot
  scene 的 processed data/checkpoint 不存在，M3 初始 readiness 为 `source available / assets missing`。

## M0 完成证据

- 首个工程失败实例：
  `/root/autodl-tmp/runs/dynamic_editing_v2/DR-V2-M0-BOOTSTRAP-01/20260802T114342Z__bootstrap-s0`；
- 正式完成实例：
  `/root/autodl-tmp/runs/dynamic_editing_v2/DR-V2-M0-BOOTSTRAP-01/20260802T115419Z__bootstrap-s0-r3`；
- 空 shell/tmux shell bootstrap：`PASS/PASS`；网络源：`4/4`；
- AD-GS 冻结资产：`6/6`；DGGT preload：bytes/SHA-256 `PASS`；
- 定向测试：`7 passed`；
- source audit：[`DR_V2_M0_SOURCE_AUDIT.md`](DR_V2_M0_SOURCE_AUDIT.md)。

## M1 完成证据

- 1-view mean：`PSNR 20.707359 / SSIM 0.856031 / LPIPS(Alex) 0.135780 / 1.785527 s`；
- 3-view mean：`PSNR 21.165262 / SSIM 0.771051 / LPIPS(Alex) 0.165553 / 4.517659 s`；
- AD-GS same-target 216/216：1-view `34.581860 / 0.951918 / 0.062490`，3-view
  `34.894344 / 0.951711 / 0.061447`；
- regional coverage：AD-GS `216`、DGGT 1-view `72`、DGGT 3-view `216`；
- 对照不是 matched leaderboard：DGGT 不用 pose/逐场景优化且在 294x518 推理，AD-GS 使用 pose、
  138 帧逐场景 60k 优化且在 900x1600 渲染。

## M2 完成证据

- 原始/合格/冻结 actor：`scene-0230 58/16/2`、`scene-0242 53/20/2`、
  `scene-0255 56/6/2`；
- 冻结 token：0230 `af663976... / 18c7f0c5...`，0242 `40f087d8... / 2c820a79...`，
  0255 `f4aa30b8... / 80c08b99...`；
- exact token mapping `4356/4356`；代表 panel 自动 QA `18/18`；视觉 QA `12/12`；
- `raw_annotations.provenance=nuscenes_raw_2hz`；`interpolated_visualization=[]`；
- DriveStudio frame 不可用时显式写 null 和 `assets_missing_until_m3`，不猜测。

## M3 完成证据

- DriveStudio `e59bda4` / MIT / native StreetGS 3-camera config；scene index `179`；
- 30k checkpoint：`386,398,646` bytes，`step=30000`，SHA-256 `8ed40576...a73f9e`；
- high-support token `af663976...` 映射为 `true id 13 / column 13 / model 5 / 2,683 Gaussians`；
- registry `24 actors / 23 non-empty / 1 explicit unavailable empty slice`；
- 27/27 edit-smoke PNG；14/14 invariant/effect checks 通过；峰值显存 `8,241 MiB`；
- final readiness `available=11 / missing=0 / incompatible=0`；r12 terminal=`done`；
- r8 的 post-render memory guard、r9–r11 的恢复 schema/依赖 fail-closed 实例全部保留，未覆盖。

## M4 完成证据

- full clip：`196 frames × 3 cameras × 3 variants = 1,764` RGB，另有逐帧 depth/opacity/
  dynamic opacity/target mask 与 source/edited footprint；
- 9/9 MP4、1,176/1,176 paired metric rows、16/16 checks 和内嵌图 QA 页面齐全；
- lateral/delete non-target PSNR=`93.394483/95.598042`，LPIPS(Alex, 256px)=
  `5.260851e-09/3.052960e-09`；目标 effect energy=`0.055526/0.031926`；
- SE(3) 最大平移误差 `3.814697e-06 m`，rotation/size/canonical drift=`0`，跨相机 world
  transform mismatch=`0`；
- 正式运行 `685.3 s`，峰值显存 `8,543 MiB`，峰值 cgroup `58,478,706,688 bytes`，
  `oom=0 / oom_kill=0`；最终 terminal=`done`；
- r1 保留 float32 容差过严的 blocked smoke；r5/r6 保留调试控制器被外层 timeout/tmux
  中断后的证据；正式 r7 使用独立 nohup controller 完成，没有覆盖旧实例。

## 下一步：只执行 M5

对 scene-0230/0242/0255 的冻结 high-support 与 boundary-support actor 执行四类预注册编辑，
完成 pseudo-hole、真值分层、感知诊断和跨场景 failure matrix。M5 提交前不进入 M6。
