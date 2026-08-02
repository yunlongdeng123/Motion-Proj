# Research Status

- 更新时间：2026-08-02
- 当前路线：动态驾驶场景可编辑重建与失败诊断 V2
- 当前里程碑：`DR-V2-M1-DGGT-REPAIR-01`
- 状态：`pending / user-authorized / M0 done`
- 当前门禁：M0 已通过并解锁 M1；M1 提交前不得进入 M2
- 权威计划：[`DYNAMIC_DRIVING_EDITING_DIAGNOSTIC_PLAN_V2.md`](DYNAMIC_DRIVING_EDITING_DIAGNOSTIC_PLAN_V2.md)
- V2 授权前 Git 基线：`1e83ad5b`（`main` / `origin/main`）
- 当前分支：`research/dynamic-editing-v2`
- M0 正式 run：
  `/root/autodl-tmp/runs/dynamic_editing_v2/DR-V2-M0-BOOTSTRAP-01/20260802T115419Z__bootstrap-s0-r3`

## 当前裁决

`DR-V2-M0-BOOTSTRAP-01` 已按计划完成：独立分支、V2 文档入口、项目级镜像配置、空 shell/tmux
bootstrap、source audit、冻结资产复核和定向测试均通过。V1 数值和 `rejected` 终态未覆盖。

当前唯一动作是执行 `DR-V2-M1-DGGT-REPAIR-01`。只有 M1 形成 18/18 1-view 结果或可信 upstream
`blocked` 证据，并完成独立事实源更新和 commit，才可进入 M2。

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

## 下一步：只执行 M1

1. 新建 `/root/autodl-tmp/envs/dggt-v2` 并固定 Python/PyTorch/CUDA 构建；
2. 复核 DGGT official commit、model revision、license 与 preload provenance；
3. 按 upstream `python setup.py install` 构建 pointops2，并做 CUDA forward/backward；
4. 分开保存 untouched `difix/diffusion` 失败和最小 compatibility patch；
5. 完成固定 18-window 1-view、3-view diagnostic 与 common-observation 诊断，或形成可信 upstream blocked；
6. 更新事实源并独立提交 M1，之后才允许进入 M2。
