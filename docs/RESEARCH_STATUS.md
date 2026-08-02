# Research Status

- 更新时间：2026-08-02
- 当前路线：动态驾驶场景可编辑重建与失败诊断 V2
- 当前里程碑：`DR-V2-M0-BOOTSTRAP-01`
- 状态：`pending / user-authorized / not started`
- 当前门禁：只允许执行 M0；不得提前安装 `dggt-v2`、运行 GPU inference 或进入 M1
- 权威计划：[`DYNAMIC_DRIVING_EDITING_DIAGNOSTIC_PLAN_V2.md`](DYNAMIC_DRIVING_EDITING_DIAGNOSTIC_PLAN_V2.md)
- V2 授权前 Git 基线：`1e83ad5b`（`main` / `origin/main`）
- V2 run：尚未创建

## 当前裁决

用户已授权新的 V2 诊断路线，但本次工作只完成执行前整理：归档 V1 文档、按现场校准 V2、清理 V2
不需要的可再生中间产物。它不是 V2 M0，也没有解锁 M1。

下一次清空 context 后的唯一动作是执行 V2 M0，并按计划先创建
`research/dynamic-editing-v2` 分支或独立 worktree。不得把本次预整理改写成 M0 `done`。

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

## 下一步：只执行 M0

M0 必须：

1. 读取本状态、失败账本、V2 实验注册表、V2/V1 计划和 `AGENTS.md`；
2. 检查 Git、最终资产 hash、GPU/cgroup/磁盘和无活跃 run；
3. 创建 V2 分支或 worktree；
4. 修正根 `README.md` 的 V7 过时入口，并协调 `AGENTS.md` 与 V2 的“禁止全局 conda init”规则；
5. 创建并验证项目级镜像配置与 `scripts/bootstrap_autodl_v2.sh`；
6. 完成 V2 指定的 source audit；
7. 更新 PLAN / STATUS / EXPERIMENTS 并只提交 M0。

M0 前不得创建 `/root/autodl-tmp/envs/dggt-v2`，不得运行 DGGT、DriveStudio 训练或任何正式 GPU task。
