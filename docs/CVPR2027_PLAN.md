# Motion-Proj CVPR 2027 可持续研发计划

最后更新：2026-07-11
当前阶段：P2 前视训练矩阵 smoke 完成，准备固定 mini cache 调参
状态词：`pending` / `running` / `blocked` / `done` / `rejected`

## 已锁定研究决策

- 主线为完整 nuScenes 前视主实验、五个非前视相机零样本泛化、OpenDWM 第二骨干验证。
- SVD 用于单张 4090 上的快速开发、消融和调参；SVD 仅是 image-to-video 骨干，不支持未来 ego/layout 控制。
- 可控驾驶世界模型结论必须由 OpenDWM 支持。A100/H20 只用于最终迁移与论文级实验。
- 主视觉指标采用标准 nuScenes FID/FVD；动力学指标采用 DrivingGen 的 temporal、agent disappearance、trajectory consistency，不宣称自建 benchmark。
- `L_real` 独立消费 clean latent；V2 投影只使用合成错误或 generated replay，不直接投影干净 GT。
- LoRA rank 固定为 16。任何自动修复都创建带父 run ID 的派生实验，不改变原实验语义。
- 本地 JSONL、TensorBoard、SQLite/Optuna 是默认记录路径；W&B 不进入关键路径，只允许显式离线导出。

## 里程碑与验收门槛

| ID | 时间 | 状态 | 目标与验收 | 证据 | 下一步 |
|---|---|---|---|---|---|
| P0-GEOMETRY-01 | 7 月 | done | 几何 mask、LiDAR 标定深度、static drift 量纲、cache 失效和 gate 修复；100 个合成错误中至少 70% 投影后能量下降 | 数据源码纳入 Git 后由 `0b4a189` 重验：`p0-geometry-synth100-s20260711-0b4a1899-e109eb12` 为 95/100 改善并通过；`p0-geometry-mini5-0b4a1899-96306871` 复现 eligible fraction 均值 62.35% 和唯一失败项 `eligible_gate` | 保留 `temporal_gap` 5/20 未改善与真实覆盖率 62.35% 作为边界，不事后调门槛 |
| P0-RUNTIME-02 | 7 月 | done | 配置 schema、原子 cache、精确 checkpoint/resume、实验注册表和任务状态可测试 | `6c6261f` 的 P2 8-clip latent 实测：step 2 强制中断并自动恢复后，12-step 指标、LoRA、optimizer、sampler 和 RNG 与不间断对照逐位一致 | 保留单卡、`num_workers=0` 与确定性 CUDA 为正式训练约束 |
| P1-PROJECTION-01 | 8 月上旬 | done | V2 使用 synthetic/replay，人工检查至少 70% target 更合理 | `0b4a189` 完整追踪数据源码的检查包 `p1-projection-manual20-s20260711-0b4a1899-9487020a`：与原人工 review 包的 20 个视频及 case metadata 逐项完全一致，迁移同一份 review 后为 20/20 reasonable，review fingerprint `3d00cbe6` | 结论限于 synthetic corruption target 合理性；进入 P2，不外推生成质量 |
| P2-FRONT-01 | 8 月中下旬 | running | 官方 700/150 scene split，3,425/732 个 8 帧 clip；Base、real-only、flow、synthetic、replay、full 和核心消融 | 数据与 8-clip latent v4 已就绪；中断恢复逐位等价通过；base/real-only/flow/synthetic/replay/full 的 2-step smoke 均完成；replay mining 经重审计 static-drift 门控后 8-clip 为 1/8 kept（eligible 均值 65.2%，不降门槛） | 构建固定 mini synthetic latent cache，接入 Optuna trial 评估摘要后启动 16×100 调参 |
| P3-CAMERA-01 | 9 月上旬 | pending | 同一前视 checkpoint 零样本评估五相机；至少四个改善，macro 视觉质量退化不超过 5% | 待记录 | P2 主模型冻结后启动 |
| P4-OPENDWM-01 | 9 月下旬至 10 月 | pending | CTSD 3.5 baseline 达官方指标 10% 相对误差；三天失败则切换 CTSD 2.1 | 待记录 | A100/H20 资源确认后启动 |
| P5-PAPER-01 | 10 月 | pending | Full/OpenDWM 三种子与 bootstrap 95% CI；动力学显著改善且 FVD 相对退化不超过 5% | 待记录 | 10 月 10 日冻结主表，20 日冻结消融 |

## 工程验收

- 每次运行保存完全解析的 `resolved.yaml`；配置在运行期间只读，schema 启动时验证。
- cache/train/eval 每个 stage 均有 fingerprint、manifest 和完成标记；重复运行跳过已完成单元。
- checkpoint 原子目录包含 adapter、optimizer、step、sampler、Python/NumPy/Torch/CUDA RNG、resolved config、cache/dataset fingerprint 和 git commit。
- `resume=auto` 只选择兼容 fingerprint 的最后一个完整 checkpoint，`max_steps` 是目标总步数。
- 生成评估以 `(checkpoint, seed, clip)` 记录状态；已完成任务跳过，summary 可独立重聚合。
- 每个 run 目录不可复用，必须保留 manifest、resolved config、JSONL、stdout、checkpoint、summary 和父子关系。

## 7–9 小时调参协议

- 固定 mini scene-level train/val split，Optuna 使用本地 SQLite。
- 16 trial × 100 step，前 4 个续跑至 300 step，前 2 个续跑至 800 step；9 小时硬截止后不再启动 trial。
- 搜索域：`lr=1e-5..5e-5`（log）、`lambda_proj=0.03..0.3`、`beta_anchor=0.1..1.0`、`bound_B={3,4,6,8}`、tube 上界 `{0.25,0.35,0.45}`。
- NaN、eligible fraction `<70%` 或 LPIPS 比 base 恶化 `>5%` 立即淘汰。
- 排名为 normalized static-drift improvement 与 track-acceleration improvement 的均值；LPIPS 是约束和同分决胜项。
- 搜索结束只生成参数范围、曲线、淘汰原因、top-5 和下一轮建议，不自动扩大搜索。

## 资源与运行守护

- 预算：前视+LiDAR 约 31GB，六相机+LiDAR 约 60GB，多版本 cache/实验不超过 30GB；数据盘当前假设约余 207GB。
- 单张 4090 串行开发；不训练基础模型，不在 SVD 上实现同步六视角。
- 正式 worker 使用 `tmux + flock` 单实例；heartbeat 60 秒，watchdog 5 分钟检查进程、日志、GPU、磁盘、NaN 和 checkpoint 新鲜度。
- 子进程异常最多恢复三次，等待 1/5/15 分钟；配置错误直接 fatal。NaN 只允许从健康 checkpoint 派生 `lr*0.5` 重试一次；训练 OOM 失败，评估 OOM 减小 decode chunk。
- 实例不可达期间登记 `resume-pending`；重新可达后显式启动 worker 恢复。容器内程序不能把已关机的 AutoDL 实例重新开机。

## 风险

- LiDAR 与单目深度的尺度/外参错误会让 static energy 无意义；在 P0 阶段阻断所有主实验。
- SVD 的生成质量不能替代可控性证据；OpenDWM 复现门槛必须先过。
- 当前 mini 数据仅 49 个 clip，任何 mini 结果只用于链路和超参筛选。
- CVPR 2027 截稿尚未公布；目前按上一届 11 月中旬倒排，官方时间发布后立即更新。

## 变更日志

| 日期 | commit | 变更 | 原因 |
|---|---|---|---|
| 2026-07-11 | `f11645b` | 建立持久化计划、事实源和可恢复运行基础设施 | 避免研究决策只存在于对话和不可恢复脚本中 |
| 2026-07-11 | `094ff59` | 将未跟踪文件纳入 worktree 指纹，Git 状态不可用时 fail closed | 防止正式实验把未提交实现误记为 clean commit |
| 2026-07-11 | `8c8afef4` | 首次完成真实 mini5 与合成 100-case 验收 | 记录 62.35% 真实覆盖率；后续发现数据源码受 `.gitignore` 误伤，最终 clean-code 证据由 `0b4a189` 重验替代 |
| 2026-07-11 | `0e3b5e7` | 启动 P1 synthetic nuScenes 人工检查协议并导出首批 20-case 检查包 | P0 关闭后进入投影 target 合理性验证 |
| 2026-07-11 | `6fe74f9` | 分离检查包导出与人工 review 聚合 provenance | 防止 `aggregate-only` 用当前 commit 覆盖原始导出证据 |
| 2026-07-11 | `0b4a189` | 将误被 `data/` 规则忽略的数据源码纳入 Git，固化官方 split manifest，并重验 P0/P1 | 修复历史 clean manifest 未覆盖数据构建源码的溯源缺口；确认 P0 数值与 P1 视频逐项不变 |
| 2026-07-11 | `e6bb6d6` | cache schema v3 分离 clean/y/x_dagger 并拒绝 stale/fingerprint 混用 | 落实 `L_real` 独立消费 clean latent 和 V2 不投影 clean GT 的锁定决策 |
| 2026-07-11 | `fa375b2` | 固定 cuBLAS/cuDNN/PyTorch 确定性算法并在 P2 禁用 xFormers | 同 seed 独立训练从第 2 步开始分叉，单纯播种不足以支持严格恢复验收 |
| 2026-07-11 | `6c6261f` | 隔离 DataLoader RNG，并移除 Accelerate DataLoaderShard 的预取偏移 | checkpoint 的 sampler 不得越过尚未训练的 batch；恢复与连续训练最终逐位一致 |
| 2026-07-11 | `9d9b28e` | projector 总能量严格下降；replay 训练改读 stage fingerprint | 禁止空 track 虚报 energy_decreased；避免 mining fingerprint 与 cache_config 冲突 |
| 2026-07-11 | `ced5e35` | replay 能量门改为对 x_dagger 重审计 static drift | 生成帧无 GT track 时 e_dyn static 投影前后不变，不能作为能量下降证据 |
