# Motion-Proj CVPR 2027 可持续研发计划

最后更新：2026-07-11
当前阶段：P1 投影 target 人工检查
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
| P0-GEOMETRY-01 | 7 月 | done | 几何 mask、LiDAR 标定深度、static drift 量纲、cache 失效和 gate 修复；100 个合成错误中至少 70% 投影后能量下降 | `8c8afef4` clean run：`p0-geometry-synth100-s20260711-8c8afef4-e109eb12` 为 95/100 改善并通过；`p0-geometry-mini5-8c8afef4-96306871` 如实记录真实 eligible fraction 均值 62.35% 和唯一失败项 `eligible_gate` | 启动 P1，建立 synthetic/replay target 的人工检查协议；保留 `temporal_gap` 5/20 未改善作为边界案例 |
| P0-RUNTIME-02 | 7 月 | done | 配置 schema、原子 cache、精确 checkpoint/resume、实验注册表和任务状态可测试 | `f11645b`，`python -m pytest -q` 为 26 passed | 在 mini 训练中做中断恢复演练 |
| P1-PROJECTION-01 | 8 月上旬 | running | V2 使用 synthetic/replay，人工检查至少 70% target 更合理 | `0e3b5e7` 已导出 20-case 检查包 `p1-projection-manual20-s20260711-0e3b5e79-9487020a`（能量下降 20/20，eligible 均值 69.27%，待 review） | 人工填写 reviews.jsonl 并完成 70% reasonable 验收 |
| P2-FRONT-01 | 8 月中下旬 | pending | 官方 700/150 scene split，3,425/732 个 8 帧 clip；Base、real-only、flow、synthetic、replay、full 和核心消融 | 待记录 | 数据清单和 split fingerprint |
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
| 2026-07-11 | `8c8afef4` | 完成真实 mini5 与合成 100-case 的 clean commit 正式验收 | 以可重聚合证据关闭 P0；保留 62.35% 真实覆盖率而不事后调低门槛 |
| 2026-07-11 | `0e3b5e7` | 启动 P1 synthetic nuScenes 人工检查协议并导出首批 20-case 检查包 | P0 关闭后进入投影 target 合理性验证 |
