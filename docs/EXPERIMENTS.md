# Motion-Proj 实验事实源

这里只记录进入比较表的实验、失败结论和最终参数选择。原始 trial 指标保存在各 run 的 `metrics.jsonl`、SQLite 与 summary 中，不向 Git 追加流水日志。

## 比较实验

| Run ID | 状态 | commit | 数据/cache fingerprint | 方法 | seed | 关键结果 | 证据路径 | 结论 |
|---|---|---|---|---|---:|---|---|---|
| `legacy-mini-v1-2000` | completed | pre-2026-07-11 | legacy/unknown | SVD LoRA Motion-Proj V1 | unknown | 仅确认 2,000 step 链路和 adapter 保存成功 | `/root/autodl-tmp/runs/motionproj_v1` | 不进入正式主表；需固定 seed 生成比较 |

## 可信度验收

| 日期 | Run ID | 状态 | commit | 协议/数据 | seed | 关键结果 | 证据路径 | 结论 |
|---|---|---|---|---|---:|---|---|---|
| 2026-07-11 | `p0-geometry-synth100-s20260711-0b4a1899-e109eb12` | completed | `0b4a189` | `synthetic-object-track-v1` | 20260711 | 95/100 投影后目标轨迹能量下降，finite/mask 有效率 100%，最低 eligible fraction 78.44%；`temporal_gap` 为 15/20，其余四类均为 20/20 | `/root/autodl-tmp/runs/p0-geometry-synth100-s20260711-0b4a1899-e109eb12/{resolved.yaml,manifest.json,metrics.jsonl,summary.json,COMPLETE}`；config fingerprint `e109eb12`；cache `not-applicable:synthetic-object-track-v1` | 数据源码首次完整纳入 Git 后数值逐项复现；通过 P0 的 70% 合成错误验收，但不外推为 RGB/FVD 或驾驶可控性结论 |
| 2026-07-11 | `p2-resume-interrupted12-6c6261f` / `p2-resume-continuous12-6c6261f` | completed | `6c6261f` | P2 8-clip latent schema v4 / real-only / 12-step / 单卡 | 1234 | interrupted run 在 step 2 被 SIGKILL（137），worker 等待 60 秒后从完整 checkpoint 恢复；两条 run 的逐 step 指标、512 个 LoRA tensor、optimizer、sampler 和 Python/NumPy/Torch/CUDA RNG 全部相同，adapter `max_abs_diff=0.0` | `/root/autodl-tmp/runs/p2-resume-{interrupted12,continuous12}-6c6261f/{manifest.json,metrics.jsonl,events.jsonl,ckpts/step_000000012_final}`；config `ad090e66` / `d92fa375`；cache directory fingerprint `e2e50a40` | 通过中断—自动恢复—不间断逐位等价验收；结论限于单卡、`num_workers=0` 的 real-only 路径 |
| 2026-07-11 | `p2-train-{base,flow2,synthetic2,full2,replay2}-*` | completed | `5da76bf`/`ba53f20`/`9d9b28e`/`ced5e35` | 8-clip latent v4 路径 smoke；各 0–2 step | 1234 | base 冻结写 COMPLETE；flow/synthetic/full/replay 均完成 2 step 并写出 `summary.json`+`COMPLETE`；replay 使用重审计后合法 1-sample cache | `/root/autodl-tmp/runs/p2-train-base-5da76bf`、`p2-train-flow2-ba53f20`、`p2-train-synthetic2-ba53f20`、`p2-train-full2-9d9b28e`、`p2-train-replay2-ced5e35` | 训练矩阵路径可通；不代表生成质量或正式超参 |

## 失败与拒绝结论

| 日期 | Run/Trial | 状态 | 结论 | 证据 | 后续 |
|---|---|---|---|---|---|
| 2026-07-11 | legacy cache | rejected | 旧 cache 无 schema/完成标记/fingerprint，且部分历史 metadata 曾含 NaN，不能作为可信投影证据 | `/root/autodl-tmp/cache/projection/*/metadata.json` | 由新 writer 幂等重建 |
| 2026-07-11 | `p0-geometry-mini5-5ff8e8c0-96306871` | completed | 首轮审计运行完成但验收失败，唯一失败项为 `eligible_gate`；数值已由 clean commit 正式 run 完整复现 | `/root/autodl-tmp/runs/p0-geometry-mini5-5ff8e8c0-96306871/{summary.json,manifest.json,metrics.jsonl}`；manifest commit `5ff8e8c0`，但审计新增文件运行时尚未被 Git 跟踪 | 保留为溯源不完整的历史 run，不覆盖；正式证据改用 `p0-geometry-mini5-8c8afef4-96306871` |
| 2026-07-11 | `p0-geometry-mini5-8c8afef4-96306871` | completed | 数值有效但代码溯源不完整：未锚定的 `.gitignore` 曾将 `motion_proj/data/` 排除在 commit 与 dirty 检测外 | 原 run 保留；manifest commit `8c8afef4`、config fingerprint `96306871` | 不再作为最终 clean-code 证据；由 `p0-geometry-mini5-0b4a1899-96306871` 替代 |
| 2026-07-11 | `p0-geometry-mini5-0b4a1899-96306871` | completed | 完整追踪数据源码后的审计仍仅 `eligible_gate` 失败：eligible 均值 62.3507%、最小值 54.7917%，0/5 达到 70%；深度-LiDAR Pearson 均值/最小值 0.8780/0.8480，其余检查均通过 | `/root/autodl-tmp/runs/p0-geometry-mini5-0b4a1899-96306871/{resolved.yaml,manifest.json,metrics.jsonl,summary.json,COMPLETE}`；commit `0b4a189`；config `96306871`；split `v1.0-mini:CAM_FRONT:first-5`；seed 1234 | 逐项复现旧 run；接受 62.35% 为适用范围，不调整 70% 门槛 |
| 2026-07-11 | `p2-resume-{interrupted12,continuous12}-f1b568c` | rejected | 两次同 seed、同 12-step 的未中断训练从第 2 步开始分叉；449/512 个最终 LoRA tensor 不同，`max_abs_diff=3.2711e-4`，证明只统一 RNG 播种不足以保证 CUDA 确定性 | 两个 run 目录下的 `metrics.jsonl` 与 `ckpts/step_000000012_final` | 由 `fa375b2` 固定确定性 CUDA 算法并禁用 P2 xFormers；本组不作为恢复证据 |
| 2026-07-11 | `p2-resume-{interrupted12,continuous12}-fa375b2` | rejected | step 2 强制中断和 60 秒自动恢复均成功，但恢复第 3 步 `loss/real=0.4505`，对照为 `0.5697`；checkpoint sampler position 为 3，越过了 Accelerate 预取但尚未训练的 batch | `/root/autodl-tmp/runs/p2-resume-{interrupted12,continuous12}-fa375b2/{events.jsonl,metrics.jsonl,ckpts/step_000000002/training_state.pt}` | 不降低等价门槛；由 `6c6261f` 移除 DataLoaderShard 预取并隔离 loader RNG 后重验 |
| 2026-07-11 | `smoke-replay8-5da76bf` | rejected | 曾报 1/8 kept，但 kept 样本 `total_before==total_after`；projector 用空 track 的 obj/prior `0<=0` 虚报能量下降 | `/root/autodl-tmp/cache/p2-front/smoke-replay8-5da76bf/` | 由 `9d9b28e`/`ced5e35` 修复后作废 |
| 2026-07-11 | `smoke-replay8-9d9b28e` | rejected | 改用总能量后 8/8 因 energy 拒绝；根因是无 track 时 e_dyn static 读同一审计 state，投影前后不变 | `/root/autodl-tmp/cache/p2-front/smoke-replay8-9d9b28e/_stage/{manifest.json,metrics.jsonl}` | 由 `ced5e35` 改为对 x_dagger 重审计 static drift |

## 人工检查（P1）

| 日期 | Run ID | 状态 | commit | 协议/数据 | seed | 关键结果 | 证据路径 | 结论 |
|---|---|---|---|---|---:|---|---|---|
| 2026-07-11 | `p1-projection-manual20-s20260711-0e3b5e79-9487020a` | completed | `0e3b5e7` | `projection-target-manual-v1` / `v1.0-mini:CAM_FRONT:synthetic-20` | 20260711 | 人工 review 为 20/20 reasonable；但导出 commit 未包含当时被误忽略的数据源码 | 原 run 完整保留；review fingerprint `3d00cbe6` | verdict 有效但不单独作为最终 clean-code 证据；迁移前先与 `0b4a189` 重导出包逐文件核对 |
| 2026-07-11 | `p1-projection-manual20-s20260711-0b4a1899-9487020a` | completed | `0b4a189` | `projection-target-manual-v1` / `v1.0-mini:CAM_FRONT:synthetic-20` | 20260711 | 20 个 mp4 与 20 个 case metadata 均和人工已审包完全一致；复用同一评审后 20/20 reasonable、0 borderline，合理率 100%，能量下降 20/20，eligible 均值 69.27% | `/root/autodl-tmp/runs/p1-projection-manual20-s20260711-0b4a1899-9487020a/{panels,cases,reviews.jsonl,summary.json,manifest.json,COMPLETE}`；config `9487020a`；review `3d00cbe6` | 通过 70% target 合理性门槛，关闭 P1；结论仅限 synthetic corruption，不代表生成质量 |

## 数据清单（P2）

| 日期 | Run ID | 状态 | commit | split | scene / clip | fingerprint | 证据路径 | 结论 |
|---|---|---|---|---|---:|---|---|---|
| 2026-07-11 | `p2-data-train-k8-0b4a1899-5d7cc689` | completed | `0b4a189` | nuScenes v1.0-trainval 官方 train / CAM_FRONT | 700 / 3,425 | `5d7cc689` | `/root/autodl-tmp/runs/p2-data-train-k8-0b4a1899-5d7cc689/{split_manifest.json,summary.json,manifest.json,resolved.yaml,COMPLETE}` | scene 数、clip 数和 clip ID 唯一性全部通过 |
| 2026-07-11 | `p2-data-val-k8-0b4a1899-72ce633b` | completed | `0b4a189` | nuScenes v1.0-trainval 官方 val / CAM_FRONT | 150 / 732 | `72ce633b` | `/root/autodl-tmp/runs/p2-data-val-k8-0b4a1899-72ce633b/{split_manifest.json,summary.json,manifest.json,resolved.yaml,COMPLETE}` | 与 train scene 不相交，全部清单门槛通过 |
| 2026-07-11 | `trainval-front-extract` | completed | `0b4a189` | v1.0-trainval / CAM_FRONT + LIDAR_TOP keyframes | 34,149 / 34,149 files | N/A | `/root/autodl-tmp/data/nuscenes/.trainval-front-extract/{summary.json,extract.log}` | 两个通道缺失均为 0；体积约 5.09GB + 23.71GB，未解压无关相机与 sweeps |
| 2026-07-11 | `p2-cache-smoke-rgb2` | completed | `9c6da23` | 官方 train 前 2 clip / synthetic / RGB cache schema v3 | 2 clips | stage `a02d3c92`；sample `d5549c79` | `/root/autodl-tmp/cache/p2-front/smoke-rgb/{_stage/manifest.json,_stage/resolved.yaml,_stage/COMPLETE}` 及两个样本目录 | 两条 cache 均为 schema v3；clean、y、x_dagger 分离，全部 finite，mask 范围为 [0,1] |
| 2026-07-11 | `p2-cache-smoke-latent8-v4` | completed | `3e0c41e` | 官方 train 前 8 clip / synthetic / latent cache schema v4 | 8 clips | stage `3518a430`；sample `ce38bd7e` | `/root/autodl-tmp/cache/p2-front/smoke-latent-v4-aa9ffe3/{_stage,*/metadata.json,*/COMPLETE}` | 8 个样本均为 `(8,4,32,56)` latent，flow 为 `(7,32,56,2)`，总计 5.4MB；clean/y/x_dagger、flow/confidence、seed/source/fingerprint 完整 |
| 2026-07-11 | `p2-replay-mine8-reaudit` | completed | `ced5e35` | 冻结 2-step synthetic adapter；8 clip；`energy_gate=reaudit-static-drift-v1`；2 inference steps | 8 clips | stage `a5973bb5` | `/root/autodl-tmp/cache/p2-front/smoke-replay8-reaudit/_stage/{manifest.json,metrics.jsonl,COMPLETE}` | 8/8 static drift 下降（例 59.1→30.1、35.0→9.6）；1/8 kept；eligible 均值 65.23%（min 55.56%、max 70.64%）；不降低 70% 门槛；弱 parent 与覆盖边界下 keep 率低属预期 |

## 参数选择

| 日期 | 范围 | 已锁定选择 | 依据 |
|---|---|---|---|
| 2026-07-11 | 开发期容量 | LoRA rank 16 | 避免把容量变化混入方法比较 |
| 2026-07-11 | 调参范围 | 见 `docs/CVPR2027_PLAN.md` | 固定搜索空间，防止事后扩域 |

## 登记规则

- 状态只使用 `queued/running/retrying/completed/pruned/failed`。
- promoted、rejected、failed trial 均保留，不复用 run 目录、不覆盖历史结果。
- 每条正式结论必须同时给出 commit、配置、数据/cache fingerprint、seed 和证据路径。
- 文档中的汇总必须能从运行目录或注册表重新生成；不手工修饰原始指标。
