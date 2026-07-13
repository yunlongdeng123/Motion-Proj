# Motion-Proj 实验事实源

这里只记录进入比较表的实验、失败结论和最终参数选择。原始 trial 指标保存在各 run 的 `metrics.jsonl`、SQLite 与 summary 中，不向 Git 追加流水日志。

## 比较实验

| Run ID | 状态 | commit | 数据/cache fingerprint | 方法 | seed | 关键结果 | 证据路径 | 结论 |
|---|---|---|---|---|---:|---|---|---|
| `legacy-mini-v1-2000` | completed | pre-2026-07-11 | legacy/unknown | SVD LoRA Motion-Proj V1 | unknown | 仅确认 2,000 step 链路和 adapter 保存成功 | `/root/autodl-tmp/runs/motionproj_v1` | 不进入正式主表；需固定 seed 生成比较 |

## P2 V1 调参归档（里程碑 `P2-V2-ARCHIVE-00`）

V1 使用固定 synthetic latent schema v4 cache（32 clips）、同一 Base 和 4 个固定 validation clips，完成 16 个 100-step trial，并将排序前 4 个续训至 300 step。所有 run 均为 clean commit `ae826a1`、seed `20260711`，cache fingerprint 为 `3f5d80ac1690028784849227782db4323e63f766c796e00acedf5d43d2d63315`。原始证据位于 `/root/autodl-tmp/runs/p2-tune-mini/{base_metrics.json,optuna.sqlite3,trials/}`。

Base 指标为：static drift `8.2095`、track acceleration `4.3953`、LPIPS `0.5088`、eligibility `85.51%`。第 4 个 validation clip 没有有效 track，但历史 evaluator 将 acceleration 写成了 `0.0`；该值只用于复现 V1 当时的排序，不作为 V2 有效 object metric。V2 必须将无有效 component 记为 invalid 并单独报告 coverage。

### 16 × 100-step

| trial | lr | lambda_proj | beta_anchor | B | tube upper | static ↓ | acceleration ↓ | LPIPS ↓ | eligibility | score ↑ |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 3.89e-5 | 0.2422 | 0.8471 | 8 | 0.35 | 10.4153 | 9.0077 | 0.4663 | 83.19% | -0.6590 |
| 1 | 1.14e-5 | 0.1248 | 0.7099 | 4 | 0.45 | 9.3451 | 7.2857 | 0.4865 | 84.81% | -0.3980 |
| 2 | 1.01e-5 | 0.0464 | 0.9627 | 4 | 0.25 | 9.5166 | 6.9921 | 0.4867 | 85.26% | -0.3750 |
| 3 | 1.51e-5 | 0.2360 | 0.6385 | 3 | 0.45 | 9.7457 | 7.5815 | 0.4905 | 81.85% | -0.4560 |
| 4 | 2.55e-5 | 0.0648 | 0.5242 | 8 | 0.25 | 10.1903 | 7.3292 | 0.4741 | 83.51% | -0.4544 |
| 5 | 2.44e-5 | 0.2287 | 0.3196 | 4 | 0.35 | 10.3559 | 8.0341 | 0.4808 | 83.86% | -0.5447 |
| 6 | 1.60e-5 | 0.2974 | 0.3029 | 6 | 0.45 | 9.7869 | 8.3829 | 0.4906 | 82.00% | -0.5497 |
| 7 | 1.87e-5 | 0.1459 | 0.5256 | 6 | 0.45 | 9.8345 | 6.2189 | 0.4879 | 82.92% | -0.3064 |
| 8 | 1.09e-5 | 0.0421 | 0.5969 | 8 | 0.25 | 9.4404 | 7.3830 | 0.4950 | 83.17% | -0.4148 |
| 9 | 2.69e-5 | 0.1408 | 0.9707 | 3 | 0.45 | 9.8010 | 6.4164 | 0.4754 | 80.40% | -0.3268 |
| 10 | 5.00e-5 | 0.1559 | 0.1558 | 6 | 0.45 | 9.8610 | 5.7143 | 0.4453 | 86.37% | -0.2506 |
| 11 | 4.93e-5 | 0.1565 | 0.1353 | 6 | 0.45 | 10.4404 | 7.6173 | 0.4524 | 86.09% | -0.5024 |
| 12 | 3.45e-5 | 0.1771 | 0.4250 | 6 | 0.45 | 10.4206 | 9.7582 | 0.4751 | 84.43% | -0.7447 |
| 13 | 1.95e-5 | 0.1028 | 0.1088 | 6 | 0.45 | 9.9107 | 8.0349 | 0.4922 | 82.93% | -0.5176 |
| 14 | 4.91e-5 | 0.1893 | 0.3116 | 6 | 0.45 | 9.8533 | 6.7548 | 0.4605 | 81.94% | -0.3685 |
| 15 | 3.28e-5 | 0.0979 | 0.4707 | 6 | 0.35 | 10.9156 | 8.4987 | 0.4757 | 83.81% | -0.6316 |

### 4 × 300-step

| trial | static ↓ | acceleration ↓ | LPIPS ↓ | eligibility | score ↑ |
|---:|---:|---:|---:|---:|---:|
| 7 | 11.1561 | 9.2454 | 0.4832 | 77.91% | -0.7312 |
| 9 | 9.4849 | 6.4871 | 0.4685 | 69.94% | -0.3156 |
| 10 | 12.2478 | 6.5092 | 0.4502 | 76.46% | -0.4864 |
| 14 | 13.4246 | 7.8267 | 0.4586 | 71.40% | -0.7080 |

搜索空间为 `lr=1e-5..5e-5`（log）、`lambda_proj=0.03..0.3`、`beta_anchor=0.1..1.0`、`B∈{3,4,6,8}`、`tube_upper∈{0.25,0.35,0.45}`，LoRA rank 固定为 16。16 个 100-step trial 的 score 全部为负；4 个 300-step 续训均未反转动力学退化，且 eligibility 整体下降。LPIPS 的改善不能抵消 static/track 的稳定恶化。

因此 `P2-V1-TUNE-01` 判定为 `rejected`。未启动 800-step：已有 300-step 证据显示继续暴露同一 32-clip cache 不会反转趋势，反而增加重复暴露和 eligibility 下降风险。`t10` 仅是失败配方中相对最高的 100-step trial，不登记为有效超参数，也不作为 V2 replay parent。

## 可信度验收

| 日期 | Run ID | 状态 | commit | 协议/数据 | seed | 关键结果 | 证据路径 | 结论 |
|---|---|---|---|---|---:|---|---|---|
| 2026-07-11 | `p0-geometry-synth100-s20260711-0b4a1899-e109eb12` | completed | `0b4a189` | `synthetic-object-track-v1` | 20260711 | 95/100 投影后目标轨迹能量下降，finite/mask 有效率 100%，最低 eligible fraction 78.44%；`temporal_gap` 为 15/20，其余四类均为 20/20 | `/root/autodl-tmp/runs/p0-geometry-synth100-s20260711-0b4a1899-e109eb12/{resolved.yaml,manifest.json,metrics.jsonl,summary.json,COMPLETE}`；config fingerprint `e109eb12`；cache `not-applicable:synthetic-object-track-v1` | 数据源码首次完整纳入 Git 后数值逐项复现；通过 P0 的 70% 合成错误验收，但不外推为 RGB/FVD 或驾驶可控性结论 |
| 2026-07-11 | `p2-resume-interrupted12-6c6261f` / `p2-resume-continuous12-6c6261f` | completed | `6c6261f` | P2 8-clip latent schema v4 / real-only / 12-step / 单卡 | 1234 | interrupted run 在 step 2 被 SIGKILL（137），worker 等待 60 秒后从完整 checkpoint 恢复；两条 run 的逐 step 指标、512 个 LoRA tensor、optimizer、sampler 和 Python/NumPy/Torch/CUDA RNG 全部相同，adapter `max_abs_diff=0.0` | `/root/autodl-tmp/runs/p2-resume-{interrupted12,continuous12}-6c6261f/{manifest.json,metrics.jsonl,events.jsonl,ckpts/step_000000012_final}`；config `ad090e66` / `d92fa375`；cache directory fingerprint `e2e50a40` | 通过中断—自动恢复—不间断逐位等价验收；结论限于单卡、`num_workers=0` 的 real-only 路径 |
| 2026-07-11 | `p2-train-{base,flow2,synthetic2,full2,replay2}-*` | completed | `5da76bf`/`ba53f20`/`9d9b28e`/`ced5e35` | 8-clip latent v4 路径 smoke；各 0–2 step | 1234 | base 冻结写 COMPLETE；flow/synthetic/full/replay 均完成 2 step 并写出 `summary.json`+`COMPLETE`；replay 使用重审计后合法 1-sample cache | `/root/autodl-tmp/runs/p2-train-base-5da76bf`、`p2-train-flow2-ba53f20`、`p2-train-synthetic2-ba53f20`、`p2-train-full2-9d9b28e`、`p2-train-replay2-ced5e35` | 训练矩阵路径可通；不代表生成质量或正式超参 |
| 2026-07-12 | `p2-v2-cond16-s20260712-fff5ccb-97d2d05d` | completed | `fff5ccb` | nuScenes val 固定 16 clips；冻结 SVD Base；25 inference steps；无 adapter | 20260712–20260727 | GT-ego/self-estimated/identity residual 均值分别为 19.2887/0.9320/2.1164；自动 finite、首帧冻结、无 GT 泄漏检查全部通过；人工 16/16 review 为 self 8 yes、4 no、4 uncertain，decisive 合理率 66.67% < 70% | `/root/autodl-tmp/runs/p2-v2-condition/p2-v2-cond16-s20260712-fff5ccb-97d2d05d/{resolved.yaml,manifest.json,condition_validity.jsonl,condition_validity_summary.json,condition_validity_panel,reviews.jsonl,COMPLETE}`；config `97d2d05d`；review `abbac56d` | H0 确认：SVD 不得使用 future GT ego static target；self-estimated static projector V1 因前景/伪影传播未过人工门槛，SVD static replay branch 标为 blocked，不启动训练 |
| 2026-07-12 | `p2-v2-api-unit-and-svd-structure` | completed | `5bd7a18` | `P2-V2-API-01` 工程门槛；预注册 6 sigma；tiny/full SVD-XT 结构；无训练 | 7 / 11 | float32 双向变换误差 `<1e-5`，bf16 通过 `2e-3` 容差；完整 SVD-XT 选中 temporal/spatial `128/0`，adapter tensor `256`，可训练参数 `3,319,808`；全量 `105 passed` | `tests/test_svd_parameterization.py`、`tests/test_lora_scope.py`；本地权重 `/root/autodl-tmp/weights/svd-xt` 只作结构 smoke | raw-v、sigma floor、anchor adapter 恢复和 temporal-only fail-closed 门槛通过；只解锁 V2 loss/gradient audit，不构成 rollout 收益证据 |
| 2026-07-13 | `p2-v2-grad-{base,v1}-s20260713-*` | completed | `ce52feb` / `63d9bd0` | legacy synthetic schema-v4 固定 2 sample × 6 sigma；无 optimizer update；V1 审计只读加载 `t10-300` adapter | 20260713 | 全部梯度 finite；V1 adapter 的 median `|g_x0|/|g_real|=2.7559`、median `|g_direct-v|/|g_real|=62.7214`；direct-v/anchor cosine 未触发负冲突阈值；spatial direct-v GradRMS 在 11/12 行超过 temporal 的 2 倍 | `/root/autodl-tmp/runs/p2-v2-gradient-audit/p2-v2-grad-s20260713-ce52feb-legacyv4/{gradient_audit.jsonl,gradient_audit_summary.json,gradient_norm_by_sigma.csv,gradient_cosine_matrix.csv,selected_modules.txt,COMPLETE}`；`/root/autodl-tmp/runs/p2-v2-gradient-audit/p2-v2-grad-v1-s20260713-63d9bd0-legacyv4/{gradient_audit.jsonl,gradient_audit_summary.json,gradient_norm_by_sigma.csv,gradient_cosine_matrix.csv,selected_modules.txt,COMPLETE}` | 仅确认参数化/梯度工程风险边界，保留 residual-v + trust-region；不得把 synthetic 输入或 V1 adapter 外推为 Base replay 或 rollout 改善 |
| 2026-07-13 | `p2-v2-gen04-unit` | running | pending | CPU 合成平移流；不生成 replay、不更新参数 | 0 | 轨迹 provider 单测覆盖无 GT 隔离、分层 query、F/B 拒绝、生成背景接线与配置 fail-closed；真实 Base panel 尚未执行 | `tests/test_generated_tracks.py` | 仅为工程门槛；不得登记 object replay 改善，也不得解除人工 panel 门槛 |
| 2026-07-13 | `p2-v2-gen04-panel1-s20260713-3cb8445` | completed / static review blocked | `3cb8445` | 冻结 SVD Base、val clip 0、25 inference steps、无 adapter；不写 replay cache | 20260713 | 自动检查通过、`uses_future_gt_track=false`；72 个分层 query，62 条有效轨迹，长度中位数 5，survival 44.44%，correction coverage 6.01%；已填 static verdict 为 `no` | `/root/autodl-tmp/runs/p2-v2-condition/p2-v2-gen04-panel1-s20260713-3cb8445/{manifest.json,resolved.yaml,cases/cond-00-clip-0000-seed-20260713.json,condition_validity_panel/cond-00-clip-0000-seed-20260713.mp4,reviews.jsonl,condition_validity_summary.json,COMPLETE}` | 评审再次确认 self-estimated static correction 在车体/护栏伪影处错位、撕裂，static branch 继续 blocked；该旧表单不含点轨迹 verdict，不能作为 GEN-04 object gate，需独立 8-case track review |
| 2026-07-13 | `p2-v2-gen04-track8-s20260713-59c3f05-net1` | completed / review passed | `59c3f05` | nuScenes val 固定 clip `0/96/192/288/384/480/576/672`；冻结 SVD Base；25 inference steps；无 adapter；不写 replay cache | 20260713–20260720 | 8/8 自动门禁通过：`uses_future_gt_track=false`、有效轨迹 45–72、长度中位数 3–8 帧、survival 29.17%–84.72%、correction coverage 4.27%–7.36%；`point_track_valid` 为 8/8 decisive、7 yes、1 no，合理率 87.5% ≥ 70% | `/root/autodl-tmp/runs/p2-v2-condition/p2-v2-gen04-track8-s20260713-59c3f05-net1/{resolved.yaml,manifest.json,cases,condition_validity_panel,reviews.jsonl,condition_validity_summary.json,COMPLETE}`；config `db483268`；experiment `5c7188bca4fe`；review `00babf16e071` | GEN-04 通过，解锁无 future-GT object component 与 P2-V2-REPLAY-05 schema/preflight；static branch 继续 blocked，不构建 replay cache 或启动训练 |
| 2026-07-13 | `replay-v5-smoke-s20260713-38a1665-objectonly` | completed | `38a1665` | nuScenes train / CAM_FRONT condition 0；冻结 SVD Base；25 inference steps；1 condition × 2 seeds；latent schema V5 | 20260713–20260714 | 2/2 kept；`parent_kind=base`、无 adapter/GT、首帧冻结、VAE/RGB/latent/residual 对齐均通过；static mask/confidence 均为 0，object mask 非空；object energy `1.6338→0.7115`、`0.1661→0.0288` | `/root/autodl-tmp/cache/p2-v2/replay-v5-smoke-s20260713-38a1665-objectonly/{_stage/manifest.json,_stage/resolved.yaml,_stage/metrics.jsonl,_stage/COMPLETE,*/metadata.json,*/COMPLETE}`；stage fingerprint `73af97ab718dc9` | 仅通过 schema/preflight，不代表 replay 质量或训练收益；开始分层 64×2 candidate，20-case 人工门禁前不得训练 |
| 2026-07-13 | `p2-v2-replay05-candidate64x2-s20260713-a41dfa4-objectonly` | completed / review passed | `a41dfa4` / `8d750f3` | nuScenes train / CAM_FRONT 全 split 均匀 64 conditions；冻结 SVD Base；25 inference steps；2 seeds；latent schema V5 object-only；固定 20-case 独立复核 | 20260713–20260714 | 128 candidate，122 kept、6 条空 object mask 硬拒绝；全 122 条 formal V5 reader/collate audit 通过；有效 object coverage 均值/最小/最大 `2.79%/0.09%/7.13%`；20/20 review 完整，16 decisive 全 yes、4 uncertain，合理率 100% | cache `/root/autodl-tmp/cache/p2-v2/p2-v2-replay05-candidate64x2-s20260713-a41dfa4-objectonly/{_stage/manifest.json,_stage/resolved.yaml,_stage/metrics.jsonl,_stage/summary.json,_stage/COMPLETE}`；cache fingerprint `e2e3a3b35f6d`；review `/root/autodl-tmp/runs/p2-v2-replay/p2-v2-replay05-review20-s20260713-8d750f3/{manifest.json,summary.json,reviews.jsonl,COMPLETE}`；review fingerprint `ccd9c5fa26d8` | P2-V2-REPLAY-05 通过，仅解锁 object-only 8-pair capacity pilot；static branch 保持 blocked，不构成生成质量或 rollout 收益结论 |
| 2026-07-13 | `p2-v2-pilot03-capacity-e200-s20260713-b4c2608` | completed / capacity failed | `b4c2608` | V5 object-only 固定 8 pair（train 4 / held-out 4）；固定 sigma/epsilon/z_sigma noise bank；temporal-only rank-16 LoRA；E residual-v + trust-region；200 update | 20260713 / 20260714 | target error `0.0137314→0.0105117`（下降 23.45% < 80%）；outside teacher drift 6.52% > 2%，frame-0 drift 0.1948；gradient finite/nonzero、roundtrip `7.38e-6`、direction cosine 0.4975 | `/root/autodl-tmp/runs/p2-v2-pilot/p2-v2-pilot03-capacity-e200-s20260713-b4c2608/{manifest.json,resolved.yaml,noise_bank.pt,metrics.jsonl,variants/E/adapter.safetensors,summary.json,COMPLETE}`；cache `e2e3a3b35f6d`；noise bank `e3071be549ae`；experiment `3b774e07bc7f` | E 未过 capacity gate；不进入 rollout/16×8 泛化/长训练。下一步仅固定同一 noise bank 跑 D，以定位 trust-region 是否为主因 |

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
| 2026-07-12 | `p2-v2-cond16-s20260712-fff5ccb-97d2d05d` / self-estimated static V1 | rejected | residual 数值下降但人工合理率仅 66.67%；失败集中为高覆盖 mask 把车辆或 Base 既有伪影当作背景传播，产生路面色块、拖影，另有 4 case 因 Base 灾难性失真无法判定 | 同 run 的 `reviews.jsonl`、`review_contact_sheets/`、`condition_validity_summary.json` | 不调整 70% 阈值、不重用 run ID；GT static 只保留 debug；`P2-V2-API-01` 已关闭，主线进入 `P2-V2-GRAD-02`，generated point-track 保持后续解锁 |

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
| 2026-07-11 | `p2-tune-mini-cache32-skiptracks` | completed | `8f27415` | 官方 train / CAM_FRONT；synthetic latent schema v4；`fill_policy=skip-empty-tracks-until-max` | 32 kept / 33 scanned（跳过 1 无轨迹） | stage `3845e398`；sample `ce38bd7e` | `/root/autodl-tmp/cache/p2-front/tune-mini-synth32-v2/_stage/{manifest.json,resolved.yaml,COMPLETE}` 与 32 个样本目录 | 固定 mini cache 就绪，可作 7–9h Optuna 训练输入；不代表生成质量 |
| 2026-07-11 | `p2-tune-base-metrics-hielig` / `tune-smoke2` | completed | `2f0476b` | val 固定高覆盖 clip 0/162/396/414；base 8-step 生成；2-step synthetic trial | seed 20260711 | base `lpips=0.509`、`eligible=0.855`；smoke summary 含全部 Optuna 字段且 `prune_reason=None` | `/root/autodl-tmp/runs/p2-tune-mini/base_metrics.json`；`/root/autodl-tmp/runs/p2-tune-mini/trials/tune-smoke2/summary.json`；低覆盖旧集保留为 `base_metrics.lowelig-0-100-200-300.json` | trial 评估桥可通；不把 2-step 分数当作正式超参 |

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
