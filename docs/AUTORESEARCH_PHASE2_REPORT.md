# Motion-Proj Autoresearch Phase 2 Report

日期：2026-07-14
范围：C0 official-SVD parity、P0 dynamics projector、P1 RGB/VAE target、E0 independent evaluator。
资源约束：单张 RTX 4090 24 GB；不训练生成器、不扩写 V5 cache、不自动 push。

## 1. Executive decision

**选择：E — 停止当前 explicit dynamics projection 核心方向。**

一句话结论：P0 能产生机器合格的连续 point-track correction，但当前
`projected tracks → crop/resize/paste RGB → VAE/hybrid latent` 无法稳定构成合法、可观察、可解码的
counterfactual target；独立 rollout evaluator 又未因官方 checkpoint 缺失而获得验证，因此不能把更多训练
预算投入这一监督链。

**Fallback：D — 继续诊断，尚不足以选择新机制。**只有新的 target construction 与可校验的官方独立
evaluator 都分别通过只读 gate 后，才重新预注册一个新问题；fallback 不包含 F1-R、short-chain、feature
head、zero-conv/refiner 或生成器训练。

这不是“所有 endpoint 或 continuous feature relation 都失败”的结论。停止仅针对当前 explicit projection
链及其共享的 RGB/VAE target；A1 corrigendum 保留 F0/F1 的严格外推边界。

## 2. 当前 Git 与环境事实

- 本阶段最后一个代码提交：`a712587` (`research(eval): 建立独立长时点轨迹评价器`)；它之前的 C0/P0/P1
  代码与证据提交依次为 `5b1bb6f`、`dfef913`、`960c4c2`，P1 gate 文档为 `f188698`。
- E0 formal manifest 记录的干净 Git commit 是
  `a71258720686feae6639b5f9cf68b2ae75d279f6`，`dirty=false`。
- 运行环境来自 E0 manifest：Python 3.10.20、PyTorch 2.4.1+cu121、CUDA 12.1、NVIDIA RTX 4090；seed
  `20260713`；cache fingerprint `e2e3a3b35f6d1af9a4c4a0ac4d7c38d116dfafd6af7151721b75b4edbbea1a39`。
- E0 开始前 GPU 使用为 1 MiB / 0%，正式 preflight 没有启动训练或生成任务。
- E0 code commit 前完整测试为 `150 passed, 2 warnings`；官方 CoTracker3 local hub entry 在
  `pretrained=False` 下可成功构造。

## 3. F0/F1 结论边界修订

- **F0 实际证明：**在固定 replay pair、`sigma=0.05`、noise、teacher-relative residual-`v`、共享 temporal
  LoRA 和当前 preserve 定义下，没有 checkpoint 同时满足 correction 与 mask 外/frame-0 locality；不得继续扫
  preserve weight、学习率或该单-pair pilot 步数。
- **F0 未证明：**所有 endpoint、所有 sigma/pair/mask policy 或具有独立局部参数子空间的机制均失败；也未证明
  raw-`v` one-step 与 decoded RGB/rollout locality 等价。
- **F1 实际证明：**旧 projector 在 frozen raw SVD feature probe 中的 correction 很小，stride-8 层有
  94.97% 小于半个 cell，且没有现成的 projected relation signal；故旧 target 不可直接启动 F2/F3。
- **F1 未证明：**bilinear/soft-argmax/Gaussian continuous relation 永远不可学习。F1-R 原应以 target
  separation、gradient SNR 和 sub-cell calibration 判断，而非单独用 `<0.5 cell`；本轮不运行它的原因是
  P1/E0 前置失败。

## 4. C0 — SVD parity

- **status：pass（legacy one-step mismatch 被显式保留）。**
- **正式 run：**`autoresearch-c0-conditioning-s20260714-v2`，clean commit `b36e042`，Diffusers 0.31.0，
  condition index 0，25 steps，seed `2026071401`。
- **first mismatch：**legacy `build_conditioning()` 的 fps time ID 为 7，official branch 为 6；condition
  noise、image embedding 和 image latent 的语义也不同。
- **通过事实：**`svd_official_v1` 下 official Diffusers pipeline、实际 backbone wrapper 和 versioned
  candidate 的 added IDs、condition noise、initial latent、每步 raw/CFG/scheduler output、final latent 和
  decoded RGB 均为 0 差异，rerun exact。
- **旧 cache 影响：**V5 Base rollout 的 generation provenance 不被否定，但其 stored legacy one-step context
  不得作为新的 one-step-to-rollout transfer claim，也不静默重建 122 条 cache。
- **证据：**`/root/autodl-tmp/runs/autoresearch-c0-conditioning-s20260714-v2/`；v1 的 device 类型失败 run
  保留在 `...-v1/`。

## 5. P0 — Projector validity

- **status：machine pass / awaiting_reviews。**
- **eligible strata：**`background` 仅 preservation/negative relation；正向 correction 仅限
  `dynamic_residual` 与 `foreground_candidate` 的 point-track tube component。
- **正式 run：**`autoresearch-p0-projector-s20260714-v1`，8 个 frozen Base indices，351 tracks；无 adapter、
  future GT 或 cache write。
- **机器合格候选：**P-UNC 唯一 eligible，101 条 primary track、290 个 corrected point，所有 primary
  correction SNR `>=1`；frame-0/visibility/time-index/support violations 均为 0；net-displacement
  median/p10=1；direction median=1；turn preservation=95.40%；dynamic-degree median ratio=0.862。
- **拒绝的候选：**P-CON 的 turn=88.79%、dynamic-degree=0.736；P-CUR frame-0 max=10.165 px、visibility
  expansion=127、turn=83.05%、dynamic-degree median=0.112。
- **synthetic：**P-UNC 保留 clean motion、改善 5/5 high-SNR single-frame outlier，并拒绝/不放大
  sub-uncertainty jitter。
- **human review：**12 个 panel 和 `reviews.template.jsonl` 已生成，尚无 verdict；它不被自行升格为最终
  pass。
- **证据：**`/root/autodl-tmp/runs/autoresearch-p0-projector-s20260714-v1/`。

## 6. P1 — RGB/VAE target validity

- **status：fail（machine hard gate）。**
- **正式 run：**`autoresearch-p1-target-s20260714-v2`，clean commit `960c4c2`，P-UNC 的 7 个含 primary
  component frozen Base clips；index 114 只有 background preservation，未伪装成空 target failure。
- **full/hybrid：**7/7 frame-0 RGB/latent exact；hybrid outside latent RMS/Base RMS 最大 0.00871，但这不能
  抵消目标不可实现性。hybrid target LPIPS 最大 0.06805，高于 0.05；同一 index 的 full VAE reconstruction
  也为 0.06805，说明问题不只是 mask 截断。
- **decoded trajectory realization：**index 34 的连续 P-UNC correction 经 integer crop/paste 后为 0 个
  target RGB changed pixel，故没有可学习的 RGB trajectory realization。
- **duplication/occlusion：**1 个 source-retention duplication proxy；588 个实际 moved-component 的
  overlap 缺少 depth/occlusion order。
- **结论：**当前 RGB crop/resize/paste + masked/dilated hybrid 不能提供合法 counterfactual；不以扩 mask、
  忽略 source 或引入大规模视频编辑模型规避。P1 v1 的过宽 occlusion-proxy scope bug 已保留，并由 v2 修正后
  重跑，失败结论未变。
- **证据：**`/root/autodl-tmp/runs/autoresearch-p1-target-s20260714-v2/`；保留 v1 在
  `...-v1/`。

## 7. E0 — Independent evaluator

- **status：blocked。**
- **provider：**官方 [CoTracker3 repository](https://github.com/facebookresearch/co-tracker) 的 offline
  predictor；local repository 固定在 `82e02e8029753ad4ef13cf06be7f4fc5facdda4d`。query 是 evaluator 自身
  的 first-frame grid，输入禁止 cache generated tracks、P0/P1 outputs、future GT 与 source-future metadata。
- **实现边界：**使用 evaluator-only robust affine background fit，并仅报告 camera-compensated image-plane
  velocity/acceleration/jerk；无有效 track 显式为 invalid，不回退至 RAFT/KLT。
- **block 事实：**`scaled_offline.pth` 不在
  `/root/autodl-tmp/weights/cotracker3/`，官方 checkpoint URL 在本机连接被拒；run 记录
  `fallback_used=false`、`uses_future_gt=false` 和 checkpoint SHA256=`null`（因为文件不存在）。
- **provider 选择：**只读扫描未发现可验证的预装 CoTracker/TAPIR/PIPS/TAPNet 等独立 tracker checkpoint
  或 Python package；不为填补空缺安装未核验依赖或改用与 RAFT 同源的 provider。
- **repeatability / synthetic sanity / human alignment：**均未执行，绝不写作 0、pass 或 fail metric；没有
  overlay panel，也没有 human verdict。
- **证据：**`/root/autodl-tmp/runs/autoresearch-e0-evaluator-s20260714-v1/`，状态为 `blocked` 但保留
  manifest、resolved config、metrics、summary 和 `COMPLETE` 终态标记。

## 8. F1-R

- **status：not run。**
- **原因：**必要条件为 C0 pass、P0 human pass、P1 pass、E0 pass；实际为
  `pass / awaiting_reviews / fail / blocked`。P1 fail 单独就阻断 feature supervision，E0 block 同时禁止
  rollout transfer 解释。
- **sub-cell calibration、TV/JS、gradient SNR、actual distinguishability：**未计算；不以缺失数值代替结论。

## 9. Route comparison

| Route | Evidence for | Evidence against | Decision |
|---|---|---|---|
| Endpoint | C0 parity pass；P0 P-UNC 机器不变量合格。 | F0 locality fail；P1 legality hard fail。 | rejected |
| Projected feature | 连续 relation 不应只按旧 F1 cell 统计否定。 | P1 target 不合法、P0 人审未完成、F1-R/E0 条件不成立；与 [Track4Gen](https://openaccess.thecvf.com/content/CVPR2025/html/Jeong_Track4Gen_Teaching_Video_Diffusion_Models_to_Track_Points_Improves_Video_CVPR_2025_paper.html) 邻域拥挤。 | rejected |
| Short-chain | 仅当合法 one-step 不能 transfer 到 rollout 时才有意义。 | 还未有合法 one-step target 或独立 rollout evaluator。 | rejected |
| Reward/preference | 可作为不同研究主题。 | 丢失 explicit projector；SHIFT/DenseDPO/VideoGPA 等邻域拥挤，且超出单卡当前证据链。 | rejected / out of scope |
| Stop | P1 正中 endpoint counterfactual 停止条件；E0 无法验证 rollout。 | 不外推为所有未来机制都失败。 | **selected (E)** |

## 10. 最近邻工作与撞车分析

- Track4Gen 已覆盖 SVD feature correlation、soft-argmax、refiner、zero-conv 与 temporal fine-tuning；在
  target 未合法化前实现类似模块没有可答辩的新边界。
- VideoREPA、MoAlign、SARA、Geometry Forcing 与 PhysAlign 已使 generic feature/geometry relation 训练高度
  拥挤；它们不能自动解决 generated RGB counterfactual 的 source/occlusion legality。
- ShortFT 与 SIFT 已覆盖不同形式的 short-chain/shortcut motion alignment；short-chain 不能把 P1 的零 RGB
  correction、duplication 或无 depth-order overlap 变成合法监督。
- E0 的设计本应切断“同一 RAFT 既造 target 又评分”的 circularity；官方 checkpoint 不可用时，诚实结论是
  缺少独立 rollout evidence，而不是换一个非等价 tracker。

## 11. 最终主路线、fallback 与推荐机制

- **Method name：**无；停止当前 explicit dynamics projection 核心。
- **Core supervision：**无；P1 否决当前 projected RGB/VAE target。
- **Trainable modules：**无。
- **Stop-gradient path：**Base、RAFT auditor、P0 projector、VAE、feature hook 与 optional independent
  tracker 全部保持 no-grad。
- **Locality mechanism：**不再训练；当前 shared temporal LoRA 的 locality 已由 F0 否决。
- **Novelty boundary：**当前不存在可成立的 method claim；保留的是可复现的 negative decision evidence。
- **Fallback D：**仅重新验证独立 evaluator 和全新 target construction，均为只读 diagnostics。

## 12. 明确停止做什么

- 不继续 current temporal-LoRA endpoint 的 LR/preserve-weight/update-step sweep；
- 不运行 F1-R、F2、F3、O1、short-chain、zero-conv/refiner 或任何生成器训练；
- 不扩容 cache、不改写 V5 cache、不用大模型视频编辑器掩盖 P1 failure；
- 不用 RAFT/KLT/其他 tracker 代替 E0 的官方 CoTracker3 provider；
- 不把 P0 的 machine pass 或 C0 parity pass 写成 rollout quality improvement；
- 不自动 push。

## 13. Reviewer 2 最可能的五个攻击点

1. **“Projector 只是平滑。”**P0 用 displacement/direction/turn/dynamic-degree、support 与 uncertainty
   约束缓解了这一点，但人审仍待完成；不将它宣称为最终 physical truth。
2. **“Counterfactual 不是视频。”**P1 给出直接反证（zero RGB realization、LPIPS、duplication、depth-order
   overlap），所以本报告停止而不是训练后挑指标。
3. **“评价 circular。”**E0 设计隔离了 tracker provenance，但没拿到权重；因此不声称 independent rollout
   improvement。
4. **“F0/F1 的失败被过度外推。”**A1 明确保留它们的有限范围；最终 E 的依据是 P1 legality，而不是 cell
   threshold。
5. **“方法只是 Track4Gen + smoother。”**没有合法 target 与 independent rollout causal ordering 时不提出该
   方法；这避免把邻近工作重命名为贡献。

## 14. 下一轮最多三个实验（均未排程）

1. 若获得官方 CoTracker3 offline checkpoint，记录 URL、repository commit、文件 SHA256 后以新 run ID
   重跑 E0 的 rerun、扰动、synthetic 和 12-panel review；不换 provider。
2. 若提出不依赖大型视频编辑模型的新 renderer，先在 P1 的 7 个 frozen clips 上验证 source removal、
   depth/occlusion order、decoded trajectory realization 与 VAE round-trip；不训练生成器。
3. 完成 P0 的 12-case human review，作为 projector 证据的独立人工补充；它不能反转 P1 machine fail。

## 15. GPU、磁盘与时间预算

- 当前后续训练预算：**0 GPU-hour**；没有自动后台任务。
- 条件性 E0 重跑上限：`<=1.5 GPU-hour`，只读 8 个已有 clips；新 checkpoint 仅放数据盘、必须记录实际 hash，
  不加入 Git。
- 条件性 target legality diagnostic 上限：`<=0.5 GPU-hour`，只读既有 7 个 clips；不建新大 cache、不用
  大模型 inpainting/video-editing。
- P0 human review：GPU 0；任何超过这些边界或要求训练生成器的提案都必须重新立项，而不是 Phase 2 的延续。

## 16. 证据路径

- A1 / route decision：`docs/AUTORESEARCH_ROUTE_DECISION.md`。
- C0：`/root/autodl-tmp/runs/autoresearch-c0-conditioning-s20260714-v2/`。
- P0：`/root/autodl-tmp/runs/autoresearch-p0-projector-s20260714-v1/`。
- P1：`/root/autodl-tmp/runs/autoresearch-p1-target-s20260714-v2/`，以及保留的 v1 scope-bug evidence。
- E0：`/root/autodl-tmp/runs/autoresearch-e0-evaluator-s20260714-v1/`。
- 代码/config/test：`motion_proj/diagnostics/{svd_conditioning_parity,projector_validity,target_validity,evaluator_validity}.py`、
  `motion_proj/eval/independent_tracks.py`、`configs/diagnostics/autoresearch_{c0_conditioning,p0_projector,p1_target,e0_evaluator}.yaml`、
  对应 `tests/test_*.py`。
- 实验索引：`docs/EXPERIMENTS.md`；预注册：`docs/AUTORESEARCH_PHASE2_PREREGISTRATION.md`；文献矩阵：
  `docs/AUTORESEARCH_LITERATURE_MATRIX.md`。

## 17. Final status

`C0=pass; P0=machine pass / awaiting_reviews; P1=fail; E0=blocked; F1-R=not run.`

因此当前仓库不含新的生成器训练结果，也不做任何 rollout-quality 提升声明。
