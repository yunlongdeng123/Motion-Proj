你现在接手 Motion-Proj 的第二轮 autonomous research。当前 Codex 对话没有历史记录，因此必须以远端仓库、Git 提交、实验产物和本文为事实源，不得根据名称猜测当前状态。

# 0. 连接与工作目录

连接后先定位 Motion-Proj 仓库，不要默认旧实例路径。开始任何修改前执行：

```bash
pwd
git status --short
git rev-parse HEAD
git log -8 --oneline
git branch --show-current
nvidia-smi
ps aux | grep -E 'python|train|motion_proj' | grep -v grep
```

预期最近提交包括但不限于：

```text
6845411  research(diagnostics): 预注册 endpoint locality 审计
9200467  research(diagnostics): 添加 SVD 特征可分辨性探针
171ec2a  fix(diagnostics): 隔离 track 中值算子的确定性限制
72ac28c  fix(diagnostics): 在首个 CUDA 调用前配置 cuBLAS
d46a881  docs(autoresearch): 固化路线决策与实验门槛
```

这些哈希只用于核对，不得强制 reset。若当前 HEAD 更新，先阅读新增提交并说明差异。

必须先阅读：

```text
AGENTS.md
docs/CVPR2027_PLAN.md
docs/EXPERIMENTS.md
docs/AUTORESEARCH_ROUTE_DECISION.md
docs/AUTORESEARCH_EXPERIMENT_PLAN.md
docs/AUTORESEARCH_LITERATURE_MATRIX.md
```

环境激活遵守 `AGENTS.md`：

```bash
source /root/miniconda3/etc/profile.d/conda.sh
conda activate /root/autodl-tmp/envs/motionproj
export PYTHONPATH=.
```

禁止自动 push。

---

# 1. 本轮任务

当前核心问题仍是：

> Motion-Proj 应继续 endpoint projection、转向 projected feature/track relation、采用 short-chain rollout，还是停止当前核心机制？

但上一轮已经完成：

* A0：仓库、参数化、LoRA、replay provenance 审计；
* F0：当前 endpoint locality 诊断；
* F1：当前 projector correction 与 frozen SVD feature grid 的分辨率审计；
* 一轮 2024–2026 最近邻论文检索。

不要重新运行 F0/F1，也不要立即训练 feature head、zero-conv 或 short-chain。

本轮目标是完成真正未关闭的四个问题：

```text
C0：Official SVD conditioning / generation parity
P0：Dynamics projector physical validity
P1：RGB / VAE counterfactual target validity
E0：Independent rollout evaluator validity
```

只有四项完成后，才能重新作路线决策。

如果 C0、P0、P1、E0 全部通过，可运行一次修订版 `F1-R`，但仍禁止运行 F2/F3。

---

# 2. 必须先修正的结论边界

上一轮诊断结果有效，但当前文档中的部分表述过强。不得删除历史证据，应追加一个 conclusion-boundary corrigendum。

## 2.1 F0 能证明什么

F0 实际证明：

> 在固定 replay pair、固定 (\sigma=0.05)、固定 noise、当前 teacher-relative residual-(v) target、共享 temporal LoRA 和当前 preserve 定义下，修正 `preserve_weight` 语义后仍没有找到同时满足预注册 correction 与 locality 门槛的点。

F0 足以否决：

* 继续扫描 preserve weight；
* 继续提高学习率；
* 延长当前单 pair shared-LoRA endpoint pilot；
* 将旧 preserve bug 解释为唯一失败原因。

F0 不足以证明：

* 所有 endpoint projection 都失败；
* 所有 sigma、pair、mask policy 都没有解；
* hard-gated residual branch 也必然失败；
* decoded RGB 或完整 rollout 的 locality 与 raw-(v) gate 完全等价。

## 2.2 F1 能证明什么

F1 实际证明：

> 当前 projector 的 correction 很小；在当前 8 pairs 中，绝大多数 correction 小于现有 SVD feature stride 的半个 cell，且 frozen raw feature 没有提供明显的现成 projected relation signal。

F1 是高风险信号，但以下推理不成立：

```text
correction < 0.5 feature cell
⇒ continuous feature target 必然不可学习
```

因为现有 probe 已使用或可以使用：

* bilinear feature sampling；
* Gaussian relation target；
* continuous soft-argmax；
* correlation distribution；
* sub-cell interpolation。

修订版 F1-R 必须以 target separation、gradient signal 和 sub-cell calibration 为主，`<0.5 cell fraction` 只能作为描述性指标，不能单独作为 hard fail。

## 2.3 首个提交

先更新：

```text
docs/AUTORESEARCH_ROUTE_DECISION.md
docs/AUTORESEARCH_EXPERIMENT_PLAN.md
```

增加“结论边界修订”章节，不覆盖原始 F0/F1 数值。

建议提交：

```text
docs(autoresearch): 收紧 F0 与 F1 结论边界
```

提交正文必须列出：

* 保留的证据；
* 不再允许的外推；
* 本轮 C0/P0/P1/E0 gate graph；
* 验证命令；
* 不涉及模型训练。

---

# 3. 本轮 gate graph

严格执行：

```text
A1 conclusion corrigendum
        │
        ├──────────────> C0 official-SVD parity
        │
        ├──────────────> P0 projector validity
        │                       │
        │                       └──pass──> P1 RGB/VAE target validity
        │
        └──────────────> E0 independent evaluator validity

C0 + P0 + P1 + E0 all pass
        │
        └──> F1-R revised feature signal audit
                    │
                    ├──pass──> 输出下一轮 F2/O1 计划，但本轮不训练
                    └──fail──> D 或 E 路线决策
```

C0、P0 和 E0 可以独立实现，但每个任务必须有独立 commit、run ID 和 summary。

禁止：

* Optuna；
* 300/800-step；
* formal Trainer V2 集成；
* feature head 训练；
* zero-conv/refiner 训练；
* short-chain；
* 新建大规模 cache；
* 修改原有 V5 cache；
* 自动 push。

---

# 4. C0：Official SVD conditioning parity

## 4.1 研究问题

当前 raw-(v) 代数已与 Diffusers scheduler 对齐，但项目的 conditioning/generation 路径可能没有精确复现官方 Stable Video Diffusion pipeline：

* `fps` 是否应在 added-time ID 前减 1；
* conditioning image 是否实际加入 `noise_aug_strength` 对应噪声；
* image latent encode 是否一致；
* conditional/unconditional CFG branch 是否一致；
* guidance scale schedule 是否一致；
* generator、scheduler state、latent initialization 是否一致；
* generation 输出是否可逐步或近似逐位复现。

必须回答：

> 当前 V5 Base rollout 是否由与官方 SVD pipeline 等价的 generation protocol 产生？

## 4.2 先审计代码

重点读取：

```text
motion_proj/backbones/svd_backbone.py
motion_proj/backbones/base.py
motion_proj/eval/generate_eval.py
motion_proj/replay/mine.py
configs/model/svd.yaml
configs/**/*.yaml
```

对照本地安装的 Diffusers 官方：

```python
StableVideoDiffusionPipeline
```

不得用记忆重写官方行为。若可联网，只使用 Diffusers 官方仓库和官方文档。

## 4.3 实现隔离诊断

新增：

```text
motion_proj/diagnostics/svd_conditioning_parity.py
configs/diagnostics/autoresearch_c0_conditioning.yaml
tests/test_svd_conditioning_parity.py
```

至少比较：

1. 官方 Diffusers pipeline；
2. 当前 `SVDBackbone.generation()`；
3. 修正 parity 后的候选实现。

固定：

```text
同一模型权重
同一 conditioning image
同一 generator seed
同一 initial latent
同一 inference steps
同一 fps
同一 motion_bucket_id
同一 noise_aug_strength
同一 min/max guidance scale
同一 dtype/device
```

逐项比较：

* added time IDs；
* conditioning image noise；
* image latent；
* initial video latent；
* 每个 scheduler timestep；
* model input scaling；
* conditional raw output；
* unconditional raw output；
* CFG output；
* scheduler step output；
* final latent；
* decoded frames。

不要只比较最终 RGB；必须定位首次出现差异的位置。

## 4.4 兼容性要求

若发现当前 protocol 与官方不一致：

* 不得静默改变旧 V5 cache 的语义；
* 新增显式 protocol version，例如：

```yaml
generation:
  protocol: svd_official_v1
```

* 旧协议标记为 legacy；
* manifest 保存：

  * Diffusers version；
  * scheduler config fingerprint；
  * conditioning protocol；
  * fps transform；
  * noise augmentation；
  * CFG settings；
* 明确旧 replay 是否需要在未来重建；
* 本轮不立即重建 122 条 cache。

## 4.5 通过标准

至少满足：

* added time IDs 精确一致；
* condition-noise tensor 精确一致；
* initial latent 精确一致；
* scheduler timesteps 精确一致；
* conditional/unconditional raw output 在 dtype 合理误差内一致；
* final latent/RGB 在预注册误差内一致；
* 两次 rerun 完全可复现。

如果当前 Base protocol 与官方不同，必须说明该差异是否足以影响旧 F0/F1，以及旧 F0/F1 哪些机制结论仍然有效。

## 4.6 产物

```text
runs/autoresearch-c0-conditioning-<unique-id>/
  manifest.json
  metrics.jsonl
  summary.json
  tensor_diffs.json
  figures/
  COMPLETE
```

建议提交：

```text
research(backbone): 审计并版本化 SVD conditioning 协议
```

---

# 5. P0：Dynamics projector physical validity v2

## 5.1 研究问题

当前 projector 的主要风险不是“平滑不够”，而是：

* 将 `background`、`dynamic_residual`、`foreground_candidate` 全部作为 positive object tracks；
* projector 内部移动 frame-0 track，但渲染后又硬冻结 frame 0；
* acceleration RMS 中位只剩约 13.6%；
* jerk RMS 中位只剩约 5.6%；
* 部分轨迹 turn sign 改变；
* correction 大量处于亚像素量级；
* 可能把 tracker noise、合理加减速、转弯曲率一起抹平。

必须回答：

> (p^\dagger) 是否是物理上更合理且可识别的 counterfactual trajectory，而不是“更接近匀速/静止”的轨迹？

## 5.2 不要直接修改正式 projector

先在 diagnostics 中实现候选 projector，避免污染已有 cache 和正式代码。

新增：

```text
motion_proj/diagnostics/projector_validity.py
configs/diagnostics/autoresearch_p0_projector.yaml
tests/test_projector_validity.py
```

最多比较四种候选：

```text
P-ID：identity，不修正
P-CUR：当前 smoother
P-CON：constrained robust smoother
P-UNC：uncertainty-gated constrained smoother
```

不得一次引入更多复杂方法。

## 5.3 Strata 语义必须分开

将 query strata 明确处理为：

### `background`

用途：

* preservation；
* background/camera motion estimation；
* negative relation；
* 估计 tracker noise floor。

禁止默认作为 positive moving-object projection。

### `dynamic_residual`

用途：

* 主要 dynamics correction candidate；
* 只有在 motion residual 显著高于 uncertainty 时才修正。

### `foreground_candidate`

用途：

* 高优先级候选；
* 仍需通过 visibility、support、confidence 和 uncertainty gate。

### `uncertain`

* ignore 或低权重；
* 不允许强制投影。

代码和文档中将误导性的 `object component` 改称：

```text
point-track tube component
```

除非已经存在 detector/instance identity，不得声称是 object-instance supervision。

## 5.4 硬约束

所有候选必须满足：

[
p^\dagger_{i,0}=p^b_{i,0}
]

以及：

* 不扩张 visibility；
* 不在 absent frame 产生点；
* 不越过有效 support；
* 不修改无足够置信度的轨迹；
* 不改变有效时间索引；
* 不使用 future GT ego、box 或 track。

建议约束：

[
\frac{|p^\dagger_T-p^\dagger_0|}
{|p^b_T-p^b_0|+\epsilon}
\in [r_{\min},r_{\max}]
]

[
\cos(\bar v^\dagger,\bar v^b)\ge c_{\min}
]

\operatorname{sign}(\mathrm{turn}^b)
]

但阈值必须先在 synthetic calibration 上预注册，不得运行后修改。

## 5.5 不允许以“低 acceleration”作为唯一目标

候选能量可包含：

\lambda_d\sum_t w_t\rho(p_t-p_t^b)
+
\lambda_a\sum_t\rho(\Delta^2p_t)
+
\lambda_j\sum_t\rho(\Delta^3p_t),
]

但必须同时保留：

* net displacement；
* mean velocity；
* motion direction；
* turn direction；
* dynamic degree；
* visibility；
* support。

需要报告：

[
\frac{\mathrm{accel}*{after}}*
*{\mathrm{accel}*{before}+\epsilon},
\qquad
\frac{\mathrm{jerk}*{after}}*
*{\mathrm{jerk}*{before}+\epsilon}
]

但不设置“越接近 0 越好”。

## 5.6 估计 uncertainty/noise floor

Projector correction 必须与 tracker uncertainty 比较，而不是只与 feature stride 比较。

至少实现一种低成本 uncertainty estimate：

* RAFT forward-backward consistency；
* 对输入施加极小 deterministic photometric perturbation 后重跑；
* query jitter 后的 track variance；
* provider 自带 confidence/visibility。

定义：

\frac{|p^\dagger_{i,t}-p^b_{i,t}|}
{\sigma^{track}_{i,t}+\epsilon}.
]

如果 correction 小于 tracker uncertainty，不应将其作为强训练 target。

## 5.7 Synthetic calibration

必须构建或复用小规模已知轨迹：

* 匀速；
* 合理匀加速；
* 刹车；
* 平滑转弯；
* 并线；
* 加入 tracker jitter；
* 单帧 outlier；
* 遮挡后恢复。

已知 clean trajectory 仅用于 synthetic calibration。

必须验证：

* 对 clean acceleration/turn 不过度平滑；
* 对 jitter/outlier 有效；
* 不通过静止化获益；
* frame 0 不移动；
* visibility/support 正确。

## 5.8 Generated replay audit

最多使用现有 8 clips，不生成新大 cache。

按 strata 分开报告：

* correction px；
* correction / uncertainty；
* net displacement ratio；
* mean velocity ratio；
* direction cosine；
* turn preservation；
* dynamic-degree ratio；
* acceleration ratio；
* jerk ratio；
* visibility/support violations；
* valid corrected fraction。

不要以 `<0.5 feature cell` 作为 P0 hard gate。

## 5.9 人工审查包

输出至少 12 个可视化 case：

```text
Base RGB
original tracks
projected tracks
confidence/uncertainty
per-stratum labels
before/after velocity/accel/jerk
difference overlay
```

生成 `reviews.template.jsonl`。

如果没有可信人工 review，不得自行把 P0 标记为最终 `done`；标记：

```text
awaiting_reviews
```

可以给出机器门槛结论，但必须把 human gate 单独列出。

## 5.10 P0 通过条件

候选 projector 至少满足：

* synthetic clean trajectories 不被系统性错误修正；
* synthetic noisy trajectories 显著接近 clean；
* frame-0 exact；
* visibility expansion 为 0；
* support violation 为 0；
* net displacement median ratio 位于预注册合理区间；
* direction median 高；
* turn preservation 高；
* dynamic degree 不系统下降；
* 主要 correction 的 SNR 高于预注册阈值；
* 不依赖 background positive projection；
* 人工 target 合理率达到预注册门槛。

如果所有 correction 均低于 tracker uncertainty，则结论应是：

> 当前 generated tracker/projector 无法提供足够可靠的 explicit correction signal。

不得人为放大 correction。

建议提交：

```text
research(projector): 校准置信度约束的轨迹投影
```

---

# 6. P1：RGB / VAE counterfactual target validity

只有 P0 至少获得一个 machine-eligible projector 后运行。

## 6.1 研究问题

当前 endpoint target 路径是：

```text
projected tracks
→ 16×16 crop/resize/paste
→ projected RGB
→ VAE encode
→ latent delta
→ latent mask 截断
```

已知风险：

* patch 被复制但 source 未移除；
* 无 occlusion/depth reasoning；
* overlap 和 feather 可能产生 ghosting；
* VAE receptive field 使 mask 外 latent 变化；
* 只保留 (M\odot\Delta z) 会形成未必对应真实 RGB 的 hybrid latent；
* track target 与最终渲染 target 可能不一致。

必须回答：

> 当前 endpoint target 是否对应一个可解码、可观察、物理语义一致的合法 counterfactual video？

## 6.2 必须比较的 target

对最多 8 个 P0 通过样本构建：

### Full projected latent

[
z_{\mathrm{full}}=E(X^\dagger)
]

### Current hybrid latent

z^b+M\odot(z_{\mathrm{full}}-z^b)
]

### Dilated hybrid latent

z^b+\widetilde M\odot(z_{\mathrm{full}}-z^b)
]

其中 (\widetilde M) 只能来自预注册 dilation/soft mask，不做连续搜索。

### Decode–reencode consistency

[
\hat X=D(z_{\mathrm{hybrid}})
]

[
\hat z=E(\hat X)
]

比较：

[
|\hat z-z_{\mathrm{hybrid}}|
]

和 projected-track realization。

## 6.3 诊断指标

记录：

* frame-0 RGB/latent exactness；
* mask 内/outside RGB L1、LPIPS；
* mask 内/outside latent RMS；
* full vs hybrid latent distance；
* decode–reencode error；
* projected track 是否在 decoded target 中真实实现；
* source duplication；
* ghosting；
* occlusion violation；
* texture stretching；
* subject identity；
* target motion direction；
* correction coverage。

必须单独报告：

```text
background
dynamic_residual
foreground_candidate
```

## 6.4 不要先实现复杂 compositor

第一步先证明当前 target 是否有效。

只有明确定位为 source duplication/occlusion 问题后，才能实现一个最小修复，例如：

* source removal；
* inpainting；
* depth-aware occlusion；
* alpha composition。

一次只允许一种修复，不要同时实现完整视频编辑系统。

## 6.5 Endpoint 路线停止条件

以下任一成立，当前 RGB/latent endpoint 主路线应标记 `rejected`：

1. hybrid latent 解码后无法实现 projected trajectory；
2. source duplication 或 ghosting 是系统性的；
3. 修正区域外产生明显视觉变化；
4. target 与 P0 projected track 不一致；
5. 需要大规模视频修复模型才能构造合法 target；
6. target correction 大部分低于 tracker uncertainty；
7. full latent 有效而 masked hybrid latent 无效，且 full latent 会破坏局部监督假设。

## 6.6 产物

新增：

```text
motion_proj/diagnostics/target_validity.py
configs/diagnostics/autoresearch_p1_target.yaml
tests/test_target_validity.py
```

运行目录：

```text
runs/autoresearch-p1-target-<unique-id>/
  manifest.json
  metrics.jsonl
  summary.json
  panels/
  reviews.template.jsonl
  COMPLETE
```

建议提交：

```text
research(projector): 审计 RGB 与 VAE 投影目标合法性
```

---

# 7. E0：Independent rollout evaluator validity

## 7.1 研究问题

当前训练 target、projector energy 和评价部分高度依赖 RAFT-chain track。

必须建立冻结、独立的 evaluator，避免：

```text
同一 tracker 产生 target
→ 同一 tracker 判断 target 改善
→ 同一 tracker 评价生成模型
```

候选优先使用官方 CoTracker3；如果不可用，可选择另一个官方、机制独立的长时点 tracker，但必须解释原因。

## 7.2 实现边界

新增 provider/evaluator，不替换现有训练 auditor：

```text
motion_proj/eval/independent_tracks.py
motion_proj/diagnostics/evaluator_validity.py
configs/diagnostics/autoresearch_e0_evaluator.yaml
tests/test_independent_evaluator.py
```

如果需要外部权重：

* 使用官方仓库/官方模型；
* 权重放数据盘；
* 记录 URL、commit、hash；
* 不提交模型权重；
* 不无理由修改全局 requirements；
* 优先 optional dependency。

## 7.3 Query 协议

评价 query 必须固定且可复现。

分别评估：

* background points；
* dynamic-residual points；
* foreground candidates。

不得用 future GT box/track。

如训练 tracker 与 evaluator 使用不同 query grid，需要设计可比的 clip-level metrics，而不是强行逐点对应。

## 7.4 Validity 测试

在已有 Base rollout 上测试：

### Repeatability

相同输入重复运行，检查：

* point coordinates；
* visibility；
* track survival；
* aggregate acceleration/jerk。

### Perturbation stability

施加极小、不会改变语义的 perturbation：

* 编解码；
* 轻微亮度；
* 轻微 resize roundtrip。

评价指标不应剧烈变化。

### Synthetic sanity

在已知合成平移、加速、转弯、遮挡视频上验证：

* acceleration 排序；
* jerk 排序；
* survival；
* occlusion；
* dynamic degree。

### Human alignment

输出至少 12 个 track overlay panels，区分：

* valid；
* drift；
* identity switch；
* occlusion failure；
* low-texture failure。

没有人工 review 时，不得宣称 evaluator 完全有效。

## 7.5 Camera/background motion

必须明确：

* 是否先移除 estimated background/camera motion；
* background motion 模型是什么；
* dynamic residual 如何定义；
* acceleration/jerk 是像素坐标还是 camera-compensated coordinates；
* 透视缩放是否会造成伪加速度。

禁止把所有像素 acceleration 直接称为“物理加速度”。

建议命名：

```text
camera-compensated image-plane acceleration
camera-compensated image-plane jerk
```

## 7.6 通过标准

至少满足：

* rerun 稳定；
* synthetic ordering 正确；
* visibility/occlusion 基本可信；
* 低纹理失败被识别并降低权重；
* 与 RAFT-chain 不完全同源；
* 人工 overlay 基本一致；
* 指标 coverage 和 invalid handling 明确；
* 无有效 track 时返回 invalid，而不是 0。

建议提交：

```text
research(eval): 建立独立长时点轨迹评价器
```

---

# 8. 修订版 F1-R：仅在四项通过后执行

只有以下全部满足才运行：

```text
C0 pass
P0 machine pass + review gate resolved
P1 pass
E0 pass
```

F1-R 仍然只做只读 feature audit，不训练 head。

## 8.1 不再使用单一 resolution hard gate

保留：

\frac{|p^\dagger-p^b|}
{\mathrm{stride}_\ell}
]

但它只作描述性统计。

必须增加：

### Target distribution separation

[
\mathrm{TV}
\left(
H(p^b),H(p^\dagger)
\right)
]

[
D_{\mathrm{JS}}
\left(
H(p^b),H(p^\dagger)
\right)
]

### Relation gradient signal

对冻结 feature 或临时 leaf tensor 计算：

[
|\nabla_F L_{\mathrm{relation}}|_{\mathrm{RMS}}
]

并与：

* float/bfloat numerical variation；
* repeated-forward variation；
* tracker uncertainty；

比较，形成 signal-to-noise ratio。

### Sub-cell calibration

人工构造 feature-grid shift：

```text
0.05 cell
0.10 cell
0.25 cell
0.50 cell
1.00 cell
```

验证：

* Gaussian target TV/JS 单调变化；
* soft-argmax 输出单调变化；
* relation loss gradient 非零且单调；
* observed/projected target 可区分。

### Actual target distinguishability

按 strata 报告：

* binary AUC；
* target TV/JS；
* gradient SNR；
* projected vs observed soft-argmax difference；
* frozen feature PCK；
* uncertainty-normalized correction。

## 8.2 F1-R 判断

Feature route 只有在以下情况才应停止：

* actual projected target distribution 几乎无法与 observed target 区分；
* relation gradient 接近数值噪声；
* artificial sub-cell calibration 本身失败；
* actual correction 低于 tracker uncertainty；
* P0/P1 target 本身不可信。

不得仅因：

```text
correction < 0.5 cell
```

而停止。

如果 F1-R 通过：

* 输出 F2/O1 的正式预注册计划；
* 不在本轮训练 feature head；
* 不实现 zero-conv。

建议提交：

```text
research(diagnostics): 复核连续特征关系信号
```

---

# 9. 文献更新要求

不需要重新做一份泛泛综述，但在涉及方法选择时必须联网核对 2024–2026 一手来源。

重点检查：

```text
Track4Gen
SG-I2V
MotionDirector
SHIFT
MoAlign
Geometry Forcing
VideoREPA
PhysAlign
SARA
ShortFT
SIFT
DenseDPO
VideoGPA
```

只使用：

* 论文原文；
* CVF/ECCV/ICLR/NeurIPS/OpenReview；
* arXiv；
* 官方项目页；
* 官方 GitHub。

本轮重点回答：

1. Track4Gen 是否支持 sub-cell continuous correlation，而不是离散 cell 分类；
2. Track4Gen 的 refiner/zero-conv 是否意味着 raw frozen feature failure 不能直接外推；
3. generic feature relation 已被哪些工作覆盖；
4. explicit self-rollout dynamics projector 仍可保留什么创新边界；
5. short-chain 只能在什么失败模式下成为合理 fallback；
6. endpoint counterfactual target 的合法性是否在现有工作中有成熟解决方式。

更新现有：

```text
docs/AUTORESEARCH_LITERATURE_MATRIX.md
```

不要为了增加篇数添加不相关论文。

---

# 10. 最终路线决策

完成 C0/P0/P1/E0，必要时完成 F1-R 后，重新选择：

```text
A. Endpoint projection distillation
B. Projected track / feature relation distillation
C. Short-chain / truncated rollout training
D. 继续诊断，尚不足以选择机制
E. 停止当前 explicit dynamics projection 核心方向
```

只能选择一个主结论和一个 fallback。

## 10.1 选择 A 的最低条件

Endpoint 只有在以下全部满足时可选：

* P0 物理轨迹有效；
* P1 counterfactual RGB/latent target 有效；
* target 不存在系统 duplication/occlusion/hybrid-latent 问题；
* C0 protocol 已对齐；
* E0 evaluator 有效；
* 能提出一个不依赖继续扫描 shared-LoRA preserve weight 的 locality mechanism。

如果仍只能使用当前 shared temporal LoRA + soft preserve，不得选择 A。

## 10.2 选择 B 的最低条件

Feature route 只有在以下全部满足时可选：

* P0/P1 target 有效；
* F1-R target separation 和 gradient SNR 通过；
* sub-cell calibration 通过；
* correction 高于 tracker uncertainty；
* 与 Track4Gen/MoAlign/Geometry Forcing/PhysAlign 有清晰创新边界；
* 预计单卡可验证。

不得仅因为 endpoint 失败就默认选择 B。

## 10.3 选择 C 的最低条件

Short-chain 只有在：

* one-step target 合法；
* one-step held-out 可学习；
* locality 可控；
* 独立 evaluator 有效；
* 唯一剩余失败是 one-step 到 25-step rollout transfer；

时才能作为主路线。

当前尚不满足，不要提前实现。

## 10.4 选择 E 的条件

如果：

* projector correction 低于 tracker uncertainty；
* 轨迹投影无法在保留合理运动的同时修正错误；
* RGB/latent counterfactual target 不合法；
* feature relation signal也不可识别；
* independent evaluator 无法可靠验证；

则应明确停止当前 explicit projection 核心机制，不要通过扩大数据、模型或训练预算掩盖失败。

---

# 11. 输出文档

新增：

```text
docs/AUTORESEARCH_PHASE2_PREREGISTRATION.md
docs/AUTORESEARCH_PHASE2_REPORT.md
```

更新：

```text
docs/AUTORESEARCH_ROUTE_DECISION.md
docs/AUTORESEARCH_EXPERIMENT_PLAN.md
docs/AUTORESEARCH_LITERATURE_MATRIX.md
docs/EXPERIMENTS.md
```

## `AUTORESEARCH_PHASE2_REPORT.md` 必须包含

1. Executive decision；
2. 当前 Git/环境事实；
3. F0/F1 结论边界修订；
4. C0 结果；
5. P0 结果；
6. P1 结果；
7. E0 结果；
8. F1-R 结果或未运行原因；
9. Route A–E 评分；
10. 最近邻工作撞车分析；
11. 最终主路线；
12. fallback；
13. 明确停止做什么；
14. Reviewer 2 最可能的 5 个攻击点；
15. 下一轮最多三个实验；
16. GPU、磁盘与时间预算；
17. 所有证据路径。

---

# 12. 测试、运行与提交规范

每个任务开始前：

```bash
git status --short
```

每个代码提交前：

```bash
PYTHONPATH=. pytest -q
git diff --cached --check
git diff --cached
```

不得通过修改 `sys.path` 掩盖错误；按仓库标准使用 `PYTHONPATH=.`。

建议提交顺序：

```text
docs(autoresearch): 收紧 F0 与 F1 结论边界
research(backbone): 审计并版本化 SVD conditioning 协议
research(projector): 校准置信度约束的轨迹投影
research(projector): 审计 RGB 与 VAE 投影目标合法性
research(eval): 建立独立长时点轨迹评价器
research(diagnostics): 复核连续特征关系信号   # conditional
docs(autoresearch): 固化第二阶段路线决策
```

每个 commit 只处理一个逻辑主题，不自动 push。

正式 run 必须包含：

```text
resolved config
manifest
Git commit
dirty status
model and cache fingerprints
seed
metrics.jsonl
summary.json
COMPLETE / FAILED / awaiting_reviews
```

失败 run 不删除、不覆盖、不复用 run ID。

---

# 13. 资源与停止规则

本轮总边界：

* 单张 RTX 4090 24 GB；
* 不超过现有 8–16 clips；
* 不超过 24 pairs；
* 不训练生成器；
* F1-R 只读；
* 单个任务显存不超过 22 GB；
* 不建立新大 cache；
* 不运行超过 200 updates 的任何任务；
* 不并行占用多个大模型。

遇到以下情况立即暂停并汇报，不继续猜测：

1. 当前 HEAD 与文档描述严重不一致；
2. worktree 有无法归因的用户修改；
3. C0 无法复现官方 SVD；
4. 旧 V5 replay 实际不是 claimed Base；
5. 发现 future GT 泄漏；
6. projector correction 低于 tracker uncertainty；
7. P1 target 系统性不合法；
8. independent evaluator 不稳定；
9. 需要引入大型视频编辑模型；
10. 需要超过本轮资源边界；
11. 关键 gate 需要人工复核但没有 review 结果。

人工 gate 未完成时，状态必须写：

```text
awaiting_reviews
```

不能自行视为通过。

---

# 14. 过程汇报

执行过程中每完成一个 gate，输出一次简短中间汇报：

```text
## Gate
C0 / P0 / P1 / E0 / F1-R

## Status
pass / fail / blocked / awaiting_reviews

## Verified facts
- ...

## Most important evidence
- ...

## Artifacts
- ...

## Decision impact
- ...

## Git
- commit:
- worktree:
```

如果某个 gate fail，按依赖图停止后续相关任务，但继续执行与其独立的 gate。例如：

* P0 fail：停止 P1/F1-R，但仍可完成 C0/E0；
* C0 fail：停止使用旧 protocol 做新生成，但仍可完成静态 projector 代码审计；
* E0 fail：不得作 rollout 性能结论。

---

# 15. 最终终端回复格式

```text
## Final decision

选择：A / B / C / D / E

一句话结论：

Fallback：

## Current repository state

- HEAD:
- branch:
- worktree:
- latest commits:

## Conclusion-boundary correction

- F0 actually proves:
- F0 does not prove:
- F1 actually proves:
- F1 does not prove:

## C0 — SVD parity

- status:
- first mismatch:
- protocol version:
- impact on old cache:

## P0 — Projector validity

- status:
- eligible strata:
- physical invariants:
- correction / uncertainty:
- human review:

## P1 — Target validity

- status:
- full latent:
- hybrid latent:
- duplication/occlusion:
- decoded trajectory realization:

## E0 — Independent evaluator

- status:
- provider:
- repeatability:
- synthetic sanity:
- human alignment:

## F1-R

- status:
- run / not run reason:
- sub-cell calibration:
- target TV/JS:
- gradient SNR:
- actual target distinguishability:

## Route comparison

| Route | Evidence for | Evidence against | Decision |
|---|---|---|---|
| Endpoint | | | |
| Projected feature | | | |
| Short-chain | | | |
| Reward/preference | | | |
| Stop | | | |

## Recommended next mechanism

- Method name:
- Core supervision:
- Trainable modules:
- Stop-gradient path:
- Locality mechanism:
- Novelty boundary:
- Expected single-GPU cost:

## Next three experiments

1.
2.
3.

## Stop conditions

- ...

## Files created or changed

- ...

## Tests and runs

- ...

## Git status

- commits:
- clean/dirty:
- uncommitted files:
- push:
```

最重要的研究纪律：

> 不要因为 F0 失败就宣布所有 endpoint 无效，也不要因为 F1 中 correction 小于半个 feature cell 就宣布连续 feature relation 无法学习。先证明 projector target 在物理上有效、在 RGB/VAE 中是合法 counterfactual、SVD generation protocol 与官方一致、独立 evaluator 可信；然后才讨论监督应该落在 endpoint、feature 还是 short-chain。
