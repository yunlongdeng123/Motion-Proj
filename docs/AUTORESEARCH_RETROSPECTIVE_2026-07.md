# Motion-Proj 2026-07-11 至 2026-07-14 实验复盘

> 目的：保留本轮已经证伪或被阻断的想法、证据边界和停止理由，避免以后仅凭“好像试过”而重复无效的训练、调参或 target 设计。
>
> 这是复盘文档，不替代实验事实源。数值和正式状态以 docs/EXPERIMENTS.md、docs/CVPR2027_PLAN.md 及各 run 的 manifest.json、resolved.yaml、metrics.jsonl、summary.json 为准。

* 记录日期：2026-07-14
* 覆盖范围：P2 V1/V2 训练诊断与 Autoresearch Phase 2（C0、P0、P1、E0）
* 复盘基线：Git commit 2e07773
* 当前决定：停止当前 explicit dynamics projection 监督链；不启动新的生成器训练、F1-R 或 rollout 比较

## 1. 先给结论

本轮没有得到一个可继续训练的 Motion-Proj 方法，但得到了足够明确的负结论：

> 当前 Base rollout → 轨迹平滑/投影 → RGB crop/resize/paste → VAE/hybrid latent → shared temporal LoRA 的链条，不能稳定产生合法、可观察、可解码的 counterfactual target。

最关键的停止依据不是某个损失没有下降，而是 P1 证明当前 target construction 本身不合法：连续的 point-track correction 经过整数 crop/paste 后可以变成零 RGB 改变；同时存在超阈值的 VAE round-trip 距离、source retention 和没有 depth order 的组件重叠。在这种 target 上继续训练，任何看似改善的 loss 或 LPIPS 都不能解释为学到了正确动力学。

这不是“所有 endpoint、feature relation 或几何监督都失败”的结论。它只否定当前共享 RGB/VAE counterfactual 及共享 temporal LoRA 的实现组合。保留下来的可复用资产见第 8 节。

## 2. 一页总览

| 尝试 | 结果 | 当前可得结论 | 不能得出的结论 |
|---|---|---|---|
| V1 synthetic cache 上的 projection distillation 调参 | rejected | 该 V1 配方的 LPIPS 改善与静态/轨迹动力学退化同时出现，不值得继续扫超参或长训 | 所有投影蒸馏、所有 projector 都无效 |
| 用数据集 future ego 定义 SVD static target | rejected | SVD 未接收 future ego 条件，GT-ego residual 不能作为正式生成 rollout 的监督 | GT ego 几何本身错误；可控 backbone 上不能使用该信号 |
| self-estimated static projector V1 | blocked | 高覆盖背景 mask 会把车辆和 Base 伪影传播为背景修正，人工合理率未过门槛 | 所有无 GT 的背景运动估计都不可行 |
| C/D/E temporal-LoRA 单步容量与 locality | blocked | 当前 shared temporal LoRA 能拟合单 pair correction，却不能同时保持 mask 外与 frame 0 的 locality | 有独立局部参数子空间的 endpoint 机制必然失败 |
| F0 preserve-weight endpoint sweep | hard fail | 固定 pair、sigma、noise 和 shared temporal LoRA 下，没有一个 checkpoint 同时满足 correction 与 locality | 任意 sigma、pair、mask 或 endpoint target 都没有可行点 |
| F1 frozen raw SVD feature probe | hard fail | 旧 projector 的 correction 在现有 feature grid 上过小，且没有现成 relation signal | 连续的 bilinear、Gaussian、soft-argmax relation 永远不可学习 |
| P0 轨迹 projector 候选 | 仅 P-UNC machine pass | uncertainty-gated point-space correction 可保留多项运动不变量 | 点空间合格自动意味着 RGB/VAE target 合法 |
| P1 RGB/VAE target construction | machine hard fail | crop/resize/paste 与 hybrid latent 不能提供当前所需的反事实视频 target | 更大 mask、更多训练或更强 loss 能修复 target 语义 |
| E0 官方 CoTracker3 独立 evaluator | machine pass / awaiting reviews | evaluator 的重跑与跨扰动排序稳定；可作为独立诊断候选 | 已完成人工对齐、已标定绝对 jerk，或有任何新模型 rollout 改善 |

## 3. 失败尝试一：V1 synthetic projection distillation

### 原始想法

在 32 个 synthetic latent cache 上，以 synthetic corruption 的投影 target 训练 SVD LoRA；训练目标包含 low-noise absolute x0 projection regression、real loss、full-image anchor，以及 spatial/temporal mixed LoRA。预期通过调学习率、projection 权重、anchor 权重、tube 参数等找到一个同时改善视觉和动力学的点。

### 实际观察

16 个 100-step trial 均正常结束，排名前 4 个继续到 300 step；没有 NaN、OOM 或异常恢复。相对 Base，100-step 中表面最好的 t10 的 LPIPS 从 0.5088 降到 0.4453，但 static drift 从 8.2095 升到 9.8610，track acceleration 从 4.3953 升到 5.7143。t10 继续到 300 step 后，static drift 进一步升到 12.2478，track acceleration 升到 6.5092，eligibility 从 85.51% 降到 76.46%。

在 32 个 cache clip、micro batch 1、gradient accumulation 8 的设置下，100/300 个 optimizer step 分别对应约 25/75 次平均 replay exposure。长训没有反转趋势。

### 为什么在当前配方下不继续

观察到的稳定模式是：重建类损失和 LPIPS 可以较快改善，但静态背景与对象轨迹动力学同时变差。继续增加同一小 cache 的暴露次数只会扩大这种风险；它不是一个“再多跑几百 step 就会翻转”的信号。

可能的贡献因素包括 synthetic corruption 与 Base rollout 误差分布不匹配、low-noise 未加权 x0 对 raw-v 的有效梯度弱、full-image anchor 与 mask 内 correction 的目标冲突、mixed LoRA 先拟合外观，以及 replay 样本重复暴露。它们是诊断假设，而非已被逐一证明的单因果解释。

### 明确不应重试的操作

* 不继续 V1 Optuna。
* 不启动 t10-800，也不把 t10 当成可用超参数。
* 不在这 32 个 synthetic cache 上继续增加 lambda_proj、训练步数或 replay 挖掘。
* 不把 LPIPS 改善解释为动力学改善。

### 证据

* /root/autodl-tmp/runs/p2-tune-mini/
* docs/EXPERIMENTS.md 的 P2 V1 调参归档
* docs/CVPR2027_PLAN.md 第 1–3 节

## 4. 失败尝试二：把数据集未来 ego 当作自由生成 SVD 的静态约束

### 原始想法

用 nuScenes source clip 的 future ego pose 计算 expected static flow，并将其作为 SVD 生成 rollout 的静态几何 target。

### 实际观察

冻结 SVD Base 的 16-case condition validity 诊断中，GT-ego、identity 和 self-estimated residual 均值分别为 19.2887、2.1164、0.9320。SVD 是 image-to-video 模型，并未接收 source clip 的 future ego pose；因此 GT-ego residual 主要衡量“生成结果与未条件化未来分支的差异”，而不是模型必须满足的物理误差。

这确认了 H0：future ego 条件不匹配。GT-ego 只保留为 synthetic/debug 工具，不再用于正式 SVD replay target。

### 自估背景分支为什么也没有继续

self-estimated static V1 试图只用 generated RGB 估计背景运动，避免 future GT 泄漏。虽然数值 residual 降低，但 16 个 case 的人工复核为 8 yes、4 no、4 uncertain；decisive 合理率 66.67%，低于预注册的 70%。失败集中在高覆盖 background mask 将车辆或 Base 已有伪影当作背景传播，形成路面色块和拖影。

因此，数值 residual 更小并不足以使 target 合法。该 static branch 标为 blocked，而不是靠放宽人工阈值或修改 review 标签恢复。

### 这件事留下的规则

监督 target 必须来自模型实际接收的条件，或由生成 RGB 自身可验证地估计。不能把数据集拥有、但模型从未看到的未来状态当成生成错误。

### 证据

* /root/autodl-tmp/runs/p2-v2-condition/p2-v2-cond16-s20260712-fff5ccb-97d2d05d/
* docs/CVPR2027_PLAN.md 第 5 节

## 5. 失败尝试三：当前 shared temporal LoRA 的 correction/locality 折中

### 原始想法

在冻结 Base 的 object-only V5 replay 上，以 temporal-only rank-16 LoRA 进行单步 target 拟合。C、D、E 分别检查 absolute direct-v、teacher-relative residual-v、以及 residual-v 加 continuous trust-region scaling。所有版本使用相同 8 个 pair、相同 noise bank、相同 200 update 上限。

### 实际观察

| 版本 | target error 降幅 | mask 外 drift | frame 0 drift | 结果 |
|---|---:|---:|---:|---|
| C：absolute direct-v | 0.95% | 20.59% | 0.6094 | failed |
| D：teacher-relative residual-v | 1.28% | 16.84% | 0.4077 | failed |
| E：D 加 trust-region | 23.45% | 6.52% | 0.1948 | failed |

预注册要求是 target error 至少下降 80%、mask 外 teacher drift 不超过 2%、frame 0 近数值零。C/D/E 都不满足。

为了分清“不可学习”与“不可局部”，只对固定 index 34、固定 sigma 0.05、固定 noise 做了单-pair 2×2 诊断。高学习率可使 correction 降低 95% 以上；加入 preserve 后仍有 8.45% mask 外 drift 和 0.2324 frame-0 drift。

### 为什么在当前设计下不继续

这个对照说明 correction 不是完全不可拟合，raw-v 代数、cache/VAE 对齐和 temporal-only LoRA 的基本容量也没有整体失效。真正没有同时满足的是共享 temporal adapter 的局部性约束：为了修改小的局部 target，它也改变了非目标区域与第一帧。

关闭 trust-region 的 D 比 E 更差，说明 trust-region 不是 E 失败的唯一根因；单纯继续扫 preserve weight、learning rate、LoRA rank 或 step 数没有预注册依据，也已经被固定 F0 sweep 排除。

### F0 的补充证据

F0 在同一 single-pair/shared-temporal-LoRA/raw-v 框架下扫描 lambda_preserve = 0、0.25、1、4、16。11 个 checkpoint 达到 correction 条件，但 0 个同时满足 correction fraction 不高于 20%、mask 外 raw-v ratio 不高于 2%、frame-0 raw-v max 不高于 1e-6。

这就是“当前 endpoint 失败”的精确含义。它不等于所有 endpoint 机制失败：拥有独立局部参数子空间、不同 mask policy、不同 pair/sigma 或不同 target 定义的机制没有被这个实验否定。

### 明确不应重试的操作

* 不在同一 shared temporal LoRA pilot 上继续扫 preserve weight、learning rate、LoRA rank 或 update count。
* 不把单-pair 的 95% correction 降幅外推为 8-pair capacity、rollout 或生成质量。
* 不因为单一 loss 看起来下降而跳过 mask 外和 frame-0 gate。

### 证据

* /root/autodl-tmp/runs/p2-v2-pilot/p2-v2-pilot03-capacity-c200-s20260713-9dd4c88/
* /root/autodl-tmp/runs/p2-v2-pilot/p2-v2-pilot03-capacity-d200-s20260713-0ca869a/
* /root/autodl-tmp/runs/p2-v2-pilot/p2-v2-pilot03-capacity-e200-s20260713-b4c2608/
* /root/autodl-tmp/runs/autoresearch-f0-endpoint-s20260713-6845411/
* docs/CVPR2027_PLAN.md 第 14 节
* docs/AUTORESEARCH_EXPERIMENT_PLAN.md 的 F0

## 6. 失败尝试四：直接在冻结 SVD raw feature 上学习旧 projector relation

### 原始想法

不先改 target，而是在冻结 Base SVD 的若干 feature hook 上用 correlation、soft-argmax 等关系表示，直接分辨 observed track 与 dynamics-projected track；若存在足够强的 feature signal，则再考虑 feature head 或 short-chain。

### 实际观察

在 8 个固定 cache index、7 个实际 hook path、sigma 为 0.05/0.2/1 的只读 probe 中，最细的 stride-8 层仍有 94.97% correction 小于半个 feature cell，且没有一个层满足预注册的 resolution/tracking promotion 条件。

### 为什么在当前输入下不继续

旧 projector 的绝大多数改变量在当前 raw feature grid 上几乎不可分辨，且没有现成 projected relation signal。直接把此 target 交给 feature head，极可能只是在数值噪声或插值误差上训练。

不过，“小于半个 cell”只能说明现有离散 probe 的风险很高，不能证明 continuous relation 永远不存在。bilinear sampling、Gaussian relation target、soft-argmax、correlation distribution 和 sub-cell calibration 都没有被 F1 普遍否定。原本可用 F1-R 重新审计，但它依赖合法的 P1 target；P1 已 machine fail，所以不应绕过前置条件。

### 证据

* /root/autodl-tmp/runs/autoresearch-f1-features-s20260713-72ac28c/
* docs/AUTORESEARCH_EXPERIMENT_PLAN.md 的 F1 与 F1-R

## 7. Phase 2：把“能否训练”拆成四个独立 gate

Phase 2 的价值不在于提出更多训练损失，而在于先拆开四个常被混在一起的问题：generation parity、point-track projector、RGB/VAE target、独立 evaluator。

### 7.1 C0：官方 SVD conditioning parity 不是方法失败，但揭示了旧 transfer claim 不能成立

首个 C0 v1 因诊断调用把字符串 device 传给 Diffusers 而失败。该 run 被保留，未用日志掩盖或复用 run ID。修复后，v2 在 Diffusers 0.31.0、25 steps 下证明 official pipeline、实际 backbone wrapper 与版本化 candidate 的 added IDs、condition noise、initial latent、逐步 raw/CFG/scheduler output、final latent、RGB 都是 0 差异，rerun exact。

同时，legacy build_conditioning 与 official branch 不等价：legacy 的 fps time ID 为 7，official 为 6，condition noise、image embedding 与 latent 语义也不同。

因此：

* official generation parity 是通过的，可作为未来生成协议的可靠基线；
* 旧 V5 Base rollout 的 generation provenance 不被否定；
* 但旧 cache 保存的 legacy one-step context 不能支撑新的 one-step-to-rollout transfer claim；
* 不静默重建 cache，也不把 C0 pass 写成训练收益。

证据：/root/autodl-tmp/runs/autoresearch-c0-conditioning-s20260714-v1/ 与 /root/autodl-tmp/runs/autoresearch-c0-conditioning-s20260714-v2/。

### 7.2 P0：简单 smoother 不能安全替代 uncertainty-gated projector

P0 在 8 个 frozen Base clip、351 条重建 track 上比较 P-ID、P-CUR、P-CON、P-UNC。P-CUR 会改动 frame 0（最大 10.165 px）、扩张 visibility（127），turn preservation 只有 83.05%，dynamic-degree median ratio 0.112。P-CON 虽更保守，但 turn preservation 88.79%、dynamic-degree median ratio 0.736，仍不过 gate。

P-UNC 是唯一 machine-eligible 候选：101 条 primary track、290 个 corrected point 的 SNR 都不低于 1；frame-0/visibility/time-index/support violation 都为 0；turn preservation 95.40%，dynamic-degree median ratio 0.862。合成集还显示它不系统性平滑干净运动，能改善高 SNR 单帧 outlier，并拒绝/不放大 sub-uncertainty jitter。

这说明“把轨迹平滑一点”不够。只有 confidence/uncertainty、support、visibility 与运动不变量同时受约束时，点空间 correction 才暂时可信。

但 P0 仍是 machine pass / awaiting reviews，12 个 panel 尚无人审；更重要的是，点空间合格不是 RGB target 合格。P1 正是为检验两者之间缺失的一步而运行。

证据：/root/autodl-tmp/runs/autoresearch-p0-projector-s20260714-v1/。

### 7.3 P1：当前 RGB crop/resize/paste + VAE/hybrid target 是本轮的 hard fail

P1 使用 P-UNC 的 7 个含 primary component frozen-Base clip，比较 full VAE、masked hybrid 与 radius-1 dilated hybrid。不使用 adapter、future GT，也不写 cache。

通过的部分不能掩盖失败：

| 检查 | 结果 | 含义 |
|---|---|---|
| frame 0 RGB/latent | 7/7 exact | 首帧冻结实现正确 |
| hybrid mask 外 latent ratio | 最大 0.00871，小于 0.02 | 表面局部性符合阈值 |
| decoded RGB trajectory realization | index 34 的 changed pixel count 为 0 | 连续轨迹 correction 被整数 compositor 量化掉，目标在 RGB 中不可观察 |
| full/hybrid target LPIPS | 最大 0.06805，大于 0.05 | 问题不只是 masked hybrid 边界；full VAE 也同样越过阈值 |
| source duplication | 1 个 proxy | 移动物体后 source 未被正确处理 |
| 无 depth order 的 moved-component overlap | 588 个 | 遮挡/重叠语义没有定义 |

这是一个 renderer/target semantics 问题，而不只是把 mask 扩大、把 preserve 权重调大或增加训练数据的问题：

1. point-track correction 是连续坐标；当前 renderer 是整数 crop/resize/paste。小的连续移动可以在离散像素域变成零变化。
2. 把局部 source crop 复制到新位置不等于生成运动后的物体；source removal、遮挡层次、背景显露和 texture consistency 都没有被定义。
3. 即使 mask 外 latent 很小，decoded target 仍可能无法与投影 RGB 对齐。小的 latent locality 数值不能替代合法视频反事实。

因此 P1 的 machine_pass 为 false、status 为 fail。它单独阻断 endpoint A、F1-R、feature head、short-chain 与任何生成器训练。人审模板只能补充证据，不能覆盖 machine hard fail。

P1 v1 曾把未移动 dense query overlap 计入 occlusion proxy；该 scope bug 被保留，v2 改为只统计实际 integer-paste move 后重跑，machine failure 仍然存在。这一点很重要：结论来自修复后的更窄统计范围，而不是错误计数。

证据：/root/autodl-tmp/runs/autoresearch-p1-target-s20260714-v1/ 与 /root/autodl-tmp/runs/autoresearch-p1-target-s20260714-v2/。

### 7.4 E0：独立 evaluator 已有机器稳定性，但不能替代 target legality

E0 的目标是打破“RAFT 既造 target 又评分”的循环。它只读取 generated RGB、evaluator 自己的 first-frame grid 和官方 CoTracker3 offline weights，不读 cache track、P0/P1 output、future GT 或 source-future metadata。

本轮保留了两个工程性失败，且没有假装它们是通过：

* v1：官方 scaled offline checkpoint 缺失，本机官方下载被拒；没有用 RAFT、KLT 或其他 provider 代替。
* v2：checkpoint 上传后，8/8 track valid、rerun 和 synthetic 检查看似通过；但 survival-threshold correlation 错把 baseline 与其自身过滤子集相关，不能验证扰动稳定性。v2 被保留为 scope-bug evidence。

修复后的 v3 使用实际的 Base-vs-photometric/codec/resize 跨 clip 四项 aggregate rank。8/8 real clip valid；同视频重跑的坐标、visibility 与 aggregate 差均为 0；最小 rank correlation 为 resize acceleration 的 0.97619，超过原先锁定的 0.8 阈值。12 个 overlay 都可解码。

可用结论严格只有：

* CoTracker3 v3 对本协议中的排序是机器稳定的；
* resize 的绝对 aggregate delta 中位数/最大值为 10.03%/31.84%，因此不能把绝对 image-plane jerk 当作物理标定；
* 0/12 human verdict，所以仍是 awaiting_reviews；
* E0 没有比较任何训练模型，不能产生 rollout-quality improvement 结论，更不能修复 P1。

证据：/root/autodl-tmp/runs/autoresearch-e0-evaluator-s20260714-v1/、/root/autodl-tmp/runs/autoresearch-e0-evaluator-s20260714-v2/、/root/autodl-tmp/runs/autoresearch-e0-evaluator-s20260714-v3/。

## 8. 哪些东西没有失败，值得复用

负结果不等于整套基础设施无用。下列资产可以在新的、重新预注册的问题中复用，但不自动授权继续当前训练：

| 资产 | 当前可信范围 | 不应被误用为 |
|---|---|---|
| C0 的 svd_official_v1 protocol | matched inputs 下与官方 Diffusers SVD pipeline exact parity | 旧 legacy one-step context 的 transfer 证明 |
| P-UNC projector | point-track space 的机器不变量、SNR 与合成 sanity 通过 | 已有合法 RGB/video renderer |
| E0 CoTracker3 v3 | evaluator-only 输入下的重跑与排序稳定 | 绝对物理 jerk、人工对齐或新模型收益 |
| V5 Base replay provenance/cache audit | 已有 Base/no-future-GT/object-only 候选的可追溯性 | RGB target 已经合法或 rollout 改善 |
| 固定 noise bank、per-pair worst-case gate、manifest/summary | 可复现的失败定位框架 | 用平均值掩盖局部失败 |

## 9. 跨实验得到的工作原则

### 9.1 先问 target 是否存在，再问模型能否拟合

P1 表明最危险的顺序是先训练，再用 loss 或视觉指标解释 target。对于任何新的 supervision，至少先验证：

* 要求的连续 correction 是否在最终 RGB/latent 表示中实际可观察；
* source removal、disocclusion、遮挡排序和 identity 是否被明确定义；
* decode/encode 后目标仍然表达同一件事；
* target 的局部性不只是 mask 数值小，而是保持了视频语义。

### 9.2 单-pair 可拟合不是方法成立

高学习率单-pair 可以把 correction 降到 95% 以上，却仍无法控制 mask 外和 frame 0。以后任何 overfit feasibility 都必须同时报告 target、outside、frame-0、per-pair worst case 与 provenance，不得只报告优化曲线。

### 9.3 评价器与 target builder 必须隔离

E0 的价值在于不复用训练 RAFT 的 track、future GT 或 projector output。独立 evaluator 即使通过，也只能验证评价协议本身；它不能倒推出 target 合法，更不能替代因果性训练比较。

### 9.4 人工复核是 gate，不是最后的装饰

P0/E0 的 machine pass 均仍待 12-case review。未完成 review 的正确状态是 awaiting_reviews，而不是“基本通过”。反过来，P1 的 machine hard fail 也不能靠主观 review 覆盖。

### 9.5 保留错误 run 比覆盖它更有价值

C0 v1、P1 v1、E0 v1/v2 都被保留并显式标注问题；修复后使用新 run ID 重跑。这使得之后可以区分“研究假设失败”和“诊断实现错误”，也避免用更漂亮的后验结果覆盖历史。

## 10. 当前禁止与重新开启条件

### 当前明确禁止

* 不继续 current temporal-LoRA endpoint 的 learning-rate、preserve-weight、update-step 或 rank sweep。
* 不训练 feature head、zero-conv/refiner、short-chain、F1-R、F2/F3 或生成器。
* 不扩写或改写 V5 cache，不把 synthetic target 合理性外推为 rollout 收益。
* 不通过扩大 mask、忽略 source、接入大型视频编辑模型，或替换 evaluator provider 来绕过 P1/E0 gate。
* 不用 future GT ego、track、box 或 source-future metadata 构造未条件化 SVD 的正式 target。

### 唯一合理的重新开启路径

这必须是一个新问题，而不是当前路线的隐式补丁：

1. 完成已有 P0/E0 的 human review。它们只能完成相应证据链，不能反转 P1。
2. 提出不依赖大型视频编辑模型的新 counterfactual construction。
3. 在相同的 P1-style legality gate 上，用现有冻结 clip 先验证 decoded trajectory realization、source removal、depth/occlusion order、VAE round-trip 与局部性。
4. 只有新 target 和独立 evaluator 都成立后，重新预注册模型参数化、训练目标、停止条件与预算。
5. 在新的单-pair feasibility 同时满足 correction/locality/frame-0 后，才讨论更大 cache、rollout 或训练。

## 11. 证据索引与阅读顺序

如果只需确认当前决策，先读：

1. docs/AUTORESEARCH_PHASE2_REPORT.md
2. docs/AUTORESEARCH_ROUTE_DECISION.md
3. docs/EXPERIMENTS.md 的 Autoresearch Phase 2 表

如果需要复查某个失败原因，再读对应 run：

| 主题 | 首选证据 |
|---|---|
| V1 调参失败 | /root/autodl-tmp/runs/p2-tune-mini/ |
| future ego 条件不匹配 | /root/autodl-tmp/runs/p2-v2-condition/p2-v2-cond16-s20260712-fff5ccb-97d2d05d/ |
| C/D/E capacity 与单-pair locality | /root/autodl-tmp/runs/p2-v2-pilot/ |
| F0 endpoint | /root/autodl-tmp/runs/autoresearch-f0-endpoint-s20260713-6845411/ |
| F1 feature probe | /root/autodl-tmp/runs/autoresearch-f1-features-s20260713-72ac28c/ |
| C0 parity | /root/autodl-tmp/runs/autoresearch-c0-conditioning-s20260714-v2/ |
| P0 projector | /root/autodl-tmp/runs/autoresearch-p0-projector-s20260714-v1/ |
| P1 target legality | /root/autodl-tmp/runs/autoresearch-p1-target-s20260714-v2/ |
| E0 evaluator | /root/autodl-tmp/runs/autoresearch-e0-evaluator-s20260714-v3/ |

最终状态保持为：

> C0 = pass；P0 = machine pass / awaiting reviews；P1 = fail；E0 = machine pass / awaiting reviews；F1-R = not run。

因此当前没有新的生成器训练结果，也没有任何 rollout-quality 提升声明。
