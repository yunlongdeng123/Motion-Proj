# 历史原始记录 017

> 冻结历史，不是当前执行规则。当前规则见 [scaling law](../../../auto-research_scaling_law.md) 与 [AGENTS](../../../AGENTS.md)。
> 原文件：`docs/RESEARCH_FAILURES.md`；迁移前提交 `abbd431c`。原有相对链接按原目录解释，证据路径原样保留。
> 按索引读取相关小节；旧哈希、过度门控和资源指令均不重新生效。

## V71-F36: M34 identifiable anchor mass fails under normalized Gaussian-energy returns（2026-09-05）

- run=`run://worldsim_v71/WS-V71-M34-PRODUCER-EVIDENTIAL-ANCHOR-AUTHORITY-01/20260905T043000Z__m34-producer-anchor-authority-s71134-r1`；
- evidence=occupied correlation=`0.4624`通过，证明producer evidence能辨识GT；但all/hazard/clear early分别
  `+0.223/+0.177/+0.447pp`，all hit=`-0.753pp`；decisions=`2/4`；
- root cause=当前部署把空间Gaussian energy做沿射线softmax；这个全局归一化不是按深度有序的
  transmittance/first-hit概率，局部mass变化没有return单调性；
- anti-repeat=不调M34 seed/epoch/loss/capacity，不按provenance或UNKNOWN硬删；迁移按bin alpha与前缀
  transmittance构造termination distribution；
- claim impact=M34 checkpoint不进入external；可辨识性修复成立，但amplitude composition被拒绝。
  下一failure ID=`V71-F37`。

## V7.1 M34 pre-registration note — isolate anchor evidence from completion（2026-09-05）

M34只训练anchor authority，M8 geometry/trajectory冻结，children保持unit，避免把anchor矛盾、completion几何与
动态建模再次混合。若性能通过但predicted-vs-GT occupied相关低于0.25，仍视为全局降权捷径而拒绝；任一
冻结decision失败不调epochs/loss/seed，登记`V71-F36`并关闭该family。下一failure ID=`V71-F36`。

## V7.1 M33 outcome note — hard provenance labels remain invalid（2026-09-05）

M33精确物化1004 Actor/297,535 anchors，无工程失败。PROJECT仅占13.75%，但其held-out occupied mass均值
`0.466`高于KEEP的`0.425`；因此PROJECT不是错误标签，KEEP也不是安全标签。监督中58.72% anchor同时含
FREE/OCCUPIED票，后续必须学习连续evidential authority并保留UNKNOWN，禁止按provenance删除或阈值重标。
build-vs-GT occupied相关`0.514`支持继续M34，但不构成性能claim。下一failure ID=`V71-F36`。

## V7.1 M33 pre-registration note — provenance/evidence is a data-interface repair（2026-09-05）

M31/M32把失败定位到anchor输入可辨识性，故M33只补producer-side sidecar，不把held-out监督伪装为输入，
不靠UNKNOWN mask或surface filter制造一致性。若精确1004-Actor replay、anchor顺序或source adapter失败，
登记`V71-F36`；M33没有模型性能decision，sidecar完成也不构成物理改善claim。下一failure ID=`V71-F36`。

## V71-F35: M32 build-geometry-only evidential authority worsens every early-return stratum（2026-09-05）

- run=`run://worldsim_v71/WS-V71-M32-EVIDENTIAL-GAUSSIAN-AUTHORITY-01/20260905T030000Z__m32-evidential-gaussian-authority-s71132-r3`；
- outcome=all/hazard/clear early delta=`+0.3357/+0.3384/+0.3221pp`，all hit=`+0.4869pp`，1/3 decisions；
- optimization=loss与evidential CE正常下降，无NaN/OOM；categorical NLL只改善`0.0031`，非数值失败；
- diagnosis=canonical position/scale/type/Actor-size只足以学习anchor/child全局mass，不包含M31已暴露的source-frame、
  KEEP/PROJECT、temporal/view support或build-ray contradiction evidence，单primitive authority不可辨识；
- prevention=不调evidence loss weight、unknown pseudocount、width、epoch、anchor scale、bins、median、seed；
- next=在corpus producer端保留per-anchor build evidence/provenance，以held-out GT为supervision重新训练；禁止按M31
  heldout contradiction做deployment filter；
- claim impact=连续mass本身不构成物理改善；M32 checkpoint不进入external。下一failure ID=`V71-F36`。

## V71-F34: M32 evidential target mass also retained inference identity（2026-09-05）

- run=`run://worldsim_v71/WS-V71-M32-EVIDENTIAL-GAUSSIAN-AUTHORITY-01/20260905T025000Z__m32-evidential-gaussian-authority-s71132-r2`；
- symptom=first batch在`target_mass * log_softmax`报同一inference-tensor saved-for-backward错误；
- exposure=与r1相同，0 optimizer/history/holdout metric；
- root cause=r1只普通化了model inputs，遗漏单独InferenceMode中构造且需参与weight-gradient公式的soft target；
- recovery=在context外clone `authority_target_masses_t`；PyTorch官方clone边界不变，科学合同完全不变；
- claim impact=同源tensor lifecycle错误，不是独立科学尝试。下一failure ID=`V71-F35`。

## V71-F33: M32 frozen features retained inference-tensor identity（2026-09-05）

- run=`run://worldsim_v71/WS-V71-M32-EVIDENTIAL-GAUSSIAN-AUTHORITY-01/20260905T024000Z__m32-evidential-gaussian-authority-s71132-r1`；
- symptom=first train batch的Linear forward报`Inference tensors cannot be saved for backward`；
- exposure=659 cache Actors/M8 frozen geometry和593 train primitive evidence targets已构造；0 optimizer step、0
  training history、0 holdout metric/decision；
- root cause=geometry/features分配于`torch.inference_mode()`，即使requires-grad=false也不能被autograd保存用于
  trainable weight gradient；
- literature/open-source response=PyTorch官方Gradient Modes与`SavedVariable`源码明确要求在InferenceMode外clone
  后再参与autograd；
- recovery=只clone frozen child center/scale和authority feature；不解冻、不改数值/GT/model/protocol；
- claim impact=纯tensor lifecycle错误，不是科学负结果。下一failure ID=`V71-F34`。

## V7.1 M32 pre-registration note — authority mass cannot move geometry（2026-09-05）

M19允许field decoder补偿geometry，M20允许center/scale吸收ray loss并出现scale inflation；M32将两条路径都冻结。
唯一trainable变量是primitive `FREE/OCCUPIED/UNKNOWN` soft mass，GT primitive evidence CE与GT categorical return CE
共同监督，部署只连续使用occupied mass。不存在argmax、hard UNKNOWN mask、surface filter或binary threshold。
若任一hazard/clear early恶化或all hit下降超过1pp，按一次性合同关闭本family，不做权重/伪计数/容量恢复。
下一failure ID=`V71-F33`。

## V7.1 M31 outcome note — contradiction is mixed, PROJECT is overrepresented（2026-09-05）

Canonical r3完整匹配66/66 Actors；anchors early=`19.579%`。first-return provenance为KEEP
`12,955/19,424=66.70%`、PROJECT `6,469/19,424=33.30%`；PROJECT只占13.36% anchor points，因此
相对点占比过度贡献约2.49倍，但删除PROJECT仍不能解决KEEP多数矛盾。raw-query到KEEP只改善`0.629pp`，
UNKNOWN-query自身early仅`1.290%`；不能把失败归因于UNKNOWN mask。

moving/quasi-static差异仅`+0.369pp`，target-frame ordinal既有早期峰值也有后期小样本上升，不支持单调pose-drift
解释。下一阶段禁止按M31结果做post-hoc provenance/time filter；应把cross-frame GT ray的occupied/free/unknown
状态直接写入连续field supervision，使endpoint authority可学习且UNKNOWN保留。r3无新failure；下一ID仍
=`V71-F33`。

## V71-F32: M31 treated processed-recovery scenes as raw LiDAR scenes（2026-09-05）

- run=`run://worldsim_v71/WS-V71-M31-ANCHOR-CONTRADICTION-ATTRIBUTION-01/20260905T014500Z__m31-anchor-contradiction-attribution-r2`；
- symptom=`scene-0001`进入raw compiler后缺少已清理的`.pcd.bin`，抛`FileNotFoundError`；
- exposure=0 scene compiled、0 Actor geometry/metric row；只读了cache manifest/IDs和nuScenes metadata；
- root cause=r2修复scene locator但未恢复S2的per-scene producer lineage；processed-added Actors来自
  `compile_processed_scene`，不是raw `compile_source_scene`；
- literature/open-source response=MLflow dataset lineage要求把实际dataset source连接到run input；r3直接复用S2
  canonical recovery config与adapter，不下载/重建另一份数据，也不加入hash/checksum/fingerprint；
- recovery=manifest中processed scenes走DriveStudio processed adapter，其余走raw adapter；Actor/geometry/compiler/
  metric全部保持；
- claim impact=纯入口失败，不是科学负结果。下一failure ID=`V71-F33`。

## V71-F31: M31 source index omitted processed-recovery scenes（2026-09-05）

- run=`run://worldsim_v71/WS-V71-M31-ANCHOR-CONTRADICTION-ATTRIBUTION-01/20260905T013000Z__m31-anchor-contradiction-attribution-r1`；
- symptom=首个selected `scene-0001`在`compile_source_scene`入口抛`KeyError`；
- exposure=0 scene compiled、0 Actor geometry row、0 literal first-return metric；只有缓存Actor ID/scene metadata被读取；
- root cause=M31沿用了raw S2 builder的train+reserve index，但当前corpus manifest是
  `train_corpus_target_met_with_processed_recovery`；66个holdout横跨63 scenes，其中20个scene来自processed recovery，
  不在旧raw split；
- literature/open-source response=MLflow/Kedro均将实际dataset source/catalog作为run input lineage；本项目采用现有
  cache Actor的scene names作为只读locator，不引入dataset hash、checksum或fingerprint；
- recovery=r2按exact selected cache scenes构造index；不替换Actor，不改P2 compiler、tolerance、strata或metric；
- claim impact=纯入口失败，不是科学负结果。下一failure ID=`V71-F32`。

## V7.1 M31 pre-registration note — provenance diagnosis is not post-hoc repair（2026-09-05）

M30已证明learned completion之前的immutable anchors也有`19.58%` early contradiction。M31只恢复S2编译器
已有的query/KEEP/PROJECT provenance，并用完全相同的literal ray operator归因；不会按结果删点、改label、调
tolerance或生成checkpoint。SCPNet指出多帧moving traces可污染completion label，DualAD强调显式ego/object motion
compensation，但两者在本阶段只支持“先定位supervision producer”的决策。若后续修正，必须在GT/supervision端
独立冻结并经过训练，禁止包装为推理后处理。下一failure ID=`V71-F31`。

## V7.1 M30 outcome note — interval semantics supported, utility sharply bounded（2026-09-05）

Canonical=`run://worldsim_v71/WS-V71-M30-EVIDENTIAL-RETURN-INTERVAL-01/
20260904T203000Z__m30-evidential-return-interval-r1`。99,208 rays上set-order violation=`0`，因此
`d_possible <= d_known`的结构实现正确。但GT empirical bracketing仅`56.69%`，`47.03%`射线
的known upper endpoint为无穷，有限区间Actor-mean q90=`0.326m`（hazard=`0.343m`）。

更重要的负边界是immutable observed anchors本身仍产生`19.58%` early returns，表明矛盾早于
completion children：canonical multi-frame fusion、pose/box annotation、rigid-shape假设或visibility可能已将build
endpoint置于held-out free-before-hit。因此M30只支持“保留epistemic interval”，不支持紧致
collision/safety certificate。无新工程failure ID；下一ID仍`V71-F31`。

## V7.1 M30 pre-registration note — interval coverage assumption is not a certificate（2026-09-05）

M30以`S_known subset S_possible`的代码构造确保`d_possible <= d_known`，但
`S_true subset S_possible`是待检验的completion coverage assumption，不能从集序数学自动得出。
因此empirical target bracketing、finite interval与width只是utility边界；禁止声称deterministic collision/safety
certificate。M30不使用M29 rejected checkpoint，不硬删UNKNOWN，不扫区间阈值。下一failure ID=`V71-F31`。

## V7.1 M29 outcome note — symmetric GT tail objective rejected（2026-09-05）

Canonical=`run://worldsim_v71/WS-V71-M29-GT-TAIL-SURFACE-TUBE-01/
20260904T200000Z__m29-gt-tail-surface-tube-s71126-r1`。4 epochs正常完成，无NaN/OOM。相对M8：

- target-to-surface top-10% tail改善`0.924mm`，frame-mean改善`0.843mm`，Chamfer改善`0.418mm`；
- surface-to-target top-10% tail恶化`0.362mm`，hazard early rate由`26.3718%`升至`26.6908%`
  （`+0.3190pp`），hit recall微降`0.0222pp`；
- 5项预注册decision仅3项通过；几何与射线梯度冲突占`86.6--90.6%` batches，
  mean pre-projection cosine约`-0.38-- -0.42`。

这是核心科学负结果：对称tail loss仍可通过增加GT coverage而将少量generated surface推入
unsupported/free-before-hit区域，不能同时解决completion与physical false return。按预注册停止该分支，
不扫tail fraction/weight/seed，M29 checkpoint不升格为external candidate。无新工程failure ID；下一ID仍
=`V71-F31`。

## V71-F30: M29 pre-run YAML check command quoting failure（2026-09-05）

- stage=远端pre-run最小验证；0 dataset load、0 optimizer step、0 quality/metric exposure；
- symptom=PowerShell到SSH的嵌套`python -c` quotation被本地shell拆分，远端收到缺失表达式的
  `import`，本地又误将`open(...)`当作cmdlet；
- root cause=命令传输层引号，与M29代码、YAML内容、GPU、数值或数据无关；
- recovery=去掉内嵌Python expression，只运行无二次引号的`python -m py_compile`，已通过；
- unchanged=tail fraction/weight、seed、epochs、lr、data split、decision、run ID均未变；
- next failure ID=`V71-F31`。

## V7.1 M29 pre-registration note — mean geometry不是worst-tail bound（2026-09-05）

M8的mean Chamfer/plane/frame/first-return只约束平均误差，不能推出worst-case clearance边界；
hard Hausdorff又易被单个LiDAR/annotation outlier主导。M29冻结双向最差10%距离均值作为
GT-supervised tail tube，且必须与M8 hazard early/Chamfer/hit同时构成Pareto改善。
禁止把这个可解释尾部目标写成deterministic safety certificate；无论结果如何均不扫参。
预注册时尚未触发failure；随后pre-run引号失败已登记为`V71-F30`，当前下一ID=`V71-F31`。

## V7.1 M28 outcome note — 权限边界已进入代码与论文（2026-09-05）

`authority_contract.py`已以三个兄弟类型和无visual参数的pure physical API实现；
`py_compile`通过。论文Method已补入GT-supervision origin、appearance non-interference proposition、
SE(3) corollary与明确non-claims，完整TinyTeX/latexmk编译成功（15页、2,331,081 bytes）。
没有通过数值post-hoc对比冒充证明，没有读取M21 partial quality，也没有新增训练、阈值或过滤。
未触发新failure；下一failure ID仍为`V71-F30`。

