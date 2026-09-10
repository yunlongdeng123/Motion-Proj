> 历史归档（2026-09-10）：以下为原版本事实与指令快照；当前执行以 docs/RESEARCH_STATUS.md 为准。

# V7.3 CAPA基线适配

## 完整CAPA r2结果（2026-09-08）

r2 `20260907T225000Z__population-build-tta-chunked-s7305-r2` 已完成31窗口×100步=3100更新与489Actor评价。每步官方随机3视图、最终每窗口全部24视图联合推理；官方早期patch LoRA393216参数，每窗口从同一初始化重置，开发窗口也仅使用自己的build TTA。code eb42f835，8277.17s、GPU allocated峰值8.08370GiB、RSS13.74811GiB。R1全anchor分配失败已被保持所有anchors/residuals的分块仿射对齐修复，没有减少最终视图。

75开发Actor/5日志：hit17.8032%、early10.9508%、miss58.4555%、free0.120341m、距离0.203133m、recall76.8964%，空表面8→3，所有分母保留。相对M1原生融合，hit−4.974pp、95% [−13.315,+1.334]pp；free−0.029963m、[−0.061784,+0.011995]m；距离+0.027708m、[+0.000880,+0.075590]m。此任务迁移没有一致优势；不能把其窗口TTA与共享适配预算混写，也不能据此否定CAPA原任务结论。

移动9Actor/2日志：hit11.2711%、early4.8243%、miss69.1382%、free0.037237m、距离0.139751m、recall88.2605%，独立样本量很小。完整/可用输入/移动分层与PCA、原生融合、r9的配对在 `docs/autoresearch/worldsim_v73/m2/global/capa_r2_analysis.json`。最终全场景与新日志评价待主模型选择后统一收口。


## 分块后实际运行进展

CAPA修订r2已实际启动：run `WS-V73-M2-CAPA-01/20260907T225000Z__population-build-tta-chunked-s7305-r2`，codeeb42f835，PID31466，日志 `/root/autodl-tmp/controller_logs/v73_population_capa_r2.log`。原始完整VGGT和393216个LoRA参数成功加载，首窗口已越过原OOM位置进入真实反向/优化，官方step0/10/20/30/40的L1读数为3.8186/1.8611/2.0940/1.4381/1.2993。随机视图子集不同，不能把这五个数当成同样本学习曲线或开发效果。随后首个scene-0015窗口已实际完成全部100步、全24视图联合推理、LoRA/深度保存及其Actor评价；适配与保存267.033s，累计GPU allocated峰值8.08180GiB，现进入scene-0071。首窗口已跨过完整优化/推理路径，其余30窗口仍待完成，不称整个基线完成或所有规模资源问题已解决。

最近GPU进程占用r5 12358MiB、Ada r1 1582MiB、CAPA r2 9880MiB，不能再叠加新GPU作业。当前GPU余量限制属于并发调度；正常训练继续，先等现有作业释放资源，再启动已准备的event单因素对照与Ada轴修订r2，不为关机强行结束正常任务。Ada r1最近epoch17/1567次优化更新，原完整初始化输出已保留；其轴未迁移限制仍按F07报告。


---


## 真实首轮资源失败与保留信息的修订

CAPA r1实际载入了原始本地VGGT及393216个可训练LoRA参数，但首窗口在官方仿射对齐sort处OOM，尚未完成第一个优化step。失败code277c9771、PID30545已退出、峰值allocated8.60038GiB；当时r5约12.07GiB、Ada r1约1.54GiB并发占用，不能据此宣告单作业必须加卡。r1/status.json与traceback完整保留，未影响另两项正常训练。

按要求先检索[CAPA官方对齐](https://github.com/nv-dvl/capa/blob/main/capa/utils/alignment.py)、[MoGe官方分块求解](https://github.com/microsoft/MoGe/blob/main/moge/utils/alignment.py)和[PyTorch显存文档](https://docs.pytorch.org/docs/stable/notes/cuda.html)，定位全锚点×全观测的中间矩阵。已将仿射锚点按128分块，保留官方所有点、所有锚点、同一weighted-median求解和全局scatter_min选择，GPU抽样seed和12000点上限不变，不缩小RGB/视图。一次CPU数值对比（2例×129点、噪声/离群/零权重、234有效锚点、chunk7）scale/shift最大差均0；这是执行优化对比，不代表完整CAPA已成功。代码=`capa_alignment.py`；记录=`capa_chunked_alignment_comparison.json`。修订r2 `20260907T225000Z__population-build-tta-chunked-s7305-r2` 待提交后重新执行实际100步/31窗口。


---


## V7.3 CAPA全窗口实际适配运行登记（2026-09-08）

准备运行 `WS-V73-M2-CAPA-01/20260907T223500Z__population-build-tta-s7305-r1`，沿用已记录的官方100步rank4/alpha8、patch_embed qkv LoRA、每步3/24视图随机采样、最终全24视图联合推理；31个build窗口、完整489 Actor队列。所有744视图已有有效稀疏深度条件，fit/dev均只在各自build输入上窗口内TTA，逐窗口重置；额外时刻仅评价。协议、对齐及native-only边界见 `docs/WORLDSIM_V7_3_CAPA_BASELINE.md`。

已静态核对模块路径：主工程NativeGeometryPyramid只在实例化时加载普通VGGT，而该CAPA入口未实例化它；实际CAPA wrapper优先加载其VGGT_VPT。因此没有发现此前担心的普通VGGT抢先导入问题，不添加清模块之类补丁。上游DINO与跨视图aggregator均有nonreentrant checkpoint路径，现有train()覆盖这些路径；保留全部图像与视图信息。开启官方已有每10步loss日志并记录窗口累计显存峰值，不额外建重复测试。

本登记不代表权重加载或LoRA训练已成功。计划在r8训练/最终评价退出并有显存余量后直接执行真实完整运行，避免并发争抢被误判为单作业资源不足；当前不建立后台自动启动队列。AdaPoinTr r1与主joint r5保持各自配置。F01/F02/F03/F04/F05/F07继续active，F06直接数据配置缓解，下一编号V73-F08。整个V7.3未完成，shutdown=false。

---


## 输入条件构建结果

CAPA build条件桥已在全部31个现有窗口实际执行完成：run `WS-V73-M2-CAPA-DATA-01/20260907T194500Z__surround-build-conditions-r1`，codebbd438fa，CPU构建3.537s。744视图共有2516975个有效稀疏轴向深度像素（fit1993519/dev523456），无零条件视图；该数量按视图计数，同一物理点可投影到多个视图，不是独立3D点数。只保留每像素最近的正测量，未读取额外时刻标签，未重复保存RGB或稠密条件张量。记录=`docs/autoresearch/worldsim_v73/coverage/capa_build_conditions_r1.json`。

这只确认实际数据桥已执行；CAPA权重载入、LoRA优化及物理评价尚未运行，不能记成完成基线。后续运行入口与协议差异见 `docs/WORLDSIM_V7_3_CAPA_BASELINE.md`。完整队列r5 PID18843已到epoch3，LiDAR-only r6 PID20389已到epoch5，二者保持原配置，峰值分别10.196/0.181GiB，磁盘约130GiB可用，无OOM。尚未完成的正常长训练继续运行，不因阶段结束关机。

下一独立数据工作：只在fit日志检查同一已知Actor整个可用轨迹内是否还有额外LiDAR记录，判断能否为主模型与AdaPoinTr提供更充分的真实训练目标。目前r4/r5的额外目标只含短窗口内2个未输入时刻；官方PCN使用完整表面目标，不能把稀疏目标强行解释为完整GT。此检查依据已有build Actor身份、只读轨迹和载荷可用性，不依据模型分数；dev不扩展训练标签，当前两条训练不改输入或目标。发现资源/数据缺口时再先检索官方优秀方案并迁移，不从短窗监督负结果外推整个路线。

failure_ledger_delta=update baseline data preparation + V73-F05；F01/F02/F03/F04/F05继续active，F06直接数据配置缓解，下一编号V73-F07。整个V7.3未完成，shutdown=false。

---

# V7.3 CAPA基线适配

LiDAR-only完整队列r6已启动（code98d9fa32，PID20389，run `20260907T193500Z__population-lidar-only-extra-time-s7304-r6`），通过初始表面评价后进入真实训练。观测query梯度非零，native组为0，峰值0.181GiB；没有视觉前缀或DPT参与。r5 PID18843同时正常运行，GPU进程占用约12.1GiB+r6约0.53GiB，当前无OOM或资源不足。两条run的输入/标签协议保持登记配置，等待完整结果再比较。

CAPA基线桥接代码已准备，模型优化尚未启动/验证。入口=`scripts/evaluate_worldsim_v73_capa.py`，数据桥=`motion_proj/worldsim_v73/capa_inputs.py`。直接调用本地官方CAPAProtocol和VGGT LoRA配置，不复制v72含checksum/固定旧split的包装器；依赖peft0.19.1、omegaconf2.3.1、colorlog6.10.1、huggingface_hub0.36.2及原始VGGT本地权重均已在motionproj环境，不需要新环境、下载或升级Torch。

具体迁移边界：保留原378×672图像和K，build投影深度量化到像素时取最近正测量；不读取额外fit/heldout时刻。采用官方100步/rank4/alpha8/qkv设置；官方实际trainable选择是patch_embed内LoRA，其他注入LoRA冻结，这与主候选可训练DPT不同。官方每步随机取10%帧，本窗口为3帧；最终全24帧联合推理。为了保留反向通路并控制激活，启用上游aggregator已有的nonreentrant checkpointing，不缓存可训练路径最终特征。官方逐图像scale+shift从build测量拟合，作为CAPA协议差异明确记录，不能冒充主方法的窗口共享固定scale。

CAPA每窗口重置后适配，包括dev窗口自身的build输入；其额外时刻始终只评价，不能将这种TTA称为“dev完全不反传的共享模型”。这属于部署时允许的稀疏输入适配，而不是读取dev留出标签调参。输出米制深度按相机曝光时刻与已知Actor轨迹规范化，接同固定大小PCA三角片融合与原始首回波评价；有图像native支持但无Actor build LiDAR时允许native-only表面，完全无支持明确miss。输出密度与观测可用比例单独记录，不把点到面转换结果当成CAPA论文原任务结果。

先执行31个窗口的CPU条件构建，保存计数与时间/视图来源，不把已有RGB再复制到磁盘。模型优化等待当前GPU长训练完成或有足够实测余量后启动，不因人为并发争抢造成OOM而宣称需要加卡。后续按实际单作业资源选择保留信息的执行优化；这不改变用户不以24GB限制研究的要求。CAPA运行仍需原始基座及完整反向；若正常单作业仍不足，再按已授权流程保存、无任务关机并提示加卡。

failure_ledger_delta=update V73-F05（同cohort控制已训练）及baseline preparation；F01/F02/F03/F04/F05仍active，F06直接数据配置缓解但不宣称彻底解决，下一编号V73-F07。整个V7.3持续进行，shutdown=false。

官方来源：[CAPA项目](https://research.nvidia.com/labs/dvl/projects/capa/)。本地官方源码目录：`/root/autodl-tmp/external/worldsim_v72/capa`。
