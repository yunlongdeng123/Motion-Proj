# V7.3 CAPA基线适配

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
