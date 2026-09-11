# 历史原始记录 011

> 冻结历史，不是当前执行规则。当前规则见 [scaling law](../../../auto-research_scaling_law.md) 与 [AGENTS](../../../AGENTS.md)。
> 原文件：`docs/RESEARCH_FAILURES.md`；迁移前提交 `abbd431c`。原有相对链接按原目录解释，证据路径原样保留。
> 按索引读取相关小节；旧哈希、过度门控和资源指令均不重新生效。

## V7.3 米制射线管解析结果与完整队列进度（2026-09-08）

米制射线管解析r2完成：run `WS-V73-M3-FREE-VISIBILITY-01/20260907T191000Z__analytic-severity-r2`，codee1d328f8，1.031s，峰值0.0002GiB。5m平面片的几何coverage同为0.839767时，首回波20m得到tube intrusion12.428555m，首回波6m得到0.671814m，分别符合coverage×14.8m和coverage×0.8m；横向平移梯度分别−86.7176和−4.68744，轴向梯度均−0.839767。原硬中心束只有轴向梯度约−1。重复相同表面结果相同；首回波前无冲突及管外时均为0。这说明同一几何代理在该解析配置中保留侵入严重度和轮廓梯度，不证明真实数据提升。

证据=`docs/autoresearch/worldsim_v73/m3/free_severity_analytic_r2.json`；设计更新=`docs/WORLDSIM_V7_3_FREE_VISIBILITY_DESIGN.md`。无支持区域仍无此项梯度，必须依靠几何coverage/native数据通路恢复支持，不能把它写成已解决F03或首事件似然。3cm有限宽度、离散像素和局部AA仍有代理偏差，未来只在固定架构/标签下单独比较。

完整队列r5 PID18843正常推进：code8831def5，已完成489对象的LiDAR PCA初始基线（`lidar_baseline.json`已写出），正在原生联合模型的训练前评价；共享744冻结前缀，观测RSS33.4GiB。run `20260907T190000Z__population-shared-native-extra-time-s7304-r5`，日志 `/root/autodl-tmp/controller_logs/v73_population_joint_r5.log`。它仍是已登记的range free、371fit Actor、30epochs/11130更新；本轮新米制射线管代码不会修改它已加载的程序。等待其实际训练/summary，正常作业不重复启动、不提前终止。完整队列训练后再用同信息pointwise/LiDAR-only与原生融合比较；场景组合和新日志确认仍未完成。

failure_ledger_delta=update V73-F02（解析代理证据，不是实测缓解）；F01/F03/F04/F05仍active，F06仅当前直接数据配置缓解，下一编号V73-F07。没有资源不足或不可抗力，shutdown=false，整个研究继续。

---

## V7.3 完整队列运行与米制射线管候选（2026-09-08）

完整队列r5已启动：code8831def5，PID18843，run `WS-V73-M2-GLOBAL-ACTOR-01/20260907T190000Z__population-shared-native-extra-time-s7304-r5`，日志 `/root/autodl-tmp/controller_logs/v73_population_joint_r5.log`。载入完成，371fit/67dev可输入Actor、51无输入对象均在队列，744共享冻结视图前缀，观测RSS约32.9GiB；正在真实训练前表面评价。保持已登记的range free/额外fit时刻标签/30epoch配置不变。

R1–R4开发权衡图已整理为 `docs/autoresearch/worldsim_v73/m2/global/mechanism_tradeoffs.png` 与 `.pdf`：六个核心读数同时列出，明确5日志范围、R1支持塌缩和R4增加训练标签预算。图仅汇总已有数值，未重跑评价；原始表格与日志配对区间继续保留。当前不能把LiDAR PCA的低侵入与高缺失分开选择性比较。

训练运行期间准备一个针对已定位F02的独立解析机制实验：`WS-V73-M3-FREE-VISIBILITY-01/20260907T191000Z__analytic-severity-r2`。r3几何覆盖代理不区分浅/深侵入，已有真实结果表现为侵入次数下降而总距离增加。先查[nvdiffrast官方插值与antialias接口](https://nvlabs.github.io/nvdiffrast/)：原始raster z/w不传播位置梯度，须显式插值米制顶点距离；antialias为轮廓提供局部位置梯度。因此候选将每个像素的二值覆盖乘以真实首回波前的侵入米数，再做同一个固定Gaussian footprint积分，即 mean_ray sum_pixel w(pixel) * 1[几何首面存在] * max(d_observed - 0.2 - d_surface, 0)。权重分母固定，不随点/面数量变化；像素距离来自同一显式首表面，无opacity/existence。

新增可选 `beam_tube_range`（单位m），原 `beam_tube` 保持覆盖比例、`range` 保持字面中心束距离，r5不切换。解析实验用同一个5m平面片比较首回波6m与20m，检验损失是否按侵入严重度缩放，以及横向/轴向位置梯度是否同时存在；同时记录无free冲突、管外与重复表面情形。它只回答具体梯度语义，不预设真实数据收益，也不解决射线管完全缺支持、裁剪/可见性切换或全局正确性。固定3cm仍为优化宽度，不是标定激光束；偏移像素不作为新增传感器真值。是否进入全队列训练，待r5结果后决定。

仅此一次解析实验与正常r5不冲突；不重复smoke/回归，不新增hash/指纹。failure_ledger_delta=update V73-F02 candidate，F03缺支持event与F04场景拼接仍未实现验证，F05新日志确认待办，F06直接深度约束配置缓解，下一编号V73-F07。资源充足，shutdown=false，持续研究。

---

## V7.3 额外时刻结果与完整Actor队列训练（2026-09-08）

额外fit时刻监督r4完成：run `20260907T184000Z__surround-extra-time-labels-s7304-r4`，code9829ce58，30epochs/600更新，1209.15s，峰值9.848GiB、RSS26.171GiB，原生project最大变化0.000998。训练native fallback4/600、原生最终depth梯度非零600/600，最终25/25保留native支持。该轮输入不变，仅增加fit窗口额外时刻的真实surface/free标签。数据边界已写入manifest、评价行和汇总。

开发5日志结果：hit33.35%、early14.63%、miss36.69%、free0.14904m、观测表面距离0.07685m、0.2m召回93.28%。相对r2（26.78%/17.35%/28.55%/0.12457m/0.07421m/91.64%），命中和召回上升，early下降，但miss和侵入距离上升，无全面胜利。fit额外时刻已作为训练标签：fit距离0.06496m、召回95.15%、free0.84021m，不能把这组fit指标称为留出泛化；侵入在训练测量上仍明显。r4开发free中87.78%来自非本Actor首回波束的贡献，仍非纯背景真值。证据=`docs/autoresearch/worldsim_v73/m2/global/r4_*.json`，含r4−r2日志配对bootstrap区间。只有5个dev日志、其中2个运动Actor，不能据此宣布几何路线失败。

下一正式扩大训练覆盖：`WS-V73-M2-GLOBAL-ACTOR-01/20260907T190000Z__population-shared-native-extra-time-s7304-r5`。数据=`WS-V73-M2-GLOBAL-DATA-01/20260907T180000Z__window-rigid-population-r2`，所有489已知窗口刚体对象均保留；371个fit可输入Actor训练，67个dev可输入Actor只评价，43fit/8dev无build LiDAR对象输出缺失并纳入适用评价分母。317/371 fit有额外时刻正回波；fit去重build点总数194680，额外时刻原始正返回96301（后者非唯一点数）。默认图像/时间窗口与模型容量不削减，冻结前缀共享，DPT逐步重算，缺相机对象保留LiDAR路径。

固定配置：从M1r3 DPT重新初始化，query seed7304，joint/native_surface，native_data_weight1、range free0.5、fit_label_times=all_window，lr1e-5，30epochs=11130 Actor更新；不从r4继续，避免原25Actor过度曝光。逐Actor等权训练，指标先Actor内按束/点计数，再按日志等权；fit与dev共31场景但仍20/5独立日志。r5检验数据覆盖与充分训练，不与25Actor结果直接声称配对提升。LiDAR-only同解码器与等容量pointwise控制需使用同一完整cohort、相同fit标签及轮次；原生几何微调+融合和认真训练的补全/稀疏适配强基线仍待完成。当前不增event或新free代理，避免数据/目标同时变化。

新日志只增加轻量参数组梯度统计和fallback原因，不改优化：区分“无可用Actor相机位姿”和“有视图但预测native支持消失”；扩展队列中预期缺相机的LiDAR分支不能记为同一种F06塌缩。共享norm裁剪仍1。已有较大总梯度而几何/free长期冲突，需要判断梯度量级而不能猜；已先核对[GradNorm/ICML2018](https://proceedings.mlr.press/v80/chen18a.html)及[PCGrad作者实现](https://github.com/tianheyu927/PCGrad)，前者处理任务梯度量级，后者处理方向冲突。本次不照搬多任务算法，也不把参数组norm当成逐损失冲突证据；只有后续实际诊断支持时才单独改变优化。

r4已结束，r5提交后启动；磁盘131GiB可用，90GiB cgroup，当前无资源不足。failure_ledger_delta=update V73-F02/F05，F01/F03/F04仍active，F06在直接数据监督配置缓解，下一编号V73-F07。整个V7.3尚未完成，shutdown=false。长训练进行时继续不冲突的强基线/场景组合实现和文献迁移，完成后分析再推进；完成全研究才保存/push并无任务关机。

---

## V7.3 额外时刻训练进行中及侵入归属（2026-09-08）

额外时刻监督r4已启动并进入真实共享训练：run `WS-V73-M2-GLOBAL-ACTOR-01/20260907T184000Z__surround-extra-time-labels-s7304-r4`，code9829ce58，PID17680，日志 `/root/autodl-tmp/controller_logs/v73_global_extra_time_r4.log`。首个记录阶段原生最终depth梯度非零，额外时刻target点数非零，native候选保留，峰值9.848GiB。30epochs/600更新，训练仍运行，不能加载或改写运行配置，结束后读取summary/最新checkpoint再做统一分析。冻结CPU前缀已按场景共享；DPT每步重新执行。当前无资源不足，不关机。

利用r2/r3已保存的每帧计数和均值完成free归属分解，没有重跑模型或射线评价。开发5日志等权mean free中，r2本Actor贡献0.02249m、非本Actor贡献0.10208m（后者占81.95%）；r3分别0.01216m、0.13427m（后者91.69%）。总侵入频率从6.24%下降至3.28%，但总侵入距离从0.12457m增至0.14643m。因此，当前代理降低事件频率，却没有同时降低非本Actor首回波之前的误表面侵入严重程度。分解对象是同一真实原始束的唯一box归属代理；“非本Actor”包括其他Actor、背景、未归属和歧义，不等于已知纯背景。不能将该比例称为背景真值。

归属结果与脚本：`docs/autoresearch/worldsim_v73/m2/global/r2_free_attribution.json`、`r3_free_attribution.json`，`scripts/attribute_worldsim_v73_free_intrusion.py`。按每帧真实束数量恢复侵入总量、扣除本Actor量，再在Actor/日志汇总；只有浮点舍入级负差截为0。F02进一步定位为跨归属真实束一致性问题；F04场景拼接仍待真实背景参与后的全局硬排序检验，本分析不替代场景级实验。

后续顺序：收口r4的fit训练误差与dev留出结果；在相同fit标签预算上推进完整窗口438可输入Actor的共享训练及LiDAR-only/pointwise强控制，51无输入对象保留缺失评价，不再以25Actor诊断代替主结果；认真比较原生几何适配+统一融合。再在可恢复几何的架构上推进event缺支持处理与背景/Actor组合，最终使用新日志。完整cohort仍只有20fit/5dev独立日志。F01/F02/F03/F04/F05 active，F06在当前直接数据监督配置缓解，下一编号V73-F07。按计划持续研究并及时push；只在整个V7.3完成或确有不可解决的资源不足时履行已授权的无任务关机。

---

## V7.3 射线管结果、窗口覆盖与额外时刻监督（2026-09-08）

射线管r3已完成：run `20260907T175000Z__surround-native-beam-tube-s7304-r3`，code075f32bb，30epochs/600更新，1220.53s，峰值9.864GiB、RSS26.147GiB。相对r2只将range free换成固定3cm/32像素射线管几何覆盖代理，权重0.5不变；两种目标单位不同，该比较检验此具体配置，不能声称最优权重下全面优越。训练fallback 0/600，原生最终depth非零梯度600/600，最终25/25有native候选。

开发5日志：hit30.20%、early6.98%、miss34.13%、free0.14643m、观测表面距离0.08391m、0.2m召回93.29%。r2对应26.78%/17.35%/28.55%/0.12457m/0.07421m/91.64%。早交点频率下降，但全原始束平均侵入加重且miss上升，不能宣称风险F02解决。fit20日志free0.41012m（r2 0.56707m）；训练本身也未达到LiDAR PCA的一致性。r3开发free仍远高于LiDAR PCA 0.00429m。未做严重侵入归属诊断前，不将其原因预先定为背景/某查询子集。原始结果与按日志配对区间保存至 `docs/autoresearch/worldsim_v73/m2/global/r3_*.json`。

完整窗口刚体枚举完成：run `WS-V73-M2-GLOBAL-DATA-01/20260907T180000Z__window-rigid-population-r2`，code9539f000。489个已知轨迹与build时刻有交集的刚体Actor中，438个有build LiDAR输入，51个无build LiDAR，visual-only初始化尚未接入，显式记录输入缺失。fit414/ready371，dev75/ready67；ready但零实际Actor相机时刻分别14/7；ready但无相机-LiDAR投影对应分别25/9。缺相机对象保留LiDAR路径；无输入对象在后续评价中输出空表面，纳入miss/覆盖分母，距离记不可用。该枚举仍是已知轨迹窗口范围，不是真实可见Actor全集。独立日志仍20fit/5dev，不能靠增加Actor数冒充新日志确认。完整index和分项计数已入Git。

下一实验r4：`WS-V73-M2-GLOBAL-ACTOR-01/20260907T184000Z__surround-extra-time-labels-s7304-r4`，保持原25Actor控制队列/M1r3初始化/seed7304/30epochs/600更新/原生数据weight1/range free0.5。相对r2仅扩大fit几何标签：输入仍为4个build时刻的LiDAR和24视图，只在fit的surface coverage与free损失加入窗口内额外时刻真实测量。原生2D depth损失、尺度估计、query种子与视觉读取都仍只用build输入。dev不反传，dev额外时刻仍评价专用。合并点标签做坐标去重；默认build模式保持原输入点顺序与采样，避免无意改变既有实验。

依据先前核对的[AdaPoinTr/PCN官方部分输入与目标分离接口](https://raw.githubusercontent.com/yuxumin/PoinTr/master/datasets/PCNDataset.py)，当前只对输入点做coverage监督不足以充分检验补全学习。r4是监督范围机制比较，不是相对较少训练标签基线的公平最终胜利。fit旧字段heldout_time仅表示未输入，r4将它标记为training_labels，不能再称训练集留出泛化；dev才保留evaluation_only。后续强控制需要相同fit标签预算。暂不同时更改query数量、event或free代理，避免无法归因。

当前r3和数据准备已结束，无GPU训练/数据任务；r4提交后启动。资源足够，shutdown=false，整个V7.3未完成。failure_ledger_delta=update V73-F02/F05；F01资源上界与F03缺支持event/F04场景拼接仍active，F06只在当前直接数据监督配置缓解，下一编号V73-F07。继续研究；全流程完成后保存/push并确认无训练、评价、数据任务及启动队列，才关闭远端。

---

## V7.3 扩大窗口Actor覆盖并保留缺观测对象（2026-09-08）

射线管r3 PID15884正常训练（code075f32bb，run `20260907T175000Z__surround-native-beam-tube-s7304-r3`），观测峰值9.864GiB。当前运行配置不改，等待30epoch后统一硬表面评价。

并行准备 `WS-V73-M2-GLOBAL-DATA-01/20260907T180000Z__window-rigid-population-r2`。已有相机投影覆盖清单是346fit/58dev Actor（65/6个速度>2m/s对象），但不能把这404个对象当成所有LiDAR支持对象。新入口按31个既有场景的build LiDAR时刻与只读轨迹，枚举全部已知刚体车辆；依据输入/元数据，不依据预测/heldout质量。仍沿用原20fit/5dev日志，不改曝光身份、不宣称新来源确认。

有LiDAR但没有相机对应/可插值相机时刻的Actor保留LiDAR查询路径。没有build LiDAR点的对象也写入cohort并保留原始束记录，当前visual-only初始化未接入，明确输出缺失；其可评价回波计入miss、正点覆盖计0，空表面距离记不可用，不能记成零误差或静默剔除。相机投影观测数不等于实例可见性真值。单时刻轨迹仅在其准确已知时间使用，不外推运动。

为扩展到同场景多个Actor，同一window/view的冻结aggregator前缀在CPU共享一次；每次优化仍重新运行可训练DPT，绝不缓存跨优化步的DPT最终输出。这样避免按Actor重复持有同一前缀导致内存随Actor数量虚增，不削减相机/时间输入。模型内部引用以(scene,owner)区分上下文，几何更新语义不变。新代码只供后续run，运行中r3已经加载075f32bb，不重启。

已核对[AdaPoinTr/PCN官方数据接口](https://raw.githubusercontent.com/yuxumin/PoinTr/master/datasets/PCNDataset.py)：部分输入与目标点集分开组织。当前共享模型的coverage主要监督输入build点，本身不能充分证明缺失表面学习；后续需要区分fit侧额外传感器标签与dev留出时刻，并按同一信息预算训练强控制。本次只扩展输入覆盖和缓存，不同时修改标签来源，避免与free机制实验混杂。

数据准备提交后启动；资源90GiB cgroup/3090/磁盘约131GiB可用，当前无资源不足，shutdown=false。failure_ledger_delta=update V73-F05（覆盖/选择范围）及F01（共享前缀执行方式），F02仍active，F06仅直接测量配置缓解，下一编号V73-F07。event、背景/Actor统一遮挡与新日志确认仍待推进。全V7.3完成后遵照用户要求保存/push、无任务及启动队列后关机。

---

## V7.3 同曲面射线管覆盖监督（2026-09-08）

`WS-V73-M3-FREE-VISIBILITY-01` 一次解析机制实验已完成（code c6129bcc，run `20260907T174500Z__analytic-r1`）。5m处、横向偏移2.5cm的12cm曲面片遮挡20m返回：硬range侵入14.8m、整体平移梯度[0,0,-1]；有限射线管覆盖0.83977、平移梯度[-5.8593,0,0]。负梯度更新给出退出横向射线管的方向。位于原始首回波后或完全在管外时均为0；重复同一表面覆盖和梯度均不变。证据=`docs/autoresearch/worldsim_v73/m3/free_visibility_analytic_r1.json`。这只说明代理能提供不同局部几何方向，不保证真实数据收益、任意可见性切换或缺失支持出生。

下一受控训练r3：task `WS-V73-M2-GLOBAL-ACTOR-01`，run `20260907T175000Z__surround-native-beam-tube-s7304-r3`。从同一M1 r3初始权重/seed7304、20fit/5dev Actor、30epochs/600更新，保留native-data weight1、coverage及弱envelope，仅把free训练目标从硬range侵入换为同一三角曲面的有限射线管覆盖。free weight0.5、宽度0.03m、32×32分辨率、每步原始512束不变。覆盖比例与米制range是不同单位，不声称权重相同等于梯度强度相同；另外持续记录原始硬free米值，最终统一硬表面评价。

将原始返回−0.2m设为free终点，nvdiffrast栅格化几何占据、antialias提供轮廓位置梯度；相机/轨迹/表面半径/opacity均无新可训练控制。固定3σ范围和高斯足迹是训练代理，不能把横向偏移射线称为新增观测真值。边界裁剪、正好对称或完全离开有限支持仍可能无梯度，宽度在掠射表面上的偏差需结合hard hit/early/miss判断。保留真实depth与coverage的支持恢复路径，不以event替代。

运行依赖：motionproj保留Torch2.4.1+cu121，新增nvdiffrast0.3.3、ninja1.13.2；仅复用`/root/autodl-tmp/envs/worldsim-v72-pointr`的CUDA12.1编译器，库仍在motionproj运行。源码`/root/autodl-tmp/external/worldsim_v73/nvdiffrast` tag v0.3.3，扩展缓存`/root/autodl-tmp/torch_extensions/worldsim_v73`；不升级Torch，不新建环境，不添加校验工件。来源：[官方API/SIGGRAPH Asia 2020](https://nvlabs.github.io/nvdiffrast/)、[兼容版本源码](https://github.com/NVlabs/nvdiffrast/tree/v0.3.3)。

r2已结束，r3提交后启动；当前无资源不足、shutdown=false。failure_ledger_delta=update V73-F02（已得到横向代理梯度，实际收益待测），F01:F05仍active，F06在直接测量路径范围缓解；下一编号V73-F07。完整同信息控制、更多动态独立日志、event和背景组合仍pending，不把这一次解析结果当成V7.3完成。

---

## V7.3 原生梯度恢复，但自由空间冲突仍在（2026-09-08）

共享原生数据监督r2已完成（code37bda7d4，run `20260907T171500Z__surround-shared-native-data-s7304-r2`），30epochs/600更新，1259.42s，峰值9.848GiB、RSS26.152GiB，native project最大变化0.001115。与r1同初始头/seed/数据/架构，仅增加build Actor轴向depth Huber。

V73-F06在此配置下缓解：训练fallback=2/600，原生最终depth非零梯度=600/600，最终25/25 Actor均保留native候选（r1最终0/25）。这是恢复数据梯度的直接证据，不是物理指标胜利；F03关于event缺失支持的风险仍未解除。

开发5日志最终hit26.78%/early17.35%/miss28.55%/free0.12457m，观测表面距离0.0742m；r1为hit36.20%/early15.68%/miss42.15%/free0.03028m。相较原生+LiDAR融合hit27.22%/early8.37%/free0.05903m，也没有物理指标支配。fit20日志free0.56707m，训练集本身仍有侵入，不能只归因为跨日志泛化。恢复native支持保住了更多覆盖，也重新暴露了过剩表面；F02继续active。

原始结果、按日志配对区间及r2−r1比较：`docs/autoresearch/worldsim_v73/m2/global/r2_summary.json`、`r2_log_analysis.json`、`r2_minus_r1.json`；报告更新=`docs/WORLDSIM_V7_3_M2_GLOBAL_RESULTS.md`。没有读取source/external测试，dev只有2个运动日志的边界不变。

下一机制问题：当前硬首交点range free只提供支持内交点位置梯度，缺少退出射线管的轮廓梯度。已先查[nvdiffrast官方文档/SIGGRAPH Asia 2020](https://nvlabs.github.io/nvdiffrast/)：栅格化本身不产生可见性位置梯度，antialiasing负责轮廓梯度。准备在同一三角表面上构造有限宽度的已观测free射线管覆盖代理，并保持原硬首交点评价；不加入opacity/existence，不以关闭native通路降loss，不增加新的世界表示。

迁移范围：每条真实束以已知原始首回波−0.2m限定free终点，32×32局部正交投影、固定3cm训练宽度、有限3σ范围，按几何覆盖积分获得轮廓梯度；宽度是优化代理，不声称已标定真实光束。先做一次前后表面/横向梯度的解析机制实验，识别裁剪/有限支持限制，再进入同架构比较。当前Torch2.4.1+cu121，复用保留的v72-pointr CUDA12.1编译器；安装官方nvdiffrast v0.3.3和ninja，不升级Torch、不新建环境。

当前无训练任务，渲染依赖准备进行中；资源充足，整个V7.3尚未完成，shutdown=false。failure_ledger_delta=mitigate V73-F06 + update V73-F02，F01:F05仍active，F06仅在r2测量监督范围缓解，下一编号V73-F07。按用户授权继续研究，完成全流程后才无任务关机。

---

## V7.3 原生测量梯度恢复实验已启动（2026-09-08）

共享几何恢复实验r2已启动：PID14038，code37bda7d4，task `WS-V73-M2-GLOBAL-ACTOR-01`，run `20260907T171500Z__surround-shared-native-data-s7304-r2`；日志`/root/autodl-tmp/controller_logs/v73_global_native_data_r2.log`。已进入initial_evaluation，随后自动执行20fit Actor×30epochs；5dev日志只评价。训练代码加载M1六相机r3 checkpoint，不继承退化的共享r1权重；保持seed7304和原架构，新增真实build投影depth Huber weight1。

当前r1和融合评价任务均已结束，无其他研究作业/待启动批次；当前唯一研究任务为r2。磁盘约132GiB可用，r1实测9.825GiB GPU/26GiB RSS支持当前完整输入，无资源不足。三本台账已记录r1最终25/25 fallback、548/600次训练fallback及48/600次原生最终depth非零梯度。不要重复启动r2、改它的配置或把initial_evaluation误判卡死。

下一次接续先读r2 status/进程：正常则开展不冲突工作；done后按与r1相同日志/原始束汇总并比较native支持、真实表面和early/free。若仅保住梯度仍不改善物理指标，继续针对已定位机制查优秀论文/官方开源后迁移；不把退化回退写为route B成功。等容量逐点/LiDAR-only控制、更多动态独立日志、event及完整背景组合仍未完成。failure_ledger_delta=none，F01:F06仍active，下一编号V73-F07。

自动跟进worldsim-v7-3已同步计划最新修订（当前revision3），每15分钟接续；全V7.3完成或确实资源不足时，仍按用户授权保存/push、确保无训练/评价/数据任务及启动队列后shutdown并暂停跟进。当前shutdown=false，无需用户确认。

---

## V7.3 首次共享训练完成：原生支持退化（2026-09-08）

共享r1已完成，code8e175195，run `20260907T165500Z__surround-shared-native-s7304-r1`，30epochs/600更新，1140.09s，峰值9.825GiB、RSS26.102GiB，native project最大变化0.000407。

开发5日志：最终hit36.20%/early15.68%/miss42.15%/free0.03028m/表面距离0.0904m；LiDAR PCA hit28.55%/early4.74%/miss66.38%/free0.00429m/表面距离0.1234m；同数量原生+LiDAR融合hit27.22%/early8.37%/miss55.52%/free0.05903m/表面距离0.0809m。fit20日志最终hit40.88%/early18.15%/miss37.33%/free0.04220m/表面距离0.0967m。全部按未输入时刻的真实原始束/正点统计，先Actor内计数加权、后日志等权。

F06已形成完整负结果：训练548/600次LiDAR fallback，原生depth最终输出有非零梯度的更新48/600次；最终fit fallback=20/20，dev fallback=5/5，初始0/25。loss下降不能解释成原生depth表面适配成功，也不能把features路径说成完全无梯度。当前受支持筛选影响，无法由该run确立视觉/空间交互的净优势；继续缺失支持恢复实验，不切主任务。

完整报告=`docs/WORLDSIM_V7_3_M2_GLOBAL_RESULTS.md`；原始结果和独立日志配对区间=`docs/autoresearch/worldsim_v73/m2/global/r1_summary.json`、`r1_log_analysis.json`；训练曲线和PDF保存在原run的`figures/`。LiDAR PCA与生成方法表面密度不同，native融合与生成方法数量相同；尚需同一查询LiDAR-only/逐点以及更强补全和融合比较，不能宣称主表胜利。

下一run `20260907T171500Z__surround-shared-native-data-s7304-r2` 将从同一M1 r3初始头/seed7304训练，保持30epoch和原设置，仅将`--native-data-weight`由0改1（已提交）。每个可用build Actor投影观测直接约束原生轴向depth Huber beta0.2；投影计数不是独立LiDAR样本量。其梯度不依赖当前box内候选存在，开发日志仍只评价；同源真实LiDAR输入预算不变。r1已无训练进程，r2提交后启动；没有资源不足，shutdown=false。failure_ledger_delta=update V73-F06/F02，F01:F06仍active，下一编号V73-F07。

---

## V7.3 同信息融合显示覆盖与侵入权衡（2026-09-08）

原生+LiDAR融合run `20260907T170500Z__native-lidar-fusion-r1` 已完成，code5d58e6fe，25Actor、67.20s，峰值2.687GiB、RSS6.825GiB，native fallback=0/25。fit20日志中16个运动Actor、11个build点数<100；development5日志中2个运动、2个build<100。所有对象均有未输入时刻Actor回波，未按评价质量选择。

开发日志等权：LiDAR PCA hit28.55%/early4.74%/miss66.38%/free0.00429m，观测表面距离0.1234m/recall85.50%；native+LiDAR融合hit27.22%/early8.37%/miss55.52%/free0.05903m，距离0.0809m/recall94.17%。覆盖改善伴随early/free退化，没有形成物理指标支配。融合输出min(N,1024)+512个patch，LiDAR PCA为min(N,1536)；报告密度差异，不能把全部变化归因于图像信息。此融合与共享查询模型的patch数量/固定尺度相同。

报告=`docs/WORLDSIM_V7_3_M2_GLOBAL_FUSION_RESULTS.md`；证据=`docs/autoresearch/worldsim_v73/m2/global/fusion_r1_summary.json`、`fusion_r1_log_analysis.json`。本方法为已训练原生DPT的简单融合参照，不是CAPA/AdaPoinTr/TSDF，也未完成最强同信息基线集合。failure_ledger_delta=update V73-F02（支持增加/侵入权衡）；F01:F06仍active，下一编号V73-F07。

共享r1 PID12315仍在训练30epochs，不能把中间fallback或loss下降写为最终结果；随后运行已提交的native-data r2。当前融合任务已结束，资源未不足；整个研究未结束，因此shutdown=false。

---

## V7.3 同信息原生与LiDAR融合比较（2026-09-08）

`WS-V73-M2-GLOBAL-FUSION-01` 同信息简单融合基线已实现，run `20260907T170500Z__native-lidar-fusion-r1`。复用完成的M1六相机r3头与同一25个Actor，不新增训练、不读取新时间目标选择输出。原生depth→已知轨迹规范坐标→box归属点，与build LiDAR合并；保留min(1024,N) LiDAR证据中心+512原生FPS中心，使用合并点集局部PCA法向及相同0.06m、8三角面/patch读出。与共享模型输出数量一致，原生点完全缺失时保留LiDAR并报告。没有自由空间删除、凸包或opacity。

本基线将补足早期native-only未融合LiDAR的信息差异，仍只是简单融合参考，不冒充CAPA、AdaPoinTr或TSDF。若其表现差，不能称已超过所有强融合。共享r1仍运行，native直接数据监督r2按既定计划随后串行训练；资源足够时融合评价可与r1并行，优先保持训练完整信息。failure_ledger_delta=none，F01:F06仍active；下一编号V73-F07。三本台账和报告随结果完成更新，整个V7.3尚未完成，不关机。

---

## V7.3 共享训练暴露原生支持逃避通道（2026-09-08）

共享跨Actor r1已真实训练（code8e175195，PID12315，run `20260907T165500Z__surround-shared-native-s7304-r1`），20fit/5development日志，输入准备25/25ready。上下文仍为六相机24view；Actor轨迹可插值的实际投影视图7–24，完整报告，未用预测质量剔除对象。资源实测峰值9.825GiB、RSS约26.1GiB，无OOM，无需关机。

新增 **V73-F06：预测支持筛选导致原生通路退化，active**。初始化25/25 Actor均有native表面候选，r1 epoch1/2/3的LiDAR fallback分别2/20、11/20、16/20次；对应depth最终输出非零梯度18/20、9/20、4/20次。证据=`docs/autoresearch/worldsim_v73/m2/global/r1_support_interim.json`。当前还在30epoch运行，不冒充最终结论；features路径仍可能收到梯度，不能写成全部视觉梯度为零。

代码归因：`native_surface_seeds`按当前预测深度反投影后的box归属选点，选空时回退LiDAR。coverage/free仅作用最终曲面；一旦选空，原生最终depth参数不再直接参与表面位置损失。原生支持移出归属域可以伴随free下降，形成无需opacity的隐藏逃避通道。现有证据足以标记机制风险，尚不能区分free、coverage与共享优化各自的因果占比。该风险细化F02/F03，不能外推视觉几何路线失败。

先核对[CAPA原论文](https://arxiv.org/html/2602.14751v1)的稀疏测量直接驱动适配，以及[VGGT官方训练](https://raw.githubusercontent.com/facebookresearch/vggt/main/training/README.md)的原生几何头训练。迁移为一个受控候选：保留相同初始化、数据、架构、lr、seed7304、30epoch和free目标，增加当前build Actor在正确相机像素上的米制轴向Huber(beta0.2，weight1.0)。该损失使用真实build LiDAR、标定与只读轨迹，绕过预测框内选点，因此选空时仍可恢复native depth；不蒸馏旧深度，不加入零位移先验，不改轨迹或放宽归属框。

下一run `20260907T171500Z__surround-shared-native-data-s7304-r2`，参数`--native-data-weight 1`，待r1完成后串行启动。默认weight0保留r1方法可复现。将报告native支持/fallback、depth数据项与输出梯度、同一硬曲面指标；保留无视觉/无对应对象的LiDAR路径。不能以减少fallback代替表面真实改善。最终event和背景组合尚未完成，V7.3继续；failure_ledger_delta=add V73-F06，下一编号V73-F07。

---

