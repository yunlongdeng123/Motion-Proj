# V7.3 跨日志共享几何训练结果

## r3侵入归属与r4启动

额外时刻监督r4已启动并进入真实共享训练：run `WS-V73-M2-GLOBAL-ACTOR-01/20260907T184000Z__surround-extra-time-labels-s7304-r4`，code9829ce58，PID17680，日志 `/root/autodl-tmp/controller_logs/v73_global_extra_time_r4.log`。首个记录阶段原生最终depth梯度非零，额外时刻target点数非零，native候选保留，峰值9.848GiB。30epochs/600更新，训练仍运行，不能加载或改写运行配置，结束后读取summary/最新checkpoint再做统一分析。冻结CPU前缀已按场景共享；DPT每步重新执行。当前无资源不足，不关机。

利用r2/r3已保存的每帧计数和均值完成free归属分解，没有重跑模型或射线评价。开发5日志等权mean free中，r2本Actor贡献0.02249m、非本Actor贡献0.10208m（后者占81.95%）；r3分别0.01216m、0.13427m（后者91.69%）。总侵入频率从6.24%下降至3.28%，但总侵入距离从0.12457m增至0.14643m。因此，当前代理降低事件频率，却没有同时降低非本Actor首回波之前的误表面侵入严重程度。分解对象是同一真实原始束的唯一box归属代理；“非本Actor”包括其他Actor、背景、未归属和歧义，不等于已知纯背景。不能将该比例称为背景真值。

归属结果与脚本：`docs/autoresearch/worldsim_v73/m2/global/r2_free_attribution.json`、`r3_free_attribution.json`，`scripts/attribute_worldsim_v73_free_intrusion.py`。按每帧真实束数量恢复侵入总量、扣除本Actor量，再在Actor/日志汇总；只有浮点舍入级负差截为0。F02进一步定位为跨归属真实束一致性问题；F04场景拼接仍待真实背景参与后的全局硬排序检验，本分析不替代场景级实验。

后续顺序：收口r4的fit训练误差与dev留出结果；在相同fit标签预算上推进完整窗口438可输入Actor的共享训练及LiDAR-only/pointwise强控制，51无输入对象保留缺失评价，不再以25Actor诊断代替主结果；认真比较原生几何适配+统一融合。再在可恢复几何的架构上推进event缺支持处理与背景/Actor组合，最终使用新日志。完整cohort仍只有20fit/5dev独立日志。F01/F02/F03/F04/F05 active，F06在当前直接数据监督配置缓解，下一编号V73-F07。按计划持续研究并及时push；只在整个V7.3完成或确有不可解决的资源不足时履行已授权的无任务关机。

---

# V7.3 跨日志共享几何训练结果

## r3：早交点减少，侵入严重程度未改善

射线管r3已完成：run `20260907T175000Z__surround-native-beam-tube-s7304-r3`，code075f32bb，30epochs/600更新，1220.53s，峰值9.864GiB、RSS26.147GiB。相对r2只将range free换成固定3cm/32像素射线管几何覆盖代理，权重0.5不变；两种目标单位不同，该比较检验此具体配置，不能声称最优权重下全面优越。训练fallback 0/600，原生最终depth非零梯度600/600，最终25/25有native候选。

开发5日志：hit30.20%、early6.98%、miss34.13%、free0.14643m、观测表面距离0.08391m、0.2m召回93.29%。r2对应26.78%/17.35%/28.55%/0.12457m/0.07421m/91.64%。早交点频率下降，但全原始束平均侵入加重且miss上升，不能宣称风险F02解决。fit20日志free0.41012m（r2 0.56707m）；训练本身也未达到LiDAR PCA的一致性。r3开发free仍远高于LiDAR PCA 0.00429m。未做严重侵入归属诊断前，不将其原因预先定为背景/某查询子集。原始结果与按日志配对区间保存至 `docs/autoresearch/worldsim_v73/m2/global/r3_*.json`。

完整窗口刚体枚举完成：run `WS-V73-M2-GLOBAL-DATA-01/20260907T180000Z__window-rigid-population-r2`，code9539f000。489个已知轨迹与build时刻有交集的刚体Actor中，438个有build LiDAR输入，51个无build LiDAR，visual-only初始化尚未接入，显式记录输入缺失。fit414/ready371，dev75/ready67；ready但零实际Actor相机时刻分别14/7；ready但无相机-LiDAR投影对应分别25/9。缺相机对象保留LiDAR路径；无输入对象在后续评价中输出空表面，纳入miss/覆盖分母，距离记不可用。该枚举仍是已知轨迹窗口范围，不是真实可见Actor全集。独立日志仍20fit/5dev，不能靠增加Actor数冒充新日志确认。完整index和分项计数已入Git。

下一实验r4：`WS-V73-M2-GLOBAL-ACTOR-01/20260907T184000Z__surround-extra-time-labels-s7304-r4`，保持原25Actor控制队列/M1r3初始化/seed7304/30epochs/600更新/原生数据weight1/range free0.5。相对r2仅扩大fit几何标签：输入仍为4个build时刻的LiDAR和24视图，只在fit的surface coverage与free损失加入窗口内额外时刻真实测量。原生2D depth损失、尺度估计、query种子与视觉读取都仍只用build输入。dev不反传，dev额外时刻仍评价专用。合并点标签做坐标去重；默认build模式保持原输入点顺序与采样，避免无意改变既有实验。

依据先前核对的[AdaPoinTr/PCN官方部分输入与目标分离接口](https://raw.githubusercontent.com/yuxumin/PoinTr/master/datasets/PCNDataset.py)，当前只对输入点做coverage监督不足以充分检验补全学习。r4是监督范围机制比较，不是相对较少训练标签基线的公平最终胜利。fit旧字段heldout_time仅表示未输入，r4将它标记为training_labels，不能再称训练集留出泛化；dev才保留evaluation_only。后续强控制需要相同fit标签预算。暂不同时更改query数量、event或free代理，避免无法归因。

当前r3和数据准备已结束，无GPU训练/数据任务；r4提交后启动。资源足够，shutdown=false，整个V7.3未完成。failure_ledger_delta=update V73-F02/F05；F01资源上界与F03缺支持event/F04场景拼接仍active，F06只在当前直接数据监督配置缓解，下一编号V73-F07。继续研究；全流程完成后保存/push并确认无训练、评价、数据任务及启动队列，才关闭远端。

| 开发日志指标 | r2 | r3 | r3−r2 日志bootstrap95%区间 |
|---|---:|---:|---|
| hit_rate | 0.26784 | 0.30202 | [-0.01204, 0.09454] |
| early_rate | 0.17347 | 0.06981 | [-0.17821, -0.03699] |
| miss_rate | 0.28547 | 0.34129 | [-0.02818, 0.13921] |
| free_intrusion_m | 0.12457 | 0.14643 | [-0.04850, 0.12314] |
| surface_distance_m | 0.07421 | 0.08391 | [-0.00960, 0.03427] |
| surface_recall_02 | 0.91643 | 0.93289 | [-0.03594, 0.06884] |

---

# V7.3 跨日志共享几何训练结果

## r2：直接测量恢复原生通路

共享原生数据监督r2已完成（code37bda7d4，run `20260907T171500Z__surround-shared-native-data-s7304-r2`），30epochs/600更新，1259.42s，峰值9.848GiB、RSS26.152GiB，native project最大变化0.001115。与r1同初始头/seed/数据/架构，仅增加build Actor轴向depth Huber。

V73-F06在此配置下缓解：训练fallback=2/600，原生最终depth非零梯度=600/600，最终25/25 Actor均保留native候选（r1最终0/25）。这是恢复数据梯度的直接证据，不是物理指标胜利；F03关于event缺失支持的风险仍未解除。

开发5日志最终hit26.78%/early17.35%/miss28.55%/free0.12457m，观测表面距离0.0742m；r1为hit36.20%/early15.68%/miss42.15%/free0.03028m。相较原生+LiDAR融合hit27.22%/early8.37%/free0.05903m，也没有物理指标支配。fit20日志free0.56707m，训练集本身仍有侵入，不能只归因为跨日志泛化。恢复native支持保住了更多覆盖，也重新暴露了过剩表面；F02继续active。

原始结果、按日志配对区间及r2−r1比较：`docs/autoresearch/worldsim_v73/m2/global/r2_summary.json`、`r2_log_analysis.json`、`r2_minus_r1.json`；报告更新=`docs/WORLDSIM_V7_3_M2_GLOBAL_RESULTS.md`。没有读取source/external测试，dev只有2个运动日志的边界不变。

下一机制问题：当前硬首交点range free只提供支持内交点位置梯度，缺少退出射线管的轮廓梯度。已先查[nvdiffrast官方文档/SIGGRAPH Asia 2020](https://nvlabs.github.io/nvdiffrast/)：栅格化本身不产生可见性位置梯度，antialiasing负责轮廓梯度。准备在同一三角表面上构造有限宽度的已观测free射线管覆盖代理，并保持原硬首交点评价；不加入opacity/existence，不以关闭native通路降loss，不增加新的世界表示。

迁移范围：每条真实束以已知原始首回波−0.2m限定free终点，32×32局部正交投影、固定3cm训练宽度、有限3σ范围，按几何覆盖积分获得轮廓梯度；宽度是优化代理，不声称已标定真实光束。先做一次前后表面/横向梯度的解析机制实验，识别裁剪/有限支持限制，再进入同架构比较。当前Torch2.4.1+cu121，复用保留的v72-pointr CUDA12.1编译器；安装官方nvdiffrast v0.3.3和ninja，不升级Torch、不新建环境。

当前无训练任务，渲染依赖准备进行中；资源充足，整个V7.3尚未完成，shutdown=false。failure_ledger_delta=mitigate V73-F06 + update V73-F02，F01:F05仍active，F06仅在r2测量监督范围缓解，下一编号V73-F07。按用户授权继续研究，完成全流程后才无任务关机。


| 开发日志指标 | r1 | r2 | r2−r1 日志bootstrap95%区间 |
|---|---:|---:|---|
| hit_rate | 0.36200 | 0.26784 | [-0.17778, -0.01095] |
| early_rate | 0.15680 | 0.17347 | [-0.03846, 0.07179] |
| miss_rate | 0.42153 | 0.28547 | [-0.57005, 0.11262] |
| free_intrusion_m | 0.03028 | 0.12457 | [-0.00128, 0.27456] |
| surface_distance_m | 0.09037 | 0.07421 | [-0.04555, 0.00693] |
| surface_recall_02 | 0.90505 | 0.91643 | [-0.04231, 0.05944] |

---

# V7.3 跨日志共享几何训练结果

共享r1已完成，code8e175195，run `20260907T165500Z__surround-shared-native-s7304-r1`，30epochs/600更新，1140.09s，峰值9.825GiB、RSS26.102GiB，native project最大变化0.000407。

开发5日志：最终hit36.20%/early15.68%/miss42.15%/free0.03028m/表面距离0.0904m；LiDAR PCA hit28.55%/early4.74%/miss66.38%/free0.00429m/表面距离0.1234m；同数量原生+LiDAR融合hit27.22%/early8.37%/miss55.52%/free0.05903m/表面距离0.0809m。fit20日志最终hit40.88%/early18.15%/miss37.33%/free0.04220m/表面距离0.0967m。全部按未输入时刻的真实原始束/正点统计，先Actor内计数加权、后日志等权。

F06已形成完整负结果：训练548/600次LiDAR fallback，原生depth最终输出有非零梯度的更新48/600次；最终fit fallback=20/20，dev fallback=5/5，初始0/25。loss下降不能解释成原生depth表面适配成功，也不能把features路径说成完全无梯度。当前受支持筛选影响，无法由该run确立视觉/空间交互的净优势；继续缺失支持恢复实验，不切主任务。

完整报告=`docs/WORLDSIM_V7_3_M2_GLOBAL_RESULTS.md`；原始结果和独立日志配对区间=`docs/autoresearch/worldsim_v73/m2/global/r1_summary.json`、`r1_log_analysis.json`；训练曲线和PDF保存在原run的`figures/`。LiDAR PCA与生成方法表面密度不同，native融合与生成方法数量相同；尚需同一查询LiDAR-only/逐点以及更强补全和融合比较，不能宣称主表胜利。

下一run `20260907T171500Z__surround-shared-native-data-s7304-r2` 将从同一M1 r3初始头/seed7304训练，保持30epoch和原设置，仅将`--native-data-weight`由0改1（已提交）。每个可用build Actor投影观测直接约束原生轴向depth Huber beta0.2；投影计数不是独立LiDAR样本量。其梯度不依赖当前box内候选存在，开发日志仍只评价；同源真实LiDAR输入预算不变。r1已无训练进程，r2提交后启动；没有资源不足，shutdown=false。failure_ledger_delta=update V73-F06/F02，F01:F06仍active，下一编号V73-F07。


| 分组/方法 | hit | early | miss | free (m) | 可观测表面距离 (m) | recall@0.2m |
|---|---:|---:|---:|---:|---:|---:|
| fit/lidar_pca | 0.2298 | 0.0638 | 0.6791 | 0.01265 | 0.1264 | 0.8217 |
| fit/native_lidar_fusion | 0.2192 | 0.1662 | 0.5111 | 0.42169 | 0.0811 | 0.9390 |
| fit/initial | 0.2732 | 0.2797 | 0.3466 | 0.89025 | 0.0796 | 0.9396 |
| fit/final | 0.4088 | 0.1815 | 0.3733 | 0.04220 | 0.0967 | 0.8695 |
| development/lidar_pca | 0.2855 | 0.0474 | 0.6638 | 0.00429 | 0.1234 | 0.8550 |
| development/native_lidar_fusion | 0.2722 | 0.0837 | 0.5552 | 0.05903 | 0.0809 | 0.9417 |
| development/initial | 0.4087 | 0.1377 | 0.3187 | 0.23701 | 0.0758 | 0.9382 |
| development/final | 0.3620 | 0.1568 | 0.4215 | 0.03028 | 0.0904 | 0.9051 |

开发日志相对同数量native+LiDAR融合的配对变化（最终−基线）：

| 指标 | 均值变化 | 日志bootstrap95%区间 | 改善日志 |
|---|---:|---|---:|
| hit_rate | 0.08978 | [0.03850, 0.14234] | 4/5 |
| early_rate | 0.07312 | [0.00043, 0.14581] | 1/5 |
| miss_rate | -0.13371 | [-0.19021, -0.06154] | 4/5 |
| free_intrusion_m | -0.02875 | [-0.12068, 0.02406] | 1/5 |
| surface_distance_m | 0.00946 | [-0.02173, 0.04655] | 3/5 |
| surface_recall_02 | -0.03661 | [-0.05901, -0.01421] | 0/5 |

已选Actor中fit有16个速度>2m/s，dev只有2个；5个开发日志不是全新来源确认。点归属为注释box代理、时间为scan级，缺少逐点实例/时间真值。表面距离和recall只针对可观测正点，不能当作完整表面Chamfer/F-score。

原生最终depth无梯度不等于全部视觉特征无梯度。图中按每epoch的fit更新比例统计，不是epoch末全Actor重新评价的支持覆盖率。
