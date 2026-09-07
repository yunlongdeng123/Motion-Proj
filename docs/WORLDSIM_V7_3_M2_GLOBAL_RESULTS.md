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
