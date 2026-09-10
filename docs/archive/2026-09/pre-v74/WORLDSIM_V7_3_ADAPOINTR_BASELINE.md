> 历史归档（2026-09-10）：以下为原版本事实与指令快照；当前执行以 docs/RESEARCH_STATUS.md 为准。

# V7.3 AdaPoinTr：完整微调、坐标迁移与表面密度

更新于2026-09-08。官方完整AdaPoinTr已完成两次30轮任务微调；Y-up迁移版r2作为当前正确接口基线，r1保留为坐标接口对照。r2能改善测量表面覆盖，但尚未形成一致的物理优势。固定输出密度诊断进一步表明，匹配预算的采样显著影响字面交点，而简单使用全部输出也会放大自由空间侵入。

## 实现与公平性边界

使用[作者仓库](https://github.com/yuxumin/PoinTr)、[官方模型](https://raw.githubusercontent.com/yuxumin/PoinTr/master/models/AdaPoinTr.py)和[PCN配置](https://raw.githubusercontent.com/yuxumin/PoinTr/master/cfgs/PCN_models/AdaPoinTr.yaml)，复用本机PoinTr_AdaPoinTr_4603257源码及官方epoch353的AdaPoinTr_PCN.pth。完整512查询、16384点输出、384维、6层编码器/8层解码器，32494657个参数全部可训练。没有替换为小型外挂头。

所有原始build LiDAR点进入模型，不按目标或质量裁剪；少于512槽位时重复已有点，不算新增观测。DGCNN精确kNN按查询轴512分块，保留全部keys与邻居数。以只读Actor尺寸最大边做各向同性尺度归一化，不从target估计中心、尺度或旋转。r2按[NormalizeObjectPose](https://github.com/yuxumin/PoinTr/blob/master/datasets/data_transforms.py)和[PCN KITTI脚本](https://github.com/wentaoyuan/pcn/blob/master/test_kitti.py)的车辆接口交换Y/Z：输入(x,y,z)→(x,z,y)，所有输出逆变换后在同一米制Actor坐标监督；轨迹、标定不变。r1使用本项目Z-up轴，是明确的未迁移对照。

当前31场景/25日志、489个metadata/build窗口刚体Actor，371可输入fit对象更新，67可输入dev对象只评价；51空输入对象保留缺失。fit监督使用全轨迹实测点与原始首返回射线，dev后续时刻不优化。稀疏target不是完整表面，故损失为target→匹配曲面覆盖+0.5硬首返回free+0.05软box envelope+0.1 target→coarse覆盖+0.1局部稀疏去噪覆盖，不使用全局预测→稀疏target惩罚。该任务损失不同于官方完整PCN GT损失，不能称精确复现PCN基准。

物理主表在16384点上FPS选择min(build点数,1024)+512中心，使用0.06m间距、每片9顶点8三角形、20个原生近邻PCA法向。FPS/邻域/PCA架每步重算但停止其离散/方向梯度，中心位置接收真实表面梯度；无凸包、opacity或可学习半径。完整原生点集另存，不能把这一转换的全部错误直接归因于原生点预测。

FP32 AdamW lr1e-4、weight decay5e-4、epoch21后lr×0.9、clip norm10、seed7307，4个变长Actor逐个反向累积。两轮各30epoch/11130呈现/2790更新，均保存model/optimizer/scheduler及RNG。不是从头复现官方600epoch训练，单一预算也不意味着穷尽该基线。

| 运行 | 代码 | 实际耗时含初始/最终评价 | GPU allocated峰值 | RSS峰值 |
| --- | --- | ---: | ---: | ---: |
| r1 Z-up | 23981936 | 4926.63s | 1.00658GiB | 2.11761GiB |
| r2 PCN Y-up | acd91103 | 5109.12s | 1.00658GiB | 2.07924GiB |

运行根目录`/root/autodl-tmp/runs/worldsim_v73/WS-V73-M2-ADAPOINTR-01/`；r1为`20260907T221000Z__population-full-track-pcn-s7307-r1`，r2为`20260907T231000Z__population-full-track-pcn-yup-s7307-r2`。两进程均已正常结束。

## 75开发Actor、5独立日志

Actor内按射线/测量点计数汇总，日志内Actor等权，日志间等权。距离是一向target→surface，未知区域没有完整精度/Chamfer定义。8个空输入及23个无自有heldout返回保留；无回波对象的该项为不可评价，不能充当成功。

| 方法 | hit | early | miss | free m | target→surface m | recall@0.2m |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| LiDAR PCA | 21.2448% | 4.3527% | 73.4104% | 0.017708 | 0.304790 | 65.4182% |
| 同full_track LiDAR-only r7 | 31.7549% | 11.8390% | 49.8974% | 0.071575 | 0.209524 | 74.5468% |
| 米制beam free r8 | 31.0028% | 5.5909% | 56.8336% | 0.034320 | 0.229553 | 71.9452% |
| 首事件r9 | 32.4688% | 6.0060% | 54.5686% | 0.038543 | 0.227363 | 72.4665% |
| Ada r1最终 | 16.0514% | 11.3132% | 48.9238% | 0.146560 | 0.097579 | 85.6566% |
| Ada r2初始 | 9.8625% | 2.9657% | 51.9989% | 0.107698 | 0.171134 | 72.8113% |
| Ada r2最终 | 14.2906% | 8.5389% | 54.2423% | 0.122057 | 0.113749 | 84.6032% |

r2相对自身初始化：距离−0.057385m，日志bootstrap95%[−0.086634,−0.037315]m；recall+11.792pp，[+4.380,+18.482]pp；early+5.573pp，[+2.968,+8.172]pp。相对r1：free−0.024503m，[−0.086788,+0.032201]m；hit−1.761pp，[−6.766,+3.705]pp；距离+0.016169m，[+0.000341,+0.035096]m。修正轴接口是必要迁移，不能当作已证明的性能原因。

r2相对r8：hit−16.712pp，[−23.434,−8.407]pp；free+0.087737m，[+0.038663,+0.136812]m；recall+12.658pp，[+2.480,+26.112]pp。相对r9：hit−18.178pp，[−26.080,−8.322]pp；free+0.083514m，[+0.030030,+0.136998]m。网络/辅助损失/更新预算不同，不能把这些对照解释为只改变一种架构机制。

移动子集仅9Actor/2日志：r2 hit4.1164%、early8.0231%、miss76.3141%、free0.057109m、距离0.104197m、recall98.8670%。0.2m邻近覆盖不能代替字面交点，少量日志也不足以支持移动泛化主张。完整逐对象、共有输入、移动分层和配对保存在`docs/autoresearch/worldsim_v73/m2/global/adapointr_r2_analysis.json`；r1同目录单独保留。

## 固定输出密度诊断

针对上述脱节先检索[APSS SIGGRAPH2007](https://cgl.ethz.ch/research/past_projects/apss/)和[2DGS SIGGRAPH2024](https://github.com/hbb1/2d-gaussian-splatting)，采用其区分点集支持、局部表面定义和求交的研究思路；未实现或声称复现这两项方法。没有引入可学习透明度或用target选择支持。

`WS-V73-M2-ADAPOINTR-DENSITY-01/20260908T030000Z__development-native-vs-matched-r1`，code5dfa3948，CPU10.6413s/RSS0.66930GiB，只读所有75旧开发对象最终输出。全部16384点作为同尺度PCA曲面中心，其他几何参数不变；匹配表面成绩复用原结果。CPU BVH与原GPU主表存在数值实现差异，密度本身也显著改变三角面预算，因此只作敏感性分析。

| 表示 | 测量target→几何 m | recall@0.2m | hit | early | miss | free m |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 匹配中心点集 | 0.162732 | 77.7732% | — | — | — | — |
| 原生16384点集 | 0.088114 | 86.9506% | — | — | — | — |
| 匹配PCA表面 | 0.113749 | 84.6032% | 14.2906% | 8.5389% | 54.2423% | 0.122057 |
| 全密度PCA表面 | 0.053369 | 87.9661% | 39.0141% | 39.3882% | 18.2534% | 0.341008 |

全密度−匹配：hit+24.724pp，[+14.343,+36.328]pp；early+30.849pp，[+16.340,+42.835]pp；miss−35.989pp，[−45.407,−25.494]pp；free+0.218951m，[+0.108226,+0.334258]m。移动2日志的全密度hit63.2750%、early34.8396%、miss1.0245%、free0.168564m。采样转换影响真实首交点已得到直接证据，但加密同时覆盖错误前表面；不能把原匹配负结果全归给模型，也不能把加密命中收益宣称为物理成功。若扩大该基线，应让监督覆盖实际最终读出的表面，而非只提高评价密度。

证据`adapointr_density_r1_summary.json`及图`V73_ADAPOINTR_DENSITY.png/pdf`位于上述m2/global目录，绘图只使用保存结果，已检查排版。原生点和匹配曲面全部保留。

## 同背景场景结果与当前决策

`WS-V73-M4-SCENE-COMPOSITION-01/20260908T030000Z__development-adapointr-yup-carved-r6`，code5dfa3948，CPU3.8597s/RSS0.77354GiB。使用同四build雕刻背景和全部固定r2匹配表面，6场景/5日志/每方法416704原始heldout束，无网络重跑或更新。cohort11886束的日志均值：hit11.0410%、early33.6000%、miss36.5152%、free0.951842m。场景按射线和场景聚合，不能直接和Actor均值混用。

相对同背景Ada r1，cohort hit−0.682pp、[−2.892,+1.941]pp，free+0.001128m、[−0.016595,+0.014165]m，未见一致改善。相对r8，hit−11.418pp、[−15.326,−8.430]pp，free+0.042653m、[+0.011555,+0.089395]m。相对PCA，miss−8.009pp、[−14.314,−1.704]pp，但hit−10.874pp、[−21.286,−0.461]pp，free+0.048074m、[+0.011879,+0.098823]m。新全密度表面尚未做全场景组合，不能挪用r6为其结果。

F07坐标接口已落实，历史r1限制保留；F02覆盖/free冲突、F03缺支持与F04场景问题仍在。当前没有据此否定视觉几何适配或切换主任务。主joint及匹配native-only训练继续，Ada控制不取代路线B主假设；整个V7.3尚未完成。
