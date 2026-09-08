# V7.3 AdaPoinTr强补全控制

## V7.3 AdaPoinTr坐标修订结果、AV2外部输入完成与旧域场景实测（2026-09-08）

AdaPoinTr Y-up r2 `WS-V73-M2-ADAPOINTR-01/20260907T231000Z__population-full-track-pcn-yup-s7307-r2` 已完整结束，code acd91103，30epoch/11130呈现/2790更新，5109.12s，GPU allocated峰值1.00658GiB、RSS2.07924GiB；PID37887已退出。保持r1输入/全轨迹标签/seed/预算，仅迁移官方PCN轴接口。75开发Actor/5日志：hit14.2906%、early8.5389%、miss54.2423%、free0.122057m、target→surface距离0.113749m、recall@0.2m84.6032%。相对自身初始化，距离−0.057385m、日志bootstrap95%[−0.086634,−0.037315]m，recall+11.792pp、[+4.380,+18.482]pp；early+5.573pp、[+2.968,+8.172]pp。相对r1，free−0.024503m、[−0.086788,+0.032201]m，hit−1.761pp、[−6.766,+3.705]pp，距离+0.016169m、[+0.000341,+0.035096]m，不能宣称坐标修复带来一致物理改善。F07接口修订已执行并完成训练，原r1局限保留；物理缺陷归入仍active的F02/F03，不扩大成完整补全路线失败。

相对r8，Ada r2 hit−16.712pp、[−23.434,−8.407]pp，free+0.087737m、[+0.038663,+0.136812]m，recall+12.658pp、[+2.480,+26.112]pp；相对r9，hit−18.178pp、[−26.080,−8.322]pp，free+0.083514m、[+0.030030,+0.136998]m。移动9Actor/2日志hit4.1164%、early8.0231%、miss76.3141%、free0.057109m、距离0.104197m、recall98.8670%，再次说明0.2m邻近覆盖不能代替字面命中。8个原空输入和23个无自有留出返回仍在全体报告；原生16384点与匹配曲面分别保存。完整配对见 `docs/autoresearch/worldsim_v73/m2/global/adapointr_r2_analysis.json`。

针对覆盖/首交点脱节，先查阅[APSS SIGGRAPH2007](https://cgl.ethz.ch/research/past_projects/apss/)与[2DGS SIGGRAPH2024官方实现](https://github.com/hbb1/2d-gaussian-splatting)。前者体现局部拟合对稀疏点集表面定义的作用，后者显式区分有向surfel求交与网格提取；不迁移可学习opacity替错误前景变透明。先登记一次固定输出诊断 `WS-V73-M2-ADAPOINTR-DENSITY-01/20260908T030000Z__development-native-vs-matched-r1`：只读r2最终点集，在全部75旧开发Actor上对比原生点集覆盖、匹配中心覆盖和全部16384点同0.06m/20邻居PCA显式化的硬交点。它改变输出密度，作为敏感性分析而非匹配预算主表，不按结果挑Actor/半径。另登记r2匹配表面的雕刻背景组合 `WS-V73-M4-SCENE-COMPOSITION-01/20260908T030000Z__development-adapointr-yup-carved-r6`。两项当前尚未执行；不预先归因于支持稀疏或法向。

外部数据 `WS-V73-M4-AV2-DATA-01/20260908T020000Z__external20-common-windows-r1` code f70d099c 已完整导出/合并，2061.27s，父RSS3.66828GiB、子最大RSS1.26955GiB，PID41941及子任务已退出。原20日志全部保留，936刚体Actor：878 ready、58 unavailable_input，221移动>2m/s、108无速度；1个无可用Actor相机时刻，119个无自有heldout回波。560个完整相机输入，原始11723402返回=build7815152+heldout3908250，姿态未知返回0、框归属重叠7077。输入/标注值已处理，但未做外部网络推理、模型质量评分、共享优化或按覆盖更换身份。新域最终确认尚未发生，训练仍仅旧fit；索引归档至coverage/av2_external20_index.json。

旧AV2逐束场景三项登记均以code ad457f3c成功执行：数据34.5845s/RSS0.85305GiB，原背景评价11.3142s/RSS0.96797GiB，雕刻背景11.0018s/RSS0.94923GiB。21Actor、187494原始heldout束，cohort5257、移动cohort88、框边界2708，单条历史开发日志，不能用射线量或单日志bootstrap当独立确认。2916480背景三角面中仅按build自由空间删除211100（7.24%）；71498条build冲突束降至0，此结论仅此窗口中央束/当前容差，不能外推所有几何无侵入。

旧AV2仅背景全束free0.733877→0.513045m、early20.5169%→14.1498%、hit44.3774%→48.5205%，miss31.1071%→32.1333%。雕刻背景cohort：PCA hit62.9446%/early9.5872%/miss21.6663%/free0.060124m；native fusion 42.6479%/33.3080%/18.2043%/0.359073m；r9 54.2134%/9.2068%/17.2912%/0.074238m。移动仅88束时PCA hit21.59%/miss73.86%，r9 44.32%/37.50%，native fusion22.73%/64.77%；不从这一旧单日志选外部赢家。背景对cohort的提前遮挡仅约0.38%，与nuScenes约25.19%明显不同，需要分别报告数据与背景误差；F04仍active。详见m4/av2_old_scene_r1_summary.json、av2_old_scene_r2_summary.json、av2_old_scene_data_index.json。

主joint r5恢复PID39009与native-only full_track r11 PID38460仍运行，占GPU约13810/2920MiB；未启动r10。磁盘约116GiB可用，当前没有真实资源不足证据。F01/F02/F03/F04/F05 active，F06直接数据约束缓解，F07坐标接口已落实，F08已恢复但原退出原因未知，下一编号V73-F09。整个V7.3未完成，shutdown=false；继续自动推进，最终必须保存/push并确认训练、评价、数据及启动控制进程全部结束才关机。

---


## V7.3 AdaPoinTr首轮完整结果、event真实启动与独立日志对策（2026-09-08）

AdaPoinTr r1 `WS-V73-M2-ADAPOINTR-01/20260907T221000Z__population-full-track-pcn-s7307-r1` 已完成并正常退出：code23981936，完整模型32494657个可训练参数，30epoch/11130 Actor呈现/2790优化更新，4926.63s含489对象初始/最终评价，GPU allocated峰值1.00658GiB、RSS2.11761GiB。最终75开发对象/5日志：hit16.05%、early11.31%、miss48.92%、free0.14656m、target→surface距离0.09758m、recall@0.2m85.66%。相对自身初始化，距离−0.11974m、日志bootstrap95%[−0.27337,−0.02175]m，recall+18.54pp、[+7.71,+30.93]pp，hit+6.68pp、[+3.16,+10.20]pp；early+6.38pp、[+2.47,+10.82]pp，free+0.03040m、[−0.01929,+0.07727]m。几何覆盖可学习，但不是一致的物理改善。

相对同full_track标签的r7，Ada r1 hit−15.70pp、[−26.16,−2.97]pp；free+0.07499m、[−0.02406,+0.17408]m；相对r8，free+0.11224m、[+0.02594,+0.20383]m，recall+13.71pp、[+2.70,+27.37]pp。移动开发9对象/2日志的hit7.14%、early4.39%、miss18.56%、free0.08765m、距离0.15679m、recall45.61%，大量返回较晚，不能只看miss降低。8个空输入开发对象及23个无自有留出回波均保留。全体、共有输入、移动分层和配对见m2/global/adapointr_r1_analysis.json。r1未迁移PCN Y-up坐标的F07限制保留，修订r2尚未运行；不能仅凭r1否定完整补全基线。


---


## V7.3 AdaPoinTr官方坐标接口修订（2026-09-08）

V73-F07：首轮AdaPoinTr r1使用本项目Z-up Actor轴输入PCN预训练模型，尚未迁移其车辆坐标约定。依据适配前物理误差继续查阅官方源码，确认[PoinTr NormalizeObjectPose](https://github.com/yuxumin/PoinTr/blob/master/datasets/data_transforms.py)及[PCN test_kitti](https://github.com/wentaoyuan/pcn/blob/master/test_kitti.py)在车辆规范化后显式交换Y/Z，输出再逆变换。该证据说明现有预训练接口有可修正的域差异，但并不单独证明r1全部误差由此造成。

已为后续训练加入input_frame=pcn_y_up，输入(x,y,z)→(x,z,y)，所有coarse/fine/denoised输出变回Actor米制坐标后计算同一损失与表面。保持已知box最大维度的各向同性尺度、原输入/全轨迹fit标签/seed/网络/预算不变；不依据target拟合轴或尺度。矩阵仅定义模型坐标接口，不修改Actor轨迹和相机标定。保留input_frame=actor用于明确复现原r1。

r1 PID28663继续原已加载代码，记录不覆盖，作为未迁移预训练轴的对照；不把其最终结果独自作为认真迁移的强基线。登记修订版 `WS-V73-M2-ADAPOINTR-01/20260907T231000Z__population-full-track-pcn-yup-s7307-r2`，与r1相同30epoch/全参数微调，待当前Ada作业完成后启动，不设后台自动重启或重复队列。r2当前仅实现/登记，未训练；F07状态active，下一failure编号V73-F08。F01/F02/F03/F04/F05仍active，F06直接数据配置缓解。主joint r5与米制free r8照常，整个V7.3未完成，shutdown=false。

---


完整模型已实际载入并进入首轮任务微调，初始化489对象评价已完成；最终训练结果尚未取得。主joint r5与free目标r8继续原配置，本工作不替换V7.3的预训练视觉几何适配主线。

来源为[AdaPoinTr作者代码](https://github.com/yuxumin/PoinTr)、[官方模型](https://raw.githubusercontent.com/yuxumin/PoinTr/master/models/AdaPoinTr.py)、[PCN配置](https://raw.githubusercontent.com/yuxumin/PoinTr/master/cfgs/PCN_models/AdaPoinTr.yaml)。本机复用PoinTr_AdaPoinTr_4603257源码、AdaPoinTr_PCN.pth和v72-pointr环境。使用完整512查询/16384输出、384维、6层编码器/8层解码器及官方去噪结构，载入全部预训练权重，所有模型参数允许更新；没有改成旧4096输出或只训练末端小头。现有checkpoint包含32496706个模型Tensor元素（含BN缓冲），准确可训练参数数由运行manifest记录。

输入保留原build Actor全部LiDAR点及只读尺寸，不用后续测量初始化。官方DGCNN在第一层会构建完整N×N距离矩阵，因此仅将精确kNN的查询轴分成512块、保留全部keys与原邻居数，避免通过裁剪输入适配显存。输入不足512槽位时重复已有点，重复不算新增观测；输入超过512时不限制点数，模型内部原始FPS层仍按官方结构工作。坐标保留本项目Actor规范轴，除以已知最大box维度做各向同性归一化，输出乘回米制；不从target拟合中心/尺度/旋转。

官方PCN损失假定完整GT且fine与GT同点数，对稀疏真实标签不能直接照搬。此次任务适配保留完整网络与去噪分支，但使用：同一匹配预算显式曲面的观测target→surface覆盖、原始首返回前的直接free、软box envelope；另加权重0.1的target→coarse覆盖、权重0.1的局部稀疏去噪覆盖。后者从真实fit观测中选去噪查询附近的已测点，做已测点→局部denoised输出单向距离；不强迫全部预测靠近不完整target。未知区域不因没有LiDAR点被标为空。主free权重0.5，envelope0.05，与当前query控制一致；附加项和优化器预算差异明确报告，不能称完全相同目标或官方PCN基准复现。

物理主评价从全部16384预测点以FPS选min(build,1024)+512个中心，与query方法的曲面片预算相同，使用同0.06m PCA三角片、同真实首交点/覆盖/free算子。训练时每步更新邻域与PCA架，但这些离散选择和PCA方向在本步停止梯度，仅将表面位置梯度传回所选预测中心，避免退化特征值处的特征向量梯度不稳定；该条件梯度近似需要保留为限制。既有只读LiDAR PCA基线的数值语义不变。初始与最终完整16384点输出单独保存，后续可评价原生密度/点集与显式化差异，不把下采样转换的错误直接算成原生点预测错误。

训练登记 `WS-V73-M2-ADAPOINTR-01/20260907T221000Z__population-full-track-pcn-s7307-r1`：371个可输入fit Actor、67个可输入dev Actor，51个原输入缺失对象仍完整报告为空。仅fit读取既有全轨迹标签，开发无优化。30epoch/11130 Actor呈现，按4个变长Actor逐个反向累积，预计2790次优化更新；AdamW lr1e-4、weight_decay5e-4，epoch21后乘0.9，gradient clip10，seed7307，全FP32。此为完整预训练模型任务微调，不宣称用本数据从头复现官方600epoch PCN训练。若最终拟合不足，再依据实际训练和开发现象扩大预算或调整；不预先把固定30epoch的负结果当作充分失败。

初始化评价用于建立未做本轮任务适配的强起点，最终评价与之及r7/r8比较。相同固定LiDAR PCA结果直接复用；每epoch保存完整model/optimizer/scheduler及RNG状态，不新增校验和或门控。实际GPU/cgroup峰值由真实运行记录，不因与其他作业并发竞争就宣布单作业资源不足。

实际运行代码23981936，PID28663，载入官方epoch353 checkpoint，32494657个可训练参数。最近状态epoch4/290次优化更新，GPU allocated峰值1.00239GiB，非零完整模型梯度；此为训练过程快照，最终结果尚未完成。每epoch保存状态的原计划已实际执行。

适配前完整开发75对象/5日志：literal hit9.37%、early4.94%、miss54.97%、free0.11617m、target到surface距离0.21732m、recall@0.2m67.11%。相对固定PCA，hit差−11.87pp、日志bootstrap95%[−21.74,−2.01]pp；free差+0.09846m、[+0.02770,+0.16252]m；miss差−18.44pp、[−30.88,−3.52]pp。距离差−0.08747m、[−0.19164,+0.01970]m，recall差+1.70pp、[−11.13,+10.12]pp。尚不能用适配前的域差异、稀疏输入与匹配曲面转换结果代表认真微调后的强基线。共有输入与运动分层、原始逐Actor数据及全部配对见 `docs/autoresearch/worldsim_v73/m2/global/adapointr_r1_initial_analysis.json`；23个无留出自有回波和8个无输入空预测保留在完整报告。

源码：`motion_proj/worldsim_v73/adapointr_baseline.py`、`scripts/train_worldsim_v73_adapointr.py`。F02/F05与场景F04仍active；整个V7.3未完成，shutdown=false。
