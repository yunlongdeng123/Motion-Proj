# V7.3 独立确认数据准备

## V7.3 AdaPoinTr坐标修订结果、AV2外部输入完成与旧域场景实测（2026-09-08）

AdaPoinTr Y-up r2 `WS-V73-M2-ADAPOINTR-01/20260907T231000Z__population-full-track-pcn-yup-s7307-r2` 已完整结束，code acd91103，30epoch/11130呈现/2790更新，5109.12s，GPU allocated峰值1.00658GiB、RSS2.07924GiB；PID37887已退出。保持r1输入/全轨迹标签/seed/预算，仅迁移官方PCN轴接口。75开发Actor/5日志：hit14.2906%、early8.5389%、miss54.2423%、free0.122057m、target→surface距离0.113749m、recall@0.2m84.6032%。相对自身初始化，距离−0.057385m、日志bootstrap95%[−0.086634,−0.037315]m，recall+11.792pp、[+4.380,+18.482]pp；early+5.573pp、[+2.968,+8.172]pp。相对r1，free−0.024503m、[−0.086788,+0.032201]m，hit−1.761pp、[−6.766,+3.705]pp，距离+0.016169m、[+0.000341,+0.035096]m，不能宣称坐标修复带来一致物理改善。F07接口修订已执行并完成训练，原r1局限保留；物理缺陷归入仍active的F02/F03，不扩大成完整补全路线失败。

相对r8，Ada r2 hit−16.712pp、[−23.434,−8.407]pp，free+0.087737m、[+0.038663,+0.136812]m，recall+12.658pp、[+2.480,+26.112]pp；相对r9，hit−18.178pp、[−26.080,−8.322]pp，free+0.083514m、[+0.030030,+0.136998]m。移动9Actor/2日志hit4.1164%、early8.0231%、miss76.3141%、free0.057109m、距离0.104197m、recall98.8670%，再次说明0.2m邻近覆盖不能代替字面命中。8个原空输入和23个无自有留出返回仍在全体报告；原生16384点与匹配曲面分别保存。完整配对见 `docs/autoresearch/worldsim_v73/m2/global/adapointr_r2_analysis.json`。

针对覆盖/首交点脱节，先查阅[APSS SIGGRAPH2007](https://cgl.ethz.ch/research/past_projects/apss/)与[2DGS SIGGRAPH2024官方实现](https://github.com/hbb1/2d-gaussian-splatting)。前者体现局部拟合对稀疏点集表面定义的作用，后者显式区分有向surfel求交与网格提取；不迁移可学习opacity替错误前景变透明。先登记一次固定输出诊断 `WS-V73-M2-ADAPOINTR-DENSITY-01/20260908T030000Z__development-native-vs-matched-r1`：只读r2最终点集，在全部75旧开发Actor上对比原生点集覆盖、匹配中心覆盖和全部16384点同0.06m/20邻居PCA显式化的硬交点。它改变输出密度，作为敏感性分析而非匹配预算主表，不按结果挑Actor/半径。另登记r2匹配表面的雕刻背景组合 `WS-V73-M4-SCENE-COMPOSITION-01/20260908T030000Z__development-adapointr-yup-carved-r6`。两项当前尚未执行；不预先归因于支持稀疏或法向。

外部数据 `WS-V73-M4-AV2-DATA-01/20260908T020000Z__external20-common-windows-r1` code f70d099c 已完整导出/合并，2061.27s，父RSS3.66828GiB、子最大RSS1.26955GiB，PID41941及子任务已退出。原20日志全部保留，936刚体Actor：878 ready、58 unavailable_input，221移动>2m/s、108无速度；1个无可用Actor相机时刻，119个无自有heldout回波。560个完整相机输入，原始11723402返回=build7815152+heldout3908250，姿态未知返回0、框归属重叠7077。输入/标注值已处理，但未做外部网络推理、模型质量评分、共享优化或按覆盖更换身份。新域最终确认尚未发生，训练仍仅旧fit；索引归档至coverage/av2_external20_index.json。

旧AV2逐束场景三项登记均以code ad457f3c成功执行：数据34.5845s/RSS0.85305GiB，原背景评价11.3142s/RSS0.96797GiB，雕刻背景11.0018s/RSS0.94923GiB。21Actor、187494原始heldout束，cohort5257、移动cohort88、框边界2708，单条历史开发日志，不能用射线量或单日志bootstrap当独立确认。2916480背景三角面中仅按build自由空间删除211100（7.24%）；71498条build冲突束降至0，此结论仅此窗口中央束/当前容差，不能外推所有几何无侵入。

旧AV2仅背景全束free0.733877→0.513045m、early20.5169%→14.1498%、hit44.3774%→48.5205%，miss31.1071%→32.1333%。雕刻背景cohort：PCA hit62.9446%/early9.5872%/miss21.6663%/free0.060124m；native fusion 42.6479%/33.3080%/18.2043%/0.359073m；r9 54.2134%/9.2068%/17.2912%/0.074238m。移动仅88束时PCA hit21.59%/miss73.86%，r9 44.32%/37.50%，native fusion22.73%/64.77%；不从这一旧单日志选外部赢家。背景对cohort的提前遮挡仅约0.38%，与nuScenes约25.19%明显不同，需要分别报告数据与背景误差；F04仍active。详见m4/av2_old_scene_r1_summary.json、av2_old_scene_r2_summary.json、av2_old_scene_data_index.json。

主joint r5恢复PID39009与native-only full_track r11 PID38460仍运行，占GPU约13810/2920MiB；未启动r10。磁盘约116GiB可用，当前没有真实资源不足证据。F01/F02/F03/F04/F05 active，F06直接数据约束缓解，F07坐标接口已落实，F08已恢复但原退出原因未知，下一编号V73-F09。整个V7.3未完成，shutdown=false；继续自动推进，最终必须保存/push并确认训练、评价、数据及启动控制进程全部结束才关机。

---


## V7.3 外部20日志通用输入已启动与场景结果图（2026-09-08）

`WS-V73-M4-AV2-DATA-01/20260908T020000Z__external20-common-windows-r1` 已实际启动，codef70d099c，CPU父进程PID41941，日志 `/root/autodl-tmp/controller_logs/v73_av2_external20_export_r1.log`。最新3/20日志导出完成，原身份顺序、四build/两heldout时间和全部七相机保留；后续日志由该父进程顺序执行，属于关机前必须结束的数据控制进程。尚未完成整个20日志合并。

原始图像/点云和标注数值已用于格式转换、逐点坐标、已知轨迹归属与输入/heldout分离，不能再写“新数据数值未读取”。没有运行新域网络推理、重建质量评分或共享梯度更新；方法选择仍依据既有开发日志。exporter没有按新输入覆盖或预测好坏挑选/删除身份，空输入也保留。每日志处理日志、input_protocol和case均在注册run下，真实质量确认仍待最终模型选择后进行。

背景雕刻结果图由已保存r4/r5汇总生成，未重复推理：`docs/autoresearch/worldsim_v73/m4/V73_BUILD_FREE_BACKGROUND.png/pdf`，脚本 `scripts/plot_worldsim_v73_background_carving.py`。上排显示5日志背景free/early/miss的配对变化，下排显示同背景组合的cohort free/hit/miss；均值与个别日志都保留，明确Ada r1轴限制及不同训练预算。已检查排版并修正一处长标题截断，未新增模型smoke/回归。

后续AV2全场景组合需保持逐点时刻：当前nuScenes场景入口使用每扫描一个Actor刚体矩阵，不能直接冒充AV2逐点已知轨迹。下一实现应按每束时间把原点/方向转入同一Actor规范系，再通过其固定BVH求交并全局硬排序；背景使用参考ego已补偿端点和逐点束原点。先在旧AV2开发窗口接通，20新日志不用于选择几何时序约定。

三项训练正常，最近Ada Y-up r2为epoch25；r5恢复和native-only full_track继续，未启动r10。F01/F02/F03/F04/F05/F07仍active，F06直接数据配置缓解，F08恢复推进但退出原因未知，下一编号V73-F09。整个V7.3未完成，shutdown=false；必须完成研究、保存/push并确认训练、评价、数据及其启动控制进程全部结束后才关机。

---



2026-09-08，仅元数据/文件存在性调查，未读取候选图像、点云、标注或模型质量。

nuScenes本地原始载荷调查耗时10.05s：当前31窗口属于25日志，磁盘完整七传感器关键帧足够形成四build时刻的共35场景/27日志；当前日志外仅scene-0139和scene-0379。二者分别出现在V6.4 fit和V5诊断/开发配置，不能因不在当前25日志中就重用为全新确认。逐场景可用性见 `docs/autoresearch/worldsim_v73/coverage/log_payload_inventory_r1.json`。

更完整的身份调查发现：nuScenes trainval全部850场景只属于68日志，而旧V7.2角色表已覆盖全部68条（train54、legacy2、exposure_unknown2、dev4、route_select3、source_test3）。因此按旧记录保守排除后，没有无历史角色的新trainval日志。公开归档确实挂载于 `/root/autodl-pub/nuScenes/Fulldatasetv1.0/Trainval`，十个blob约294GiB，可解决旧日志缺相机载荷问题，但不能创造日志独立性。尚未为此扫描/解压整套归档，也不把更多相邻场景计作独立样本。

对应F05，继续当前开发训练，同时迁移到官方公开AV2 Sensor train中尚未使用的日志作为外部确认候选。官方train/val名称不会决定本研究是否训练：最终确认所选train日志在V7.3中仅供固定方法构建与评价，不能进入共享参数或超参数选择。跨数据集结果需明确是新日志与传感器域同时改变，不能伪称同分布确认，也不能与现有nuScenes开发集混合统计。

依据[AV2官方说明](https://argoverse.org/av2.html)、[官方下载说明](https://argoverse.github.io/user-guide/getting_started.html)与[官方传感器格式](https://github.com/argoverse/user-guide/blob/main/guide/src/datasets/sensor.md)：使用已公开的标定、时间戳、刚体轨迹与周视相机；保留每台相机自己的采集时刻。AV2点云已做ego运动补偿且在egovehicle坐标，双LiDAR来源和束原点需按官方字段重建，不能直接套nuScenes单传感器扫描模型。实现/对应问题在已经使用的AV2开发日志上处理；最终候选先依身份和时间元数据选取，待方法定型后评价，不按预测结果筛日志。

官方S3 train目录包含700日志，已有仓库身份引用排除1条，699条候选中按日志名字典序选择前20条；名单在列举所选日志的对象元数据之前写入。名单及精确载荷见 `configs/worldsim_v73/av2_external_confirmation_r1.json`，脚本 `scripts/prepare_worldsim_v73_av2_confirmation.py`；后续运行复用既有名单，不因下载或质量失败换日志。元数据准备98.49s，无原始图像/点云/标注值读取。

使用LiDAR原始序号[5,15,20,30]作build、[10,25]作留出，对应约0.5/1.5/2.0/3.0s与1.0/2.5s；七周视相机按每个build时刻最近实际采集时间读取，所有时间戳保留。20日志所需760文件共343260344字节（327.36MiB），目录元数据无缺项；相机与LiDAR时间差范围−23.10至+18.11ms，不能将它们硬置为同一时刻。只复制28相机帧、6个LiDAR扫描和4个标定/轨迹/标注文件每日志，不为选帧下载整个1TB数据集，不使用额外三维目标选择输入。

下载后也不立即开展外部模型选择。格式迁移必须先处理AV2双LiDAR来源/ego补偿、纳秒时间与只读Actor姿态插值、前相机纵向画幅和内参变换。当前query模块的6项nuScenes相机嵌入也不能直接索引为7路AV2：保留全部7路观测，在旧AV2开发日志上确定基于已知标定/方向的接口映射或无数据集ID表示，不能临时丢相机或给第7台加入未经训练随机向量后声称可靠零样本结论。该项尚未实现，不是最终确认已经完成。

当前20条日志身份已登记，精确载荷下载已启动（code c00020fb，PID34904，760文件），已于977.74s后全部复制成功、返回0且下载进程退出；日志在 /root/autodl-tmp/controller_logs/v73_av2_confirmation/，尚未解码新图像/点云或开展外部评价。磁盘约128GiB可用，该卡点是数据独立性与格式迁移，不是要求立即关机的资源不足。
