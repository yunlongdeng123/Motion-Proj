> 历史归档（2026-09-10）：以下为原版本事实与指令快照；当前执行以 docs/RESEARCH_STATUS.md 为准。

# V7.3 fit全轨迹监督

## V7.3 全轨迹LiDAR-only控制最终完成（2026-09-08）

`WS-V73-M2-GLOBAL-ACTOR-01/20260907T203500Z__population-lidar-only-track-labels-s7304-r7` 已完成，code d78e99ae，30epoch/11130更新，3519.80s包含最终完整评价（initial与固定PCA复用），GPU allocated峰值0.19448GiB、RSS2.987GiB；PID23188已退出。输入仍为原build，只有fit surface/free标签扩展到全轨迹；51个原输入不可用对象继续保留空预测，没有把后续测量作为推理输入。开发75对象/5日志、23个无留出自有回波、8个无预测。

| 同一完整开发队列 | literal hit | early | miss | free m | 观测target到surface m | recall@0.2m |
|---|---:|---:|---:|---:|---:|---:|
| 固定LiDAR PCA | 21.24% | 4.35% | 73.41% | 0.01771 | 0.30479 | 65.42% |
| LiDAR-only r6，短窗标签 | 31.01% | 14.02% | 49.31% | 0.08375 | 0.24118 | 72.42% |
| LiDAR-only r7，全轨迹标签 | 31.75% | 11.84% | 49.90% | 0.07157 | 0.20952 | 74.55% |

r7−r6独立日志配对：hit+0.75pp，bootstrap95%[−2.10,+3.37]pp（3/5日志改善）；early−2.18pp，[−4.70,−0.37]pp（4改善、1相同）；free−0.01218m，[−0.02226,−0.00210]m（3改善、1变差、1相同）；target距离−0.03165m，[−0.08340,−0.00251]m（4/5改善）；recall+2.12pp，[−0.03,+4.44]pp。miss+0.58pp，[−4.10,+5.27]pp。更充分的训练观测缓解部分侵入/定位问题，尚未形成全面优势；尤其free仍明显高于PCA。此收益属于标签范围变化，不能归因于视觉或空间交互结构。

原build LiDAR-ready开发分层为67对象/5日志，其中16无留出自有回波：r7 hit33.77%、early12.47%、miss46.77%、free0.07548m、target距离0.20952m、recall78.89%。该分层由原输入元数据定义，与模型质量无关；完整75对象仍为主报告。已知速度>2m/s子集仅9对象/2日志（3无留出自有回波、1空预测）：r7 hit17.73%、early6.84%、miss73.64%、free0.03667m、距离0.08915m、recall85.42%；相对r6有物理侵入改善和命中/缺失退化，不能凭2日志宣称动态泛化成立。

fit旧短窗口时刻也属于训练标签：r7 hit40.30%、early9.50%、miss45.01%、free0.04752m、距离0.15537m、recall79.40%；这些不是独立确认，也不是对整条全轨迹所有标签的完整评价。所有标签仍是稀疏真实观测与框归属代理，不称完整表面GT。

完整summary、原PCA/初始化/final逐Actor记录、r6与native fusion配对、运动及共有输入分层均已归档 `docs/autoresearch/worldsim_v73/m2/global/population_lidar_r7_summary.json` 与 `population_lidar_r7_analysis.json`，采用既有结果汇总，不重跑基线。native fusion与r7标签预算和无LiDAR输入处理有差异，结果表不混为同监督架构比较。

当前主joint r5 PID18843（最近epoch11）、米制beam_tube_range r8 PID26975（最近epoch2、GPU峰值0.33034GiB）正常训练。r8与r7输入/全轨迹标签/架构/seed相同，仅free目标变化；等待真实最终结果决定下一对策。完整主模型后还需用与r7/r8相同标签预算训练joint/原生保守控制，并推进CAPA、AdaPoinTr、event和新日志；不从本LiDAR控制推导视觉路线失败。

failure_ledger_delta=update V73-F02/F05（更多fit观测有局部作用，动态独立样本仍不足）；F01/F02/F03/F04/F05继续active，F06直接数据配置缓解，下一编号V73-F07。场景背景r2及配对已完成，scene无残留作业。整个V7.3未完成，shutdown=false；按既有自动研究持续推进，真正完成且确认无训练/评价/数据/任务队列后再关机。


---

## 实际标签构建完成

完整cohort LiDAR-only r6完成（code98d9fa32，run `20260907T193500Z__population-lidar-only-extra-time-s7304-r6`）：371fit训练Actor、67dev可输入Actor，30epoch/11130更新，2904.27s含自身initial/final评价，峰值0.181GiB、RSS2.964GiB。固定PCA基线从r5复用，其计算成本不包括在r6 wall time中；该run与joint r5并发，不用于单作业速度排名。489对象全部记录，43fit/8dev无输入对象保留空预测；dev75对象中23个无留出自有回波，不把它们当作有几何GT，miss/覆盖与无输出计数同时报告。

dev5日志均值：LiDAR PCA→训练LiDAR-only为 hit21.24%→31.01%，early4.35%→14.02%，miss73.41%→49.31%，free0.01771m→0.08375m，观测target到surface距离0.30479m→0.24118m，0.2m覆盖65.42%→72.42%。hit/覆盖/距离在5/5日志改善，但early/free没有日志改善（4差、1相同）。相对PCA，hit配对差+9.76pp、日志bootstrap95%区间[+4.79,+14.73]pp；free差+0.06605m、区间[+0.03014,+0.09334]m。纯LiDAR生成器同样有覆盖与侵入冲突，因此不能把F02全部归因为高维视觉。当前joint r5还在训练，不能用其25Actor旧结果与完整r6直接比较。

新增按已知build窗口平均速度>2m/s的运动子集汇总（只分析已有结果，不重跑模型）：dev9Actor仅2日志，其中3个无留出自有回波、1个无预测。r6运动子集hit18.70%、miss71.85%、free0.04875m、距离0.11049m；该独立样本量不足以形成动态泛化主张。总体5日志也仍为旧开发集，不是新日志确认。完整结果与配对/运动分层=`docs/autoresearch/worldsim_v73/m2/global/population_lidar_r6_*.json`。

fit全轨迹标签构建也完成（code1c19f2c2，run `20260907T200000Z__fit-track-measurements-r2`）：175.86s、RSS8.553GiB、磁盘356MiB。414对象中407有正观测，合计1690284个Actor内精确坐标去重点（非独立表面样本），7748633条相关原始束，11214个Actor×扫描、9782个标签专用时刻；原build点合计194680。43个无build LiDAR对象中36个在更长时段有标签，仍不将这些后来测量作为输入。更充分观测不是完整表面GT，未知区域语义不变。索引=`docs/autoresearch/worldsim_v73/coverage/fit_track_targets_r2_index.json`，目标Tensor留在单独run目录。

后续r7登记：`WS-V73-M2-GLOBAL-ACTOR-01/20260907T203500Z__population-lidar-only-track-labels-s7304-r7`。相对r6仅将fit surface/free标签替换为独立全轨迹目标；build输入、seed7304、同decoder、lr1e-5、range free0.5、30epoch/11130更新保持相同。新参数--fit-targets仅在fit损失加载target_points_actor_m/target_rays，predict仍只读原case.points_actor_m；native数据辅助项（未来joint使用）也只读build像素。开发集不加载此目录。有效label_times记full_track，fit原留出短窗口已属训练标签；无输入、未参加优化对象不再标记training_labels。复用r6相同输入/同seed模型的initial评价和r5固定PCA结果，避免相同算子重复运行；复用路径写入manifest。

同时登记完整cohort原生保守参考：`WS-V73-M2-GLOBAL-FUSION-01/20260907T203500Z__population-native-lidar-fusion-r2`。使用已认真训练的M1r3 DPT，固定其参数，每个24视图窗口解码一次后供所有Actor共享米制深度；这是评价期固定输出缓存，不是训练中缓存可训练路径。按原规范轨迹融合native+LiDAR、同0.06m PCA曲面片；没有Actor build LiDAR但有native支持时允许native-only，完全无支持则明确缺失。所有489对象均纳入，不能再沿用旧25Actor融合结果代表全队列。其fit标签预算仍为M1 build深度监督、没有全轨迹标签，差异单列，不宣称同标签架构胜负；原生强控制后续仍可用相同几何标签优化。

r5正常运行；r6和标签生成已结束。r7及固定融合评价提交后启动，预计额外GPU开销小于另一套24视图反向，仍以实际占用为准，不把并发争抢当成单作业资源不足。CAPA数据已备、模型未运行；AdaPoinTr接同真实标签与统一表面、场景组合、独立新日志仍待推进。failure_ledger_delta=update V73-F02/F05；F01/F03/F04仍active，F06直接数据配置缓解，下一编号V73-F07。整个V7.3未完成，shutdown=false。

---

# V7.3 fit全轨迹监督

fit全轨迹载荷盘点完成：run `WS-V73-M2-EXTENDED-TARGETS-01/20260907T195000Z__fit-track-payload-inventory-r1`，code856aa525，38.90s。25fit场景合计1004个关键帧LiDAR均在本地；414个窗口内已知Actor中，403个在当前短窗口之外还有可插值轨迹与可用LiDAR，合计9064个额外Actor×关键帧机会。该数量不是新增正点数，尚须实际提取归属观测；11个无额外机会对象保留。没有新增日志或dev标签，输入Actor选择仍依据原build/metadata。记录=`docs/autoresearch/worldsim_v73/coverage/fit_track_payload_inventory_r1.json`。

真实fit标签生成已启动（code1c19f2c2，PID21982，日志 `/root/autodl-tmp/controller_logs/v73_fit_track_targets_r2.log`；已生成首批25对象，进程RSS约8.1GiB，最终数量待完成汇总）：`WS-V73-M2-EXTENDED-TARGETS-01/20260907T200000Z__fit-track-measurements-r2`，入口 `scripts/prepare_worldsim_v73_track_targets.py`。使用每次LiDAR扫描的只读Actor姿态，将该轨迹内全部可用关键帧正点累积到规范坐标，同时保留原始束首回波/歧义归属及已观测free。仍为box+0.1m、重叠排除的归属代理和scan级时间近似，不声称成为完整表面GT或精确实例分割。非build记录明确为fit_label_time。

标签保存为单独目录的target_points_actor_m与target_rays，原始actor-data/build输入/图像/metric scale/query seed一律不替换，当前r5/r6保持短窗训练配置直至完成。414个fit对象包括43个当前无build LiDAR者，标签文件保留其输入缺失身份，不能将后来测量偷偷拿来当输入。有更多观测支持仍不等于未知区域已知：后续训练不得将所有预测到稀疏标签距离都解释为几何错误。主模型与AdaPoinTr等控制需要同一新标签预算后才可比较，不能把较多监督单独包装成架构收益。

为避免每Actor重复读取同一扫描并计算全场景box归属，数据构建按场景共享world points、传感器原点与membership，再按当前Actor姿态投影；处理完一个场景释放缓存。默认短窗读取语义保持不变，include_track仅用于新的fit标签入口，不新增校验门控、hash或重复回归。

当前r5 PID18843和r6 PID20389正常训练；本标签生成用CPU并保留已有raw载荷，不需要下载或新环境。CAPA数据桥已完成，但CAPA模型优化未启动。failure_ledger_delta=update V73-F05（从短窗口到更充分真实训练标签的准备，不是已测性能提升）；F01/F02/F03/F04/F05仍active，F06在直接数据配置缓解，下一编号V73-F07。资源充足，shutdown=false，继续研究。
