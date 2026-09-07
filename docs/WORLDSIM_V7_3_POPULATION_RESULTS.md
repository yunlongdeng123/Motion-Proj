# V7.3 完整窗口Actor队列结果

完整cohort LiDAR-only r6完成（code98d9fa32，run `20260907T193500Z__population-lidar-only-extra-time-s7304-r6`）：371fit训练Actor、67dev可输入Actor，30epoch/11130更新，2904.27s含自身initial/final评价，峰值0.181GiB、RSS2.964GiB。固定PCA基线从r5复用，其计算成本不包括在r6 wall time中；该run与joint r5并发，不用于单作业速度排名。489对象全部记录，43fit/8dev无输入对象保留空预测；dev75对象中23个无留出自有回波，不把它们当作有几何GT，miss/覆盖与无输出计数同时报告。

dev5日志均值：LiDAR PCA→训练LiDAR-only为 hit21.24%→31.01%，early4.35%→14.02%，miss73.41%→49.31%，free0.01771m→0.08375m，观测target到surface距离0.30479m→0.24118m，0.2m覆盖65.42%→72.42%。hit/覆盖/距离在5/5日志改善，但early/free没有日志改善（4差、1相同）。相对PCA，hit配对差+9.76pp、日志bootstrap95%区间[+4.79,+14.73]pp；free差+0.06605m、区间[+0.03014,+0.09334]m。纯LiDAR生成器同样有覆盖与侵入冲突，因此不能把F02全部归因为高维视觉。当前joint r5还在训练，不能用其25Actor旧结果与完整r6直接比较。

新增按已知build窗口平均速度>2m/s的运动子集汇总（只分析已有结果，不重跑模型）：dev9Actor仅2日志，其中3个无留出自有回波、1个无预测。r6运动子集hit18.70%、miss71.85%、free0.04875m、距离0.11049m；该独立样本量不足以形成动态泛化主张。总体5日志也仍为旧开发集，不是新日志确认。完整结果与配对/运动分层=`docs/autoresearch/worldsim_v73/m2/global/population_lidar_r6_*.json`。

fit全轨迹标签构建也完成（code1c19f2c2，run `20260907T200000Z__fit-track-measurements-r2`）：175.86s、RSS8.553GiB、磁盘356MiB。414对象中407有正观测，合计1690284个Actor内精确坐标去重点（非独立表面样本），7748633条相关原始束，11214个Actor×扫描、9782个标签专用时刻；原build点合计194680。43个无build LiDAR对象中36个在更长时段有标签，仍不将这些后来测量作为输入。更充分观测不是完整表面GT，未知区域语义不变。索引=`docs/autoresearch/worldsim_v73/coverage/fit_track_targets_r2_index.json`，目标Tensor留在单独run目录。

后续r7登记：`WS-V73-M2-GLOBAL-ACTOR-01/20260907T203500Z__population-lidar-only-track-labels-s7304-r7`。相对r6仅将fit surface/free标签替换为独立全轨迹目标；build输入、seed7304、同decoder、lr1e-5、range free0.5、30epoch/11130更新保持相同。新参数--fit-targets仅在fit损失加载target_points_actor_m/target_rays，predict仍只读原case.points_actor_m；native数据辅助项（未来joint使用）也只读build像素。开发集不加载此目录。有效label_times记full_track，fit原留出短窗口已属训练标签；无输入、未参加优化对象不再标记training_labels。复用r6相同输入/同seed模型的initial评价和r5固定PCA结果，避免相同算子重复运行；复用路径写入manifest。

同时登记完整cohort原生保守参考：`WS-V73-M2-GLOBAL-FUSION-01/20260907T203500Z__population-native-lidar-fusion-r2`。使用已认真训练的M1r3 DPT，固定其参数，每个24视图窗口解码一次后供所有Actor共享米制深度；这是评价期固定输出缓存，不是训练中缓存可训练路径。按原规范轨迹融合native+LiDAR、同0.06m PCA曲面片；没有Actor build LiDAR但有native支持时允许native-only，完全无支持则明确缺失。所有489对象均纳入，不能再沿用旧25Actor融合结果代表全队列。其fit标签预算仍为M1 build深度监督、没有全轨迹标签，差异单列，不宣称同标签架构胜负；原生强控制后续仍可用相同几何标签优化。

r5正常运行；r6和标签生成已结束。r7及固定融合评价提交后启动，预计额外GPU开销小于另一套24视图反向，仍以实际占用为准，不把并发争抢当成单作业资源不足。CAPA数据已备、模型未运行；AdaPoinTr接同真实标签与统一表面、场景组合、独立新日志仍待推进。failure_ledger_delta=update V73-F02/F05；F01/F03/F04仍active，F06直接数据配置缓解，下一编号V73-F07。整个V7.3未完成，shutdown=false。
