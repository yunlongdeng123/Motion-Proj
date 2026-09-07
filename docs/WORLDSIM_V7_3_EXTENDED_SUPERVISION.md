# V7.3 fit全轨迹监督

fit全轨迹载荷盘点完成：run `WS-V73-M2-EXTENDED-TARGETS-01/20260907T195000Z__fit-track-payload-inventory-r1`，code856aa525，38.90s。25fit场景合计1004个关键帧LiDAR均在本地；414个窗口内已知Actor中，403个在当前短窗口之外还有可插值轨迹与可用LiDAR，合计9064个额外Actor×关键帧机会。该数量不是新增正点数，尚须实际提取归属观测；11个无额外机会对象保留。没有新增日志或dev标签，输入Actor选择仍依据原build/metadata。记录=`docs/autoresearch/worldsim_v73/coverage/fit_track_payload_inventory_r1.json`。

真实fit标签生成已启动（code1c19f2c2，PID21982，日志 `/root/autodl-tmp/controller_logs/v73_fit_track_targets_r2.log`；已生成首批25对象，进程RSS约8.1GiB，最终数量待完成汇总）：`WS-V73-M2-EXTENDED-TARGETS-01/20260907T200000Z__fit-track-measurements-r2`，入口 `scripts/prepare_worldsim_v73_track_targets.py`。使用每次LiDAR扫描的只读Actor姿态，将该轨迹内全部可用关键帧正点累积到规范坐标，同时保留原始束首回波/歧义归属及已观测free。仍为box+0.1m、重叠排除的归属代理和scan级时间近似，不声称成为完整表面GT或精确实例分割。非build记录明确为fit_label_time。

标签保存为单独目录的target_points_actor_m与target_rays，原始actor-data/build输入/图像/metric scale/query seed一律不替换，当前r5/r6保持短窗训练配置直至完成。414个fit对象包括43个当前无build LiDAR者，标签文件保留其输入缺失身份，不能将后来测量偷偷拿来当输入。有更多观测支持仍不等于未知区域已知：后续训练不得将所有预测到稀疏标签距离都解释为几何错误。主模型与AdaPoinTr等控制需要同一新标签预算后才可比较，不能把较多监督单独包装成架构收益。

为避免每Actor重复读取同一扫描并计算全场景box归属，数据构建按场景共享world points、传感器原点与membership，再按当前Actor姿态投影；处理完一个场景释放缓存。默认短窗读取语义保持不变，include_track仅用于新的fit标签入口，不新增校验门控、hash或重复回归。

当前r5 PID18843和r6 PID20389正常训练；本标签生成用CPU并保留已有raw载荷，不需要下载或新环境。CAPA数据桥已完成，但CAPA模型优化未启动。failure_ledger_delta=update V73-F05（从短窗口到更充分真实训练标签的准备，不是已测性能提升）；F01/F02/F03/F04/F05仍active，F06在直接数据配置缓解，下一编号V73-F07。资源充足，shutdown=false，继续研究。
