# 当前研究状态

更新：2026-10-17（Asia/Singapore）。当前任务 `WS-V82-DRIVINGEDITOR-SHADOW-REBUILD-20261017 / E0_full_s0`。用户已开GPU并授权实验，仍禁止subagent。最新范围收紧为相机清楚可见且同帧真实LiDAR支持充分的7个Actor：新5、6、12、18、25、26、37；旧0/1/2分别对应新5/26/18，不按所有43个LiDAR轨迹做delete。

真实49帧输入导出与实际CUDA ABI/反传已通过；318083高斯、43轨迹、8263跨帧道路伪参考。旧三Actor147对完整mask内外监督齐全；辅助403/2107对mask未知，显式allow-partial-masks排除其mask损失，仍用RGB/LiDAR；两条无训练LiDAR轨迹不构建资产。18项CPU检查通过。工程检查与相机/LiDAR准入、组件图、计时记录见[GPU记录](autoresearch/worldsim_v82/drivingeditor_shadow_rebuild_20261017/v3_gpu_r3/)。[源码与命令](../scripts/worldsim_v82/drivingeditor_shadow_rebuild_20261017/) · [方法组织与历史CPU检查](v82/DRIVINGEDITOR_SHADOW_REBUILD_20261017.md)。

E0已在AutoDL的shadow_rebuild_20261017_v3/E0_full_s0启动，从原始测量初始化，不继承GN11。1000步几何预览已由根代理核对，7辆均可在独立动态分支辨认；白车新Actor18仍模糊（证据帧body线性RGB MAE0.1421），Actor12部分轮廓和Actor37毛刺尚待改善。因此同配置latest仅继续几何到5000步，再用同7帧复查，未通过前不进入材质。当前控制器E0_gate5000_controller.json，日志E0_gate5000.log。初始30k/10k/5k测速约11.29h，不含增密、教师、I/O、参考刷新和连续评价；阶段完成后更新ETA。b-c-15心跳ACTIVE，每10分钟核对PID、日志、数值与阶段，不重复启动，不关闭GPU。

后续完整49帧评价只对准入7辆逐车删除。旧三辆按真实track ID与旧实验配对同轴±1m、±15°/±30°连续编辑；新增四辆若无旧独立资产只对照场景外观。65流包括全动态独立RGB/raw alpha、全场景动态可见RGB/alpha、每辆独立/可见alpha、删除后剩余动态与运动编辑动态RGB。静态分支单独输出只作诊断，不是其他不可见车辆delete成功。49帧物理位姿已CPU核对；旧投影有-0.125像素偏移，GT缩放差2.4097/255，后续用共同相机/GT/线性显示域只读重渲染后再报逐像素差异，原生结果保留。GPU配对尚未执行。

画质、完整视频和当前problem是否解决尚未判定，人工verdict=null。道路候选非人工语义GT，跨帧参考非无影GT，始终遮挡表面与单视频光照仍有不可辨识性；没有完整间接光或车身BRDF。主预算45k，消融只保留接口，不自动无限追加。继续回看旧raw106–108范围外黑片、±30°资产破面和原位阴影加权误差未改善，[历史记录](v82/DRIVINGEDITOR_SHADOW_SEQUENCE_20261016.md)不覆盖。

其他项目由[EXPERIMENTS](EXPERIMENTS.md)定位。`failure_ledger_refs=[V82-F01]`，`failure_ledger_delta=none`；当前为工程与输入检查，没有新增科学否定。原dirty工作保持。
