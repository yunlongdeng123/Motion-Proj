# 当前研究状态

更新：2026-10-17（Asia/Singapore）。当前最高优先级是动态分支清理：道路板片、残影、异常几何与车身保真。任务 `WS-V82-DRIVINGEDITOR-SHADOW-REBUILD-20261017 / E1_balanced_s0`；用户禁止subagent。只对相机和同帧真实LiDAR证据清楚的7辆做delete：新5、6、12、18、25、26、37，旧0/1/2分别对应5/26/18。

E0从原始测量初始化、未加载旧18k或GN11。5000步同7帧body误差均下降，但动态银车a37带道路板片，白车a18仍破碎（0.1421→0.0924），不能进入材质。原E0@5000已保存且停止。冻结逐点诊断a37远离已知车体区域85.17%动态贡献来自框底15cm内种子；原外部梯度因全局多实例归一化弱化，完整外域没有被删除。底部点归属也可能有问题，未确认由单项机制完全解释。

E1使用明确R4实例平衡配置，从保存的5k模型/Adam/RNG继续1000步到global6000，只改变已登记的mask归一化/外部权重与车身采样，不改原数据/旧检查点。21CPU检查通过。当前控制器E1_repair1000_controller.json，日志E1_repair1000.log，源implementation_r4，active_experiment.json供监控读取。6000步必须对照E1_preview_006000、diagnostic_006000_r4与父5k证据；检查外泄贡献、车身和背景。修复不足则查真实LiDAR底部归属，不盲加步数。动态分支完整49帧与7车delete未验收前不进入材质/阴影。[修复组件与记录](autoresearch/worldsim_v82/drivingeditor_shadow_rebuild_20261017/v3_gpu_r4/) · [源码](../scripts/worldsim_v82/drivingeditor_shadow_rebuild_20261017/)。

主预算45k包含父5k，当前修复不增加总步数，不自动排七臂。b-c-15每10分钟检查PID、日志、阶段和数值，不重复启动、不关闭GPU，别人任务优先。初始11.29h测速不含增密、教师/I-O/评价；按实际阶段更新ETA。

输入49帧/46训练，318083真实LiDAR种子、43轨迹、8263跨帧道路伪参考。旧三车147对完整mask内外齐全，辅助403/2107对未知只排除mask损失，仍用RGB/LiDAR；两条无训练LiDAR轨迹不造资产。道路候选不是语义GT，无独立无影或反事实编辑GT。旧四辆新增Actor如无旧独立资产仅作旧场景外观参考。

49帧物理相机CPU核对一致，旧native投影-0.125像素、GT缩放差2.4097/255；共同相机/GT/显示线性域的只读GPU重渲染尚未执行。新RGB实际为Lanczos，camera_pairing_audit_r2修正r1误写bilinear的文字，不改已测像素。旧原生结果保留。

最终交付仍需65条49帧编辑/静动态分支视频、逐帧指标和raw106–108/大角度案例。旧18k[结果](v82/DRIVINGEDITOR_SHADOW_SEQUENCE_20261016.md)保留，不把工程检查当画质通过。人工verdict=null、problem_solved=null。其他工作见[EXPERIMENTS](EXPERIMENTS.md)。failure_ledger_refs=[V82-F01]，failure_ledger_delta=none；原dirty用户工作保留。
