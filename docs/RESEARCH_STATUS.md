# 当前研究状态

更新：2026-10-18。同一a6的旧GN11 30K＋实例30K冻结核对已完成：原生49帧逐像素匹配旧基线，另做49帧真实轨迹框几何选择、固定raw105/113/137精确贡献及共同物理视角。检查点与暂停标记保留，参数更新0，无subagent。[同一a6的图、组件和数值](autoresearch/worldsim_v82/drivingeditor_shadow_rebuild_20261017/v3_backend_restore_r15/old30k_a6_20261018/report.html)。

旧版a6由静态/PVG分支表达，没有独立owner6。旧观测图更自然，但其a6车身像素依赖非局部高斯；旧贡献集独立渲染出现明显拉伸和外溢。贡献集包含真实背景，不能当作车辆归属真值；没有证明旧版有同一批、同样归属的车底碎片，也没有把旧版的局限当成R15质量通过。

R15最小阴影修复仍是显式install的内存适配器，登记范围内数值/梯度/速度通过；生产训练入口、loss、配置、CUDA ABI及原checkpoint未改。[此前有限收口实验](autoresearch/worldsim_v82/drivingeditor_shadow_rebuild_20261017/v3_backend_restore_r15/closure_20261018/report.html)。a6车底外延与a37区域静态残影未解决；道路证据不足，不执行归属迁移或按高度裁底。原45K保持暂停，不新增R16/R17。

本次全部GPU渲染/审计完成、原始输出保存下载、代码数值push后，于2026-10-18 16:04（新加坡时间）执行AutoDL关机；命令返回0，随后SSH关闭。未检查云计费控制台。没有GPU、worker或controller待执行。[关机前检查与观察记录](autoresearch/worldsim_v82/drivingeditor_shadow_rebuild_20261017/v3_backend_restore_r15/old30k_a6_20261018/shutdown.json)。不根据历史控制器自动重开。

human_verdict=null，problem_solved=null。最新task=WS-V82-R15-OLD30K-A6-20261018，failure_ledger_refs=[V82-F06]，failure_ledger_delta=none（追加既有F06的基线证据）。49帧为同窗口重建视角，三锚帧为固定坏例诊断，不冒称独立测试或配对编辑真值。用户其他checkout的dirty文件保留。
