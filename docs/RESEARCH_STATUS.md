# 当前研究状态

更新：2026-10-03。分支 `research/worldsim-v7.7-target-protected-editing`，主机 wm-3090-1001。

用户收敛为 O/N/U/Q 小条件分支：先验证“哪里有车／背景／未知”对真实 DELETE 的价值，主干冻结，身份系统、学习state encoder、2DGS/surfel后置。停止扩建旧数据工厂。

当前 task `WS-V77-TARGET-PROTECTED-20260929/r45` 已完成 CPU 准备：12旧训练例、4合成开发评测例、8固定真实DELETE；14个缺失LiDAR文件已补齐。O是GT框保守内核、N是排除实体后实测背景返回、U保留未知、Q为启发式可信度；不声称自动估计或精确分割。GT相机/车辆框为POC辅助，Y仅监督。初始整框O误标道路和不必要地面拟合限制N的对照均保留；统一用内核O与背景返回N，未逐例调规则。

157888参数四通道Adapter及240帧CPU合同通过。新训练0步，真实网络四通道GPU前后向未执行，模型收益未验证。固定160步，只更新Adapter；随后12评测例×3臂=36窗，比较关闭分支／全未知／正常条件，不选择性更换评测例。人工verdict空。

用户最新要求：先完成CPU准备，通知后再开GPU。当前无GPU训练/推理，也无等待自动启动GPU的控制器或定时任务。本地条件预览 `outputs/v77-onuq-r45/index.html`。细节、覆盖比例和组件图见 [r45报告](v77/TARGET_PROTECTED_ONUQ_ADAPTER_R45.md)。稀疏N与粗几何是本次POC的明确边界；真实DELETE跨scene收益仍未达到。

旧Q060条件正例、r7等全部历史权重及反例保留。磁盘仍约44GiB可用，本轮没有清理旧文件。沿用 [V77-F02](research_failures/entries/V77-F02.md)。下一步由用户开GPU后先验证冻结/零初始化，再跑有界训练与固定真实评测。
