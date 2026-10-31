# 当前研究状态

当前里程碑：**WS-V43-R4-CPU-20261031 / run_r1 已完成 CPU 入口、合法候选冻结与任务登记；原生渲染和科学评价未执行。** [本轮独立报告与组件图](autoresearch/v43_ownership_20261028/decision_r4_cpu_20261031/report.html)；依据用户修订的 V43 方法提案 R4，先聚焦固定表示内的共享编辑决策，D/C 分开。

实际范围：原 M0 10K、同世界 hybrid8 10K、真实首场原生 30K 三配置，两个已有开发祖先，登记 12 个 none/delete/shift/rotate 任务。每配置三个来源组、八个 keep/rebind 共享方案；defer 只在查询层。两个仿真重建不是独立世界或封存测试。候选行依次为 392468/667559、490823/1157894、2703/918850；小规模指三组二元动作域，不指少量高斯，也不代表行级或全局最优。

输入边界：sim 保留既有 simulator_sensor_oracle 掩码/轨迹及无噪稀疏 LiDAR 角色；真实首场用原始 DEVA 关联审核、标注辅助轨迹和实测 LiDAR 行。旧 M1 的独立语义/深度类别证据不进入候选筛选或资格，历史原行账本仅为可选隔离诊断。完整 checkpoint bundle、event/gauge/camera/preparation 纳入冻结上下文，一份路由跨帧和全部操作复用；原模型参数不变。

验收：实际 0.5 核 CPU、单线程、CUDA 隐藏，程序墙钟 106.639 秒、峰值 RSS 590.891 MiB；13 项合同及实际产物检查通过，独立代码与产物审核完成。12 项资格全部 unknown，阈值 null；f_geom/bg 未测，f_view 是方位扇区代理，f_motion 是 CPU 字段一致性。原生 Scene/CUDA 恢复、两种 none、FP32 合成和实际共享编辑仍待验。新增渲染/奖励/训练更新均为 0，D/C 数值 unavailable，human_verdict=null。

本地共享诊断平台已原位治理：共同入口四场景、每场景一个链接；三个场景页和旧道路场景页提前展示影像与几何，高级控件折叠，移除进展横幅和独立报告导航。真实首场候选仅作为固定参考帧的只读输入预览。报告在对话中单独交付，不追加到平台。静态检查和脚本语法通过；浏览器工具安全策略阻止 file://，交互验收未完成。HorizonForge 补充包仅有演示视频等，无原始场景数据；官方项目页核查时未提供场景/代码下载入口，未接入候选。

下一步：在真实 GPU 环境做原生恢复和实际共享路由两种 none，继而测量合法资格指标并独立校准阈值、保持域硬约束和联合操作目标；对冻结八方案执行完整原生评价后才计算相应有限域差距。多个独立祖先上出现合法可学习的排序信号后，再开始 L0。小时级实验仍须先完成约十分钟真实保存、评价、完整恢复和下一阶段入口 smoke；本轮 CPU 检查不替代它。

现场：`/root/autodl-tmp/drivingeditor_repro/campaign/v43_ownership_20261028/v43_decision_r4_cpu_20261031/run_r1`。本轮代码位于 `scripts/worldsim_v82/v43_ownership_20261028/v43_decision_r4/`。旧 checkpoint、日志和失败证据保留。此前来源控制仍见[原报告](autoresearch/v43_ownership_20261028/source_evidence_20261030/index.html)，不覆盖原 M0/M1/M2 或真实 R2。

failure_ledger_refs=[V82-F01]，failure_ledger_delta=none。用户本次为 CPU 模式开机，保留实例，不执行关机。
