# 当前研究状态

更新：2026-10-13T19:51:12.274453+08:00（新加坡）。当前任务 `WS-V82-DRIVINGEDITOR-INSTANCES-20261013/r1`：用户要求v82实例编辑能力，随后收敛为只用GMC多实例场景测试。原银车/红车不新增实验；已经做好的全车辆CPU准备与旧结果保留。[协议与组件图](v82/DRIVINGEDITOR_INSTANCE_EDITOR_20261013.md) · [机器状态](v82/DRIVINGEDITOR_INSTANCE_EDITOR_20261013.json)。

已实现真实框/track分配LiDAR、局部实例Gaussian、整体刚体pose、同步协方差/局部SH、命名删/移/转/场景内替换、全场景可见贡献监督与完整checkpoint入口。车辆没有PVG重复动态，首阶段固定轨迹及背景；原SGCM只继承背景训练结果，新static→rigid克隆未做。NPS及2024活跃监督保留。代码在scripts/worldsim_v82/drivingeditor_instances；不是识别网络或无3D框方案，也不是已证明论文创新。

GMC真实输入：两个raw track分别绑定DEVA77/82，137174/160901点，共298075；49×2全部98已审查mask。laser注释有效均49，FRONT可见分别7/39。其他停车仍静态；600k数据seed仅过滤两车框后579855，实际GN11继承模型删点后的点数待GPU测。旧49窗口raw23–71和原作者窗口未知边界保留。

本地67CPU检查通过/1CUDA实测待验；本地与远端真实Python3.9均通过GMC49时刻单实例删除/平移/旋转他车参数不变与core CPU状态恢复，CUDA未初始化。完整scene GPU模型/Adam/RNG恢复、照片重建和编辑质量未实测。框可能含道路，车影及Gaussian跨框尾部未解决；不能由身份标签或CPU数学合同认证成功补景。

19:41 SGT真实启动等待controller PID20065，远端instance_editor_20261013/queue_gmc_r1。最新只读状态waiting_for_other_tasks，无GMC GPUworker。等待不导入Torch、不持共享gpu_slot.lock，5分钟稀疏检查/24小时超时。另一legacy49训练worker10164及其controller10161持共享锁是只读快照，属于其他任务；不修改、不暂停、不误杀。StreetCrafter优先；CPU最多4线程/nice10。已有每小时心跳drivingeditor-30k更新为本GMC实例任务，其他自动化不恢复；无变化安静。

下步等待空闲执行仅GMC300步GPU兼容短测，再实际49帧分层/两车编辑核验。兼容通过后同一固定GT轨迹run续至总30k上限、10k/20k/30k完整保存，不追加scene/seed/loss网格或自动开放位姿。竞争出现则保存priority_yield退出，工程异常保留资产定位；最终新增结果接入已有outputs/DrivingEditor_final_report.html，不再增多HTML入口。新GPU显存和速度尚未测，旧3090初筛不能代替新实现实测。

原C00/G10/N01/GN11各30k和GMC/红车GN11各30k共180k已完成；原18节点及真实报告保留。GN11−C00全图+0.3254dB/车体−0.1287dB，旧模块gate passed，human GN11/新scene null，完整恢复未测；新实例表示不自动套旧模块gate。此前2026-10-13 12:03:48 SGT关机请求及原自动化暂停已收口；当前主机重新在线并有其他用户任务，此新阶段没有关机、清理、恢复他人队列的授权。

SSH仅wm-vgpu-1011；新根/root/autodl-tmp/drivingeditor_repro/campaign/instance_editor_20261013，Python /root/autodl-tmp/envs/drivingeditor/bin/python。原矩阵hybrid30k_20261012与followup_scene_prep_20261013、全部checkpoint/rawFP32/输入/失败尝试保留。原静态分配与车影诊断见[F01](research_failures/entries/V82-F01.md)，环境能力迁移教训参见[F03](research_failures/entries/V82-F03.md)；本轮首次CPU解释器缺SciPy的partial保留，改已有解释器成功，无共享环境安装。failure_ledger_refs=[V82-F01,V82-F02,V82-F03]，本轮delta=none。保留scripts/run_worldsim_v72_e3_render_bridge.py无关dirty，子agent默认gpt-6.1-sol/max。
