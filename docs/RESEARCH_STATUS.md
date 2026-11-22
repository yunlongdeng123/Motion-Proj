# 当前研究状态

更新：2026-11-22（Asia/Singapore）。

`A20-JOINT-RECONSTRUCTION-20261122`：[统一高斯场报告](autoresearch/a20_joint_reconstruction_20261122/report.md)、[HTML对照](autoresearch/a20_joint_reconstruction_20261122/index.html)、[实际执行账目](autoresearch/a20_joint_reconstruction_20261122/actual_execution_ledger.json)、[恢复索引](autoresearch/a20_joint_reconstruction_20261122/restore_index.json)。归档完成，车辆灰模与场景交接未通过。原始a20 8925行与TRELLIS原生410080行统一为同一可学习对象坐标场，无重采样；两源分别8014/310240行位置改变，几何阶段颜色冻结。三次从共同初始化独立训练40步，每次另有10smoke+2丢弃重放+1丢弃材质探针+1丢弃联合探针，共正式120、额外42、实际162次更新；所选资产为第三次40步+整场移除检验0步。累计退休旧源811，有效旧8114+T410080=418194/总419005。246个不透明面返回点的渲染期望距离MAE/P90为0.006290/0.013866米；单对象可见区覆盖99.80%/99.50%，场景内可见贡献15.50%/34.38%，其他车辆RGB变化171/359像素。六视灰模仍有前轮长杆与后窗/尾门异常，材质和联合精修正式均0；scene/road/其他对象/背景更新0。完整actor/Adam/RNG恢复、本地资产SHA/finite/来源行ID及202界面状态QA已核验。旧固定贡献与24k/410k历史组保留。scene_qualified=false，root_report_complete=true；human_verdict/problem_solved=null；failure_ledger_refs=[V82-F06]；failure_ledger_delta=updated:V82-F06。GPU按用户要求保持开机。

免责声明：测距为alpha归一化的原生渲染期望距离，不能等同整体几何精度；期望深度法向为深度梯度诊断。数值满足与灰模、场景验收分开；仅BUILD86/110进入本轮监督，封存帧未打开；三次40步是独立试验，不是同一资产连续120步。
