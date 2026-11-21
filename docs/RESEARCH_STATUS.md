# 当前研究状态

更新：2026-11-21（Asia/Singapore）。

`ROAD-SURFACE-RESIDUAL-20261121-R2`：认证残余修复完成，[报告与组件](autoresearch/module2_surface_residual_20261121_r2/report.md)、[合同](autoresearch/module2_surface_residual_20261121_r2/contract.json)、[指标](autoresearch/module2_surface_residual_20261121_r2/bench/metrics.json)。直接父为第一轮外观终态，实际激活13个旧inactive锚点、追加1071个新空间位置，并在原不可变尺度上界内复用107个第一轮新增行；不追加同中心副本。道路总行596100、active162348；累计两轮真append7239、旧行activation3121，历史892另列。覆盖纯hinge400步、外观1200步；这是残余采样与覆盖目标共同变化，不是仅密度消融。

所有安全动作的源像素均达到alpha>=.95；认证道路C低alpha总数raw86/110从第一轮前3895/7431降至809/303，S从520/2810降至0/0。剩余认证C缺口全属于本轮实际joint K拒绝候选，仍不能推导所有3DGS几何不可行。未认证C编辑域5934/7077保留，不称已确认道路。五指定视角K的RGB/alpha exact0、逐字段父prefix保护和完整状态恢复通过；不外推任意相机或连续参数区间。固定最终几何的4F/0更新颜色0/1区间诊断独立记账，只支持逐像素RGB盒区间下界，不是HF或全局容量证书。另在同1191行、原normalizer和目标下独立核验有界L-BFGS-B，实际2FG/1接受步，总loss只降4.51e-8；总目标含冻结父区域常数，不能据此认定新覆盖颜色收敛或3DGS达到上限。正式1200步终态不替换。原第一轮提交与资产保留；未接受整体画质、未晋升默认场景。human_verdict/problem_solved=null，failure_ledger_refs=[V82-F06]，failure_ledger_delta=updated:V82-F06。GPU保持开机。
