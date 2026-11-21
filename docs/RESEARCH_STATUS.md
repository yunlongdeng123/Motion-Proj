# 当前研究状态

更新：2026-11-21（Asia/Singapore）。

`ROAD-SURFACE-ADAPTIVE-20261121-R1`：1152新教师三轮重训与覆盖驱动真增密两条固定链均完成，正在归档；[三轮训练](autoresearch/module2_common1152_20261121/report.md)、[表面覆盖与外观](autoresearch/module2_surface_adaptive_20261121/report.md)。教师固定legacy/T0 step600/seed101，不按学生误差选图。先启用3108个旧inactive锚点，再一次残余选择实际追加6168个普通道路高斯；覆盖200+400、外观1200步，父活动行几何与覆盖冻结。历史892是旧格点激活，不计新增几何。

认证道路C低alpha像素raw86/110由3895/7431降至1262/1000，S由520/2810降至43/63，仍未完全消除黑缝；认证C无支持1055/737继续保留。未认证C编辑域5934/7077单列，不能称已确认道路或用任意平面强填。最坏允许足迹和持久化保护仅覆盖五个指定视角，不外推任意相机。零更新诊断另将opacity乐观放宽至1（正式上界.999），只做5次前向，不改正式表或给出全局容量证书。未通过整体画质验收、未晋升默认场景；human_verdict/problem_solved=null，failure_ledger_refs=[V82-F06]，failure_ledger_delta=updated:V82-F06。GPU保持开机。
