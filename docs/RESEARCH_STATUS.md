# 当前研究状态

更新：2026-11-22（Asia/Singapore）。

`OMNI-HQ-VEHICLE-TEACHER-20261122`：[灰模结果与组件](autoresearch/omni_hq_vehicle_teacher_20261122/report.md)、[HTML报告](autoresearch/omni_hq_vehicle_teacher_20261122/index.html)、[预注册门槛](autoresearch/omni_hq_vehicle_teacher_20261122/audits/source/preregistered_gate.json)、[恢复入口](autoresearch/omni_hq_vehicle_teacher_20261122/restore_index.json)。高质量黑皮卡后3/4单图驱动冻结Omni，前3/4仅结构核对，CPU matte保持原RGB。原bbox `[1,.416,.5]` 与raw XYZ语义轴不符，仅固定同图、seed101、50步、CFG4.5、256作一次轴序修正重放为 `[.5,.416,1]`；原候选保留并单列工程资源。修正候选四轮／两轴、驾驶舱／货厢可辨，后轮、轮拱与底盘仍局部熔连。proper rotation＋统一尺度＋固定有界配准仅用raw86/110；全部1030实测代理点均值／P90为9.93／38.02cm，未过预注册7.5／15cm门，轮廓和各轴尺寸门通过。结论 `GREY_GATE_FAILED_NO_MATERIAL_NO_STUDENT`，材质0次、学生0步，真实观测锁定融合未实现，未替换场景。两次请求共100采样步、进程墙钟154.95秒、峰值分配7.68GiB；正式与工程重放分账。报告与43文件审计取回核对完成；既有车辆／道路历史保留，GPU保持开机。human_verdict/problem_solved=null，failure_ledger_refs=[V82-F06]，failure_ledger_delta=updated:V82-F06。

免责声明：参考图为车型先验；1030点按对象框与遮挡框保护筛选，为几何归属代理，未经语义逐点核验。轮廓只覆盖保守可见核心；86/110同时用于有界配准，属于拟合集，134/179/198未参与生成、配准或选择。本轮局部失败及原bbox工程问题不能据此判定Omni能力上限。
