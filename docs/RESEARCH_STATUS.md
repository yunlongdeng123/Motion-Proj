# 当前研究状态

更新：2026-11-22（Asia/Singapore）。

`VEHICLE-ROAD-JOINT-20261121`：本轮执行与交付收口，[实际路线与组件](autoresearch/vehicle_road_joint_20261121_r1/report.md)、[训练合同](autoresearch/vehicle_road_joint_20261121_r1/contract.json)、[49条细节记录](autoresearch/vehicle_road_joint_20261121_r1/delivery/detail_metrics.json)。Scene065删除前方a9白Mazda，保留并补全后方a20黑皮卡与道路；原生30K步终态转换为对象局部刚体GS，新增624个有界车体/轮胎先验高斯。下半主体持续遮挡，隐藏形状和材质为推断。车辆200步＋联合200步；失败smoke6、成功smoke9、独立恢复probe2另列，总417次，额外更新未写入最终资产。原训练封包与7阶段均保留，当前候选未替换既有道路基线。

训练、保存、完整状态恢复与独立进程五帧重载已验证；75行局部归属sidecar无新增训练，KEEP保持、外域变化小于1/255，但主要a9白块残影、先验轮胎粗糙及局部车辆退化仍在。物理投影盒与roadlegal零交集不代表真实贡献隔离：prior raw86/110合法道路区仍有457/482像素的车辆贡献大于1e-3，越界足迹进入道路教师域，画质与贡献隔离未通过。下一步优先核定跨域大足迹拆分/归属及受测距约束的对象表面，本轮不追加训练。最终资产取回402文件/58,547,980字节，正式源与小型记录选定归档；GPU保持开机。human_verdict/problem_solved=null，failure_ledger_refs=[V82-F01,V82-F06]，failure_ledger_delta=updated:V82-F06。
