# 当前研究状态

更新：2026-10-17（Asia/Singapore）。用户禁止subagent/新chat。DrivingEditor⑥车辆资产混入道路仍未验收；①②④未开始。最新run为idsplat_frontend_r13。[点集、架构、真实对应与边界](autoresearch/worldsim_v82/drivingeditor_shadow_rebuild_20261017/v3_idsplat_frontend_r13/README.md)，[实验索引](EXPERIMENTS.md)。

用户已停止B/C。A3000完整49x65输出与旧GN11+18k视觉对照已交付：A的背景/道路细节与白车有明显回退，仍带道路，不是改进模型。B已3000、视频导出被用户停止，1500/3000检查点完整；C0。旧控制器与后代已退出，不恢复旧E6/SAM2或B/C队列。

最后一次IDSplat前端+infra先验点集：同原始BUILD/同冻结共同SAM3或显式DEVA回退mask/固定GT/同implementation_r12。实际mask提升+深度+DBSCAN保护细部，DINOv3对46张真实训练图提特征并做56规划配对，47得官方RANSAC变换、9对应不足。DINO只独立对照，不覆盖GT。其他36Actor/静态种子、全49RGB/监督和GT均核验不变。

道路误入代理a5 1874→2、a26 483→8、a37 3835→165，6车保体达标；a12 body95.59%→75.98%、下部89.47%→60.53%，7车门槛未通过。主要缺口在共同mask提升，DBSCAN次要；a12没有正道路冲突返回。代理不是手工归属真值、真影比例或编辑成功率；已有QUERY用于开发，非新盲测。全部7车点图由本代理检查。没有据QUERY调参，也没有新训练/新编辑视频。

本轮新增场景更新0，历史累计19500..19549（包含弃用成本）；不为耗完预算继续。现阶段停止继续训练，保存代码与中文HTML/点集接口/待定来源，核对所有任务和锁等待退出且无其他GPU任务后按用户授权AutoDL关机，心跳暂停。是否实际关机以本地电源回执为准；未获用户继续指令不自动恢复。

delete范围仍仅5/6/12/18/25/26/37，旧0/1/2→5/26/18；其余4无旧独立资产只能外观对照。原GN11/v2/18k、原始数据、历史失败与检查点保留，用户dirty工作不动。human_verdict=null，problem_solved=false；failure_ledger_refs=[V82-F01]，本轮failure_ledger_delta=none。当前结论是最后前端的全车保体门槛未过，不是IDSplat整体无效或后端训练已验证失败。
