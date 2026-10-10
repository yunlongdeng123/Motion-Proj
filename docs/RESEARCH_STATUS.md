# 当前研究状态

更新：2026-10-10（新加坡）。分支 `research/worldsim-v8.1-seen-to-scene`；SSH `wm-3090-1009`；checkout `/root/autodl-tmp/motion_proj_v81`；唯一task/run `WS-V81-SEEN-TO-SCENE-P1-20261009/r1`。

用户最新范围为7500完整checkpoint保存后停止新增训练，完成5000／7500质量、传播收益和v77/failure对照。训练与评估已完成，服务器保持开机，不恢复训练。P0工程闭环完成，P1整体未通过，P2/P3未开始；正式四指标未计算、`protocol_verified=false`，不宣称论文复现或DELETE收益。

5000→7500新增2500更新，梯度有限、冻结梯度0；完整7500断点4.833GB在数据盘 `paper_bidirectional_m4/checkpoint_retained_step007500/`，phase/train软链不是备份。独立助手全部150帧审核：小外扩滑板、海豚和白鲸有局部进步，只有白鲸`.125`达到2分，其余五窗hold，整体hold。固定7500的零更新传播对照六窗均 `no_clear_gain`；普通支全150PNG精确重放，输入合同通过。条件coverage／latent／像素响应不等于结构动作收益，推理干预不能代替等预算重训。[评估与组件图](v81/P1_STEP7500_EVALUATION_R1.md)、[实测](v81/P1_STEP7500_EVALUATION_R1.json)。

66个新视频／1650帧与1650张PNG实际解码，三名gpt-6-sol/xhigh/no-fast助手直接审核全部编号图板。深色 `outputs/v81-paper-p1/step7500_review.html` 包含学习曲线、传播两支与中间条件；人工 `human_verdict=null`。此前file浏览限制未绕过，未宣称浏览器视觉QA。v77 DELETE与v81外扩不做跨任务排名；V77-F02只增补评审边界，无新增根因或方法族否定，卡更新后重建索引。

100／500／1000／2000／5000／7500及失败媒体保留；512诊断只保留模型，不可精确续训。清理已释放105.95GB，原RGB／标注／评分保留，不重下已解压归档。[清理与恢复](v81/STORAGE_CLEANUP_20261010.md)。Git仅代码和轻量结果，源码ZIP小于100MB。

三项评估交付后结束本轮监控，没有后续自动训练队列。周额度UTC08:10按授权消费一张重置卡，余一张，原始记录在work/v81-quota-reset，不重复消费。
