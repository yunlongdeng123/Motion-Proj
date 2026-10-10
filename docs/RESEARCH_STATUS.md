# 当前研究状态

更新：2026-10-10（新加坡）。分支 `research/worldsim-v8.1-seen-to-scene`；SSH `wm-3090-1009`；checkout `/root/autodl-tmp/motion_proj_v81`；唯一task/run `WS-V81-SEEN-TO-SCENE-P1-20261009/r1`。

用户最新授权继续Seen-to-Scene正式复现：从7500完整原件新增2500至累计10000；每1000步保存，每2500步助手复盘是否有收益并决定下一段，全局每5000步复盘后清旧正式断点，保留最新与上一复盘点。每小时心跳检查，服务器保持开机，无human in loop。这覆盖此前7500后仅评估的范围，历史评估记录保留。[执行策略与组件图](v81/P1_REVIEW_CYCLES_R1.md)。P0工程闭环完成，P1训练体系与局部学习已建立，最终能力尚未验收，P2/P3未开始；正式四指标未计算、`protocol_verified=false`，不宣称论文复现或DELETE收益。

5000→7500新增2500更新，梯度有限、冻结梯度0；完整7500断点4.833GB在数据盘 `paper_bidirectional_m4/checkpoint_retained_step007500/`，phase/train软链不是备份。独立助手全部150帧审核：小外扩滑板、海豚和白鲸有局部进步，只有白鲸`.125`达到2分，其余五窗hold，整体hold。固定7500的零更新传播对照六窗均 `no_clear_gain`；普通支全150PNG精确重放，输入合同通过。条件coverage／latent／像素响应不等于结构动作收益，推理干预不能代替等预算重训。[评估与组件图](v81/P1_STEP7500_EVALUATION_R1.md)、[实测](v81/P1_STEP7500_EVALUATION_R1.json)。

66个新视频／1650帧与1650张PNG实际解码，三名gpt-6-sol/xhigh/no-fast助手直接审核全部编号图板。深色 `outputs/v81-paper-p1/step7500_review.html` 包含学习曲线、传播两支与中间条件；人工 `human_verdict=null`。此前file浏览限制未绕过，未宣称浏览器视觉QA。v77 DELETE与v81外扩不做跨任务排名；V77-F02只增补评审边界，无新增根因或方法族否定，卡更新后重建索引。

新训练断点写数据盘`paper_bidirectional_m4/review_cycles/train`，首轮8000／9000／10000，训练器不提前trim。10000完成验证和复盘后执行明确清单清理，诊断副本、P0、原RGB、评分和失败媒体不清。此前空间清理释放105.95GB；最新实际数据盘余197.45GB，不将余量变化归因于本轮。[此前清理与恢复](v81/STORAGE_CLEANUP_20261010.md)。Git仅代码和轻量结果，源码ZIP小于100MB。

当前正式7500→10000已实际启动，父64754／trainer64820；启动快照已完成7501..7508八次更新，最近loss与梯度有限、冻结梯度0，GPU仅此trainer，约6秒/步。PID与步数需实时核对，快照不是实时状态。[启动实测](v81/P1_REVIEW_CYCLES_R1.json)。新每小时心跳`v8-1-p1-2500`已ACTIVE，负责复盘、清理与下一段调度；控制器每段只运行2500及六窗，随后退出等待助手收益gate，禁止无gate长训。质量hold不自动等同于研究停止；有局部结构/动作学习依据可继续，无明确收益先自主有界排查。周额度实读剩93%，UTC08:10已消费一张重置卡，余一张；再次实际剩余<3%才按授权使用，不重复消费。
