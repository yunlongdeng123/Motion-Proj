# 当前研究状态

更新：2026-10-11（新加坡）。分支 `research/worldsim-v8.1-seen-to-scene`；SSH `wm-3090-1009`；checkout `/root/autodl-tmp/motion_proj_v81`；P1 task/run `WS-V81-SEEN-TO-SCENE-P1-20261009/r1`。

用户最新优先级：Seen-to-Scene完成在途10000完整checkpoint及六窗验证后，先不继续12500，GPU空闲运行已下载的DGGT Waymo官方推理与车辆编辑。用户已确认VGGT是口误，并授权按论文补公开CLI尚未接入的高斯删除、平移与插入，不能因缺CLI停在能力说明。每小时检查，服务器保持开机，无human in loop。[DGGT流程与证据](v81/DGGT_WAYMO_INFERENCE_R1.md)。此前每1000保存、2500复盘与5000倍数清理规则保留；此阶段不根据continue gate自动恢复S2S。[P1策略](v81/P1_REVIEW_CYCLES_R1.md)。P0闭环完成，P1训练体系与局部学习已建立，最终能力尚未验收，P2/P3未开始；四指标未计算、`protocol_verified=false`。

5000→7500新增2500更新，梯度有限、冻结梯度0；完整7500断点4.833GB在数据盘 `paper_bidirectional_m4/checkpoint_retained_step007500/`，phase/train软链不是备份。独立助手全部150帧审核：小外扩滑板、海豚和白鲸有局部进步，只有白鲸`.125`达到2分，其余五窗hold，整体hold。固定7500的零更新传播对照六窗均 `no_clear_gain`；普通支全150PNG精确重放，输入合同通过。条件coverage／latent／像素响应不等于结构动作收益，推理干预不能代替等预算重训。[评估与组件图](v81/P1_STEP7500_EVALUATION_R1.md)、[实测](v81/P1_STEP7500_EVALUATION_R1.json)。

66个新视频／1650帧与1650张PNG实际解码，三名gpt-6-sol/xhigh/no-fast助手直接审核全部编号图板。深色 `outputs/v81-paper-p1/step7500_review.html` 包含学习曲线、传播两支与中间条件；人工 `human_verdict=null`。此前file浏览限制未绕过，未宣称浏览器视觉QA。v77 DELETE与v81外扩不做跨任务排名；V77-F02只增补评审边界，无新增根因或方法族否定，卡更新后重建索引。

新训练断点写数据盘`paper_bidirectional_m4/review_cycles/train`，首轮8000／9000／10000，训练器不提前trim。10000完成验证和复盘后执行明确清单清理，GPU正跑DGGT时延后；诊断副本、P0、原RGB、评分和失败媒体不清。此前空间清理释放105.95GB；UTC16:36实际数据盘余121.91GB、系统1.65GB，不将余量变化归因于本轮。[此前清理与恢复](v81/STORAGE_CLEANUP_20261010.md)。Git仅代码和轻量结果，源码ZIP小于100MB。

当前正式7500→10000父64754／trainer64820继续运行；UTC16:44快照9762步，最近loss与梯度有限、冻结梯度0，GPU仅64820，约5.9秒/步。8000/9000已保存；10000、六窗和复盘待完成，快照不是实时状态。DGGT独立run `WS-V81-DGGT-WAYMO-INFERENCE-20261011/r1`已完成官方CPU预处理199帧和SegFormer前视8帧，实际启动等待队列父79455，CPU依赖齐全，GPU推理尚未执行。它等P1完整10000、六窗和父子退出后持共享GPU锁串行官方基线、四分支高斯编辑与Difix。已根据RGB排除语义ROI邻车误分，逐帧mask为人工编辑控制，自动3D实例关联未验证；需核对实际输出；未运行不宣称编辑成功。[轻量状态](v81/DGGT_WAYMO_INFERENCE_R1.json)。心跳`v8-1-p1-2500`已改为每小时监控DGGT优先队列，不自动12500。周额度实读剩91%，已用一张重置卡、余一张；再次实际剩余<3%才用，不重复消费。
