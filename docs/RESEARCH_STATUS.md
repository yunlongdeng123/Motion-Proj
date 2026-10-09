# 当前研究状态

更新：2026-10-10（新加坡）。分支 `research/worldsim-v8.1-seen-to-scene`；SSH `wm-3090-1009`；checkout `/root/autodl-tmp/motion_proj_v81`；唯一task/run `WS-V81-SEEN-TO-SCENE-P1-20261009/r1`。用户授权自主复现、subagent审核，无human in loop；本次GPU已开，计算/交付完成且没有其他作业时关机。此前额度余21%，未触发低于3%的重置。

P0工程闭环完成；P1尚未通过，P2/P3未开始。旧AllFrames/reference_m4各1000 hold及原断点/输出保留。新paper-bidirectional-m4修正双向pull/源mask、Adam/wd0；同协议100→500已完成追加400步与6窗25帧验证。独立两名助手全帧审核hold：原生未可靠跟随输入结构/运动；滑板/.33唯一匹配100→500未见结构进步，不自动扩长训。数值/梯度正常不替代生成质量。

固定输入64步诊断已完成，不计入正式500或泛化评测。同clip/latent/noise/GT flow与CLIP，实际64更新，514份Adam状态都增加64，无冻结参数梯度。teacher单sigma加权MSE 0.191143→0.149839；独立前后25帧Gaussian QUERY按可见RGB生成，权重、优化器与RNG另存。teacher与QUERY角色明确分开，独立审核只支持固定片段纹理有限改善，原生主体动作与接缝仍失败。

此前18个单步条件隔离与共享Gaussian CLIP对照均完成：完整首帧只改变粗色彩，未单独恢复结构；后续帧确实到达融合并影响输出。原始SVD1024×576标准fp16/CPU卸载有鸟体结构，256²两精度没有；尺寸/画幅/精度同时变化，不作单因素归因。完整UNet与现fp16转FP32全部1428张量/15.25亿值初值相同，实际可训练416张量也等价，不为variant差异重训。

数据6包完整：1951有效train视频/19313窗口，DAVIS90+附录YT60固定，valid排除正式60。正式四指标未算，protocol_verified=false；YT60 valid_all_frames与论文official test名称、FVD两倍率汇总仍有口径差距。单3090/bf16与完整GT训练条件/可见QUERY差异继续明示。数据盘保存两个诊断终态后仅3.6GB；按既有清理授权仅删非关键400中间断点，保留100/500及全部诊断终态，实际空闲恢复约8.1GB。

正常后续/重复首帧控制也已完成：同Gaussian/CLIP/time逐值相同，重放正常支与原QUERY全25帧逐像素相同；融合条件差.687231、原生差.058893，只证明有响应，独立审核仍未见正确动作。随后单因素将teacher首帧CLIP从完整GT改为visible，保持其余缓存逐值相同，从相同500起分别64更新；训练前QUERY完全相同。该支teacher .191211→.149171，实际64/64与514份Adam+64再次核实；独立审核未见相对完整GT CLIP支的主体结构或动作收益，未放行长训。正式论文训练没有被该诊断协议替换。

GPU短计算已完成，无后续自动训练队列；先交付/推送，再检查全部真实进程和控制器并按授权关机。下一步CPU准备围绕固定单片段的完整训练噪声覆盖：当前64只拟合单sigma，尚未验证能覆盖Gaussian QUERY的全部去噪档位，不能据此否定充分训练的容量。保持论文架构/损失/数据，不改P2/P3，不自动开100K。

本地 `outputs/v81-paper-p1/condition_gap_review.html` 展示500六窗、两组固定64前后、后续帧控制与全部旧诊断；模型/数据/外部源码/媒体不入Git，源码ZIP<100MB。完整复现目标保持active；人工verdict=null。[实测](v81/P1_CONDITION_GAP_GPU_R1.json) · [组件图与协议报告](v81/P1_PROGRESS_AUDIT_R1.md) · [正式评测协议](v81/YOUTUBE_VOS_P1_R1.md)。failure_ledger_refs=[V77-F02]，failure_ledger_delta=none。
