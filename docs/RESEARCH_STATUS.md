# 当前研究状态

更新：2026-10-10（新加坡）。分支 `research/worldsim-v8.1-seen-to-scene`；主机 `wm-3090-1009`，checkout `/root/autodl-tmp/motion_proj_v81`；唯一task/run `WS-V81-SEEN-TO-SCENE-P1-20261009/r1`。用户重新开启GPU，授权自主复现与subagent审核，无human in loop；计算完成、交付且无其他作业时关机。周额度实际剩21%，两张重置卡可用，未触发低于3%的重置要求。

P0工程闭环完成；P1尚未通过，P2/P3未开始。旧AllFrames/reference_m4各1000步hold，原产物/断点保留。新paper-bidirectional-m4修正双向pull/源mask、Adam/wd0；OOM采用FCNet完整时轴激活重算及等价ternary分块修复。正式100步短/36帧全视频仍不合格。额外单片段32步属于诊断，末步权重未保存，不计入正式或泛化评测。

固定100权重18组CLIP×flow×实际sigma单步前向已完成，更新0；实际scheduler与CPU准备一致。独立助手看18张f00/f12/f24；完整CLIP改变粗色彩但未恢复结构，黑/灰RAFT没有稳定画质排序。低噪声输入含真实latent，不是自由生成成功。随后固定可见黑洞flow/传播/时间/采样器、完全相同Gaussian，只切CLIP的两条25帧QUERY也未恢复侧区；完整GT首帧不足以单独修复这一个样本。后续帧此前已证明到达融合并影响输出，不能称完全不看后续帧。

原始SVD三个sanity完成：256²全管线bf16与全FP32均结构突变；1024×576原始16:9首帧、标准fp16/CPU卸载全25帧有自然鸟体/枝叶/草地，独立最低sanity通过。尺寸、画幅与精度同时变化，不作单因素归因，也不是P1外扩成绩。

当前同协议从正式100断点恢复模型/Adam/RNG，限定总500步（追加400），数据/mask/seed/网络/学习率/参数范围不变。控制器4097、trainer4100，以实时进程为准；快照已记录112步，数值/三组件梯度正常。到500后生成原固定3例×2倍率并退出，助手全帧审核后决策，不自动1000或100K。关键100断点另以硬链接保留。控制器首次因审核JSON字段层级错配在启动训练前退出，保留栈后修正嵌套字段，不绕过审核判据。

数据6包完整，train1951有效视频/19313窗口；DAVIS90+附录YT60固定，valid排除正式60。正式四指标未算、protocol_verified=false。单3090/bf16、完整首帧CLIP/GT flow训练与可见QUERY差距继续明示；论文文字重建与逐行公开协议分开，无P2/P3创新。数据盘18GB空闲，当前短预算可容纳断点，长训前再核对空间。

本地 `outputs/v81-paper-p1/index.html` 保留历史，新 `condition_gap_review.html` 展示18前向、2条完整QUERY、3条SVD sanity与最新训练快照。GPU仍用于训练，旧关机提示已标历史；媒体/模型/数据/外部源码不入Git，源码ZIP<100MB。下一步评500相对100的结构/逐帧内容利用，再选择有证据的短控制；完整复现目标保持active。

[本轮GPU实测](v81/P1_CONDITION_GAP_GPU_R1.json) · [协议与组件图](v81/P1_PROGRESS_AUDIT_R1.md) · [固定评测协议](v81/YOUTUBE_VOS_P1_R1.md)。failure_ledger_refs=[V77-F02]，failure_ledger_delta=none；尚无方法科学否定，人工verdict始终留空。
