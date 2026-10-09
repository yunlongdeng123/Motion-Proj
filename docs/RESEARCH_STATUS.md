# 当前研究状态

更新：2026-10-10（新加坡）。分支 `research/worldsim-v8.1-seen-to-scene`；SSH `wm-3090-1009`；checkout `/root/autodl-tmp/motion_proj_v81`；唯一task/run `WS-V81-SEEN-TO-SCENE-P1-20261009/r1`。用户已重新开GPU，最新要求按完整目标持续auto research；质量不通过后继续定位与有界对照，不作为全任务停止点。本次保持开机，不执行旧关机安排。周额度最新余15%、可用重置卡2张，尚未触发此前3%阈值。

P0工程闭环完成；P1尚未通过，P2/P3未开始。旧AllFrames/reference_m4各1000 hold及原断点/输出保留。新paper-bidirectional-m4修正双向pull/源mask、Adam/wd0；同协议100→500已完成追加400步与6窗25帧验证。独立两名助手全帧审核hold：原生未可靠跟随输入结构/运动；滑板/.33唯一匹配100→500未见结构进步，不自动扩长训。数值/梯度正常不替代生成质量。

固定输入64步诊断已完成，不计入正式500或泛化评测。同clip/latent/noise/GT flow与CLIP，实际64更新，514份Adam状态都增加64，无冻结参数梯度。teacher单sigma加权MSE 0.191143→0.149839；独立前后25帧Gaussian QUERY按可见RGB生成，权重、优化器与RNG另存。teacher与QUERY角色明确分开，独立审核只支持固定片段纹理有限改善，原生主体动作与接缝仍失败。

此前18个单步条件隔离与共享Gaussian CLIP对照均完成：完整首帧只改变粗色彩，未单独恢复结构；后续帧确实到达融合并影响输出。原始SVD1024×576标准fp16/CPU卸载有鸟体结构，256²两精度没有；尺寸/画幅/精度同时变化，不作单因素归因。完整UNet与现fp16转FP32全部1428张量/15.25亿值初值相同，实际可训练416张量也等价，不为variant差异重训。

数据6包完整：1951有效train视频/19313窗口，DAVIS90+附录YT60固定，valid排除正式60。正式四指标未算，protocol_verified=false；YT60 valid_all_frames与论文official test名称、FVD两倍率汇总仍有口径差距。单3090/bf16与完整GT训练条件/可见QUERY差异继续明示。按既有存储授权，P0旧step1和已完成固定噪声probe完整权重外移至系统盘，原路径保留符号链接和恢复记录；不是外部耐久备份。正式100/500源、数据、全部视频和诊断内容均保留。实际空间随保存变化，不能以移动文件逻辑大小冒充立即回收。

正常后续/重复首帧控制也已完成：同Gaussian/CLIP/time逐值相同，重放正常支与原QUERY全25帧逐像素相同；融合条件差.687231、原生差.058893，只证明有响应，独立审核仍未见正确动作。随后单因素将teacher首帧CLIP从完整GT改为visible，保持其余缓存逐值相同，从相同500起分别64更新；训练前QUERY完全相同。该支teacher .191211→.149171，实际64/64与514份Adam+64再次核实；独立审核未见相对完整GT CLIP支的主体结构或动作收益，未放行长训。正式论文训练没有被该诊断协议替换。

重采样diffusion sigma/epsilon控制已完成64/64 Adam更新，514状态各+64，冻结梯度0，QUERY-before与旧500的25张原生PNG完全相同。固定teacher .191143→.182955，独立全25帧图像审核未见明确主体结构/展翼收益，不把单片段64步阴性视作充分容量失败。实际sigma范围.055625–216.222702，仍固定GT flow、VAE posterior、条件噪声和CLIP且无CFG dropout；不是完整论文训练或全噪声覆盖证明。[噪声控制](v81/P1_DIFFUSION_NOISE_CONTROL.md)。

随后部署eaaff94c，只把FCNet输入改成visible RGB估计的RAFT flow，完整GT flow仍作损失监督，原teacher不变并另测visible-flow teacher。实际64/64完成，514状态各+64、64个sigma与上支逐值相同、GT监督flow逐值保留、QUERY-before全25帧完全相同。GT teacher .191143→.182593；visible teacher .191211→.182725。6-sol xhigh独立全25帧审核仍未见相对GT-flow支的主体结构/展翼收益；不以误差定质量。15项CPU测试本地/远端通过，独立代码复核修掉QUERY卸载后teacher评估的设备切换问题。状态见run/visible_flow_controller.json；权重保存独立格式，不混入正式训练。[光流控制与组件图](v81/P1_VISIBLE_FLOW_CONTROL.md)。

光流短控制收口后，已从正式500恢复原论文协议至总1000，仅追加500更新、相同六窗验证；不是接续任一诊断权重或一次放行100K。UTC21:43:19实测513步，父3613/子3616存活，真实命令/状态见run/bounded1000_state.json和paper_bidirectional_m4/controller_state.json。恢复模型/Adam/scheduler及全部RNG，数据/mask/seed/模块范围不改；预计训练约50分钟再六窗验证。到1000按全25帧独立审核后选择下一有依据的控制，质量hold不停止整体研究。15分钟静默监控v8-1-p1已接续，仅实质结果/失败/需行动时通知；本轮保持开机，不等人工开GPU。

本地 `outputs/v81-paper-p1/condition_gap_review.html` 展示500六窗、固定64/CLIP/重采样前后、后续帧控制与全部旧诊断；本轮新增16条诊断视频已完整解码，光流结果在完成审核后接入页面。模型/数据/外部源码/媒体不入Git，源码ZIP<100MB。完整复现目标未达成；人工verdict=null。[实测](v81/P1_CONDITION_GAP_GPU_R1.json) · [组件图与协议报告](v81/P1_PROGRESS_AUDIT_R1.md) · [正式评测协议](v81/YOUTUBE_VOS_P1_R1.md)。failure_ledger_refs=[V77-F02]，failure_ledger_delta=none。
