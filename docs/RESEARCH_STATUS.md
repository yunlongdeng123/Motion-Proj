# 当前研究状态

更新：2026-10-10（新加坡）。分支 `research/worldsim-v8.1-seen-to-scene`；SSH `wm-3090-1009`；checkout `/root/autodl-tmp/motion_proj_v81`；唯一task/run `WS-V81-SEEN-TO-SCENE-P1-20261009/r1`。用户授权自主完成P1复现，无human in loop；质量hold后继续有界排查，不等用户、不随便关AutoDL。本次保持开机，旧关机证据只作历史记录。

P0工程闭环完成；P1尚未通过，P2/P3未开始。正式paper-bidirectional-m4从500恢复模型/Adam/scheduler/全部RNG，原协议追加500至总1000，实际432训练视频、5.95秒/步、峰18.41GiB；所有记录梯度有限、冻结梯度0。相同三valid×两倍率六窗全部25帧生成，两名gpt-6-sol/xhigh/no-fast助手审完150帧，质量hold。白鲸局部轮廓改善，仍不随位置/尺度/姿态；滑板视角与跳跃、海豚摆尾仍未恢复。中心真实硬写回不算原生保持。[实测与组件图](v81/P1_PROGRESS_AUDIT_R1.md)。

随后完成零训练完整GT逐帧latent oracle两支自由采样：固定正式1000首个valid00f88c4f0a/.33、seed2026/25步，只替换CFG有条件condition，其他Gaussian/CLIP/time/RAFT/可见VAE输入逐值相同，普通支重放与正式1000原始25PNG完全一致。58.26秒、峰9.47GiB。独立全25帧审核发现非法GT condition确实改变构图，却仍无可辨滑板者或正确视角/跳跃。GT VAE三帧重建保留姿态，不能外推完整时序。非法GT不能当正式QUERY或方法收益，也不能唯一定位传播或否定U-Net。

下一项是同权重、同x_t/sigma/CLIP/time的配对单步EDM x0诊断，并导出c_skip*x_t基线，避免把输入保留的GT当网络恢复；CPU准备中。它是有界定位，不改正式训练协议、不加sigma网格/新网络/P2/P3，不默认追加100K。此前固定噪声/visible CLIP/重采样sigma/visible flow短64均保留，诊断权重不混入正式恢复。

1951有效YouTube训练视频与六数据包在；DAVIS90/附录YT60名单不改，valid排除正式60。正式四指标未计算、protocol_verified=false；GT训练条件/可见QUERY、单3090 bf16、YT60 valid_all_frames来源与论文official test名称、FVD两倍率汇总差异均明示。[正式协议](v81/YOUTUBE_VOS_P1_R1.md)。关键100/500/1000断点、失败媒体和所有输入保留。两个完成的旧诊断权重外移系统盘、原路径symlink与恢复记录保留，非外部耐久备份。完成1000后数据盘约3.3GB、系统盘约6.5GB，下一次保存权重前须核对空间。

本地 `outputs/v81-paper-p1/condition_gap_review.html` 保留所有历史及新增500/1000五列六窗、非法oracle两支原生/写回。新增24+8视频实际解码，完整PNG/图板/独立评分留同run；代码、轻量实测入Git，媒体/权重/外部源码不入Git，源码ZIP<100MB。human_verdict=null；failure_ledger_refs=[V77-F02]（证据边界）、failure_ledger_delta=none，本轮不新增方法否定。周额度最近余13%、2张重置卡，未触发3%阈值，未消费重置卡。15分钟监控v8-1-p1持续，正常无变化静默；完整目标未达成。[轻量结果](v81/P1_CONDITION_GAP_GPU_R1.json)。
