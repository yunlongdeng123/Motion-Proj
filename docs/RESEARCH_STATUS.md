# 当前研究状态

更新：2026-10-10 01:32（新加坡）。分支 `research/worldsim-v8.1-seen-to-scene`；主机 `wm-3090-1009`，checkout `/root/autodl-tmp/motion_proj_v81`。唯一 task/run：`WS-V81-SEEN-TO-SCENE-P1-20261009/r1`。用户授权持续完整复现、subagent审核、无human in loop；先核对论文差距与短周期实测，不盲目长训。[协议](v81/YOUTUBE_VOS_P1_R1.md) · [审计与组件图](v81/P1_PROGRESS_AUDIT_R1.md)。

P0工程闭环完成，非论文能力。旧All Frames完成1000步，固定验证7窗独立完整25帧QA未达可用画质。同条件3种采样诊断没有修复结构；旧阶段hold、断点/原输出保留。reference_m4从原始SVD XT1.1、RAFT、ProPainter重新初始化，目前到838步，约4.5秒/步，梯度有限、冻结梯度0；但传播仍继承固定公开模块，不能称论文双向参考传播已正确实现。

新的CPU反证：两帧点从x4移到x5，公开参考分支一支将未来帧内容拉到x6而非x4；仅过去帧可见时，末帧两支都得不到证据。已确认方向错误与两个分支共用未来父链，证据在同run/reference_direction_audit。reference_m4/quality_gates/step001000.json预置工程hold；完成1000完整保存及固定验证后，不放行5000。

已接入独立paper-bidirectional-m4候选：目标到最近过去/未来的direct flow、参考链组合、来源mask屏蔽、原FB一致性与原细化/融合网络。25帧最稀m4为42条flow而非旧24，静态FCNet mask扩至K+1；稀疏flow序列分布和显存仍需实测。新旧协议不允许交叉恢复，控制器跳过既存断点/结果也核对协议。独立接线审核通过，CPU全套59项通过。排队worker46090只等旧阶段完成保存/验证并hold，再串行执行新协议两步训练及一窗推理；不同时启动两个GPU任务，不自动放行1000/100K。新子阶段paper_bidirectional_m4，状态在根bidirectional_handoff.json；尚无该候选GPU结果。

数据6包完整、train1951有效视频/19313窗口。正式来源固定DAVIS90+附录YT60全帧，验证排除正式60ID；GT flow/首帧完整RGB CLIP沿公开训练，QUERY仅visible RGB。评测入口已修整套DAVIS或YT缺失仍被放行的漏洞。本地前25帧生成/前16帧评分只作为明确命名的本地协议，作者长视频/预处理/FVD汇总未确认；protocol_verified=false，四指标未实际计算，不能宣称论文表1已复现。

审核HTML本地outputs/v81-paper-p1保留12生成窗、54视频与数据/论文对照，新增CPU方向反证和工程hold；87本地链接无缺失，54视频全部解码通过。媒体、数据、模型、外部源码与完整断点不入Git；源码ZIP约34.3MB、上限100,000,000 bytes。当前不关机，不加入P2/P3、不加载作者编辑权重。failure_ledger_refs=[V77-F02]，failure_ledger_delta=none；这是复现工程/协议差距，非科学否定。
