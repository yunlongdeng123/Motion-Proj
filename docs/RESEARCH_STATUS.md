# 当前研究状态

更新：2026-10-09T18:28:08.961841+00:00（UTC）。分支 `research/worldsim-v8.1-seen-to-scene`；主机 `wm-3090-1009`，checkout `/root/autodl-tmp/motion_proj_v81`。唯一task/run `WS-V81-SEEN-TO-SCENE-P1-20261009/r1`。用户授权自主复现和subagent审核、无human in loop；GPU作业结束且无其他任务时关机。周额度上次实际剩26%，未触发低于3%的重置要求。

P0工程闭环完成；P1尚未通过，P2/P3未开始。旧AllFrames与reference_m4分别1000步hold，原产物/断点保留。新paper-bidirectional-m4修正论文双向pull/源mask，Adam/wd0；第二步OOM通过完整时轴激活重算及等价加权ternary分块修复。新正式阶段只到100步；25帧验证与36帧全视频均完成，独立全帧审核仍不合格。没有排队1000/100K。

已完成用户要求三项诊断：VAE/传播/原生中间解码、共享CLIP/时间/增强噪声/初始latent的正常后续帧对重复首帧、单训练片段32步teacher与纯Gaussian QUERY。后续帧确实进入融合条件并影响最终输出；可见中心在VAE与fused仍保留，最终native失真。不能断言“完全不看后续帧”，也不能把响应当质量。孤立past/future解码较弱，不足以诊断某模块故障。

固定训练片段32步不计入正式100步或泛化评测；teacher带噪真值、GT flow/完整首帧CLIP，QUERY仅visible。teacher误差轻微下降，结构结果以独立审核为准。原32步诊断未保存最终权重，保留日志/图像/seed与源100断点；新入口补保存供后续复核，不为此重复计算。[实测](v81/P1_CONDITION_DIAGNOSTICS_R1.json) · [协议](v81/YOUTUBE_VOS_P1_R1.md) · [审计与组件图](v81/P1_PROGRESS_AUDIT_R1.md)。

下一步先在固定teacher/noise条件中隔离训练与QUERY的CLIP/flow差距和去噪响应，再决定有界学习预算。当前100+32不足以否定模型容量；不凭loss下降直接追加长训。保留外扩任务、数据/seed/正式ID，不加入P2/P3或作者编辑权重。

数据6包完整、train1951有效视频/19313窗口；DAVIS90+附录YT60固定、valid排除正式60。full-video入口已完成36帧预检，未证明所有长片显存足够。正式四指标未算，预处理/FVD口径仍有未知，protocol_verified=false。CPU全套84项通过只代表工程验证。媒体/数据/权重/外部源码不入Git，源码ZIP<100MB。

审核页同步本地outputs/v81-paper-p1，包含100步短/全视频、完整条件路径与固定片段前后原生/写回。GPU作业已完成；保存/推送和确认所有进程、队列为空后按本次授权关机，不停止无关工作。failure_ledger_refs=[V77-F02]，failure_ledger_delta=none；本轮不是方法科学否定。
