# 当前研究状态

更新：2026-10-10 00:30（新加坡）。分支 `research/worldsim-v8.1-seen-to-scene` 继承 v7.7 的 `000ad1f0`；主机 `wm-3090-1009`，checkout `/root/autodl-tmp/motion_proj_v81`。唯一 task/run：`WS-V81-SEEN-TO-SCENE-P1-20261009/r1`，活动子阶段 `reference_m4`。用户授权完整论文复现、subagent审核、无human in loop；最新要求先核对数据/训练与原文，不能七天后才发现基础问题。[协议](v81/YOUTUBE_VOS_P1_R1.md) · [审计与组件图](v81/P1_PROGRESS_AUDIT_R1.md)。

P0工程闭环完成但仅两步，非论文能力或DELETE收益。P1旧公开All Frames路径已完成1000步、3条固定valid×两倍率及1条现成feedforward。两个独立6-sol/xhigh助手检查原生与硬合成全部25帧：相比step2已有天空、建筑或动物结构，但仍形变、重复、近乎冻结和边界断裂，画质未过。旧质量门hold，1000断点和全部原输出保留；不把可见中央硬写回当生成成功。

同条件采样诊断已完成：一次缓存CLIP/RAFT/FCNet/传播latent与初始噪声，仅切公开路径、无反演+标准CFG、反演首支+标准CFG。公开重放25帧逐像素一致；另两模式相对公开RGB MAE为0.00721895/0.00128656，独立完整25帧审核未发现结构修复。该单窗未支持“修采样即可修好画质”，也不证明公开inverse参数化无害；结果和边界在run的sampler_controlled/step001000。

已修正论文m=4参考链训练入口：仅可见中心选参考对、非相邻成对RAFT、同序flow_pairs_info传播、按实际source/destination的两方向warp监督。不能只改传播参数而仍用相邻RGB监督。P1静态双侧mask显式检查；日志/断点记录协议，旧无字段checkpoint视为All Frames，跨协议恢复报错。49项CPU测试通过。reference_m4从原始SVD XT1.1、RAFT、ProPainter光流权重新初始化，真实两步反向传播通过，损失6.4800/4.6685，三个可训练组件梯度有限非零、冻结梯度0，峰值分配21.318GiB；未载作者编辑权重。

旧闲置控制器35923已安全停止，旧state及handoff保留。新控制器40743：run根下reference_m4/controller_state.json，传播reference-m4、主推理paper-feedforward；先验证两步断点，再恢复至新1000步和5000步助手质量门，另保留公开literal诊断。旧1000不计入论文参考链训练。当前只到新1000检查点，不无条件放行100K；不等待人工审核、不加入P2/P3创新。

数据6包完整校验提取，不重下。train3471视频/94588JPEG，1951严格>25帧、19313连续窗口。JPEG文件名每5递增，包内连续25张不等于原视频逐帧，作者实际训练采样率尚未确认。正式评测固定DAVIS90+附录YT60全帧，同一150ID/两倍率；验证排除正式60ID，不用test调参。单3090/bf16/CPU卸载、公开训练GT flow/首帧完整CLIP、AdamW与论文Adam、fps差异明示；protocol_verified=false。

审核HTML同步本地outputs/v81-paper-p1，包含真实训练输入、源图/mask、旧1000的7窗及同条件3窗；50视频全部逐帧解码，78本地链接无缺失。新参考链结果按独立阶段标注补入。尚无正式PSNR/SSIM/LPIPS/FVD，不宣称论文指标或DELETE收益。

数据、模型、外部源码、视频与完整断点不入Git，源码ZIP上限100,000,000 bytes。当前不关机。failure_ledger_refs=[V77-F02]，failure_ledger_delta=none；工程/协议差距记录到同run，不伪造方法失败。
