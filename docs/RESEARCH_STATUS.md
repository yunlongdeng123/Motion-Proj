# 当前研究状态

更新：2026-10-09 23:33（新加坡）。分支 `research/worldsim-v8.1-seen-to-scene` 继承 v7.7 的 `000ad1f0`；主机 `wm-3090-1009`，checkout `/root/autodl-tmp/motion_proj_v81`。当前唯一 task/run：`WS-V81-SEEN-TO-SCENE-P1-20261009/r1`。用户要求完整论文复现、subagent 审核、不设置 human in loop；最新要求先看数据、进度与原文差距，避免七天后才发现基础问题。[P1 协议](v81/YOUTUBE_VOS_P1_R1.md) · [短周期审计及组件图](v81/P1_PROGRESS_AUDIT_R1.md)。

P0 工程闭环已完成，但仅两步优化，原生输出结构失真，不是论文能力或 DELETE 收益。P0 输入、两个断点和18视频保留在旧 run；P1 从原始 SVD XT1.1、RAFT、ProPainter 光流补全重新初始化，不恢复 P0 或作者编辑权重。[P0 结果](v81/P0_R1_RESULTS.json)。

数据已按用户指定切到 HF-Mirror 固定 YouTube-VOS2019 镜像，停止已确认的三个旧 Google worker并保留部分下载/失败证据。train、valid、test、DAVIS2017和额外全帧valid双卷全部通过完整校验并提取。train共3471视频/94588 JPEG，1951视频满足公开loader的>25帧。论文附录60 ID全在2019valid；已改用原始valid_all_frames，所选60条均36–180帧。独立subagent确认150条冻结基准精确匹配DAVIS90+附录60，不替换ID、不重复帧。下载来源和完整150条清单在P1 run。

23:31快照739步，最近100步5.823秒/更新、数值和梯度有限，纯训练剩余估算6.69天；画质不能据损失判断。训练PID29673仍完成既定1000步。已保留旧控制器状态并仅替换其父控制器29301，交接PID33980等待完整1000断点后启用新版控制器，GPU训练未中断。新版1000/5000步均跑固定3个valid×两倍率后等待assistant质量门，1000另做feedforward诊断，不自动进入后续长训。8项质量门CPU测试通过；既有CPU33项曾通过。审核页同步本地outputs/v81-paper-p1，含真实step1/2/300输入、源图、mask及已完成生成视频。

两个独立6-sol/xhigh subagent确认公开代码与论文存在实质差距：当前训练实际是All Frames顺序传播，未采用论文m=4参考链；公开test.py的inverse存在B1→B2广播/CFG配对与v_prediction当epsilon使用风险。先固定1000断点比较literal/feedforward，但两现成路径另有洞区填值/RNG差异，不能直接作因果归因。训练EDM/VAE标度未发现本仓库独有的确定错配。训练GT flow/首帧CLIP、AdamW及推理fps差异均继承公开代码；单3090/bf16/CPU卸载明示。protocol_verified=false；未经短诊断不盲目放行100K。

正式指标按固定FollowYourCanvas代码，PSNR/SSIM/LPIPS AlexNet/FVD I3D，前16帧，两倍率分别报告及平均，原生与硬合成分开；依赖已就绪，尚无P1正式指标。最新完成生成仍step2：独立subagent查全部25帧，原生强色彩/结构失真，硬合成中央逐像素保留可见输入；不能以两步判定最终失败。下一步等1000完整断点，短推理与独立审核，必要时保持hold并修正有证据的协议/工程问题，不改测试名单讨好分数。assistant_verdict独立，human_verdict留空；验证例排除附录60测试ID，不等待人工审核。

数据、模型、外部源码、视频和完整断点均不入Git，源码ZIP上限100,000,000 bytes。当前不关机；没有复用旧v7.7实验或启动创新训练。failure_ledger_refs=[V77-F02]，failure_ledger_delta=none；工程/数据缺口记录到本run，不伪造方法失败。
