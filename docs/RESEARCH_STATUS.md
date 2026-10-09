# 当前研究状态

更新：2026-10-09。分支 `research/worldsim-v8.1-seen-to-scene` 继承 v7.7 的 `000ad1f0`；主机 `wm-3090-1009`，checkout `/root/autodl-tmp/motion_proj_v81`。当前唯一 task/run：`WS-V81-SEEN-TO-SCENE-P1-20261009/r1`。用户要求完整论文复现、subagent 审核、全过程不设置 human in loop。[P1 协议及组件图](v81/YOUTUBE_VOS_P1_R1.md)。

P0 工程闭环已完成，但仅两步优化，原生输出结构失真，不是论文能力或 DELETE 收益。P0 输入、两个断点和18视频保留在旧 run；P1 从原始 SVD XT1.1、RAFT、ProPainter 光流补全重新初始化，不恢复 P0 或作者编辑权重。[P0 结果](v81/P0_R1_RESULTS.json)。

数据已按用户指定切到 HF-Mirror 固定 YouTube-VOS2019 镜像，停止已确认的三个旧 Google worker并保留部分下载/失败证据。train、valid、test、DAVIS2017和额外全帧valid双卷全部通过完整校验并提取。train共3471视频/94588 JPEG，1951视频满足公开loader的>25帧。论文附录60 ID全在2019valid；已改用原始valid_all_frames，所选60条均36–180帧。独立subagent确认150条冻结基准精确匹配DAVIS90+附录60，不替换ID、不重复帧。下载来源和完整150条清单在P1 run。

P1真实GPU两步预检通过，三个可训练组件梯度有限/非零、冻结梯度0；第2步allocated峰值21.26GiB。公开推理首次因内外FP16/BF16 autocast冲突退出，保留失败日志；literal-public改为与源码一致的FP16并禁用外层autocast缓存，不改反演/CFG算法。25帧原生/硬合成验证已完成。控制器PID29301从第2步断点成功恢复，正在训练至1000步，然后按1000步保存/固定valid验证继续100K。初测每步约5.7–6秒，100K训练约7天，保存/验证另加；这是短期吞吐估计，后续更新。完整v8.1 CPU测试33 passed。

训练沿用公开源码GT flow/首帧CLIP条件、masked RGB条件VAE，AdamW1e-5；P0的可见条件训练不同，不能混报。推理使用固定官方m=4参考链和literal-public pipeline，隐藏真值不进QUERY条件。论文Adam/纯Gaussian描述与源码AdamW/额外inverse遍历存在差异；论文也未公开完整评测细节，结果明示protocol_verified=false，不凭指标接近宣称完全等价。RTX3090采用bf16/冻结编码器CPU卸载适配，源码内部fp16行为另记。

正式指标按固定FollowYourCanvas代码，PSNR/SSIM/LPIPS AlexNet/FVD I3D，前16帧，两倍率分别报告及平均，原生与硬合成分开；依赖已就绪，尚无P1正式指标。独立subagent已查看step2的f00/f12/f24及全部25帧联系图：原生强色彩/结构失真，时序明显变化；硬合成中央逐像素保留可见输入。工程链路通过不代表生成质量合格，两步尚未收敛，继续原定预算。assistant_verdict单独记录，human_verdict留空；验证例排除附录60测试ID，不等待人工审核。

数据、模型、外部源码、视频和完整断点均不入Git，源码ZIP上限100,000,000 bytes。当前不关机；没有复用旧v7.7实验或启动创新训练。failure_ledger_refs=[V77-F02]，failure_ledger_delta=none；工程/数据缺口记录到本run，不伪造方法失败。
