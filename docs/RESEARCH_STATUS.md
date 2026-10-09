# 当前研究状态

更新：2026-10-09。分支 `research/worldsim-v8.1-seen-to-scene` 继承 v7.7 的 `000ad1f0`。主机 `wm-3090-1009`，独立 `/root/autodl-tmp/motion_proj_v81` checkout；原v7.7代码与资产保留。

task/run：`WS-V81-SEEN-TO-SCENE-20261009/r1`。路线为原始 SVD + RAFT + ProPainter 光流补全，自有数据/训练/推理闭环；P0→P1论文外扩复现→P2驾驶真实显露DELETE基线→固定P2后P3。当前用nuScenes推进方法适配，不能混报YouTube-VOS论文复现。[路线](v81/SEEN_TO_SCENE_P0.md) · [r1协议与components图](v81/NUSCENES_P0_R1.md)。

CPU准备完成：6官方train+2官方val场景，共200张原始1600×900 CAM_FRONT RGB，保留sample_data.next链与sweeps；每段25帧约2秒，实际50/100ms混合、平均约12Hz。只提取缺失tgz成员，未整包解压；固定25帧、256²、左右各84像素外扩洞。20项CPU测试通过，真实官方传播组件小尺寸forward/backward通过；均不替代真实SVD优化验收。输入HTML在本地 `outputs/v81-nuscenes-p0/index.html`，16个输入视频解码通过。[输入证据](v81/P0_R1_INPUTS.json)。

原始SVD XT 1.1三个fp16组件已完整：换到ModelScope `shareAI/svd_1.1`，与官方固定revision的大小、SHA-256一致，下载成品再次验证；30秒实测约35.25MB/s。网络助手和临时文件不提交。三份YouTube-VOS（train.tar/test.zip/valid.tar）续传worker保留，尚无完整压缩包；详情 `r1/downloads/`，配额HTML不能算数据。收口数据盘约94.9GiB可用，不需清盘。

P0工程闭环已完成：固定6+2输入清单，真实优化第1步→退出并恢复至第2步→独立val scene-0562的25帧/25步生成→审核页。第1步传播梯度因合法条件dropout为skipped；第2步FCNet、传播细化、SVD时序参数均有有限非零梯度，冻结组件梯度0，峰值allocated约21.16GiB。两步不同训练片段的loss不能用来判断收敛。[轻量结果](v81/P0_R1_RESULTS.json)。

修复两项工程错误并保留失败日志：恢复Adam状态后显存不足，在观测特征完成后将冻结VAE/CLIP/RAFT移回CPU；推理读入将5维批次错误传给4维mask函数，改为先遮蔽再添加批次维度。保持数据、架构、mask、seed、两步预算不变，从已有checkpoint续跑。本次输入/训练契约5项测试通过，其中含真实推理读入回归。审核页本地 `outputs/v81-nuscenes-p0/index.html` 共18视频：8例输入预览，仅scene-0562具有原生生成及可见区合成；其余7例未生成。初始化不用DriveEditor/Seen-to-Scene编辑微调权重。

重要边界：采用公开train.py全帧传播，未复现论文m=4/SSIM参考链和DDIM inversion；条件RAFT只读可见RGB，完整RGB flow仅作teacher监督。3090采用bf16 UNet autocast/float32参数，记录真实梯度、冻结范围和峰值，不能用两步烟测宣称生成/时序/DELETE收益。

助手查看scene-0562原生f00发现色彩/结构明显失真，尚非可用外绘基线；该单帧不作时序评分。下一步是P1：补齐论文参考选择、inversion和正式训练/评测协议；等待YouTube-VOS合法下载，或继续明确标作nuScenes方法适配，不能把两步P0当作论文能力复现。当前不追加训练预算。

GitHub源码ZIP仍按100,000,000 bytes上限检查，历史docs/autoresearch通过export-ignore排除附件但完整Git历史不改写。新模型、数据、第三方代码和视频均在仓库外。本轮不关机；GPU任务结束，YouTube-VOS下载保留。完成本轮交付后删除用户授权的SVD/P0监控；人工verdict留空。`failure_ledger_refs=[V77-F02]`，`failure_ledger_delta=none`（两项工程错误已记入本run，不推导新的模型失败）。
