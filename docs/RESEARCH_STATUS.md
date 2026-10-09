# 当前研究状态

更新：2026-10-09。分支 `research/worldsim-v8.1-seen-to-scene` 继承 v7.7 的 `000ad1f0`。主机 `wm-3090-1009`，独立 `/root/autodl-tmp/motion_proj_v81` checkout；原v7.7代码与资产保留。

task/run：`WS-V81-SEEN-TO-SCENE-20261009/r1`。路线为原始 SVD + RAFT + ProPainter 光流补全，自有数据/训练/推理闭环；P0→P1论文外扩复现→P2驾驶真实显露DELETE基线→固定P2后P3。当前用nuScenes推进方法适配，不能混报YouTube-VOS论文复现。[路线](v81/SEEN_TO_SCENE_P0.md) · [r1协议与components图](v81/NUSCENES_P0_R1.md)。

CPU准备完成：6官方train+2官方val场景，共200张原始1600×900 CAM_FRONT RGB，保留sample_data.next链与sweeps；每段25帧约2秒，实际50/100ms混合、平均约12Hz。只提取缺失tgz成员，未整包解压；固定25帧、256²、左右各84像素外扩洞。20项CPU测试通过，真实官方传播组件小尺寸forward/backward通过；均不替代真实SVD优化验收。输入HTML在本地 `outputs/v81-nuscenes-p0/index.html`，16个输入视频解码通过。[输入证据](v81/P0_R1_INPUTS.json)。

原始SVD XT 1.1访问权限已恢复，官方固定版本组件正在仓库外续传；网络助手和临时文件不提交。三份YouTube-VOS（train.tar/test.zip/valid.tar）各已挂一个续传worker，Google当前返回quota HTML，按退避重试，未获得完整压缩包。详情 `r1/downloads/`；下载HTML不能算数据。数据盘约109GiB起始可用，可容纳本轮组件与三包，不需清盘。

真实训练/推理代码已接入；唯一 `run_p0.py` 控制器等待SVD完整组件后，冻结6+2输入清单，执行优化第1步→退出并恢复至第2步→独立val25步生成→视频审核页。到本次快照，真实优化0步、生成0窗，P0尚未通过。原生与可见区合成分列；遇工程错误停止留日志，不扩大预算。初始化不用DriveEditor/Seen-to-Scene编辑微调权重。

重要边界：采用公开train.py全帧传播，未复现论文m=4/SSIM参考链和DDIM inversion；条件RAFT只读可见RGB，完整RGB flow仅作teacher监督。3090采用bf16 UNet autocast/float32参数，记录真实梯度、冻结范围和峰值，不能用两步烟测宣称生成/时序/DELETE收益。

GitHub源码ZIP仍按100,000,000 bytes上限检查，历史docs/autoresearch通过export-ignore排除附件但完整Git历史不改写。新模型、数据、第三方代码和视频均在仓库外。本轮不关机、不新建自动化；人工verdict留空。`failure_ledger_refs=[V77-F02]`，`failure_ledger_delta=none`。
