# P1：固定片段的 diffusion 噪声分布控制

task/run：`WS-V81-SEEN-TO-SCENE-P1-20261009/r1`。本页是 CPU 准备，新增真实训练/推理均为 **0**。P1 仍未通过，正式四指标未计算。failure_ledger_refs=[V77-F02]，failure_ledger_delta=none。

此前两组固定输入64步，训练与teacher测量均复用同一 sigma、epsilon、GT flow、VAE后验和CLIP。其单步去噪改善约22%，Gaussian QUERY仍不可靠跟随主体动作。不能用一个噪声点的拟合结果判断整个去噪轨迹的学习能力；下一轮只改变训练的 diffusion sigma/epsilon。

```mermaid
flowchart LR
    X[同一25帧训练片段] --> C[固定 GT flow / VAE / CLIP]
    S[每步采样 sigma 和 epsilon] --> T[FCNet / 参考传播 / SVD\n64次Adam更新]
    C --> T
    T --> E[固定sigma单步测量\n与旧诊断可比]
    T --> Q[仅可见RGB + 同Gaussian\n25帧原生QUERY]
    Q --> A[独立全帧图像审核]
```

两支从相同正式500权重和Adam状态开始；新支复用旧 `fixed_input_capacity_step500_64/fixed_inputs.pt`。片段 `0fc958cde2/start2`、25帧256²、每侧mask .33、seed2026，QUERY seed2036/25采样步。学习率1e-5、Adam/wd0、损失与可训练参数不变，仍不使用作者编辑权重。

唯一成组改变的是每次训练重新采样 `sigma~LogNormal(.7,1.6)`（调用现有官方 util）和 `epsilon~N(0,I)`。sigma实际值逐步写入原有updates.jsonl；不截断、不择优重采样。其余GT flow、VAE posterior、条件噪声、完整首帧CLIP、time IDs/fps7都复用缓存，仍无CFG dropout。teacher继续使用原缓存，不被训练噪声污染；QUERY仍仅可见RGB/fps6、同Gaussian初值。它匹配论文的diffusion噪声分布，但**不是完整论文训练协议**；64次随机抽样也不保证覆盖每一个采样档位。

GPU启动前先核对真实进程/队列、GPU、磁盘与源断点；不覆盖任何旧输出，不自动扩步数。仅执行以下一项：

```bash
cd /root/autodl-tmp/motion_proj_v81
OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 /root/autodl-tmp/envs/motionproj/bin/python -m motion_proj.worldsim_v81.p1_fixed_capacity_probe \
  --checkpoint /root/autodl-tmp/runs/worldsim_v81/WS-V81-SEEN-TO-SCENE-P1-20261009/r1/paper_bidirectional_m4/train/p1-checkpoint-000500.pt \
  --fixed-inputs /root/autodl-tmp/runs/worldsim_v81/WS-V81-SEEN-TO-SCENE-P1-20261009/r1/fixed_input_capacity_step500_64/fixed_inputs.pt \
  --output-dir /root/autodl-tmp/runs/worldsim_v81/WS-V81-SEEN-TO-SCENE-P1-20261009/r1/fixed_clip_resampled_diffusion_step500_64 \
  --training-diffusion-noise resampled --teacher-clip-source full-gt --updates 64 --seed 2026 --query-steps 25
```

结束后核对实际Adam增量与冻结梯度、QUERY-before与旧500原生全25帧逐像素一致，再由6sol xhigh（no fast）对两支QUERY-after的全部25帧独立看图。主体轮廓、动作跟随、接缝是判断依据；GT写回中心不计入生成能力。teacher误差只是诊断，不替代视觉通过。

若原生结构/运动改善，只能说本片段此预算的单sigma/epsilon限制有所缓解；sigma与epsilon共同改变，不能单独归因。若无改善，也不能断言充分训练后的模型容量不足。两种结果都不自动放行100K或进入P2。终态使用独立格式保存，不能进入正式训练恢复。之前同规模GPU计算约3–4分钟，含检查/编码/审核预计10–20分钟；本次尚未实测。

CPU验证：固定与重采样模式均执行真实小模块反向更新；确认只有sigma/epsilon改变、teacher缓存保持、CLIP/time IDs未变，且FCNet/传播仍重算。当前8项测试通过。远端已关机，新代码尚未部署或GPU验证；旧结果全部保留。
