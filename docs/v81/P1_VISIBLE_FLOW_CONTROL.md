# P1：训练传播光流来源的单因素控制

task/run：`WS-V81-SEEN-TO-SCENE-P1-20261009/r1`。P1未通过，正式四指标未计算。failure_ledger_refs=[V77-F02]，failure_ledger_delta=none。本页是有界诊断，不替换正式论文训练协议。

重采样diffusion sigma/epsilon的64步仍未恢复训练片段的展翼动作。下一项只检查训练使用完整RGB的RAFT条件、QUERY使用可见RGB的RAFT条件这一差异；不能提前断言它是根因。

```mermaid
flowchart LR
    V[遮罩后的真实视频] --> R[冻结RAFT\n相同参考对]
    R --> F[可训练FCNet]
    F --> P[参考潜变量传播]
    P --> S[SVD时间层\n64次Adam更新]
    G[完整真实视频] --> GT[GT flow与GT latent\n仅损失监督]
    GT --> S
    S --> Q[仅可见RGB的同seed QUERY]
    Q --> A[原生25帧独立审核]
```

与已完成 `fixed_clip_resampled_diffusion_step500_64` 配对，两支都从正式500权重及Adam开始。同片段 `0fc958cde2/start2`、25帧256²、mask每侧.33、完整首帧CLIP、固定VAE后验与条件噪声、Adam lr1e-5/wd0、seed2026、每步LogNormal(.7,1.6) sigma与Gaussian epsilon、64更新；QUERY seed2036/25步/fps6不变。

唯一新因子：FCNet输入由缓存完整GT flow改为冻结RAFT对 `visible_rgb` 的估计。参考对不变、20 iter/chunk2；RAFT预计算置于CPU/CUDA RNG fork内。原 `cache.flow` 始终保存完整GT，供光流L1/warp监督；只额外保存 `flow_input`，不能把可见flow误当GT。teacher缓存不被修改，保持旧GT条件的固定单步测量；另报告使用visible-flow条件的teacher前后值，两个口径分开。

原训练mask、QUERY膨胀3×3 mask仍有差异，完整首帧CLIP/fps7训练与visible CLIP/fps6推理也未一起改动，因此本控制不是完全匹配推理的训练。visible-flow和resampled-noise终态各用独立格式，不能被正式P1恢复。QUERY-before必须与既有500原生全部25帧逐像素相同；逐步sigma也必须与已完成重采样分支一致。

CPU接线验证15项通过：输入与GT监督分离、teacher不污染、RNG前后相同、训练模块梯度、旧控制契约。运行前仍需核对GPU真实作业与空闲存储。质量hold只否定该具体预算下的结果，主任务继续定位下一差异，不自动关机、不等待人工审核。
