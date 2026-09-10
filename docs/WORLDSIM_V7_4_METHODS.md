# V74 方法实现记录

当前阶段是首次实现，尚无 DEV 结果或晋级裁决。三候选独立，两个数据集各自 FIT 定标；相机关闭。

```mermaid
flowchart LR
  B[BUILD 点与射线] --> P[本数据集 FIT 定标]
  P --> A[WEX 共享域 / 责任交换]
  P --> R[RIF 场 / 区间根 / 细分]
  P --> C[DCS LP 需求 / 新片 / 整数选择]
  A --> S[固定三角表面 ≤4096面]
  R --> S
  C --> S
  S --> E[硬首交点 + 最近点]
  Q[隔离 QUERY 真值] --> E
```

## 实际接入与前序

[OSQP 官方](https://osqp.org/docs/solver/) 的线性约束 QP用于 WEX 子问题，显式正见证松弛只诊断冲突。近似冲突关联不称最小不可行核/精确证书。联合责任搜索与固定/逐射线控制共用子问题。
[SciPy HiGHS MILP](https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.milp.html) 是通用混合约束控制；先最大化解释的见证数，再固定责任求同一个域 QP，MILP 线性主目标与域二次代价分开记录。
[HiGHS LP 对偶接口](https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.linprog.html) 用于后续 DCS 定价。
[Minimal Neural Atlas](https://arxiv.org/abs/2207.14782) 已有可学习域；A0 是 PU 思想与同射线输入的任务适配软域控制，未复现其完整官方训练模型，不能标成官方 MNA。
[DMTet](https://research.nvidia.com/labs/toronto-ai/DMTet/) 的同场等值面与 [NKSR](https://github.com/nv-tlabs/NKSR) 的局部几何/稀疏重建是 B 的前序与外部控制候选；当前没有声称复现。

## FIT 定标

`scripts/calibrate_worldsim_v74_fit.py` 只读 FIT BUILD，跨帧局部 PCA 平面的测距残差用于统一数值容差；两数据集独立，不打开 DEV QUERY。nuScenes 326 个≥3点对象、129812有效比较；AV2 719对象、672669比较。epsilon 为90分位并限制 .02–.20m：.174519874/.20m；评价带仍±.2m。超过容差的观测不排除评价。
半宽取 BUILD 近邻间距中位数的四倍并限制 .15–.8m，得到 .156881495/.15m；这不是验证调参。配置 `configs/worldsim_v74/tournament.json` 在 DEV 质量出现前落盘。首次实现中的实际数值/成本问题仅允许 FIT 修正后说明。

WEX 初始64片、4×4顶点；域裁剪共享边交点。空正确候选只允许已有承载片的两轮有限平移，不生成新片。输出仍可存在未解释观测，不能以求解器退出码冒称物理满足。机制首个run只测试80个解析共享域析取子问题，尚不满足计划要求的80组三维几何机制证据。
