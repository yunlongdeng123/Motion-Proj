# WorldSim V7.1 M17 — Monotone LiDAR Ray Survival Plan

状态：`FROZEN`（2026-09-04）  
任务：`WS-V71-M17-MONOTONE-RAY-SURVIVAL-01`  
假设：`WS-V71-H-M17-MONOTONE-TERMINATION`

## 1. Blocker and change of representation

M16用完整GT FREE rays把observable提高到约93.5%并保住hit，但hazard early恶化11.98%。同一query scalar可沿ray任意
多次变号，因此“最早zero crossing”不受训练样点顺序约束。继续增加FREE points或删early crossings属于采样/后处理补救。

M17把首交改写为survival process。local evidence network只预测`σ(x)>=0`；有序ray samples通过
`T_i=exp(-sum_{j<=i} σ_j Δ_j)`得到survival，`F_i=1-T_i`是termination CDF。其非负性和单调性是表示的数学边界，
不是训练后检查。NeuRAD以opacity-times-transmittance渲染LiDAR expected depth；Neural LiDAR Fields进一步建模two-way
transmittance和first-return peak。本实验只迁移与几何首交直接相关的最小机制，不引入intensity/ray-drop。

## 2. GT and objective

- 每条target LiDAR ray在Actor AABB entry/exit间均匀采32点；
- GT CDF在target depth前为0、从target所在bin起为1；
- balanced NLL分别平均pre-hit survival与at/after-hit termination，避免ray长度改变类权重；
- termination weights `w_i=(1-exp(-σ_iΔ_i))T_{i-1}`的归一化期望深度接受GT L1；
- 以上构成单一joint survival objective，不再用PCGrad把hit与FREE拆开；
- network输入仍是query-local M8 children/features/relative coordinates；M8参数冻结。

## 3. Deployment and frozen protocol

- deployment在同一AABB ray上累积CDF，取首次`F>=0.5`的中位termination depth；`0.5`是分布中位定义，不调优；
- seed=`71119`，epochs=`6`，32 train samples，64 evaluation samples，maximum training rays=`128`；
- M8 point五门保持；hazard early相对M8降低`>=5%`；all hit delta `>=-1pp`；
- 禁止CDF threshold、density scale、sample count、loss、capacity、seed、epoch扫描；失败登记`V71-F22`；
- 不读Source Final或未完成AV2 aggregate。

## 4. Separation boundary

M17只验证几何首交与ray physics。dynamic/static、actor motion、image information和安全risk仍保持独立；survival field通过也不能
替代跨域AV2确认。

## Sources

- [NeuRAD official implementation, CVPR 2024](https://github.com/georghess/neurad-studio)
- [Neural LiDAR Fields project, ICCV 2023](https://research.nvidia.com/labs/toronto-ai/nfl/)
