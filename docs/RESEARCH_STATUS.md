# 当前研究状态

更新：2026-10-13（新加坡）。分支 `research/worldsim-v8.2-drivingeditor-debug`。当前task/run为 `WS-V82-DRIVINGEDITOR-HYBRID30K-20261012/r1`。用户要求按“论文未明确改变的机制优先继承2024代码”重排实验，完成准备后等用户把 `wm-vgpu-1011` 切回GPU再运行。后续subagent默认 `gpt-6.1-sol / max`。本轮CPU准备完成，真实GPU训练0步，GPU首步/完整模型恢复及质量仍待验证。

四臂C00/G10/N01/GN11分别关闭双模块/仅corrected SGCM/仅NPS/两者，固定scene0017_000_048的49 FRONT开发视图、seed0、scale4 W480×H320，从零30k，每10k完整保存，四臂总120k。共同恢复raw动态alpha完整MSE100、静态RGB1、24活跃RGB系数1/5/5、寿命/速度正则、两项平滑/总alpha熵及LiDAR深度衰减；Eq7 sky与Eq16逆深度为显式共同改写，记录旧/新目标梯度差异。NPS保留sum/L2公式与.01，只按臂flag开启。[组件图、继承规则、矩阵与停止条件](v82/DRIVINGEDITOR_HYBRID30K_PLAN.md)。

SGCM保留Eq8、Hd与地面拟合，加入实际训练视图的SHM/时间alpha/中心投影支撑代理及一次性clone谱系，generic/SGCM append/prune/clone/split与18字段capture/Adam对齐；不冒称代理为遮挡后的真实贡献。实际contrib是每像素序号，不用作逐点贡献。Hd约360.36m²，三帧投影明显宽于body mask，保持定义做消融，不据此推断彩噪根因。共同修复canonical二值mask（避免719黑色body像素遗漏）、仅增密阶段reset、独立配对采样与RNG安全评价。静态时间/速度图正则不保证raw SHM不动，因此每节点另保存固定raw10位姿与全部49 timestamp静态变化控制。

远端新源码、工具和四配置位于 `/root/autodl-tmp/drivingeditor_repro/campaign/hybrid30k_20261012/`。正式DrivingEditor Python3.9/Torch2.2.2环境CPU损失6组、SGCM18项、集成7组检查通过，默认run_matrix入口只输出等待GPU，不导入renderer或开始训练。检查点必须包括模型、Adam、RNG、独立采样器、scale/EMA/SGCM谱系；GPU串行独占，异常中止队列，不自动调loss/分辨率或清资产。`eval=false`结果只标全观测重建；当前mask仍body-only缺shadow、M-detector仍未成为真实训练预处理。scene8仅后续候选、不排队。

原S＋V 2k CPU方案被新矩阵取代并保留，未执行、不入队。已有B1/B2/V/S的49帧30k→32k原结果继续保留：V/S改善静态背景却损害整体/车体，未认证修复；完整paper、50k、多场景与编辑仍未通过。[旧交接](v82/DRIVINGEDITOR_DEBUG_HANDOFF.md)。新方案有30k horizon/从零状态，不和旧50k continuation混比较。

`failure_ledger_refs=[V82-F01,V76-F03]`、`failure_ledger_delta=none`、`human_verdict=null`；不改写历史failure或代填质量评分。v8.1 DGGT/Seen-to-Scene等其他研究及所有旧自动化继续暂停，不因本次准备恢复；当前只有明确固定的四臂，无关资产和旧检查点均保留，电源操作仍由用户负责。
