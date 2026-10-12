# 当前研究状态

更新：2026-10-13 00:53（新加坡）。分支 `research/worldsim-v8.2-drivingeditor-debug`。当前task/run为 `WS-V82-DRIVINGEDITOR-HYBRID30K-20261012/r1`。用户已把 `wm-vgpu-1011` 切回GPU并授权启动，按“论文未明确改变的机制优先继承2024代码”执行。串行controller2704、C00 worker2705运行正常；最新归档观测2400/30000步，G10/N01/GN11排队。真实GPU前向/反向/优化与1/1000梯度诊断已通过；10k检查点、完整GPU模型恢复与质量尚未验收。后续subagent默认 `gpt-6.1-sol / max`。

四臂C00/G10/N01/GN11分别关闭双模块/仅corrected SGCM/仅NPS/两者，固定scene0017_000_048的49 FRONT开发视图、seed0、scale4 W480×H320，从零30k，每10k完整保存，四臂总120k。共同恢复raw动态alpha完整MSE100、静态RGB1、24活跃RGB系数1/5/5、寿命/速度正则、两项平滑/总alpha熵及LiDAR深度衰减；Eq7 sky与Eq16逆深度为显式共同改写，记录旧/新目标梯度差异。NPS保留sum/L2公式与.01，只按臂flag开启。[组件图、继承规则、矩阵与停止条件](v82/DRIVINGEDITOR_HYBRID30K_PLAN.md)。

SGCM保留Eq8、Hd与地面拟合，加入实际训练视图的SHM/时间alpha/中心投影支撑代理及一次性clone谱系，generic/SGCM append/prune/clone/split与18字段capture/Adam对齐；不冒称代理为遮挡后的真实贡献。实际contrib是每像素序号，不用作逐点贡献。Hd约360.36m²，三帧投影明显宽于body mask，保持定义做消融，不据此推断彩噪根因。共同修复canonical二值mask（避免719黑色body像素遗漏）、仅增密阶段reset、独立配对采样与RNG安全评价。静态时间/速度图正则不保证raw SHM不动，因此每节点另保存固定raw10位姿与全部49 timestamp静态变化控制。

远端新源码、工具和四配置位于 `/root/autodl-tmp/drivingeditor_repro/campaign/hybrid30k_20261012/`。正式DrivingEditor Python3.9/Torch2.2.2环境CPU损失6组、SGCM18项、集成7组检查通过，默认run_matrix入口只输出等待GPU，不导入renderer或开始训练。检查点必须包括模型、Adam、RNG、独立采样器、scale/EMA/SGCM谱系；GPU串行独占，异常中止队列，不自动调loss/分辨率或清资产。`eval=false`结果只标全观测重建；当前mask仍body-only缺shadow、M-detector仍未成为真实训练预处理。scene8仅后续候选、不排队。

GPU识别为RTX4080 SUPER、32760MiB可见显存。资源监控从初始化开始累计PyTorch allocated/reserved峰，并每秒采样nvidia-smi进程/设备；独立监控文件保存各阶段及每臂总体，不将父子峰值相加。2400步进程采样峰3.23GiB、设备采样峰3.26GiB、reserved峰3.11GiB，最近100步约10.29步/秒，仅为早期观测；尚未覆盖增密终态、LPIPS与完整评价，不能认证3090 24GiB迁移。四臂完整30k后按2GiB初筛余量与目标3090实测评估。[启动与资源快照](autoresearch/worldsim_v82/drivingeditor_hybrid30k_20261012/r1/launch_gpu_snapshot_20261013.json)。

首次C00在相机CUDA初始化处停滞，0训练步，已终止owned worker并将所有资产归档至`attempts/initialization_1`；没有删除或改变科学配置。监控改为明确posix_spawn并禁止fork fallback，远端新增CPU7项验证通过，第二次初始化约15秒且已持续训练。具体原生锁源未确认，工程证据见[V82-F02](research_failures/entries/V82-F02.md)。

原S＋V 2k CPU方案被新矩阵取代并保留，未执行、不入队。已有B1/B2/V/S的49帧30k→32k原结果继续保留：V/S改善静态背景却损害整体/车体，未认证修复；完整paper、50k、多场景与编辑仍未通过。[旧交接](v82/DRIVINGEDITOR_DEBUG_HANDOFF.md)。新方案有30k horizon/从零状态，不和旧50k continuation混比较。

用户要求安排后隔半小时到一小时检查，现已创建当前聊天的每小时心跳`drivingeditor-30k`：只检查本四臂，控制器自行运行；无可行动变化保持安静，新完整10k节点、显存接近容量、异常或完成才简报。全部完成后核验与汇总质量/静态性/梯度/资源报告，再暂停此心跳。旧`drivingeditor-50k`及v8.1 DGGT/Seen-to-Scene等其他自动化保持暂停。

`failure_ledger_refs=[V82-F01,V82-F02,V76-F03]`、`failure_ledger_delta=V82-F02`、`human_verdict=null`；不改写历史failure或代填质量评分。当前只有明确固定的四臂，无关资产和旧检查点均保留，电源操作仍由用户负责。
