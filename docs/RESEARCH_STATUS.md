# 当前研究状态

更新：2026-10-14T01:11:51.332165+08:00。当前任务WS-V82-DRIVINGEDITOR-INSTANCES-20261013/r1，仅GMC两动态实例。300步GPU兼容、六项×49观察帧编辑合同、真实完整模型/Adam/RNG恢复通过；从同checkpoint续至总30k，PID37747，当前证据step23300；10k/20k已保存且CPU结构/finite/Adam/RNG/policy通过，30k待完成；10k/20k GPU实际恢复未认证。

[协议与组件图](v82/DRIVINGEDITOR_INSTANCE_EDITOR_20261013.md) · [状态](v82/DRIVINGEDITOR_INSTANCE_EDITOR_20261013.json) · [新GPU证据](autoresearch/worldsim_v82/drivingeditor_instances_20261013/r1/gpu_smoke_edits_resume_20261013/)。旧CPU输入/合同证据保留。114CPU检查通过，实际CUDA联合遮挡/梯度另行通过。真实98mask/298075实例点，GN11继承静态删两框中心后613405；停车保持静态。固定背景/环境、固定GT轨迹、固定拓扑，车辆无PVG重复运动，不自动开放pose或joint background；旧SGCM仅继承背景，没有新克隆。NPS原公式保留。

真实300步车体偏黑透明、道路patch与残影均保留，工程支持继续原预算，不代表收敛质量。人工评价null；编辑后GT不存在。恢复证明仅首次更新前模型/Adam/RNG精确值，不认证下一步确定性或全部未来checkpoint；旧300配置没有raw快照的边界保留，今后policy锁定。GPU短测设备采样峰3.850GiB是下界，新30k和目标3090仍待验证。

每半小时心跳drivingeditor-gmc ACTIVE；原drivingeditor-30k PAUSED，StreetCrafter及其他GPU任务优先，不共跑；CPU≤4/nice10。新竞争任务出现保存priority_yield退出，无自动抢资源续训。仅SSH wm-vgpu-1011；新根/root/autodl-tmp/drivingeditor_repro/campaign/instance_editor_20261013。默认CPUvalidate，禁止重复启动已退出的300队列或当前worker。

银车/红车不新增实验。旧六组180k与统一报告全部保留，旧关机路线已结束，此新阶段绝不关机、清理、暂停或终止他人任务。保留scripts/run_worldsim_v72_e3_render_bridge.py无关dirty，子agent默认gpt-6.1-sol/max。

本半小时节点证据：[10k/20k检查](autoresearch/worldsim_v82/drivingeditor_instances_20261013/r1/heartbeat_20261014_0055/)。3888 NV采样0错误，设备采样峰4.887GiB为运行中下界，活跃training阶段PyTorch峰待monitor关闭。10k单帧预览车体更成形、道路灰白补片保留，不能代替最终49帧审查。最终评价/统一HTML接入源码正在CPU准备，无新GPU或训练阶段。
