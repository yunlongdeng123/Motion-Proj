# 当前研究状态

更新：2026-10-14T02:26:09.149475+08:00。当前任务WS-V82-DRIVINGEDITOR-INSTANCES-20261013/r1，仅GMC两动态实例。固定背景/轨迹基线已从真实step300续到总30000（新增29700），worker37747退出。10k/20k/30k checkpoint全部通过CPU结构、finite、Adam/RNG结构和policy绑定检查；post30k_r1另在GPU实际逐值恢复完整模型/Adam与8组状态字段，三节点均PASS、优化更新0次。完整测试118通过、1项CUDA默认跳过，独立实际CUDA遮挡/梯度合同已通过。

[协议与组件图](v82/DRIVINGEDITOR_INSTANCE_EDITOR_20261013.md) · [状态](v82/DRIVINGEDITOR_INSTANCE_EDITOR_20261013.json) · [30k与预览证据](autoresearch/worldsim_v82/drivingeditor_instances_20261013/r1/final30k_partial_preview_20261014/)。真实98mask/298075实例点，GN11继承静态删两框中心后613405；停车保持静态。背景/环境、GT轨迹和拓扑全程固定，车辆无PVG重复运动，pose/joint background未开启；旧SGCM仅继承背景，没有新克隆，NPS原公式保留。

post30k_r1评价的CPU GT/CUDA SSIM设备不匹配已仅以GT.to(composite.device)修复，失败资产保留并登记[V82-F04](research_failures/entries/V82-F04.md)。post30k_r2在真实StreetCrafter CPU准备优先请求出现后主动yield，保留28个observed帧（raw23..50）的FP32与420张原生PNG；独立CPU核验finite、PNG对应round(clip(FP32)*255)、真实prepared masks和旧GMC GT逐像素一致。独立partial manifest已在outputs/drivingeditor_GMC_instances_30k/preview_partial，已接入现有统一HTML的#instances-preview，420 PNG/旧HTML可恢复/导航源码static QA通过，GUI QA未做；固定相机0帧、30k六项编辑0项、评价结束模型/Adam只读核验未到达，不能用28帧均值替代完整49帧质量。

恢复证明只覆盖首次更新前保存的模型/Adam/RNG精确状态，不认证下一步CUDA数值确定性；旧checkpoint的认证False原样保留。人工质量评价null，编辑后GT不存在。28帧分层像素审查已完成：皮卡较清楚、Jeep前段透明、static道路补片保留；8-bit PNG支持检查静态遮挡或残留几何，但具体原因未认证，没有修改loss或增加训练。历史300步偏黑透明车体、道路patch和残影、旧300配置没有raw快照的边界及10k单帧预览全部保留。30k续训wall4969.2793s，另有300步短测59.442s；PyTorch allocated/reserved峰2569314816/4880072704B，NV进程/设备采样峰5236588544/5247074304B，4971采样0errors。采样峰为下界，目标3090容量/速度未认证。

每半小时心跳drivingeditor-gmc ACTIVE；原drivingeditor-30k PAUSED。StreetCrafter及其他工作优先，不共跑、不kill或暂停他人；CPU≤4/nice10，CPU审计单线程。仅SSH wm-vgpu-1011，新根/root/autodl-tmp/drivingeditor_repro/campaign/instance_editor_20261013。02:18 SGT实际snapshot有StreetCrafter GPU进程71279、1352MiB，当前不启动新导出；等待资源空闲后只在新目录导出完整49 observed+49 fixed-camera及30k六项编辑，无自动抢资源重试，不再训练、不追加银车/红车实验。

历史运行中证据：[step23300与10k/20k检查](autoresearch/worldsim_v82/drivingeditor_instances_20261013/r1/heartbeat_20261014_0055/)。旧六组180k、统一报告/离线包与全部失败资产保留，旧关机路线已结束，此阶段不关机、不清理其他资产。保留scripts/run_worldsim_v72_e3_render_bridge.py无关dirty，子agent默认gpt-6.1-sol/max。
