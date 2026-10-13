# v82 DrivingEditor：GMC 双实例固定轨迹基线

更新：2026-10-14T00:10:45.044812+08:00。仅 GMC scene0008 raw23–71；其他两场景没有新增实验。真实300步GPU短测完成，两车各删除、局部前移0.5m和局部yaw30度覆盖49观察帧，共294对。工程合同和实际恢复通过，已从step300续到总30000（新增29700），worker PID37747，当前快照step1100；10k/20k/30k完整保存。StreetCrafter及其他GPU任务优先，CPU最多4线程/nice10，竞争出现保存priority_yield后退出，无自动抢占续跑。

![组件图](../autoresearch/worldsim_v82/drivingeditor_instances_20261013/r1/gpu_smoke_edits_resume_20261013/DrivingEditor_instance_editor_components.png)

车辆使用静态局部Gaussian与真实整体刚体轨迹，协方差和局部SH方向同步；没有车辆PVG重复运动。两条原始track明确绑定DEVA77/82，137174/160901个真实LiDAR点，共298075，98/98真实可见mask；有效轨迹各49帧，FRONT可见7/39。600k准备seed仅排除这两框后579855；实际GN11继承背景经Gaussian中心框内删除后613405点，两种计数不能混用。停车仍在静态背景。

逐实例监督来自完整场景的同一次深度排序alpha*T，而不是孤立实例覆盖度。实际安装CUDA测试得到可见union0.392、孤立union0.980，alpha梯度有限并通过身份贡献求和。首阶段静态/环境冻结、真实轨迹固定、拓扑固定，SGCM只继承原GN11背景结果，没有新static→rigid克隆；NPS原公式及RGB/静态RGB/深度衰减保留，不宣称绝对静态或新SGCM迁移完成。

294组FP32独立CPU复核有限性、逐帧GT/合成/static/vehicles配对、独立static前后逐位相同和删除目标贡献精确0。非编辑实例Gaussian世界位置、旋转、尺度、外观与opacity在实际GPU执行中逐位不变；它的可见像素仍可因合法遮挡变化。旧GMC49帧GT PNG复核最大byte差0。84张接触页保留全图和真实mask40pxpadding共同ROI，空mask帧用全图。

300步真实图像仍有车体偏黑、半透明、纹理和轮廓不足，删除后道路浅灰模糊patch及暗残影。工程视觉审核保留这些坏例并允许按原预算训练，不能称照片编辑质量通过。背景Gaussian中心移除不处理跨框尾部、车影或遮挡后未观测表面；没有编辑后GT、重照明或OmniRe公平对照。human_quality_verdict仍null，不把300步与旧30k硬比，不套旧四臂1dB判据。

114项CPU检查通过，CPU suite的CUDA项默认跳过；另一个独立真实CUDA前后向测试通过。完整场景模型/Adam与sampler/Python/NumPy/TorchCPU/allCUDA RNG已从真实300步checkpoint在GPU逐字段逐值恢复核验；续训入口也在第一步更新前重复核验。此证明的范围是保存状态恢复，不证明下一步CUDA数值确定性或未来所有节点可恢复；旧checkpoint认证字段保留False。旧300步没有raw配置快照，策略根据其真实effective_config与现有reference重建，历史配置内容认证边界明确保留；后续checkpoint保存实际解析policy，锁定seed、损失、LR horizon、冻结/位姿开关及拓扑。

300步总wall59.442秒，末300步训练吞吐约6.95步/秒；PyTorch allocated/reserved峰2.105/3.508GiB，进程/设备采样峰3.840/3.850GiB、60采样0errors。它只描述短测，不能认证新30k全程或3090容量/速度；采样峰是下界，仍须完整训练、编辑/评价及目标机验证。各编辑操作memory_summary原样保留远端。此新阶段不关机、不改其他任务、不改共享环境。

主输出：`runs/gmc_two_actor_fixed30k_r1`；启动凭据：`queue_gmc30k_r1/launch.json`；先前短测：`queue_gmc_r1/gpu_smoke`；六组编辑：`edits_smoke_300_r1`。每小时检查保持开启，无变化安静。最终真实收敛结果接入已有DrivingEditor_final_report.html，不新增HTML入口。旧六组180k、旧报告/离线包和所有失败资产保留。
