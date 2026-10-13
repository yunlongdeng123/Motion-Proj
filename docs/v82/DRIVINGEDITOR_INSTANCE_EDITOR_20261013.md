# v82 DrivingEditor：GMC 双实例固定轨迹基线

更新：2026-10-14T02:26:09.149475+08:00。仅GMC scene0008 raw23–71，其他两场景没有新增实验。真实step300 checkpoint已续到总30000（新增29700），worker37747退出，训练完成。10k/20k/30k全部通过CPU结构、finite、Adam/RNG结构与policy绑定检查；post30k_r1另实际GPU恢复三节点完整模型/Adam与sampler/Python/NumPy/TorchCPU/allCUDA RNG，均逐字段逐值PASS、优化更新0次。完整测试118通过、1项CUDA默认跳过；独立真实CUDA遮挡与梯度合同另行通过。

![组件图](../autoresearch/worldsim_v82/drivingeditor_instances_20261013/r1/gpu_smoke_edits_resume_20261013/DrivingEditor_instance_editor_components.png)

车辆使用静态局部Gaussian与真实整体刚体轨迹，协方差和局部SH方向同步，没有车辆PVG重复运动。两条原始track明确绑定DEVA77/82，137174/160901个真实LiDAR点，共298075，98/98真实可见mask；有效轨迹各49帧，FRONT可见7/39。600k准备seed仅排除这两框后579855；实际GN11继承背景经Gaussian中心框内删除后613405点，两种计数不能混用。停车仍在静态背景。

逐实例监督来自完整场景的同一次深度排序alpha*T。实际安装CUDA测试得到可见union0.392、孤立union0.980，alpha梯度有限并通过身份贡献求和。背景/环境、GT轨迹和拓扑全程固定，pose refinement/joint background未开启，SGCM只继承原GN11背景结果，没有新static→rigid克隆；NPS原公式及RGB/静态RGB/深度衰减保留，不宣称绝对静态或新SGCM迁移完成。seed、损失、LR horizon30000、冻结/位姿开关及拓扑均由保存的实际policy锁定。

| Checkpoint | 文件大小（B） | CPU结构/finite/绑定 | 实际GPU模型/Adam/RNG恢复 |
| --- | ---: | --- | --- |
| 10000 | 935060588 | PASS | PASS，更新0次 |
| 20000 | 935060588 | PASS | PASS，更新0次 |
| 30000 | 935060524 | PASS | PASS，更新0次 |

CPU检查使用map_location=cpu、单线程/nice10，没有模型恢复、RNG恢复或优化步骤。GPU证明覆盖首次更新前保存状态的精确恢复，不认证下一步CUDA数值确定性；旧checkpoint和训练status中的restoration_verified=False原样保留，实际runtime proof另存。真实300步checkpoint也已在GPU完成相同恢复核验，续训入口在首次step301更新前重复核验。旧300步没有raw配置快照，策略根据其真实effective_config与现有reference重建，历史配置内容认证边界保留。

30k续训总wall4969.2793秒，另有300步短测59.442秒。完整续训monitor已关闭：PyTorch allocated/reserved峰2569314816/4880072704B（2.393/4.545GiB），NV进程/设备采样峰5236588544/5247074304B（4.877/4.887GiB），4971采样、0errors。采样峰为下界，PyTorch峰不含其他CUDA分配；目标3090容量/速度和完整后续评价资源仍未认证。历史短测的allocated/reserved峰2.105/3.508GiB、进程/设备采样峰3.840/3.850GiB及60采样0errors继续保留，各编辑memory_summary原样留在远端。

post30k_r1先完成三节点恢复，再在场景SSIM评价遇到CPU GT/CUDA composite设备不匹配；只以GT.to(composite.device)修复，没有改渲染、GT像素、损失或训练参数，失败目录保留，登记[V82-F04](../research_failures/entries/V82-F04.md)。post30k_r2在真实StreetCrafter CPU准备优先请求出现后主动yield，保留28个observed帧（raw23..50）。固定相机0帧、30k六项编辑0项、评估末尾完整模型/Adam只读核验未到达，后续只在资源空闲时使用新目录完成49 observed+49 fixed-camera和六项编辑，不再训练、不自动抢资源重试。02:18 SGT实际snapshot中StreetCrafter GPU进程71279使用1352MiB，当前不启动新导出。

独立CPU检查直接读取保留的28份原始FP32 NPZ，核验有限性、两ID与480×320原生尺寸；420张PNG逐像素等于round(clip(FP32,0,1)*255)，真实prepared body masks与旧GMC GT PNG逐像素一致。本地独立partial manifest位于outputs/drivingeditor_GMC_instances_30k/preview_partial，该预览已接入现有统一DrivingEditor_final_report.html的#instances-preview；实际static QA通过420 PNG路径、旧HTML逐字可恢复和导航源码存在，尚未进行GUI QA。保留逐帧CPU指标，partial SSIM为null；不计算或宣称完整49帧聚合质量，不用28帧均值代替49帧结论。camera只保真实source JSON标定来源，未独立序列化GPU矩阵；单独30k GPU恢复证明不能替代评估末尾只读证明。human_quality_verdict仍null，30k照片编辑质量尚未认证；已实际审查全部28帧的四页同尺度分层：皮卡较清楚，Jeep前段半透明，后5帧改善；static和composite均保留灰白道路补片。本批非空body分母为7/28，完整49帧应为7/39。原8-bit PNG诊断显示Jeep前段孤立车辆raw union接近1，而joint实例alpha仅约0.35–0.70，static可见贡献补足；应优先检查静态遮挡或残留几何，具体Gaussian、深度和原因尚未定位，没有据此修改loss或训练。

历史300步编辑证据完整保留：两车各删除、局部前移0.5m和局部yaw30度覆盖49观察帧，共294组。FP32独立CPU复核有限性、逐帧GT/合成/static/vehicles配对、独立static前后逐位相同和删除目标贡献精确0。非编辑实例Gaussian世界位置、旋转、尺度、外观与opacity在实际GPU执行中逐位不变，其可见像素仍可因合法遮挡变化。旧GMC49帧GT PNG复核最大byte差0；84张接触页保留全图和真实mask40pxpadding共同ROI，空mask帧用全图。这些是300步合同，不能当作30k六项编辑已导出。

历史300步图像保留车体偏黑、半透明、纹理和轮廓不足，以及删除后道路浅灰模糊patch与暗残影；当时工程视觉审核允许原预算续训，没有认定照片编辑质量通过。背景Gaussian中心移除不处理跨框尾部、车影或遮挡后未观测表面，没有编辑后GT、重照明或OmniRe公平对照，不套旧四臂1dB判据。历史step23300快照、10k/20k CPU receipt、累计3888采样0errors与10k raw47单帧预览均保留；当时车体较成形而静态道路灰白补片仍在，单帧预览不能代替最终49帧审查。

主输出runs/gmc_two_actor_fixed30k_r1，启动凭据queue_gmc30k_r1/launch.json，历史短测queue_gmc_r1/gpu_smoke，历史六项编辑edits_smoke_300_r1。StreetCrafter及其他工作优先，CPU最多4线程/nice10，审计单线程；每半小时心跳drivingeditor-gmc保持ACTIVE，原drivingeditor-30k PAUSED。此阶段不关机、不修改共享环境、不kill或暂停他人。最终证据接入既有统一HTML入口；旧六组180k、旧报告/离线包和所有失败资产保留，不追加银车/红车实验。
