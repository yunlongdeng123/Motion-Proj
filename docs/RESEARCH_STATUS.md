# 当前研究状态

更新：2026-10-13（新加坡）。分支 `research/worldsim-v8.2-drivingeditor-debug`。当前task/run为 `WS-V82-DRIVINGEDITOR-HYBRID30K-20261012/r1`。C00已完整完成30k及10k/20k/30k保存评价，controller2704已自动串行转入G10（02:30读取时8500步）；N01/GN11继续排队。本轮用户要求查看C00动静态效果，已从保存的FP32数组用CPU导出深色离线HTML，未占用训练GPU。完整GPU模型/Adam/RNG恢复与四臂质量增量仍待核验；子agent默认 `gpt-6.1-sol / max`。

四臂C00/G10/N01/GN11分别关闭双模块/仅corrected SGCM/仅NPS/两者，固定scene0017_000_048的49 FRONT开发视图、seed0、scale4 W480×H320，从零30k，每10k完整保存，四臂总120k。共同恢复raw动态alpha完整MSE100、静态RGB1、24活跃RGB系数1/5/5、寿命/速度正则、两项平滑/总alpha熵及LiDAR深度衰减；Eq7 sky与Eq16逆深度为显式共同改写，记录旧/新目标梯度差异。NPS保留sum/L2公式与.01，只按臂flag开启。[组件图、继承规则、矩阵与停止条件](v82/DRIVINGEDITOR_HYBRID30K_PLAN.md)。

SGCM保留Eq8、Hd与地面拟合，加入实际训练视图的SHM/时间alpha/中心投影支撑代理及一次性clone谱系，generic/SGCM append/prune/clone/split与18字段capture/Adam对齐；不冒称代理为遮挡后的真实贡献。实际contrib是每像素序号，不用作逐点贡献。Hd约360.36m²，三帧投影明显宽于body mask，保持定义做消融，不据此推断彩噪根因。共同修复canonical二值mask（避免719黑色body像素遗漏）、仅增密阶段reset、独立配对采样与RNG安全评价。静态时间/速度图正则不保证raw SHM不动，因此每节点另保存固定raw10位姿与全部49 timestamp静态变化控制。

远端新源码、工具和四配置位于 `/root/autodl-tmp/drivingeditor_repro/campaign/hybrid30k_20261012/`。正式DrivingEditor Python3.9/Torch2.2.2环境CPU损失6组、SGCM18项、集成7组检查通过，默认run_matrix入口只输出等待GPU，不导入renderer或开始训练。检查点必须包括模型、Adam、RNG、独立采样器、scale/EMA/SGCM谱系；GPU串行独占，异常中止队列，不自动调loss/分辨率或清资产。`eval=false`结果只标全观测重建；当前mask仍body-only缺shadow、M-detector仍未成为真实训练预处理。scene8仅后续候选、不排队。

C00 10k/20k/30k节点合同核验通过：每节点四个模型/状态文件非零且大小匹配manifest，49个frame ID/timestamp一致，196张四通道PNG为W480×H320，49个raw NPZ的FP32 RGB/alpha/depth与boolean mask header尺寸正确。静态时间控制各49rows/49NPZ、固定pose0017010/anchor=-.28。核验仅查JSON/stat/header，未加载模型或sidecar；HTML另读取保存的数组导出PNG，但没有加载或恢复checkpoint。[10k/20k receipt](autoresearch/worldsim_v82/drivingeditor_hybrid30k_20261012/r1/C00_checkpoint_contract_10000_20000_20261013.json) · [C00终态](autoresearch/worldsim_v82/drivingeditor_hybrid30k_20261012/r1/C00_30000_summary.json)。

GPU识别为RTX4080 SUPER、32760MiB可见显存。C00全程（包含增密、三次完整评价/LPIPS/时间控制）allocated峰5.23GiB、reserved峰6.29GiB、进程采样峰6.53GiB、设备采样峰6.58GiB，采样无错误；耗时4975.12秒/82.92分钟，静态986061点、动态1028966点。采样峰仍是下界，不能相加父子峰值。C00符合24GiB减2GiB的容量初筛，其他三臂与目标3090实测仍待完成，不认证迁移后速度。G10在02:30观测时累计设备采样峰6.60GiB且无错误，未完成全程。[C00终态与资源](autoresearch/worldsim_v82/drivingeditor_hybrid30k_20261012/r1/C00_30000_summary.json)。

C00 30k全49帧等权：全图合成PSNR34.0624dB、车体合成32.7312dB、非天空车外静态34.0154dB、同域动态alpha均值0.0001710。RGB指标先clip[0,1]，alpha未预mask；不将目录名hybrid_raw误解为未clamp PSNR。原始分支查看中raw10车位下仍有黑色残影；固定pose改变timestamp的静态RGB MAE平均0.02271、最大0.03940，静态alpha MAE平均0.00321，不能宣称补景或绝对静态性已验收。此处没有编辑GT或独立heldout分数，等待G10/N01/GN11同控制比较。

用户查看页位于本聊天 `outputs/drivingeditor_C00_30k/index.html`，可播放/逐帧/同步车体ROI，包含全部49帧GT/合成/静态/动态、固定标尺alpha/body mask及固定相机49模型时刻；`outputs/DrivingEditor_C00_30k.zip`可携带整个离线素材。共441张PNG，均从保存FP32 clip[0,1]转8bit，无body预mask或逐图归一化。浏览器已核对四卡、末帧、ROI和固定时间页；页面保留实际残影，不作美化。

首次C00在相机CUDA初始化处停滞，0训练步，已终止owned worker并将所有资产归档至`attempts/initialization_1`；没有删除或改变科学配置。监控改为明确posix_spawn并禁止fork fallback，远端新增CPU7项验证通过，第二次初始化约15秒且已持续训练。具体原生锁源未确认，工程证据见[V82-F02](research_failures/entries/V82-F02.md)。

原S＋V 2k CPU方案被新矩阵取代并保留，未执行、不入队。已有B1/B2/V/S的49帧30k→32k原结果继续保留：V/S改善静态背景却损害整体/车体，未认证修复；完整paper、50k、多场景与编辑仍未通过。[旧交接](v82/DRIVINGEDITOR_DEBUG_HANDOFF.md)。新方案有30k horizon/从零状态，不和旧50k continuation混比较。

用户要求安排后隔半小时到一小时检查，现已创建当前聊天的每小时心跳`drivingeditor-30k`：只检查本四臂，控制器自行运行；无可行动变化保持安静，新完整10k节点、显存接近容量、异常或完成才简报。全部完成后核验与汇总质量/静态性/梯度/资源报告，再暂停此心跳。旧`drivingeditor-50k`及v8.1 DGGT/Seen-to-Scene等其他自动化保持暂停。

`failure_ledger_refs=[V82-F01,V82-F02,V76-F03]`、`failure_ledger_delta=V82-F02`、`human_verdict=null`；不改写历史failure或代填质量评分。当前只有明确固定的四臂，无关资产和旧检查点均保留，电源操作仍由用户负责。
