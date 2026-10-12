# 当前研究状态

更新：2026-10-12（新加坡）。分支 `research/worldsim-v8.2-drivingeditor-debug`。用户在新聊天明确要求继续 AutoDL DrivingEditor 复现，并将后续subagent默认改为 `gpt-6.1-sol / max`。当前 task/run `WS-V82-DRIVINGEDITOR-SV-20261012/r1`：独立S＋V联合候选已在远端完成CPU源码准备，7组实际候选检查通过，GPU恢复harness尚待接入，新增GPU训练0步。远端 `wm-vgpu-1011` SSH可达，但目前CPU模式，0.5核/2GiB、无GPU设备；用户自行切回GPU。本轮不启动旧场景队列、自动化或电源操作。

下一步从同一saved30k恢复SV到32k，保留50k学习率horizon、seed0和已有B1的2,000步相机/time_shift/LR轨迹。S系数1，沿用历史V外域opacity项系数.05，加在真实未预mask动态alpha上，与原前景项组成双域监督；使用独立输出，保留原S/V冻结源码。CPU源码、数学和配置检查后，GPU恢复时另接独立SV harness并做首步真实backward核验；旧单变量guard/receipt不能直接复用。当前body-only mask缺shadow；双域监督不等于完整论文预处理。[组件图、固定协议与边界](v82/DRIVINGEDITOR_SV_RESUME.md)。

已有银色车Waymo FRONT raw0–48共49帧的B1/B2/V/S均从同一saved30k完成2,000步到32k。B1/B2全图PSNR为36.43076/36.43047dB，静态道路仍差、动态层仍含背景；V/S背景static PSNR为27.33124/31.76342dB，composite却降到35.90406/35.44799dB，并有车体代价。SV联合尚无真实GPU结果，不能将CPU检查判为修复完成。完整paper复现、从零/50k、多场景与编辑效果仍未通过。[既有交接](v82/DRIVINGEDITOR_DEBUG_HANDOFF.md)保留完整原证据及停止时快照；本次用户明确续接撤销DrivingEditor的研究暂停，不自动恢复其他任务。

相关工程failure为[V82-F01](research_failures/entries/V82-F01.md)：替换opacity目标时丢失掩码外直接监督、长训前分层验收不足。[后续双域默认](v82/DRIVINGEDITOR_OPACITY_DEFAULTS.md)仍有效。当前 `failure_ledger_refs=[V82-F01,V76-F03]`、`failure_ledger_delta=none`、`human_verdict=null`。reset/保存顺序、depth投影顺序、SGCM不可见clone、静态时间、初始化污染下界及M-detector/shadow预处理缺口仍按既有证据解释，不合并为唯一彩噪根因，不因新候选改写历史卡。

原数据、30k与32k检查点、FP32数组、媒体和冻结源仍在原本地/远端路径。四件30k恢复入口已stat存在，本次不在2GiB内存中加载约2.78GB模型文件；实际完整恢复待GPU阶段核验。当前可用数据盘约75GB，不需为本轮重新下载数据。轻量Git保存patch、工具、协议和已有证据索引，不冒称大型资产完整备份。

v8.1 DGGT与Seen-to-Scene等其他研究的暂停和历史结果继续保留，不因本次DrivingEditor续接恢复。当前只有SV一个候选，不开loss权重网格；是否进入从零训练和更多场景由固定49帧的分层、车体与整体质量共同决定。
