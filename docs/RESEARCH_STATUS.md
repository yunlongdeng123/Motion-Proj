# 当前研究状态

更新：2026-10-11（新加坡）。分支 `research/worldsim-v8.1-seen-to-scene`；SSH `wm-3090-1009`；checkout `/root/autodl-tmp/motion_proj_v81`。

DGGT Waymo官方基线与车辆高斯删除、右移、同场景复制插入已实际完成，随后官方Difix处理四支共16帧；task/run为`WS-V81-DGGT-WAYMO-INFERENCE-20261011/r1`，状态`complete_assistant_reviewed`。原始noop与官方float trace逐值差0，保存PNG因量化最多差1灰阶。独立gpt-6-sol/xhigh/no-fast助手全部四帧审核：操作已生效，但删除留明显黑灰洞，Difix未补完整道路；移动/复制有邻车重叠和右缘裁切，视觉质量hold。人工verdict=null，不能由单场景四帧宣称长时序或论文编辑质量通过。[流程、组件图与实测](v81/DGGT_WAYMO_INFERENCE_R1.md) · [轻量结果](v81/DGGT_WAYMO_INFERENCE_R1.json)。

深色`outputs/v81-dggt-waymo/index.html`包含四支原生、Difix、输入、选择轮廓和alpha；64PNG、16MP4/64帧均解码，80媒体链接无缺失、JS语法通过。未声称浏览器视觉QA。首次Difix依赖错误在隔离推理venv内用diffusers0.32.2修复，只续缺失Difix，不重跑基线或编辑，不改S2S环境/权重。原始失败栈保留，服务器保持开机。

Seen-to-Scene P1 task/run `WS-V81-SEEN-TO-SCENE-P1-20261009/r1` 已从7500完成新增2500更新，8000/9000/10000均实际保存，10000完整原件4,833,164,855B、scheduler10000、Adam514、RNG齐全，协议不变；均速5.957秒/步、loss和梯度有限、冻结梯度0。六固定valid窗各25帧推理完，两助手实际审核全部150帧：滑板与白鲸宽侧区有明确相对进步，窄侧区有小幅学习；海豚收益不明确且有新伪影。最终质量hold，训练收益continue；当前未启动12500。[周期记录与复盘](v81/P1_REVIEW_CYCLES_R1.md) · [证据](v81/P1_REVIEW_CYCLES_R1.json)。10000页为`outputs/v81-paper-p1/step10000_review.html`；7500传播六窗`no_clear_gain`仍为既有推理诊断，本轮未重复。

授权滚动清理已在10000完整校验、六窗、真实助手复盘及GPU空闲后完成：保留10000+7500完整原件，逐项清单移除旧正式文件；数据盘实际增加14.500GB、系统盘4.833GB。UTC17:30核对GPU无进程、数据盘余131.20GB、系统盘6.46GB，服务器保持开机。诊断源/P0/基座/原RGB/评分/视频和DGGT产物保留，软链不是外部备份。此前105.95GB清理见[恢复边界](v81/STORAGE_CLEANUP_20261010.md)。Git仅轻量代码和结果，源码ZIP小于100MB。

P0闭环完成；P1训练体系和局部学习已建立，完整四指标尚未评测、protocol_verified=false；P2/P3未开始。DGGT和v77 DELETE输入/任务不同，不用上述重建分数作横向质量排名；当前无确定新科学根因，failure_ledger_delta=none，既有V77-F02边界保留。每小时监控只核对当前任务实际状态，不重复完成推理或依据旧prompt启动12500/关机。周额度最新实读剩85%、余一张授权重置卡；再次真实剩<3%才使用。
