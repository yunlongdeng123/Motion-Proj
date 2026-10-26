# 当前研究状态

更新：2026-10-27。分支`research/worldsim-v8.2-drivingeditor-v42`。Task：`WS-V82-PERSISTENT-WORLD-20261027`。

用户已授权按最小端到端基线→针对性测试/失败发现→固定编辑与生成后端改进背景修复的三阶段持续自主研究。当前主研究链路为证据驱动的背景联合修复；①编辑操作与③预训练后端先复用。协议和组件图见[当前执行计划](autoresearch/persistent_world_20261027/PLAN.md)，代码入口为`scripts/worldsim_v82/persistent_world_20261027/`。当前campaign=`/root/autodl-tmp/drivingeditor_repro/campaign/persistent_world_20261027`。负结果保留并导向下一轮，不设置额外科学止损点；单次输入/方法/步数与工程成本仍固定登记。

原生V41-B完整重建主副本冻结；编辑起点为用户选中的`edit_smoke_preserve/full199/checkpoint1000_from0.pth`。强透射`edit_visibility`是未选对照，不以提高道路贡献代替最终质量。此前错误绑定的更正见[历史机制证据](../scripts/worldsim_v82/drivingeditor_shadow_rebuild_20261017/r15/gn11_stage1_v42_20261024/controlled_smoke_20261026/C_MECHANISM.md)。

已完成49帧原生输出、状态重载和同后端视频闭环。主通路编辑前后最大RGB差0，raw102独立重载差0。两个上游状态均已运行预训练ProPainter；生成前3D帧与最终视频分别保存。三个原型优化视图中修复原生MAE0.00769，整洞生成后0.05012，暴露有效内容被覆盖。最简单证据路由及同来源直接RGB锚点强控制继续检验；不把输入证据域内的保护效果称泛化。

49帧参考诊断已生成；来源138–169曾进入199帧初始化，只是同片段跨视图一致性，不是独立GT。选择删除、侧移、复用插入、相机变轨/再显露的六项49帧原生基线已保存。原B没有逐行对象身份，当前七时刻框投票仍有867871/1030499行未决；保留未决行，并检查实际贡献，单对象编辑尚未解决。

持久SQLite状态保存26088世界候选单元的逐来源帧、修复版本、请求及独立的生成假设。7848为观测支持候选、13778为有支撑但未满足颜色资格、4462未知；候选与修复不冒称已核实三维表面。生成结果不自动升级为观测或可信几何。

当前训练阶段：`smoke_restore_train_10min`；权威运行状态是campaign/training_status.json，PID/start_ticks见对应dispatch。队列先100更新实际保存/评价，再独立进程完整状态恢复并真实训练至少600秒，之后fresh 10000更新可配置多视图修复基线。训练视图94/110/126；固定199初始化、几何、继承损失与学习率。原B更新0；smoke/必要重放单列，不能用旧149秒短跑代替本轮smoke。健康训练不重启、不修改配置。

GPU NVML每秒采样与CUDA allocator峰分开保存。DiffuEraser历史目录大权重未完成，当前正在新目录按固定revision补齐；ProPainter完整可用。新目录`/root/autodl-tmp/models/persistent_world_diffueraser_20261027`。后端不同不作为②收益，完整权重加载和实际推理通过后才称接入。

接续：收齐smoke/10K与49帧评价；检查简单直接参考控制能解释多少收益；依据失败决定几何、污染或持续记忆算法。严格道路撤源必须在初始化/编码前移除全部关联证据，不能使用已见答案的原生B冒充未观测输入。跨场景适配与独立测试尚未完成，240例仍是规模目标。代码/协议与结果取回、推送需逐轮核实；当前队列和下载仍在推进，禁止提前关机。无卡模式不因空闲关机。

旧V4/V41正式队列不恢复；V41-R/B 30K已完成并封存，旧7003工程更新继续保留，不混入当前新账。本轮`quality_accepted=false`、`human_verdict=null`；`failure_ledger_refs=[V82-F01,V82-F06]`，新增失败卡尚未收口，不自动判质量接受。
