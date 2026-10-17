# 当前研究状态

更新：2026-10-17（Asia/Singapore）。最新用户要求**暂停R14，先审计并恢复旧后端重建能力**，覆盖此前“跑完再关机”的自动执行计划。控制器3605、flock4792、worker4793已退出，b-c-15已PAUSED；无自动恢复、无新训练、GPU未关机。禁止subagent/新chat，旧B/C不恢复。[后端继承审计](autoresearch/worldsim_v82/drivingeditor_shadow_rebuild_20261017/v3_r14/backend_audit/README.md)，[实验索引](EXPERIMENTS.md)。

R14最后日志5350，按50步日志间隔实际已执行5350..5399；完整可恢复checkpoint为5000，模型/Adam/RNG/输入保留并另存，SHA256=01affea31778f9548be812ffbdf1fed72779b4d4c64bf122ee7b6542f08db2a5。未持久化350..399更新不计作可恢复；历史19500..19549成本另记。原3k失败/修复/审计证据保留。

CPU读取实际旧30k与R14检查点确认：旧915840静态点、静态和Actor三阶SH、6×1024×1024环境图；R14初始静态80617、5k静态83456，全场景45万上限、32×64环境图和简化RGB/视角残差。几何阶段道路直接RGB被排除，5k道路24746点中仅1809有正干净参考（7.31%）。这是表示、覆盖、监督和优化共同改变，不能归因IDSplat无效，亦不能承诺多训恢复。点比例不是像素面积覆盖或零梯度比例。

修正方向：以旧训练/渲染后端为主体继承覆盖、SH、EnvLight、原优化调度与完整重建监督，再接SAM3/IDSplat、pixelvalid、独立运动、持续归属/实际owner和几何位姿阴影；静态道路无时间自由度保留。继承设计不等于自动复用已烘焙阴影的GN11权重，从零与温启动条件须明确。旧18k为冻结父场景的局部修复，不作为新主方法。当前只完成审计/状态修正，未实施新模型或启动训练。

⑥动态资产带道路仍未解决；旧3k图里白车/背景/删除缺口不接受。49帧65流和完整22400计划均未完成且已挂起。delete范围仅7车5/6/12/18/25/26/37；旧0/1/2→5/26/18，其余4无旧独立资产。原始数据、旧GN11/v2/18k、所有失败与用户dirty工作保留。完整mask绘图已交付，但不是语义验收。

task_id=WS-V82-DRIVINGEDITOR-SHADOW-REBUILD-20261017；human_verdict=null，problem_solved=null，failure_ledger_refs=[V82-F01]，failure_ledger_delta=none。
