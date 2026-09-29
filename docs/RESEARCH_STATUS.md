# 当前研究状态

2026-09-29，v77；`WS-V77-TARGET-PROTECTED-20260929/r1`。用户已授权GPU造数，本轮SAM2和试产完成，进入用户逐帧全检；未训练、未运行新DriveEditor前向、架构未改。[当前报告](v77/TARGET_PROTECTED_GPU.md)。

实际49个合成case / 11个receiver scene，独立抽帧{'reject': 7, 'pass': 28, 'uncertain': 14}，仅28例可进入人工全检；通过类型{'background': 9, 'single_actor': 19}。尚未收到人工全检回传、训练准入0。约50例合格目标未达、密集多车0；不得把候选数/重复短窗当独立合格scene数。

P/D是30帧控制；W是依据官方训练长度的固定10帧短窗，规则变化在质量协议v2保留。49个全部实际产物、失败和未确定例可在本地`outputs/v77-target-protected-synthetic/index.html`查阅，正式人工列表只含pass。此前来源审核页独立保留，来源通过不等于合成通过。

来源/实例污染、光照与自车前景顺序错误已更新同一V77-F02。当前池密集提案有界细化及未知包络诊断均0，不继续相同扫描；下一步用可投放空间与保护车相对几何先选新来源，再做RGB/分割，以补齐类型和场景多样性。人工逐帧确认前不正式微调；只保留X可观测reference、真实Y仅作监督。

run：`/root/autodl-tmp/runs/worldsim_v77/WS-V77-TARGET-PROTECTED-20260929/r1`。GPU任务/CPU渲染均结束；没有新建自动化或电源操作，当前阶段不继承旧审计关机授权。subagent默认gpt-6-sol / xhigh，禁止fast。failure_ledger_refs: [V77-F02]；failure_ledger_delta: updated V77-F02。
