# 当前研究状态

2026-09-30，v77；`WS-V77-TARGET-PROTECTED-20260929/r4`。用户要求只用CPU。新来源准备及几何预案完成，[本轮报告](v77/TARGET_PROTECTED_CPU_R4.md)。32窗口/31scene，26新来源scene；固定10帧唯一曝光匹配29例，通过实际曝光几何29例。旧30帧门槛与最近邻对照保留，未复制帧或放宽时间限制。

真实LiDAR与旧r3物理规则得到4个CPU预案/4scene：{'single_actor': 4}。独立来源抽帧{'pass': 4}，待GPU精确SAM2共4个保护实例；队列disabled，未运行模型。包络交叠不能认证精确遮挡，本轮新合成0、训练0、人工null。

本地`outputs/v77-target-protected-r4/index.html`展示预案；r3真实合成与人工逐帧页不覆盖。下一步必须用户开GPU后获取新保护mask，再检查完整五项合同；候选不因规划通过自动准入。真实Y不变、DriveEditor架构不变，旧27技术候选与r3的24例保持独立。CPU工作已收口，GPU不可自启，未设自动化或关机。subagent默认gpt-6-sol/xhigh，禁fast。

run：`/root/autodl-tmp/runs/worldsim_v77/WS-V77-TARGET-PROTECTED-20260929/r4`。failure_ledger_refs: [V77-F02]；failure_ledger_delta: updated V77-F02。
