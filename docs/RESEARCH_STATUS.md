# 当前研究状态

2026-09-30，v77；`WS-V77-TARGET-PROTECTED-20260929/r5`。用户最新明确资源要求是CPU，本轮连续规划和精确准入入口完成。[本轮报告](v77/TARGET_PROTECTED_CPU_R5.md)。r4的4个空间预案保持原曝光、donor、位姿与门槛，扩成40帧/16视频，160视频帧实际解码；没有实际synthetic-X，训练准入0。

四列为真实Y+B定位框、A洞规划上界、masked-Y条件规划、米制位姿。全帧隐藏像素条件探针和ego底部保守禁入通过；真实保护车SAM2 mask仍缺，不能认证精确遮挡或时序视觉通过。复用r4 Sol来源抽帧，未重复AI看图；人工verdict全部空白。

`iteration5/admit_exact.py`已检查4例全部waiting_inputs。真实PNG来源字段、数值连续性与独立mask QA同时通过后才调用原精确几何/遮挡关卡。4项准入测试和本地静态文件/JS验证通过；浏览器交互未测。没有推理、训练、自动化或电源操作。下一步需用户确认解除CPU限制，再运行r4 SAM2保护mask、独立QA、精确关卡、实际合成与人工全检。远端当前已读到RTX3090可用，未自动加载模型。

本地 `outputs/v77-target-protected-r5/index.html`；run `/root/autodl-tmp/runs/worldsim_v77/WS-V77-TARGET-PROTECTED-20260929/r5`。r4负对照/旧资产、r3实际合成和全帧人工页保持独立。failure_ledger_refs: [V77-F02]；failure_ledger_delta: none。
