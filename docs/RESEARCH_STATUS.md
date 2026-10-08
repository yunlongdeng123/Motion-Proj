# 当前研究状态

更新：2026-10-08。分支 `research/worldsim-v7.7-target-protected-editing`，主机 wm-3090-1001。

当前唯一run `WS-V77-TARGET-PROTECTED-20260929/r50`：先扩大真实DELETE单帧结构排查，再按失败造数据。按用户最新要求，subagent已独立检查87例f00/f05/f09真实原图与目标crop、可用B参考；0/1分36例退队列，补齐20个官方train场景、42个2分目标，每景2–3个不同实例。uncertain输入0例未准入，人工verdict仍空。请求配置gpt-6-sol/xhigh，未启用fast，工具未返回实际模型身份。评分是抽帧输入准入，不代替SAM身份或生成验收。

统一基线r46官方DriveEditor原权重+r21完整SAM，seed42/25steps/10帧/1024×576；无Adapter、微调、时间模块修改或累计多车删除。官方SDK时刻检查和全部准入原视频实际解码通过。缓存池非700景均匀总体样本。另5景不看RGB、不训练，旧val隔离集不动。后车证据分开登记：{'weak': 10, 'absent': 21, 'sufficient': 9, 'uncertain': 2}；A合格不保证B隐藏区域证据充分。

当前无GPU，SAM2/DriveEditor均0任务、训练0步；CPU完成后已停在等用户开卡。开卡后先完整SAM+实例检查，再每个合格目标固定一次DELETE。保存原生与写回，粗分薄膜、后车变形、车身向道路延伸；不拿失败生成当训练Y、不逐例调seed或洞规则、不自动微调。r49未见稳定收益的边界及旧产物保留，不恢复其训练。

本地 `outputs/v77-real-delete-r50/index.html` 只显示20景2分队列，退队列原图和理由单独归档。[组件图、输入与停止规则](v77/REAL_DELETE_STRUCTURE_R50.md)，同一失败卡[V77-F02](research_failures/entries/V77-F02.md)。数据盘约余91GiB，无清理或电源操作。GPU批次预留约1–2小时，开卡后按实际耗时更新。
