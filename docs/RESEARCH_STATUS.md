# 当前研究状态

更新：2026-10-08。分支 `research/worldsim-v7.7-target-protected-editing`，主机 wm-3090-1001。

当前唯一run `WS-V77-TARGET-PROTECTED-20260929/r50`：按用户最新要求先扩大真实DELETE结构排查，再根据失败造数据和验证内部空间层。20个官方nuScenes train开发场景、49个独立单车目标已完成CPU准备与原视频审核页；抽f05后6例树木/护栏遮挡、逆光或暗处输入暂排除，43例/18景待GPU。另留5景不看RGB、不训练。复用缓存，非700景均匀总体样本。34例有后车投影代理，不代表后车外观证据已确认充分。

统一基线 r46 官方 DriveEditor 原权重+r21完整SAM，seed42/25steps/10帧/1024×576；无Adapter、微调、时间模块修改或累计多车删除。官方SDK关键帧/插值规则检查与49段原视频实际解码通过；主agent只抽f05核对原目标，SAM身份要开卡后单独查。人工verdict始终空。

当前无GPU，SAM2/DriveEditor均0任务、训练0步；CPU完成后已停在等用户开卡。开卡后先完整SAM+实例检查，再每个合格目标固定一次DELETE。保存原生与写回，粗分薄膜、后车变形、车身向道路延伸；不拿失败生成当训练Y、不逐例调seed或洞规则、不自动微调。r49未见稳定收益的边界及旧产物保留，不恢复其训练。

本地审核 `outputs/v77-real-delete-r50/index.html`；[组件图、输入与停止规则](v77/REAL_DELETE_STRUCTURE_R50.md)，同一失败卡[V77-F02](research_failures/entries/V77-F02.md)。数据盘约余91GiB，无清理或电源操作。GPU批次预计约1–2小时，开卡后按实际首批耗时更新。
