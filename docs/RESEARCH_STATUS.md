# 当前研究状态

更新：2026-10-09。分支 `research/worldsim-v7.7-target-protected-editing`，CPU 主机 `wm-vgpu-1008`。

`WS-V77-TARGET-PROTECTED-20260929/r52` 已关闭：矩形道路 proxy 与矩形硬拼接伪标签均退出训练，仅保留工程诊断及失败证据。训练入口检查退役标记并拒绝续训。已完成三次64步与官方/r47对照，best1单例未通过，不续128、不扩case；历史分数、候选、关键checkpoint与全部结果保留。[r52报告](v77/SINGLE_CASE_FINETUNE_R52.md)。

新 run `r53` 只准备 SAM3 真轮廓入口：用户指定 ModelScope 的 sam3.pt 3,450,062,241字节已下载，CPU权重解析及官方模块导入通过；edit_mask、r47 hole、新二值alpha同步自测通过。发现旧wrapper将SAM轮廓改为外接矩形，旧接口留作复现，新入口绕开该转换。GPU推理0窗、训练0步、SAM3身份与覆盖尚未验证；真实轮廓本身不提供去车后的正确Y。[组件图与准备记录](v77/REAL_INSTANCE_MASK_R53.md)。

数据盘按用户授权实际释放203.687GiB（218.7GB）；SAM3安装下载后约可用208.4GiB。保留当前nuScenes、关键模型和全部runs；旧AV2 sensors/闲置权重/4个环境的删除清单及恢复入口见[清理记录](autoresearch/worldsim_v77/storage_cleanup_20261009/README.md)。

CPU准备已完成，当前CUDA不可用，无训练/推理控制器。下一步按用户最新请求先讨论DELETE掩码与去噪机制，不启动新架构或训练；开GPU后才做R001十帧真实实例分割核验。本轮无关机或自动化动作。人工verdict仍留空；同一失败卡[V77-F02](research_failures/entries/V77-F02.md)已更新。
