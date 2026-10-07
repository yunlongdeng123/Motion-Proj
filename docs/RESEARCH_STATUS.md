# 当前研究状态

更新：2026-10-08。分支 `research/worldsim-v7.7-target-protected-editing`，主机 wm-3090-1001。

唯一run `WS-V77-TARGET-PROTECTED-20260929/r49` 已完成64步、20新窗口及两阶段五例三帧独立助手review。GPU和控制器已空，没有自动追加训练或定时任务；已通知用户可切CPU。本次没有电源操作。

没有观察到相对r47/r48_64稳定的真实DELETE视觉升级：A034/A061薄膜与额外轮廓仍在，correct/wrong/no_RGB与同权重routing_off目视近同；A022仍逊于r46，两个合成例保住旧收益。工程检查、偏置生效、主干冻结和同预算输入顺序通过，不等于视觉收益；人工verdict与整段时序未判。

默认仍r46官方原权重+r21完整SAM，不推广r49。空间对应稀疏，f05洞内无N；更正来源概括：A061有2、M003有1个N源patch，非全程全无N。下一项先检验已知B查询更明确选择对应B来源、未知保持原路径；不追加步数，不与主干去噪控制同时修改。完整组件图与结果见[r49报告](v77/TARGET_PROTECTED_SPATIAL_BINDING_R49.md)，同一失败卡[V77-F02](research_failures/entries/V77-F02.md)。

本地审核`outputs/v77-priors-r49/index.html`含9卡、75视频并实际解码；原生/写回、参考/路由控制、三帧图板均保留。数据盘约余90.8GiB，原权重/输入/断点均在，无清理需求。
