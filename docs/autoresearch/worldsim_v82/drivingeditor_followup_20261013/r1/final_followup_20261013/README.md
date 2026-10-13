# DrivingEditor 两追加场景最终证据

`WS-V82-DRIVINGEDITOR-GN11-GATE-OTHER-SCENES-20261013/r1`，2026-10-13。原四臂120k加GMC/红车各GN11 30k，总180k；固定seed0/49 FRONT/10k三节点。此目录保存红车终态、两场景最后只读快照、数值/阶段资源与实际像素/浏览器检查记录。没有新增arms、seeds、loss或训练预算。

`red_report_metadata`含真实CPU导出与原小记录；`red_terminal`保存两次collector源/状态和原快照。attempt1在真实30k/所有节点检查后因工程目录假设停止，partial原样保留；attempt2按事件真实metrics路径完成全部复算。`red_nodes`是当时10k/20k过程审计，不覆盖30k终态；`pre_power_audit.json`是发信号前的只读快照，不能作为关机请求或平台电源证明。

所有18个节点仅实际验证四文件stat/header、事件、49FP32/PNG/header和固定49时间；不读取大checkpoint模型payload，也未恢复Adam/RNG，restoration_verified=false。完整模型、FP32原资产与失败尝试保留AutoDL。HTML离线包与本目录均不是大检查点备份。

实际必需像素审查为全部49观察帧×5通道及固定49时间×2通道，343图；完整序列来源明确为代理实际看图报告，root直接核对raw79及固定raw113–116。额外98个alpha_s/body图不从文件存在推定已审查。工程判读保留静态道路残影、时变、绿线/浅粉反事实伪影；人工verdict=null。无同scene C00，不认证模块因果收益/回退；没有heldout或反事实GT。

六组完整训练最大观测6.902GiB；真实DEVA设备采样10.279GiB为本轮含预处理最大观测，SAM已记录加载与三帧推理峰。3090仅22GiB预算容量初筛，采样是下界，目标机未经实测。无新增科学failure；沿用V82-F03工程兼容故障记录。最终关机信号/自动化暂停记录收口后单独附入，不从SSH断开推出平台已关机。
