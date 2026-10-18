# 当前研究状态

更新：2026-10-18。当前 task=WS-V82-R15-IDENTITY-COMPARISON-20261018，failure_ledger_refs=[V82-F06]，failure_ledger_delta=none。用户要求补齐旧版a6/a37真实owner、R15 a5/a26/a18独立与删除，并按0/1/unknown排查。无subagent，无R16/R17，无R15续训。[三值组件、交互中心与全量谱系证据](autoresearch/worldsim_v82/drivingeditor_shadow_rebuild_20261017/v3_backend_restore_r15/identity_20261018/tristate/index.html)。

历史旧版只有a5/a26/a18三个独立owner，a6/a37由静态/PVG表达。历史旧3及R15冻结4K已完成49×3／49×5按真实身份独立RGB、alpha、可见alpha和单车删除，R15同时正确移除太阳caster、输出关阴影删除。补齐旧5从同一GN11 30K、原始LiDAR重新初始化，原3身份种子位置/RGB逐位一致，原旧30K协议训练进行中；不能将未完成训练当作最终基线。原旧3、GN11和R15检查点均保留。[本轮范围与状态](autoresearch/worldsim_v82/drivingeditor_shadow_rebuild_20261017/v3_backend_restore_r15/identity_20261018/status.json)。

实测证据status原码0=unknown、1=Actor、2=static，概率通道[static,Actor,unknown]，均为buffer而非Adam参数。a6/a37内部初始及4K均全1；2067112个来源三元组匹配的谱系后代状态/概率/硬owner无变化。unknown从123484到463520是增密计数，不是学习失败的转移率。a6当前框外中心25905/29834、a37为13681/14573来自初始框内父点，须区分种子错误与训练几何演化；框外不等于道路错误真值。旧版二值控制适用。原生SAM正区落在R15 unknown中，a6=39.04%、a37=0.25%，落在已知负域均为0%；SAM不作为边界GT，R15输入未改。

GPU当前已由用户重新开启并用于旧5完整30K。随后顺序执行旧5渲染及初始/4K三值Gaussian真实支撑投影；后者队列必须完成，不能只看旧训练队列结束就关机。交付对照、视频和报告并push后，核对所有worker/controller退出，再按本次明确授权关机。此前16:04关机记录是历史状态，不能冒充本次终态。

R15最小阴影修复仍为显式内存适配器；生产入口、loss、默认配置、CUDA ABI与原checkpoint未改。[此前收口实验](autoresearch/worldsim_v82/drivingeditor_shadow_rebuild_20261017/v3_backend_restore_r15/closure_20261018/report.html)、[历史同一a6核对](autoresearch/worldsim_v82/drivingeditor_shadow_rebuild_20261017/v3_backend_restore_r15/old30k_a6_20261018/report.html)。a6碎片与a37静态残影未解决；human_verdict=null，problem_solved=null。49帧为同一重建窗口，无配对编辑GT，不冒称独立测试；保留用户其他dirty文件。
