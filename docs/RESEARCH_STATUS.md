# 当前研究状态

更新：2026-09-26 02:26（Asia/Singapore）。工作分支：`research/worldsim-v7.6-ego-view-densification`。本文件只放当前快照；V7.5 完成证据见 [r9 收尾](autoresearch/worldsim_v75/downstream_bench/r9-closeout/README.md)，逐项任务见 [EXPERIMENTS](EXPERIMENTS.md)，失败边界见 [RESEARCH_FAILURES](RESEARCH_FAILURES.md)。

## 当前方向

V7.6 正在研究 ego-view densification：以 **VAD-GS 全新训练**连接原有反事实仿真代码，编辑 ego 相机轨迹，保持 actor 的原始世界时间和位姿。当前优先解决复现设置，使用相机 0–4 官方时间 test 检查基础重建，相机 5 作为额外未见相机外推。AutoDL GPU 已由用户开启；不恢复 HUGSIM checkpoint。组件及早期证据见 [V7.6 工程记录](v76/P0_ENGINEERING.md)，最新证据见[复现审计](v76/P0_REPRODUCTION_AUDIT.md)。

用户指定的两个优先诊断均已完成：COLMAP 138,944 点在 `<0.6 px` 后剩 33,930 点（24.42%），最终 15,756 点进入背景初始化；16k 官方时间 test 的 75 视图为 PSNR 22.7010 / SSIM 0.7486 / LPIPS 0.2313，基础重建已起来。此前相机 0 的61帧对照混合训练和留出，不能替代官方 test。

发现 [V76-F01](research_failures/entries/V76-F01.md)：COLMAP image ID 被错误当作有序相机/帧编号，300/305 图像错配，15,754/15,756 个 COLMAP 初始化点的 visibility 与真实 track 不一致，法线来源也受影响。已修复为按名字连接实际视图表，3 项回归和305图像核验通过；尚未重新初始化或重新训练。旧 P0 于约17,725迭代主动停止，GPU 上已无该训练进程；4k/8k/16k权重、原始输入、日志与评估完整保留。它不是训练崩溃，不恢复旧 P0 checkpoint，也不再以该run的30k收敛作为方法判决。修复不证明画质已改善，稀疏匹配的独立影响仍未测量。

最新自主执行run为 **VADGS-P0R1-000**，01:52启动官方 exhaustive matching（COLMAP3.7 CPU，12线程），02:26正常推进到匹配block[4/7,1/7]，尚无新checkpoint。旧SIFT和输入位姿复用，匹配表清空于新副本，新几何/初始化/模型均独立。有限控制器PID22094，匹配子进程PID22095；状态入口 `/root/autodl-tmp/runs/v76_ego_view/VADGS-P0R1-000/pipeline_state.json`，按它与实时进程核实，勿重复启动。队列会自动进入三角化、全新30k训练和基础评估。控制条件与架构见 [P0R1执行记录](v76/P0R1_EXECUTION.md)。

scene-0230/0255已各有366张深度与366张DSINE法线，固定官方六视图的法线轴/符号核验通过。SAM固定12视图测试发现 [V76-F02](research_failures/entries/V76-F02.md)：框提示将前景车/路牌误绑定给被遮挡行人；数值ID合法不等于可见身份正确。各6张动态mask已完整移到证据目录，训练入口有阻断标记，尚未启动同场景训练。后台只补背景SAM区域（02:26约27/366与6/366），进入P0R1 GPU阶段时自动退出让资源。数据合同、反例及唯一预注册的可见二维检测关联fallback见[先验报告](v76/MATCHED_SCENE_PRIORS.md)。此问题来自本适配器，不影响P0R1官方样例输入。

## V7.5 截至收尾的已确认状态

- r9 版 **24 个 10 秒 paired case 获人工批准**，分为 6 个 ego 与 18 个对象 case；这取代旧文档的 `approval.status=pending` 现状。早期 OmniDreams 24 对生成仍是旧轮证据，不冒称 r9 新推理。
- HUGSIM 在 7 个场景完成 ground/scene 各 30k 与导出，24 个 case 均有 100 帧、10 Hz 的事实和反事实输出及自动指标。输入从官方 12 Hz 流程适配到 10 Hz；scene-0242 稀疏 COLMAP 与 scene-0998 相机更新存在质量风险。指标中的 outcome、轨迹和对象身份缺失值不补零。
- DriveEditor 的 18 个对象 case 有 10 帧原生 factual/counterfactual；其中 11 个另有完整 10 秒迭代 counterfactual。其 6 个 ego case 不属于原生编辑接口，另 7 个对象 case 没有完整迭代结果。原生 1 秒和迭代 10 秒不是同一评价分母；能力边界记于 [V75-F02](research_failures/entries/V75-F02.md)。
- ReSim 停止，GaussianDWM 仅有接口 smoke；RecEdit-Drive CPU 预检通过但专用 checkpoint 缺失，未启动推理。不存在可诚实计算的六模型 24-case 总分或统一排名。
- V7.5 保留包有 119 个 MP4、逐 case 结果、配置和两份人工评分工作簿；119/119 视频哈希已校验。2026-09-25 清理后旧模型权重、HUGSIM 七场景 checkpoint 及中间输入已退役，保留视频不是可直接恢复的训练状态。证据及复现边界见 [收尾记录](autoresearch/worldsim_v75/downstream_bench/r9-closeout/README.md)。

V7.5 之前的自然状态、对象移除、速度干预及接近任务控制仍由 [V7.5 文档](v75/PROBLEM.md)与 [实验索引](EXPERIMENTS.md)保留。旧版 V7.4 状态见[归档](archive/2026-09/v74-0920/STATUS_PRE_V75.md)，不会作为现行授权或当前资源状态。

## 下一门禁

继续P0R1：三角化后核对过滤与正确track映射，4k优先完成工程渲染和几何门禁，30k检查官方test与Camera5外推、横移和actor时间戳，再按证据推进同场景；同场景必须先过V76-F02可见身份门禁，不把空mask或裁切遮挡物当修复。帧20的+3.5m进入近物遮挡，需选可通行方向/帧位。V7.5 HUGSIM输出只作历史数据对照，不用其checkpoint初始化V7.6。

## 当前电源授权

用户最新明确要求睡觉期间自主持续推进，工作完成到足以交付后自行收口并关闭AutoDL。仅对 `wm-3090-0811` 生效。收口前保存资产、取回报告和小型证据并push v76，确认无训练/评价/渲染/数据任务及后续启动控制器，将heartbeat暂停后调用 `/usr/bin/shutdown` 并验证。当前仍有全量匹配、后续训练队列与先验准备，**尚未满足关机条件，尚未关机**。这段授权不写入AGENTS或scaling law规则文档。
