# 当前研究状态

更新：2026-09-26 08:14（Asia/Singapore）。工作分支：`research/worldsim-v7.6-ego-view-densification`。本文件只放当前快照；V7.5 完成证据见 [r9 收尾](autoresearch/worldsim_v75/downstream_bench/r9-closeout/README.md)，逐项任务见 [EXPERIMENTS](EXPERIMENTS.md)，失败边界见 [RESEARCH_FAILURES](RESEARCH_FAILURES.md)。

## 当前方向

V7.6 正在研究 ego-view densification：以 **VAD-GS 全新训练**连接原有反事实仿真代码，编辑 ego 相机轨迹，保持 actor 的原始世界时间和位姿。当前优先解决复现设置，使用相机 0–4 官方时间 test 检查基础重建，相机 5 作为额外未见相机外推。AutoDL GPU 已由用户开启；不恢复 HUGSIM checkpoint。组件及早期证据见 [V7.6 工程记录](v76/P0_ENGINEERING.md)，最新证据见[复现审计](v76/P0_REPRODUCTION_AUDIT.md)。

用户指定的两个优先诊断均已完成：COLMAP 138,944 点在 `<0.6 px` 后剩 33,930 点（24.42%），最终 15,756 点进入背景初始化；16k 官方时间 test 的 75 视图为 PSNR 22.7010 / SSIM 0.7486 / LPIPS 0.2313，基础重建已起来。此前相机 0 的61帧对照混合训练和留出，不能替代官方 test。

发现 [V76-F01](research_failures/entries/V76-F01.md)：旧COLMAP image ID被当作有序相机/帧编号，300/305图像错配，15,754/15,756个初始化点的visibility与真实track不一致，法线来源也受影响。按名字映射修复已通过回归，且已核验P0R1实际新初始化：15,475个COLMAP点、55,557条保存可见性边全部正确，错误行数0。旧P0约17,725迭代主动停止，4k/8k/16k权重及原始输入/日志/评估保留；不恢复旧权重。修复后的4k工程检查已通过，最终画质仍待30k；不能单独归因于映射或匹配器。

最新run **VADGS-P0R1-000** 已完成穷举、新三角化、4k工程门禁与8k官方时间test。8k同一75视图为 **PSNR22.7086 / SSIM0.7520 / LPIPS0.2348**，较4k分别+0.9694/+0.0220/−0.0379；五个相机均值三项均改善，固定五相机图已查看，近车/行人等仍有细节失真。4k的六偏移×32actor实际世界位置/旋转不变；+3.5m进入近树遮挡，−3.5m可见街道。两个一次性检查已退出，不重复启动。08:14主训练14828/30,000，控制器PID22094、训练PID27160仍运行；9k附近的体素扩展/NumPy空集合与比值警告已保存，最近loss有限且未见异常退出，尚不能判定警告无害，下个检查点继续核验。Camera5外推和30k结果仍待主队列。主状态入口 `/root/autodl-tmp/runs/v76_ego_view/VADGS-P0R1-000/pipeline_state.json`；指标、边界与architecture见 [P0R1执行记录](v76/P0R1_EXECUTION.md)。新初始化实际track已核验0错配；穷举后保留点数未大幅增长，数量不代表覆盖或画质。

scene-0230/0255各有366张深度与366张DSINE法线。背景SAM现各366/366；scene-0255的CPU续做98张于07:48完成，PID28434已退出。732张背景PNG逐张解码、尺寸/uint8/标签编码与文件名核验通过，逐文件SHA-256已保存；背景完成不解除动态身份门禁。[V76-F02](research_failures/entries/V76-F02.md) 的唯一fallback先通过6个固定可见正例/4个遮挡反例；后续固定0/40/60帧、36视图跨帧扩展完成，47个投影对象关联28个。确认可见公交车ID3被判成truck而漏标，固定同类关联的完整输入召回未过。按推理前停止规则，本轮停止这项扩展，不改阈值/类别或换检测器，保留全部sidecar和保护标记；同场景训练和HUGSIM同场景比较未完成。详见[先验与身份控制报告](v76/MATCHED_SCENE_PRIORS.md)。

## V7.5 截至收尾的已确认状态

- r9 版 **24 个 10 秒 paired case 获人工批准**，分为 6 个 ego 与 18 个对象 case；这取代旧文档的 `approval.status=pending` 现状。早期 OmniDreams 24 对生成仍是旧轮证据，不冒称 r9 新推理。
- HUGSIM 在 7 个场景完成 ground/scene 各 30k 与导出，24 个 case 均有 100 帧、10 Hz 的事实和反事实输出及自动指标。输入从官方 12 Hz 流程适配到 10 Hz；scene-0242 稀疏 COLMAP 与 scene-0998 相机更新存在质量风险。指标中的 outcome、轨迹和对象身份缺失值不补零。
- DriveEditor 的 18 个对象 case 有 10 帧原生 factual/counterfactual；其中 11 个另有完整 10 秒迭代 counterfactual。其 6 个 ego case 不属于原生编辑接口，另 7 个对象 case 没有完整迭代结果。原生 1 秒和迭代 10 秒不是同一评价分母；能力边界记于 [V75-F02](research_failures/entries/V75-F02.md)。
- ReSim 停止，GaussianDWM 仅有接口 smoke；RecEdit-Drive CPU 预检通过但专用 checkpoint 缺失，未启动推理。不存在可诚实计算的六模型 24-case 总分或统一排名。
- V7.5 保留包有 119 个 MP4、逐 case 结果、配置和两份人工评分工作簿；119/119 视频哈希已校验。2026-09-25 清理后旧模型权重、HUGSIM 七场景 checkpoint 及中间输入已退役，保留视频不是可直接恢复的训练状态。证据及复现边界见 [收尾记录](autoresearch/worldsim_v75/downstream_bench/r9-closeout/README.md)。

V7.5 之前的自然状态、对象移除、速度干预及接近任务控制仍由 [V7.5 文档](v75/PROBLEM.md)与 [实验索引](EXPERIMENTS.md)保留。旧版 V7.4 状态见[归档](archive/2026-09/v74-0920/STATUS_PRE_V75.md)，不会作为现行授权或当前资源状态。

## 下一门禁

继续P0R1至30k，审核主队列的官方test、Camera5外推、横移和actor世界变换；4k工程门禁已完成。同场景输入仍缺可靠跨帧动态mask，本轮有限fallback已按停止规则收口，不能把局部正例通过当作数据齐全。可在P0R1完成且结论/未完成范围清楚后按用户授权结束本轮扩展。帧20的+3.5m进入近物遮挡，ego压力测试需区分可通行方向/帧位。V7.5 HUGSIM输出只作历史数据对照，不用其checkpoint初始化V7.6。

## 当前电源授权

用户最新明确要求睡觉期间自主持续推进，工作完成到足以交付后自行收口并关闭AutoDL。仅对 `wm-3090-0811` 生效。收口前保存资产、取回报告和小型证据并push v76，确认无训练/评价/渲染/数据任务及后续启动控制器，将heartbeat暂停后调用 `/usr/bin/shutdown` 并验证。背景先验任务已完成，当前仍有全新训练与后续评价队列，**尚未满足关机条件，尚未关机**。这段授权不写入AGENTS或scaling law规则文档。
