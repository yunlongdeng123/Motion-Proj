# 当前研究状态

更新：2026-09-26 00:53（Asia/Singapore）。工作分支：`research/worldsim-v7.6-ego-view-densification`。本文件只放当前快照；V7.5 完成证据见 [r9 收尾](autoresearch/worldsim_v75/downstream_bench/r9-closeout/README.md)，逐项任务见 [EXPERIMENTS](EXPERIMENTS.md)，失败边界见 [RESEARCH_FAILURES](RESEARCH_FAILURES.md)。

## 当前方向

V7.6 正在研究 ego-view densification：以 **VAD-GS 全新训练**连接原有反事实仿真代码，编辑 ego 相机轨迹，保持 actor 的原始世界时间和位姿，再用留出相机及几何门禁评估。AutoDL GPU 已由用户开启，P0 训练正在运行；不恢复 HUGSIM checkpoint。组件、输入及 4k/8k 中途证据见 [V7.6 工程记录](v76/P0_ENGINEERING.md)。

P0 使用官方 nuScenes `000` 的 61 帧，相机 0–4 为模型候选输入，相机 5 完全留出。2026-09-26 00:53 时约 15,470/30,000 迭代；4k、8k 权重可用。4k 横移渲染的相机与 actor 时间戳门禁通过。留出相机 5 的 4k→8k PSNR 为 7.996→8.045 dB，而训练相机 0 对照为 20.999→22.648 dB；大面积空白尚在。此为未收敛阶段观察，30k 才作最终判断。帧 20 的 +3.5 m 方向进入近物遮挡，需选可通行方向/帧位。

## V7.5 截至收尾的已确认状态

- r9 版 **24 个 10 秒 paired case 获人工批准**，分为 6 个 ego 与 18 个对象 case；这取代旧文档的 `approval.status=pending` 现状。早期 OmniDreams 24 对生成仍是旧轮证据，不冒称 r9 新推理。
- HUGSIM 在 7 个场景完成 ground/scene 各 30k 与导出，24 个 case 均有 100 帧、10 Hz 的事实和反事实输出及自动指标。输入从官方 12 Hz 流程适配到 10 Hz；scene-0242 稀疏 COLMAP 与 scene-0998 相机更新存在质量风险。指标中的 outcome、轨迹和对象身份缺失值不补零。
- DriveEditor 的 18 个对象 case 有 10 帧原生 factual/counterfactual；其中 11 个另有完整 10 秒迭代 counterfactual。其 6 个 ego case 不属于原生编辑接口，另 7 个对象 case 没有完整迭代结果。原生 1 秒和迭代 10 秒不是同一评价分母；能力边界记于 [V75-F02](research_failures/entries/V75-F02.md)。
- ReSim 停止，GaussianDWM 仅有接口 smoke；RecEdit-Drive CPU 预检通过但专用 checkpoint 缺失，未启动推理。不存在可诚实计算的六模型 24-case 总分或统一排名。
- V7.5 保留包有 119 个 MP4、逐 case 结果、配置和两份人工评分工作簿；119/119 视频哈希已校验。2026-09-25 清理后旧模型权重、HUGSIM 七场景 checkpoint 及中间输入已退役，保留视频不是可直接恢复的训练状态。证据及复现边界见 [收尾记录](autoresearch/worldsim_v75/downstream_bench/r9-closeout/README.md)。

V7.5 之前的自然状态、对象移除、速度干预及接近任务控制仍由 [V7.5 文档](v75/PROBLEM.md)与 [实验索引](EXPERIMENTS.md)保留。旧版 V7.4 状态见[归档](archive/2026-09/v74-0920/STATUS_PRE_V75.md)，不会作为现行授权或当前资源状态。

## 下一门禁

VAD-GS P0 到 30k 后，复测相机 5 的 PSNR/SSIM/LPIPS、横移 0/0.5/1/2/3.5 m 和 actor 世界时间戳；再推进 scene-0230/0255 的同场景先验和 VAD-GS 全新训练。V7.5 的 HUGSIM 输出用于历史基线与数据对照，不用其 checkpoint 初始化 V7.6。
