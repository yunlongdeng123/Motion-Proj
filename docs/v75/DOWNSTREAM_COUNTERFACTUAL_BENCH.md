# V7.5 下游反事实 benchmark 准备

> 本页记录 V7.5 早期设计与准备时的资格状态。后续 r9 24-case 已获人工批准，并完成部分模型路径；最新执行分母、保留资产和未覆盖能力见 [r9 收尾证据](../autoresearch/worldsim_v75/downstream_bench/r9-closeout/README.md)。以下历史 `manual_pending` 不代表当前批准状态。

## 结论

V7.5 调整为：**在高斯重建状态上施加反事实干预，并以编辑遵循、物理合理性、环境不变性及下游响应为核心发现问题。** 第一阶段不是做大而全的 leaderboard，而是冻结 24 个 paired edits（四类各 6 个），逐模型运行并保留六个独立维度，不合成总分。

这六个系统并非同一种模型。统一的是 case、输出和评价合同，而不是强行统一它们不存在的输入能力：

| 系统 | 在本 bench 中的角色 | 允许的原生/适配入口 | 禁止的等价替换 |
|---|---|---|---|
| OmniDreams | 轨迹/结构化条件生成器 | ego 条件原生；非 ego 结构化条件为 V7.5 适配 | 不把适配接口写成论文原生对象编辑能力 |
| ReSim | ego action world model | ego 速度与横向轨迹 | 不把 ego `fut_traj` 当非 ego actor 编辑 |
| DriveEditor | 非 ego 对象视频编辑器 | reposition/insert/delete，逐帧 3D box 可表达运动 | 不在单 24GB GPU 上伪称官方配置可运行 |
| GaussianDWM | 下游 understanding/generation consumer | 在 paired 编辑输出上做 grounding、planning 或 future generation | CVPR 公共版没有 4D edit API，不作为编辑器计分 |
| Street Gaussians | 显式高斯重建/组合基线 | actor 刚体轨迹、移除、重组；复用历史证据 | 历史 checkpoint 已退役，不把旧图当本轮新推理 |
| HUGSIM | 高斯闭环参考模拟器 | ego/actor 更新、插入及闭环 outcome | 没有对齐导出场景时不启动空跑 |

该边界来自各官方发布：ReSim 公共接口是 future ego trajectory；DriveEditor 官方推理强调对象编辑且声明单卡需要超过 32GB；GaussianDWM CVPR release 明确只有 QA/world generation、没有 4D edit；Street Gaussians 提供 reconstruction/rendering；HUGSIM 提供闭环场景与 actor 更新。官方入口见 [ReSim](https://github.com/OpenDriveLab/ReSim)、[DriveEditor](https://github.com/yvanliang/DriveEditor)、[GaussianDWM](https://github.com/dtc111111/GaussianDWM)、[Street Gaussians](https://github.com/zju3dv/street_gaussians)、[HUGSIM](https://github.com/hyzhou404/HUGSIM) 与 [OmniDreams/FlashDreams](https://github.com/nv-tlabs/omni-dreams)。

## Architecture components

![V7.5 downstream counterfactual benchmark architecture](../autoresearch/worldsim_v75/downstream_bench/architecture-components.svg)

数据流是：同一事实观测与重建状态 `(x,a,z)` 进入 paired case compiler；只改变一个变量得到 `z′`；各方法 adapter 输出事实/反事实视频或状态；六轴 evaluator 与 GaussianDWM 下游 consumer 分别回答“改动是否成立”和“改动是否改变下游理解/规划”。

## Pilot case 合同

`scripts/build_worldsim_v75_downstream_bench.py` 从现有三个 nuScenes DriveStudio 10Hz 场景的实例轨迹做 CPU 筛查，生成 24 个**候选**，不是自动通过资格：

- actor slowdown/acceleration：6 个，其中 3 个 ego、3 个 non-ego；
- actor lateral relocation/lane change：6 个，其中 3 个 ego、3 个 non-ego；
- actor removal：6 个 non-ego；
- actor insertion：6 个 non-ego。

每个 case 都记录 factual/counterfactual 控制、目标实体、branch frame、共享前缀、预期 outcome 与不变范围。CPU 几何资格脚本已让 24/24 case 通过 source-track coverage、目标可见性与反事实初始碰撞 gate，状态为 `geometry_pass_manual_pending`。原处理数据没有 map/lane layer，因此道路有效性不能自动给真值；24 张 review sheet 已生成，而 `road_validity` 与 `human_verdict` 仍保持 `null`，必须由人工审阅，脚本不会代填。

三个来源只够做工程 pilot 和问题发现，不够支持发生率、泛化或模型排名。后续若扩展数据，应新增冻结来源而不是在这三个场景内继续堆更多高度相关 case。

## 六个维度

评价协议在 `configs/worldsim_v75_downstream_bench/evaluation_protocol.json`。本轮报告固定：

1. Adherence：干预有没有执行；
2. Physics：运动、尺度、接触和遮挡是否合理；
3. Environment preservation：背景、相机和非目标对象是否保持；
4. Outcome：paired 最终状态是否按干预逻辑分叉；
5. Trajectory adherence：ADE/FDE、中心/yaw 或轨迹差；
6. Object/background preservation：资产身份、目标外区域与移除后的背景是否保持。

A/P/E 同时允许 single 与 paired 证据，O 只接受 paired。缺失值保持 missing，不写成 0；unsupported 不是模型失败；pilot 不输出 composite/overall score。

## GPU-free 准备结果

本轮完成：

- 官方源码冻结：新建 ReSim sparse checkout、DriveEditor、GaussianDWM、Street Gaussians 浅克隆；复用已有 FlashDreams 与 HUGSIM checkout；
- 统一 model registry：记录源码 revision、角色、能力矩阵、环境/权重/数据路径和资源边界；
- case builder：从实例元数据筛持续 vehicle track，生成 24 个 paired 候选；
- preflight：只检查 source/environment/weights/data/prior evidence，不调用 GPU；
- run planner：按固定顺序生成每模型可执行 case，当前明确 `execution_enabled=false`；
- result schema/evaluator：六维独立校验与均值汇总，拒绝 composite score；
- 单元测试：覆盖 24-case 平衡、单变量合同、能力分层与无总分汇总。

进一步完成了可直接复用的输入和下载层：

- ReSim：六个合法 ego case 已分别编译 factual/counterfactual trajectory JSON 与 YAML；公开 transformer、VAE、T5 分片按官方目录布局保存；
- DriveEditor：18 个 non-ego case 已编译为 18 个独立 official-format pickle，避免一次反序列化约 784MB 合并对象；每项均验证为 10 帧 900×1600 RGB、3×3 内参与 0/255 mask，并提供 batch runner；
- GaussianDWM：只取 scene 0179/0191/0204 的公开 sampled Gaussians，共 18 个 camera view、715 帧，未下载约 304GiB 全量训练集；
- HUGSIM：官方 scene-0383 sample reconstruction input、导出场景、配套 scenarios/map cache，以及该场景五档 scenario 实际引用的 6 个 3DRealCar 资产均纳入 materializer；
- 下载与 preflight 都把邻接 `.aria2`、`.part`/`.incomplete` 和显式最小尺寸视为未完成，避免稀疏预分配或断点文件被误报为 ready；
- 六套环境用独立 Python 解释器检查核心 import；不加载权重、不建 CUDA tensor、不执行模型前向。

## GPU-free 阶段已经暴露的问题

这些问题本身就是 pilot 的第一批发现，不能等到 GPU 推理后再处理：

1. **模型角色不可互换。** ReSim 只原生接受 ego future trajectory；GaussianDWM CVPR release 是 paired 世界的下游 consumer；把它们都写成“actor editor”会让 Adherence 分数失去含义。
2. **GaussianDWM 发布资产与 loader 不闭合。** 官方 sampled `.pt` 实际是裸 `torch.Tensor[N,14]`，但 CVPR loader 的 `_load_one` 只接受 NumPy 或 mapping。bench 保留原文件，并另生成只加 `{"packed": tensor}` 容器的兼容副本；数值不改。官方 torch 2.6 环境实测原文件报 `TypeError`，包装后得到 `[16000,14] float32`，合同差异已确认。
3. **DriveEditor 参考资产存在实例分离风险。** 源数据动态 mask 是聚合 mask，不是逐实例 mask。编译器以目标 3D box 投影选择连通域并记录面积/中心偏移，但 contact sheet 中仍有遮挡或弱 crop；这些 case 必须通过人工 reference review，不能因 pickle 结构合法就算输入合格。
4. **HUGSIM 的 sample data 不是可运行权重。** 2.4GB `sample_data/data.zip` 是重建输入；闭环还需要发布页中的导出 `scene.pth/cfg.yaml/ground_param.pkl`、scenario、地图缓存与 3DRealCar。发布的 scenario 仍保留旧 `/postprocess/shadow.pth` 后缀，bench 按上游 `export_multiple_scenes.py` 的逻辑规范化；资产 registry 使用精确文件，不再用任意 `.ply` 误判 ready。
5. **道路 gate 缺数据支持。** DriveStudio pilot 场景没有 map/lane layer；平滑轨迹、可见性和无初始碰撞只证明几何上可执行，不证明目标仍在可行驶区域。因此 24 个 case 仍不能自动进入正式榜单。

Street Gaussians 的历史结论保留为 `prior_evidence_reusable`：既有 matched reconstruction/actor composition 结果可用于预期与 adapter 设计，但 checkpoint 已在存储退役中释放，所以新的 paired render 需要恢复 checkpoint 或重训，不能把历史 evidence 冒充本轮推理。

实际 CPU preflight 由 `preflight.json` 记录；这里的“缺权重”是 fail-closed 状态，不会触发 GPU 调用：

| 系统 | 当前 readiness | 已确认边界 |
|---|---|---|
| OmniDreams | `ready_for_gpu_preflight` | source、环境、2B 权重和 pilot 数据均在位 |
| ReSim | `ready_for_gpu_preflight` | 23.7GB transformer、VAE、T5、独立环境及 6 个 ego paired adapter 均在位；只跑 ego trajectory |
| DriveEditor | `source_only_missing_weights` | 环境、227MB demo 和 18 个 official-format 输入在位；12.1GB model 遇 Google Drive 24h quota，且单 3090 不满足官方显存要求 |
| GaussianDWM | `ready_for_gpu_preflight` | 22.2GB 模型全套 config/tokenizer、715 帧兼容 Gaussian 与独立环境在位；只进入 consumer 流 |
| Street Gaussians | `prior_evidence_reusable` | 历史证据可复用；旧 checkpoint 已退役，保留环境缺 `simple_knn`/rasterizer 且无卡主机无 `nvcc` |
| HUGSIM | `ready_for_gpu_preflight` | 官方导出 scene-0383、map、5 scenarios、6 cars 和环境在位；完整 closed loop 仍需外部 AD client，且不是 179/191/204 对齐场景 |

六环境核心 import 为 5/6 通过：OmniDreams、ReSim、DriveEditor、GaussianDWM、HUGSIM 通过；Street Gaussians 按上述编译扩展缺失 fail-closed。ReSim 另固定 `pyarrow==14.0.2`，修复上游 `datasets==2.14.4` 对新 pyarrow 已删除 API 的依赖；DriveEditor 固定 `setuptools<81`。GaussianDWM 官方测试为 55 passed、1 skipped，QA/world 两个 CLI 的 `--help` 均可解析。

生成物位于 `docs/autoresearch/worldsim_v75/downstream_bench/`：`cases.json`、`qualification.json`、`assets.json`、`environment-smoke.json`、`preflight.json` 和 `run-plan.json`。planner 把 GaussianDWM 的 24 个 case 明确记为 `consumer_only`（生成数 0、消费数 24），并把 24 个 case 的人工资格数保持为 0，避免把 CPU geometry pass 当成人工合格。

## 开卡后的顺序与停止规则

固定顺序为 OmniDreams → ReSim → DriveEditor → GaussianDWM → Street Gaussians → HUGSIM。GaussianDWM 的时序位置是先完成公共包/QA/world inference smoke，再消费已有 producer 的 paired outputs；其编辑维度记为 consumer-only。

每个模型按以下规则推进：

1. 先通过 source、environment、weights、data、case qualification 五个 gate；
2. 先跑 1 个 paired smoke，再跑该模型合法的 frozen cases；
3. OOM、官方单机资源要求不满足、输入接口需要伪造或输出合同缺失时立即停止，不自动降分辨率或换任务；
4. 任何 proxy/adapter 结果与原生结果分栏；
5. 先报告 per-case 六维失败，再决定是否值得补数据或研究机制。

当前 GPU-free 阶段不执行模型推理、不开训练、不恢复旧队列。
