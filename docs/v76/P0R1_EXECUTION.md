# V7.6 P0R1：修复后全新复现与自主收口

实验ID：`VADGS-P0R1-000`。上游 VAD-GS commit `77e27686d84b64be4643d518cea77b34bf9718bf`，加入 [V76-F01](../research_failures/entries/V76-F01.md) 的按图像名字映射修复。当前状态只见 [RESEARCH_STATUS](../RESEARCH_STATUS.md)。

## Architecture components

```mermaid
flowchart LR
  A[官方 RGB / SIFT / 标定位姿] --> B[全量 exhaustive matching]
  B --> C[新三角化点 / 正确 track 映射]
  L[官方 LiDAR / 分割 / 法线] --> D[全新初始化]
  C --> D
  D --> E[VAD-GS 从零训练30k]
  E --> F[0–4 官方时间test]
  E --> G[相机5外推 / ego横移]
  F --> H[审核与保存 / v76推送]
  G --> H
  H --> I[充分收口后检查任务与控制器 / AutoDL关机]
```

## 本次控制条件

- 使用官方样例 `000` 的帧0–60、相机0–4、230梯度训练视图及75官方时间test。相机5继续作为独立外推诊断。
- 只复用旧run中同一官方RGB的SIFT关键点/描述子和输入标定位姿。新数据库的 `matches`、`two_view_geometries` 初始行数均为0；原库不修改。复用输入模型的 `points3D.txt` 确认空文件，旧三角化、input_ply和任何checkpoint均不复用。
- 采用官方 `exhaustive_matcher` 与三角化参数；本机COLMAP为3.7无CUDA构建，CPU 12线程、seed0。匹配集合恢复为穷举，不冒称CPU与上游GPU的浮点实现完全相同。
- `cfg.resume=False`，训练前断言新run不存在 `trained_model`；训练、模型和先验相关超参数沿用官方样例。相机/帧映射采用已验证的名字连接，原P0被INVALIDATED标记保护。
- 这是联合纠正工程错误与匹配偏离的复现检查，不能单独分解两项变化的因果收益。官方时间test仍有初始化先验使用边界，见[审计报告](P0_REPRODUCTION_AUDIT.md)。

## 执行与证据

运行目录：`/root/autodl-tmp/runs/v76_ego_view/VADGS-P0R1-000`。配置：`/root/autodl-tmp/external/worldsim_v75/VAD-GS/configs/v76/nuscenes_000_repro_exhaustive.yaml`。

`run_p0r1_pipeline.py` 是有限顺序队列，运行状态写入 `pipeline_state.json`，包括阶段、PID、返回值、日志和完成阶段。队列依次执行全量匹配、三角化、30k全新训练、75视图官方test、61视图Camera5外推、帧20横移0/0.5/1/2/3.5m。任何阶段失败即记录并退出，交由research scaling law分类修复；不无限自动重试。4k检查点出现时由当前任务优先完成工程渲染与track/几何门禁；30k结果需人工任务审核后才能成为阶段结论。

配置、输入复用说明及控制器源码归档于 [p0r1_exhaustive](../autoresearch/worldsim_v76/p0r1_exhaustive/)。启动命令为：

```bash
cd /root/autodl-tmp/external/worldsim_v75/VAD-GS
/root/autodl-tmp/envs/vadgs-v76/bin/python script/v76/prepare_p0r1.py
nohup /root/autodl-tmp/envs/vadgs-v76/bin/python script/v76/run_p0r1_pipeline.py \
  > /root/autodl-tmp/runs/v76_ego_view/VADGS-P0R1-000/pipeline.log 2>&1 < /dev/null &
```

两入口均拒绝重复创建或重复启动。`prepare_p0r1.py` 已执行，不再重跑。代码编译检查通过，COLMAP三角化线程参数已由本机help验证；实际全量匹配已启动并完成初始区块。

## 同场景准备与资源协调

CPU匹配期间已利用空闲GPU完成scene-0255的366张Depth Anything V2灰度深度先验，日志为其数据目录下 `depth_prior.log`；scene-0230的该项已有。DSINE官方代码与权重在准备中，模型来自官方[hub入口](https://github.com/baegwangbin/DSINE/blob/main/hubconf.py)指向的下载地址。SAM ViT-H已有本地checkpoint，可用于后续实例和背景分割；正式推理前需验证object ID编码、相机坐标与法线PNG编码。先验生成属于同场景适配，不能标为与官方示例完全同源。

官方VAD-GS的动态mask是三通道ID图，背景255，各channel可包含对象ID；背景SAM是uint8区域图。对象ID必须与 `instances_info.json` 的track一致。法线读取为RGB解码后整体取负并按c2w旋转，DSINE导出必须通过官方样例方向检查后使用。避免同时在单卡上执行主训练和重型SAM/DSINE推理。

## 停止与关机范围

2026-09-26用户最新明确授权：“我先睡了，你自己持续推进，如果都完成差不多就可以停下来，然后帮我autodl关机”。该授权适用于本任务的 `wm-3090-0811`，不需要再次确认；不延伸到其他主机，也不等于立即关机。

按原P0→同场景→ego stress顺序推进，根据有效证据与资源达到可交付阶段后可停止扩展，明确保留未解决范围。单个GPU空闲、训练退出、工程失败或checkpoint落盘都不是关机充分条件。收口时保存输入、关键权重、正反结果和报告，将小型证据取回本地并提交push v76；确认没有训练/评价/渲染/先验任务及会启动它们的控制器，再将当前heartbeat设为PAUSED，最后调用AutoDL官方 `/usr/bin/shutdown` 并检查结果。官方关机入口见[省钱说明](https://api.autodl.com/docs/save_money/)。不把“发出关机命令”写成已验证停机。

`failure_ledger_refs: [V76-F01]`；`failure_ledger_delta: none`。目前没有新训练指标，也没有方法成功或失败结论。
