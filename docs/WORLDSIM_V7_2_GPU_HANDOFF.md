# WorldSim V7.2 GPU 开机交接

更新：2026-09-07

GPU 已恢复为 1×RTX 3090 24 GiB；无卡阶段的 G0/G1 与开卡后的 AdaPoinTr capability 均已完成。

## 已就绪

- branch：`research/worldsim-v7.2-task-first-completion-lidar`
- CPU implementation commit：`691619c5f20de4af838a20d416a6818aa377b2b0`
- G0 canonical：`20260906T153519Z__g0-raw-fusion-cpu-r1`
- G1 canonical：`20260906T160409Z__g1-actor-tsdf-cpu-r3`
- G0/G1：相同 66 Actors、34 logs、五个密度预算和 evaluator；G1 峰值 RSS=`0.717 GiB`
- AdaPoinTr adapter：`20260906T154442Z__adapointr-legacy-export-r1`
- AdaPoinTr official checkpoint：`/root/autodl-tmp/external/worldsim_v72/checkpoints/AdaPoinTr_PCN.pth`
- source snapshots：PoinTr `4603257`、DyNFL `b6d03de`、LiDAR4D `4d6abbd`
- runtime：`/root/autodl-tmp/envs/worldsim-v72-pointr`，Torch `2.4.1+cu121`，CUDA toolkit `12.1.105`
- extensions：Chamfer 与 PointNet++ 均已按 `sm_86` 编译并通过 forward/backward
- capability：FP32 batch 48 峰值 reserved `19.63GiB`；AMP batch 64 峰值 `19.79GiB`、约 `107 samples/s`
- checkpoint：4096-point decoder 可载入 333/335 tensors，仅重置输出维度相关的 2 个 tensors

## 当前执行顺序

1. 先评测官方 AdaPoinTr zero-shot，单独标记外部预训练暴露。
2. 以 AMP batch 64 运行 600-epoch pretrained-adapted 与 scratch 配方；二者分表，训练期间不读 holdout。
3. 以同一 66-Actor cohort 重跑 G2 M8 surface，按 64/128/256/512/native 接入统一 evaluator，完成 G0--G3 geometry table。
4. W0--W4 使用同一 categorical reader；W3/W4 先匹配输入与容量，再单独增加 F/O/U 辅助监督。
5. 完成一个 LiDAR4D official sequence，或在取得 Waymo 授权后完成 DyNFL scene capability；之后才允许 D1 选路。

## 仍缺外部条件

- DyNFL：未具备 Waymo 数据授权/预处理资产。
- LiDAR4D：未下载完整 KITTI-360 sequence；现有 tracking smoke 不能替代。
- 独立数据：nuScenes 明确未暴露候选仅 10 logs；本地 AV2 80 logs 全部已暴露。

无需重跑 G0/G1，也无需重新生成 AdaPoinTr adapter。当前从 G3 三种初始化对照继续。
