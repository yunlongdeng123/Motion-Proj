# WorldSim V7.2 GPU 开机交接

更新：2026-09-07

无卡实例可完成的 research、协议、数据适配、G0 raw fusion 和 G1 Actor TSDF 同协议诊断已完成。下一步需要至少 1×RTX 3090 24 GiB。

## 已就绪

- branch：`research/worldsim-v7.2-task-first-completion-lidar`
- CPU implementation commit：`691619c5f20de4af838a20d416a6818aa377b2b0`
- G0 canonical：`20260906T153519Z__g0-raw-fusion-cpu-r1`
- G1 canonical：`20260906T160409Z__g1-actor-tsdf-cpu-r3`
- G0/G1：相同 66 Actors、34 logs、五个密度预算和 evaluator；G1 峰值 RSS=`0.717 GiB`
- AdaPoinTr adapter：`20260906T154442Z__adapointr-legacy-export-r1`
- AdaPoinTr official checkpoint：`/root/autodl-tmp/external/worldsim_v72/checkpoints/AdaPoinTr_PCN.pth`
- source snapshots：PoinTr `4603257`、DyNFL `b6d03de`、LiDAR4D `4d6abbd`

## 开卡后按顺序执行

1. 只读核对 `nvidia-smi`、driver、GPU 型号/显存、`nvcc` 与磁盘；不假设无卡模式的软件状态。
2. 建立 AdaPoinTr 隔离环境，安装与实测 toolkit 匹配的 PyTorch；编译 Chamfer 与 PointNet++。旧 KNN_CUDA import 在 AdaPoinTr 当前 graph 路径未启用，不先增加该依赖。
3. 载入官方 PCN checkpoint，记录兼容/不兼容 keys；以 512-point partial、4096-point decoder 做一个 forward/backward batch，测显存与 wall。
4. 根据单 batch 与单 epoch实测冻结 batch/accumulation/epoch 预算，再启动 G3。外部预训练与从头训练分表，训练中不读取 source/external final。
5. 以同一 66-Actor cohort 重跑 G2 M8 surface，按 64/128/256/512/native 接入统一 evaluator，完成 G0--G3 geometry table。
6. W0--W4 使用同一 categorical reader；W3/W4 先匹配输入与容量，再单独增加 F/O/U 辅助监督。
7. 完成一个 LiDAR4D official sequence，或在取得 Waymo 授权后完成 DyNFL scene capability；之后才允许 D1 选路。

## 仍缺外部条件

- DyNFL：未具备 Waymo 数据授权/预处理资产。
- LiDAR4D：未下载完整 KITTI-360 sequence；现有 tracking smoke 不能替代。
- 独立数据：nuScenes 明确未暴露候选仅 10 logs；本地 AV2 80 logs 全部已暴露。

开卡后无需重跑 G0/G1，也无需重新生成 AdaPoinTr adapter。先完成 1--3 的 capability；若单 batch正常，再按实测资源继续 G2/G3 与 W0--W4。
