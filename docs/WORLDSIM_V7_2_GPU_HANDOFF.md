# WorldSim V7.2 GPU 执行状态

更新：2026-09-07

GPU 已恢复为 1×RTX 3090 24 GiB。G0--G3 几何图谱和 AdaPoinTr 三种初始化对照已经完成；当前不需要重跑这些任务。

## 已完成

- branch：`research/worldsim-v7.2-task-first-completion-lidar`
- G0：`20260906T153519Z__g0-raw-fusion-cpu-r1`
- G1：`20260906T160409Z__g1-actor-tsdf-cpu-r3`
- G2：`20260906T174000Z__g2-m8-matched-s71110-r1`
- G3 official zero-shot：`20260906T165000Z__g3-adapointr-official-zs-r3`
- G3 pretrained-adapted 600 epochs：`20260906T165500Z__g3-adapointr-transfer-s7203-r1`
- G3 scratch 600 epochs：`20260906T174200Z__g3-adapointr-scratch-s7204-r1`
- 相同 evaluator：66 Actors、34 logs、99,208 条正回波射线、64/128/256/512/native 五档密度
- source/external test read：`false/false`

AdaPoinTr 独立运行环境为 `/root/autodl-tmp/envs/worldsim-v72-pointr`，Torch=`2.4.1+cu121`，CUDA toolkit=`12.1.105`。Chamfer 与 PointNet++ 已按 `sm_86` 编译；AMP batch 64 实测峰值约 `19.96GiB`。两次 600-epoch 训练分别使用约 52 分钟，训练 Actor 全量常驻 GPU。

## 已得到的关键结论

- 官方 PCN zero-shot 存在严重域差，不能作为拒绝 AdaPoinTr 的依据。
- 官方预训练适配版在所有固定预算下稳定优于 scratch，说明预训练确有迁移价值。
- G3 pretrained 在 64/128/256/512 固定点数下仍未越过 G0/G1 简单前沿；4096 点原生输出的 CD 优势含明显密度因素。
- G2 的 early 更低，但以 CD/F-score 和 hit 损失为代价，形成保守权衡点而非全面胜出。
- 详细主表见 `docs/WORLDSIM_V7_2_D0_GPU_RESULTS.md`。

## 下一执行顺序

1. 固定 G0/G2 几何和同一 categorical reader，完成 W0--W4：单位权重、build-only 支持率、局部密度归一化、单标量 MLP、F/O/U head。
2. 先比较 W3/W4 的同回波损失版本，再单独加入 F/O/U 辅助监督，避免混淆结构与额外标签。
3. 在 legacy diagnostic 上只作机制筛选；分配干净 dev/route-select 数据后才允许 D1。
4. 路线 B 的 capability 仍需完整 KITTI-360 sequence，或正式 Waymo 授权与预处理资产。

## 仍缺外部条件

- DyNFL：Waymo 数据授权／预处理资产未具备；Nerfstudio 0.3.4 与 CUDA 扩展未安装。
- LiDAR4D：未下载完整 KITTI-360 sequence；现有 KITTI tracking smoke 不能替代。
- 独立数据：nuScenes 明确未暴露候选仅 10 logs；本地 AV2 80 logs 全部已暴露。
- 旧 cohort 只有正回波 target，不能支持 full-return 主张。

无需重跑 G0--G3，也无需重新生成 AdaPoinTr adapter。D1、source test 和 external test 继续冻结。
