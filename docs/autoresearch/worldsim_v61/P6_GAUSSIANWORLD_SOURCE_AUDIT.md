# WorldSim V6.1 P6 GaussianWorld source audit

记录时间：2026-08-22T12:55:00Z

## 决策

按 V6.1 计划的既定优先级，P6 首先执行 GaussianWorld pretrained 的 3090 capability smoke；只有该路线因
官方权重、数据合同或单卡资源失败时，才审计 OccWorld。ME-2 的 `V61-F09` 只拒绝 Hunyuan actor surface，
不外推为 learned occupancy rejection。

## 一手来源与冻结资源

- paper：<https://arxiv.org/abs/2412.10373>；
- official repository：<https://github.com/zuosc19/GaussianWorld>；
- official source commit：`b43629eaecffd5a7cbaac1a55517766e6263e4fc`；
- streaming checkpoint：`298029831` bytes，SHA-256=
  `5477081191e6d7fb8ac880bf5b99230b944a7cea76b0d87175c789a51cb97be3`；
- R101-DCN backbone：`177818375` bytes，SHA-256=
  `1ee46d1c3294a4ff51733013d1f27b08269c1435376dece9e8d70fa6c22a8ccf`；
- official temporal metadata：`530760430` bytes，SHA-256=
  `302fcb86cc5be290a0b3987da3753034ae4eea0bb711fd75af5b80c27ca5ab54`。

官方 README 明确提供 GaussianWorld streaming checkpoint、六相机 nuScenes 输入以及
`200×200×16 @ 0.5m` 的 SurroundOcc 配置。正式实验不下载 SurroundOcc label：模型 head 的 `label` 参数只被用于
恢复输出 spatial shape，P6 传入固定 empty dummy tensor，并把 `dummy_label_role=shape_only_not_truth`、
`surroundocc_label_read=false` 写入 worker evidence。不得用 dummy label 计算 loss、metric、calibration 或 acceptance。

## 本机数据兼容性

官方 train metadata 有 700 scenes；`scene-0048` 与 `scene-0242` 均存在，各 40 个 2Hz keyframes，官方 camera
顺序固定为 `FRONT / FRONT_RIGHT / FRONT_LEFT / BACK / BACK_LEFT / BACK_RIGHT`。DriveStudio 六相机 ID 对应为
`[0,2,1,5,3,4]`。本机 `scene-0048` 原始六相机/keyframe 文件完整；`scene-0242` 原始包不完整，但既有
DriveStudio scene191 保存了开发目标帧的六相机 JPEG、camera-to-world extrinsics、intrinsics 与 lidar-to-world pose。
适配只计算 `camera_from_lidar = inv(global_from_camera) @ global_from_lidar`，不读取 LiDAR 点、O_method、O_eval 或
confirmation 内容。

P6 只使用 `scene-0048/frame52` 做单帧 capability smoke，验证权重可完整载入、六相机投影合同成立、输出 finite 且
occupied/empty 都非空、peak VRAM `<22GiB`。它不产生 28-case 方法结论，也不据此选择 semantic/confidence threshold。

## 为什么不先接 OccWorld

GaussianWorld 是计划的第一优先级，且它直接消费本机已有的六相机图像与标定。OccWorld 的官方实现与权重以过去
occupancy 序列为输入；在本项目当前阶段直接接入会要求额外的 occupancy annotation/adapter，或者把 ME-0 oracle
误当模型输入，都会削弱“predicted occupancy 替换 oracle”的因果归因。因此只有 GaussianWorld P6 capability 失败时，
才允许一次 OccWorld source/resource audit，而不是同时铺开两个安装与参数分支。

## 环境、许可与停止规则

隔离环境遵循官方 installation 版本：Python3.8.16、Torch2.0.0+cu118、MMCV2.0.1、MMDetection3.0.0、
MMSegmentation1.0.0、MMDetection3D1.1.1、spconv2.3.6，并只为 RTX3090 编译两个官方 CUDA op。全局 Conda 的
废弃 `pkgs/free` 镜像不被修改；新环境单独使用 conda-forge 创建，随后仍按官方版本安装。

官方 repository 没有顶层通用 LICENSE；其中 local aggregator 源码声明只允许 non-commercial research/evaluation。
因此本轮仅在 AutoDL 内部科研运行，不分发代码、checkpoint 或模型输出，不把预测用作其他模型训练数据。

P6 若通过，直接预注册并执行一次 ME-3 development；若 P6 失败，只根据明确的 source/data/resource 根因决定是否
审计 OccWorld，不调图像尺寸、camera order、权重、类别映射或模型参数。ME-3 若出现 false-safe，优先保留 UNKNOWN/
abstain，禁止通过降低阈值恢复 yield。

## P6 正式结果

Canonical run：

```text
run://worldsim_v61/WS-V61-P6-GAUSSIANWORLD-3090-SMOKE-01/20260822T132526Z__gaussianworld-smoke-s1-r1
```

source=`95c842a883652f679cb1bee93bf1db0e3092c5b2`。官方 checkpoint 以 `0 missing / 0 unexpected`
完整载入，输出 shape=`1×18×200×200×16` 且 finite；occupied=`29608`、empty=`610392`。单帧 inference=
`0.8524s`、worker wall=`3.0384s`、peak=`2.1499GiB`，17 项 gate 全部通过。正式 worker 明确记录
`surroundocc_label_read=false`、`training_started=false`、`confirmation_content_read=false`，dummy label 仍只提供
shape。gate/summary/resource/manifest/terminal=`dd59fd9e...133 / da079429...b21 / b6dc3b48...9ac /
24b19cbb...0d9 / 8f886211...ab7`。因此不审计 OccWorld，直接进入冻结的 ME-3 GaussianWorld development。
