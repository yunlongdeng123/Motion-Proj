# WorldSim V6.1 ME-3 GaussianWorld failure and backend audit

记录时间：2026-08-22T14:00:30Z

## 结论

GaussianWorld 的 3090 capability 成立，但它在固定 ME-3 上不能作为 safety authority：预测臂虽然完整保留 oracle 的
`10/28` 接受集合与 mask-area yield，却产生 `10/10` false-safe。该结果不是 coverage 不足，也没有证据支持通过
confidence threshold、坐标修补或重复回测恢复。GaussianWorld 路线按 `V61-F11` 停止。

计划内后端按输入和权重契约审计后，只有 IR-WM 同时满足“vision-centric current state + 官方任务权重”。先运行一次
capability smoke；它不消耗 ME-3 唯一恢复。只有 capability pass 才允许一轮 separately preregistered IR-WM ME-3，
否则直接结束 learned occupancy 路线。

## Canonical evidence

```text
run://worldsim_v61/WS-V61-ME3-PREDICTED-OCC-01/20260822T134559Z__predicted-occ-s1-r1
```

- source=`4c048ecd2db834ae494deb998947136f9918d9bb`；
- predicted=`10 ACCEPT / 0 ABSTAIN / 18 REJECT`，false-safe=`10`；
- oracle=`10/28`，predicted/oracle yield fraction=`1.0`；
- 4/4 target outputs，28/28 decisions，两个 scene workers overlap；
- wall=`28.3599s`，GPU peak sum upper bound=`4.4688GiB`；
- training/confirmation/calibration/threshold selection 均为 false；
- 唯一失败 gate=`predicted_zero_false_safe`。

10 个接受例都来自 scene-0242。route-support 的隐藏 observed-FREE conflict ratio=`0.766..0.958`、observed support
约=`0.013..0.019`、relative depth consistency≈`0.402..0.406`；actor removal/disocclusion conflict=`0.159..0.328`、
support=`0.419..0.495`、relative depth consistency=`0.036..0.053`。这说明 accepted geometry 与隐藏观测 FREE 的冲突
是系统性的，而不是一个边界 case。

## 坐标与类别审计

GaussianWorld 官方 `gaussian_occ_head.py` 用 `x/y/z` meshgrid 生成 `200×200×16 @0.5m` 网格；可视化与 metric
源码一致地把 class1..16 视为 occupied、class17 视为 empty、class0 排除。ME-3 的 class mapping 没有颠倒。

DriveStudio 官方 nuScenes preprocess 保存原始 camera-to-world 与 lidar-to-world，NuScenes 的 `OPENCV2DATASET` 为
identity。将 scene-0242 official temporal metadata keyframe10/11 与 DriveStudio processed frame50/55 直接比较，后相机
`lidar2img` 差异约 `1e-10`，前相机最大矩阵差约 `0.04..0.10`，与 closest-image 异步 timestamp 相符。这排除了
camera order、world/lidar 方向、x/y swap 或额外 OpenCV 轴翻转。没有依据改 adapter。

## 不采用 post-hoc uncertainty threshold

- [ReliOcc](https://arxiv.org/abs/2409.18026) 用 uncertainty learning 与 offline calibration 缩小 camera occupancy
  的 reliability gap；它不是 frozen checkpoint 上即插即用的阈值。
- [α-OCC](https://arxiv.org/abs/2406.11021) 用 uncertainty propagation 与 hierarchical conformal prediction，仍需要
  calibration/prediction-set 合同。
- [OCCUQ](https://arxiv.org/abs/2503.10605) / [official code](https://github.com/ika-rwth-aachen/OCCUQ) 使用 GMM
  uncertainty module 与拟合过程；其公开结果还显示 naive max-softmax/entropy 的 OoD discrimination 很弱。

这些方法可进入后续“重新训练+独立 calibration split”的新研究，但不能作为当前 confirmation-locked、no-training
ME-3 的事后修复。现在从同一28例上挑 confidence threshold 会同时构成 tuning 与 truth leakage。

## 后端资格审计

### OccWorld：不进入

[OccWorld paper](https://arxiv.org/abs/2311.16038) 与
[official repository](https://github.com/wzzheng/OccWorld) 的任务是从过去 3D Occupancy 预测未来 Occupancy。
当前项目若接入它，必须提供 GT/predicted past occupancy；前者把 ME-0 oracle 偷渡回 predictor，后者又依赖一个尚未通过的
camera occupancy backend。它不能独立替换当前 `O_predicted`，故不做 3090 smoke。

### Drive-OccWorld main：不进入

[Drive-OccWorld paper](https://arxiv.org/abs/2408.14197) 与
[official repository](https://github.com/yuyang-cloud/Drive-OccWorld) 是 vision-centric，但主分支没有发布任务 checkpoint；
官方 release 为空，安装文档只下载 R101-DCN backbone。官方 evaluation 需要自备 epoch checkpoint、fine-grained occupancy
或约650GB Cam4DOcc annotations，并以8 GPU脚本为主。缺少任务权重时不能做 inference-only recovery。

### IR-WM：只进入 capability smoke

[IR-WM paper](https://arxiv.org/abs/2510.16729) 的官方 branch commit=
`a83e4a24a8dbf008e5fe4e02d7efb692c1bec582`；它从历史相机建立 current BEV state，再预测 action-conditioned residual。
[Official weights](https://huggingface.co/Jianbiao/IR-WM) revision=
`36b16b55d21f773f080f5980b195aa2ece5b9358`，公开 fully/semi/tightly-decoupled 三个 checkpoint。本轮固定最少耦合的
`MMO_MSO_with_plan_fully_decoupled.pth`：`941598147` bytes，SHA-256=
`8e1816dc290df28f1e89d0b88eddb451b9ab20fc85a8ef6449d9226ceeacd1ce`。

smoke 只抽取 official current-state occupancy logits，不评价未来 planning：固定 scene-0048 两历史帧六相机、官方归一化、
标定与 ego motion；不读 `gt_occ`、nuScenes-Occupancy、O_method/O_eval 或 confirmation。要求 checkpoint state `0/0`、
current output finite 且 FREE/OCCUPIED 非空、peak `<22GiB`、wall `<1200s`。失败即止；通过后再冻结 ME-3 mapping、
UNKNOWN policy 与唯一恢复 gate。

## Anti-loop boundary

1. 不再运行 GaussianWorld threshold/grid/schedule/checkpoint/axis 变体。
2. 不跑可由现有10个 conflict直接推导出的“observed-FREE veto → 0 accepted”回测。
3. 不安装 Drive-OccWorld main 的650GB标注链，不训练或校准 uncertainty。
4. IR-WM capability 只允许一次 source/data/resource 闭包；不能 truth-free current-state inference 即停止。
5. capability pass 后最多一轮 ME-3 IR-WM scientific recovery，false-safe必须为0，不降低任何阈值。
