# 当前：V8.1 有限补证收口（2026-09-13）

WS-V81-CLOSE-01完成：10旧候选内部复核；8新AV2日志840格模型前筛查，3日志18项新推理。旧440项未重跑、无训练。木板墙最强法向候选参考不稳；灰墙297点/0.103m好例保留。VGGT两个新日志有0.429/0.290m可恢复差距，单图度量锚点后MAE0.018/0.472/0.337m；DVGT两个后向单相机坐标/覆盖合同未建立，不计科学failure或稳健负例。

四环证据不足，V8.2 NO_GO；当前共同失效子命题降低优先级，停止本批次。其他8方法已纳入方法族badcase report，DGGT等方法级实证仍缺，条件下游扩展未触发；不能宣称整个V8.1已测完。无自动续跑、确认reserve封存、未关机。人工verdict=null；failure_ledger_refs=V81-F01/F02/F03/F04；failure_ledger_delta=V81-F05。

[有限收口报告](WORLDSIM_V8_1_BOUNDED_CLOSEOUT_REPORT.md)；[方法族badcase report](WORLDSIM_V8_1_METHOD_BADCASE_REPORT.md)；[推进判定](WORLDSIM_V8_1_TO_V8_2_DECISION.md)。以下为历史。

# GPU阶段补记（2026-09-13）

DVGT-1与raw VGGT均已各完成220项官方权重推理，实际Torch2.4.1+cu121/BF16/3090环境已运行6图和18图；不宣称作者整套评测复现。DVGT导出须除gt_scale_factor=0.1、RDF→FLU并使用首输入相机ego时间戳，修正见[V81-F03](research_failures/entries/V81-F03.md)。raw VGGT相机基线定尺度与区域外INPUT LiDAR尺度控制分表。DGGT仍只有CPU合同与权重准备，没有本轮GPU/renderer结果；其他不可运行方法没有新增复现。

[当前科学报告](WORLDSIM_V8_1_SCIENTIFIC_REPORT.md)。以下为CPU方法审计历史。

# V8.1 方法可执行性审计

2026-09-13；task `WS-V81-CPU-01`。这里的“代码/权重公开”不等于已经通过本机 GPU 推理验证。原计划中的方法级 failure 全部仍为待检验假说。

| 方法 | 官方版本 / checkpoint | 本轮用途与边界 |
|---|---|---|
| DVGT-1 | [官方仓库](https://github.com/wzzheng/DVGT)，commit `51cf3f6d11fdff8bc7e2bbe1a88f71665ccb2236`；[官方权重](https://huggingface.co/RainyNight/DVGT-1/tree/4918b6ed3547a00fd217d602fabb84ada180dba4) | 第一主模型。固定 DVGT-1，不混入同仓库 DVGT-2。点图在首帧 ego 系，原生宽512、patch16、2Hz；原生输出不作 ROI 真值尺度拟合。伪标签生产失败是否传到推理仍待实验。 |
| VGGT | [官方仓库](https://github.com/facebookresearch/vggt)，commit `a288dd0f14786c93483e45524328726ab7b1b4ce`；[VGGT-1B](https://huggingface.co/facebook/VGGT-1B/tree/860abec7937da0a4c03c41d3c269c366e82abdf9) 既有官方 safetensors | VGGD 的 prior probe。原生宽518、patch14；同时保留原始深度单位及相机基线定尺度后的深度。使用 dataset calibration 定尺度，不读取 held-out LiDAR。不能称为 VGGD 输出。 |
| DGGT | [官方仓库](https://github.com/xiaomi-research/dggt)，commit `a3276d2bbe4cbb03bcc117830b1836110a27adeb`；[官方模型库](https://huggingface.co/xiaomi-research/dggt/tree/735ac9a6486057b1eb886c33a8c6dc79e0b43214) | 模型库确有 `model_latest_nuscenes.pt`，README 的“其他权重稍后发布”已落后于模型库。准备原生 core 输出与 Gaussian tensors；完整动态渲染/扩散 refinement 待第一轮 geometry 结果后按需进入，不能把 core depth 当最终 Gaussian 渲染深度。 |
| DriveMVS | [官方仓库](https://github.com/Akina2001/DriveMVS)，commit `7c3f6d811667509a6d6572709e23ac1f28cc96cf` | 仅 README，0 个 Python 文件；Tier B / literature boundary。已准备同点数不同位置的 INPUT_PROMPT，未复现 DriveMVS。 |
| FocusGS | [官方项目页](https://focusgs.github.io/) | Code 实际为 `https://github.com/YOUR REPO HERE` 占位链接；Tier B。ambiguity localization / completion failure 未检验。 |
| VGGD | [论文 v1](https://arxiv.org/html/2608.10682v1)；[官方仓库](https://github.com/JHLin42in/VGGD)，commit `fb95aa1b4f8ce616bafb056160f4670eef0af99e` | README + LICENSE，0 个 Python 文件；Tier B。prior lock-in / recovery 不能用 raw VGGT 结果代替。 |
| PointForward | [论文 v1](https://arxiv.org/abs/2605.11594v1) | 作者页面写代码待发布；本轮不进可复现实验主表。 |
| LGS | [论文 v1](https://arxiv.org/abs/2608.11077v1) | LiDAR 驱动的结构干预路线，论文域为 Waymo/PandaSet。未建立本机官方 checkpoint/eval contract；只保留扩展假说。 |
| ReconDrive | [官方仓库](https://github.com/TuojingAI/ReconDrive)，commit `d2bc397b724d6cc021da22f8f57ad6af1cc53e3c` | 已确认有63个Python文件；未因此自动通过 checkpoint / nuScenes evaluation contract。主 failure 明确后再扩展。 |
| P2GS | 原计划指定的 CVPR 2026 一手论文 | 仅 photometric confound 观察位；未完成本轮可运行性审核，不列执行模型。 |

资源和运行版本记录见 `docs/autoresearch/worldsim_v81/` 及完整 run 的 `official_assets.json` / `runtime_versions.json`。Git commit 用普通版本追溯，没有引入哈希或指纹门控。

推理数据合同：使用完全相同的原始 RGB、frame/camera/ROI，各方法保留自己的官方 resize。输出按实际 resize 的 pixel-center affine 回到原图查询，避免“统一 resize”改变 baseline。DVGT-1 的 DINOv3 完整代码固定到 `6876159a11b4df116f30f667f8c9888617df0751`；仅在构造期间关闭冗余 DINO 单独权重下载，随后用官方完整 checkpoint 严格加载全部参数。这不改变前向网络或模型权重。

参考几何坐标变换参照 [nuScenes 官方 devkit](https://github.com/nutonomy/nuscenes-devkit/blob/master/python-sdk/nuscenes/utils/data_classes.py)。本轮按每扫描 ego pose 做运动补偿；原始点文件没有逐点时间戳，因此不声称已完成精确逐点 deskew。参考支撑质量与最终 scientific acceptance 分开记录。

最终CPU状态：DVGT-1/VGGT-1B/DGGT nuScenes完整权重均已落盘，三个模型strict meta参数名/形状加载通过；三个官方预处理在真实6相机输入上通过。没有GPU前向。记录见meta_*.json与cpu_input_contracts.json。Torch2.4.1+cu121与DVGT官方建议2.8/12.8不同；实际GPU兼容性待开卡验证，不将meta通过称为完整复现。
