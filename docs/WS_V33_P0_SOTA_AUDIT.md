# WS-V33 P0 路线与 SOTA 审计

- 日期：2026-08-11
- Task：`WS-V33-P0-ROUTE-SOTA-AUDIT-01`
- 状态：`done`
- 项目基线：`a055fc6727dddacd194665d5c997a1fe47c2d2f4`
- 分支：`research/worldsim-v3.3-object-maintenance`
- 事实配置：[`../configs/worldsim_v33/p0_sources_v1.yaml`](../configs/worldsim_v33/p0_sources_v1.yaml)
- 审计器：[`../scripts/audit_worldsim_v33_sources.py`](../scripts/audit_worldsim_v33_sources.py)
- canonical run：`20260810T171744Z__p0-source-audit-s0-r2`
- config/summary/manifest/status SHA-256：
  `29c167fe050d074f626884c0eba7b67fd6fd56c8493adc4c6be0d390f09b9ae2` /
  `08806b5f197d524207aa5d527b9976a993042b6451fa0cc9b0458a20b3a1d68a` /
  `2603ff0e037931aef8f8c84606038bd748600c99cef1a2a29cc82c621c51a12d` /
  `91096d0eae7616f2c68d133796922b49d475220c0c18fb7c438ca3655a32072d`
- 下一授权：仅 `WS-V33-S1-OBJECT-AWARE-GS-01`

r1 的全部 source/asset/gate 审计已通过，但其 manifest 没有冻结 auditor、module 与 test 源码，因此只保留为
noncanonical `done` 证据。r2 新增三份 source snapshot，SHA-256 分别为
`96024691ea6a26e312d21120205648f9b33bd68fcaf1dd68ff017799dea3c065`、
`5760318f8e44e538fb039977d68d570a09bc246a997b9b3bbc69f7e77314fec3`、
`0fe51b7c0accebf27349d544d79f2d9c4cd31f53b98b395b5482f8e293f7a778`；未改写 r1。

## 1. 裁决摘要

P0 将 V3.3 的可执行边界收缩为三类：

1. 有官方源码、许可与现成资产的能力可以进入后续阶段；
2. 只有论文思想、没有官方 runnable source 的能力只能做 `inspired implementation`；
3. gated 权重、缺作者导出模型或许可受限的能力保持 `weights_blocked/audit_only`，不通过手写大模型或从零训练绕过。

| 路线 | 审计固定点 | 许可/权重 | P0 裁决 |
|---|---|---|---|
| Meta SAM 3.1 | `96914d2425f9` / tree `573deb167702` | Meta SAM License；HF 未登录、无本地 checkpoint | `weights_blocked`；S1 使用 V3.2 SAM2.1 exact fallback |
| OP2GS | arXiv `2605.20044` | 未发现官方 runnable source/license/weights | `source_not_released`；只实现 OP2GS-inspired instance-opacity sidecar |
| Inpaint360GS | `d54c893285c6` / tree `671626f4825c` | Apache-2.0；CropFormer/LaMa 外部权重 | `executable` source；S2 adapter 后做单卡最小 preflight |
| GS-RoadPatching | `468f81258758` / tree `d4505b9be1b5` | 无 LICENSE；仓库仅静态项目页，算法源码文件数为 0 | `source_not_released`；实现 GS-RoadPatching-inspired RoadPatch-Lite |
| 3D-GIMP | arXiv `2607.20789` | 未发现官方 runnable source/license/weights | `source_not_released`；只审计，不手写完整系统 |
| FocusGS | arXiv `2607.28834` | 未发现官方 runnable source/license/weights | `source_not_released`；只吸收 spatial delta / erase-insert 接口思想 |
| R3D2 | `3fc6e317d9fe` / tree `759dd48d47b7` | Apache-2.0；只有 `sd-turbo/taesd` base，仓库无作者导出 R3D2 pipeline | `weights_blocked`；禁止单卡从零训练 |
| GOR-IS | `eb36accba0a3` / tree `d03c79c35016` | 自定义非商业研究许可；无 pretrained manifest | `audit_only`；不阻塞 RoadPatch 主线 |
| LiDAR-EVS | arXiv `2603.14763` | 未发现官方 runnable source/license/weights | `source_not_released`；仅保留 R0 后 future interface |
| Asset Harvester | 继承 V3.2 `767b2439ce47` | Apache-2.0；V3.2 三权重与 2-view actor 已 exact | `executable`；V3.3 只研究 train-only 自动 view selection |

`executable` 在本报告中表示“官方 source/安装链可进入后续独立环境 preflight”，不表示本次 P0 已执行模型或证明 RTX 3090 质量/资源结论。

## 2. V3.2 immutable 基线复核

P0 对实际文件重新计算 SHA-256，不只引用文档：

| 资产 | bytes | SHA-256 | 结果 |
|---|---:|---|---|
| V3.1 D2 checkpoint | 578,819,674 | `1a061247a753c0d8c9aa7835a52efa2ab1ddc79141a6168adc18b9748de66e7c` | exact |
| V3.2 S2 checkpoint | 579,269,554 | `3d6e13d47291f5b5949ff3adf5598b6e0cffb930c4cbff2200c6e708d82e6e0f` | exact |
| V3.2 S3 2-view actor | 3,798,161 | `b0c1f413e1a462292a1e3396ad45b8a8fc10f87f647e4bc3e1b98a4c8913caf0` | exact |
| V3.2 R0 mixed checkpoint | 432,347,490 | `6d4e4c489f53bf4e7de3f5c405ec37dc63d3f79155aad5237fe175ce0fcd7e5d` | exact |
| V3.2 R0 chunk manifest | 141,427 | `af7b402e0b171b11f8c22e4123002f4f844db746ea72f53b77c3de878bf0947d` | exact |

R0 `status.json` 仍为 `done`，8/8 gates 为真；九个 V3.2 定向测试文件重跑得到 `36 passed in 2.50s`。第一次直接调用 `pytest` 因项目未 editable install、根目录不在 `PYTHONPATH` 而在 collection 阶段失败；按仓库真实入口使用 `PYTHONPATH=.` 后全部通过，没有测试逻辑失败。

## 3. 官方源码与许可细节

### 3.1 SAM 3.1

- 官方 `main` HEAD、local commit 与 tree exact；
- README 明确要求 Python `>=3.12`、PyTorch `>=2.7`、CUDA `>=12.6`；
- SAM 3.1 `model_builder.py` 指向 `facebook/sam3.1/sam3.1_multiplex.pt`；
- 当前主环境的 `hf auth whoami` 返回 `Not logged in`，HF cache 中没有 SAM3.1；
- P0 不请求或绕过 gated access，也不下载 checkpoint。

因此 S1 的主可执行输入是 V3.2 canonical SAM2.1 masks。未来用户合法取得 SAM3.1 权重后，必须用新 task/protocol/run 解锁 P1/P2/P3 arm。

### 3.2 OP2GS 与 FocusGS

两篇论文分别提供 dual instance opacity 与 spatial delta/erase-insert 的清晰表示思想，但审计时没有可固定的官方实现、根许可证或 checkpoint。V3.3 可以独立实现相应思想，必须命名为 `OP2GS-inspired`、`FocusGS-inspired`，不得写成 reproduction。

### 3.3 Inpaint360GS

官方仓库完整包含 object-aware 3DGS training、object removal、LaMa color/depth inpainting 与 3D inpainting pipeline。上游只声明在 CUDA 11.8 / RTX 4090 验证，默认 `resolution=2`，并允许按显存选择 `1/2/4/8`。P0 没有安装依赖、下载 CropFormer/LaMa 权重或执行训练。S2 只允许在 StreetGS adapter、split 和正式分辨率合同冻结后做一个 scene-0230/object 的最小单卡 preflight；若必须静默降正式分辨率、改变 heldout 或越过 24 GiB，则记录 `blocked_single_3090`。

### 3.4 GS-RoadPatching 与 3D-GIMP

GS-RoadPatching 官方 GitHub 当前只有 HTML/CSS/JS/图片，零个 `.py/.cu/.cpp/.sh` 算法文件，也没有根许可证。3D-GIMP 的 arXiv 记录没有给出可固定的官方仓库。因此 RoadPatch-Lite 必须是本项目的受控 inspired implementation：只用 native Background donor、train/dev 搜索、heldout confirmation、1/2/4 m patch、top-K=5、刚性 BEV placement 和完整 donor provenance。

### 3.5 R3D2 与 GOR-IS

R3D2 已公开训练、验证、export、单图/视频 eval 代码，但 README 的首次运行只会下载 `stabilityai/sd-turbo` 与可选 `madebyollin/taesd` base；仓库没有作者训练并导出的 R3D2 pipeline。按计划不得在单卡 3090 从零训练主模型。

GOR-IS 已公开 intrinsic material/light、nvdiffrast 与 OptiX gtracer 链，但许可证明确限定 non-commercial research/evaluation，依赖与 torch/CUDA 没有冻结版本，也没有 pretrained manifest。它只作研究比较接口，不是 V3.3 必做路径。

## 4. 机器与资源合同

- GPU=`NVIDIA GeForce RTX 3090 24,576 MiB`，compute capability=`8.6`；
- driver=`580.105.08`，driver-reported CUDA=`13.0`；
- cgroup memory max=`96,636,764,160 bytes`，`oom/oom_kill=0/0`；
- P0 后数据盘约剩 `40 GiB`，高于 `20 GiB` 停止门，但没有空间支持无约束复制和大型权重矩阵；
- 五个可审计 source checkout 合计约 `636 MiB`；
- P0 没有训练、模型推理、依赖安装、>5 GB 权重下载或 DriveStudio mutation；
- GPU stage、tmux/controller 均未启动。

## 5. P0 关闭与 S1 解锁

P0 只解锁 `WS-V33-S1-OBJECT-AWARE-GS-01`：

1. 固定 scene-0230 high/boundary actor 和 identity 三元组；
2. exact 使用 train-only SAM2.1 canonical masks；
3. base RGB means/scales/quats/SH/opacity 全部 immutable；
4. 只学习独立 `instance_opacity_logit` sidecar；
5. 先 synthetic renderer correctness，再 100-step real smoke；
6. heldout 只做最终指标，不进入超参数和 candidate 选择；
7. 无提升时合法收口为 `object_field_no_gain`，不得追加事后 loss 或更换 actor。

S2–S5 在 S1 收口前仍未授权。
