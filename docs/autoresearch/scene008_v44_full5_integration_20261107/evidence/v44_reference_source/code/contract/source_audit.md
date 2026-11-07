# v44 输入与源码审计

2026-11-07：实际远端纯 CPU 预检 **PASS_V44_CPU_INPUT_CONTRACT**，552 项检查通过。核验 132 个小文件的 SHA256，共 241,285,078 字节；未读取或哈希多 GB 原生 checkpoint 的张量载荷。训练更新、重新校色求解、GPU 渲染均为 0。

![v44 组件与数据流](architecture.svg)

## 去蓝修复实际发生在哪里

RCR-1 将原 ownership 训练遗漏的全图联合 RGB 监督补回，系数为 1。实际执行源码位于：

`C:/Users/dengyunlong/Documents/Codex/2026-11-05/10k-rgb-v1-v41-b-case/outputs/delete_comparison_review/source_archive/actual_sources/native/executed/global_10k/executed_global_rgb_probe.py`

`LossProbeOwnership.terms()` 先保留原 `Ownership.terms()`，在 global 模式添加 `photometric_rgb_loss(result['render'], target, ssim, args.lambda_dssim)`。训练合同保留 static2、其他 loss、普通 XY/time 增长、SGCM、初始化、采样、学习率和正式 10K 步数。修复不是删车推理时的颜色过滤，也不是后拟合原生 SH。原推理源码审计记录显示，native SH 从 checkpoint 加载，road gate 只改临时 opacity，道路细色场只改独立 road RGB。

原始评审与采用范围：

`C:/Users/dengyunlong/Documents/Codex/2026-11-05/10k-rgb-v1-v41-b-case/outputs/delete_comparison_review/index.html`

`C:/Users/dengyunlong/Documents/Codex/2026-11-05/10k-rgb-v1-v41-b-case/outputs/delete_comparison_review/evidence/inference_code_audit.json`

旧 ownership native10K 的 lineage64990 有 64 颗蓝色 SH；RCR-1 native10K 有 7 颗，shader 输出黑色。该单项实验支持恢复 RGB 约束抑制持续正 B，未确定原来出现正 B 或两版 clone 数量不同的唯一原因。

## 可直接复用的模型与精确依赖

远端 campaign 根为 `/root/autodl-tmp/drivingeditor_repro/campaign`。本轮已实际确认以下文件存在，并通过输入契约核对：

| 组件 | 精确路径（相对 campaign 根） | 核验方式 |
|---|---|---|
| RCR 原生 10K | `scene008_blue_global_rgb_probe_20261105_r1/global_10k` | 四个 checkpoint 的既有 size/mtime 身份、三个 metadata 的 SHA |
| fresh road 1K | `scene008_blue_global_road1k_20261105_r1/formal/checkpoint001000.pt` | 小 checkpoint SHA、CPU step=1000、parent、repair tensor hash |
| fresh contributor/gate | 同 road run 的 `inputs/contributors.npz` | 82,427 个唯一静态索引，与 checkpoint ids 逐位相同，索引落于本版 1,042,776 个静态行 |
| road baseline | 同 checkpoint 的 `road.dc` | 143,248 道路行，`.5+C0*dc[:,0]`，原始 baseline hash 精确匹配 |
| 原 coarse/fine | `scene008_blue_global_M2_final_20261105_r1/coarse6x5`、`fine11x9` | 精确 source receipt、binding、系数 SHA、99×3 有限有界 world 系数、baseline、11×9 knots |
| 当前 C2 资产 | `scene008_a6_insert_artifacts_20261105/surface_fit/combined_v1/observed_fit/formal/complete_a6.npz` | SHA `0b1f69e13d0e54328734325c8f118bc27dc7b75080869f2d52f55038afc0b20d` |
| 当前曲面 posterior | 同 artifacts run 的 `surface_fit/masked_depth_v1/posterior.npz` | SHA `b98a0d932730fbf1462e555eb7d5eb931030b51fe68848990f073b9752879e06` |
| 当前 SurfaceLOD | 同 artifacts run 的 `source/scale/surface_lod_asset.py` | SHA `fb3c746a04bf17b1f84b127e44769b4ff294567e98aec744131ddc3f3c4d76ca` |

原生文件身份按原 RCR-1 receipt 精确保留：

| 文件 | 字节数 | mtime_ns |
|---|---:|---:|
| `chkpnt10000.pth` | 826934296 | 1793884537536887316 |
| `dynamic_chkpnt10000.pth` | 839984408 | 1793884540706054677 |
| `env_light_chkpnt10000.pth` | 226494608 | 1793884542546760397 |
| `training_state10000.pth` | 44824 | 1793884709726852182 |

该 10K 模型使用 raw089–137 的 49 帧 clip，并已保存完整训练状态；旧收据有真实 10000→10002 完整状态恢复证据。旧 RCR-1 输出保存了 49 keep 和 raw105/113/130/137 四帧 delete，不能把它们写成旧实验已验收 49 delete。v44 可以直接用现存模型渲染 49 帧，无需重跑 native10K。

## 组合边界与最小整合方案

RCR-1 的原 final runtime 是固定完整 M2 的专用实现，会强查 M2 的 27 项张量、degree3 和文件哈希。旧 M2 SHA 为 `d84133bb1a9767e00301857cc1d8f293f506faaafc58da598c8d0d7c0c8ae0e6`。当前 C2 与 SurfaceLOD 已有独立 schema 和持久状态，不能用旧 M2 替换，也不能把完整 M2 的 27 tensor 假定直接套到当前资产。

最小整合沿用已存的 RCR native10K、专属于此拓扑的 fresh road/gates，物理排除所有 native a6 行（包含后代），插入当前同一 C2 或同 S/F SurfaceLOD 的 rows。保留 native a37 的原时间包络；delete 只排除 a37。原生、资产、a37 与道路进入同一次原 CUDA 共享深度排序，再加 `(1-alpha)*EnvLight`。本合同不改变相机、FoV、原生 SO、曲面、材质、opacity 或采样算法。

当前 C2 有 45,314 行：42,536 chart + 2,778 retained。虽然文件声明 SH3，所有行的非 DC 系数精确为零。CPU 已核对 geometry/provenance hash、RGB 与 DC、唯一整数 UV、posterior 重建 canonical XYZ/sigma，以及 raw106 参考相机和 local pose。SurfaceLOD 必须随 wrapper 一起复用，裸 NPZ 默认 dense rows 不等价于 LOD。

原 fine 的 `binding_sha256=6a955f0fa78646264366d985e45b481b7bf8698998764927bf906e5730943e00` 包含旧完整 M2，因此**原 fine 是精确 source binding，不是针对新 C2 重新标定的场**。本轮 v44 明确采用 `frozen_transfer_from_rcr1_m2`：保留原 field 系数和原 source receipt，不改写旧 metadata，也不假造新 binding。fine SHA 为 `131bc37ed2711701ef5526caed8021a5971fa10a56ae670d3cd6a82dffc250e3`；它是 baseline 的绝对 residual，计算 `clamp(baseline + basis @ world, 0, 1)`，不再叠加 coarse residual。

换资产会改变共享排序下的可见性和 alpha×T，因此冻结转移的输入一致性不构成新组合的画质验收。其是否仍满足去蓝、道路与受保护区域质量，留待 v44 自己的真实 GPU 对照。

## 银灰亮片与既有范围

raw130/raw137 银灰亮片是保留问题。原 RCR-1 已在 raw137 的 315 个新增亮点核得实际 premultRGB 增量：native static `(+0.293346,+0.333195,+0.366756)`；environment×T `(-0.200625,-0.191717,-0.186205)`；road 约 `(+0.000019,+0.000022,+0.000028)`；M2/a37 为零；最终净增量 `(+0.092740,+0.141500,+0.180580)`。净增亮由静态背景主导，环境×T 仍带车形结构。

这些点原来全部被 a37 遮挡，且均在当前 road prior 支持外；fit/native_keep/repair/vehicle/marking 权重为零。具体静态行、EnvLight 与透射率的各自空间份额、唯一训练原因仍未隔离。换用当前 C2 不自动解决这项问题；旧用户采用范围仅为异常蓝色 RGB 修复，未证明所有区域画质无回退。HELD9 只对道路颜色 target 留出，不是独立 native 泛化集。

## 本阶段交付与收据

- `v44_contract.py`：可导入的纯 CPU checker；`validate_contract(path)` 返回严格 PASS 或 BLOCKED。没有 scene/runtime/wrapper/renderer import，只有 numpy、torch 的 CPU 反序列化与 scipy spline。
- `input_contract.json`：实际远端路径、固定 SHA、原 source receipt 身份、冻结转移政策。
- `cpu_preflight_receipt.json`：远端实际 552 项 PASS；GPU 渲染与画质标记仍为 false。
- `cpu_preflight_attempt1_blocked.json`：保留首次工程阻塞，原因是 Torch checkpoint 的 tuple 与 JSON list 直接比较；现改用 canonical JSON hash，数值与实验方法未改。
- `evidence/rcr1_source_receipt.json`：原始 RCR-1 运行收据逐字复制，SHA `57909831f49aaba80ab38cfc1ec4ee682fb5eee8092e5b20b5f81be9dd8fb8f6`。
- `evidence/existing_remote_snapshot.json`：本次只读采集的额外小文件 SHA 与原 selection/knots。

本阶段远端仅新增独立 `scene008_v44_rgb_asset_20261107_r1/contract`。未覆盖旧 campaign、模型或源目录，未启动训练或 GPU 渲染，未关机。
