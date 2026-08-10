# V3.3 S3 自动视图选择与完整 Actor 资产

## 1. 裁决

`WS-V33-S3-ASSET-VIEWSELECT-01` 已在单卡 RTX 3090 上收口为 `done`。

- high-support 生产策略：`A4_auto_4view`；
- high-support heldout：相对 V3.2 manual 2-view，IoU `+0.023490`、boundary F1 `+0.059889`，
  PSNR `-0.015760 dB`、LPIPS `+0.008527`，四项冻结门全部通过；
- boundary-support：自动选择与官方生成链执行成功，但相对 immutable D2 native actor 的开发集保留门失败，
  生产 override 明确 `ABSTAIN_GENERATED_OVERRIDE`；未读取 boundary heldout；
- scene-0242/0255 条件确认未执行：当前没有与本任务冻结协议一致的 V3.3 S1/S2 mask/actor 输入链，且
  boundary transfer 已拒绝；不得混用旧 V3 资产补表；
- 生成背面只声明 completeness/consistency，不声明 GT correctness。

S3 的突破不是修改 Asset Harvester 网络，而是把人工冻结 `frame 91 / frame 51+91` 升级为训练侧、可复现、
防泄漏的 observation selection，并用开发集选臂、一次性 heldout 确认和 native boundary transfer gate 阻止
“能生成即能生产”的错误结论。

## 2. 冻结输入与上游

- scene：`scene-0230`，DriveStudio processed root=`trainval/179`；
- D2 checkpoint SHA-256：
  `1a061247a753c0d8c9aa7835a52efa2ab1ddc79141a6168adc18b9748de66e7c`；
- train SAM2 manifest SHA-256：
  `efc7d82145dfe0cf3c1bfccf230b0756f152ed5b306610b2634ddbcdbabab98b`；
- heldout mask manifest SHA-256：
  `f99cc1150a5c964bb9348b7b3dabf4b26ed4edcea9e2bdaf75a12bc6f4897c1a`；
- high identity：dataset ID `13` / token `af663976...c5c29` / rigid index `5`；
- boundary identity：dataset ID `41` / token `18c7f0c5...31d4` / rigid index `21`；
- NVIDIA Asset Harvester source：`767b2439ce47a8b2513038ae0fb2073026f89ee8`，tree clean；
- diffusion/lifting/camera weights SHA-256：
  `92a25a61...2270 / 576a4250...6ff8 / 4e698b07...52885`；
- VAE revision=`ca69e17e97609e64ce055115a6515215109b1f50`；
- C-RADIO revision=`c3ab05bfe0a70819f9af060dfb0424d12eaf03a4`；
- HF 全程 offline，未下载或漂移传递依赖。

## 3. 训练侧视图选择

候选只来自 train frames，并显式排除：

- 19 个 heldout frames；
- 10 个预留 development frames；
- actor 不可见、SAM mask 未接受、面积不足、D2 effect 不足、严重截断的 view。

每个候选真实渲染 D2 `original/delete`，形成 counterfactual effect；单视图分数为：

```text
Q_view =
0.20 × area
+ 0.20 × mask confidence
+ 0.15 × Laplacian sharpness
+ 0.20 × D2 visible fraction
- 0.15 × nearby-dynamic/box occlusion
- 0.10 × boundary truncation
```

连续量先用候选池 5/95 percentile 做 robust normalization。集合分数在 `ΣQ_view` 上增加 circular yaw、时间、
相机多样性；同相机至少间隔 5 帧，任意视图 yaw 至少相差 3°，用固定 beam width `512` 做确定性 1/2/4-view
搜索。manifest 完整保存 candidate/selected、score components、yaw、mask、occlusion、sharpness、输入文件 SHA。

formal 重跑必须提供 diagnostic 的 expected selection/input SHA；不相等立即失败。高支持和边界角色的
diagnostic/formal 都得到 byte-exact 相同结果。

## 4. High-support canonical 链

### 4.1 Selector r2

```text
/root/autodl-tmp/runs/worldsim_v33/WS-V33-S3-ASSET-VIEWSELECT-01/
20260810T201345Z__s3-viewselect-high-formal-s0-r2
```

- candidates=`130`，eligible=`119`；
- `heldout_read=false`，`reserved_development_read=false`；
- 1-view：`f091 CAM_FRONT_LEFT`；
- 2-view：`f000 CAM_FRONT + f091 CAM_FRONT_LEFT`；
- 4-view：`f011 CAM_FRONT + f083/f089/f094 CAM_FRONT_LEFT`；
- selection/input/summary/status SHA-256：
  `192e5035...b7be9 / 34b1e09e...ef2e7 / 65485a63...0311 / b4c0af50...376c`；
- wall=`203.05 s`，peak CUDA allocated/reserved=`8,315,726,336 / 8,860,467,200 bytes`。

### 4.2 官方 Asset Harvester r3

```text
/root/autodl-tmp/runs/worldsim_v33/WS-V33-S3-ASSET-HARVEST-01/
20260810T201830Z__s3-asset-high-formal-s0-r3
```

- 官方 30-step、CFG `2.0`、BF16、CPU offload；
- 三份 PLY 均非空，SHA-256：
  `13ff42b6...299b / 9bb7f925...8c4a / 9e9875b8...6ca0`；
- inference manifest SHA-256=`e33fcc65...2d90`；
- wall=`160.189 s`，peak NVIDIA memory=`20,137 MiB`；
- cgroup high/OOM/OOM-kill=`0/0/0`，memory pressure=`false`；
- source commit/tree、三套权重、VAE/C-RADIO revision 和离线 cache 均进入 manifest。

### 4.3 StreetGS importer r4

```text
/root/autodl-tmp/runs/worldsim_v33/WS-V33-S3-ASSET-IMPORT-01/
20260810T202210Z__s3-import-high-formal-s0-r4
```

三臂全部执行 PLY→actor NPZ、确定性重序列化和写后重载：

| Arm | Gaussian | bytes | asset SHA-256 |
|---|---:|---:|---|
| A1 | 101,988 | 3,885,576 | `00397310...5290` |
| A2 | 100,783 | 3,871,536 | `1a6d9300...6859` |
| A4 | 99,241 | 3,791,327 | `06d5db85...ec13` |

A4 enriched manifest SHA-256=`4590c1bd...7343`；所有 arm 的 `reload_exact=true`、
`deterministic_reserialization=true`。

### 4.4 Development r13

```text
/root/autodl-tmp/runs/worldsim_v33/WS-V33-S3-ASSET-EVAL-01/
20260810T205300Z__s3-eval-high-development-formal-s0-r13
```

development views 固定为 `f005/c0, f025/c0, f065/c1, f085/c1`，不读 heldout。统一渲染
`original/lateral+1m/delete`。

| Arm | IoU | Boundary F1 | LPIPS | PSNR dB | max outside L1 | max lateral fragmentation |
|---|---:|---:|---:|---:|---:|---:|
| A0 manual-2 | 0.669876 | 0.517563 | 0.090755 | 17.718824 | 0.028807 | 0 |
| A1 auto-1 | 0.658477 | 0.497629 | 0.092720 | 18.480375 | 0.004445 | 0 |
| A2 auto-2 | 0.664463 | 0.544166 | 0.095948 | 17.980733 | 0 | 0 |
| A4 auto-4 | **0.701490** | **0.604799** | 0.098533 | 17.694031 | 0.022481 | 0 |

三条 auto arm 的六项 retention gate 全部通过；冻结 metric order 后选择 A4。decision/summary/status SHA-256=
`28d4f75c...82bf / c3a9be4f...56e0 / 98ab859e...619e`。

### 4.5 一次性 Heldout r14

```text
/root/autodl-tmp/runs/worldsim_v33/WS-V33-S3-ASSET-EVAL-01/
20260810T205600Z__s3-eval-high-heldout-formal-s0-r14
```

只携带 A0 与 development 冻结的 A4；views=`f020/c0, f040/c0, f060/c1, f090/c1`，无选择或优化。

| Arm | IoU | Boundary F1 | LPIPS | PSNR dB | cross-view consistency |
|---|---:|---:|---:|---:|---:|
| A0 manual-2 | 0.704974 | 0.505017 | 0.094170 | 17.025697 | 0.940240 |
| A4 auto-4 | **0.728464** | **0.564906** | 0.102697 | 17.009936 | 0.937789 |

delta=`IoU +0.023490 / boundary F1 +0.059889 / LPIPS +0.008527 / PSNR -0.015760 dB`；四项
heldout gate 全过，decision=`accepted`。decision/summary/status SHA-256=
`795ecbc5...8032 / 7e329c0d...a767 / 2afeda52...39cc`。D2 checkpoint before/after SHA exact，
`optimization_performed=false`，最终 evaluator snapshot 与提交源码 byte-exact。

## 5. Boundary-support transfer

### 5.1 Selector/AH/import

- selector formal r8=`20260810T203700Z__s3-viewselect-boundary-formal-s0-r8`；
- candidates/eligible=`135/127`，无 heldout/development read；
- A4=`f039 CAM_FRONT + f146/f151/f156 CAM_FRONT_LEFT`，yaw=`17.95°→84.53°`；
- selection/input SHA-256=`ba05b224...505f / 5cb0ab6c...5856`；
- AH r9=`20260810T204100Z__s3-asset-boundary-formal-s0-r9`，wall=`161.119 s`，peak=`20,137 MiB`，
  inference manifest SHA-256=`2bd2d011...54bd3`；
- importer r11=`20260810T204600Z__s3-import-boundary-formal-s0-r11`；A4=`94,835` Gaussian、
  `3,632,764 bytes / 9b2295e5...5dd1f`，manifest=`2ec79f14...d79f`，reload/re-serialization exact。

### 5.2 Native retention gate r12

V3.2 没有 boundary manual asset，因此禁止把 high actor 的 A0 错配过来。评估器声明 `A0_native`：直接用
immutable D2 actor 渲染 original/lateral/delete；A4 仍走 remove+insert。同一固定 development views 为
`f045/c0, f065/c0, f125/c1, f145/c1`。

| Arm | IoU | Boundary F1 | LPIPS | PSNR dB | max lateral fragmentation |
|---|---:|---:|---:|---:|---:|
| A0 native | 0.666562 | 0.555343 | 0.015414 | 31.006882 | 0.004427 |
| A4 auto-4 | 0.624832 | 0.492141 | 0.111043 | 16.396747 | 0 |

A4 的 IoU、boundary、LPIPS、PSNR retention gates 失败，decision 回退 `A0_native`。r12 summary/decision/status
SHA-256=`8ce2cfbc...11fd / 52f4f6eb...b543 / 6772104c...18d`。可视化显示 A4 存在蓝灰色拉伸和车体形变，
与指标一致。因此：

```text
boundary_support generated asset = research artifact
boundary_support production override = ABSTAIN_GENERATED_OVERRIDE
boundary heldout read = false
```

## 6. 失败与不可改写 run

- view-selector r0 在首个候选发现 `obj_to_world` 实际为 list，而旧 parser 只接受 JSON string；无 selection 输出，
  修复为两种 schema 都做 4×4/finite 校验后新建 r1/r2；不得续写 r0；
- boundary importer r10 因编排层传错 inference manifest SHA 被 importer fail-closed；r10 不续写，正确 SHA 用于 r11；
- high r5/r6 已得到与最终完全相同的指标，但随后 evaluator 增加 native baseline 支持，源码快照不再等于提交态；
  r13/r14 用同一冻结资产与视图重跑并作为 canonical；
- boundary development 失败是生成资产相对 native retention 的研究负结果，不是 AH 工程失败；不得通过读取 boundary
  heldout、放宽门或换 actor 复活。

## 7. 验证与后续接口

- S3 专项=`11 passed`；
- V3.3 + V3.2 定向回归=`63 passed`；
- 五个 S3 Python 入口/module 均通过 `py_compile`；
- staged `git diff --check` 通过；
- canonical selector/AH/import/eval 的 source snapshots 与对应执行源码 exact；最终 r13/r14/r12 evaluator
  snapshots 与提交态 exact；
- 无 D2 checkpoint mutation、无训练/optimizer step、无 heldout view selection。

S4 只允许使用：high-support A4 asset SHA=`06d5db85...ec13`。boundary 生成资产只作审计，不得进入 production
delta。S4 必须继续 immutable base + erase/insert、compose/rollback SHA exact 和独立 provenance。
