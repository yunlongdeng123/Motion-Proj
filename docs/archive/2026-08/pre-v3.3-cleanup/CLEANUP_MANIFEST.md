# V3.3 启动前存储清理账本

- 执行日期：2026-08-11
- 范围：V3.2 以及之前路线中 V3.3 不再使用的过程产物
- 执行性质：存储维护；未启动 V3.3 task、run、下载、训练或评测
- 当前分支：`research/worldsim-v3.3-object-maintenance`
- 清理程序与机器清单：
  `/root/autodl-tmp/cleanup_manifests/20260811-pre-v33-space-reclaim/`

## 结果

| 项目 | 清理前 | 清理后 | 变化 |
|---|---:|---:|---:|
| 文件系统可用空间 | 42,818,088,960 bytes | 270,524,063,744 bytes | +227,705,974,784 bytes |
| 约合 GiB | 39.9 GiB | 251.9 GiB | +212.1 GiB |
| 文件系统使用率 | 87% | 17% | -70 percentage points |

dry-run 与正式执行使用同一清单：

- 整目录目标：63 条，apparent bytes=`189,344,502,399`；
- 非 canonical payload 文件：20,746 条，bytes=`55,106,030,330`；
- `delete_dirs.tsv` SHA-256=`5aa0da92feeabc26fcbb49e00bf1319a392dbb2bad690818862b332c284f6f3e`；
- `delete_files.tsv` SHA-256=`6382c153799baf730890326e0d01338c2dcf5684f2eb02333da61347a282d27f`。

apparent size 会受文件系统分配与硬链接影响，实际释放量以 `df` 差值为准。

## 删除类别

- V1/V2/V3.1/V3.2 非 canonical run 中的 checkpoint、tensor/point payload、图像、视频和 panel；
- V3.1 D2 canonical 的 5 个中间 checkpoint，保留 `checkpoint_final.pth`；
- DGGT/AD-GS/旧 SAM2 环境、checkpoint 与不再使用的源码 checkout；
- 旧 nuScenes/AD-GS/OccGS/ReSim 数据副本及 scene-specific raw staging；
- pip、Conda、torch 和不再使用的 Hugging Face cache；
- V3.2 未选或阻塞路线的第三方 checkout；
- Harmonizer full diffusion pickle，保留 non-temporal JIT；
- 旧 S1 prompt v1/v2 与失败准备目录，保留 canonical v3。

旧 run 的轻量 manifest/config/summary/terminal/log/source snapshot 未删除。空的 `/root/autodl-tmp/cache`、
`checkpoints`、`weights` 根目录在清理后重新建立，以保持稳定路径。

## V3.3 白名单与校验

以下资产在清理前后 SHA-256 exact：

| 资产 | SHA-256 |
|---|---|
| V3.1 D2 FP32 final | `1a061247a753c0d8c9aa7835a52efa2ab1ddc79141a6168adc18b9748de66e7c` |
| S1 high sidecar | `0119e15d979c1c69cdaea23d9c4716ae66f323972cd4abf3027fb43593ce22cb` |
| S1 boundary sidecar | `25b1eaecb2bb53af852ec2c5fae1abf53636359f9714e2de6e9a330187f22050` |
| S2 generated-background | `3d6e13d47291f5b5949ff3adf5598b6e0cffb930c4cbff2200c6e708d82e6e0f` |
| S3 selected actor | `b0c1f413e1a462292a1e3396ad45b8a8fc10f87f647e4bc3e1b98a4c8913caf0` |
| R0 mixed checkpoint | `6d4e4c489f53bf4e7de3f5c405ec37dc63d3f79155aad5237fe175ce0fcd7e5d` |
| R0 chunk manifest | `af7b402e0b171b11f8c22e4123002f4f844db746ea72f53b77c3de878bf0947d` |
| AH multiview diffusion | `92a25a615a78999477df764ed6616c0cf1dd8e08c5c12cf776dd03bbcbc72270` |
| AH TokenGS lifting | `576a4250e373547c6864cc3fa6ec310b7c66dd06b8025d609ec6681405896ff8` |
| AH camera estimator | `4e698b079a667054c825e84247c72da0cdc3272808d95f34dc33d23ef3d52885` |
| Harmonizer non-temporal JIT | `ece8e2daa914e8c2a027a2da94e0eb2064491d5b3fd8514009fae9a442e06e90` |

额外驻留检查：

- processed scene `179/191/204` 均存在；
- `motionproj` / `drivestudio` / `worldsim-v32-asset-harvester` 三个 Python 环境可执行；
- PyTorch 分别为 `2.4.1+cu121` / `2.1.2+cu118` / `2.10.0+cu128`；
- DriveStudio、Asset Harvester、Inpaint360GS、Harmonizer 与 Asset Harvester 离线 HF cache 均存在；
- 六个 V3.1/V3.2 canonical run 目录均存在。

## 恢复与不可逆边界

- 已删除缓存、公开源码和环境可按冻结 commit/revision 重建；
- 已删除 raw staging 可从原始数据重新物化；
- 已删除的旧 run 大载荷本机不可直接恢复，必须以历史轻量证据解释；若未来确需复现，创建新 task/protocol/run，
  不得修改旧 terminal；
- canonical 白名单与清理清单均保留机器 SHA 证据，因此本次清理不改变既有研究结论。
