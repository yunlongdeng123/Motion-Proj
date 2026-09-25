# V7.6 P0：复现与评估设置审计

记录时间：2026-09-26（Asia/Singapore）。本轮按用户要求先做两个诊断：COLMAP `<0.6 px` 过滤统计、相机 0–4 官方配置的时间留出帧评估。结果支持基础重建已运行，同时发现实际初始化中的工程错误；不支持否定 VAD-GS。当前执行状态见 [RESEARCH_STATUS](../RESEARCH_STATUS.md)。

## Architecture components

```mermaid
flowchart LR
  A[相机0–4 RGB / 位姿] --> B[COLMAP 点与 tracks]
  B --> C[误差与空间过滤]
  C --> D[按图像名字绑定视图]
  L[LiDAR / 法线先验] --> E[初始化几何与 visibility]
  D --> E
  E --> F[VAD-GS 全新训练]
  F --> G[高斯 + sky 渲染]
  G --> H[0–4 时间留出帧 / 基础重建]
  G --> I[相机5 / 未见方向外推]
```

## COLMAP 实测

`VADGS-P0-000-COLMAP-AUDIT` 直接读取原运行的 `points3D.bin`、`images.bin` 以及保存的 `input_ply/points3D_bkgd.{ply,npy}`，不重建或覆盖旧输入。完整[点与 track 统计](../autoresearch/worldsim_v76/reproduction-audit/colmap_audit.json)。

| 阶段 | 点数 | 含义 |
|---|---:|---|
| 原 COLMAP 模型 | 138,944 | 平均误差 1.120710 px，中位数 1.002675 px |
| 严格 `error < 0.6` | 33,930 | 保留 24.4199%；剩余点平均误差 0.381874 px |
| 后续空间过滤并进入初始化 | 15,756 | 与保存的 PLY 尾部 float32 坐标和顺序逐点精确一致 |
| 其他背景初始化点 | 214,123 | LiDAR 体素等；总背景点 229,879 |

原 track 观测边 672,617 条，误差过滤后剩 114,694 条（17.05%）。点数和观测边确实显著减少，但**保留比例单独不能证明稀疏匹配比 exhaustive 差，更不能给出它对画质的因果贡献**。LiDAR 仍占背景初始化的主要部分，COLMAP 点占 6.85%。空间过滤是官方代码既有行为，包括相机附近/低于相机及场景范围筛选，不将其全部归因于误差门限。

## 已确认工程错误：V76-F01

`lib/utils/drivestudio_utils.py` 原实现用 `(image_id-1)//num_frames` 和 `%num_frames` 推断相机、帧，再读取该视图的法线并设置 visibility。当前数据库中 **300/305 张图**不满足该假设。官方 COLMAP 明确区分有序索引与无序 ID，IMAGE_ID 不能当作视图序号，见[格式说明](https://colmap.github.io/format.html#indices-and-identifiers)。

对实际进入初始化的 15,756 个 COLMAP 点逐一核验：

- 保存的 visibility 全部复现了旧 ID 算术规则，证实错误已进入训练输入。
- **15,754/15,756 行**与依据 `images.bin` 文件名恢复的真实 track 不一致。
- 保存边数与正确边数均为 59,793，但交集仅 3,370 条（5.64%）；单独检查边数不能发现错绑。
- 按原 track 条目计，31,186 条绑定到了错误相机，58,408 条绑定到了错误帧。相同索引还用于选择法线图，因此法线来源也受影响；未测量由此引起的画质损失大小。

这段假设存在于所用上游 commit `77e27686d84b64be4643d518cea77b34bf9718bf`；本机生成的 COLMAP ID 顺序触发了它。尚不能把触发原因归于 sparse matcher，也不能因此判定论文方法失败。失败卡见 [V76-F01](../research_failures/entries/V76-F01.md)。

修复已部署到外部 VAD-GS 工作树：用 `images.bin` 的 `cam_N/frame.png` 名字连接实际 `(frame, camera) → view index` 表，同时修正 visibility 和法线图来源。旧文件备份为 `lib/utils/drivestudio_utils.py.codexbak.20260926-id-mapping`。三个回归测试覆盖乱序/不连续 ID、非零起始帧与相机子集、重复视图拒绝；实际 305 张图核验 0 错配；编译与 diff 检查通过。复现补丁与脚本见[归档目录](../autoresearch/worldsim_v76/reproduction-audit/)。**修复源代码不会修复旧初始化或 checkpoint；后续必须新建 run 并重新初始化。**

## Camera 0–4 官方时间留出帧

`VADGS-P0-000-TEST-16000` 显式加载现有 16k 权重，直接使用原始 `Scene.getTestCameras()`：15 个帧号 `4,8,...,60` × 5 相机，共 **75 视图**；梯度训练视图 230，交集为 0。保留原始 `is_val=True` 和官方 actor 位姿插值逻辑，未把它们改成训练时间戳。

| 相机 | 视图数 | PSNR ↑ | SSIM ↑ | LPIPS ↓ |
|---|---:|---:|---:|---:|
| 0 | 15 | 22.1295 | 0.7645 | 0.2224 |
| 1 | 15 | 20.1831 | 0.6713 | 0.2963 |
| 2 | 15 | 23.8995 | 0.8027 | 0.1809 |
| 3 | 15 | 22.8017 | 0.7138 | 0.2783 |
| 4 | 15 | 24.4913 | 0.7906 | 0.1786 |
| 全部 | 75 | **22.7010** | **0.7486** | **0.2313** |

使用官方 PSNR/SSIM 与 LPIPS Alex v0.1，全图评价，并保留官方保存/读取 PNG 的 8-bit 量化口径。附带的浮点 PSNR 为 22.7022 dB，量化差异约 0.0011 dB。逐视图结果见[metrics.json](../autoresearch/worldsim_v76/reproduction-audit/official_test_16k_metrics.json)，固定帧 20 的[GT/render/acc 对照](../autoresearch/worldsim_v76/reproduction-audit/official_test_16k_frame20.png)未按画质挑选。

75 视图接近纯白（RGB 各通道 >250/255）的平均比例为 0.1785%；白且 `acc<0.1` 的平均比例为 0.00362%。帧 20 的五相机图没有相机 5 那样的大面积白洞，但车辆、人物和高频纹理仍有模糊/伪影。低 acc 也可对应正常天空，不能把它直接当作全部几何空洞。评估进程正常退出，未发生 OOM。

这是**官方配置的时间留出**，不是完全未见数据：官方初始化阶段仍使用所有候选帧的 RGB、LiDAR、COLMAP 与先验。它也不是完整论文 benchmark 复现或最终收敛质量。此前相机 0 的 61 帧对照混合训练帧与留出帧，不能与本表当作同一评价集合；还存在 `is_val` 对 actor 插值的区别。

## 结论和研究边界

相机 5 从未进入候选训练视图、初始化可见性或 COLMAP 图像集合，其评估属于额外的未见相机外推。渲染器确实执行 `rgb += sky_color * (1-acc)`，且 sky cubemap 默认接近白色；这一代码路径能解释未训练方向的白色填充，但本轮未对相机 5 逐像素分解归因，不能仅由白色比例推断全部是低 acc。

本轮可以确认：基础重建已起来；复现初始化存在明确的 track/法线来源映射错误；稀疏 COLMAP 是另一项尚未量化影响的预处理偏离。不能据这次 run 宣称 VAD-GS 表示失败，也不能等待它到 30k 后忽略初始化错误。旧 P0 权重保留为诊断证据，修复后应使用新初始化和官方 exhaustive 对照，按相同官方 test split 检查。

- 原始证据根：`/root/autodl-tmp/runs/v76_ego_view/VADGS-P0-000/{v76_diagnosis,v76_official_test_16k,input_ply,trained_model}`。
- 资源：1×RTX 3090；COLMAP 审计为 CPU 只读；75 张测试使用现有 16k 权重，无新训练。
- `failure_ledger_refs`: `[V76-F01]`；`failure_ledger_delta`: `V76-F01`。
- 人工 verdict：null。
