# Scene008 a6 射线排序与局部接管

承接[局部接管](../scene008_a6_continuous_takeover_20261105/README.md)。PVD、道路颜色场、原生几何和已采用的裁剪保持固定。黑底为主实验，另用相同资产复核无编辑和删 a37 的街景。新增重建训练 0 步。

~~~mermaid
flowchart LR
  A["固定原生场景与 PVD"] --> C["实际光栅候选与每条射线的 alpha"]
  B["真实 RGB、LiDAR 与保护域"] --> D["逐行遮挡影响和约束"]
  C --> O["射线排序对照"]
  C --> D
  D --> E["持久不透明度系数"]
  E --> F["49 帧黑底、无编辑和删车"]
~~~

复用同一个私有渲染器的光栅输出缓冲区，逐射线重放。中心顺序在 49 帧黑底和两种街景状态的 RGB、alpha、深度辅助和标签贡献逐像素一致。诊断比较裁剪后条件深度中位数和三等份光学厚度采样。采用相同候选、材料和总不透明度。单点采用体深度中位数，三份采样等分光学厚度。原生中心深度辅助保留。两种按 ±0.25 或 ±0.75 m 限制相对中心的移动。零位移分片控制区分终止阈值影响。仅修正裁剪行且保护通过的最好黑底增益为 0.0018 个百分点，街景增益 0.0018 个百分点。全量改序破坏可见车身。正式方案保持中心顺序。

在实际参与光栅的行上，固定顺序和贡献集合，求每行不透明度变化对下游 PVD 和保护域的局部导数。线性规划跨 49 帧分配衰减，显露率仅按预先固定的下部像素计算。初版局部模型在 raw137 实际重放失败。采用版冻结该帧实际命中保护像素的身份，且为 raw97、112、113 的非目标 RGB 和 alpha 变化使用保守的一阶估计。所有候选再由原光栅路径复核，实际门槛沿用上一轮。每个行身份的衰减上限 25%，包含同源子行，跨帧和编辑状态固定。

| 固定域指标 | 上一轮 | 本轮 |
| --- | ---: | ---: |
| 黑底下部显露 | 50.851% | 50.934% |
| 删 a37 街景下部显露 | 48.777% | 48.859% |
| 黑底已观测 a6 RGB MAE | 0.018738 | 0.018743 |
| 黑底实测深度 MAE (m) | 0.269921 | 0.269882 |
| 无编辑道路 MAE | 0.031688 | 0.031688 |
| 删 a37 道路 MAE | 0.018911 | 0.018904 |

分母仍为同一批下部 PVD 的独立透明度，共 26,229 个固定像素。重开账本后独立重放 49 帧黑底和两种编辑状态，指标记录精确相同。该验证使用同场景观测，隐藏区域缺少独立几何真值。

~~~bash
/usr/local/cuda-11.8/bin/nvcc -O3 -arch=sm_89 --shared -Xcompiler -fPIC -o libreplay_quantile.so source/replay.cu
python source/depth_order.py --family median
python source/depth_order.py --family quantile
python source/order_controls.py
python source/support.py
python source/replay_support.py
~~~

命令在远端本次 campaign 目录运行，依赖同级上一轮资产、DrivingEditor 环境和先前的私有光栅器。

[排序对照](order_controls.json) · [逐帧指标](fixed_metrics.csv) · [持久系数](support_accepted.npz) · [独立重放](replay_receipt.json)。
