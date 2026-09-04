# WorldSim V7.1 M33 — Producer-side Anchor Evidence Sidecar

日期：2026-09-05  
状态：frozen

## 问题

M31证明锚点 early-return 同时来自 KEEP 与 PROJECT；M32进一步证明仅用 center/scale/type/Actor size
学习连续 FREE/OCCUPIED/UNKNOWN mass 无法辨识具体矛盾。继续调 mass、loss 或阈值没有科学含义。缺口位于
数据生产接口：1004-Actor corpus 丢弃了 anchor 的 provenance、source query、canonical build support 和
build-ray evidence。

## 冻结动作

1. 精确重放既有 721 raw + 283 processed-recovery Actor producer，只接受当前 corpus 中相同
   `(scene_name, track_id)`；不替换 Actor，不重写原 cache。
2. diagnostics 只新增 KEEP/PROJECT 到 source-query 与 canonical-surface 的索引，以及 canonical 的
   hit/temporal/view/origin build 统计。
3. 为每个 anchor 写独立 sidecar：provenance、source frame/ray、projection displacement、canonical
   build support、build LiDAR FREE/OCCUPIED/UNKNOWN counts/masses。
4. held-out native LiDAR 只写入 `supervision_*` 字段，按射线“端点前 FREE、端点附近 OCCUPIED、遮挡后或
   无支持 UNKNOWN”定义；绝不进入 `input_*` 字段。

## 边界

- M33 只做数据处理，不训练、不写 checkpoint、不调参；
- 不读取 selection/source-final/external，不读取 M21 AV2 partial quality；
- 不做 post-hoc 删除或 metric surface filtering；
- 不增加哈希、校验和、指纹；仅保留一次必要的 anchor 顺序对齐不变量；
- 固定一组 0.20 m beam、0.12 m build endpoint、0.20 m supervision endpoint 容差，不 sweep。

## 完成定义

1004 个现有 cache Actor 全部一一物化，sidecar anchor 数与原 cache 对齐，输出 corpus manifest、紧凑
Actor index 与 `COMPLETE`。完成后 M34 只能把 `input_*` 用作模型输入，把 `supervision_*` 用作训练目标；
M33 数据不构成性能 claim。
