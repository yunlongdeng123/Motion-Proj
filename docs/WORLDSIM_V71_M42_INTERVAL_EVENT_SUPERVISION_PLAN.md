# WorldSim V7.1 M42 — GT Interval-event CDF Supervision

日期：2026-09-05  
状态：frozen

## 根因

M41证明分族总measure守恒、F/O/U可辨识且point-bin NLL下降仍不足以控制部署median early event。Nominal CE忽略
depth顺序；LidaRF（CVPR 2024）的LiDAR sight loss则直接约束GT depth邻域的ray-weight/CDF。M42把验收事件本身
定义为GT supervision，不在输出后filter。

## 唯一变化

- 保留M41全部表示：从M35/M38初始化、每Actor anchor/child reference total分别守恒、M8 geometry冻结；
- 删除single target-bin CE与expected-depth L1代理；在32-bin训练分布上直接计算：
  `L_safe=-log P(d >= d_gt-0.20m)`，`L_hit=-log P(|d-d_gt| <= 0.20m)`；两者等权1.0；
- anchor/child GT F/O/U soft CE各权重0.5并持续训练，防止event loss成为不可解释重排；
- 64-bin部署仍取同一categorical CDF median；训练的safe event直接要求early-side累计概率低；
- 固定4 epochs/seed71142；不调tolerance/weight/bin/median/total/scale/seed，不threshold/delete/filter。

## 判定

相对原unit-energy baseline要求all与hazard/clear early不增、all hit最多降1pp，并要求anchor/child occupied
correlation都≥0.25；另报告相对M39。失败登记`V71-F42`并关闭当前interval-event fine-tune，不做权重或容差恢复；
无external/M21 partial read。
