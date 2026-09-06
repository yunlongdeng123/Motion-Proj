# WorldSim V7.1 M37 — Supervised Completion-child Transmittance

日期：2026-09-05  
状态：frozen

## 隔离问题

M36把M35过度首碰定位到unit completion-child opacity。M37冻结M35 anchor authority、M8 child geometry/scale和
trajectory，只训练独立child evidential head，不重新混合observed anchor与completion。

## Child输入与GT

- 输入为每个parent candidate的冻结build-only M8标准化特征（已含build F/O/U、ray、normal、local support），
  加child局部位置、abs位置、scale、相对parent residual及四槽one-hot；
- held-out LiDAR按0.20 m beam/endpoint构造每child连续FREE/OCCUPIED/UNKNOWN target；无支持为UNKNOWN，
  FREE/OCCUPIED冲突保留soft mass；
- anchor head读取M33输入并完全冻结；child head单独PointNet编码；
- ordered analytic transmittance first-return NLL与child evidential CE共同训练，geometry无梯度；
- 单seed 71137、6 epochs；不调optical/global/Gaussian scale，不阈值、不删除/filter。

## 判定

同一593 train / 66 exposed holdout，相对原unit-energy baseline要求all/hazard/clear early不增、all hit最多降1pp，
并要求predicted-vs-GT child occupied相关≥0.25。另报告相对M35 unit-child arm变化。任一失败则不进external且
停止该双头参数化；不读M21 partial。下一failure ID=`V71-F38`。
