# WorldSim V7.1 M38 — Native Pre-hit Free-space Survival

日期：2026-09-05  
状态：frozen

## 纠偏依据

M37证明completion-child authority可由producer build context辨识，并相对unit-child显著降低early；但whole-ray
categorical NLL几乎不变，且相对原baseline仍有5.73pp early。ALSO（CVPR 2023）直接从LiDAR sensor origin到
return构造occupancy监督；Neural LiDAR Fields（ICCV 2023）把return/drop建模为物理射线过程。因此M38把
GT first return之前的观测FREE interval直接写入训练目标，而不是在部署后删除primitive。

## 唯一变化

- 初始化M37 child checkpoint；冻结M35 anchor head、M8 child center/scale、trajectory与所有输入；
- 对每条训练ray，在GT endpoint前0.20m之外计算completion-child累计解析Gaussian optical thickness：
  `L_free = sum_{k: d_k < d_gt - 0.20} tau_child,k = -log T_child,pre`；
- 保留M37 ordered categorical NLL、depth L1和child F/O/U evidential CE；`L_free`权重固定1.0；
- 单seed 71138，继续4 epochs；不调margin/weight/scale/segment/seed，不阈值、不删除/filter；
- 只惩罚completion-child optical mass；frozen anchor的残余early边界不由child head吸收。

## 判定

同一593 train / 66 exposed holdout，相对原unit-energy baseline要求all及hazard/clear early不增、all hit最多降1pp，
且child occupied相关不低于0.25；同时报告相对M37的early/hit、pre-hit optical thickness与no-return变化。任一失败
登记`V71-F39`并关闭当前对称F/O/U + additive optical-mass参数化；不做weight/margin恢复，不读external或M21 partial。
