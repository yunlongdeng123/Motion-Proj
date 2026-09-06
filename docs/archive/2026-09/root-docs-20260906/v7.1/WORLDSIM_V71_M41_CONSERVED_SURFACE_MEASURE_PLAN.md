# WorldSim V7.1 M41 — Conserved Evidential Surface Measure

日期：2026-09-05  
状态：frozen

## 纠偏

M40的ray/GT supervision和可辨识性都正常，却把mean child occupied推到0.663。因为每个parent有四个children，
unconstrained categorical objective可通过改变anchor/child family总measure获得hit，重现sampling-density shortcut。
M41将F/O/U semantic probability与渲染surface measure显式分权。

## 表示与训练

- 从最后通过的M39组成初始化：M35 anchor head + M38 child head；不用失败M40 checkpoint；
- 对每个Actor，reference heads定义anchor与child两个family的GT-supervised总occupied measure；两者分别冻结；
- 当前head仍输出逐primitive F/O/U probability并接受soft evidential CE；进入ray renderer前，在每个family内归一化
  当前occupied probability，再乘对应reference total，只允许族内重分配；
- 该conserved surface measure同时用于32-bin categorical训练和64-bin CDF部署；M8 center/scale/trajectory冻结；
- 固定4 epochs/seed71141及M40损失权重；不调total、loss、bin、scale、median或seed，不threshold/delete/filter。

## 判定

相对原unit-energy baseline要求all与hazard/clear early不增、all hit最多降1pp，并要求raw semantic anchor/child
occupied correlation都≥0.25；另报告相对M39及family-total residual。失败登记`V71-F41`并关闭分族守恒候选，不作
normalization/loss恢复；无external/M21 partial read。
