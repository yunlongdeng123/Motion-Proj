# WorldSim V7.1 M39 — Categorical Authority Composition Audit

日期：2026-09-05  
状态：frozen

## 目的

M38证明native pre-hit supervision有效，但additive optical thickness把endpoint authority与Gaussian前翼hazard耦合。
NeuS/VolSDF说明generic volume density与surface存在系统偏差；项目内部M18也已证明直接categorical first-return
distribution能避免弥散density的“积少成多”。M39因此只隔离composition，不训练或修改authority/geometry。

## 冻结设计

- 同一66个exposed holdout Actors；M8 center/scale/trajectory、M35 anchor head、M37/M38 child heads全冻结；
- 对同一64个ray depth bins计算continuous-authority weighted Gaussian energy并直接softmax为first-return categorical
  distribution；部署取同一CDF median；无additive optical thickness/no-return重归一化；
- 主candidate事前固定为M38 anchor+child authority；M37 arm只作配对描述，不允许结果后择优；
- M34 anchor+unit-child arm与原unit-energy baseline用于定位；不训练、不改scale/bin/threshold，不删除/filter；
- 判定M38相对原baseline的all及hazard/clear early不增、all hit最多降1pp。失败即登记`V71-F40`并关闭
  scalar-authority + isotropic-energy组合；不读external/M21 partial。
