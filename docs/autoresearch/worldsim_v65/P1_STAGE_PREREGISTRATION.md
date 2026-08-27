# P1 Continuous-Trajectory Signal Atlas Preregistration

Task：`WS-V65-P1-CONDITION-SIGNAL-ATLAS-01`。

这是 Tier-L train-only 机制实验，不读取 V6.5 正式 selection。

- R0：冻结 V6.4 full-native q0；
- R1：冻结 q0 64D hidden/logit，增加 `10→32→16` 连续 trajectory encoder、FiLM interaction、32D delta head；
- scene/unit split：16 scenes，每 scene 前 8 units train、后 4 units evaluation；
- sample：train 4096/unit，evaluation 8192/unit；
- seed=0，40 epochs，batch=16384，AdamW；
- model feature 不含硬 route corridor、stratum 或 scene ID；
- 1.5m corridor 只构造 fixed-opportunity evaluation denominator；
- perturbation：evaluation unit 内 shuffle trajectory feature，只作机制诊断；
- positive signal：AUROC gain `>=0.005`、fixed-route relative reduction `>=5%`、scene lower>higher、真实条件
  AUROC 高于 shuffle；
- no sweep、no second arm、no formal V6.5 claim。

若无清晰信号，按 Stop 1 关闭 trajectory family；若有信号，只解锁 fresh P2/T0，不直接解锁 attention。
