# WorldSim V6.5 P0 Protocol

状态：`complete_direct_research_profile`（2026-08-27）。

- source：`research/worldsim-v6.4-native-uq@add2f3f`；
- branch：`research/worldsim-v6.5-task-conditioned-authority`；
- immutable baseline：V6.4 full-native MLP、C0、M0、M1；
- scene discipline：所有 V6.4 quality-read scenes 均为 Tier L；
- first research slice：P1 continuous trajectory signal atlas，随后 T0 residual；
- exposure：V6.5 formal selection/calibration/confirmation/test 均未读取；
- compute：1×RTX 3090 足够完成 P1–P6 faithful minimum；
- I/O：只读复用 canonical sidecar，先物化 compact cache，再持续 GPU 训练；
- validation：仅做配置可解析、窄入口和主实验产物验证；
- prohibited：哈希/校验和/指纹、scene-ID inference、阈值/seed/capacity sweep、过量 smoke/回归。

本文件只冻结归因边界，不把 P0 变成新的验证项目。
