# P7R Sensor-supported Actor-local Surface Repair Migration Freeze

P7 triage通过但未修改physical geometry，不能把handled candidate直接写成artifact repair。按卡点协议检索后冻结以下迁移：

- [NeuRAD（CVPR 2024）](https://openaccess.thecvf.com/content/CVPR2024/html/Tonderski_NeuRAD_Neural_Rendering_for_Autonomous_Driving_CVPR_2024_paper.html)
  将动态Actor与sensor modeling显式结合，支持actor edit；迁移为Actor shell和sensor surface分层，而非替换backbone。
- [Neural Scene Graphs（CVPR 2021）](https://openaccess.thecvf.com/content/CVPR2021/papers/Ost_Neural_Scene_Graphs_for_Dynamic_Scenes_CVPR_2021_paper.pdf)
  用track transformation分解static/dynamic对象；迁移为Actor canonical track/box保持物理身份，local surface独立修复。
- [Cam4DOcc（CVPR 2024）](https://openaccess.thecvf.com/content/CVPR2024/html/Ma_Cam4DOcc_Benchmark_for_Camera-Only_4D_Occupancy_Forecasting_in_Autonomous_Driving_CVPR_2024_paper.html)
  强调instance 4D occupancy与backward flow；迁移为复用项目已有motion-compensated Actor hit，而不训练新4D backbone。
- [Cam4DOcc official code](https://github.com/haomo-ai/Cam4DOcc)保留instance目录与4D occupancy基准接口，支持本项目
  将Actor-local repair作为显式instance artifact而非whole-scene filter。

冻结P7R：继续使用P7 L0固定290个action states，不改budget/score/model。对action state，把native Actor-owned boundary
只保留能映射到motion-compensated same-Actor sensor hit的primitive，其余转UNKNOWN；未action state保持原geometry。
Actor canonical box、ID、class、track、trajectory和hazard attributes全部沿用P6，不删除Actor。target evidence只用于
评估repaired conflict，不进入action或primitive-retention rule。

Gates在formal read前固定：point conflict reduction>=50%、Actor/collision-shell/ID/trajectory retention=1、Actor removed=0、
hazard proxy shift=0、overall emitted boundary fraction>=40%、clean boundary retention>=40%、scene yield=1。通过也只支持
consumed legacy sensor-supported surface repair capability，不支持RGB、fresh distribution或RL-ready claim。
