# V6.7 ray-evidence migration freeze

V6.6 one-voxel isotropic support提高clean retention，却将conflict reduction降到41.79%。外部检索后冻结最小迁移：

- [ALSO（CVPR 2023）](https://openaccess.thecvf.com/content/CVPR2023/html/Boulch_ALSO_Automotive_Lidar_Self-Supervision_by_Occupancy_Estimation_CVPR_2023_paper.html)
  利用sensor location构造有方向的occupancy supervision；迁移为使用现有ray termination，不训练新backbone。
- [Accurate Training Data using Evidence Theory（CVPR 2024）](https://openaccess.thecvf.com/content/CVPR2024/html/Kalble_Accurate_Training_Data_for_Occupancy_Map_Prediction_in_Automated_Driving_CVPR_2024_paper.html)
  显式区分LiDAR evidence与unknown/contradiction；迁移为不把未知空间静默当occupied。
- [SelfOcc（CVPR 2024）](https://openaccess.thecvf.com/content/CVPR2024/html/Huang_SelfOcc_Self-Supervised_Vision-Based_3D_Occupancy_Prediction_CVPR_2024_paper.html)
  用signed-distance/rendering约束3D geometry；当前只迁移surface两侧的方向性语义，不引入渲染训练。

项目V62 evidence已保存`behind_hit`：沿LiDAR endpoint之后若干voxel标记方向性occlusion support。V6.7只允许把它
与same-Actor one-voxel proximity做AND；exact hit永远保留。禁止中间radius、target-dependent rule或gate relaxation。
