# 实际仿真影响研究

task WS-SIM-IMPACT-01；run 20260915-r1。目标仍active；详见docs/WORLDSIM_SIMULATION_IMPACT.md及V74-H2-F14。

环境：官方HUGSIM/LTF使用envs/hugsim-impact/bin/python；四官方前馈模型使用envs/worldsim-v81/bin/python。绝对资源根为/root/autodl-tmp，运行资产位于runs/worldsim_simimpact/WS-SIM-IMPACT-01/20260915-r1。setup_runtime保留初始工程接通过程；后续依赖版本以runtime_packages_final.json为准，包含Kornia0.6.12及额外实际导入依赖。

run_native_baseline使用官方控制、渲染和动力学；默认原生50ms停止。--controller-fixed-iterations是取消墙钟停止的单列诊断控制，不是同实时预算。--collision-sample-stride只改变碰撞点采样；--collision-cloud可接入显式外部点集。输出目录已存在会拒绝覆盖；不要为了重跑删除旧结果。模拟seed20260915，复用官方前向入口的重建seed7403，两者均写入输出或登记。

已完成18次闭环1056步、24次四模型前向、48组碰撞几何替换。infer_reconstruction只读inputs的BUILD RGB，不访问QUERY。prepare_lidar_reference按BUILD前2/QUERY后6分开保存；build_geometry_swap只读BUILD_00作普通尺度锚点。它使用原生语义、采样、可见性和已知标定，属于诊断适配器，不是官方端到端前馈系统。

replay_collision_geometry在完全相同位姿上调用官方碰撞函数；audit_paired_controls核查共同轨迹前缀及目标支持覆盖。终止中心替换覆盖为0，48组未改变碰撞不是模型无害证据。相机对应仍待解决，不把叠图当精确真值。

初版data.pkl/official_eval.json记录pre-step诊断约定。reevaluate_official_format写official_post_data.pkl/official_post_eval.json，遵循官方closed_loop.py的post-step约定；指标比较优先后者，原在线碰撞标志以trace/summary为准。

作图draw_simimpact使用导出的visual_bundle与paired_control_audit.json，以Matplotlib输出PNG/PDF/SVG，不使用生成式图像填补数据。图F01真实RGB/稀疏LiDAR对应仍为provisional，F02固定轨迹配对结果不依赖该对应。

禁止以旧H2计划恢复训练或关机。原始资产、权重和大数组保留仓库外，代码和小型证据在本仓库。
