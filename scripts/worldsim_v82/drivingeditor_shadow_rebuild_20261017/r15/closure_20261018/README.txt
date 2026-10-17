R15最小阴影修复与有限实验（2026-10-18）

packed_shadow.py是已验收的内存适配器，不修改checkpoint schema或原生产入口。
在现有SceneModel构造/load_state_dict后，第一次buffers之前调用一次：
    from packed_shadow import install
    install(model, dtype=torch.float64)
model.buffers把同一真实Actor的主轴因子传给太阳积分，编辑位姿与删除也共享。
要求torch2.2.2+cu118、原DrivingEditor光栅/EnvLight依赖。没有安装或替换CUDA ABI。
保留5方向、全部caster、旧损失与分辨率。当前修复只验收到报告中的6帧数值/梯度和6帧吞吐范围。
原训练保持暂停；本轮没有恢复45K，也没有原地改写生产目录。

run_numeric_full.py须先通过，再运行run_performance_gate.py；性能通过才允许owner_entry_audit。
run_a37_appearance_control.py是稀疏外观强控制，不是主训练，不可把参考MSE下降当区域残影通过。
propose_a6_geometry.py使用全部独立世界道路证据，但此次仍无≥3帧正支持，不执行归属修改。
factor_integral.py是无粗筛FP64参考；run_cpu_checks.py核查独立积分和梯度。
所有runner的--source/--base/--output指向原campaign与新结果目录；禁止覆盖main/latest.pt。
数值/计时/审计结果见同目录上级报告，实验ID与停止原因见closure_summary.json。
原始checkpoint和逐点NPZ仍保留服务器；不将大权重或大量逐点数组推入Git。
