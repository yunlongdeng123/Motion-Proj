# 从零实例场景与车辆关联阴影：实现与验证协议

任务 `WS-V82-DRIVINGEDITOR-SHADOW-REBUILD-20261017 / v3_code_cpu`。
用户要求从原始测量开始建立适于后续光照分解的场景，先解决道路细节、阴影轮廓、车身质量。执行状态统一见 [RESEARCH_STATUS](../RESEARCH_STATUS.md)；历史局部道路替换结果继续由[上一轮实验](DRIVINGEDITOR_SHADOW_SEQUENCE_20261016.md)定位，不覆盖历史证据。

![输入测量、三阶段训练、共享Actor几何射线积分与编辑合成组件图](../autoresearch/worldsim_v82/drivingeditor_shadow_rebuild_20261017/v3_code_cpu/architecture.svg)

## 实现

完整源码、损失归一化、数据契约、命令和外部依据见 [源码说明](../../scripts/worldsim_v82/drivingeditor_shadow_rebuild_20261017/README.md)。

- Stage1由训练帧真实LiDAR初始化非平面表面高斯与独立Actor，训练几何/opacity/canonical颜色；静态层没有time/velocity/lifetime。道路参考来自真实跨时刻观测，几何更新后重新投影，不使用旧手选修复ROI或原渲染混合。
- Stage2默认固定几何和opacity，训练道路逐高斯albedo、太阳/环境光与Actor外观；同一车辆高斯几何和opacity直接用于主渲染与可微遮光积分。道路接收面的条件G-buffer共享全场景alpha*T，Actor正常暗面/反射保留在新训练canonical radiance中。没有五盒体或自由形变RGB阴影代理。
- Stage3小学习率联合，位置相对前一阶段最多2cm、尺度/opacity有界；持续保留全图采样RGB/patch SSIM、稀疏LiDAR深度、完整mask内外、可见实例、车身重建与教师、跨帧道路监督。

完整mask在未预乘GT的逐Actor raw alpha上按域分别归一化：mask内趋1、mask外趋0，默认各0.05；可见实例alpha*T另计0.1。未知mask默认阻止正式训练，不伪装成背景。允许部分mask须通过显式参数并登记输入缺口。

## 验证范围

[CPU原始检查记录](../autoresearch/worldsim_v82/drivingeditor_shadow_rebuild_20261017/v3_code_cpu/cpu_validation.json)记录17项通过及21文件AST检查。含独立数值积分、解析梯度、同几何编辑、mask外梯度、冻结边界、原始数据导出、测试RGB隔离、消融初始化隔离、道路接收属性/Actor贡献隔离、增密后的模型/Adam/RNG逐位恢复及连续编辑。另一个5帧合成夹具完成30次三阶段更新（12/12/6），示例帧线性RGB MAE从0.053931降至0.045594；40张PNG检查删全部关影一致与固定相机静态一致。上述数字是CPU工程验证，不是真实驾驶数据结果，也不能代替实际CUDA内核验收。

静态道路是LiDAR局部低表面/法线候选，不冒充语义GT；缺少光照变化或始终遮挡的道路仍不可辨识。太阳积分是3D密度直接光近似，opacity以薄轴穿心光学厚度标定、6σAABB粗剔除；不是IRGS完整间接传输或完整车辆BRDF。真实GPU成本与画质须实测。

raw101/113/125从新分支种子、道路参考与训练帧采样排除；历史研究用过这些帧，不能称整个研究过程从未见过的最终测试集。无反事实编辑/无影GT时保持指标证据边界，人工verdict不代填。

## 后续实验协议

[阶段预算与消融入口](../autoresearch/worldsim_v82/drivingeditor_shadow_rebuild_20261017/v3_code_cpu/experiment_plan.json)。30k/10k/5k只是初始预算；先审计实际49帧输入和全部Actor mask，再测CUDA ABI/梯度、各阶段吞吐/显存，才决定总时长与检查间隔。进度工具只读，不自动开机、恢复或启动控制器。

7个入口：full、no_shadow、no_clean_road、no_body_teacher、frozen_joint_geometry、hard_sun、opacity_stage2。no_clean_road同时去除亮参考初始化、损失和刷新，避免消融泄漏先验。每臂同数据划分/初始几何/seed、独立配置与输出；先完整方法，再按结果选择必要消融。

评价通过相机+同帧真实LiDAR准入表选择目标；逐车删除只涵盖证据清楚的Actor。全动态保留独立RGB/raw alpha和全场景alpha*T可见RGB/alpha；每辆保留独立/可见alpha、删除后剩余动态。旧三车另有同轴±1m、±15°/±30°连续编辑与编辑动态RGB。7个目标、3个运动目标共65流×49帧；背景-only为分支诊断，不算其他轨迹delete通过。回看raw106–108与大角度车身，不能仅凭平均误差接受。具体准入、实测CUDA/梯度/测速与输入缺口见[GPU记录](../autoresearch/worldsim_v82/drivingeditor_shadow_rebuild_20261017/v3_gpu_r3/)，执行状态仍只见RESEARCH_STATUS。

## 一手依据

[LumiMotion论文](https://arxiv.org/html/2604.10994v1)和[官方阶段脚本](https://github.com/joaxkal/LumiMotion/blob/main/bash_scripts/enerf-actor1.sh)支持几何先行、后续材质/光照与重建约束的组织；[IRGS官方实现](https://github.com/fudan-zvg/IRGS)支持适于光照的表面与冻结几何策略。这里未复现其动态形变或OptiX系统。[Gaussian Shadow Casting](https://arxiv.org/html/2401.06116v1)提供各向异性密度解析射线积分依据；本代码将其绑定到同一Actor几何。[RNG](https://arxiv.org/html/2409.19702v3)作为后续深度/边缘方向，本实现尚未加入其网络。

`failure_ledger_refs=[V82-F01]`；`failure_ledger_delta=none`，本轮为新代码与合成CPU检查，没有新增真实科学否定结论。人工verdict保留null。
