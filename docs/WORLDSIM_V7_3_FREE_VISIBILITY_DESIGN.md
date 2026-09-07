# V7.3 射线管可见性梯度实验

`WS-V73-M3-FREE-VISIBILITY-01` 一次解析机制实验已完成（code c6129bcc，run `20260907T174500Z__analytic-r1`）。5m处、横向偏移2.5cm的12cm曲面片遮挡20m返回：硬range侵入14.8m、整体平移梯度[0,0,-1]；有限射线管覆盖0.83977、平移梯度[-5.8593,0,0]。负梯度更新给出退出横向射线管的方向。位于原始首回波后或完全在管外时均为0；重复同一表面覆盖和梯度均不变。证据=`docs/autoresearch/worldsim_v73/m3/free_visibility_analytic_r1.json`。这只说明代理能提供不同局部几何方向，不保证真实数据收益、任意可见性切换或缺失支持出生。

下一受控训练r3：task `WS-V73-M2-GLOBAL-ACTOR-01`，run `20260907T175000Z__surround-native-beam-tube-s7304-r3`。从同一M1 r3初始权重/seed7304、20fit/5dev Actor、30epochs/600更新，保留native-data weight1、coverage及弱envelope，仅把free训练目标从硬range侵入换为同一三角曲面的有限射线管覆盖。free weight0.5、宽度0.03m、32×32分辨率、每步原始512束不变。覆盖比例与米制range是不同单位，不声称权重相同等于梯度强度相同；另外持续记录原始硬free米值，最终统一硬表面评价。

将原始返回−0.2m设为free终点，nvdiffrast栅格化几何占据、antialias提供轮廓位置梯度；相机/轨迹/表面半径/opacity均无新可训练控制。固定3σ范围和高斯足迹是训练代理，不能把横向偏移射线称为新增观测真值。边界裁剪、正好对称或完全离开有限支持仍可能无梯度，宽度在掠射表面上的偏差需结合hard hit/early/miss判断。保留真实depth与coverage的支持恢复路径，不以event替代。

运行依赖：motionproj保留Torch2.4.1+cu121，新增nvdiffrast0.3.3、ninja1.13.2；仅复用`/root/autodl-tmp/envs/worldsim-v72-pointr`的CUDA12.1编译器，库仍在motionproj运行。源码`/root/autodl-tmp/external/worldsim_v73/nvdiffrast` tag v0.3.3，扩展缓存`/root/autodl-tmp/torch_extensions/worldsim_v73`；不升级Torch，不新建环境，不添加校验工件。来源：[官方API/SIGGRAPH Asia 2020](https://nvlabs.github.io/nvdiffrast/)、[兼容版本源码](https://github.com/NVlabs/nvdiffrast/tree/v0.3.3)。

r2已结束，r3提交后启动；当前无资源不足、shutdown=false。failure_ledger_delta=update V73-F02（已得到横向代理梯度，实际收益待测），F01:F05仍active，F06在直接测量路径范围缓解；下一编号V73-F07。完整同信息控制、更多动态独立日志、event和背景组合仍pending，不把这一次解析结果当成V7.3完成。
