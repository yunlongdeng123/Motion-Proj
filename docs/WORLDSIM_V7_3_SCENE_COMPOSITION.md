# V7.3 场景级背景与Actor组合

当前为实现和实验登记，尚未取得真实场景结果。研究风险F04需要在同一背景中比较全局首交点，Actor单独读出不足以判断遮挡和边界。

采用[Street Gaussians（ECCV 2024）作者实现](https://github.com/zju3dv/street_gaussians/)的背景与规范Actor分解方式，保留已知刚体轨迹；不迁移外观透明度作为物理读出。按[Open3D0.19官方RaycastingScene](https://www.open3d.org/docs/release/python_api/open3d.t.geometry.RaycastingScene.html)，为各规范三角面建立CPU加速结构，将原始束逆变换到该Actor坐标，求交后与静态背景统一取最近的正距离及owner。刚体旋转不改变单位方向长度，参数t仍是米制距离。固定曲面不需要梯度，复用现有worldsim-v72-lidar4d环境（Torch2.1.0+cu121、Open3D0.19.0、SciPy1.13.1），仅补装129kB的ijson3.5.1；当前训练环境不变。

背景只来自原4个build扫描，在各自扫描时刻剔除所有已知注释框+0.1m内的点，包括非刚体和未参加重建的已知物体；不是仅在最后位置裁剪。剩余点世界坐标累积，固定0.06m间距、9顶点/8三角面的PCA局部片，使用全部build背景点，不做凸包、自动补洞、opacity或闭合假设。未被采样的背景仍可缺失。注释不完整和框归属错误仍会产生残留/误剔除；不能声称语义真实背景已经完备。

评价读取同一短窗口内未输入的原始首回波束，保留所有有效距离，包括非刚体和未重建对象。各方法共享同一背景、轨迹、原始束和硬三角读出。单列all raw returns、cohort自有返回、框外背景代理返回、其他注释对象、重叠歧义、注释框表面0.2m边界带。此边界带由真实回波端点与已知框定义，是诊断代理，不能称为真实Actor接触边界。报告literal hit/early/late/miss、直接free侵入、带returned计数的条件MAE、背景返回被Actor抢先、Actor返回被背景抢先、归属错配和近深度重叠。

聚合先在scene内按真实束计数，再scene等权到log、独立log等权；与此前Actor内加权→log均值的统计单位不同，不能直接横比数值。全场景平均可能被背景主导，cohort与背景分项必须并列。原始GT只有首返回，后方表面不被额外标为错误或自由空间；没有反事实GT的轨迹编辑仅作组合能力验证。

入口：`prepare_worldsim_v73_scene_geometry.py`（build背景/评价束）；`evaluate_worldsim_v73_scene_composition.py`（已完成各方法surface文件）；`scene_readout.py`（BVH/只读刚体组合）。缺artifact直接报未完成，合法空表面保留为miss，避免将工程缺文件伪装为模型缺失。

首次数据登记：`WS-V73-M4-SCENE-DATA-01/20260907T211000Z__development-build-background-r1`，原有6开发scene/5日志；不使用新source作为开发，也不新增优化。首次比较登记：`WS-V73-M4-SCENE-COMPOSITION-01/20260907T211000Z__development-pca-lidar-native-r1`，background-only诊断、LiDAR PCA、已完成完整LiDAR-only r6、native+LiDAR fusion r2。r6比融合有更多短窗fit标签，监督预算差异保留；尚未完成的joint r5不读取中途结果冒充最终预测。

只进行一次解析遮挡/刚体读出核验，预期两Actor先后位于5/7m、背景10m；平移前后的全局owner变化有解析真值。它检验逆变换和遮挡读出实现，不等于真实场景验证。随后直接运行真实开发队列，不新增连续门控或回归套件。

F04仍active：全局排序解决读出语义，不自动修复背景ghost、注释误差、曲面片撕裂或未知背景；等实际分项结果后再选择迁移对策。F03首事件、强基线与新日志确认也未完成，shutdown=false。
