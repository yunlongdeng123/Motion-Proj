# MM-TRELLIS：针对 a20 的接口核查

2026-11-22。只读论文、官方代码及输入规范；本次未安装或运行该模型。

![新版组件与灰模门](C:/Users/dengyunlong/Documents/Codex/2026-11-16/xian/outputs/vehicle_completion_prior_20261122/architecture.svg)

MM-TRELLIS 在原 TRELLIS 图生三维权重上进行测试时引导，没有额外训练一套车辆专用生成权重。它循环注入多张车辆图像，在去噪时对齐测距点与生成体素，并引导已观测占据位置；未命中的空间不作为空体素监督。这一点适合稀疏车辆测距，不能把缺测处直接压空。[论文](https://arxiv.org/html/2606.24301v1)

官方推理代码已公开，使用 `microsoft/TRELLIS-image-large`。输入为分割车辆图与预处理点云，输出包含 Gaussian PLY 和纹理网格；默认至少两张图。本文方法的输入域适配不等于车辆权重后训练，也不等于已有 Amodal3R 式显式可见／遮挡掩码接口。[代码与权重入口](https://github.com/HongliXiao/MM-TRELLIS)、[输入规范](https://github.com/HongliXiao/MM-TRELLIS/blob/main/docs/INPUT_FORMAT.md)

a20 的帧数不能代替视角覆盖：当前选定后视帧缺少侧面、正面信息。MM-TRELLIS 可以作为传感器约束教师的对照，不能仅因有连续帧就认定优先于遮挡感知模型。其论文也报告截断图像与噪声点云会引起缺面或多余结构。

接入时保存原始测距与对称／滤波派生点的来源。默认示例使用对称化、聚类清理后的点云；这些派生结构不能计入真实测量数量或测距验收。

免责声明：上述为官方能力与当前输入的适配判断，尚无 MM-TRELLIS 在 a20 上的运行或质量证据。
